# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Sanity checks for the generator-only robustness fine-tuning wiring
(src/audioseal_robust). These do NOT test watermark quality/robustness --
they test that the training loop's core correctness property holds:
gradients flow end to end from the loss back through the frozen detector and
the frozen-but-differentiable attack module into the generator, while the
detector and attack module themselves never accumulate gradient and are
never touched by the optimizer.

Uses tiny locally-constructed generator/detector (not a downloaded
pretrained checkpoint) so this runs fast and offline.
"""

import copy
import math
import os
import sys

import pytest
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from audioseal.builder import (
    AudioSealDetectorConfig,
    AudioSealWMConfig,
    DecoderConfig,
    DetectorConfig,
    SEANetConfig,
    create_detector,
    create_generator,
)
from audioseal_robust.attacks import (
    DiffEraseAttack,
    IdentityAttack,
    MBDAttack,
    SampledReconstructionAttack,
    SGMSEAttack,
)
from audioseal_robust.config import load_config, load_eval_config
from audioseal_robust.evaluate import (
    SNR_AUDIBLE_DB,
    SNR_NEAR_SILENT_DB,
    STOCK_DETECTOR_CARD,
    STOCK_GENERATOR_CARD,
    apply_stock_baseline,
    build_eval_attacks,
    check_watermark_snr,
    evaluate_attack,
    evaluate_perceptual,
    prepare_eval_batches,
    stock_baseline_verdict,
)
from audioseal_robust.losses import PsychoacousticMelLoss, detection_loss
from audioseal_robust.metrics import watermark_report, watermark_snr_db
from audioseal_robust.train import embed_watermark


def _tiny_seanet_config() -> SEANetConfig:
    return SEANetConfig(
        channels=1,
        dimension=16,
        n_filters=4,
        n_residual_layers=1,
        ratios=[2, 2],
        activation="ELU",
        activation_params={"alpha": 1.0},
        norm="none",
        norm_params={},
        kernel_size=3,
        last_kernel_size=3,
        residual_kernel_size=3,
        dilation_base=2,
        causal=False,
        pad_mode="constant",
        true_skip=True,
        compress=2,
        lstm=1,
        disable_norm_outer_blocks=0,
    )


def _tiny_generator(nbits: int):
    cfg = AudioSealWMConfig(
        nbits=nbits,
        seanet=_tiny_seanet_config(),
        decoder=DecoderConfig(final_activation=None, final_activation_params=None, trim_right_ratio=1.0),
        normalizer=False,
    )
    return create_generator(cfg)


def _tiny_detector(nbits: int):
    cfg = AudioSealDetectorConfig(
        nbits=nbits,
        seanet=_tiny_seanet_config(),
        detector=DetectorConfig(output_dim=8),
        normalizer=False,
    )
    return create_detector(cfg)


def test_embed_watermark_hits_target_snr_range_per_example():
    """Regression test for the mentor-requested SNR-targeting mechanism:
    each example in the batch should independently land within
    [snr_db_min, snr_db_max], not just the batch as a whole -- and gradients
    must still flow (this replaces a plain x + get_watermark(...) that was
    trivially differentiable)."""
    torch.manual_seed(0)
    nbits = 4
    generator = _tiny_generator(nbits)

    x = torch.randn(8, 1, 4000)
    message = torch.randint(0, 2, (8, nbits))
    x_wm = embed_watermark(generator, x, message, snr_db_min=24.0, snr_db_max=36.0)

    delta = x_wm - x
    achieved_snr_db = 20 * torch.log10(x.norm(dim=-1) / delta.norm(dim=-1).clamp_min(1e-8))
    assert bool(((achieved_snr_db >= 23.9) & (achieved_snr_db <= 36.1)).all())
    assert x_wm.requires_grad


def test_prepare_eval_batches_is_seeded():
    generator = _tiny_generator(nbits=4).eval()
    cfg = load_eval_config(["eval_dir=.", "nbits=4", "n_eval_batches=1"])
    dataloader = [torch.randn(2, 1, 4000), torch.randn(2, 1, 4000)]

    torch.manual_seed(123)
    first, first_diagnostics = prepare_eval_batches(generator, dataloader, cfg, torch.device("cpu"))
    torch.manual_seed(123)
    second, second_diagnostics = prepare_eval_batches(generator, dataloader, cfg, torch.device("cpu"))

    assert len(first) == 1  # n_eval_batches caps the dataloader
    assert all(torch.equal(a, b) for a, b in zip(first[0], second[0]))
    assert first_diagnostics == second_diagnostics


def test_prepare_eval_batches_measures_snr_before_any_attack():
    """The watermark diagnostic must describe the generator's own output:
    one value per clip actually evaluated, matching a direct measurement of
    the (clean, watermarked) pair the batches carry."""
    generator = _tiny_generator(nbits=4).eval()
    cfg = load_eval_config(["eval_dir=.", "nbits=4", "n_eval_batches=2"])
    dataloader = [torch.randn(3, 1, 4000), torch.randn(3, 1, 4000), torch.randn(3, 1, 4000)]

    batches, diagnostics = prepare_eval_batches(generator, dataloader, cfg, torch.device("cpu"))

    assert diagnostics["n_clips"] == 6  # 2 batches x 3 clips, not 3 batches
    expected = torch.cat([watermark_snr_db(x, x_wm) for x, x_wm, _ in batches])
    # quantile(0.5), not median(): on an even number of clips torch.median
    # returns the lower of the two middle values instead of interpolating.
    assert diagnostics["snr_db"]["p50"] == pytest.approx(
        torch.quantile(expected, 0.5).item(), abs=1e-4
    )
    assert diagnostics["snr_db"]["min"] == pytest.approx(expected.min().item(), abs=1e-4)
    assert diagnostics["delta_rms"] > 0.0


def test_check_watermark_snr_warns_on_near_silent_watermark():
    diagnostics = watermark_report(
        torch.full((10,), SNR_NEAR_SILENT_DB + 5.0), torch.full((10,), 1e-9)
    )

    warnings = check_watermark_snr(diagnostics)

    assert any("near-silent" in w for w in warnings)


def test_prepare_eval_batches_flags_a_collapsed_generator():
    """The failure mode this whole diagnostic exists for: a generator whose
    delta has collapsed to zero. It has to be caught here, pre-attack, as
    "the generator emitted nothing" -- otherwise it surfaces only as
    near-chance detection, which looks identical to a broken detector."""

    class CollapsedGenerator(torch.nn.Module):
        def get_watermark(self, x, message=None):
            return torch.zeros_like(x)

    cfg = load_eval_config(["eval_dir=.", "nbits=4", "n_eval_batches=1"])

    _, diagnostics = prepare_eval_batches(
        CollapsedGenerator(), [torch.randn(4, 1, 4000)], cfg, torch.device("cpu")
    )

    assert diagnostics["delta_rms"] == 0.0
    assert diagnostics["delta_rms_min"] == 0.0
    # A zero delta must saturate, not blow up: an inf here would poison the
    # aggregates and hide the very thing being diagnosed.
    assert math.isfinite(diagnostics["snr_db"]["p50"])
    assert diagnostics["snr_db"]["p50"] > SNR_NEAR_SILENT_DB

    warnings = check_watermark_snr(diagnostics)
    assert any("near-silent" in w for w in warnings)
    assert any("exactly-zero" in w for w in warnings)


def test_prepare_eval_batches_sees_collapse_that_normalization_hides():
    """The realistic collapse, and the reason the raw pre-scaling numbers are
    measured at all: delta is tiny but NOT exactly zero, so
    embed_watermark's per-clip scaling amplifies it straight back up to the
    configured target. Post-scaling `snr_db` is then pinned at that target and
    reports the same value a healthy generator would -- only `raw_snr_db` can
    still tell the two apart."""

    class NearlyCollapsedGenerator(torch.nn.Module):
        def get_watermark(self, x, message=None):
            return 1e-9 * torch.randn_like(x)

    cfg = load_eval_config(["eval_dir=.", "nbits=4", "n_eval_batches=1", "watermark_snr_db=30.0"])

    _, diagnostics = prepare_eval_batches(
        NearlyCollapsedGenerator(), [torch.randn(4, 1, 4000)], cfg, torch.device("cpu")
    )

    # Post-scaling: indistinguishable from healthy, and the delta is nonzero
    # so delta_rms doesn't give it away either.
    assert diagnostics["snr_db"]["p50"] == pytest.approx(30.0, abs=0.1)
    assert diagnostics["delta_rms"] > 0.0
    # Pre-scaling: the collapse is plainly visible.
    assert diagnostics["raw_snr_db"]["p50"] > SNR_NEAR_SILENT_DB
    assert diagnostics["raw_delta_rms"] < 1e-6

    # The normalization itself is working, so the assert must NOT fire -- but
    # the generator is broken, so the warning must.
    warnings = check_watermark_snr(diagnostics, snr_db_range=(cfg.watermark_snr_db, cfg.watermark_snr_db))
    assert any("near-silent" in w for w in warnings)
    assert any("raw" in w for w in warnings)


def test_prepare_eval_batches_stays_quiet_on_a_healthy_generator():
    """Counterpart to the collapse test: a generator emitting a sane delta
    passes both the target assertion and every quality gate."""

    class HealthyGenerator(torch.nn.Module):
        def get_watermark(self, x, message=None):
            return 0.01 * torch.randn_like(x)

    cfg = load_eval_config(["eval_dir=.", "nbits=4", "n_eval_batches=1", "watermark_snr_db=30.0"])

    _, diagnostics = prepare_eval_batches(
        HealthyGenerator(), [torch.randn(4, 1, 4000)], cfg, torch.device("cpu")
    )

    assert diagnostics["snr_db"]["p50"] == pytest.approx(30.0, abs=0.1)
    assert SNR_AUDIBLE_DB < diagnostics["raw_snr_db"]["p50"] < SNR_NEAR_SILENT_DB
    assert check_watermark_snr(diagnostics, snr_db_range=(cfg.watermark_snr_db, cfg.watermark_snr_db)) == []


def test_check_watermark_snr_assert_fires_when_normalization_misses_target():
    """The assertion reads the POST-scaling median even though the warnings
    read the raw one -- a normalization that doesn't hit its target is a
    wiring bug, and must not be masked by a healthy-looking raw delta."""
    diagnostics = watermark_report(
        torch.full((8,), 12.0),  # post-scaling: nowhere near the 30 dB target
        torch.full((8,), 0.05),
        raw_snr_db=torch.full((8,), 30.0),  # raw: perfectly healthy
        raw_delta_rms=torch.full((8,), 0.01),
    )

    with pytest.raises(AssertionError, match="outside the configured"):
        check_watermark_snr(diagnostics, snr_db_range=(30.0, 30.0))


def test_check_watermark_snr_warns_on_audible_watermark():
    diagnostics = watermark_report(torch.full((10,), SNR_AUDIBLE_DB - 5.0), torch.full((10,), 0.1))

    warnings = check_watermark_snr(diagnostics)

    assert any("audible" in w for w in warnings)


def test_check_watermark_snr_flags_exactly_zero_delta():
    """A zero delta reads as a *large* dB value (it's the denominator), so
    it must be called out from delta_rms, not inferred from the SNR."""
    snr_db = watermark_snr_db(torch.randn(4, 1, 4000), torch.randn(4, 1, 4000).clone())
    diagnostics = watermark_report(torch.full((4,), 30.0), torch.tensor([0.01, 0.01, 0.0, 0.01]))

    warnings = check_watermark_snr(diagnostics)

    assert any("exactly-zero" in w for w in warnings)
    assert snr_db.isfinite().all()


def test_check_watermark_snr_is_quiet_on_a_healthy_watermark():
    torch.manual_seed(0)
    diagnostics = watermark_report(
        24.0 + 6.0 * torch.rand(50), 0.01 + 0.001 * torch.rand(50)
    )

    assert check_watermark_snr(diagnostics) == []
    # A configured normalization target that the median sits inside passes.
    assert check_watermark_snr(diagnostics, snr_db_range=(24.0, 36.0)) == []


def test_check_watermark_snr_warns_on_wide_per_clip_spread():
    """A healthy median can hide clips with no usable watermark -- the
    spread is what makes that visible, so it gets its own gate."""
    snr_db = torch.linspace(5.0, 60.0, 40)  # median ~32 dB, but p5..p95 is huge
    diagnostics = watermark_report(snr_db, torch.full((40,), 0.01))

    warnings = check_watermark_snr(diagnostics)

    assert diagnostics["snr_db"]["p50"] == pytest.approx(32.5, abs=1.0)
    assert any("spread" in w for w in warnings)


def test_check_watermark_snr_asserts_against_configured_target():
    diagnostics = watermark_report(torch.full((10,), 30.0), torch.full((10,), 0.01))

    # Fixed target, +/- 1 dB: 30 dB passes against 30, fails against 25.
    check_watermark_snr(diagnostics, snr_db_range=(30.0, 30.0))
    check_watermark_snr(diagnostics, snr_db_range=(29.5, 29.5))
    with pytest.raises(AssertionError, match="outside the configured"):
        check_watermark_snr(diagnostics, snr_db_range=(25.0, 25.0))
    with pytest.raises(AssertionError, match="outside the configured"):
        check_watermark_snr(diagnostics, snr_db_range=(15.0, 20.0))


def test_stock_baseline_flag_sets_config_mode():
    assert load_eval_config(["eval_dir=."]).stock_baseline is False

    cfg = load_eval_config(["eval_dir=.", "--stock-baseline", "n_eval_batches=3"])

    assert cfg.stock_baseline is True
    assert cfg.n_eval_batches == 3  # the flag doesn't disturb the dotlist overrides


def test_apply_stock_baseline_overrides_checkpoints_and_attacks():
    cfg = load_eval_config([
        "eval_dir=.",
        "--stock-baseline",
        "generator_checkpoint=./checkpoints/generator_epoch10.pth",
        "eval_attacks=[identity,sgmse]",
    ])

    cfg = apply_stock_baseline(cfg)

    assert cfg.generator_checkpoint == STOCK_GENERATOR_CARD
    assert cfg.detector_checkpoint == STOCK_DETECTOR_CARD
    assert list(cfg.eval_attacks) == ["identity"]
    assert list(cfg.held_out_attacks) == []


def test_stock_baseline_verdict_passes_on_healthy_numbers():
    results = {
        "attacks": {"identity": {"detection_rate": 1.0, "bit_accuracy": 1.0}},
        "watermark": {"snr_db": {"p50": 30.0}},
    }

    passed, lines = stock_baseline_verdict(results)

    assert passed
    assert all("PASS" in line for line in lines)


def test_stock_baseline_verdict_checks_raw_snr_under_normalization():
    """Under SNR normalization the post-scaling median equals the target by
    construction, so checking it would PASS a generator emitting nothing.
    The verdict must read raw_snr_db instead."""
    results = {
        "attacks": {"identity": {"detection_rate": 1.0, "bit_accuracy": 1.0}},
        # Perfectly on-target post-scaling, but the generator emitted ~nothing.
        "watermark": {"snr_db": {"p50": 30.0}, "raw_snr_db": {"p50": 140.0}},
    }

    passed, lines = stock_baseline_verdict(results)

    assert not passed
    assert any("raw_snr_db" in line and "FAIL" in line for line in lines)


@pytest.mark.parametrize(
    "detection, bit_acc, median_snr",
    [
        (0.5, 1.0, 30.0),  # detector doesn't fire -> harness bug
        (1.0, 0.51, 30.0),  # message doesn't survive the round trip
        (1.0, 1.0, 80.0),  # generator emits an effectively-zero delta
        (1.0, 1.0, 5.0),  # perturbation is grossly too loud
    ],
)
def test_stock_baseline_verdict_fails_on_each_broken_axis(detection, bit_acc, median_snr):
    results = {
        "attacks": {"identity": {"detection_rate": detection, "bit_accuracy": bit_acc}},
        "watermark": {"snr_db": {"p50": median_snr}},
    }

    passed, lines = stock_baseline_verdict(results)

    assert not passed
    assert any("FAIL" in line for line in lines)


def test_stock_baseline_verdict_fails_when_a_number_is_missing():
    passed, lines = stock_baseline_verdict({"attacks": {}, "watermark": {}})

    assert not passed
    assert all("MISSING" in line for line in lines)


def test_evaluate_attack_uses_prepared_batches():
    class Detector(torch.nn.Module):
        def forward(self, x):
            detected = (x.abs().sum(dim=(1, 2)) > 0).float()
            presence = torch.stack([1 - detected, detected], dim=1).unsqueeze(-1)
            message = (x[:, 0, :4] > 0).float()
            return presence, message

    message = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]])
    x = torch.zeros(2, 1, 8)
    x_wm = x.clone()
    x_wm[:, 0, :4] = message * 2 - 1
    cfg = load_eval_config(["eval_dir=.", "nbits=4"])

    metrics = evaluate_attack(
        Detector(),
        IdentityAttack(),
        [(x, x_wm, message)],
        cfg,
        torch.device("cpu"),
    )

    assert metrics["bit_accuracy"] == 1.0
    assert metrics["tpr_at_fpr"] == 1.0
    assert metrics["confusion"] == {"tp": 2, "fn": 0, "fp": 0, "tn": 2}
    assert metrics["f1"] == 1.0


def test_evaluate_perceptual_aggregates_prepared_batches(monkeypatch):
    cfg = load_eval_config(["eval_dir=.", "compute_pesq=true", "compute_sisnr=true"])
    eval_batches = [
        (torch.zeros(1, 1, 4), torch.ones(1, 1, 4), torch.zeros(1, 4)),
        (torch.zeros(1, 1, 4), torch.full((1, 1, 4), 3.0), torch.zeros(1, 4)),
    ]
    monkeypatch.setattr("audioseal_robust.evaluate.sisnr_score", lambda x, x_wm: x_wm.mean().item())
    monkeypatch.setattr(
        "audioseal_robust.evaluate.pesq_score",
        lambda x, x_wm, sample_rate: 2 * x_wm.mean().item(),
    )

    metrics = evaluate_perceptual(eval_batches, cfg, torch.device("cpu"))

    assert metrics == {"sisnr": 2.0, "pesq": 4.0}


def test_gradients_flow_to_generator_only():
    torch.manual_seed(0)
    nbits = 4

    generator = _tiny_generator(nbits)
    detector = _tiny_detector(nbits)
    detector.eval()
    for p in detector.parameters():
        p.requires_grad_(False)

    attack = SampledReconstructionAttack({"identity": IdentityAttack()}, {"identity": 1.0})
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    detector_state_before = copy.deepcopy(detector.state_dict())

    x = torch.randn(2, 1, 4000)
    message = torch.randint(0, 2, (2, nbits))

    watermark = generator.get_watermark(x, message=message)
    x_wm = x + watermark
    assert x_wm.requires_grad

    x_att, name = attack(x_wm)
    assert name == "identity"
    assert x_att.requires_grad
    assert x_att.grad_fn is not None, "graph must stay connected through the frozen attack"

    presence, m_hat = detector.forward(x_att)
    p = presence[:, 1, :].mean(dim=-1)

    det_loss = detection_loss(p, m_hat, message.float())
    perc_loss_fn = PsychoacousticMelLoss(
        sample_rate=16_000, n_fft=256, hop_length=64, win_length=256, n_mels=16
    )
    perc_loss = perc_loss_fn(x, x_wm)
    total_loss = 1.0 * det_loss + 1.0 * perc_loss

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()

    generator_grads = [p.grad for p in generator.parameters() if p.requires_grad]
    assert any(
        g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in generator_grads
    ), "expected at least one non-zero, finite gradient on a generator parameter"

    assert all(p.grad is None for p in detector.parameters()), "detector must never accumulate gradient"

    optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    detector_param_ids = {id(p) for p in detector.parameters()}
    attack_param_ids = {id(p) for p in attack.parameters()}
    assert optimizer_param_ids.isdisjoint(detector_param_ids)
    assert optimizer_param_ids.isdisjoint(attack_param_ids)

    optimizer.step()

    detector_state_after = detector.state_dict()
    for key in detector_state_before:
        assert torch.equal(detector_state_before[key], detector_state_after[key]), (
            f"detector param {key} changed after optimizer.step()"
        )


def test_sampled_attack_only_picks_enabled_branches():
    attack = SampledReconstructionAttack(
        {"identity": IdentityAttack(), "other_identity": IdentityAttack()},
        {"identity": 1.0, "other_identity": 0.0},
    )
    x = torch.randn(1, 1, 100)
    for _ in range(20):
        _, name = attack(x)
        assert name == "identity"


def test_sgmse_attack_without_checkpoint_stays_a_stub():
    attack = SGMSEAttack()
    with pytest.raises(NotImplementedError, match="constructed without a checkpoint"):
        attack(torch.randn(1, 1, 1600))


def test_sgmse_attack_missing_checkpoint_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="SGMSE checkpoint not found"):
        SGMSEAttack(checkpoint=str(tmp_path / "no_such_checkpoint.ckpt"))


class _StubScoreModel(torch.nn.Module):
    """Stand-in for sgmse.model.ScoreModel exposing only what
    SGMSEAttack.forward touches, so the whole predictor/corrector loop can be
    exercised without a multi-GB checkpoint. The SDE is the real one; only the
    score network is replaced (by a linear function of x, so the loop stays
    numerically tame and any NaN that shows up came from the transforms or the
    sampler, not from an untrained backbone)."""

    def __init__(self, n_fft: int = 510, hop_length: int = 128):
        super().__init__()
        from sgmse.sdes import OUVESDE

        self.sde = OUVESDE(theta=1.5, sigma_min=0.05, sigma_max=0.5, N=30)
        self.t_eps = 0.03
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))
        self.seen_t = []

    def forward(self, x, y, t):
        self.seen_t.append(t.detach().clone())
        return -x

    def _stft(self, sig):
        return torch.stft(
            sig, self.n_fft, self.hop_length, window=self.window, center=True, return_complex=True
        )

    def _istft(self, spec, length=None):
        return torch.istft(
            spec, self.n_fft, self.hop_length, window=self.window, center=True, length=length
        )


def test_sgmse_spec_transforms_stay_finite_on_silence():
    """The vendored spec_fwd/spec_back route |z|**0.5 through abs()/angle(),
    whose gradient is NaN at z=0 -- fine upstream (its sampler is all
    no_grad), fatal here. Silent STFT bins are exactly z=0."""
    attack = SGMSEAttack()

    spec = torch.zeros(1, 4, dtype=torch.complex64)
    spec[0, 0] = 0.3 + 0.4j  # one live bin; the rest are exact silence
    spec.requires_grad_(True)

    roundtrip = attack._spec_back(attack._spec_fwd(spec))
    (roundtrip.real.sum() + roundtrip.imag.sum()).backward()

    assert spec.grad is not None
    assert torch.isfinite(spec.grad).all()


def test_sgmse_spec_transforms_match_upstream_away_from_zero():
    """The rewrite is only allowed to change behaviour at the singularity.
    Upstream's formulas are inlined here rather than imported, so this stays
    runnable without pulling pytorch_lightning in for SpecsDataModule --
    see sgmse/data_module.py:SpecsDataModule.spec_fwd/spec_back."""
    attack = SGMSEAttack()
    factor, exponent = attack._spec_factor, attack._spec_abs_exponent
    spec = torch.randn(2, 8, dtype=torch.complex64)

    upstream_fwd = spec.abs() ** exponent * torch.exp(1j * spec.angle()) * factor
    upstream_back = (spec / factor).abs() ** (1 / exponent) * torch.exp(1j * (spec / factor).angle())

    assert torch.allclose(attack._spec_fwd(spec), upstream_fwd, atol=1e-6)
    assert torch.allclose(attack._spec_back(spec), upstream_back, atol=1e-6)


def test_sde_discretize_accepts_per_example_stepsize():
    """Per-example t* means per-example step sizes; upstream's discretize only
    broadcasts a scalar (see src/sgmse/VENDORED.md). The scalar path must be
    bit-identical to before."""
    from sgmse.sdes import OUVESDE

    sde = OUVESDE(theta=1.5, sigma_min=0.05, sigma_max=0.5, N=5)
    x = torch.randn(2, 1, 8, 4)
    y = torch.randn(2, 1, 8, 4)
    t = torch.tensor([0.2, 0.7])

    f_vec, g_vec = sde.discretize(x, y, t, torch.tensor([0.1, 0.1]))
    f_scalar, g_scalar = sde.discretize(x, y, t, torch.tensor(0.1))

    assert f_vec.shape == x.shape
    assert g_vec.shape == (2,)
    assert torch.allclose(f_vec, f_scalar)
    assert torch.allclose(g_vec, g_scalar)


def test_sgmse_attack_runs_end_to_end_and_keeps_gradients_finite():
    """The gap this closes: every other SGMSE test here stops at the
    no-checkpoint stub, so the sampler loop itself was never executed."""
    attack = SGMSEAttack(num_steps=3)
    attack._model = _StubScoreModel()

    x = torch.randn(3, 1, 4000)
    x[:, :, :1000] = 0.0  # a silent stretch -- the case that produces NaN grads
    x.requires_grad_(True)

    torch.manual_seed(0)
    x_att = attack(x)

    assert x_att.shape == x.shape
    assert torch.isfinite(x_att).all()

    x_att.pow(2).mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_sgmse_attack_samples_one_strength_per_example():
    """strength=None must spread t* across the batch: one shared draw would
    give a batch-size-8 step a single point of the robustness curve."""
    attack = SGMSEAttack(num_steps=2)
    attack._model = _StubScoreModel()

    torch.manual_seed(0)
    attack(torch.randn(4, 1, 4000))

    first_t = attack._model.seen_t[0]
    assert first_t.shape == (4,)
    assert first_t.unique().numel() == 4


def test_sgmse_attack_explicit_strength_pins_the_whole_batch():
    """evaluate.py measures one t* at a time -- a float must not be perturbed
    per example the way strength=None now is."""
    attack = SGMSEAttack(num_steps=2)
    attack._model = _StubScoreModel()

    attack(torch.randn(4, 1, 4000), strength=0.04)

    first_t = attack._model.seen_t[0]
    assert first_t.unique().numel() == 1


def test_mbd_attack_without_checkpoint_stays_a_stub():
    attack = MBDAttack()
    with pytest.raises(NotImplementedError, match="constructed without a checkpoint"):
        attack(torch.randn(1, 1, 1600))


def test_diff_erase_attack_without_checkpoint_stays_a_stub():
    attack = DiffEraseAttack()
    with pytest.raises(NotImplementedError, match="constructed without a checkpoint"):
        attack(torch.randn(1, 1, 1600))


def test_diff_erase_attack_checkpoint_without_config_raises_clear_error(tmp_path):
    fake_ckpt = tmp_path / "fake.ckpt"
    fake_ckpt.write_bytes(b"not a real checkpoint")
    with pytest.raises(NotImplementedError, match="needs `config` set"):
        DiffEraseAttack(checkpoint=str(fake_ckpt))


def test_diff_erase_attack_checkpoint_wrong_layout_fails_before_any_import(tmp_path):
    """checkpoint must live at <weights_root>/data/checkpoints/<file> -- that's
    the relative layout get_vocoder()/reload_from_ckpt hardcode. A checkpoint
    anywhere else should get a clear error, not a confusing failure deep
    inside model construction."""
    fake_ckpt = tmp_path / "fake.ckpt"  # NOT under data/checkpoints/
    fake_ckpt.write_bytes(b"not a real checkpoint")
    fake_config = tmp_path / "fake.yaml"
    fake_config.write_text("preprocessing: {}\n")
    with pytest.raises(ValueError, match="data/checkpoints"):
        DiffEraseAttack(checkpoint=str(fake_ckpt), config=str(fake_config))


def _tacotron_stft():
    """Skips unless the diff_erase extras are installed -- audioldm_train pulls
    in pandas et al. (see requirements-diff-erase.txt), which the base
    training/eval stack deliberately does not."""
    stft = pytest.importorskip("audioldm_train.utilities.audio.stft")
    return stft.TacotronSTFT(
        filter_length=1024,
        hop_length=160,
        win_length=1024,
        n_mel_channels=64,
        sampling_rate=16_000,
        mel_fmin=0,
        mel_fmax=8000,
    )


def test_audioldm_mel_reaches_the_waveform():
    """DiffEraseAttack is eval-only here, but making it trainable (the point of
    the diff_erase training direction) requires this front-end to be
    differentiable at all. `mel_spectrogram` used to call `.data`, which
    detached -- so the mel came back with grad_fn None and the detection loss
    silently stopped reaching the generator."""
    wav = (torch.randn(1, 16_000) * 0.1).clamp(-1, 1).requires_grad_(True)

    mel, *_ = _tacotron_stft().mel_spectrogram(wav)

    assert mel.grad_fn is not None
    mel.sum().backward()
    assert wav.grad is not None
    assert wav.grad.abs().sum() > 0


def test_audioldm_mel_gradients_stay_finite_on_silence():
    """Reconnecting the graph above exposes STFT.transform's
    sqrt(real**2 + imag**2), whose derivative is infinite at exactly zero --
    i.e. on digital silence, which every padded/clipped segment has."""
    wav = torch.zeros(1, 16_000, requires_grad=True)

    mel, *_ = _tacotron_stft().mel_spectrogram(wav)
    mel.sum().backward()

    assert wav.grad is not None
    assert torch.isfinite(wav.grad).all()


def test_build_eval_attacks_threads_per_attack_config_through():
    """Regression test: build_eval_attacks used to construct every attack
    with zero args, silently ignoring cfg.attack.* entirely (a config field
    like attack.sgmse.num_steps had no effect). Confirms the fix actually
    reaches the constructed module."""
    cfg = load_eval_config(["eval_dir=.", "attack.sgmse.num_steps=7"])
    attacks, skipped = build_eval_attacks(["identity", "sgmse"], torch.device("cpu"), cfg)
    assert not skipped
    assert attacks["sgmse"].num_steps == 7


def test_train_recipe_diff_erase_sets_weights():
    """recipe=diff_erase (config/recipes.yaml) should train against
    DiffErase and zero out the identity-only default.yaml fallback."""
    cfg = load_config(["recipe=diff_erase", "data.train_dir=."])
    assert cfg.attack.weights.identity == 0.0
    assert cfg.attack.weights.diff_erase == 1.0
    assert cfg.attack.weights.sgmse == 0.0


def test_train_recipe_sgmse_sets_weights():
    """The opposite direction: recipe=sgmse trains against SGMSE instead,
    leaving diff_erase at its default (disabled) weight."""
    cfg = load_config(["recipe=sgmse", "data.train_dir=."])
    assert cfg.attack.weights.identity == 0.0
    assert cfg.attack.weights.sgmse == 1.0
    assert cfg.attack.weights.diff_erase == 0.0


def test_train_recipe_cli_override_wins_over_recipe():
    """Regression test for merge order: CLI overrides must be applied AFTER
    the recipe, so an explicit CLI value for a field the recipe also sets
    still wins (letting you tweak one field without forking the recipe)."""
    cfg = load_config(["recipe=diff_erase", "data.train_dir=.", "attack.weights.diff_erase=0.5"])
    assert cfg.attack.weights.diff_erase == 0.5


def test_eval_recipe_after_sgmse_training_swaps_held_out():
    """The eval-side recipe should report sgmse (not diff_erase) as the
    trained attack and hold diff_erase out instead -- the reverse of
    default_eval.yaml's plain fallback."""
    cfg = load_eval_config(["recipe=after_sgmse_training", "eval_dir=."])
    assert list(cfg.eval_attacks) == ["identity", "bigvgan", "dac", "sgmse"]
    assert list(cfg.held_out_attacks) == ["diff_erase", "mbd"]


def test_unknown_recipe_raises_with_available_names():
    with pytest.raises(ValueError, match="Unknown train recipe"):
        load_config(["recipe=nonexistent", "data.train_dir=."])


def test_build_eval_attacks_construction_failure_is_skipped_not_fatal():
    """Regression test: once a real checkpoint path is threaded through (the
    fix above), a misconfigured attack can now fail inside __init__/
    _load_backbone instead of only at forward() time -- that must land in
    the `skipped` dict, not raise and take down the whole eval run."""
    cfg = load_eval_config(["eval_dir=.", "attack.diff_erase.checkpoint=/nonexistent/fake.ckpt"])
    attacks, skipped = build_eval_attacks(["identity", "diff_erase"], torch.device("cpu"), cfg)
    assert "diff_erase" not in attacks
    assert "config" in skipped["diff_erase"]
    assert attacks["identity"] is not None
