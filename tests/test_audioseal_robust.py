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
    AttackApplicationReporter,
    AudioLDMAttack,
    HopSkipJumpAttack,
    IdentityAttack,
    MBDAttack,
    SampledReconstructionAttack,
    SGMSEAttack,
)
from audioseal_robust.config import load_config, load_eval_config
from audioseal_robust.evaluate import (
    _print_results_table,
    build_eval_attacks,
    evaluate_attack,
    evaluate_perceptual,
    load_generator_under_test,
    prepare_eval_batches,
)
from audioseal_robust.losses import PsychoacousticMelLoss, detection_loss
from audioseal_robust.metrics import fpr_support
from audioseal_robust.model_init import build_untrained_generator
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
    first = prepare_eval_batches(generator, dataloader, cfg, torch.device("cpu"))
    torch.manual_seed(123)
    second = prepare_eval_batches(generator, dataloader, cfg, torch.device("cpu"))

    assert len(first) == 1  # n_eval_batches caps the dataloader
    assert all(torch.equal(a, b) for a, b in zip(first[0], second[0]))


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


def test_audioldm_attack_without_checkpoint_stays_a_stub():
    attack = AudioLDMAttack()
    with pytest.raises(NotImplementedError, match="constructed without a checkpoint"):
        attack(torch.randn(1, 1, 1600))


def test_audioldm_attack_checkpoint_without_config_raises_clear_error(tmp_path):
    fake_ckpt = tmp_path / "fake.ckpt"
    fake_ckpt.write_bytes(b"not a real checkpoint")
    with pytest.raises(NotImplementedError, match="needs `config` set"):
        AudioLDMAttack(checkpoint=str(fake_ckpt))


def test_audioldm_attack_checkpoint_wrong_layout_fails_before_any_import(tmp_path):
    """checkpoint must live at <weights_root>/data/checkpoints/<file> -- that's
    the relative layout get_vocoder()/reload_from_ckpt hardcode. A checkpoint
    anywhere else should get a clear error, not a confusing failure deep
    inside model construction."""
    fake_ckpt = tmp_path / "fake.ckpt"  # NOT under data/checkpoints/
    fake_ckpt.write_bytes(b"not a real checkpoint")
    fake_config = tmp_path / "fake.yaml"
    fake_config.write_text("preprocessing: {}\n")
    with pytest.raises(ValueError, match="data/checkpoints"):
        AudioLDMAttack(checkpoint=str(fake_ckpt), config=str(fake_config))


def _tacotron_stft():
    """Skips unless the audioldm extras are installed -- audioldm_train pulls
    in pandas et al. (see requirements-audioldm.txt), which the base
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
    """AudioLDMAttack is eval-only here, but making it trainable (the point of
    the audioldm training direction) requires this front-end to be
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


class _LevelDetector(torch.nn.Module):
    """Deterministic, non-trained stand-in for AudioSealDetector: decides
    "watermarked" purely from mean absolute amplitude vs. a threshold. Used
    to test HopSkipJumpAttack's search logic exactly (a real detector's
    decision surface can't be reasoned about analytically), matching the
    stub-Detector style already used by test_evaluate_attack_uses_prepared_batches."""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        level = x.abs().mean(dim=(1, 2))
        watermarked = (level > self.threshold).float()
        presence = torch.stack([1 - watermarked, watermarked], dim=1).unsqueeze(-1)
        message = torch.zeros(x.shape[0], 1)
        return presence, message


def test_hopskipjump_attack_without_checkpoint_stays_a_stub():
    attack = HopSkipJumpAttack()
    attack.bind_detector(_LevelDetector())
    with pytest.raises(NotImplementedError, match="without being enabled"):
        attack(torch.randn(1, 1, 1600))


def test_hopskipjump_attack_requires_a_bound_detector():
    attack = HopSkipJumpAttack(checkpoint="auto")
    with pytest.raises(RuntimeError, match="no detector bound"):
        attack(torch.randn(1, 1, 1600))


def test_hopskipjump_attack_finds_a_decision_flip_near_the_boundary():
    """HopSkipJumpAttack should return a waveform the detector decides
    DIFFERENTLY on than x0 itself (the "label0" framing in its class
    docstring -- correct for both the watermarked and clean branch
    evaluate_attack calls it on), landing close to the decision boundary
    thanks to binary search rather than just returning the crude random
    initialization."""
    torch.manual_seed(0)
    detector = _LevelDetector(threshold=0.5)
    attack = HopSkipJumpAttack(
        checkpoint="auto",
        num_iterations=15,
        init_num_evals=10,
        max_num_evals=30,
        init_max_trials=200,
        binary_search_steps=10,
    )
    attack.bind_detector(detector)

    x0 = torch.full((1, 1, 2000), 0.9)
    label0 = bool((detector.forward(x0)[0][:, 1, :].mean(dim=-1) > 0.5).item())
    assert label0 is True  # sanity: x0 starts on the "watermarked" side

    x_adv = attack(x0)

    assert x_adv.shape == x0.shape
    assert x_adv.dtype == x0.dtype
    label_adv = bool((detector.forward(x_adv)[0][:, 1, :].mean(dim=-1) > 0.5).item())
    assert label_adv is False  # decision flipped

    level_adv = x_adv.abs().mean().item()
    assert 0.3 < level_adv < 0.5  # close to the threshold, not raw random noise


def test_hopskipjump_attack_gives_up_gracefully_when_init_never_flips():
    """A detector that ALWAYS returns the same decision (e.g. a degenerate
    or saturated one) has no adversarial region for random init to find --
    the attack should return x0 unperturbed (and log a warning) rather than
    hang or raise, per _initialize's documented fallback."""

    class ConstantDetector(torch.nn.Module):
        def forward(self, x):
            presence = torch.zeros(x.shape[0], 2, 1)
            presence[:, 0, :] = 1.0  # always "not watermarked"
            return presence, torch.zeros(x.shape[0], 1)

    attack = HopSkipJumpAttack(checkpoint="auto", init_max_trials=5)
    attack.bind_detector(ConstantDetector())

    x0 = torch.randn(1, 1, 500)
    x_adv = attack(x0)

    assert torch.equal(x_adv, x0)
    # The give-up must be *reported*, not just logged: an unperturbed
    # watermarked example still detects, so without this it would be scored
    # as watermark robustness (see AttackApplicationReporter).
    mask = attack.pop_application_mask()
    assert mask is not None
    assert mask.tolist() == [False]


def test_hopskipjump_application_mask_reports_success_and_drains():
    """The mask is per-forward: a successful attack reports True, and
    reading it twice must not replay the previous call's result."""
    torch.manual_seed(0)
    attack = HopSkipJumpAttack(checkpoint="auto", num_iterations=2, init_max_trials=200)
    attack.bind_detector(_LevelDetector(threshold=0.5))

    attack(torch.full((1, 1, 500), 0.9))

    assert attack.pop_application_mask().tolist() == [True]
    assert attack.pop_application_mask() is None  # drained


def test_hopskipjump_initializes_from_a_bound_reference_pool():
    """With the noise fallback disabled (init_max_trials=0), the attack can
    only start from the bound pool -- so this isolates that path: it fails
    without a pool and succeeds with one."""
    detector = _LevelDetector(threshold=0.5)
    x0 = torch.full((1, 1, 500), 0.9)  # "watermarked" side

    without_pool = HopSkipJumpAttack(checkpoint="auto", num_iterations=2, init_max_trials=0)
    without_pool.bind_detector(detector)
    assert torch.equal(without_pool(x0), x0)
    assert without_pool.pop_application_mask().tolist() == [False]

    with_pool = HopSkipJumpAttack(checkpoint="auto", num_iterations=2, init_max_trials=0)
    with_pool.bind_detector(detector)
    with_pool.bind_reference_pool(torch.full((1, 1, 500), 0.1))  # "clean" side
    x_adv = with_pool(x0)

    assert with_pool.pop_application_mask().tolist() == [True]
    assert not torch.equal(x_adv, x0)
    assert bool((detector.forward(x_adv)[0][:, 1, :].mean(dim=-1) > 0.5).item()) is False


def test_hopskipjump_reference_pool_skips_the_examples_own_counterpart():
    """A pool built from the eval batches necessarily contains x0's own
    clean/watermarked counterpart. Starting from it would hand the attacker
    the original signal, so entries within _SELF_REFERENCE_REL_L2 must be
    skipped even though their detector label is the opposite one."""
    detector = _LevelDetector(threshold=0.5)
    x0 = torch.full((1, 1, 500), 0.55)  # watermarked side, just above threshold
    counterpart = torch.full((1, 1, 500), 0.48)  # opposite label, ~13% away in L2
    unrelated = torch.full((1, 1, 500), 0.1)  # opposite label, ~82% away

    only_counterpart = HopSkipJumpAttack(checkpoint="auto", num_iterations=2, init_max_trials=0)
    only_counterpart.bind_detector(detector)
    only_counterpart.bind_reference_pool(counterpart)
    assert torch.equal(only_counterpart(x0), x0)  # skipped -> no init -> gave up
    assert only_counterpart.pop_application_mask().tolist() == [False]

    with_unrelated = HopSkipJumpAttack(checkpoint="auto", num_iterations=2, init_max_trials=0)
    with_unrelated.bind_detector(detector)
    with_unrelated.bind_reference_pool(torch.cat([counterpart, unrelated]))
    assert not torch.equal(with_unrelated(x0), x0)  # the far one is still usable
    assert with_unrelated.pop_application_mask().tolist() == [True]


def test_hopskipjump_init_from_reference_can_be_disabled():
    detector = _LevelDetector(threshold=0.5)
    attack = HopSkipJumpAttack(
        checkpoint="auto", num_iterations=2, init_max_trials=0, init_from_reference=False
    )
    attack.bind_detector(detector)
    attack.bind_reference_pool(torch.full((1, 1, 500), 0.1))

    x0 = torch.full((1, 1, 500), 0.9)
    assert torch.equal(attack(x0), x0)  # pool ignored, and noise fallback disabled


def test_print_results_table_tolerates_missing_perceptual_metrics(capsys):
    """compute_pesq=false (or a missing `pesq` package) leaves that key out
    entirely -- the summary must still print, since it runs only after the
    expensive attack passes have already completed."""
    _print_results_table({"label": "t", "perceptual": {"sisnr": 30.0}, "attacks": {}})

    out = capsys.readouterr().out
    assert "SI-SNR: 30.00 dB" in out
    assert "PESQ" not in out


def test_print_results_table_flags_failures_and_thin_negatives(capsys):
    """Both caveats must appear inline in the table, not just in the log."""
    _print_results_table(
        {
            "label": "t",
            "attacks": {
                "hopskipjump": {
                    "tag": "held_out",
                    "bit_accuracy": 0.9,
                    "tpr_at_fpr": 0.8,
                    "tpr_at_fpr_attacked": 0.2,
                    "f1": 0.5,
                    "attack_failure_rate": 0.75,
                    "n_attack_failures": 3,
                    "fpr_support": {
                        "n_negatives": 4,
                        "fpr_resolution": 0.25,
                        "min_negatives_for_target": 100,
                        "supported": False,
                    },
                }
            },
        }
    )

    out = capsys.readouterr().out
    assert "did not perturb 3 example(s)" in out
    assert "perturbed-only: 0.200" in out
    assert "only 4 negatives" in out


def test_fpr_support_flags_a_sample_size_that_cannot_resolve_the_target():
    """fpr_target=0.01 needs >= 100 negatives; below that int(0.01 * N) == 0
    and the threshold degenerates to the max negative score."""
    too_few = fpr_support(n_negatives=16, target_fpr=0.01)
    assert too_few["supported"] is False
    assert too_few["min_negatives_for_target"] == 100
    assert too_few["fpr_resolution"] == pytest.approx(1 / 16)

    enough = fpr_support(n_negatives=100, target_fpr=0.01)
    assert enough["supported"] is True
    assert enough["fpr_resolution"] == pytest.approx(0.01)


def test_evaluate_attack_reports_fpr_support_and_attack_failures():
    """evaluate_attack must surface both caveats: too few negatives to
    resolve fpr_target, and examples the attack never perturbed."""

    class Detector(torch.nn.Module):
        def forward(self, x):
            detected = (x.abs().sum(dim=(1, 2)) > 0).float()
            presence = torch.stack([1 - detected, detected], dim=1).unsqueeze(-1)
            return presence, (x[:, 0, :4] > 0).float()

    class HalfFailingAttack(torch.nn.Module, AttackApplicationReporter):
        """Perturbs nothing, and reports the second example of every batch as
        an application failure -- the shape of HopSkipJump's give-up path."""

        def forward(self, x, strength=None):
            self._reset_application_mask()
            for i in range(x.shape[0]):
                self._record_application(i == 0)
            return x

    message = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]])
    x = torch.zeros(2, 1, 8)
    x_wm = x.clone()
    x_wm[:, 0, :4] = message * 2 - 1
    cfg = load_eval_config(["eval_dir=.", "nbits=4"])

    metrics = evaluate_attack(
        Detector(), HalfFailingAttack(), [(x, x_wm, message)], cfg, torch.device("cpu")
    )

    assert metrics["n_attack_failures"] == 2  # one per branch, pos + neg
    assert metrics["attack_failure_rate"] == pytest.approx(0.5)
    assert "tpr_at_fpr_attacked" in metrics
    # 2 negatives cannot resolve a 1% FPR.
    assert metrics["fpr_support"]["supported"] is False
    assert metrics["fpr_support"]["n_negatives"] == 2


def test_evaluate_attack_omits_failure_keys_for_plain_attacks():
    """Attacks that don't implement the reporter are 'applied to
    everything' -- no failure keys, so existing numbers stay comparable."""

    class Detector(torch.nn.Module):
        def forward(self, x):
            detected = (x.abs().sum(dim=(1, 2)) > 0).float()
            presence = torch.stack([1 - detected, detected], dim=1).unsqueeze(-1)
            return presence, (x[:, 0, :4] > 0).float()

    message = torch.tensor([[1, 0, 1, 0]])
    x = torch.zeros(1, 1, 8)
    x_wm = x.clone()
    x_wm[:, 0, :4] = message * 2 - 1

    metrics = evaluate_attack(
        Detector(),
        IdentityAttack(),
        [(x, x_wm, message)],
        load_eval_config(["eval_dir=.", "nbits=4"]),
        torch.device("cpu"),
    )

    assert "attack_failure_rate" not in metrics
    assert "tpr_at_fpr_attacked" not in metrics
    assert "fpr_support" in metrics  # this one is always reported


def test_load_generator_under_test_accepts_a_legacy_named_checkpoint(tmp_path):
    """Regression test: train.py's checkpoints (torch.save({"model":
    generator.state_dict()...})) use the pre-torchscripting-update flat conv
    naming (".conv.weight"/".conv.bias"), but this Python (>=3.10) build's
    AudioSealWM wraps those an extra "inner_conv" level (Moshi's SEANet) --
    load_generator_under_test's .pth branch called audioseal_load_state_dict
    directly, skipping the convert_state_dict_for_scriptable_model step that
    AudioSeal.load_generator/load_detector already apply for the model-card
    path, and raised "Missing/Unexpected key(s)" for every conv layer on a
    real fine-tuned checkpoint saved by train.py running on a different box."""
    reference = build_untrained_generator(nbits=4, device=torch.device("cpu"))
    legacy_state = {k.replace("inner_conv.", ""): v for k, v in reference.state_dict().items()}
    assert legacy_state != reference.state_dict()  # sanity: the rename actually did something

    ckpt_path = tmp_path / "generator_epoch3.pth"
    torch.save({"model": legacy_state}, ckpt_path)

    loaded = load_generator_under_test(str(ckpt_path), nbits=4, device=torch.device("cpu"))

    for (name, ref_param), (loaded_name, loaded_param) in zip(
        reference.state_dict().items(), loaded.state_dict().items()
    ):
        assert name == loaded_name
        assert torch.equal(ref_param, loaded_param)


def test_build_eval_attacks_threads_hopskipjump_config_through():
    cfg = load_eval_config(
        ["eval_dir=.", "attack.hopskipjump.checkpoint=auto", "attack.hopskipjump.num_iterations=3"]
    )
    attacks, skipped = build_eval_attacks(["identity", "hopskipjump"], torch.device("cpu"), cfg)
    assert not skipped
    assert attacks["hopskipjump"].num_iterations == 3


def test_build_eval_attacks_threads_per_attack_config_through():
    """Regression test: build_eval_attacks used to construct every attack
    with zero args, silently ignoring cfg.attack.* entirely (a config field
    like attack.sgmse.num_steps had no effect). Confirms the fix actually
    reaches the constructed module."""
    cfg = load_eval_config(["eval_dir=.", "attack.sgmse.num_steps=7"])
    attacks, skipped = build_eval_attacks(["identity", "sgmse"], torch.device("cpu"), cfg)
    assert not skipped
    assert attacks["sgmse"].num_steps == 7


def test_train_recipe_audioldm_sets_weights():
    """recipe=audioldm (config/recipes.yaml) should train against
    AudioLDM and zero out the identity-only default.yaml fallback."""
    cfg = load_config(["recipe=audioldm", "data.train_dir=."])
    assert cfg.attack.weights.identity == 0.0
    assert cfg.attack.weights.audioldm == 1.0
    assert cfg.attack.weights.sgmse == 0.0


def test_train_recipe_sgmse_sets_weights():
    """The opposite direction: recipe=sgmse trains against SGMSE instead,
    leaving audioldm at its default (disabled) weight."""
    cfg = load_config(["recipe=sgmse", "data.train_dir=."])
    assert cfg.attack.weights.identity == 0.0
    assert cfg.attack.weights.sgmse == 1.0
    assert cfg.attack.weights.audioldm == 0.0


def test_train_recipe_cli_override_wins_over_recipe():
    """Regression test for merge order: CLI overrides must be applied AFTER
    the recipe, so an explicit CLI value for a field the recipe also sets
    still wins (letting you tweak one field without forking the recipe)."""
    cfg = load_config(["recipe=audioldm", "data.train_dir=.", "attack.weights.audioldm=0.5"])
    assert cfg.attack.weights.audioldm == 0.5


def test_eval_recipe_after_sgmse_training_swaps_held_out():
    """The eval-side recipe should report sgmse (not audioldm) as the
    trained attack and hold audioldm out instead -- the reverse of
    default_eval.yaml's plain fallback."""
    cfg = load_eval_config(["recipe=after_sgmse_training", "eval_dir=."])
    assert list(cfg.eval_attacks) == ["identity", "bigvgan", "dac", "sgmse"]
    assert list(cfg.held_out_attacks) == ["audioldm", "mbd"]


def test_unknown_recipe_raises_with_available_names():
    with pytest.raises(ValueError, match="Unknown train recipe"):
        load_config(["recipe=nonexistent", "data.train_dir=."])


def test_build_eval_attacks_construction_failure_is_skipped_not_fatal():
    """Regression test: once a real checkpoint path is threaded through (the
    fix above), a misconfigured attack can now fail inside __init__/
    _load_backbone instead of only at forward() time -- that must land in
    the `skipped` dict, not raise and take down the whole eval run."""
    cfg = load_eval_config(["eval_dir=.", "attack.audioldm.checkpoint=/nonexistent/fake.ckpt"])
    attacks, skipped = build_eval_attacks(["identity", "audioldm"], torch.device("cpu"), cfg)
    assert "audioldm" not in attacks
    assert "config" in skipped["audioldm"]
    assert attacks["identity"] is not None
