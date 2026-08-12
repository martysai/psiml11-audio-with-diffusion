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

import pytest
import torch

from audioseal.builder import (
    AudioSealDetectorConfig,
    AudioSealWMConfig,
    DecoderConfig,
    DetectorConfig,
    SEANetConfig,
    create_detector,
    create_generator,
)
from audioseal_robust.attacks import DiffEraseAttack, IdentityAttack, SampledReconstructionAttack
from audioseal_robust.config import load_eval_config
from audioseal_robust.evaluate import build_eval_attacks
from audioseal_robust.losses import PsychoacousticMelLoss, detection_loss


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


def test_build_eval_attacks_threads_per_attack_config_through():
    """Regression test: build_eval_attacks used to construct every attack
    with zero args, silently ignoring cfg.attack.* entirely (a config field
    like attack.sgmse.num_steps had no effect). Confirms the fix actually
    reaches the constructed module."""
    cfg = load_eval_config(["eval_dir=.", "attack.sgmse.num_steps=7"])
    attacks, skipped = build_eval_attacks(["identity", "sgmse"], torch.device("cpu"), cfg)
    assert not skipped
    assert attacks["sgmse"].num_steps == 7


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
