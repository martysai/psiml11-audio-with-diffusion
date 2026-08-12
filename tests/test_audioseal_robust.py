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
from types import SimpleNamespace

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
from audioseal_robust.evaluate import build_eval_attacks
from audioseal_robust.losses import PsychoacousticMelLoss, detection_loss
from audioseal_robust.train import embed_watermark, validate


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


def test_validate_does_not_update_generator_and_restores_train_mode():
    """Regression test for train.py's in-training validation (data.valid_dir):
    validate() must never touch generator parameters/gradients (it's a
    read-only health check, not a second optimizer path) and must leave the
    generator back in train() mode so the next real training step behaves
    normally -- even though it puts the generator in eval() internally for
    the duration of the validation batches."""
    torch.manual_seed(0)
    nbits = 4

    generator = _tiny_generator(nbits)
    generator.train()
    detector = _tiny_detector(nbits)
    detector.eval()
    for p in detector.parameters():
        p.requires_grad_(False)

    attack = SampledReconstructionAttack({"identity": IdentityAttack()}, {"identity": 1.0})
    perc_loss_fn = PsychoacousticMelLoss(
        sample_rate=16_000, n_fft=256, hop_length=64, win_length=256, n_mels=16
    )

    params_before = [p.detach().clone() for p in generator.parameters()]
    fake_valid_batches = [torch.randn(2, 1, 4000) for _ in range(3)]
    cfg = SimpleNamespace(
        nbits=nbits,
        watermark_snr_db_min=24.0,
        watermark_snr_db_max=36.0,
        lambda_det=1.0,
        lambda_perc=1.0,
        valid_batches=2,  # less than len(fake_valid_batches): confirms the cap is honored
    )

    metrics = validate(generator, detector, attack, perc_loss_fn, fake_valid_batches, cfg, torch.device("cpu"))

    assert generator.training, "validate() must restore train() mode before returning"
    assert all(p.grad is None for p in generator.parameters()), "validate() must never leave gradients on the generator"
    assert all(
        torch.equal(before, after) for before, after in zip(params_before, generator.parameters())
    ), "validate() must never update generator parameters"
    assert set(metrics.keys()) == {"loss", "detection_loss", "perceptual_loss", "presence_prob"}


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
