# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluation harness: run a generator+detector pair through a set of
attacks (including a held-out one never seen during training) and report
robustness + perceptual metrics, logged per-run so baseline and
post-fine-tuning numbers land side by side in the same tracking experiment.

Run this FIRST, before any fine-tuning, to get the number you're trying to
beat:
    python -m audioseal_robust.evaluate eval_dir=/path/to/heldout/wavs label=baseline

Then again after fine-tuning, same eval set, pointing at the checkpoint:
    python -m audioseal_robust.evaluate eval_dir=/path/to/heldout/wavs \\
        label=finetuned_epoch10 \\
        generator_checkpoint=./checkpoints/audioseal_robust/generator_epoch10.pth

Attacks that are still stubs (see attacks.py -- bigvgan, dac, sgmse raise
NotImplementedError until a checkpoint is configured; diff_erase is wired to
DiffErase-latent, see attacks.py's DiffEraseAttack and
EvalAttackConfig.diff_erase in config.py, but still raises the same way if
its checkpoint/config/differase_root aren't set) are individually caught and
reported as "skipped", not fatal -- so this is runnable today, giving you
the identity-attack + perceptual baseline immediately, and picks up each
attack's real numbers as its checkpoint gets configured without any change
to this script.
"""

import logging
import typing as tp
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from audioseal import AudioSeal
from audioseal.loader import load_state_dict as audioseal_load_state_dict
from audioseal.models import AudioSealDetector, AudioSealWM

from .attacks import BigVGANAttack, DACAttack, DiffEraseAttack, IdentityAttack, SGMSEAttack
from .config import EvalConfig, load_eval_config
from .data import build_dataloader
from .device import resolve_device
from .metrics import bit_accuracy, confusion_counts, f1_score, pesq_score, sisnr_score, tpr_at_fpr, visqol_score
from .model_init import build_untrained_generator
from .plotting import plot_confusion_matrices, plot_robustness_curve
from .train import random_message
from .tracking import build_tracker

logger = logging.getLogger(__name__)

_ATTACK_CLASSES: tp.Dict[str, tp.Type[nn.Module]] = {
    "identity": IdentityAttack,
    "bigvgan": BigVGANAttack,
    "dac": DACAttack,
    "sgmse": SGMSEAttack,
    "diff_erase": DiffEraseAttack,
}


def load_generator_under_test(checkpoint: str, nbits: int, device: torch.device) -> AudioSealWM:
    """`checkpoint` is either a model card name / HF uri (-> vanilla
    pretrained, the baseline) or a local path to a .pth saved by train.py
    (-> a fine-tuned generator, built on the real architecture + that
    state_dict)."""
    path = Path(checkpoint)
    if path.suffix == ".pth" and path.exists():
        logger.info("loading fine-tuned generator from %s", path)
        state = torch.load(path, map_location=device, weights_only=False)
        generator = build_untrained_generator(nbits=nbits, device=device)
        audioseal_load_state_dict(generator, state["model"])
        return generator
    logger.info("loading generator checkpoint/card %r", checkpoint)
    return AudioSeal.load_generator(checkpoint, nbits=nbits, device=device)


_CONSTRUCTION_SKIP_EXCEPTIONS = (NotImplementedError, FileNotFoundError, ModuleNotFoundError)


def build_eval_attacks(
    names: tp.List[str], device: torch.device, cfg: EvalConfig
) -> tp.Tuple[tp.Dict[str, nn.Module], tp.Dict[str, str]]:
    """Builds one instance per requested attack name, constructed with that
    attack's `cfg.attack.<name>` sub-config (checkpoint, and whatever else
    that attack class takes -- see attacks.py). `identity` has no matching
    sub-config, so it's always built with defaults.

    Unlike `evaluate_attack`'s forward-time failures, a still-stubbed or
    misconfigured attack (missing checkpoint, missing companion
    config/differase_root, or a backbone package that isn't installed in
    this env) can now fail at *construction* time too, since a real
    checkpoint path is actually threaded through. That's caught here rather
    than left to crash the whole run: returns `(attacks, skipped)`, the
    latter mapping name -> error message for run()'s loop to report exactly
    like a forward-time "skipped" (see `_CONSTRUCTION_SKIP_EXCEPTIONS`).
    """
    attacks: tp.Dict[str, nn.Module] = {}
    skipped: tp.Dict[str, str] = {}
    for name in names:
        if name not in _ATTACK_CLASSES:
            raise ValueError(f"Unknown attack {name!r}, expected one of {sorted(_ATTACK_CLASSES)}")
        attack_cfg = getattr(cfg.attack, name, None)
        kwargs = tp.cast(tp.Dict[str, tp.Any], OmegaConf.to_container(attack_cfg)) if attack_cfg is not None else {}
        try:
            module = _ATTACK_CLASSES[name](**kwargs)
        except _CONSTRUCTION_SKIP_EXCEPTIONS as e:
            skipped[name] = str(e)
            continue
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
        attacks[name] = module.to(device)
    return attacks, skipped


@torch.no_grad()
def evaluate_attack(
    generator: AudioSealWM,
    detector: AudioSealDetector,
    attack: nn.Module,
    dataloader,
    cfg: EvalConfig,
    device: torch.device,
    strength: tp.Optional[float] = None,
) -> tp.Dict[str, float]:
    """Robustness metrics for one attack (optionally at one fixed t*
    strength): bit accuracy and TPR@FPR, where the "negative" (unwatermarked)
    examples are ALSO run through the same attack, so the false-positive
    rate reflects the detector's behavior on attacked-but-clean audio, not
    on unrealistically pristine audio."""
    positive_scores = []
    negative_scores = []
    bit_accs = []

    batches = 0
    for batch in dataloader:
        if batches >= cfg.n_eval_batches:
            break
        batches += 1

        x = batch.to(device)
        message = random_message(cfg.nbits, x.size(0), device=device)

        watermark = generator.get_watermark(x, message=message)
        x_wm = x + watermark

        x_att_pos = attack(x_wm, strength=strength)
        x_att_neg = attack(x, strength=strength)

        presence_pos, m_hat = detector.forward(x_att_pos)
        presence_neg, _ = detector.forward(x_att_neg)

        positive_scores.append(presence_pos[:, 1, :].mean(dim=-1).cpu())
        negative_scores.append(presence_neg[:, 1, :].mean(dim=-1).cpu())
        bit_accs.append(bit_accuracy(m_hat, message))

    positive_cat = torch.cat(positive_scores)
    negative_cat = torch.cat(negative_scores)
    confusion = confusion_counts(positive_cat, negative_cat, cfg.fpr_target)
    return {
        "bit_accuracy": sum(bit_accs) / len(bit_accs),
        "tpr_at_fpr": tpr_at_fpr(positive_cat, negative_cat, cfg.fpr_target),
        "confusion": confusion,
        "f1": f1_score(confusion),
    }


@torch.no_grad()
def evaluate_perceptual(
    generator: AudioSealWM, dataloader, cfg: EvalConfig, device: torch.device
) -> tp.Dict[str, float]:
    """No attack: x vs x_wm only."""
    sisnr_values = []
    pesq_values = []

    batches = 0
    for batch in dataloader:
        if batches >= cfg.n_eval_batches:
            break
        batches += 1

        x = batch.to(device)
        message = random_message(cfg.nbits, x.size(0), device=device)
        x_wm = x + generator.get_watermark(x, message=message)

        if cfg.compute_sisnr:
            sisnr_values.append(sisnr_score(x, x_wm))
        if cfg.compute_pesq:
            try:
                pesq_values.append(pesq_score(x, x_wm, cfg.sample_rate))
            except Exception as e:  # pesq raises on "no speech detected" for some segments
                logger.warning("pesq_score failed on a batch, skipping it: %s", e)

    metrics: tp.Dict[str, float] = {}
    if sisnr_values:
        metrics["sisnr"] = sum(sisnr_values) / len(sisnr_values)
    if pesq_values:
        metrics["pesq"] = sum(pesq_values) / len(pesq_values)
    if cfg.compute_visqol:
        try:
            metrics["visqol"] = visqol_score(x, x_wm, cfg.sample_rate)
        except NotImplementedError as e:
            logger.warning("visqol_score not available, skipping: %s", e)
    return metrics


def run(cfg: EvalConfig) -> tp.Dict[str, tp.Any]:
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    generator = load_generator_under_test(cfg.generator_checkpoint, cfg.nbits, device)
    generator.eval()
    detector = AudioSeal.load_detector(cfg.detector_checkpoint, nbits=cfg.nbits, device=device)
    detector.eval()

    all_attack_names = list(cfg.eval_attacks) + list(cfg.held_out_attacks)
    attacks, skipped_at_construction = build_eval_attacks(all_attack_names, device, cfg)
    held_out = set(cfg.held_out_attacks)

    dataloader = build_dataloader(
        cfg.eval_dir,
        sample_rate=cfg.sample_rate,
        segment_duration=cfg.segment_duration,
        batch_size=cfg.batch_size,
        num_workers=0,
        shuffle=False,
    )

    tracker = build_tracker(
        backend=cfg.tracking.backend,
        project=cfg.tracking.project,
        run_name=cfg.tracking.run_name or cfg.label,
        config={"label": cfg.label, "generator_checkpoint": cfg.generator_checkpoint},
        mlflow_tracking_uri=cfg.tracking.mlflow_tracking_uri,
        wandb_mode=cfg.tracking.wandb_mode,
    )

    results: tp.Dict[str, tp.Any] = {"label": cfg.label}
    try:
        logger.info("=== perceptual metrics (no attack) ===")
        perceptual = evaluate_perceptual(generator, dataloader, cfg, device)
        results["perceptual"] = perceptual
        tracker.log({f"perceptual/{k}": v for k, v in perceptual.items()}, step=0)
        logger.info("perceptual: %s", perceptual)

        results["attacks"] = {}
        for name in all_attack_names:
            tag = "held_out" if name in held_out else "trained"
            if name in skipped_at_construction:
                logger.warning("skipping attack=%s (%s): %s", name, tag, skipped_at_construction[name])
                results["attacks"][name] = {"tag": tag, "skipped": skipped_at_construction[name]}
                continue
            attack = attacks[name]
            logger.info("=== attack=%s (%s) ===", name, tag)
            try:
                robustness = evaluate_attack(generator, detector, attack, dataloader, cfg, device)
                results["attacks"][name] = {"tag": tag, **robustness}
                loggable = {k: v for k, v in robustness.items() if k != "confusion"}
                loggable.update({f"confusion_{k}": v for k, v in robustness["confusion"].items()})
                tracker.log({f"{name}/{k}": v for k, v in loggable.items()}, step=0)
                logger.info("%s (%s): %s", name, tag, robustness)
            except NotImplementedError as e:
                logger.warning("skipping attack=%s (%s): %s", name, tag, e)
                results["attacks"][name] = {"tag": tag, "skipped": str(e)}
                continue

            # Robustness curve: detection vs. t* (only for strength-aware attacks).
            if name in ("sgmse", "diff_erase"):
                curve = []
                for t_star in cfg.t_star_grid:
                    try:
                        point = evaluate_attack(
                            generator, detector, attack, dataloader, cfg, device, strength=t_star
                        )
                        curve.append({"t_star": t_star, **point})
                        tracker.log({f"{name}/tpr_at_fpr_vs_t_star": point["tpr_at_fpr"]}, step=int(t_star * 1000))
                    except NotImplementedError:
                        break  # strength not wired up yet either; identical failure at every t*
                if curve:
                    results["attacks"][name]["robustness_curve"] = curve
                    logger.info("%s robustness curve (detection vs t*): %s", name, curve)

        out_dir = Path(cfg.output_dir)
        confusion_path = plot_confusion_matrices(results, out_dir / f"{cfg.label}_confusion.png")
        if confusion_path is not None:
            results["confusion_matrix_plot"] = str(confusion_path)
            tracker.log_figure(confusion_path)
            logger.info("confusion matrix plot: %s", confusion_path)

        curve_path = plot_robustness_curve(results, out_dir / f"{cfg.label}_robustness_curve.png")
        if curve_path is not None:
            results["robustness_curve_plot"] = str(curve_path)
            tracker.log_figure(curve_path)
            logger.info("robustness curve plot: %s", curve_path)
    finally:
        tracker.finish()

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = load_eval_config()
    results = run(cfg)

    print("\n=== Evaluation summary ===")
    print(f"label: {results['label']}")
    print(f"perceptual: {results.get('perceptual')}")
    for name, r in results["attacks"].items():
        print(f"attack={name} ({r.get('tag')}): {r}")
    if "confusion_matrix_plot" in results:
        print(f"confusion matrix plot: {results['confusion_matrix_plot']}")
    if "robustness_curve_plot" in results:
        print(f"robustness curve plot: {results['robustness_curve_plot']}")


if __name__ == "__main__":
    main()
