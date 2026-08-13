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
NotImplementedError until a checkpoint is configured; audioldm is wired to
a pretrained AudioLDM latent-diffusion release, see attacks.py's
AudioLDMAttack and
EvalAttackConfig.audioldm in config.py, but still raises the same way if
its checkpoint/config aren't set) are individually caught and
reported as "skipped", not fatal -- so this is runnable today, giving you
the identity-attack + perceptual baseline immediately, and picks up each
attack's real numbers as its checkpoint gets configured without any change
to this script.
"""

import logging
import random
import time
import typing as tp
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

from audioseal import AudioSeal
from audioseal.loader import load_state_dict as audioseal_load_state_dict
from audioseal.models import AudioSealDetector, AudioSealWM

from .attacks import AudioLDMAttack, BigVGANAttack, DACAttack, IdentityAttack, MBDAttack, SGMSEAttack
from .config import EvalConfig, load_eval_config
from .data import build_dataloader
from .device import resolve_device
from .metrics import bit_accuracy, confusion_counts, f1_score, pesq_score, sisnr_score, tpr_at_fpr, visqol_score
from .model_init import build_untrained_generator
from .plotting import plot_confusion_matrices, plot_robustness_curve
from .train import embed_watermark, random_message
from .tracking import build_tracker

logger = logging.getLogger(__name__)

_ATTACK_CLASSES: tp.Dict[str, tp.Type[nn.Module]] = {
    "identity": IdentityAttack,
    "bigvgan": BigVGANAttack,
    "dac": DACAttack,
    "sgmse": SGMSEAttack,
    "audioldm": AudioLDMAttack,
    "mbd": MBDAttack,
}

# Attacks with a meaningful `strength` (t*) axis: these are the ones that get
# a pinned `cfg.headline_strength` for the headline number and a
# `cfg.t_star_grid` sweep for the robustness curve. Every other attack's
# forward() takes `strength` and ignores it (see attacks.py -- identity is a
# no-op, and bigvgan/dac/mbd have no natural single corruption-level knob), so
# they're left at strength=None.
_STRENGTH_AWARE_ATTACKS = ("sgmse", "audioldm")


def _reset_peak_memory(device: torch.device) -> None:
    """No-op off CUDA -- MPS/CPU have no equivalent peak-tracking counter."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_metrics(device: torch.device) -> tp.Dict[str, float]:
    """Peak GPU memory since the last `_reset_peak_memory`, in GB, plus what
    fraction of the card that is.

    `reserved` (not `allocated`) is the number to size batches against: it's
    what the caching allocator actually holds from the GPU, so it's what runs
    you out of memory. `allocated` is only the live-tensor subset of that, and
    reads lower than the pressure you'd actually hit.

    Empty dict off CUDA, so callers can just `.update()` it unconditionally.
    """
    if device.type != "cuda":
        return {}
    total = torch.cuda.get_device_properties(device).total_memory
    reserved = torch.cuda.max_memory_reserved(device)
    return {
        "peak_alloc_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "peak_reserved_gb": reserved / 1e9,
        "peak_reserved_frac": reserved / total,
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

PreparedEvalBatch = tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def build_eval_attacks(
    names: tp.List[str], device: torch.device, cfg: EvalConfig
) -> tp.Tuple[tp.Dict[str, nn.Module], tp.Dict[str, str]]:
    """Builds one instance per requested attack name, constructed with that
    attack's `cfg.attack.<name>` sub-config (checkpoint, and whatever else
    that attack class takes -- see attacks.py). `identity` has no matching
    sub-config, so it's always built with defaults.

    Unlike `evaluate_attack`'s forward-time failures, a still-stubbed or
    misconfigured attack (missing checkpoint, missing companion
    config, or a backbone package that isn't installed in
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
def prepare_eval_batches(
    generator: AudioSealWM,
    dataloader,
    cfg: EvalConfig,
    device: torch.device,
) -> tp.List[PreparedEvalBatch]:
    """Materialize one set of (clean, watermarked, message) batches, held on
    CPU and reused by every attack and every t* point, so all reported
    numbers come from identical audio and identical messages. The curve
    points read a prefix of this list (see `cfg.n_curve_batches`)."""
    prepared = []
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= cfg.n_eval_batches:
            break
        x = batch.to(device)
        message = random_message(cfg.nbits, x.size(0), device=device)

        # Same SNR-targeted scaling train.py optimizes against (see
        # embed_watermark) -- NOT raw x + get_watermark(), which would
        # measure detection at whatever amplitude get_watermark() happens to
        # produce unscaled, an operating point training never sees.
        x_wm = embed_watermark(generator, x, message, cfg.watermark_snr_db, cfg.watermark_snr_db)
        prepared.append((x.cpu(), x_wm.cpu(), message.cpu()))
    if not prepared:
        raise RuntimeError("Evaluation dataloader produced no batches")
    return prepared


@torch.no_grad()
def evaluate_attack(
    detector: AudioSealDetector,
    attack: nn.Module,
    eval_batches: tp.List[PreparedEvalBatch],
    cfg: EvalConfig,
    device: torch.device,
    strength: tp.Optional[float] = None,
    n_batches: tp.Optional[int] = None,
    progress_desc: str = "attack",
) -> tp.Dict[str, float]:
    """Robustness metrics for one attack (optionally at one fixed t*
    strength): bit accuracy and TPR@FPR, where the "negative" (unwatermarked)
    examples are ALSO run through the same attack, so the false-positive
    rate reflects the detector's behavior on attacked-but-clean audio, not
    on unrealistically pristine audio.

    `strength` is forwarded to the attack's forward() as its t*. Leave it None
    only for attacks that ignore it -- for a strength-aware one (see
    `_STRENGTH_AWARE_ATTACKS`) None means "sample a fresh random t* per
    forward call", which would give the positives and the negatives below
    *different* attack strengths and make the resulting operating point
    meaningless. Callers should pass `cfg.headline_strength` or a grid point.

    `n_batches` defaults to `cfg.n_eval_batches`; the robustness-curve loop
    overrides it with the smaller `cfg.n_curve_batches`.

    `progress_desc` labels this call's progress bar -- the diffusion attacks
    run for minutes per batch, so without it a long run is indistinguishable
    from a hung one.
    """
    if n_batches is None:
        n_batches = cfg.n_eval_batches

    positive_scores = []
    negative_scores = []
    bit_accs = []
    attack_sisnr_values = []

    progress = tqdm(eval_batches[:n_batches], desc=progress_desc, unit="batch", leave=False)
    for x_cpu, x_wm_cpu, message_cpu in progress:
        x = x_cpu.to(device)
        x_wm = x_wm_cpu.to(device)
        message = message_cpu.to(device)

        x_att_pos = attack(x_wm, strength=strength)
        x_att_neg = attack(x, strength=strength)

        presence_pos, m_hat = detector.forward(x_att_pos)
        presence_neg, _ = detector.forward(x_att_neg)

        positive_scores.append(presence_pos[:, 1, :].mean(dim=-1).cpu())
        negative_scores.append(presence_neg[:, 1, :].mean(dim=-1).cpu())
        bit_accs.append(bit_accuracy(m_hat, message))
        # SNR the attack itself imposes (watermarked-before vs. watermarked-after),
        # so t_star_grid's values can be checked against real degradation instead
        # of the hand-picked curriculum table they were calibrated from.
        attack_sisnr_values.append(sisnr_score(x_wm, x_att_pos))

    progress.close()

    positive_cat = torch.cat(positive_scores)
    negative_cat = torch.cat(negative_scores)
    confusion = confusion_counts(positive_cat, negative_cat, cfg.fpr_target)
    return {
        "bit_accuracy": sum(bit_accs) / len(bit_accs),
        "tpr_at_fpr": tpr_at_fpr(positive_cat, negative_cat, cfg.fpr_target),
        "confusion": confusion,
        "f1": f1_score(confusion),
        "attack_sisnr": sum(attack_sisnr_values) / len(attack_sisnr_values),
    }


@torch.no_grad()
def evaluate_perceptual(
    eval_batches: tp.List[PreparedEvalBatch], cfg: EvalConfig, device: torch.device
) -> tp.Dict[str, float]:
    """No attack: x vs x_wm only."""
    sisnr_values = []
    pesq_values = []

    progress = tqdm(eval_batches, desc="perceptual", unit="batch", leave=False)
    for x_cpu, x_wm_cpu, _ in progress:
        x = x_cpu.to(device)
        x_wm = x_wm_cpu.to(device)

        if cfg.compute_sisnr:
            sisnr_values.append(sisnr_score(x, x_wm))
        if cfg.compute_pesq:
            try:
                pesq_values.append(pesq_score(x, x_wm, cfg.sample_rate))
            except Exception as e:  # pesq raises on "no speech detected" for some segments
                logger.warning("pesq_score failed on a batch, skipping it: %s", e)

    progress.close()

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
    random.seed(cfg.seed)

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
        num_workers=cfg.num_workers,
        shuffle=False,
    )
    eval_batches = prepare_eval_batches(generator, dataloader, cfg, device)

    tracker = build_tracker(
        backend=cfg.tracking.backend,
        project=cfg.tracking.project,
        run_name=cfg.tracking.run_name or cfg.label,
        config={"label": cfg.label, "generator_checkpoint": cfg.generator_checkpoint},
        mlflow_tracking_uri=cfg.tracking.mlflow_tracking_uri,
        wandb_mode=cfg.tracking.wandb_mode,
    )

    results: tp.Dict[str, tp.Any] = {"label": cfg.label}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        results["gpu"] = {"name": props.name, "total_gb": props.total_memory / 1e9}
        logger.info(
            "gpu: %s, %.1f GB total | batch_size=%d segment_duration=%.1fs "
            "-- peak memory is reported per attack below, size batches off the "
            "largest peak_reserved_gb (attacks differ a lot)",
            props.name,
            props.total_memory / 1e9,
            cfg.batch_size,
            cfg.segment_duration,
        )

    # Up-front plan, so a small trial run can be sanity-checked (and its cost
    # extrapolated) before committing to the full one -- see the projection
    # printed by main() at the end.
    n_curve_attacks = sum(1 for n in all_attack_names if n in _STRENGTH_AWARE_ATTACKS and n in attacks)
    planned_batches = (
        cfg.n_eval_batches  # perceptual
        + len(attacks) * cfg.n_eval_batches  # headline, per attack
        + n_curve_attacks * len(cfg.t_star_grid) * cfg.n_curve_batches  # curve points
    )
    logger.info(
        "plan: %d attack(s) [%s], %d skipped | %d batches total "
        "(perceptual %d + headline %dx%d + curve %dx%dx%d) "
        "| batch_size=%d -> %.1f min of audio per full pass",
        len(attacks),
        ", ".join(attacks) or "-",
        len(skipped_at_construction),
        planned_batches,
        cfg.n_eval_batches,
        len(attacks),
        cfg.n_eval_batches,
        n_curve_attacks,
        len(cfg.t_star_grid),
        cfg.n_curve_batches,
        cfg.batch_size,
        cfg.n_eval_batches * cfg.batch_size * cfg.segment_duration / 60,
    )
    results["planned_batches"] = planned_batches
    run_started = time.perf_counter()

    try:
        logger.info("=== perceptual metrics (no attack) ===")
        _reset_peak_memory(device)
        perceptual = evaluate_perceptual(eval_batches, cfg, device)
        perceptual.update(_peak_memory_metrics(device))
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
                _reset_peak_memory(device)
                # Pin t* for the headline number instead of letting each
                # forward call draw its own -- see EvalConfig.headline_strength
                # for why (cost, and positives/negatives otherwise landing at
                # different strengths). None for the attacks that ignore it.
                headline_strength = cfg.headline_strength if name in _STRENGTH_AWARE_ATTACKS else None
                attack_started = time.perf_counter()
                robustness = evaluate_attack(
                    detector,
                    attack,
                    eval_batches,
                    cfg,
                    device,
                    strength=headline_strength,
                    progress_desc=f"{name} (headline)",
                )
                robustness["seconds"] = time.perf_counter() - attack_started
                robustness.update(_peak_memory_metrics(device))
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
            # Deliberately cheaper per point than the headline number above
            # (cfg.n_curve_batches, not cfg.n_eval_batches): this is one full
            # pos+neg pass per grid point, so at equal batch counts the curve
            # alone would cost len(t_star_grid)x the headline. See
            # EvalConfig.n_curve_batches.
            if name in _STRENGTH_AWARE_ATTACKS:
                curve = []
                curve_started = time.perf_counter()
                for i, t_star in enumerate(cfg.t_star_grid, start=1):
                    try:
                        point = evaluate_attack(
                            detector,
                            attack,
                            eval_batches,
                            cfg,
                            device,
                            strength=t_star,
                            n_batches=cfg.n_curve_batches,
                            progress_desc=f"{name} curve t*={t_star:g} ({i}/{len(cfg.t_star_grid)})",
                        )
                        curve.append({"t_star": t_star, **point})
                        tracker.log({f"{name}/tpr_at_fpr_vs_t_star": point["tpr_at_fpr"]}, step=int(t_star * 1000))
                    except NotImplementedError:
                        break  # strength not wired up yet either; identical failure at every t*
                if curve:
                    results["attacks"][name]["robustness_curve"] = curve
                    results["attacks"][name]["curve_seconds"] = time.perf_counter() - curve_started
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
        results["total_seconds"] = time.perf_counter() - run_started
    finally:
        tracker.finish()

    return results


def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def _print_timing_and_projection(cfg: EvalConfig, results: tp.Dict[str, tp.Any]) -> None:
    """Per-attack wall clock, plus what the same config would cost at a
    larger `n_eval_batches`.

    This is the point of running a small fraction first: cost is dominated by
    the diffusion attacks' per-batch work, which scales linearly in batch
    count, so a trial run's seconds-per-batch extrapolates. Model construction
    and checkpoint loading are one-off and NOT excluded here, so short trials
    over-estimate slightly -- the projection is a safe upper bound, which is
    the direction you want before committing to a long run.
    """
    total = results.get("total_seconds")
    if total is None:
        return

    timed = {n: r["seconds"] for n, r in results["attacks"].items() if "seconds" in r}
    print(f"\n=== Timing (n_eval_batches={cfg.n_eval_batches}, n_curve_batches={cfg.n_curve_batches}) ===")
    for name in sorted(timed, key=lambda k: -timed[k]):
        r = results["attacks"][name]
        curve_s = r.get("curve_seconds")
        curve_note = f"  (+ curve {_fmt_duration(curve_s)})" if curve_s else ""
        print(f"  {name}: {_fmt_duration(timed[name])} headline{curve_note}")
    print(f"total: {_fmt_duration(total)}")

    planned = results.get("planned_batches")
    if not planned:
        return

    # total_seconds is undifferentiated wall clock (perceptual + every
    # attack's headline + every attack's curve). Split it before projecting:
    # perceptual and headline cost scale with n_eval_batches, but curve cost
    # is fixed by n_curve_batches (held constant regardless of
    # n_eval_batches) -- scaling it along with n_eval_batches, as a single
    # blended total/planned rate would, wildly overstates a target run's
    # cost whenever curve dominates the trial run's total (it usually does:
    # len(t_star_grid) points, each its own reverse-diffusion pass).
    curve_seconds_total = sum(r.get("curve_seconds", 0.0) for r in results["attacks"].values())
    # Remainder = perceptual pass + fixed per-run overhead (model/checkpoint
    # loading) -- both are effectively one-off, not per-eval-batch, but
    # lumping them into the eval_batches-scaling pool means a short trial
    # over-estimates a longer run's per-batch rate (safe direction: an
    # over-estimate, not under).
    eval_scaling_seconds = total - curve_seconds_total
    eval_scaling_batches = cfg.n_eval_batches * (1 + len(results["attacks"]))
    per_batch = eval_scaling_seconds / eval_scaling_batches if eval_scaling_batches else 0.0

    print(f"\n~{per_batch:.1f}s per eval-batch (perceptual + headline only; curve held fixed, see below).")
    print("Projected totals (headline scaled by n_eval_batches, curve unchanged):")
    for target in (20, 50, 150):
        if target == cfg.n_eval_batches:
            continue
        projected_eval_scaling = per_batch * target * (1 + len(results["attacks"]))
        audio_min = target * cfg.batch_size * cfg.segment_duration / 60
        print(
            f"  n_eval_batches={target:<4} (~{audio_min:.0f} min of audio): "
            f"~{_fmt_duration(projected_eval_scaling + curve_seconds_total)}"
        )
    print(
        f"(curve held fixed at {_fmt_duration(curve_seconds_total)} total -- "
        f"raise n_curve_batches={cfg.n_curve_batches} if you want it more precise, "
        "that's the only thing that changes its cost)"
    )


def _print_results_table(results: tp.Dict[str, tp.Any]) -> None:
    """Human-readable stand-in for dumping `results` (a deeply nested dict,
    including a whole robustness_curve list per attack) straight to stdout --
    that's fine for the log line evaluate_attack already emits (grep-able,
    machine-parseable), but unreadable as the final on-screen summary."""
    print(f"\n{'=' * 60}")
    print(f" Evaluation summary: {results['label']}")
    print(f"{'=' * 60}")

    perceptual = results.get("perceptual")
    if perceptual:
        print("\nPerceptual (watermarked vs. clean, no attack):")
        print(f"  SI-SNR: {perceptual['sisnr']:.2f} dB    PESQ: {perceptual['pesq']:.2f}")

    print("\nAttacks:")
    col = "  {:<12}{:<10}{:>9}{:>10}{:>8}{:>9}{:>9}"
    print(col.format("name", "tag", "bit_acc", "tpr@fpr", "f1", "snr", "time"))
    print("  " + "-" * 67)
    for name, r in results["attacks"].items():
        tag = r.get("tag", "")
        if "skipped" in r:
            print(col.format(name, tag, "skipped", "", "", "", ""))
            print(f"    -> {r['skipped']}")
            continue
        time_str = _fmt_duration(r["seconds"]) if "seconds" in r else "-"
        snr_str = f"{r['attack_sisnr']:.1f}" if "attack_sisnr" in r else "-"
        print(
            col.format(
                name, tag, f"{r['bit_accuracy']:.3f}", f"{r['tpr_at_fpr']:.3f}", f"{r['f1']:.3f}", snr_str, time_str
            )
        )
        c = r.get("confusion")
        if c:
            # Actual 2x2 layout (rows=ground truth, cols=predicted), same
            # convention as plotting.py's confusion-matrix PNG -- not just a
            # flat "tp=.. fn=.. fp=.. tn=.." line.
            mcol = "      {:<14}{:>10}{:>15}"
            print("    confusion matrix:")
            print(mcol.format("", "detected", "not detected"))
            print(mcol.format("watermarked", c["tp"], c["fn"]))
            print(mcol.format("clean", c["fp"], c["tn"]))

        curve = r.get("robustness_curve")
        if curve:
            print(f"    robustness curve ({_fmt_duration(r.get('curve_seconds', 0))}, {len(curve)} points):")
            curve_col = "      {:>8}{:>10}{:>10}{:>9}"
            print(curve_col.format("t*", "bit_acc", "tpr@fpr", "snr"))
            for p in curve:
                print(
                    curve_col.format(
                        f"{p['t_star']:.4f}", f"{p['bit_accuracy']:.3f}", f"{p['tpr_at_fpr']:.3f}",
                        f"{p['attack_sisnr']:.1f}"
                    )
                )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = load_eval_config()
    results = run(cfg)

    _print_results_table(results)
    _print_timing_and_projection(cfg, results)
    if "confusion_matrix_plot" in results:
        print(f"confusion matrix plot: {results['confusion_matrix_plot']}")
    if "robustness_curve_plot" in results:
        print(f"robustness curve plot: {results['robustness_curve_plot']}")

    gpu = results.get("gpu")
    if gpu:
        peaks = {
            name: r["peak_reserved_gb"]
            for name, r in results["attacks"].items()
            if "peak_reserved_gb" in r
        }
        if peaks:
            worst_name = max(peaks, key=lambda k: peaks[k])
            worst_gb = peaks[worst_name]
            print(f"\n=== GPU memory (batch_size={cfg.batch_size}, segment_duration={cfg.segment_duration}s) ===")
            print(f"{gpu['name']}, {gpu['total_gb']:.1f} GB total")
            for name in sorted(peaks, key=lambda k: -peaks[k]):
                print(f"  {name}: {peaks[name]:.2f} GB peak reserved")
            # Deliberately not printing a "max batch size" number: peak memory
            # is not purely linear in batch size (fixed model weights sit in
            # there too, and the diffusion attacks' inner loops allocate their
            # own buffers), so a naive total/peak*batch_size extrapolation
            # would over-promise. Headroom ratio is the honest version.
            print(
                f"worst attack is {worst_name} at {worst_gb:.2f} GB "
                f"({worst_gb / gpu['total_gb']:.0%} of the card) -- "
                f"~{gpu['total_gb'] / worst_gb:.1f}x headroom before it's full"
            )


if __name__ == "__main__":
    main()
