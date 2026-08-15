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

import csv
import json
import logging
import time
import typing as tp
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm

from audioseal import AudioSeal
from audioseal.loader import align_state_dict_to_model
from audioseal.loader import load_state_dict as audioseal_load_state_dict
from audioseal.models import AudioSealDetector, AudioSealWM

from .attacks import (
    AudioLDMAttack,
    BigVGANAttack,
    DACAttack,
    HopSkipJumpAttack,
    IdentityAttack,
    MBDAttack,
    SGMSEAttack,
)
from .config import EvalConfig, load_eval_config
from .data import build_dataloader
from .distributed import (
    DistEnv,
    all_gather_scores,
    all_gather_values,
    all_reduce_max,
    cleanup_distributed,
    configure_logging,
    gather_objects,
    init_distributed,
    seed_everything,
    shard_size,
)
from .metrics import (
    bit_accuracy,
    confusion_counts,
    f1_score,
    fpr_support,
    pesq_score,
    sisnr_score,
    tpr_at_fpr,
    visqol_score,
)
from sklearn.metrics import roc_auc_score, roc_curve
from .model_init import build_untrained_generator
from .plotting import plot_confusion_matrices, plot_robustness_curve, plot_roc_curves
from .train import embed_watermark, random_message
from .tracking import NullTracker, build_tracker

logger = logging.getLogger(__name__)

_ATTACK_CLASSES: tp.Dict[str, tp.Type[nn.Module]] = {
    "identity": IdentityAttack,
    "bigvgan": BigVGANAttack,
    "dac": DACAttack,
    "sgmse": SGMSEAttack,
    "audioldm": AudioLDMAttack,
    "mbd": MBDAttack,
    "hopskipjump": HopSkipJumpAttack,
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


def _peak_memory_metrics(device: torch.device, env: DistEnv = DistEnv()) -> tp.Dict[str, float]:
    """Peak GPU memory since the last `_reset_peak_memory`, in GB, plus what
    fraction of the card that is.

    `reserved` (not `allocated`) is the number to size batches against: it's
    what the caching allocator actually holds from the GPU, so it's what runs
    you out of memory. `allocated` is only the live-tensor subset of that, and
    reads lower than the pressure you'd actually hit.

    Reduced with MAX, not mean, across ranks: the number worth knowing is the
    worst card's, because that is the one that decides whether the job OOMs.

    Empty dict off CUDA, so callers can just `.update()` it unconditionally.
    """
    if device.type != "cuda":
        return {}
    total = torch.cuda.get_device_properties(device).total_memory
    reserved = torch.cuda.max_memory_reserved(device)
    return all_reduce_max(
        {
            "peak_alloc_gb": torch.cuda.max_memory_allocated(device) / 1e9,
            "peak_reserved_gb": reserved / 1e9,
            "peak_reserved_frac": reserved / total,
        },
        env,
        device,
    )


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
        # train.py writes whichever conv naming the *training* interpreter's
        # SEANet used (flat below Python 3.10, "inner_conv"-wrapped at or above
        # it -- see builder.py), and this eval process may well be on the other
        # side of that split. align_state_dict_to_model reconciles both
        # directions off the actual keys; see its docstring for why
        # convert_state_dict_for_scriptable_model alone is not enough.
        audioseal_load_state_dict(
            generator, align_state_dict_to_model(generator, state["model"])
        )
        return generator
    logger.info("loading generator checkpoint/card %r", checkpoint)
    return AudioSeal.load_generator(checkpoint, nbits=nbits, device=device)


_CONSTRUCTION_SKIP_EXCEPTIONS = (NotImplementedError, FileNotFoundError, ModuleNotFoundError)

# Eval batches drawn on for the query-based attacks' reference pool (see
# run()). Two batches of each branch is enough variety for _initialize to
# find an opposite-class start almost immediately, while keeping its linear
# scan short -- every extra entry is a detector query per attacked example.
_REFERENCE_POOL_BATCHES = 2

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
    env: DistEnv = DistEnv(),
) -> tp.List[PreparedEvalBatch]:
    """Materialize one set of (clean, watermarked, message) batches, held on
    CPU and reused by every attack and every t* point, so all reported
    numbers come from identical audio and identical messages. The curve
    points read a prefix of this list (see `cfg.n_curve_batches`).

    Under DDP this holds only THIS rank's shard: `cfg.n_eval_batches` is a
    global total, so each rank materializes `shard_size(...)` of it from its
    own disjoint `DistributedSampler` slice. The same config therefore
    evaluates the same audio whether it runs on 1 GPU or 4 -- it just splits
    the work. Per-rank scores are pooled with `all_gather_*` before any
    metric is computed (see `evaluate_attack`).

    A rank whose shard is legitimately empty (fewer batches requested than
    there are ranks -- `shard_size` gives the surplus ranks 0) returns an
    empty list rather than raising: it still has to enter every collective
    downstream, and `evaluate_attack`/`evaluate_perceptual` are built to let
    it contribute nothing (see `_cat_or_empty`). Raising here would take the
    job down on exactly the configuration that support is meant to cover.
    """
    local_n_batches = shard_size(cfg.n_eval_batches, env)
    if env.is_distributed and local_n_batches == 0:
        logger.warning(
            "rank %d gets no batches: n_eval_batches=%d is below world_size=%d, so this rank only "
            "sits in the collectives. The numbers stay correct -- the GPU is just wasted.",
            env.rank, cfg.n_eval_batches, env.world_size,
        )
    prepared = []
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= local_n_batches:
            break
        x = batch.to(device)
        message = random_message(cfg.nbits, x.size(0), device=device)

        # Same SNR-targeted scaling train.py optimizes against (see
        # embed_watermark) -- NOT raw x + get_watermark(), which would
        # measure detection at whatever amplitude get_watermark() happens to
        # produce unscaled, an operating point training never sees.
        x_wm = embed_watermark(generator, x, message, cfg.watermark_snr_db, cfg.watermark_snr_db)
        prepared.append((x.cpu(), x_wm.cpu(), message.cpu()))
    # Compare against this rank's LOCAL expected batch count
    # (local_n_batches), not the global cfg.n_eval_batches -- under DDP each
    # rank only materializes its own shard (see shard_size above), so
    # comparing against the global total would make every multi-GPU run
    # raise here even when nothing is wrong. Still a strict "<", not just
    # "== 0": a dataloader with drop_last=True silently yields fewer batches
    # than asked for when the dataset is too small, and every downstream
    # number (including n_negatives, which sets the achievable FPR
    # resolution) would then be computed over a smaller sample than the
    # label claims -- a "full_1h" run quietly becoming an 8-minute one. A
    # rank whose shard is legitimately empty (local_n_batches == 0, see the
    # docstring above) still passes this check (0 < 0 is False), so it can
    # enter the collectives with nothing rather than raising.
    if len(prepared) < local_n_batches:
        dataset = getattr(dataloader, "dataset", None)
        n_files = len(dataset) if dataset is not None else "unknown"
        raise RuntimeError(
            f"Evaluation dataloader produced only {len(prepared)} batches, but "
            f"local_n_batches={local_n_batches} was expected on rank {env.rank} "
            f"(cfg.n_eval_batches={cfg.n_eval_batches} globally, "
            f"eval_dir={cfg.eval_dir!r} holds {n_files} usable files; "
            f"batch_size={cfg.batch_size} with drop_last=True needs at least "
            f"{local_n_batches * cfg.batch_size} on this rank). Lower "
            f"n_eval_batches, use fewer GPUs, or point eval_dir at a larger set."
        )
    return prepared


def _cat_or_empty(chunks: tp.List[torch.Tensor]) -> torch.Tensor:
    """`torch.cat` that tolerates an empty list, which it otherwise rejects.

    A rank holding a zero-size shard has no score chunks to concatenate but
    still has to take part in the all-gather, so it needs a well-formed
    zero-length tensor rather than an exception.
    """
    return torch.cat(chunks) if chunks else torch.zeros(0)


def _save_row_artifacts(
    out_dir: Path,
    row_index: int,
    x: torch.Tensor,
    x_wm: torch.Tensor,
    x_att: torch.Tensor,
    sample_rate: int,
    row_writer: "csv._writer",
    per_example_bit_acc: torch.Tensor,
    presence_pos: torch.Tensor,
    presence_neg: torch.Tensor,
    per_example_sisnr: torch.Tensor,
) -> None:
    """Writes one example's (x, x_wm, x_att) as .wav under out_dir/audio/ and
    appends its per-example metrics as a CSV row. row_index is the running
    index across all batches (not reset per batch), so filenames/row indices
    stay unique and stable across the whole eval_batches sweep."""
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for tag, wav in (("x", x), ("x_wm", x_wm), ("x_att", x_att)):
        torchaudio.save(str(audio_dir / f"{row_index:05d}_{tag}.wav"), wav.cpu(), sample_rate)
    row_writer.writerow(
        {
            "row_index": row_index,
            "bit_accuracy": per_example_bit_acc.item(),
            "presence_pos": presence_pos.item(),
            "presence_neg": presence_neg.item(),
            "attack_sisnr": per_example_sisnr.item(),
            "x_wav": f"audio/{row_index:05d}_x.wav",
            "x_wm_wav": f"audio/{row_index:05d}_x_wm.wav",
            "x_att_wav": f"audio/{row_index:05d}_x_att.wav",
        }
    )


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
    env: DistEnv = DistEnv(),
    row_artifacts_name: tp.Optional[str] = None,
) -> tp.Dict[str, tp.Any]:
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

    Beyond the headline metrics, the returned dict always carries
    `fpr_support` (whether the negative sample size can resolve
    `cfg.fpr_target` at all -- see metrics.fpr_support), and for attacks
    implementing AttackApplicationReporter also `attack_failure_rate`,
    `n_attack_failures` and, when any example failed, `tpr_at_fpr_attacked`
    (the headline metric restricted to examples the attack really
    perturbed). Both exist to stop a weak attack from reading as a robust
    model; see those keys' inline comments below.

    `row_artifacts_name` (e.g. the attack's name): when set AND
    `cfg.save_row_artifacts` is True, writes every example's (x, x_wm,
    x_att_pos) plus its own per-example metrics under
    f"{cfg.output_dir}/{cfg.label}_{row_artifacts_name}_rows/" -- see
    `_save_row_artifacts`. None (the default) skips this entirely, so
    callers that don't pass it (e.g. the robustness-curve sweep) never pay
    for it.
    """
    if n_batches is None:
        n_batches = cfg.n_eval_batches

    positive_scores = []
    negative_scores = []
    bit_accs = []
    attack_sisnr_values = []
    # Per-example "did the attack actually perturb this?" flags, kept
    # separately per branch so the applied-only metrics below stay paired
    # with the right score pool. Empty for every attack that doesn't
    # implement AttackApplicationReporter (i.e. all resynthesis attacks),
    # which is treated as "applied to everything" -- see that mixin.
    applied_pos: tp.List[torch.Tensor] = []
    applied_neg: tp.List[torch.Tensor] = []
    pop_application_mask = getattr(attack, "pop_application_mask", None)

    # `n_batches` is a global total; take this rank's share of it. The list
    # itself is already this rank's shard (see prepare_eval_batches), so this
    # is a prefix of local batches, not a re-slice of the global set.
    local_n_batches = shard_size(n_batches, env)

    # Row artifacts are written per rank. Every rank holds a different shard
    # but would otherwise write audio/00000_x.wav and rows.csv into the same
    # directory, clobbering each other's files. Under torchrun each rank gets
    # its own rank<N>/ subfolder, so row_index can stay a local counter; a
    # single-process run keeps the original flat layout unchanged.
    row_writer = None
    row_file = None
    rows_dir: tp.Optional[Path] = None
    if row_artifacts_name is not None and cfg.save_row_artifacts:
        rows_dir = Path(cfg.output_dir) / f"{cfg.label}_{row_artifacts_name}_rows"
        if env.is_distributed:
            rows_dir = rows_dir / f"rank{env.rank}"
        rows_dir.mkdir(parents=True, exist_ok=True)
        row_file = open(rows_dir / "rows.csv", "w", newline="")
        row_writer = csv.DictWriter(
            row_file,
            fieldnames=[
                "row_index",
                "bit_accuracy",
                "presence_pos",
                "presence_neg",
                "attack_sisnr",
                "x_wav",
                "x_wm_wav",
                "x_att_wav",
            ],
        )
        row_writer.writeheader()

    # A stub attack raises NotImplementedError from its first forward. That
    # has to become the SAME decision on every rank: if one rank skipped the
    # attack while another carried on into the all-gather below, the ones
    # still going would block until the NCCL timeout (~30 min) and take the
    # job down with a stack trace pointing at the wrong line. So the failure
    # is recorded, gathered, and only then re-raised -- everywhere or nowhere.
    local_failure: tp.Optional[str] = None
    row_index = 0
    progress = tqdm(
        eval_batches[:local_n_batches],
        desc=progress_desc,
        unit="batch",
        leave=False,
        disable=not env.is_main,
    )
    try:
        for x_cpu, x_wm_cpu, message_cpu in progress:
            x = x_cpu.to(device)
            x_wm = x_wm_cpu.to(device)
            message = message_cpu.to(device)

            x_att_pos = attack(x_wm, strength=strength)
            if pop_application_mask is not None:
                mask = pop_application_mask()
                if mask is not None:
                    applied_pos.append(mask)
            x_att_neg = attack(x, strength=strength)
            if pop_application_mask is not None:
                mask = pop_application_mask()
                if mask is not None:
                    applied_neg.append(mask)

            presence_pos, m_hat = detector.forward(x_att_pos)
            presence_neg, _ = detector.forward(x_att_neg)

            presence_pos_per_example = presence_pos[:, 1, :].mean(dim=-1)
            presence_neg_per_example = presence_neg[:, 1, :].mean(dim=-1)
            positive_scores.append(presence_pos_per_example.cpu())
            negative_scores.append(presence_neg_per_example.cpu())
            bit_accs.append(bit_accuracy(m_hat, message))
            # SNR the attack itself imposes (watermarked-before vs. watermarked-after),
            # so t_star_grid's values can be checked against real degradation instead
            # of the hand-picked curriculum table they were calibrated from.
            attack_sisnr_values.append(sisnr_score(x_wm, x_att_pos))

            if row_writer is not None:
                assert rows_dir is not None  # set together with row_writer
                decoded = (m_hat > 0.5).float()
                per_example_bit_acc = (decoded == message.float()).float().mean(dim=-1)
                for i in range(x.shape[0]):
                    _save_row_artifacts(
                        rows_dir,
                        row_index,
                        x[i],
                        x_wm[i],
                        x_att_pos[i],
                        cfg.sample_rate,
                        row_writer,
                        per_example_bit_acc[i],
                        presence_pos_per_example[i],
                        presence_neg_per_example[i],
                        # Per-example SI-SNR: reuses sisnr_score (rather than a
                        # separate hand-rolled formula) on a size-1 slice, so
                        # this can't silently drift from the aggregate number
                        # above -- same function, just one example at a time.
                        torch.tensor(sisnr_score(x_wm[i : i + 1], x_att_pos[i : i + 1])),
                    )
                    row_index += 1
    except NotImplementedError as e:
        local_failure = str(e)
    finally:
        progress.close()
        if row_file is not None:
            row_file.close()

    failures = [f for f in gather_objects(local_failure, env) if f is not None]
    if failures:
        raise NotImplementedError(failures[0])

    # Pool the RAW scores across ranks before computing anything. TPR@FPR and
    # the confusion matrix both hang off a threshold that is a quantile of the
    # negative-score distribution, and a quantile cannot be averaged out of
    # per-rank quantiles -- 4 ranks each thresholding their own shard is a
    # different (and wrong) operating point from one threshold over all of it.
    #
    # `local_n_batches` can legitimately be 0 on the high ranks (the curve
    # uses cfg.n_curve_batches, which can be smaller than world_size), and
    # torch.cat([]) raises. Such a rank contributes nothing but must still
    # enter every collective below, or the ranks that do have data block.
    positive_cat = all_gather_scores(_cat_or_empty(positive_scores), env)
    negative_cat = all_gather_scores(_cat_or_empty(negative_scores), env)
    bit_accs = all_gather_values(bit_accs, env)
    attack_sisnr_values = all_gather_values(attack_sisnr_values, env)

    if not bit_accs:
        raise RuntimeError(
            f"No rank produced any batch for '{progress_desc}' "
            f"(requested {n_batches} batch(es) over {env.world_size} rank(s))"
        )

    # ROC/AUC over the SAME pooled (post-all_gather) scores as everything
    # else below -- computing it from per-rank-local scores instead would
    # give each rank its own curve, the same wrong-operating-point problem
    # the comment above describes for TPR@FPR.
    try:
        y_pos = torch.ones_like(positive_cat)
        y_neg = torch.zeros_like(negative_cat)
        y = torch.cat([y_pos, y_neg]).cpu().numpy()
        scores = torch.cat([positive_cat, negative_cat]).cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y, scores)
        roc_auc = float(roc_auc_score(y, scores))
    except Exception:
        # If sklearn isn't available or computation fails, skip ROC
        fpr, tpr, thresholds, roc_auc = None, None, None, None

    confusion = confusion_counts(positive_cat, negative_cat, cfg.fpr_target)
    metrics = {
        "bit_accuracy": sum(bit_accs) / len(bit_accs),
        "tpr_at_fpr": tpr_at_fpr(positive_cat, negative_cat, cfg.fpr_target),
        "confusion": confusion,
        "f1": f1_score(confusion),
        "attack_sisnr": sum(attack_sisnr_values) / len(attack_sisnr_values),
        "roc_auc": roc_auc,
        "roc_curve": {"fpr": fpr.tolist() if fpr is not None else None, "tpr": tpr.tolist() if tpr is not None else None},
    }

    # Is the negative sample size big enough for `fpr_target` to mean
    # anything? Reported unconditionally (not just on failure) so the number
    # travels with the run rather than having to be recomputed from
    # batch_size * n_eval_batches by whoever reads the log later.
    support = fpr_support(negative_cat.numel(), cfg.fpr_target)
    metrics["fpr_support"] = support
    if not support["supported"]:
        logger.warning(
            "%s: tpr_at_fpr=%.3f was measured at fpr_target=%g from only %d negatives, "
            "which can only resolve FPR down to %.3f -- the threshold degenerated to "
            "'just above the highest negative score' and this number is a high-variance "
            "lower bound, NOT a %g operating point. Raise batch_size * n_eval_batches to "
            ">= %d negatives before quoting it.",
            progress_desc,
            metrics["tpr_at_fpr"],
            cfg.fpr_target,
            support["n_negatives"],
            support["fpr_resolution"],
            cfg.fpr_target,
            support["min_negatives_for_target"],
        )

    # Did the attack actually run on everything it was handed? An attack that
    # passed examples through unperturbed inflates tpr_at_fpr, because an
    # unattacked watermarked example still detects and is scored as
    # robustness -- see AttackApplicationReporter.
    if applied_pos or applied_neg:
        mask_pos = torch.cat(applied_pos) if applied_pos else torch.ones_like(positive_cat, dtype=torch.bool)
        mask_neg = torch.cat(applied_neg) if applied_neg else torch.ones_like(negative_cat, dtype=torch.bool)
        n_total = mask_pos.numel() + mask_neg.numel()
        n_failed = int((~mask_pos).sum().item() + (~mask_neg).sum().item())
        metrics["attack_failure_rate"] = n_failed / n_total if n_total else 0.0
        metrics["n_attack_failures"] = n_failed

        # The same headline metric restricted to examples the attack really
        # perturbed: "how robust is the watermark WHEN attacked", as opposed
        # to tpr_at_fpr's "how robust is it against this attack as
        # configured, failures included". Both are legitimate; reporting only
        # the first hides a weak search budget behind a good-looking score.
        if n_failed and mask_pos.any() and mask_neg.any():
            metrics["tpr_at_fpr_attacked"] = tpr_at_fpr(
                positive_cat[mask_pos], negative_cat[mask_neg], cfg.fpr_target
            )
            logger.warning(
                "%s: attack failed to perturb %d/%d examples (%.1f%%); tpr_at_fpr=%.3f "
                "over all examples vs. %.3f over perturbed-only. The gap is attack "
                "weakness, not model robustness.",
                progress_desc,
                n_failed,
                n_total,
                100.0 * metrics["attack_failure_rate"],
                metrics["tpr_at_fpr"],
                metrics["tpr_at_fpr_attacked"],
            )

    return metrics


@torch.no_grad()
def evaluate_perceptual(
    eval_batches: tp.List[PreparedEvalBatch],
    cfg: EvalConfig,
    device: torch.device,
    env: DistEnv = DistEnv(),
) -> tp.Dict[str, float]:
    """No attack: x vs x_wm only."""
    sisnr_values = []
    pesq_values = []
    last_pair: tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]] = None

    progress = tqdm(eval_batches, desc="perceptual", unit="batch", leave=False, disable=not env.is_main)
    for x_cpu, x_wm_cpu, _ in progress:
        x = x_cpu.to(device)
        x_wm = x_wm_cpu.to(device)
        last_pair = (x, x_wm)

        if cfg.compute_sisnr:
            sisnr_values.append(sisnr_score(x, x_wm))
        if cfg.compute_pesq:
            try:
                pesq_values.append(pesq_score(x, x_wm, cfg.sample_rate))
            except Exception as e:  # pesq raises on "no speech detected" for some segments
                logger.warning("pesq_score failed on a batch, skipping it: %s", e)

    progress.close()

    # Pool across ranks so the reported mean is over all the audio, not this
    # rank's shard. Done unconditionally (not under the `if` below) because
    # all_gather is collective: a rank that skipped it while others called it
    # would hang the job.
    sisnr_values = all_gather_values(sisnr_values, env)
    pesq_values = all_gather_values(pesq_values, env)

    metrics: tp.Dict[str, float] = {}
    if sisnr_values:
        metrics["sisnr"] = sum(sisnr_values) / len(sisnr_values)
    if pesq_values:
        metrics["pesq"] = sum(pesq_values) / len(pesq_values)
    if cfg.compute_visqol and last_pair is not None:
        try:
            metrics["visqol"] = visqol_score(*last_pair, cfg.sample_rate)
        except NotImplementedError as e:
            logger.warning("visqol_score not available, skipping: %s", e)
    return metrics


def run(cfg: EvalConfig) -> tp.Dict[str, tp.Any]:
    env, device = init_distributed(cfg.device)
    # Every rank draws the SAME messages and the same attack randomness, and
    # the sharding comes from the sampler, not from the seed. See
    # distributed.seed_everything.
    seed_everything(cfg.seed, env)

    generator = load_generator_under_test(cfg.generator_checkpoint, cfg.nbits, device)
    generator.eval()
    detector = AudioSeal.load_detector(cfg.detector_checkpoint, nbits=cfg.nbits, device=device)
    detector.eval()

    all_attack_names = list(cfg.eval_attacks) + list(cfg.held_out_attacks)
    attacks, skipped_at_construction = build_eval_attacks(all_attack_names, device, cfg)

    # Union the skip set across ranks before anyone acts on it.
    #
    # build_eval_attacks decides locally, and construction can genuinely fail
    # on one rank and not another -- a per-rank HF cache miss, a transient
    # download error, a checkpoint that only some node-local disk has. Whatever
    # the cause, an asymmetric skip set is fatal rather than merely uneven: the
    # loop below runs `for name in all_attack_names` and every non-skipped
    # attack enters evaluate_attack, which is full of collectives
    # (all_gather_scores/all_gather_values/gather_objects). A rank that skips
    # an attack the others evaluate simply never reaches those calls, so the
    # ranks that did block until the NCCL timeout (~30 min) and take the job
    # down pointing at the wrong line.
    #
    # Union, not intersection: a rank that could not build an attack cannot
    # evaluate it, so the only globally consistent choice is to skip it
    # everywhere. Each rank keeps its own reason where it has one, and adopts a
    # peer's otherwise, so the log says why rather than just that.
    if env.is_distributed:
        merged: tp.Dict[str, str] = {}
        for rank, remote in enumerate(gather_objects(skipped_at_construction, env)):
            for name, reason in tp.cast(tp.Dict[str, str], remote).items():
                merged.setdefault(name, f"{reason} (first seen on rank {rank})")
        newly_skipped = set(merged) - set(skipped_at_construction)
        if newly_skipped and env.is_main:
            logger.warning(
                "skipping %s on every rank: another rank could not construct %s",
                ", ".join(sorted(newly_skipped)),
                "it" if len(newly_skipped) == 1 else "them",
            )
        skipped_at_construction = merged
        # Drop the local instances too, so nothing downstream can reach for an
        # attack this rank is contractually skipping.
        for name in newly_skipped:
            attacks.pop(name, None)

    held_out = set(cfg.held_out_attacks)

    # Query-based attacks (currently just HopSkipJumpAttack) need the
    # detector under test at forward() time, unlike every resynthesis
    # attack -- see HopSkipJumpAttack's class docstring for why. Duck-typed
    # via bind_detector rather than threading the detector through every
    # attack's constructor/forward signature, so this stays a no-op for
    # every other attack.
    for attack in attacks.values():
        bind_detector = getattr(attack, "bind_detector", None)
        if bind_detector is not None:
            bind_detector(detector)

    # Eval is pure forward passes -- no DDP wrapper needed, no gradients to
    # sync. The parallelism is purely data sharding via the sampler.
    dataloader, _sampler = build_dataloader(
        cfg.eval_dir,
        sample_rate=cfg.sample_rate,
        segment_duration=cfg.segment_duration,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        env=env,
    )
    eval_batches = prepare_eval_batches(generator, dataloader, cfg, device, env)

    # Real-audio starting points for query-based search attacks, same
    # duck-typed opt-in as bind_detector above. Deliberately built AFTER
    # prepare_eval_batches (it needs the watermarked signals) and from both
    # branches: an evasion search on watermarked audio needs references the
    # detector calls clean, a false-positive search on clean audio needs
    # references it calls watermarked, and one pool serves both. Capped
    # because _initialize scans it linearly with a detector query per entry,
    # so an unbounded pool would add cost to every single example.
    reference_pool = torch.cat(
        [
            torch.cat([x for x, _, _ in eval_batches[:_REFERENCE_POOL_BATCHES]]),
            torch.cat([x_wm for _, x_wm, _ in eval_batches[:_REFERENCE_POOL_BATCHES]]),
        ]
    )
    for attack in attacks.values():
        bind_reference_pool = getattr(attack, "bind_reference_pool", None)
        if bind_reference_pool is not None:
            bind_reference_pool(reference_pool)
            logger.info("bound a %d-waveform reference pool for query-based init", reference_pool.shape[0])

    # Only rank 0 talks to the tracker: 4 ranks logging the same all-reduced
    # numbers to the same run would quadruple every point.
    tracker = (
        build_tracker(
            backend=cfg.tracking.backend,
            project=cfg.tracking.project,
            run_name=cfg.tracking.run_name or cfg.label,
            config={
                "label": cfg.label,
                "generator_checkpoint": cfg.generator_checkpoint,
                "world_size": env.world_size,
            },
            mlflow_tracking_uri=cfg.tracking.mlflow_tracking_uri,
            wandb_mode=cfg.tracking.wandb_mode,
        )
        if env.is_main
        else NullTracker()
    )

    results: tp.Dict[str, tp.Any] = {"label": cfg.label, "device": str(device), "world_size": env.world_size}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        results["gpu"] = {"name": props.name, "total_gb": props.total_memory / 1e9}
        logger.info(
            "gpu: %s, %.1f GB total | world_size=%d batch_size=%d (per rank) "
            "segment_duration=%.1fs -- peak memory is reported per attack "
            "below (max over ranks), size batches off the largest "
            "peak_reserved_gb (attacks differ a lot)",
            props.name,
            props.total_memory / 1e9,
            env.world_size,
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
        perceptual = evaluate_perceptual(eval_batches, cfg, device, env)
        perceptual.update(_peak_memory_metrics(device, env))
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
                    env=env,
                    row_artifacts_name=name,
                )
                robustness["seconds"] = time.perf_counter() - attack_started
                robustness.update(_peak_memory_metrics(device, env))
                results["attacks"][name] = {"tag": tag, **robustness}
                # Flatten the dict-valued entries: mlflow's log_metric only
                # takes scalars, so passing `confusion`/`fpr_support` through
                # as nested dicts would break that backend (wandb tolerates
                # them, which is exactly how it would slip through untested).
                loggable = {k: v for k, v in robustness.items() if k not in ("confusion", "fpr_support")}
                loggable.update({f"confusion_{k}": v for k, v in robustness["confusion"].items()})
                loggable.update(
                    {f"fpr_support_{k}": float(v) for k, v in robustness["fpr_support"].items()}
                )
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
                            env=env,
                        )
                        curve.append({"t_star": t_star, **point})
                        tracker.log({f"{name}/tpr_at_fpr_vs_t_star": point["tpr_at_fpr"]}, step=int(t_star * 1000))
                    except NotImplementedError:
                        break  # strength not wired up yet either; identical failure at every t*
                if curve:
                    results["attacks"][name]["robustness_curve"] = curve
                    results["attacks"][name]["curve_seconds"] = time.perf_counter() - curve_started
                    logger.info("%s robustness curve (detection vs t*): %s", name, curve)

        # Plots are written once, by rank 0 -- 4 ranks racing to write the
        # same two PNG paths would interleave partial files.
        if env.is_main:
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
            roc_path = plot_roc_curves(results, out_dir / f"{cfg.label}_roc.png")
            if roc_path is not None:
                results["roc_plot"] = str(roc_path)
                tracker.log_figure(roc_path)
                logger.info("roc plot: %s", roc_path)
        results["total_seconds"] = time.perf_counter() - run_started
        # Persist BEFORE main()'s summary printer runs: that printer has
        # crashed on a missing key before (KeyError 'pesq' under
        # compute_pesq=false), which used to throw away a whole run's
        # numbers even though every metric had already been computed.
        _write_result_files(cfg, results)
    finally:
        tracker.finish()
        cleanup_distributed(env)

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


def _write_result_files(cfg: EvalConfig, results: tp.Dict[str, tp.Any]) -> None:
    """Dump `results` to disk next to the PNGs: a flat per-attack CSV, a
    flat per-curve-point CSV, and the full nested JSON.

    The harness only ever emitted plots and a stdout table, so every number
    behind a run lived in a log that has to be re-parsed by hand (and in a
    tracking backend that may be offline/none). Each CSV row carries its own
    run-level context (label, device, sample sizes, perceptual metrics) so
    runs can simply be concatenated for cross-run comparison without a join.

    The caveat fields travel WITH the numbers on purpose: `fpr_supported`
    and `attack_failure_rate` are exactly what distinguishes a real result
    from one where the negative sample was too thin to resolve the target
    FPR, or where the attack never actually perturbed anything.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    perceptual = results.get("perceptual") or {}

    # n_positives == n_negatives: evaluate_attack scores each batch twice,
    # once on the watermarked copy and once on the clean one.
    n_headline = cfg.batch_size * cfg.n_eval_batches
    n_curve = cfg.batch_size * cfg.n_curve_batches

    run_ctx = {
        "label": results.get("label"),
        "device": str(results.get("device", "")),
        "eval_dir": str(cfg.eval_dir),
        "segment_duration": cfg.segment_duration,
        "batch_size": cfg.batch_size,
        "perceptual_sisnr": perceptual.get("sisnr"),
        "perceptual_pesq": perceptual.get("pesq"),
    }

    metrics_path = out_dir / f"{cfg.label}_metrics.csv"
    metrics_cols = list(run_ctx) + [
        "attack", "tag", "skipped", "n_eval_batches", "n_positives", "n_negatives",
        "fpr_target", "fpr_resolution", "min_negatives_for_target", "fpr_supported",
        "bit_accuracy", "tpr_at_fpr", "tpr_at_fpr_attacked", "f1", "attack_sisnr",
        "tp", "fn", "fp", "tn",
        "attack_failure_rate", "n_attack_failures",
        "roc_auc",
        "seconds", "peak_reserved_gb",
    ]
    scalar_keys = (
        "bit_accuracy", "tpr_at_fpr", "tpr_at_fpr_attacked", "f1", "attack_sisnr",
        "roc_auc",
        "attack_failure_rate", "n_attack_failures", "seconds", "peak_reserved_gb",
    )
    with metrics_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=metrics_cols, extrasaction="ignore")
        writer.writeheader()
        for name, r in results.get("attacks", {}).items():
            confusion = r.get("confusion") or {}
            support = r.get("fpr_support") or {}
            row = {
                **run_ctx,
                "attack": name,
                "tag": r.get("tag", ""),
                "skipped": r.get("skipped", ""),
                "n_eval_batches": cfg.n_eval_batches,
                "n_positives": n_headline,
                "n_negatives": support.get("n_negatives", n_headline),
                "fpr_target": cfg.fpr_target,
                "fpr_resolution": support.get("fpr_resolution"),
                "min_negatives_for_target": support.get("min_negatives_for_target"),
                "fpr_supported": support.get("supported"),
                **{k: confusion.get(k) for k in ("tp", "fn", "fp", "tn")},
                **{k: r.get(k) for k in scalar_keys},
            }
            writer.writerow(row)

    curve_rows = []
    for name, r in results.get("attacks", {}).items():
        for point in r.get("robustness_curve") or []:
            flat = {k: v for k, v in point.items() if not isinstance(v, dict)}
            curve_rows.append({
                **run_ctx,
                "attack": name,
                "n_curve_batches": cfg.n_curve_batches,
                "n_positives": n_curve,
                "n_negatives": n_curve,
                **flat,
            })
    curve_path = out_dir / f"{cfg.label}_curve.csv"
    if curve_rows:
        curve_cols = list(run_ctx) + [
            "attack", "n_curve_batches", "n_positives", "n_negatives",
            "t_star", "bit_accuracy", "tpr_at_fpr", "tpr_at_fpr_attacked", "f1",
            "attack_sisnr", "attack_failure_rate", "n_attack_failures",
        ]
        with curve_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=curve_cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(curve_rows)

    json_path = out_dir / f"{cfg.label}_results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        # default=str: results carries a few non-JSON-native values (Paths,
        # numpy/torch scalars) that are only ever read back as text.
        json.dump(results, fh, indent=2, sort_keys=True, default=str)

    results["metrics_csv"] = str(metrics_path)
    if curve_rows:
        results["curve_csv"] = str(curve_path)
    results["results_json"] = str(json_path)
    logger.info(
        "wrote %s, %s%s", metrics_path, json_path, f", {curve_path}" if curve_rows else ""
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
        # Each metric is optional: evaluate_perceptual only emits the ones
        # its compute_* flags enabled, and pesq additionally drops out when
        # its backing package is missing or every batch raised. Formatting
        # them unconditionally used to KeyError here and take down the whole
        # summary *after* the expensive attack passes had already run.
        print("\nPerceptual (watermarked vs. clean, no attack):")
        available = [
            f"{name.upper()}: {perceptual[key]:.2f}{unit}"
            for key, name, unit in (("sisnr", "si-snr", " dB"), ("pesq", "pesq", ""), ("visqol", "visqol", ""))
            if key in perceptual
        ]
        print("  " + "    ".join(available) if available else "  (none computed)")

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

        # Caveats that invalidate the tpr@fpr printed above if ignored --
        # printed inline rather than left in the log, since this table is
        # what actually gets screenshotted into a report.
        failures = r.get("n_attack_failures")
        if failures:
            print(
                f"    !! attack did not perturb {failures} example(s) "
                f"({r['attack_failure_rate']:.1%}) -- those count as detections above, "
                f"inflating tpr@fpr"
            )
            if "tpr_at_fpr_attacked" in r:
                print(f"       tpr@fpr over perturbed-only: {r['tpr_at_fpr_attacked']:.3f}")

        support = r.get("fpr_support")
        if support and not support["supported"]:
            print(
                f"    !! only {support['n_negatives']} negatives -- cannot resolve "
                f"fpr below {support['fpr_resolution']:.3f}; tpr@fpr above is a "
                f"high-variance lower bound (need >= {support['min_negatives_for_target']})"
            )

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
    # Rank-aware logging: rank 0 at INFO, the others at WARNING, so a 4-GPU
    # run doesn't print every line four times. Returns the env read from
    # torchrun's environment variables (no process group needed).
    env = configure_logging(logging.INFO)
    cfg = load_eval_config()
    results = run(cfg)

    # Every rank holds the same all-gathered results; only one prints them.
    if not env.is_main:
        return

    _print_results_table(results)
    _print_timing_and_projection(cfg, results)
    for key, caption in (
        ("metrics_csv", "metrics csv"),
        ("curve_csv", "curve csv"),
        ("results_json", "results json"),
    ):
        if key in results:
            print(f"{caption}: {results[key]}")
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
