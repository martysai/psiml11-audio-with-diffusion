# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluation metrics, split into the two axes that matter for this
project:

Robustness (measured on audio that has gone through an attack):
  - `bit_accuracy`: fraction of decoded message bits that match.
  - `tpr_at_fpr`: true-positive rate at a fixed false-positive rate --
    more honest than accuracy at a fixed 0.5 threshold, because it fixes
    the operating point that actually matters for a watermark detector
    (how often do we falsely flag clean audio) instead of an arbitrary one.
  - `confusion_counts` / `f1_score`: the same positive/negative presence
    scores as `tpr_at_fpr`, but collapsed into a binary "is the message
    still detected as present" call (TP/FP/TN/FN) at that same threshold,
    plus their F1 -- for when you want the full confusion matrix rather
    than just recall (= `tpr_at_fpr`) at one operating point.
  - `fpr_support`: whether the negative sample size can resolve the
    requested FPR at all -- a guard against quoting a TPR@FPR that is
    really a max-of-N order statistic (see its docstring).
  - a robustness *curve* (detection vs. attack strength t*) is not a single
    function here -- see `evaluate.py`, which sweeps `t_star_grid` and calls
    `tpr_at_fpr` at each point.

Perceptual (measured on watermarked vs. original, no attack):
  - `sisnr_score`, `pesq_score`: via `torchmetrics` (same library AudioCraft's
    own watermark solver uses for these, see audiocraft/solvers/watermark.py
    in the separate AudioCraft checkout -- reusing it here rather than
    reimplementing SI-SNR/PESQ from scratch).
  - `visqol_score`: stubbed, see docstring below -- ViSQOL isn't pip-installable.

Diagnostic (also watermarked vs. original, no attack -- but read *before*
believing any robustness number, not as a quality score):
  - `watermark_snr_db` / `watermark_delta_rms` / `watermark_report`: how loud
    the watermark perturbation actually is. Near-chance detection has two very
    different causes -- the generator emitting an effectively-zero delta, or a
    fine delta that something downstream fails to detect -- and these separate
    them for the price of two reductions. Deliberately NOT scale-invariant
    (unlike `sisnr_score`): the absolute size of `x_wm - x` relative to `x` is
    exactly the thing in question, and SI-SNR's per-clip rescaling would hide
    a delta that is uniformly too small.

All *_score functions import their backing library lazily (inside the
function) so that importing this module doesn't require torchmetrics/pesq
to be installed if you only need bit_accuracy/tpr_at_fpr.
"""

import logging
import math
import typing as tp

import torch

logger = logging.getLogger(__name__)


def fpr_support(n_negatives: int, target_fpr: float) -> tp.Dict[str, tp.Any]:
    """Whether `n_negatives` samples can actually resolve `target_fpr`.

    An empirical FPR estimated from N negatives is quantized to multiples of
    1/N -- you cannot measure a 1% false-positive rate with 16 samples any
    more than you can measure it with a coin flip. `_threshold_at_fpr`
    allows `int(target_fpr * N)` false positives, so once N < 1/target_fpr
    that budget floors to ZERO and the "threshold at `target_fpr`" silently
    degenerates into "just above the single highest negative score" -- a
    max-of-N order statistic with enormous variance, not a 1% operating
    point. The number still prints, which is exactly why this has to be
    reported next to it rather than left to the reader.

    Returns the resolution (1/N), the minimum N that makes `target_fpr`
    representable at all (1/target_fpr, i.e. a budget of >= 1 false
    positive), and whether this sample size clears it. `supported=False`
    means the reported TPR@FPR is a lower bound of unknown tightness and
    must not be quoted as a headline number -- raise batch_size *
    n_eval_batches until it flips to True.
    """
    resolution = 1.0 / n_negatives if n_negatives > 0 else float("inf")
    min_negatives = int(math.ceil(1.0 / target_fpr)) if target_fpr > 0 else 0
    return {
        "n_negatives": n_negatives,
        "fpr_resolution": resolution,
        "min_negatives_for_target": min_negatives,
        "supported": n_negatives >= min_negatives,
    }


def bit_accuracy(m_hat: torch.Tensor, message: torch.Tensor, threshold: float = 0.5) -> float:
    """Fraction of matching bits between the decoded message probabilities
    `m_hat` (B, nbits), in [0, 1], and the ground-truth `message` (B, nbits),
    in {0, 1}."""
    decoded = (m_hat > threshold).float()
    return (decoded == message.float()).float().mean().item()


def detection_rate(positive_scores: torch.Tensor, threshold: float = 0.5) -> float:
    """Fraction of watermarked clips whose presence probability clears a
    fixed `threshold`.

    Unlike `tpr_at_fpr` this does NOT calibrate against the negatives, which
    makes it the wrong number to quote for robustness -- but the right one
    for a diagnostic: it answers "does the detector fire at all on our
    watermarked audio" without the answer depending on how the (possibly
    equally broken) negatives happened to score. A run where `tpr_at_fpr` is
    near chance but `detection_rate` is ~1.0 means the detector fires on
    everything; both near chance means it fires on nothing.
    """
    assert positive_scores.numel() > 0
    return (positive_scores >= threshold).float().mean().item()


def _threshold_at_fpr(negative_scores: torch.Tensor, target_fpr: float) -> float:
    """Detection threshold whose empirical FPR does not exceed `target_fpr`:
    just above the (k+1)-th highest negative score, where k is the number of
    false positives the budget allows. Stepping past that score (rather than
    landing on it) keeps ties at the boundary from spending more than the
    budget. Shared by `tpr_at_fpr` and `confusion_counts` so both report
    numbers for the same operating point.

    NOTE: `n_allowed` floors to 0 whenever `negative_scores` holds fewer than
    `1 / target_fpr` samples, at which point this returns "just above the
    maximum negative score" and the resulting TPR is a high-variance lower
    bound rather than a real operating point. That degeneracy is silent here
    by design (this is a pure threshold helper); call `fpr_support` to detect
    it and report it alongside -- `evaluate.py:evaluate_attack` does.
    """
    negative_sorted, _ = torch.sort(negative_scores, descending=True)
    n_allowed = int(target_fpr * negative_sorted.numel())
    if n_allowed >= negative_sorted.numel():
        return negative_sorted[-1].item()
    boundary = negative_sorted[n_allowed]
    return torch.nextafter(boundary, torch.full_like(boundary, float("inf"))).item()


def tpr_at_fpr(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    target_fpr: float = 0.01,
) -> float:
    """True-positive rate at a fixed false-positive rate.

    Args:
        positive_scores: presence probability p (detector's "watermarked"
            class, pooled over time) measured on watermarked audio, shape (N,).
        negative_scores: the same, measured on audio that was never
            watermarked (run through the same attack, for a fair comparison
            -- see evaluate.py), shape (N,).
        target_fpr: desired false-positive rate, e.g. 0.01 for 1%.

    Finds the detection threshold on `negative_scores` that yields
    (approximately) `target_fpr`, then reports the fraction of
    `positive_scores` at or above that same threshold. This is the standard
    way to report a detector's robustness without conflating it with an
    arbitrary fixed threshold: a detector that's "90% accurate at
    threshold=0.5" can still have a terrible false-positive rate if its
    score distribution is shifted, TPR@FPR pins down the actually-relevant
    operating point.
    """
    assert positive_scores.numel() > 0 and negative_scores.numel() > 0
    threshold = _threshold_at_fpr(negative_scores, target_fpr)
    return (positive_scores >= threshold).float().mean().item()


def confusion_counts(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    target_fpr: float = 0.01,
) -> tp.Dict[str, int]:
    """Binary confusion matrix for "is the message still detected as
    present", at the same threshold `tpr_at_fpr` uses (found on
    `negative_scores` at `target_fpr`).

    Positive class = watermarked audio (message was embedded, possibly then
    attacked); negative class = audio that was never watermarked (also run
    through the same attack) -- the same pairing `evaluate.py` already
    builds for the TPR@FPR comparison.

        TP: watermarked audio, message still detected as present.
        FN: watermarked audio, message lost (the attack erased it).
        FP: clean audio, incorrectly flagged as watermarked.
        TN: clean audio, correctly flagged as clean.
    """
    assert positive_scores.numel() > 0 and negative_scores.numel() > 0
    threshold = _threshold_at_fpr(negative_scores, target_fpr)
    return {
        "tp": int((positive_scores >= threshold).sum().item()),
        "fn": int((positive_scores < threshold).sum().item()),
        "fp": int((negative_scores >= threshold).sum().item()),
        "tn": int((negative_scores < threshold).sum().item()),
    }


def f1_score(confusion: tp.Dict[str, int]) -> float:
    """F1 of the "message preserved" binary call, from a `confusion_counts()`
    dict. 0.0 if precision and recall are both 0 (no positive predictions
    at all)."""
    tp_, fp_, fn_ = confusion["tp"], confusion["fp"], confusion["fn"]
    precision = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
    recall = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _as_bct(x: torch.Tensor, x_wm: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """Normalize a (B, T) or (B, C, T) pair to (B, C, T), checking they agree."""
    if x.shape != x_wm.shape:
        raise ValueError(f"x and x_wm must have the same shape, got {tuple(x.shape)} vs {tuple(x_wm.shape)}")
    if x.dim() == 2:
        return x.unsqueeze(1), x_wm.unsqueeze(1)
    if x.dim() == 3:
        return x, x_wm
    raise ValueError(f"expected (B, T) or (B, C, T) audio, got {x.dim()}D tensor {tuple(x.shape)}")


def watermark_snr_db(x: torch.Tensor, x_wm: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-clip watermark SNR in dB: 10 * log10(||x||^2 / ||x_wm - x||^2).

    Args:
        x: clean input, (B, C, T) or (B, T), float.
        x_wm: generator output *before any attack*, same shape as `x`.
        eps: floor for the squared norms, see below.

    Returns a (B,) tensor of per-clip dB values.

    Reduced over the channel and time axes ONLY -- never over the batch.
    Pooling energy across the batch first would report a single ratio of
    summed energies, in which one loud clip's signal energy masks a quiet
    clip's missing watermark; per-clip-then-aggregate keeps that visible in
    the spread (see `watermark_report`, which reports percentiles for exactly
    this reason).

    Both sums are floored at `eps` rather than just the denominator: flooring
    the noise term alone still returns -inf for a digitally-silent clip
    (0 signal energy, 0 delta), and this is a diagnostic that gets aggregated
    -- one -inf would poison the mean and the reported min. With both floored,
    that degenerate case reads 0 dB and the accompanying `delta_rms` (exactly
    0.0) says which of the two zero cases it was.
    """
    x, x_wm = _as_bct(x, x_wm)
    delta = x_wm - x
    signal = x.pow(2).sum(dim=(1, 2)).clamp_min(eps)
    noise = delta.pow(2).sum(dim=(1, 2)).clamp_min(eps)
    return 10.0 * torch.log10(signal / noise)


def watermark_delta_rms(x: torch.Tensor, x_wm: torch.Tensor) -> torch.Tensor:
    """Per-clip RMS of the watermark perturbation `x_wm - x`, (B,).

    The companion to `watermark_snr_db`, which saturates at a large finite
    number (not inf) when the delta is exactly zero and so can't by itself
    distinguish "no watermark at all" from "an extremely quiet one". This
    reads exactly 0.0 in the former case. Absolute units (same as the audio),
    so it is not comparable across datasets -- only against zero.
    """
    x, x_wm = _as_bct(x, x_wm)
    return (x_wm - x).pow(2).mean(dim=(1, 2)).sqrt()


def _per_clip_stats(values: torch.Tensor) -> tp.Dict[str, float]:
    """mean/std/min/max + 5th/50th/95th percentiles of a (N,) tensor.

    Population std (`unbiased=False`) so a single clip reports 0.0 rather
    than nan -- this runs on whatever the eval set happened to produce.
    """
    values = values.detach().float().flatten()
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.5, 0.95], dtype=values.dtype))
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "min": values.min().item(),
        "max": values.max().item(),
        "p5": quantiles[0].item(),
        "p50": quantiles[1].item(),
        "p95": quantiles[2].item(),
    }


def watermark_report(
    snr_db: torch.Tensor,
    delta_rms: torch.Tensor,
    raw_snr_db: tp.Optional[torch.Tensor] = None,
    raw_delta_rms: tp.Optional[torch.Tensor] = None,
) -> tp.Dict[str, tp.Any]:
    """Aggregate per-clip `watermark_snr_db` / `watermark_delta_rms` values
    (concatenated over every eval batch) into the reported diagnostic.

        {"snr_db": {mean, std, min, max, p5, p50, p95},
         "delta_rms": <mean>, "delta_rms_min": <min>, "n_clips": <N>}

    `snr_db.p50` is the headline: dB is a log scale, so a few clips landing
    at the ~150 dB zero-delta saturation point drag the mean far more than
    they should. The spread (p5..p95) is the second thing to read -- a wide
    distribution means the perturbation is not being normalized per clip, so
    a good median is hiding clips with no usable watermark.

    `delta_rms_min` is reported alongside the mean so that a *subset* of
    clips with an exactly-zero delta is still unambiguous.

    `raw_snr_db` / `raw_delta_rms` are the same two measurements taken on the
    generator's output BEFORE SNR normalization scales it (see
    train.py:embed_watermark). Pass them whenever the embedding is
    normalized, and read them as the actual generator-health signal: the
    post-scaling `snr_db` is pinned to the configured target by construction
    and therefore reports the same value for a healthy generator and for one
    whose delta has collapsed to zero. They land under `raw_snr_db` and
    `raw_delta_rms` / `raw_delta_rms_min`, and are simply absent when the
    embedding is unscaled (in which case `snr_db` is already the raw number).
    """
    assert snr_db.numel() == delta_rms.numel() and snr_db.numel() > 0
    report: tp.Dict[str, tp.Any] = {
        "snr_db": _per_clip_stats(snr_db),
        "delta_rms": delta_rms.detach().float().mean().item(),
        "delta_rms_min": delta_rms.detach().float().min().item(),
        "n_clips": int(snr_db.numel()),
    }
    if raw_snr_db is not None:
        assert raw_snr_db.numel() == snr_db.numel()
        report["raw_snr_db"] = _per_clip_stats(raw_snr_db)
    if raw_delta_rms is not None:
        assert raw_delta_rms.numel() == snr_db.numel()
        raw_delta_rms = raw_delta_rms.detach().float()
        report["raw_delta_rms"] = raw_delta_rms.mean().item()
        report["raw_delta_rms_min"] = raw_delta_rms.min().item()
    return report


def sisnr_score(x: torch.Tensor, x_wm: torch.Tensor) -> float:
    """Scale-invariant SNR between clean `x` and watermarked `x_wm`, both
    (B, 1, T). Higher is better (less audible watermark). Cheap enough to
    log every eval step.

    Note this is *scale-invariant*: it rescales `x_wm` to best match `x`
    before measuring, so it is a perceptual-quality number, not a measure of
    how large the perturbation actually is. Use `watermark_snr_db` for the
    latter."""
    from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio

    metric = ScaleInvariantSignalNoiseRatio().to(x.device)
    return metric(x_wm, x).item()


def pesq_score(x: torch.Tensor, x_wm: torch.Tensor, sample_rate: int) -> float:
    """Perceptual Evaluation of Speech Quality between clean `x` and
    watermarked `x_wm`, both (B, 1, T). Needs `pip install pesq
    torchmetrics`. Uses wideband mode for sample_rate >= 16kHz, narrowband
    otherwise (PESQ only supports 8kHz/16kHz internally; torchmetrics
    resamples for you if your sample_rate differs, but AudioSeal's 16kHz
    default lines up with wideband directly)."""
    from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

    mode = "wb" if sample_rate >= 16_000 else "nb"
    metric = PerceptualEvaluationSpeechQuality(sample_rate, mode).to(x.device)
    return metric(x_wm.squeeze(1), x.squeeze(1)).item()


def visqol_score(x: torch.Tensor, x_wm: torch.Tensor, sample_rate: int) -> float:
    """TODO(setup): ViSQOL (https://github.com/google/visqol) is a Bazel
    C++ build, not a pip package -- there's no pure-Python wheel to lazily
    import here. To enable:
      1. Build the `visqol` binary per its README (or use AudioCraft's
         `audiocraft.solvers.builders.get_visqol` wrapper as a reference --
         that's a subprocess-based wrapper around the same binary, in the
         separate AudioCraft checkout, not this repo).
      2. Implement this function as: write x/x_wm to temp wavs, shell out to
         the binary, parse its MOS-LQO score out of stdout/the result proto.
    """
    raise NotImplementedError(
        "visqol_score needs a local ViSQOL binary build -- see the TODO in "
        "src/audioseal_robust/metrics.py. Set compute_visqol=false in the "
        "eval config to skip it until then."
    )
