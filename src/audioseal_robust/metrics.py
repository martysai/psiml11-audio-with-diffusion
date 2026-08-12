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
  - a robustness *curve* (detection vs. attack strength t*) is not a single
    function here -- see `evaluate.py`, which sweeps `t_star_grid` and calls
    `tpr_at_fpr` at each point.

Perceptual (measured on watermarked vs. original, no attack):
  - `sisnr_score`, `pesq_score`: via `torchmetrics` (same library AudioCraft's
    own watermark solver uses for these, see audiocraft/solvers/watermark.py
    in the separate AudioCraft checkout -- reusing it here rather than
    reimplementing SI-SNR/PESQ from scratch).
  - `visqol_score`: stubbed, see docstring below -- ViSQOL isn't pip-installable.

All *_score functions import their backing library lazily (inside the
function) so that importing this module doesn't require torchmetrics/pesq
to be installed if you only need bit_accuracy/tpr_at_fpr.
"""

import typing as tp

import torch


def bit_accuracy(m_hat: torch.Tensor, message: torch.Tensor, threshold: float = 0.5) -> float:
    """Fraction of matching bits between the decoded message probabilities
    `m_hat` (B, nbits), in [0, 1], and the ground-truth `message` (B, nbits),
    in {0, 1}."""
    decoded = (m_hat > threshold).float()
    return (decoded == message.float()).float().mean().item()


def _threshold_at_fpr(negative_scores: torch.Tensor, target_fpr: float) -> float:
    """Detection threshold on `negative_scores` that yields (approximately)
    `target_fpr`: the score at or above which exactly `target_fpr` fraction
    of negatives fall. Shared by `tpr_at_fpr` and `confusion_counts` so both
    report numbers for the same operating point."""
    negative_sorted, _ = torch.sort(negative_scores, descending=True)
    n_neg = negative_sorted.numel()
    idx = max(0, min(n_neg - 1, int(round(target_fpr * n_neg))))
    return negative_sorted[idx].item()


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


def sisnr_score(x: torch.Tensor, x_wm: torch.Tensor) -> float:
    """Scale-invariant SNR between clean `x` and watermarked `x_wm`, both
    (B, 1, T). Higher is better (less audible watermark). Cheap enough to
    log every eval step."""
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
