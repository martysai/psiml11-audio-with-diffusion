# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from audioseal_robust.metrics import (
    bit_accuracy,
    confusion_counts,
    detection_rate,
    f1_score,
    tpr_at_fpr,
    watermark_delta_rms,
    watermark_report,
    watermark_snr_db,
)


def test_bit_accuracy_perfect_and_random():
    message = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    perfect = torch.tensor([[0.9, 0.1, 0.8, 0.7]])
    assert bit_accuracy(perfect, message) == 1.0

    inverted = torch.tensor([[0.1, 0.9, 0.2, 0.3]])
    assert bit_accuracy(inverted, message) == 0.0


def test_tpr_at_fpr_separated_distributions():
    torch.manual_seed(0)
    # Watermarked audio scores high, clean audio scores low -- a detector
    # that's actually working should get ~perfect TPR at a strict FPR.
    positive_scores = 0.9 + 0.05 * torch.rand(200)
    negative_scores = 0.1 + 0.05 * torch.rand(200)

    tpr = tpr_at_fpr(positive_scores, negative_scores, target_fpr=0.01)
    assert tpr > 0.99


def test_tpr_at_fpr_identical_distributions_is_close_to_fpr():
    torch.manual_seed(0)
    # If positive and negative scores come from the same distribution (a
    # detector with no signal at all), TPR at FPR=x should be close to x --
    # by construction, the threshold is chosen so ~x fraction of negatives
    # exceed it, and positives are drawn from the same distribution.
    scores = torch.rand(5000)
    positive_scores = scores[:2500]
    negative_scores = scores[2500:]

    tpr = tpr_at_fpr(positive_scores, negative_scores, target_fpr=0.1)
    assert 0.05 < tpr < 0.2


def test_tpr_at_fpr_monotonic_in_target_fpr():
    torch.manual_seed(0)
    positive_scores = torch.rand(500)
    negative_scores = torch.rand(500)

    tpr_strict = tpr_at_fpr(positive_scores, negative_scores, target_fpr=0.01)
    tpr_loose = tpr_at_fpr(positive_scores, negative_scores, target_fpr=0.5)
    assert tpr_loose >= tpr_strict


def test_confusion_counts_separated_distributions():
    torch.manual_seed(0)
    # Same setup as test_tpr_at_fpr_separated_distributions: a working
    # detector should land almost all positives as TP and almost all
    # negatives as TN, with counts that add up to the input sizes.
    positive_scores = 0.9 + 0.05 * torch.rand(200)
    negative_scores = 0.1 + 0.05 * torch.rand(200)

    confusion = confusion_counts(positive_scores, negative_scores, target_fpr=0.01)
    assert confusion["tp"] + confusion["fn"] == 200
    assert confusion["fp"] + confusion["tn"] == 200
    assert confusion["tp"] > 195
    assert confusion["fp"] <= 2


def test_confusion_counts_does_not_exceed_fpr_budget():
    positive_scores = torch.ones(160)
    negative_scores = torch.arange(160, dtype=torch.float32)

    confusion = confusion_counts(positive_scores, negative_scores, target_fpr=0.01)
    assert confusion["fp"] == 1  # int(0.01 * 160) = 1 negative allowed through


def test_confusion_counts_zero_fpr_admits_no_negatives():
    confusion = confusion_counts(torch.ones(100), torch.zeros(100), target_fpr=0.0)

    assert confusion["fp"] == 0
    assert confusion["tp"] == 100


def test_confusion_counts_ties_at_threshold_stay_within_budget():
    negative_scores = torch.cat([torch.ones(2), torch.zeros(98)])

    confusion = confusion_counts(torch.ones(100), negative_scores, target_fpr=0.01)
    assert confusion["fp"] <= 1


def test_confusion_counts_matches_tpr_at_fpr():
    torch.manual_seed(0)
    positive_scores = torch.rand(500)
    negative_scores = torch.rand(500)

    tpr = tpr_at_fpr(positive_scores, negative_scores, target_fpr=0.1)
    confusion = confusion_counts(positive_scores, negative_scores, target_fpr=0.1)
    # TPR@FPR is exactly recall = TP / (TP + FN) at the same threshold.
    recall = confusion["tp"] / (confusion["tp"] + confusion["fn"])
    assert recall == pytest.approx(tpr)


def test_f1_score_perfect_and_no_signal():
    perfect = {"tp": 100, "fp": 0, "tn": 100, "fn": 0}
    assert f1_score(perfect) == 1.0

    no_positives_predicted = {"tp": 0, "fp": 0, "tn": 100, "fn": 100}
    assert f1_score(no_positives_predicted) == 0.0

    half_recall_full_precision = {"tp": 50, "fp": 0, "tn": 100, "fn": 50}
    assert f1_score(half_recall_full_precision) == pytest.approx(2 / 3)


def test_detection_rate_counts_scores_over_threshold():
    scores = torch.tensor([0.9, 0.6, 0.4, 0.1])
    assert detection_rate(scores) == 0.5
    assert detection_rate(scores, threshold=0.05) == 1.0
    assert detection_rate(scores, threshold=0.95) == 0.0


def _mix_at_target_snr(x: torch.Tensor, noise: torch.Tensor, target_snr_db: torch.Tensor) -> torch.Tensor:
    """x_wm = x + k * noise, with k chosen per clip so the watermark SNR is
    exactly `target_snr_db`. Reduces over every axis but the batch, matching
    watermark_snr_db."""
    reduce_dims = tuple(range(1, x.dim()))
    x_norm = x.pow(2).sum(dim=reduce_dims).sqrt()
    noise_norm = noise.pow(2).sum(dim=reduce_dims).sqrt()
    k = (x_norm / noise_norm) * 10 ** (-target_snr_db / 20)
    return x + k.reshape(-1, *([1] * (x.dim() - 1))) * noise


def test_watermark_snr_db_matches_known_target():
    torch.manual_seed(0)
    x = torch.randn(6, 1, 4000)
    noise = torch.randn(6, 1, 4000)
    target = torch.tensor([5.0, 15.0, 24.0, 30.0, 36.0, 60.0])

    x_wm = _mix_at_target_snr(x, noise, target)
    snr = watermark_snr_db(x, x_wm)

    assert snr.shape == (6,)
    assert torch.allclose(snr, target, atol=0.1)


def test_watermark_snr_db_accepts_2d_and_multichannel():
    torch.manual_seed(0)
    target = torch.full((4,), 30.0)

    x_2d = torch.randn(4, 4000)
    x_wm_2d = _mix_at_target_snr(x_2d, torch.randn(4, 4000), target)
    assert torch.allclose(watermark_snr_db(x_2d, x_wm_2d), target, atol=0.1)

    # (B, T) and its own (B, 1, T) view must agree exactly.
    assert torch.equal(watermark_snr_db(x_2d, x_wm_2d), watermark_snr_db(x_2d[:, None], x_wm_2d[:, None]))

    # Channels are reduced together with time, not treated as a batch axis.
    x_stereo = torch.randn(4, 2, 4000)
    x_wm_stereo = _mix_at_target_snr(x_stereo, torch.randn(4, 2, 4000), target)
    snr = watermark_snr_db(x_stereo, x_wm_stereo)
    assert snr.shape == (4,)
    assert torch.allclose(snr, target, atol=0.1)


def test_watermark_snr_db_is_per_clip_not_batch_pooled():
    """The whole point of reducing per clip: a loud clip in the same batch
    must not mask a quiet clip whose watermark is missing. Pooling energy
    over the batch first would report one blended ratio instead."""
    torch.manual_seed(0)
    loud = 100.0 * torch.randn(1, 1, 4000)
    quiet = 0.001 * torch.randn(1, 1, 4000)
    x = torch.cat([loud, quiet])
    target = torch.tensor([20.0, 40.0])

    x_wm = _mix_at_target_snr(x, torch.randn(2, 1, 4000), target)

    assert torch.allclose(watermark_snr_db(x, x_wm), target, atol=0.1)


def test_watermark_snr_db_zero_delta_is_large_and_finite():
    torch.manual_seed(0)
    x = torch.randn(3, 1, 4000)

    snr = watermark_snr_db(x, x.clone())

    assert torch.isfinite(snr).all()
    assert not torch.isnan(snr).any()
    assert (snr > 100).all()  # saturates at the eps floor rather than at inf


def test_watermark_snr_db_silent_clip_with_zero_delta_is_finite():
    """Degenerate case: no signal AND no watermark. Flooring only the
    denominator would give -inf here and poison every aggregate."""
    snr = watermark_snr_db(torch.zeros(2, 1, 100), torch.zeros(2, 1, 100))

    assert torch.isfinite(snr).all()
    assert torch.allclose(snr, torch.zeros(2))


def test_watermark_snr_db_rejects_mismatched_or_bad_shapes():
    with pytest.raises(ValueError):
        watermark_snr_db(torch.randn(2, 1, 10), torch.randn(2, 1, 11))
    with pytest.raises(ValueError):
        watermark_snr_db(torch.randn(2, 1, 4, 10), torch.randn(2, 1, 4, 10))


def test_watermark_delta_rms_is_zero_only_for_identical_input():
    x = torch.randn(3, 1, 100)

    assert torch.equal(watermark_delta_rms(x, x.clone()), torch.zeros(3))

    delta = torch.full((3, 1, 100), 0.25)
    assert torch.allclose(watermark_delta_rms(x, x + delta), torch.full((3,), 0.25))


def test_watermark_report_stats_and_percentiles():
    snr_db = torch.arange(101, dtype=torch.float32)  # 0..100
    delta_rms = torch.cat([torch.zeros(1), torch.ones(100)])

    report = watermark_report(snr_db, delta_rms)

    assert report["n_clips"] == 101
    assert report["snr_db"]["mean"] == pytest.approx(50.0)
    assert report["snr_db"]["min"] == 0.0
    assert report["snr_db"]["max"] == 100.0
    assert report["snr_db"]["p5"] == pytest.approx(5.0)
    assert report["snr_db"]["p50"] == pytest.approx(50.0)
    assert report["snr_db"]["p95"] == pytest.approx(95.0)
    # delta_rms_min exposes the single silent clip that the mean hides.
    assert report["delta_rms"] == pytest.approx(100 / 101)
    assert report["delta_rms_min"] == 0.0


def test_watermark_report_single_clip_has_no_nan_std():
    report = watermark_report(torch.tensor([30.0]), torch.tensor([0.01]))

    assert report["snr_db"]["std"] == 0.0
    assert report["snr_db"]["p50"] == pytest.approx(30.0)
