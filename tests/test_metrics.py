# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from audioseal_robust.metrics import bit_accuracy, confusion_counts, f1_score, tpr_at_fpr


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
