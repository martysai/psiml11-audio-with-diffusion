# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for gradient normalization (OptimConfig.normalize_grad).

`clip_grad_norm_` only ever shrinks, so in a mixed-attack recipe it silently
reweights the branches: the branch whose gradients explode is scaled down on
every step while the cheap branch passes through untouched. Measured on the
4x A100 run train-audioldm-mixed-0814-155734, median parameter-gradient norms
were 1.69 (identity, under max_norm=3.0) versus 42.94 (audioldm, over it on
every step and cut 14.3x), and the audioldm branch did not move in 1550 steps.

Normalizing rescales to exactly `max_norm` in both directions, so a step's
contribution no longer depends on which branch produced it.
"""

import math

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from audioseal_robust.train import _grad_norm, _normalize_grad_  # noqa: E402


def _module_with_grad(norm: float) -> nn.Module:
    """A module whose parameter gradients have a known global L2 norm."""
    module = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        module.weight.grad = torch.ones_like(module.weight)
        module.weight.grad *= norm / module.weight.grad.norm(2)
    return module


@pytest.mark.parametrize("start_norm", [0.5, 1.69, 42.94, 1000.0])
def test_normalizes_to_exactly_the_target(start_norm):
    """Both directions: a small gradient is scaled UP, a large one DOWN.
    Scaling up is what clip_grad_norm_ cannot do, and is the half that
    equalizes the cheap branch against the exploding one."""
    module = _module_with_grad(start_norm)
    target = 3.0

    returned = _normalize_grad_(module, target, floor=1e-6)

    assert returned == pytest.approx(start_norm, rel=1e-5), "must return the PRE-scaling norm"
    assert _grad_norm(module) == pytest.approx(target, rel=1e-5)


def test_direction_is_preserved():
    """Only the magnitude changes -- every gradient is multiplied by one
    shared scalar, so the update direction is untouched."""
    module = nn.Linear(3, 3, bias=False)
    module.weight.grad = torch.tensor(
        [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0], [-7.0, 8.0, 9.0]]
    )
    before = module.weight.grad.clone()

    _normalize_grad_(module, 3.0, floor=1e-6)

    after = module.weight.grad
    cosine = torch.nn.functional.cosine_similarity(
        before.flatten(), after.flatten(), dim=0
    )
    assert cosine == pytest.approx(1.0, abs=1e-6)


def test_below_floor_is_left_alone():
    """A gradient that is essentially numerical noise must not be amplified
    into a full-sized step."""
    module = _module_with_grad(1e-9)

    _normalize_grad_(module, 3.0, floor=1e-6)

    assert _grad_norm(module) == pytest.approx(1e-9, rel=1e-3), "should be untouched"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_gradients_raise(bad):
    """Matches clip_grad_norm_(error_if_nonfinite=True): a NaN/Inf gradient is
    a broken run, not a large one, and must not be silently rescaled."""
    module = nn.Linear(2, 2, bias=False)
    module.weight.grad = torch.full_like(module.weight, bad)

    with pytest.raises(RuntimeError, match="non-finite"):
        _normalize_grad_(module, 3.0, floor=1e-6)


def test_equalizes_two_branches_that_clipping_would_not():
    """The property this exists for.

    Reproduces the measured per-branch norms. Under clipping the branches end
    up with very different update magnitudes; under normalization they match.
    """
    target = 3.0
    identity_norm, audioldm_norm = 1.69, 42.94

    # --- what clip_grad_norm_ does -----------------------------------------
    clipped = {}
    for name, norm in (("identity", identity_norm), ("audioldm", audioldm_norm)):
        module = _module_with_grad(norm)
        torch.nn.utils.clip_grad_norm_(module.parameters(), target)
        clipped[name] = _grad_norm(module)

    assert clipped["identity"] == pytest.approx(identity_norm, rel=1e-5), "under the cap, untouched"
    assert clipped["audioldm"] == pytest.approx(target, rel=1e-5), "over the cap, cut to it"
    # The imbalance: identity contributes a materially smaller update than the
    # branch we actually care about... in the opposite direction to intent,
    # because audioldm's 42.94 was cut 14x while identity kept its full 1.69.
    ratio_clipped = clipped["audioldm"] / clipped["identity"]
    assert ratio_clipped == pytest.approx(target / identity_norm, rel=1e-5)

    # --- what normalization does -------------------------------------------
    normalized = {}
    for name, norm in (("identity", identity_norm), ("audioldm", audioldm_norm)):
        module = _module_with_grad(norm)
        _normalize_grad_(module, target, floor=1e-6)
        normalized[name] = _grad_norm(module)

    assert normalized["identity"] == pytest.approx(normalized["audioldm"], rel=1e-5)
    assert normalized["audioldm"] == pytest.approx(target, rel=1e-5)


def test_config_exposes_the_knobs_off_by_default():
    """Default must stay clipping: normalization changes the update rule for
    every recipe, and single-attack runs do not have the imbalance it fixes."""
    from audioseal_robust.config import OptimConfig

    cfg = OptimConfig()
    assert cfg.normalize_grad is False
    assert cfg.max_norm == 3.0
    assert math.isclose(cfg.normalize_grad_floor, 1e-6)
