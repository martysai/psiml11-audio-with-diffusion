# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for pinning the attack branch instead of sampling it.

`SampledReconstructionAttack.forward` draws a random branch, which is right
for training and wrong for measurement. `eval_step` used to call it that way,
so each eval point measured a *different task*: with identity and audioldm in
one recipe the eval curve alternated between two populations (loss ~0.8 vs
~5), which reads as instability or a regression rather than the
branch-switching it actually was. Simulating the seeded RNG predicted the drawn
branch for 7/7 observed draws: eval at step 0 drew identity, while eval at step
83 drew audioldm.

Sampling in eval also consumed from the shared RNG, so evaluating shifted the
branch sequence training itself saw.
"""

import random

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from audioseal_robust.attacks import IdentityAttack, SampledReconstructionAttack  # noqa: E402

SEED = 1234


class _ScaleAttack(nn.Module):
    """Stands in for a real attack; identifiable by what it does to the input."""

    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def forward(self, x, strength=None):
        return x * self.factor


def _build(weights=None):
    attacks = {"identity": IdentityAttack(), "loud": _ScaleAttack(3.0)}
    weights = weights or {"identity": 0.5, "loud": 0.5}
    return SampledReconstructionAttack(attacks, weights)


def test_branch_names_lists_only_enabled_attacks():
    attack = _build({"identity": 1.0, "loud": 0.0})
    assert attack.branch_names == ["identity"], "weight-0 branches can never be sampled"


def test_explicit_name_selects_that_branch():
    attack = _build()
    x = torch.ones(2, 1, 8)

    out, name = attack(x, name="loud")

    assert name == "loud"
    assert torch.allclose(out, x * 3.0)


def test_explicit_name_is_deterministic_across_calls():
    """The property the eval fix depends on: same name, same result, no matter
    how many times it is called or what the RNG is doing."""
    attack = _build()
    x = torch.randn(2, 1, 8)

    names = [attack(x, name="identity")[1] for _ in range(10)]

    assert names == ["identity"] * 10


def test_explicit_name_does_not_consume_the_shared_rng():
    """Evaluating must not shift the branch sequence training sees. From the
    same seed, the sampled sequence must be identical whether or not
    explicit-name calls are interleaved between the draws."""
    attack = _build()
    x = torch.ones(1, 1, 4)

    random.seed(SEED)
    expected = [attack(x)[1] for _ in range(6)]

    random.seed(SEED)
    got = []
    for _ in range(6):
        got.append(attack(x)[1])
        attack(x, name="identity")  # an "eval" in between
        attack(x, name="loud")

    assert got == expected


def test_sampling_still_happens_without_a_name():
    """The training path is unchanged: no name means draw one, and over enough
    draws both branches appear."""
    attack = _build()
    x = torch.ones(1, 1, 4)

    random.seed(SEED)
    drawn = {attack(x)[1] for _ in range(50)}

    assert drawn == {"identity", "loud"}


def test_unknown_name_raises():
    attack = _build()
    with pytest.raises(KeyError, match="unknown attack"):
        attack(torch.ones(1, 1, 4), name="nope")


def test_sampled_sequence_is_reproducible_for_a_given_seed():
    """Underpins the diagnosis: the drawn branch is a pure function of the
    seed and the call count, which is how the eval artifact was confirmed
    rather than guessed at."""
    x = torch.ones(1, 1, 4)
    attack = _build()

    random.seed(SEED)
    sequence = [attack(x)[1] for _ in range(8)]

    random.seed(SEED)
    assert [attack(x)[1] for _ in range(8)] == sequence
