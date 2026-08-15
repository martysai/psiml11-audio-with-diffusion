# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for rank-shared attack strength sampling.

`SampledReconstructionAttack` already synchronised which branch is drawn each
step, because the branches cost wildly different amounts and DDP synchronises
at every backward. The strength (t*) each branch then sampled internally came
from the *global* RNG, which distributed.seed_everything deliberately seeds per
rank -- so every rank drew a different t*.

For AudioLDM that is not merely cosmetic: t* sets the number of
reverse-diffusion steps, so the step costs whatever the deepest draw on any
rank costs. At world_size=4 that is E[max of 4 uniforms] = 0.8 * strength_max
against 0.5 for a single draw, roughly 1.6x the expected work on every step,
plus peak memory set by the unluckiest rank.
"""

import random

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from audioseal_robust.attacks import IdentityAttack, SampledReconstructionAttack  # noqa: E402


class _StrengthProbe(nn.Module):
    """Records the strength it was invoked with, like a strength-aware attack."""

    def __init__(self, strength_max: float = 0.08):
        super().__init__()
        self.strength_max = strength_max
        self.seen: list = []
        self._strength_rng: random.Random = random  # type: ignore[assignment]

    def forward(self, x, strength=None):
        if strength is None:
            strength = self._strength_rng.random() * self.strength_max
        self.seen.append(strength)
        return x


def _build(seed: int = 1234):
    probe = _StrengthProbe()
    attack = SampledReconstructionAttack(
        {"identity": IdentityAttack(), "probe": probe},
        {"identity": 0.0, "probe": 1.0},
        rng=random.Random(seed),
    )
    return attack, probe


def test_shared_rng_is_injected_into_branches():
    attack, probe = _build()
    assert probe._strength_rng is attack._rng, "branch must draw from the shared generator"


def test_two_ranks_with_the_same_seed_draw_identical_strengths():
    """The property that matters: same shared seed -> same t* sequence, so
    every rank runs the same number of diffusion steps."""
    x = torch.zeros(1, 1, 4)

    rank0, probe0 = _build(seed=1234)
    rank1, probe1 = _build(seed=1234)
    for _ in range(8):
        rank0(x)
        rank1(x)

    assert probe0.seen == probe1.seen
    assert len(probe0.seen) == 8


def test_strengths_still_vary_across_steps():
    """Sharing across ranks must not collapse the draw to a constant -- the
    point of sampling t* per step is coverage of attack strengths."""
    attack, probe = _build()
    x = torch.zeros(1, 1, 4)
    for _ in range(20):
        attack(x)

    assert len(set(probe.seen)) > 1, "t* should still differ between steps"
    assert all(0.0 <= s <= probe.strength_max for s in probe.seen)


def test_global_random_does_not_perturb_the_shared_draw():
    """Rank-local RNG use (seed_everything seeds `random` per rank, and other
    code draws from it) must not shift the shared sequence."""
    x = torch.zeros(1, 1, 4)

    baseline, probe_a = _build(seed=99)
    for _ in range(5):
        baseline(x)

    perturbed, probe_b = _build(seed=99)
    for _ in range(5):
        random.random()  # stand-in for unrelated rank-local randomness
        perturbed(x)
        random.random()

    assert probe_a.seen == probe_b.seen


def test_explicit_strength_still_bypasses_sampling():
    attack, probe = _build()
    attack(torch.zeros(1, 1, 4), strength=0.05)
    assert probe.seen == [0.05]
