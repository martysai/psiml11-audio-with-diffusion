# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-GPU (DistributedDataParallel) plumbing, shared by train.py,
evaluate.py and sanity_check.py.

The whole module is a no-op passthrough when launched as a single process:
every helper here takes a `DistEnv`, and a non-distributed `DistEnv`
(world_size=1, the default) makes each of them return its input unchanged
without touching `torch.distributed` at all. That is deliberate -- the
single-GPU code path this project started with must keep working, and keep
producing identical numbers, without a `torchrun` wrapper.

Launching
---------
Processes are expected to come from `torchrun`, which sets RANK,
LOCAL_RANK and WORLD_SIZE in the environment:

    torchrun --standalone --nproc_per_node=4 -m audioseal_robust.train ...

One process per GPU (NOT one process driving 4 GPUs via `nn.DataParallel`):
DataParallel re-scatters the model every step and serializes everything
through rank 0's Python process, which for this pipeline would be
particularly bad -- the frozen diffusion attacks (sgmse, diff_erase) are the
expensive part of every step and would all queue behind one interpreter.

What is and isn't reduced
-------------------------
DDP itself only averages *gradients*. Everything a human reads -- logged
losses, TPR@FPR, PESQ, peak memory -- is computed per rank on that rank's
own shard, so it has to be combined explicitly, which is what
`all_reduce_mean` / `all_gather_scores` / `all_reduce_max` are for. The
distinction that matters for correctness:

  - Losses are *averages over examples*, so averaging the per-rank averages
    is exact as long as every rank saw the same number of examples (which
    `drop_last=True` guarantees) -- see `all_reduce_mean`.
  - TPR@FPR and the confusion matrix are NOT averages: they are computed
    against a threshold read off a *quantile* of the negative-score
    distribution (see metrics._threshold_at_fpr). A per-rank threshold
    averaged across ranks is not the global threshold, and with 4 ranks x a
    1% FPR budget the per-rank quantile is estimated from a quarter of the
    samples. So the raw scores are gathered and the metric is computed once
    on the full set -- see `all_gather_scores` and evaluate.py.
"""

import datetime
import logging
import os
import random
import typing as tp
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from .device import resolve_device

logger = logging.getLogger(__name__)

# Generous: rank 0 can spend several minutes alone in a HF checkpoint
# download or a first-call MBD weights fetch while the others sit in a
# barrier, and the default 10 minutes has bitten this kind of pipeline before.
_DEFAULT_TIMEOUT_MINUTES = 30


@dataclass(frozen=True)
class DistEnv:
    """Immutable snapshot of the distributed context.

    The default value is the single-process case, so `DistEnv()` is a valid
    "not distributed" env that every helper in this module accepts.
    """

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: tp.Optional[str] = None

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """Rank 0. Anything with a side effect outside this process --
        experiment tracking, checkpoint files, plots, stdout summaries --
        must be guarded on this, or you get `world_size` copies of it (4
        MLflow runs per training run, 4 processes writing the same
        checkpoint path concurrently)."""
        return self.rank == 0


def env_from_environment() -> DistEnv:
    """Read the torchrun-provided RANK/LOCAL_RANK/WORLD_SIZE without
    initializing anything. Returns the single-process default when they are
    absent (i.e. plain `python -m ...`)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistEnv(rank=rank, local_rank=local_rank, world_size=world_size)


def init_distributed(
    requested_device: str = "auto",
    timeout_minutes: int = _DEFAULT_TIMEOUT_MINUTES,
) -> tp.Tuple[DistEnv, torch.device]:
    """Initialize the process group (if launched under torchrun) and resolve
    this rank's device.

    Returns `(env, device)`. When not launched under torchrun this is exactly
    `(DistEnv(), resolve_device(requested_device))` -- no process group, no
    NCCL, no behavior change.

    The returned device is rank-local (`cuda:{local_rank}`), and
    `torch.cuda.set_device` is called so that anything constructing tensors
    with a bare `"cuda"` or `.cuda()` -- including code inside the vendored
    attack backbones, which this repo does not control -- still lands on
    this rank's GPU instead of piling every rank onto cuda:0.
    """
    env = env_from_environment()

    if not env.is_distributed:
        return env, resolve_device(requested_device)

    device = resolve_device(requested_device, local_rank=env.local_rank)

    # NCCL is the only sane choice for CUDA; gloo covers CPU-only runs, which
    # is what the distributed unit tests use (and what a laptop smoke test of
    # the DDP wiring gets).
    backend = "nccl" if device.type == "cuda" and dist.is_nccl_available() else "gloo"
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(minutes=timeout_minutes),
        )
    env = DistEnv(rank=env.rank, local_rank=env.local_rank, world_size=env.world_size, backend=backend)
    logger.info(
        "distributed initialized: rank=%d/%d local_rank=%d backend=%s device=%s",
        env.rank, env.world_size, env.local_rank, backend, device,
    )
    return env, device


def cleanup_distributed(env: DistEnv) -> None:
    """Tear the process group down. Safe to call unconditionally, including
    on the single-process path and in a `finally:` after a crash."""
    if env.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def barrier(env: DistEnv) -> None:
    if env.is_distributed and dist.is_initialized():
        dist.barrier()


def seed_everything(seed: int, env: DistEnv) -> None:
    """Seed torch/random with a *rank-dependent* seed.

    Every rank must NOT draw the same random numbers: the watermark messages
    (train.random_message), the per-example target SNR, the dataset's random
    crops and each strength-aware attack's random t* are all sampled per
    step, and identical draws on 4 ranks would make a 4x larger batch that
    is only 1x more diverse -- exactly the diversity DDP is supposed to buy.

    Model *initialization* does not need matching seeds here because both the
    generator and the detector are always loaded from a checkpoint (see
    train.build_generator); DDP additionally broadcasts rank 0's parameters
    at construction time, so the replicas are identical regardless.

    The one thing that must stay in lockstep is which attack branch gets
    sampled each step -- that gets its own explicitly-shared RNG rather than
    relying on the global one, see `attack_sampling_rng`.
    """
    torch.manual_seed(seed + env.rank)
    random.seed(seed + env.rank)


def attack_sampling_rng(seed: int) -> random.Random:
    """A `random.Random` seeded identically on every rank, for
    `SampledReconstructionAttack` to pick its branch from.

    Why the branch specifically has to agree across ranks, when nothing else
    does: the branches have wildly different costs (identity is free, sgmse
    is a 30-step diffusion sampler). Since DDP synchronizes gradients at
    every backward, an unsynchronized choice makes every step cost the
    *slowest* branch drawn by *any* rank, so with 4 ranks and a small sgmse
    weight, nearly every step pays for sgmse. Sharing the draw keeps a step's
    cost equal to the branch that was actually sampled.

    A dedicated `random.Random` instance is used instead of a per-step
    broadcast because it needs no collective at all: same seed, same call
    sequence (every rank calls the attack exactly once per train step and
    once per eval step), therefore same branch -- while leaving the global
    `random`/torch seeds free to differ per rank, which is what
    `seed_everything` wants.
    """
    return random.Random(seed)


def configure_logging(level: int = logging.INFO) -> DistEnv:
    """Set up logging for a possibly-distributed run and return the env read
    from the environment (without initializing the process group).

    Under torchrun all ranks share one terminal, so every line is tagged with
    its rank -- otherwise four interleaved copies of the same message are
    indistinguishable, and it is impossible to tell "all ranks are at step
    900" from "rank 2 is 300 steps behind". Non-main ranks are also raised to
    WARNING so the per-step INFO chatter isn't printed four times; warnings
    and errors from every rank still come through, which is the half that
    matters when one GPU is the one failing.
    """
    env = env_from_environment()
    rank_tag = f"[rank{env.rank}] " if env.is_distributed else ""
    logging.basicConfig(
        level=level if env.is_main else max(level, logging.WARNING),
        format=f"%(asctime)s {rank_tag}%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    return env


def unwrap_module(module: nn.Module) -> nn.Module:
    """Strip a DistributedDataParallel wrapper, if present.

    Anything that reaches into the model's own API rather than calling
    `forward()` needs this: `state_dict()` for checkpointing (a DDP
    state_dict has every key prefixed with `module.`, which would silently
    produce checkpoints that evaluate.py cannot load), and `.eval()`/
    `.train()` toggles.
    """
    return module.module if isinstance(module, nn.parallel.DistributedDataParallel) else module


def wrap_ddp(module: nn.Module, env: DistEnv, device: torch.device) -> nn.Module:
    """Wrap in DDP when distributed, else return the module untouched."""
    if not env.is_distributed:
        return module
    device_ids = [device.index] if device.type == "cuda" else None
    return nn.parallel.DistributedDataParallel(
        module,
        device_ids=device_ids,
        output_device=device.index if device.type == "cuda" else None,
        # Every generator parameter contributes to the watermark on every
        # step, so there is nothing unused to look for -- leaving this False
        # avoids DDP's per-step graph traversal.
        find_unused_parameters=False,
        broadcast_buffers=False,
    )


def shard_size(total: int, env: DistEnv) -> int:
    """How many of `total` globally-requested items this rank should handle.

    The remainder is spread over the low ranks (rank r takes one extra if
    r < total % world_size), so the shards differ by at most one and their
    sum is exactly `total` -- no rank silently doing a whole extra batch,
    and no items dropped.

    Used for eval batch counts (`n_eval_batches`, `n_curve_batches`), which
    are documented as *global* totals so that the same config evaluates the
    same amount of audio no matter how many GPUs it runs on.
    """
    if not env.is_distributed:
        return total
    base, remainder = divmod(total, env.world_size)
    return base + (1 if env.rank < remainder else 0)


def all_reduce_mean(values: tp.Dict[str, float], env: DistEnv, device: torch.device) -> tp.Dict[str, float]:
    """Average each scalar across ranks, in one collective for the whole dict.

    Exact only when every rank contributed the same number of examples, which
    is why every dataloader in this project keeps `drop_last=True`.
    """
    if not env.is_distributed or not values:
        return dict(values)
    keys = sorted(values)
    tensor = torch.tensor([float(values[k]) for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= env.world_size
    return {k: v for k, v in zip(keys, tensor.tolist())}


def all_reduce_max(values: tp.Dict[str, float], env: DistEnv, device: torch.device) -> tp.Dict[str, float]:
    """Max of each scalar across ranks -- the right reduction for peak-memory
    numbers, where the useful figure is the worst GPU (that is the one that
    OOMs), not the average."""
    if not env.is_distributed or not values:
        return dict(values)
    keys = sorted(values)
    tensor = torch.tensor([float(values[k]) for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return {k: v for k, v in zip(keys, tensor.tolist())}


def all_gather_scores(scores: torch.Tensor, env: DistEnv) -> torch.Tensor:
    """Concatenate a 1-D CPU score tensor from every rank, in rank order.

    Ranks may contribute different lengths (a rank can hold one batch fewer,
    see `shard_size`), so this goes through `all_gather_object` rather than
    the fixed-shape `all_gather`. These are a few thousand floats per eval,
    once per attack -- not a hot path.

    This is what makes TPR@FPR / the confusion matrix correct under DDP: the
    detection threshold is a quantile of the *global* negative-score
    distribution, and cannot be recovered from per-rank summaries.
    """
    if not env.is_distributed:
        return scores
    gathered: tp.List[tp.Optional[torch.Tensor]] = [None] * env.world_size
    dist.all_gather_object(gathered, scores.detach().cpu())
    return torch.cat([tp.cast(torch.Tensor, g) for g in gathered])


def all_gather_values(values: tp.List[float], env: DistEnv) -> tp.List[float]:
    """Concatenate a per-rank list of scalars (e.g. per-batch bit accuracies,
    or per-batch PESQ values with some batches skipped) into one list.

    Kept separate from `all_reduce_mean` because these lists can legitimately
    have different lengths per rank -- pesq_score raises on batches with no
    detectable speech and those batches are dropped, so a plain mean-of-means
    would weight ranks unequally. Concatenating first and averaging once
    weights every surviving batch the same.
    """
    if not env.is_distributed:
        return list(values)
    gathered: tp.List[tp.Optional[tp.List[float]]] = [None] * env.world_size
    dist.all_gather_object(gathered, list(values))
    return [v for part in gathered for v in tp.cast(tp.List[float], part)]


def gather_objects(obj: tp.Any, env: DistEnv) -> tp.List[tp.Any]:
    """All-gather an arbitrary picklable object, one per rank, in rank order."""
    if not env.is_distributed:
        return [obj]
    gathered: tp.List[tp.Any] = [None] * env.world_size
    dist.all_gather_object(gathered, obj)
    return gathered
