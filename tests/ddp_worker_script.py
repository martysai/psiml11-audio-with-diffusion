# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Worker launched under `torch.distributed.run` by tests/test_distributed.py.

Deliberately NOT named test_*.py: pytest must not collect it, because it only
does anything meaningful when it is one of several ranks started by torchrun.
Each mode below writes its result to `--out-dir/rank<N>.pt` and the parent
test process does the asserting.

This runs the real thing -- real torchrun, real process group, real DDP
allreduce (over gloo on CPU, which is the only part that differs from a
4xA100 NCCL run) -- because the failure this is guarding against, gradients
silently not being synchronized, is invisible to any single-process test:
the loss still goes down on every rank.
"""

import argparse
import os

import torch

from audioseal.builder import (
    AudioSealWMConfig,
    DecoderConfig,
    SEANetConfig,
    create_generator,
)
from audioseal_robust.distributed import (
    all_gather_scores,
    all_gather_values,
    all_reduce_max,
    all_reduce_mean,
    cleanup_distributed,
    init_distributed,
    shard_size,
    unwrap_module,
    wrap_ddp,
)
from audioseal_robust.train import WatermarkEmbedder, embed_watermark, unwrap_generator

NBITS = 4


def tiny_generator():
    seanet = SEANetConfig(
        channels=1,
        dimension=16,
        n_filters=4,
        n_residual_layers=1,
        ratios=[2, 2],
        activation="ELU",
        activation_params={"alpha": 1.0},
        norm="none",
        norm_params={},
        kernel_size=3,
        last_kernel_size=3,
        residual_kernel_size=3,
        dilation_base=2,
        causal=False,
        pad_mode="constant",
        true_skip=True,
        compress=2,
        lstm=1,
        disable_norm_outer_blocks=0,
    )
    cfg = AudioSealWMConfig(
        nbits=NBITS,
        seanet=seanet,
        decoder=DecoderConfig(final_activation=None, final_activation_params=None, trim_right_ratio=1.0),
        normalizer=False,
    )
    return create_generator(cfg)


def _train_a_few_steps(embedder, generator, rank: int, call_through_ddp: bool):
    """Three optimizer steps on rank-dependent data.

    The data differs per rank on purpose: that makes the per-rank gradients
    differ, so the parameters can only stay equal across ranks if an allreduce
    actually happened. If they end up equal here, gradients were synchronized.
    """
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-2)
    torch.manual_seed(100 + rank)
    for _ in range(3):
        x = torch.randn(2, 1, 2000)
        message = torch.randint(0, 2, (2, NBITS))
        if call_through_ddp:
            x_wm = embed_watermark(embedder, x, message, 24.0, 36.0)
        else:
            # The bug this guards against: reaching past DDP to the raw
            # module's own method, which never triggers DDP's hooks.
            x_wm = embed_watermark(unwrap_generator(embedder), x, message, 24.0, 36.0)
        loss = x_wm.pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def mode_ddp_sync(env, out_path, call_through_ddp: bool):
    torch.manual_seed(0)  # identical init on every rank
    generator = tiny_generator()
    embedder = wrap_ddp(WatermarkEmbedder(generator), env, torch.device("cpu"))
    _train_a_few_steps(embedder, generator, env.rank, call_through_ddp)

    unwrapped = unwrap_generator(embedder)
    torch.save(
        {
            "state_dict": unwrapped.state_dict(),
            "is_raw_generator": type(unwrapped).__name__,
            "unwrap_module_type": type(unwrap_module(embedder)).__name__,
        },
        out_path,
    )


def mode_collectives(env, out_path):
    """Exercise every reduction helper with values that make a wrong
    reduction obvious (rank-dependent, different lengths per rank)."""
    rank, world = env.rank, env.world_size

    # Different lengths per rank: this is what rules out the fixed-shape
    # all_gather that would silently truncate.
    scores = torch.arange(rank + 1, dtype=torch.float32) + 10 * rank
    gathered_scores = all_gather_scores(scores, env)

    values = [float(rank)] * (rank + 1)
    gathered_values = all_gather_values(values, env)

    means = all_reduce_mean({"a": float(rank), "b": 1.0}, env, torch.device("cpu"))
    maxes = all_reduce_max({"peak": float(rank)}, env, torch.device("cpu"))

    torch.save(
        {
            "world_size": world,
            "gathered_scores": gathered_scores,
            "gathered_values": gathered_values,
            "means": means,
            "maxes": maxes,
            "shard_of_20": shard_size(20, env),
            "shard_of_7": shard_size(7, env),
        },
        out_path,
    )


def mode_idle_rank(env, out_path):
    """Fewer global batches than ranks: the high ranks get a zero-size shard.

    Reproduces the shape of what evaluate_attack does in that case -- an
    empty local contribution that must still take part in every collective,
    and must NOT be reported as a failure. Getting this wrong turns an entire
    evaluation into "every attack skipped" with exit code 0.
    """
    n_batches_global = 2
    local = shard_size(n_batches_global, env)

    scores = torch.full((3 * local,), float(env.rank)) if local else torch.empty(0)
    bit_accs = [1.0] * local

    gathered_scores = all_gather_scores(scores, env)
    gathered_values = all_gather_values(bit_accs, env)

    torch.save(
        {
            "local_shard": local,
            "gathered_scores_numel": gathered_scores.numel(),
            "gathered_values": gathered_values,
        },
        out_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True, choices=["ddp_sync", "ddp_bypass", "collectives", "idle_rank"]
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    env, _ = init_distributed("cpu")
    try:
        out_path = os.path.join(args.out_dir, f"rank{env.rank}.pt")
        if args.mode == "ddp_sync":
            mode_ddp_sync(env, out_path, call_through_ddp=True)
        elif args.mode == "ddp_bypass":
            mode_ddp_sync(env, out_path, call_through_ddp=False)
        elif args.mode == "idle_rank":
            mode_idle_rank(env, out_path)
        else:
            mode_collectives(env, out_path)
    finally:
        cleanup_distributed(env)


if __name__ == "__main__":
    main()
