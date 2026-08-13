# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Device resolution: pick a real accelerator when one is available instead
of silently training on CPU, but never hard-fail if the config asks for a
backend that isn't actually there on this machine.

Under `torchrun` this also has to pin each rank to *its own* GPU: a bare
`torch.device("cuda")` means `cuda:0` to every process, so all N ranks would
load their model replica onto card 0 and leave the other three idle (and
usually OOM). Callers pass `local_rank` for that -- in practice via
`distributed.init_distributed`, which is the only place that knows it."""

import logging
import typing as tp

import torch

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto", local_rank: tp.Optional[int] = None) -> torch.device:
    """
    Args:
        requested: "auto" (pick the best available: cuda > mps > cpu),
            or an explicit "cuda" / "mps" / "cpu" (falls back to cpu with a
            warning if the requested backend isn't available here).
        local_rank: this process's GPU index within its node, from torchrun's
            LOCAL_RANK. When the resolved device is CUDA, the index is pinned
            to it (`cuda:{local_rank}`) so each rank gets its own card.
            Ignored for non-CUDA devices, and for a `requested` string that
            already names an explicit index (e.g. "cuda:2"), which is treated
            as a deliberate override.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.info("device=auto resolved to %s", device)
        return _pin_local_rank(device, local_rank)

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("device=cuda requested but CUDA is not available here, falling back to cpu")
        return torch.device("cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        logger.warning("device=mps requested but MPS is not available here, falling back to cpu")
        return torch.device("cpu")
    return _pin_local_rank(device, local_rank)


def _pin_local_rank(device: torch.device, local_rank: tp.Optional[int]) -> torch.device:
    if local_rank is None or device.type != "cuda" or device.index is not None:
        return device
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but this node only has {torch.cuda.device_count()} visible CUDA "
            "device(s) -- check --nproc_per_node against CUDA_VISIBLE_DEVICES"
        )
    return torch.device("cuda", local_rank)
