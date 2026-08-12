# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Device resolution: pick a real accelerator when one is available instead
of silently training on CPU, but never hard-fail if the config asks for a
backend that isn't actually there on this machine."""

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> torch.device:
    """
    Args:
        requested: "auto" (pick the best available: cuda > mps > cpu),
            or an explicit "cuda" / "mps" / "cpu" (falls back to cpu with a
            warning if the requested backend isn't available here).
    """
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.info("device=auto resolved to %s", device)
        return device

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("device=cuda requested but CUDA is not available here, falling back to cpu")
        return torch.device("cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        logger.warning("device=mps requested but MPS is not available here, falling back to cpu")
        return torch.device("cpu")
    return device
