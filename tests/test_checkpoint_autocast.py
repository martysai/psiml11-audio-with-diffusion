# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for autocast state across the checkpoint boundary.

`CheckpointFunction.backward` re-runs the checkpointed function to rebuild the
activations it did not store. The autograd engine runs backward with autocast
*off*, so unless the forward's autocast state is captured and restored, that
recompute evaluates a different function than the forward did.

On CUDA this is fatal. Under train.py's bf16 autocast the saved activations are
bf16 while module parameters stay fp32, and replaying
`BasicTransformerBlock._forward`'s LayerNorm outside autocast raises

    RuntimeError: expected scalar type BFloat16 but found Float

which is exactly how the 4x A100 manifold smoke run died. On CPU the same bug
is silent -- layer_norm promotes instead of raising -- and merely produces
gradients that do not match the uncheckpointed path, so these tests assert the
invariant directly (autocast state and dtypes agree between the two calls)
rather than relying on a device-specific exception.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")
pytest.importorskip("einops")

import torch.nn as nn  # noqa: E402

_DIFFUSION_UTIL_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "audioldm_train"
    / "utilities"
    / "diffusion_util.py"
)


def _load_diffusion_util():
    """Load the real diffusion_util source without importing audioldm_train.

    Same stubbing approach as test_diffusion_checkpoint.py -- the package
    __init__ drags in the whole AudioLDM training stack, none of which
    `checkpoint()` needs.
    """
    name = "audioldm_train.utilities.diffusion_util"
    stub_names = (
        "audioldm_train",
        "audioldm_train.utilities",
        "audioldm_train.utilities.model_util",
        name,
    )
    saved = {key: sys.modules.get(key) for key in stub_names}

    package = types.ModuleType("audioldm_train")
    package.__path__ = []
    utilities = types.ModuleType("audioldm_train.utilities")
    utilities.__path__ = []
    model_util = types.ModuleType("audioldm_train.utilities.model_util")
    model_util.instantiate_from_config = None

    sys.modules["audioldm_train"] = package
    sys.modules["audioldm_train.utilities"] = utilities
    sys.modules["audioldm_train.utilities.model_util"] = model_util
    try:
        spec = importlib.util.spec_from_file_location(name, _DIFFUSION_UTIL_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


_diffusion_util = _load_diffusion_util()
checkpoint = _diffusion_util.checkpoint


class _Block(nn.Module):
    """Shaped like attention.BasicTransformerBlock._forward, which is the
    block that actually failed: LayerNorm with fp32 parameters applied to an
    autocast activation, then a Linear."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = nn.Linear(dim, dim)

    def _forward(self, x):
        return self.attn1(self.norm1(x)) + x


def _record_calls(frozen: bool):
    """Run a checkpointed block under bf16 autocast and record the autocast
    state seen on each invocation: index 0 is the forward, index 1 the
    backward recompute."""
    torch.manual_seed(0)
    block = _Block()
    for p in block.parameters():
        p.requires_grad_(not frozen)

    seen = []

    def fn(x):
        seen.append(
            {
                "autocast": torch.is_autocast_enabled("cpu"),
                "norm_out_dtype": block.norm1(x).dtype,
            }
        )
        return block._forward(x)

    x = torch.randn(4, 16, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = checkpoint(fn, (x,), block.parameters(), True)
        loss = out.float().pow(2).mean()
    loss.backward()
    return seen


@pytest.mark.parametrize("frozen", [True, False])
def test_recompute_runs_under_the_forward_autocast_state(frozen):
    seen = _record_calls(frozen)
    assert len(seen) == 2, f"expected a forward and a recompute, saw {len(seen)} call(s)"
    forward, recompute = seen
    assert forward["autocast"] is True, "test setup: forward should run under autocast"
    assert recompute["autocast"] == forward["autocast"], (
        "backward recompute ran with autocast disabled; on CUDA this raises "
        "'expected scalar type BFloat16 but found Float'"
    )


@pytest.mark.parametrize("frozen", [True, False])
def test_recompute_dtypes_match_the_forward(frozen):
    forward, recompute = _record_calls(frozen)
    assert recompute["norm_out_dtype"] == forward["norm_out_dtype"]


def _input_grad(flag: bool):
    torch.manual_seed(0)
    block = _Block()
    for p in block.parameters():
        p.requires_grad_(False)
    x = torch.randn(4, 16, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = checkpoint(block._forward, (x,), block.parameters(), flag)
        loss = out.float().pow(2).mean()
    loss.backward()
    return x.grad.detach().clone()


def test_autocast_input_grad_matches_uncheckpointed():
    """The point of checkpointing is to trade compute for memory, not to
    change the answer: recomputing under the forward's own autocast state
    reproduces the uncheckpointed gradient exactly."""
    torch.testing.assert_close(_input_grad(True), _input_grad(False), rtol=0, atol=0)
