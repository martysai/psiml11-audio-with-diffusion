# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for gradient checkpointing through a *frozen* module.

`audioldm_train.utilities.diffusion_util.CheckpointFunction.backward` calls
`torch.autograd.grad` over the tensors it was handed in forward. Passing it a
parameter with `requires_grad=False` makes autograd raise

    RuntimeError: One of the differentiated Tensors does not require grad

which is exactly what happens when AudioLDM is used as a frozen but
differentiable module: its parameters are frozen, gradients only need to flow
back through it to the input, and every checkpointed block
(openaimodel.ResBlock / AttentionBlock, attention.BasicTransformerBlock) hits
this on the first backward pass.

`checkpoint()` therefore forwards only the parameters that require grad, and
these tests pin that down: no exception, and gradients identical to the
`flag=False` path that skips checkpointing entirely.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")
pytest.importorskip("einops")

_DIFFUSION_UTIL_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "audioldm_train"
    / "utilities"
    / "diffusion_util.py"
)


def _load_diffusion_util():
    """Load the real diffusion_util source without importing audioldm_train.

    `audioldm_train/utilities/__init__.py` re-exports tools/data/model_util,
    which pull in the whole AudioLDM training stack (hifigan, PIL, librosa,
    ...) from requirements-audioldm.txt. `checkpoint()` and
    `CheckpointFunction` need none of it, so the package and the single symbol
    diffusion_util imports from it are stubbed and the file is loaded straight
    from disk -- the code under test is still the committed source, not a copy.
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


class TinyBlock(torch.nn.Module):
    """Stand-in for a checkpointed AudioLDM block: two weighted layers."""

    def __init__(self, dim=4):
        super().__init__()
        self.first = torch.nn.Linear(dim, dim)
        self.second = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.second(torch.tanh(self.first(x)))


def _make_block(dim=4, seed=0):
    torch.manual_seed(seed)
    return TinyBlock(dim)


def _run(block, x, flag):
    """Mirror how the AudioLDM blocks call checkpoint()."""
    return checkpoint(block, (x,), block.parameters(), flag)


def _backward(block, x, flag):
    """Return (input grad, {param name: param grad}) for one backward pass."""
    for param in block.parameters():
        param.grad = None
    out = _run(block, x, flag)
    out.sum().backward()
    grads = {name: param.grad for name, param in block.named_parameters()}
    return x.grad, grads


def _freeze(block, names=()):
    """Freeze every parameter, or only the named ones when `names` is given."""
    for name, param in block.named_parameters():
        if not names or name in names:
            param.requires_grad_(False)


def test_frozen_module_backward_does_not_raise():
    # The reported failure: every parameter frozen, gradient-requiring input.
    block = _make_block()
    _freeze(block)
    x = torch.randn(2, 4, requires_grad=True)

    out = _run(block, x, True)
    out.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_frozen_module_input_grad_matches_uncheckpointed():
    # Checkpointing must not change the gradient that reaches the input.
    block = _make_block()
    _freeze(block)
    inputs = torch.randn(2, 4)

    x_ckpt = inputs.clone().requires_grad_(True)
    x_plain = inputs.clone().requires_grad_(True)

    grad_ckpt, _ = _backward(block, x_ckpt, True)
    grad_plain, _ = _backward(block, x_plain, False)

    torch.testing.assert_close(grad_ckpt, grad_plain)


def test_mixed_frozen_and_trainable_params_backward_does_not_raise():
    # Partially frozen module: `first` frozen, `second` still trained.
    block = _make_block()
    _freeze(block, names=("first.weight", "first.bias"))
    x = torch.randn(2, 4, requires_grad=True)

    out = _run(block, x, True)
    out.sum().backward()

    assert x.grad is not None
    assert block.second.weight.grad is not None
    assert block.second.bias.grad is not None


def test_trainable_param_grads_match_uncheckpointed():
    # Gradients still land on the trainable parameters, with the same values
    # the non-checkpointed path produces.
    inputs = torch.randn(2, 4)
    frozen = ("first.weight", "first.bias")

    block_ckpt = _make_block()
    _freeze(block_ckpt, names=frozen)
    x_ckpt = inputs.clone().requires_grad_(True)
    input_grad_ckpt, grads_ckpt = _backward(block_ckpt, x_ckpt, True)

    block_plain = _make_block()
    _freeze(block_plain, names=frozen)
    x_plain = inputs.clone().requires_grad_(True)
    input_grad_plain, grads_plain = _backward(block_plain, x_plain, False)

    torch.testing.assert_close(input_grad_ckpt, input_grad_plain)
    for name in ("second.weight", "second.bias"):
        assert grads_ckpt[name] is not None
        torch.testing.assert_close(grads_ckpt[name], grads_plain[name])


def test_frozen_params_receive_no_grads():
    # Frozen parameters are constants during recomputation: no gradient, and
    # no silent difference from the non-checkpointed path either.
    block = _make_block()
    _freeze(block, names=("first.weight", "first.bias"))
    x = torch.randn(2, 4, requires_grad=True)

    _, grads = _backward(block, x, True)

    assert grads["first.weight"] is None
    assert grads["first.bias"] is None


def test_fully_trainable_module_grads_match_uncheckpointed():
    # The ordinary training case the upstream code was written for must keep
    # working: nothing is filtered out when every parameter requires grad.
    inputs = torch.randn(2, 4)

    block_ckpt = _make_block()
    x_ckpt = inputs.clone().requires_grad_(True)
    input_grad_ckpt, grads_ckpt = _backward(block_ckpt, x_ckpt, True)

    block_plain = _make_block()
    x_plain = inputs.clone().requires_grad_(True)
    input_grad_plain, grads_plain = _backward(block_plain, x_plain, False)

    torch.testing.assert_close(input_grad_ckpt, input_grad_plain)
    for name in grads_plain:
        assert grads_ckpt[name] is not None
        torch.testing.assert_close(grads_ckpt[name], grads_plain[name])


def test_checkpoint_accepts_a_parameter_generator():
    # The call sites pass self.parameters(), a generator -- it must not be
    # consumed before its length is needed.
    block = _make_block()
    _freeze(block)
    x = torch.randn(2, 4, requires_grad=True)

    out = checkpoint(block, (x,), block.parameters(), True)
    out.sum().backward()

    assert x.grad is not None
