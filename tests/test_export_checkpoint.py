# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for `audioseal_robust.export_checkpoint`, i.e. the step that turns a
train.py checkpoint into something a stranger with `pip install audioseal` can
open.

The two failure modes worth pinning down are the ones that actually shipped:

1. `xp.cfg` pickled `audioseal_robust.config.TrainConfig` by reference, so
   `torch.load` raised `ModuleNotFoundError: No module named 'audioseal_robust'`
   off-repo -- and `AudioSeal.parse_config` rejected the file anyway, because a
   `TrainConfig` has no `seanet` block.
2. The conv keys carried the Moshi `inner_conv` naming, which upstream's
   flat -> `inner_conv` conversion cannot undo, so the checkpoint only loaded
   on the same side of the Python 3.10 split it was written on.

`test_exported_checkpoint_pickles_no_project_types` is the direct regression for
(1): it unpickles the exported file in a subprocess that cannot import this repo
at all, which is exactly the situation a downstream user is in.
"""

import pickletools
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch

from audioseal import AudioSeal
from audioseal.loader import (
    align_state_dict_to_model,
    convert_state_dict_for_scriptable_model,
    flatten_scriptable_state_dict,
)
from audioseal_robust.export_checkpoint import export_generator
from audioseal_robust.model_init import build_untrained_generator

NBITS = 16


@pytest.fixture(scope="module")
def trained_generator():
    """Stand-in for a fine-tuned generator: the real architecture with weights
    perturbed off their init, so an export that silently dropped or reordered
    tensors could not pass by coincidence."""
    generator = build_untrained_generator(nbits=NBITS)
    torch.manual_seed(1234)
    with torch.no_grad():
        for param in generator.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    generator.eval()
    return generator


@pytest.fixture(scope="module")
def training_checkpoint(trained_generator, tmp_path_factory):
    """A byte-for-byte stand-in for what train.py writes: the state_dict plus a
    structured OmegaConf config, which is the thing that poisons the pickle."""
    from audioseal_robust.config import TrainConfig
    from omegaconf import OmegaConf

    path = tmp_path_factory.mktemp("train") / "generator_epoch18.pth"
    cfg = OmegaConf.structured(TrainConfig)
    cfg.nbits = NBITS
    torch.save({"model": trained_generator.state_dict(), "xp.cfg": cfg}, path)
    return path


@pytest.fixture(scope="module")
def exported_checkpoint(training_checkpoint, tmp_path_factory):
    out = tmp_path_factory.mktemp("export") / "generator.pth"
    # verify=True runs export_checkpoint's own reload-and-compare assertions.
    export_generator(training_checkpoint, out, nbits=NBITS, verify=True)
    return out


def test_training_checkpoint_is_not_loadable_by_stock_audioseal(training_checkpoint):
    """The premise: the raw train.py checkpoint cannot go on a model hub as-is."""
    with pytest.raises(AssertionError, match="seanet"):
        AudioSeal.load_generator(str(training_checkpoint), nbits=NBITS)


def test_exported_checkpoint_loads_with_plain_load_generator(
    exported_checkpoint, trained_generator
):
    """The payoff: one call, no nbits, no key surgery, no torch.load."""
    reloaded = AudioSeal.load_generator(str(exported_checkpoint))
    reloaded.eval()

    expected = trained_generator.state_dict()
    got = reloaded.state_dict()
    assert set(expected) == set(got)
    for key, ref in expected.items():
        assert torch.equal(ref, got[key]), key


def test_exported_checkpoint_watermarks_identically(
    exported_checkpoint, trained_generator
):
    reloaded = AudioSeal.load_generator(str(exported_checkpoint))
    reloaded.eval()

    generator = torch.Generator().manual_seed(7)
    x = torch.randn(2, 1, 16000, generator=generator)
    message = torch.randint(0, 2, (2, NBITS), generator=generator)
    with torch.no_grad():
        assert torch.equal(
            trained_generator.get_watermark(x, 16000, message=message),
            reloaded.get_watermark(x, 16000, message=message),
        )


def test_exported_state_dict_uses_upstream_flat_conv_naming(exported_checkpoint):
    state = torch.load(exported_checkpoint, map_location="cpu", weights_only=False)
    assert not any(".inner_conv." in k for k in state["model"]), (
        "published weights must use the flat naming: it is the only one both the "
        "Audiocraft (py<3.10) and Moshi (py>=3.10) SEANet builds can consume"
    )


def test_exported_config_is_the_architecture_not_the_training_run(exported_checkpoint):
    state = torch.load(exported_checkpoint, map_location="cpu", weights_only=False)
    config = state["xp.cfg"]
    assert isinstance(config, dict)
    # What AudioSeal.parse_config requires and reads.
    assert "seanet" in config
    assert config["nbits"] == NBITS
    assert config["seanet"]["dimension"] == 128
    assert config["seanet"]["ratios"] == [8, 5, 4, 2]
    # The base model's download URL has no business in a derived artifact.
    assert "checkpoint" not in config
    # The training config is preserved, just out of the load path.
    assert state["audioseal_robust"]["training_config"]["nbits"] == NBITS


def test_exported_checkpoint_pickles_no_project_types(exported_checkpoint):
    """Unpickle it where neither `audioseal_robust` nor `audioseal` exists.

    `-I` isolates the interpreter (no cwd on sys.path, no PYTHONPATH), so an
    import of anything from this repo is guaranteed to fail rather than silently
    resolve. This is the regression test for the ModuleNotFoundError that made
    the published checkpoint unusable.
    """
    script = (
        "import sys, torch;"
        f"ckpt = torch.load(r'{exported_checkpoint}', map_location='cpu', weights_only=False);"
        "assert set(ckpt) >= {'model', 'xp.cfg'};"
        "assert isinstance(ckpt['xp.cfg'], dict);"
        "assert 'audioseal_robust' not in sys.modules, sorted(sys.modules);"
        "assert 'omegaconf' not in sys.modules, sorted(sys.modules);"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        cwd=Path(exported_checkpoint).parent,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_exported_pickle_references_only_builtin_and_torch_globals(exported_checkpoint):
    """Belt-and-braces on the above, straight off the opcode stream: every
    GLOBAL the pickle resolves must come from the stdlib or torch.

    Protocol 2 (torch.save's default) emits GLOBAL, which carries the
    module/name inline; protocol 4 emits STACK_GLOBAL, which pops the two
    strings pushed just before it. Both are handled so this test does not
    quietly pass if the protocol ever changes.
    """
    with zipfile.ZipFile(exported_checkpoint) as archive:
        name = next(n for n in archive.namelist() if n.endswith("data.pkl"))
        with archive.open(name) as pkl:
            globals_, recent = set(), []
            for opcode, arg, _ in pickletools.genops(pkl):
                if opcode.name == "GLOBAL":
                    globals_.add(arg.split()[0])
                elif opcode.name == "STACK_GLOBAL":
                    assert len(recent) >= 2, "STACK_GLOBAL without its two operands"
                    globals_.add(recent[-2])
                elif opcode.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
                    recent.append(arg)

    offenders = sorted(
        g for g in globals_ if not g.split(".")[0] in ("torch", "collections", "builtins")
    )
    assert not offenders, f"pickle pulls in non-portable globals: {offenders}"


def test_align_state_dict_to_model_round_trips_both_directions(trained_generator):
    """`align_state_dict_to_model` must be an identity when the naming already
    matches, and the exact inverse of `flatten_scriptable_state_dict` when it
    does not -- that pairing is what lets one published file serve both builds.
    """
    native = trained_generator.state_dict()
    flat = flatten_scriptable_state_dict(native)
    # On py>=3.10 `flat` is genuinely renamed; below it the model is already
    # flat and this is a no-op. Either way no two keys may collide.
    assert len(flat) == len(native)

    realigned = align_state_dict_to_model(trained_generator, flat)
    assert list(realigned) == list(native)
    for key in native:
        assert torch.equal(native[key], realigned[key])

    # Already in the model's naming: must be left completely alone.
    assert align_state_dict_to_model(trained_generator, native) is native


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="the Moshi SEANet (and so `inner_conv` naming) is only built on py>=3.10",
)
def test_flatten_is_the_inverse_of_upstreams_conversion(trained_generator):
    """The property the whole export rests on: upstream's flat -> `inner_conv`
    conversion undoes our flattening exactly, keys and order alike."""
    native = trained_generator.state_dict()
    assert any(".inner_conv." in k for k in native)

    flat = flatten_scriptable_state_dict(native)
    assert len(flat) == len(native), "flattening collided two distinct keys"
    assert not any(".inner_conv." in k for k in flat)
    assert list(convert_state_dict_for_scriptable_model(flat)) == list(native)
