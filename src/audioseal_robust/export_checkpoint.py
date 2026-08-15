# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Repackage a train.py checkpoint into one `AudioSeal.load_generator` can open.

`train.py` writes `torch.save({"model": ..., "xp.cfg": cfg}, ...)`, which is the
right thing for *our* eval loop and the wrong thing to publish. Two properties of
that file make it unusable outside this repo:

1. **`xp.cfg` drags the repo into the pickle.** `cfg` is an OmegaConf
   `DictConfig` built from `OmegaConf.structured(TrainConfig)`, so its structured
   type is pickled as a `GLOBAL` reference to `audioseal_robust.config.TrainConfig`
   (and `GeneratorConfig`, `AttackConfig`, ... 14 of them). Anyone who runs
   `torch.load(...)` without this repo importable gets
   `ModuleNotFoundError: No module named 'audioseal_robust'` -- before they ever
   reach the weights. Worse, `AudioSeal.parse_model` *does* pick `xp.cfg` up when
   present and hands it to `parse_config`, which asserts `"seanet" in config` --
   and a `TrainConfig` has no `seanet`, so even with the repo installed the
   stock loader refuses the file.
2. **The conv keys are in the naming only half the world can read.** builder.py
   picks Moshi's SEANet (an extra `inner_conv` level) on Python >=3.10 and
   Audiocraft's flat naming below it. Upstream publishes the *flat* naming
   because that is the one both builds can consume -- `convert_state_dict_for_scriptable_model`
   up-converts flat -> `inner_conv` on new interpreters and no-ops on old ones.
   A checkpoint saved as `inner_conv` is stuck: that is the one direction
   upstream's loader cannot repair.

So this module rewrites both. The result is a plain `{"model", "xp.cfg"}` file
whose `xp.cfg` is the `audioseal_wm_16bits` architecture as plain dicts/lists,
and whose weights use the flat naming -- i.e. exactly the shape of upstream's own
`generator_base.pth`. Loading it needs nothing but `pip install audioseal`:

    from audioseal import AudioSeal
    generator = AudioSeal.load_generator("generator.pth")

The full training config is not lost, just demoted out of the load path: it is
converted to plain containers and stored under `"audioseal_robust"`, a key the
stock loader ignores.

Usage:

    PYTHONPATH=src python -m audioseal_robust.export_checkpoint \\
        checkpoints/.../generator_epoch18.pth dist/generator.pth
"""

import argparse
import json
import logging
import subprocess
import sys
import typing as tp
from pathlib import Path

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from audioseal import AudioSeal
from audioseal.builder import AudioSealWMConfig, create_generator
from audioseal.libs.moshi.utils.compile import no_compile
from audioseal.loader import (
    align_state_dict_to_model,
    flatten_scriptable_state_dict,
    load_local_model_config,
)
from audioseal.loader import load_state_dict as audioseal_load_state_dict

logger = logging.getLogger(__name__)

# Keys `AudioSeal.parse_config` actually reads off `xp.cfg` for a generator
# (`fields(AudioSealWMConfig)` is nbits/seanet/decoder/normalizer; everything
# else it drops). `model_type` is carried purely so the file is readable by a
# human poking at it with torch.load.
_EXPORTED_CONFIG_KEYS = ("model_type", "nbits", "seanet", "decoder", "normalizer")

_PLAIN_TYPES = (str, int, float, bool, type(None))


def _to_plain(obj: tp.Any) -> tp.Any:
    """Strip every OmegaConf/dataclass wrapper down to builtin containers.

    This is the whole point of the exercise: `torch.save` pickles types by
    reference, so anything that is not a builtin becomes an import the consumer
    must be able to satisfy.
    """
    if isinstance(obj, (DictConfig, ListConfig)):
        try:
            obj = OmegaConf.to_container(obj, resolve=True, enum_to_str=True)
        except OmegaConfBaseException as exc:
            # A dangling `${...}` in the *training* config is not a reason to
            # refuse to publish weights -- keep the interpolation verbatim.
            logger.warning("could not resolve interpolations (%s); storing them raw", exc)
            obj = OmegaConf.to_container(obj, resolve=False, enum_to_str=True)
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _assert_plain(obj: tp.Any, path: str = "xp.cfg") -> None:
    """Fail loudly here rather than in a stranger's `torch.load`."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"{path}: non-str key {k!r} of type {type(k).__name__}")
            _assert_plain(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_plain(v, f"{path}[{i}]")
    elif not isinstance(obj, _PLAIN_TYPES):
        raise TypeError(
            f"{path}: {type(obj).__module__}.{type(obj).__name__} is not a builtin; "
            "it would be pickled as an import the consumer has to satisfy"
        )


def _architecture_config(card: str, nbits: tp.Optional[int]) -> tp.Dict[str, tp.Any]:
    """The `xp.cfg` the exported file carries: the stock generator card, minus
    its `checkpoint:` download URL (which points at the *base* weights and would
    be actively misleading inside a derived artifact)."""
    raw = load_local_model_config(card)
    if raw is None:
        raise FileNotFoundError(
            f"No model card {card!r} under src/audioseal/cards/ -- "
            "the exported config has to describe a real architecture"
        )
    full = _to_plain(raw)
    config = {k: full[k] for k in _EXPORTED_CONFIG_KEYS if k in full}
    if nbits is not None:
        config["nbits"] = nbits
    if "seanet" not in config:
        raise ValueError(f"card {card!r} has no `seanet` block; parse_config would reject it")
    if "nbits" not in config:
        raise ValueError(f"card {card!r} has no `nbits`; pass --nbits explicitly")
    return config


def _git_commit() -> tp.Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _state_dict_of(checkpoint: tp.Mapping[str, tp.Any]) -> tp.Dict[str, torch.Tensor]:
    # Same precedence AudioSeal.parse_model uses.
    if "best_state" in checkpoint:
        return dict(checkpoint["best_state"]["model"])
    if "model" in checkpoint:
        return dict(checkpoint["model"])
    raise KeyError("checkpoint has neither a 'model' nor a 'best_state' key")


def export_generator(
    checkpoint: Path,
    output: Path,
    card: str = "audioseal_wm_16bits",
    nbits: tp.Optional[int] = None,
    verify: bool = True,
) -> tp.Dict[str, tp.Any]:
    """Write `output`, a portable copy of the generator weights in `checkpoint`."""
    checkpoint, output = Path(checkpoint), Path(output)
    logger.info("reading training checkpoint %s", checkpoint)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)

    state_dict = _state_dict_of(raw)
    flat = flatten_scriptable_state_dict(state_dict)
    n_renamed = sum(1 for k in state_dict if ".inner_conv." in k)
    if len(flat) != len(state_dict):
        raise ValueError("flattening `.inner_conv.` collided two distinct keys")
    logger.info("flattened %d/%d conv keys to upstream naming", n_renamed, len(state_dict))

    training_cfg = _to_plain(raw.get("xp.cfg", {}))
    if nbits is None:
        nbits = training_cfg.get("nbits") if isinstance(training_cfg, dict) else None

    config = _architecture_config(card, nbits)
    _assert_plain(config)

    provenance = _to_plain(
        {
            "architecture_card": card,
            "source_checkpoint": checkpoint.name,
            "exported_by": f"{__name__} (martysai/psiml11-audio-with-diffusion)",
            "git_commit": _git_commit(),
            "torch_version": torch.__version__,
            "training_config": training_cfg,
        }
    )
    _assert_plain(provenance, "audioseal_robust")
    # The strongest portability check available without a clean interpreter: if
    # it round-trips through JSON it is builtins the whole way down.
    json.dumps({"xp.cfg": config, "audioseal_robust": provenance})

    exported = {"model": flat, "xp.cfg": config, "audioseal_robust": provenance}
    output.parent.mkdir(parents=True, exist_ok=True)

    # Write to a sibling and only rename on a clean verify, so a failed export
    # can never leave a half-valid file sitting where someone might publish it.
    staged = output.with_name(output.name + ".staged")
    torch.save(exported, staged)
    try:
        if verify:
            verify_export(
                checkpoint_state_dict=state_dict,
                output=staged,
                config=config,
            )
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    staged.replace(output)
    logger.info("wrote %s (%.1f MiB)", output, output.stat().st_size / 2**20)
    return exported


def verify_export(
    checkpoint_state_dict: tp.Mapping[str, torch.Tensor],
    output: Path,
    config: tp.Mapping[str, tp.Any],
) -> None:
    """Prove the exported file rebuilds the same generator, weight for weight.

    The reference is built from `config` -- the very architecture block the
    export claims -- and then loaded with the *source* state dict. So this does
    not just check that the file is self-consistent, it checks that the config
    we are shipping actually describes the weights we are shipping.

    `AudioSeal.load_generator` is then called on the real artifact with no
    arguments, exercising the exact code path a downstream user hits: `nbits`
    off `xp.cfg`, and the flat -> `inner_conv` up-conversion this export exists
    to hand it.
    """
    logger.info("verifying %s through AudioSeal.load_generator", output)
    reloaded = AudioSeal.load_generator(str(output))
    reloaded.eval()

    reference = create_generator(AudioSeal.parse_config(dict(config), AudioSealWMConfig))
    audioseal_load_state_dict(
        reference, align_state_dict_to_model(reference, dict(checkpoint_state_dict))
    )
    reference.eval()

    ref_state, got_state = reference.state_dict(), reloaded.state_dict()
    if set(ref_state) != set(got_state):
        missing = sorted(set(ref_state) - set(got_state))
        extra = sorted(set(got_state) - set(ref_state))
        raise AssertionError(f"key mismatch after reload: missing={missing[:5]} extra={extra[:5]}")
    for key, ref in ref_state.items():
        if not torch.equal(ref, got_state[key]):
            raise AssertionError(f"tensor {key} changed across export")

    # A forward pass on top of the tensor comparison: it is the only thing that
    # covers the msg_processor/encoder/decoder *wiring* rebuilt from `xp.cfg`,
    # as opposed to the tensors themselves.
    #
    # Under no_compile() because Moshi's SEANet forward is @torch_compile_lazy,
    # and torch.compile's C++ backend needs a toolchain that a box doing nothing
    # but repackaging a checkpoint has no reason to have. It also keeps both
    # generators on identical kernels, so torch.equal is the right comparison.
    nbits = int(config["nbits"])
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(2, 1, 16000, generator=generator)
    message = torch.randint(0, 2, (2, nbits), generator=generator)
    with torch.no_grad(), no_compile():
        if not torch.equal(
            reference.get_watermark(x, 16000, message=message),
            reloaded.get_watermark(x, 16000, message=message),
        ):
            raise AssertionError("watermark output differs after export")
    logger.info("verified: %d tensors identical, watermark bit-exact", len(ref_state))


def main(argv: tp.Optional[tp.List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", type=Path, help="a .pth written by train.py")
    parser.add_argument("output", type=Path, help="where to write the portable .pth")
    parser.add_argument(
        "--card",
        default="audioseal_wm_16bits",
        help="model card under src/audioseal/cards/ describing the architecture",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=None,
        help="override the payload size (defaults to the training config's, then the card's)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip reloading the result through AudioSeal.load_generator",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    export_generator(
        checkpoint=args.checkpoint,
        output=args.output,
        card=args.card,
        nbits=args.nbits,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    sys.exit(main())
