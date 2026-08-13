"""Fail an AzureML job in seconds instead of hours.

Every check here is for a failure mode this project has that is either silent
or only surfaces late:

* **CPU fallback.** ``device.py:36-40`` logs a warning and returns CPU when
  ``device=cuda`` is asked for but unavailable. ``run_diffusion_swap.sh``
  passes ``device=cuda`` unconditionally, so a mis-sized compute target
  produces a job that runs, logs, and checkpoints normally -- just ~100x too
  slow, on paid A100 time.
* **Mount layout.** ``attacks.py:418-426`` requires the AudioLDM checkpoint at
  ``<weights_root>/data/checkpoints/<file>``, and the model's own YAML then
  resolves its VAE/CLAP/vocoder siblings by *relative* path after a chdir.
  A wrong mount level fails only when DiffEraseAttack is first constructed,
  which for the ``sgmse`` direction is after training has fully finished.
* **Empty data dirs.** A dataloader over an empty directory raises deep inside
  a worker, well after the model is built.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Relative paths inside audioldm_original.yaml, resolved by audioldm_train
# after attacks.py chdirs to weights_root. Kept here rather than parsed out of
# the YAML so this file has no pyyaml/omegaconf import cost.
AUDIOLDM_SIBLINGS = (
    "vae_mel_16k_64bins.ckpt",  # first_stage_config.reload_from_ckpt (line 56)
    "clap_htsat_tiny.pt",  # cond_stage_config...pretrained_path (line 138)
    "hifigan_16k_64bins.ckpt",  # get_vocoder()
    "hifigan_16k_64bins.json",
)


def fail(msg: str) -> None:
    print(f"PREFLIGHT FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def check_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        fail(
            "torch.cuda.is_available() is False, but run_diffusion_swap.sh passes "
            "device=cuda. device.py:36-40 would silently fall back to CPU and the "
            "job would burn its whole budget at roughly 1% of the expected speed. "
            "Check the compute target actually has a GPU and that the image's "
            "torch build matches the host driver.\n"
            f"  torch={torch.__version__} torch.version.cuda={torch.version.cuda}"
        )
    n = torch.cuda.device_count()
    print(f"CUDA OK: {n} device(s), torch={torch.__version__}, cuda={torch.version.cuda}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {props.name}  {props.total_memory / 1024**3:.1f} GiB  sm_{props.major}{props.minor}")


def check_audio_split(root: Path, split: str) -> None:
    d = root / split
    if not d.is_dir():
        fail(f"expected LibriSpeech split at {d} -- got a data root of {root} with children {_children(root)}")
    # data.py:36 rglobs for _AUDIO_EXTENSIONS; stop at the first hit so this
    # stays fast over 28k files on a blobfuse mount.
    for ext in (".flac", ".wav"):
        if next(d.rglob(f"*{ext}"), None) is not None:
            print(f"data OK: {split} -> {d}")
            return
    fail(f"no .flac/.wav anywhere under {d} -- data.py:WavDirDataset would build an empty dataset")


def check_audioldm(ckpt: Path) -> None:
    if not ckpt.is_file():
        fail(f"AudioLDM checkpoint not found: {ckpt}")
    # The exact assertion attacks.py:418 makes, restated early with a better error.
    if ckpt.parent.name != "checkpoints" or ckpt.parent.parent.name != "data":
        fail(
            f"AudioLDM checkpoint {ckpt} must sit at <weights_root>/data/checkpoints/<file> "
            "(attacks.py:418-426). The model asset must preserve the "
            "audioldm/data/checkpoints/ nesting -- do not flatten it."
        )
    missing = [s for s in AUDIOLDM_SIBLINGS if not (ckpt.parent / s).is_file()]
    if missing:
        fail(
            f"AudioLDM checkpoint {ckpt.name} is present but these siblings that "
            f"audioldm_original.yaml resolves by relative path are missing from "
            f"{ckpt.parent}: {', '.join(missing)}"
        )
    print(f"diff_erase OK: {ckpt}  (weights_root={ckpt.parent.parent.parent})")


def _children(p: Path) -> list[str]:
    try:
        return sorted(c.name for c in p.iterdir())[:20]
    except OSError as exc:
        return [f"<unreadable: {exc}>"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direction", required=True, choices=("diff_erase", "sgmse"))
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--sgmse-checkpoint", required=True, type=Path)
    ap.add_argument("--audioldm-checkpoint", required=True, type=Path)
    ap.add_argument("--audioldm-config", required=True, type=Path)
    args = ap.parse_args()

    check_cuda()

    for split in ("train-clean-100", "dev-clean", "test-clean"):
        check_audio_split(args.data_root, split)

    if not args.audioldm_config.is_file():
        fail(f"AudioLDM config not found: {args.audioldm_config}")

    # Both attacks are needed in both directions: recipes.yaml holds the
    # untrained one out as the generalization probe, so it still has to load
    # at eval time. The trained one is required; the held-out one is a warning
    # because run_diffusion_swap.sh degrades gracefully when it's absent
    # (lines 52-56 / 66-70) rather than failing the run.
    sgmse_required = args.direction == "sgmse"
    if args.sgmse_checkpoint.is_file():
        print(f"sgmse OK: {args.sgmse_checkpoint}")
    elif sgmse_required:
        fail(f"direction=sgmse trains against SGMSE but its checkpoint is missing: {args.sgmse_checkpoint}")
    else:
        print(f"WARNING: SGMSE checkpoint missing ({args.sgmse_checkpoint}) -- held-out sgmse eval will be skipped")

    if args.direction == "diff_erase":
        check_audioldm(args.audioldm_checkpoint)
    elif args.audioldm_checkpoint.is_file():
        check_audioldm(args.audioldm_checkpoint)
    else:
        print(
            f"WARNING: AudioLDM checkpoint missing ({args.audioldm_checkpoint}) "
            "-- held-out diff_erase eval will be skipped"
        )

    print("preflight passed")


if __name__ == "__main__":
    main()
