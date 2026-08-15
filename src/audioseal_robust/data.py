# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal wav-directory dataset, enough to make the training loop runnable
end to end. Not a replacement for AudioCraft's manifest-based dataset
pipeline (egs/, audio_dataset.py, dset configs) -- if you need dataset
sharding, metadata filtering, etc. at scale, use that instead and pass its
output into `train_step`, which only requires batches of shape (B, 1, T).

Multi-GPU: `build_dataloader` attaches a `DistributedSampler` when handed a
distributed `DistEnv`, so each rank reads a disjoint shard of the files
rather than all four ranks training on the same audio. See its docstring for
the `set_epoch` requirement that comes with it.
"""

import random
import typing as tp
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from .distributed import DistEnv


_AUDIO_EXTENSIONS = (".wav", ".flac")


class WavDirDataset(Dataset):
    """Recursively collects audio files (.wav, .flac -- anything soundfile
    can decode) under `root`, and on each __getitem__ returns a random
    `segment_duration`-second mono crop resampled to `sample_rate`. Files
    shorter than the segment are looped."""

    def __init__(self, root: str, sample_rate: int = 16_000, segment_duration: float = 1.0):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_duration * sample_rate)
        self.files = sorted(p for ext in _AUDIO_EXTENSIONS for p in self.root.rglob(f"*{ext}"))
        if not self.files:
            raise RuntimeError(f"No audio files ({', '.join(_AUDIO_EXTENSIONS)}) found under {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        # soundfile rather than torchaudio.load: recent torchaudio (>=2.9)
        # routes torchaudio.load through torchcodec unconditionally (the
        # `backend=` kwarg no longer has a non-torchcodec option), which
        # needs a system FFmpeg install to load its shared libraries --
        # unavailable on plain dev boxes. soundfile decodes wav/flac
        # directly with no FFmpeg dependency at all; only the resample step
        # below still goes through torchaudio, which is a pure tensor op
        # (no decoding backend involved) and works either way.
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (frames, channels)
        wav = torch.from_numpy(data.T)  # (channels, frames)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        n = wav.size(-1)
        if n < self.segment_samples:
            reps = self.segment_samples // n + 1
            wav = wav.repeat(1, reps)
            n = wav.size(-1)
        start = random.randint(0, n - self.segment_samples)
        return wav[:, start:start + self.segment_samples]


def build_dataloader(
    root: str,
    sample_rate: int,
    segment_duration: float,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    env: tp.Optional[DistEnv] = None,
) -> tp.Tuple[torch.utils.data.DataLoader, tp.Optional[DistributedSampler]]:
    """Returns `(dataloader, sampler)`.

    `batch_size` is PER RANK (the standard DDP convention): under
    `torchrun --nproc_per_node=4` a batch_size of 16 means 16 examples per
    GPU and an effective batch of 64. Configs are unchanged from the
    single-GPU runs; the effective batch is what scales.

    `sampler` is a `DistributedSampler` when `env` is distributed, else None.
    The caller must call `sampler.set_epoch(epoch)` every epoch when it is
    not None -- otherwise the sampler reshuffles identically every epoch, so
    each rank sees the exact same subset of files for the whole run (see
    train.py, which does this).

    `drop_last=True` on both the sampler and the loader is load-bearing under
    DDP, not just tidiness: it makes every rank run the same number of steps
    with the same number of examples. Without it a rank that runs out of
    batches first stops calling into DDP's gradient allreduce and the others
    hang waiting for it, and the per-rank metric averages combined by
    `distributed.all_reduce_mean` would be weighted wrong.
    """
    dataset = WavDirDataset(root, sample_rate=sample_rate, segment_duration=segment_duration)

    sampler: tp.Optional[DistributedSampler] = None
    if env is not None and env.is_distributed:
        if len(dataset) < env.world_size * batch_size:
            raise RuntimeError(
                f"{root} has {len(dataset)} audio files, too few for world_size={env.world_size} x "
                f"batch_size={batch_size}={env.world_size * batch_size} examples per step with "
                "drop_last=True -- every rank would get zero batches. Lower batch_size, use fewer "
                "GPUs, or point at more data."
            )
        sampler = DistributedSampler(
            dataset,
            num_replicas=env.world_size,
            rank=env.rank,
            shuffle=shuffle,
            drop_last=True,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        # Mutually exclusive with `sampler`: DistributedSampler does the
        # shuffling itself (per-epoch, seeded by set_epoch).
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    return dataloader, sampler
