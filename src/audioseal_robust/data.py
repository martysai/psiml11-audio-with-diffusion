# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal wav-directory dataset, enough to make the training loop runnable
end to end. Not a replacement for AudioCraft's manifest-based dataset
pipeline (egs/, audio_dataset.py, dset configs) -- if you need dataset
sharding, metadata filtering, etc. at scale, use that instead and pass its
output into `train_step`, which only requires batches of shape (B, 1, T)."""

import random
import typing as tp
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


class WavDirDataset(Dataset):
    """Recursively collects .wav files under `root`, and on each __getitem__
    returns a random `segment_duration`-second mono crop resampled to
    `sample_rate`. Files shorter than the segment are looped."""

    def __init__(self, root: str, sample_rate: int = 16_000, segment_duration: float = 1.0):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_duration * sample_rate)
        self.files = sorted(self.root.rglob("*.wav"))
        if not self.files:
            raise RuntimeError(f"No .wav files found under {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        wav, sr = torchaudio.load(str(path))
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
) -> torch.utils.data.DataLoader:
    dataset = WavDirDataset(root, sample_rate=sample_rate, segment_duration=segment_duration)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
