#!/usr/bin/env python3
"""Reports min/max/mean/median audio duration for a directory of .flac/.wav
files. Reads only each file's header (soundfile.info, no decode) so it stays
fast even across LibriSpeech's ~28k train-clean-100 files.

Usage:
    python3 analyze_dataset_durations.py <dataset_dir>
"""

import statistics
import sys
from pathlib import Path

import soundfile as sf

_AUDIO_EXTENSIONS = (".flac", ".wav")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python3 analyze_dataset_durations.py <dataset_dir>")
    dataset_dir = Path(sys.argv[1])
    files = sorted(p for ext in _AUDIO_EXTENSIONS for p in dataset_dir.rglob(f"*{ext}"))
    if not files:
        sys.exit(f"no audio files ({', '.join(_AUDIO_EXTENSIONS)}) found under {dataset_dir}")

    durations = []
    for path in files:
        info = sf.info(str(path))  # header-only, no decode
        durations.append(info.frames / info.samplerate)

    durations.sort()
    total = sum(durations)

    print(f"dataset: {dataset_dir}")
    print(f"files: {len(durations)}")
    print(f"total duration: {total:.1f}s ({total / 3600:.2f}h)")
    print(f"min: {durations[0]:.3f}s")
    print(f"max: {durations[-1]:.3f}s")
    print(f"mean: {statistics.mean(durations):.3f}s")
    print(f"median: {statistics.median(durations):.3f}s")
    print(f"stdev: {statistics.stdev(durations):.3f}s")


if __name__ == "__main__":
    main()
