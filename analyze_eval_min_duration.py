#!/usr/bin/env python3
"""Counts how many files in a dataset are long enough to give AudioLDMAttack
a full `min_duration`-second window without needing to zero-pad (see
attacks.py:AudioLDMAttack.forward's target_len padding, and the "10.24s gap"
analysis it was written for) -- files shorter than that get padded with
silence before AudioLDM ever sees them, which is the problem this avoids.

Reads only each file's header (soundfile.info, no decode), so it's fast even
across LibriSpeech's ~2.6k test-clean files.

Usage:
    python3 analyze_eval_min_duration.py [dataset_dir] [min_duration_seconds]
"""

import sys
from pathlib import Path

import soundfile as sf

_AUDIO_EXTENSIONS = (".flac", ".wav")


def main() -> None:
    dataset_dir = Path(
        sys.argv[1] if len(sys.argv) > 1 else "/Users/djina/Desktop/psiml/datasets/LibriSpeech/test-clean"
    )
    min_duration = float(sys.argv[2]) if len(sys.argv) > 2 else 10.24

    files = sorted(p for ext in _AUDIO_EXTENSIONS for p in dataset_dir.rglob(f"*{ext}"))
    if not files:
        sys.exit(f"no audio files ({', '.join(_AUDIO_EXTENSIONS)}) found under {dataset_dir}")

    durations = []
    for path in files:
        info = sf.info(str(path))  # header-only, no decode
        durations.append(info.frames / info.samplerate)

    total_files = len(durations)
    qualifying = [d for d in durations if d >= min_duration]
    n_qualifying = len(qualifying)

    print(f"dataset: {dataset_dir}")
    print(f"min_duration threshold: {min_duration}s (AudioLDMAttack's target_len -- see attacks.py)")
    print(f"total files: {total_files}")
    print(f"files >= {min_duration}s: {n_qualifying} ({100 * n_qualifying / total_files:.1f}% of dataset)")
    print()
    print(f"if each qualifying file is cropped to exactly {min_duration}s (one no-padding eval example each):")
    print(f"  -> {n_qualifying} examples, {n_qualifying * min_duration / 3600:.2f}h total")
    print()
    print("if using each qualifying file's FULL natural length instead:")
    print(f"  -> {sum(qualifying) / 3600:.2f}h total")


if __name__ == "__main__":
    main()
