#!/usr/bin/env python3
"""Builds a fixed-length eval subset from a LibriSpeech-style directory:
every output file is cropped to exactly `target_duration` seconds --
AudioLDM's own expected input length (see attacks.py:AudioLDMAttack.forward's
target_len padding, and the "10.24s gap" analysis this was written for) --
so AudioLDMAttack never has to zero-pad it. Files shorter than
`target_duration` are skipped entirely (no source material for a full,
unpadded window).

Unlike the header-only analyze_*.py scripts, this one actually decodes and
writes audio (cropping means new file content, not just a symlink) -- but it
only touches the qualifying subset (~586 of test-clean's 2620 files), not
the whole dataset, so it's still fast.

Usage:
    python3 build_fixed_duration_eval_set.py [source_dir] [target_duration_seconds]
"""

import sys
from pathlib import Path

import soundfile as sf

_AUDIO_EXTENSIONS = (".flac", ".wav")


def main() -> None:
    source_dir = Path(
        sys.argv[1] if len(sys.argv) > 1 else "<user-home>/Desktop/psiml/datasets/LibriSpeech/test-clean"
    )
    target_duration = float(sys.argv[2]) if len(sys.argv) > 2 else 10.24

    dest_dir = source_dir.parent / "test-clean-fixed"
    marker = dest_dir / f".fixed_{target_duration}s_complete"
    if marker.exists():
        print(f"reusing existing fixed-duration set at {dest_dir} (delete the dir to rebuild)")
        return

    files = sorted(p for ext in _AUDIO_EXTENSIONS for p in source_dir.rglob(f"*{ext}"))
    if not files:
        sys.exit(f"no audio files ({', '.join(_AUDIO_EXTENSIONS)}) found under {source_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_short = 0
    for path in files:
        info = sf.info(str(path))
        duration = info.frames / info.samplerate
        if duration < target_duration:
            skipped_short += 1
            continue

        target_frames = int(round(target_duration * info.samplerate))
        # start=0: deterministic crop (the point of this dataset is a fixed,
        # reproducible, no-padding window, not variety within the file).
        data, sr = sf.read(str(path), frames=target_frames, dtype="float32")
        out_path = dest_dir / path.name
        sf.write(str(out_path), data, sr)
        written += 1

    marker.touch()
    print(f"wrote {written} files (exactly {target_duration}s each) to {dest_dir}")
    print(f"skipped {skipped_short} files shorter than {target_duration}s")
    print(f"total: {written * target_duration / 3600:.2f}h")


if __name__ == "__main__":
    main()
