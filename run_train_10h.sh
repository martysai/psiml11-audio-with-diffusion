#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Trains on the first ~10h of audio (by cumulative duration, sorted file
# order for determinism) out of LibriSpeech train-clean-100. train.py's
# WavDirDataset (data.py) recursively scans data.train_dir for every
# .flac/.wav under it -- it has no built-in "first N hours" cutoff, so we
# build a subset directory of symlinks into the real dataset first and point
# training at that instead of the full ~100h corpus.
# DATASET_ROOT matches this server's known layout (see run_smoke_eval.sh /
# run_diffusion_swap.sh for the same /data/... convention on the eval side).
# Subset dirs default under the repo instead of DATASET_ROOT since the
# dataset mount itself may not be writable -- override any of these via env
# vars for a different layout (e.g. Colab/local, see run_diffusion_swap.sh).
DATASET_ROOT="${DATASET_ROOT:-/data/datasets/LibriSpeech}"
TRAIN_SOURCE_DIR="${TRAIN_SOURCE_DIR:-$DATASET_ROOT/train-clean-100}"
TRAIN_SUBSET_DIR="${TRAIN_SUBSET_DIR:-data/LibriSpeech/train-clean-100_10h}"
TRAIN_TARGET_MINUTES=600

# Validation: dev-clean (never in train-clean-100, so eval_step measures
# something meaningful instead of re-scoring the training set under a
# different name). 60min target -- see the reasoning this was picked with:
#   - train.py's valid_iter = itertools.cycle(valid_dataloader) caches the
#     EXACT batch sequence from one pass and replays it unchanged forever --
#     it does not reshuffle -- so the subset size sets how many DISTINCT
#     batches exist before eval starts re-scoring ones it already saw.
#   - default epochs=100 * updates_per_epoch=1000 = 100k training steps;
#     at eval_every=500 (below) that's 200 eval calls over the full run.
#   - 60min of dev-clean (~574 files at this dataset's measured ~6.27s/file
#     average) / batch_size=16 = ~35 distinct cached batches, so each is
#     revisited ~200/35 ~ 6x over the run rather than a handful of batches
#     being hammered hundreds of times. ~21% of dev-clean's ~4.7h total,
#     leaving the rest free for anything else that might want it later.
VALID_SOURCE_DIR="${VALID_SOURCE_DIR:-$DATASET_ROOT/dev-clean}"
VALID_SUBSET_DIR="${VALID_SUBSET_DIR:-data/LibriSpeech/dev-clean_60min}"
VALID_TARGET_MINUTES=60

mkdir -p logs
LOG="logs/train_10h_$(date +%Y%m%d_%H%M%S).log"
echo "logging to $LOG (tail -f $LOG to follow live, or just check it later)"

# Idempotent subset builder: reads only each file's header (sf.info, no
# decode) to sum duration, sorted file order for determinism, symlink farm
# so nothing is copied. Skips rebuilding if the target subset already has
# its completion marker.
build_subset() {
  local source_dir="$1" subset_dir="$2" target_minutes="$3"
  local marker="$subset_dir/.subset_${target_minutes}min_complete"
  if [ -f "$marker" ]; then
    echo "reusing existing ${target_minutes}min subset at $subset_dir"
    return
  fi
  echo "building ${target_minutes}min subset of $source_dir -> $subset_dir"
  rm -rf "$subset_dir"
  mkdir -p "$subset_dir"
  python3 - "$source_dir" "$subset_dir" "$target_minutes" <<'PYEOF'
import sys
from pathlib import Path
import soundfile as sf

source_dir, subset_dir, target_minutes = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
target_seconds = target_minutes * 60

files = sorted(source_dir.rglob("*.flac"))
if not files:
    sys.exit(f"no .flac files found under {source_dir}")

total_seconds = 0.0
linked = 0
for path in files:
    info = sf.info(str(path))  # header-only, no decode
    duration = info.frames / info.samplerate
    if total_seconds >= target_seconds:
        break
    (subset_dir / path.name).symlink_to(path)
    total_seconds += duration
    linked += 1

print(f"linked {linked} files, {total_seconds / 60:.2f}min")
(subset_dir / f".subset_{sys.argv[3]}min_complete").touch()
PYEOF
}

build_subset "$TRAIN_SOURCE_DIR" "$TRAIN_SUBSET_DIR" "$TRAIN_TARGET_MINUTES"
build_subset "$VALID_SOURCE_DIR" "$VALID_SUBSET_DIR" "$VALID_TARGET_MINUTES"

# PYTHONUNBUFFERED/tee/PIPESTATUS: see run_full_eval_1h.sh's comment on the
# same pattern -- keeps the log file live-updating and preserves python's
# real exit code through the pipe.
# eval_every=500: see the reasoning above VALID_TARGET_MINUTES -- 200 eval
# calls over the full 100k-step run, each one a single forward-only batch,
# negligible next to 100k forward+backward train steps.
set +e
PYTHONUNBUFFERED=1 PYTHONPATH=src python3 -m audioseal_robust.train \
  data.train_dir="$TRAIN_SUBSET_DIR" \
  data.valid_dir="$VALID_SUBSET_DIR" \
  eval_every=500 \
  device=auto \
  2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

echo "exit code: $STATUS" | tee -a "$LOG"
exit "$STATUS"
