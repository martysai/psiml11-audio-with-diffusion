#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Fast end-to-end sanity check for train.py: a couple minutes of audio, a
# couple minutes of validation audio, one epoch of a handful of steps.
# Not a robustness number -- just "does the training loop actually run
# without crashing", before committing to run_train_10h.sh's real run.
#
# DATASET_ROOT matches this server's known layout (see run_smoke_eval.sh /
# run_diffusion_swap.sh for the same /data/... convention on the eval side).
# Subset dirs default under the repo instead of DATASET_ROOT since the
# dataset mount itself may not be writable -- override any of these via env
# vars for a different layout (e.g. Colab/local, see run_diffusion_swap.sh).
DATASET_ROOT="${DATASET_ROOT:-/data/datasets/LibriSpeech}"
TRAIN_SOURCE_DIR="${TRAIN_SOURCE_DIR:-$DATASET_ROOT/train-clean-100}"
TRAIN_SUBSET_DIR="${TRAIN_SUBSET_DIR:-data/LibriSpeech/train-clean-100_smoke}"
TRAIN_TARGET_MINUTES=3

# Validation is on (eval_every set below) -- separate held-out split
# (dev-clean, never in train-clean-100) so eval_step measures something
# meaningful rather than re-scoring the training set under a different name.
VALID_SOURCE_DIR="${VALID_SOURCE_DIR:-$DATASET_ROOT/dev-clean}"
VALID_SUBSET_DIR="${VALID_SUBSET_DIR:-data/LibriSpeech/dev-clean_smoke}"
VALID_TARGET_MINUTES=2

# recipe=sgmse, same checkpoint path convention/default as run_train_10h.sh --
# smoke-tests the actual attack path run_train_10h.sh will use, not just the
# default identity-only pass-through.
SGMSE_CHECKPOINT="${SGMSE_CHECKPOINT:-/data/checkpoints/sgmse/sgmse_vb_pretrained.ckpt}"
if [ ! -f "$SGMSE_CHECKPOINT" ]; then
  echo "SGMSE_CHECKPOINT not found at $SGMSE_CHECKPOINT -- set SGMSE_CHECKPOINT to a real path" >&2
  exit 1
fi

mkdir -p logs
LOG="logs/train_smoke_$(date +%Y%m%d_%H%M%S).log"
echo "logging to $LOG (tail -f $LOG to follow live, or just check it later)"

# Idempotent subset builder (see run_train_10h.sh's comment on why this
# exists -- WavDirDataset has no built-in "first N minutes" cutoff): reads
# only each file's header (sf.info, no decode) to sum duration, sorted file
# order for determinism, symlink farm so nothing is copied.
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

# batch_size=2 (default 16 needs more files/segment than a 3min smoke subset
# reliably has, once drop_last=True is accounted for): updates_per_epoch=5,
# log_every=1, eval_every=2 so both the train step and eval_step paths
# actually get exercised within a few seconds.
# tracking.backend=none: this is a throwaway run, not a number worth a wandb
# entry -- see run_smoke_eval.sh for the opposite call on the eval side
# (that one deliberately DOES track, to be comparable against the real run).
# checkpoint_dir is separate from the real run's so this never clobbers it.
set +e
PYTHONUNBUFFERED=1 PYTHONPATH=src python3 -m audioseal_robust.train \
  recipe=sgmse \
  attack.sgmse.checkpoint="$SGMSE_CHECKPOINT" \
  data.train_dir="$TRAIN_SUBSET_DIR" \
  data.valid_dir="$VALID_SUBSET_DIR" \
  data.batch_size=2 \
  epochs=1 \
  updates_per_epoch=5 \
  log_every=1 \
  eval_every=2 \
  device=auto \
  checkpoint_dir=./checkpoints/audioseal_robust_smoke \
  tracking.backend=none \
  2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

echo "exit code: $STATUS" | tee -a "$LOG"
exit "$STATUS"
