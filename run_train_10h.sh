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

# recipe=sgmse_mixed (config/recipes.yaml): attack.weights becomes
# {identity: 0.5, sgmse: 0.5} -- half the steps train against the real SGMSE
# speech-enhancement attack, half are unattacked identity steps (see that
# recipe's comment for why not sgmse-only: no easy steps to anchor bit
# accuracy on, and full exposure to SGMSE's own gradient instability). Either
# way this is a real attack, not the default.yaml identity-only baseline
# (attacks.py:803 -- a zero-weight attack is never even a candidate for
# SampledReconstructionAttack's random pick, so identity: 1.0 / sgmse: 0.0
# made a real attack mathematically impossible before this was set).
# AudioLDM stays held out for eval (after_sgmse_training recipe), to measure
# whether robustness generalizes to a diffusion attack never seen in training.
# Same checkpoint path convention/default as run_diffusion_swap.sh.
SGMSE_CHECKPOINT="${SGMSE_CHECKPOINT:-/data/checkpoints/sgmse/sgmse_vb_pretrained.ckpt}"
if [ ! -f "$SGMSE_CHECKPOINT" ]; then
  echo "SGMSE_CHECKPOINT not found at $SGMSE_CHECKPOINT -- set SGMSE_CHECKPOINT to a real path" >&2
  exit 1
fi

# Validation: dev-clean (never in train-clean-100, so eval_step measures
# something meaningful instead of re-scoring the training set under a
# different name). 60min target -- see the reasoning this was picked with:
#   - train.py's valid_iter = itertools.cycle(valid_dataloader) caches the
#     EXACT batch sequence from one pass and replays it unchanged forever --
#     it does not reshuffle -- so the subset size sets how many DISTINCT
#     batches exist before eval starts re-scoring ones it already saw.
#   - epochs=100 * updates_per_epoch=1000 is only a CAP per epoch, not a
#     guarantee: the inner loop breaks once the dataloader itself is
#     exhausted, whichever comes first. Measured on this exact 10h subset
#     (~2921 files / batch_size=16, drop_last=True): ~167 steps/epoch, so
#     ~16,700 steps total, not 100k -- confirmed from a real run's wandb log
#     (step=16700 at epoch=99). At eval_every=100 (below) that's ~167 eval
#     calls over the full run.
#   - 60min of dev-clean (~574 files at this dataset's measured ~6.27s/file
#     average) / batch_size=16 = ~35 distinct cached batches, so each is
#     revisited ~167/35 ~ 5x over the run rather than a handful of batches
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
# eval_every=100: see the reasoning above VALID_TARGET_MINUTES -- ~167 eval
# calls over the full ~16.7k-step run, each one a single forward-only batch,
# negligible next to ~16.7k forward+backward train steps.
# lambda_perc=0.0: perceptual loss off -- isolates detection_loss (presence +
# bit) while debugging why bit_loss plateaus (see eval graphs from the
# comic-snowball-9 run), so perceptual_loss's own gradient into x_wm can't
# mask or compete with what detection_loss alone is doing. Not a permanent
# default -- re-enable (drop this override) once detection-side training is
# actually working, since perceptual quality still matters for the real run.
# lambda_bit=2.0: presence_loss is the "easy" half of detection_loss (the
# generator can win it just by embedding *something* detectable, without
# correctly encoding bits) -- that's why comic-snowball-9's presence_loss
# dropped fast while bit_loss never moved. Weighting bit_loss 2x pushes
# optimization toward the actually-hard sub-problem instead of letting it
# settle for the cheap presence-only win. See TrainConfig.lambda_bit.
set +e
PYTHONUNBUFFERED=1 PYTHONPATH=src python3 -m audioseal_robust.train \
  recipe=sgmse_mixed \
  attack.sgmse.checkpoint="$SGMSE_CHECKPOINT" \
  data.train_dir="$TRAIN_SUBSET_DIR" \
  data.valid_dir="$VALID_SUBSET_DIR" \
  eval_every=100 \
  lambda_perc=0.0 \
  lambda_bit=2.0 \
  device=auto \
  2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

echo "exit code: $STATUS" | tee -a "$LOG"
exit "$STATUS"
