#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Same 10h-of-train-clean-100 / 60min-of-dev-clean subset scheme as
# run_train_10h.sh -- see that script's comments for the full reasoning.
# Reuses the SAME default subset dirs/marker files on purpose: the subset
# only decides which *files* are visible, not what duration is cropped from
# them (that's data.segment_duration below, applied per-__getitem__ by
# WavDirDataset) -- so if you already built the 10h/60min subsets via
# run_train_10h.sh, this script picks them up as-is, no rebuild needed.
DATASET_ROOT="${DATASET_ROOT:-/data/datasets/LibriSpeech}"
TRAIN_SOURCE_DIR="${TRAIN_SOURCE_DIR:-$DATASET_ROOT/train-clean-100}"
TRAIN_SUBSET_DIR="${TRAIN_SUBSET_DIR:-data/LibriSpeech/train-clean-100_10h}"
TRAIN_TARGET_MINUTES=600
VALID_SOURCE_DIR="${VALID_SOURCE_DIR:-$DATASET_ROOT/dev-clean}"
VALID_SUBSET_DIR="${VALID_SUBSET_DIR:-data/LibriSpeech/dev-clean_60min}"
VALID_TARGET_MINUTES=60

# recipe=sgmse_audioldm_mixed (config/recipes.yaml): attack.weights becomes
# {identity: 0.5, sgmse: 0.25, audioldm: 0.25} -- trains against BOTH
# diffusion attacks instead of just sgmse (run_train_10h.sh's
# sgmse_mixed), because a sgmse_mixed-trained checkpoint measured much
# worse on held-out audioldm eval than on sgmse itself: robustness learned
# against one reconstruction attack does not automatically transfer to a
# structurally different one (sgmse: STFT-domain, input-conditioned
# enhancement; audioldm: VAE-latent bottleneck + globally-attending UNet
# regeneration). Training against both directly is the standard fix instead
# of hoping for zero-shot generalization.
SGMSE_CHECKPOINT="${SGMSE_CHECKPOINT:-/data/checkpoints/sgmse/sgmse_vb_pretrained.ckpt}"
if [ ! -f "$SGMSE_CHECKPOINT" ]; then
  echo "SGMSE_CHECKPOINT not found at $SGMSE_CHECKPOINT -- set SGMSE_CHECKPOINT to a real path" >&2
  exit 1
fi
AUDIOLDM_CHECKPOINT="${AUDIOLDM_CHECKPOINT:-/data/checkpoints/audioldm_root/data/checkpoints/audioldm-full-s-v2.ckpt}"
if [ ! -f "$AUDIOLDM_CHECKPOINT" ]; then
  echo "AUDIOLDM_CHECKPOINT not found at $AUDIOLDM_CHECKPOINT -- set AUDIOLDM_CHECKPOINT to a real path" >&2
  exit 1
fi
AUDIOLDM_CONFIG="${AUDIOLDM_CONFIG:-src/audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml}"
if [ ! -f "$AUDIOLDM_CONFIG" ]; then
  echo "AUDIOLDM_CONFIG not found at $AUDIOLDM_CONFIG -- set AUDIOLDM_CONFIG to a real path" >&2
  exit 1
fi

# data.segment_duration=10.24: AudioLDM's own expected input length (see
# audioldm_original.yaml's duration field, and the eval-side "10.24s gap"
# analysis in run_full_eval_1h.sh). Setting it here means WavDirDataset
# already hands every batch a full 10.24s window (cropped from longer
# files, tiled/repeated from shorter ones -- data.py, never zero-padded),
# so AudioLDMAttack's own internal padding path never triggers, same as the
# eval-side fix but without needing a separate fixed-duration dataset --
# WavDirDataset's existing crop/tile logic already covers it.
# NOTE this is a large jump from run_train_10h.sh's default of 1.0s
# (AudioSeal's own native training length, config.py's DataConfig default)
# -- 10.24x more audio per sample, on top of two diffusion attack backbones
# now resident in GPU memory instead of one. BATCH_SIZE below starts much
# lower than run_train_10h.sh's as a result.
SEGMENT_DURATION="${SEGMENT_DURATION:-10.24}"

# Untested starting point, not a measured value like run_train_10h.sh's
# BATCH_SIZE=8 comment could claim: 10.24x longer sequences (through
# SEANet's LSTM too, not just the attacks) plus SGMSE's 30-step and
# AudioLDM's up-to-int(1000*strength_max)=80-step differentiable reverse
# loops both potentially active per step. Lower via env var if this still
# OOMs; if it clearly doesn't, this is very likely leaving headroom on the
# table -- raise it and see.
BATCH_SIZE="${BATCH_SIZE:-2}"

mkdir -p logs
LOG="logs/train_10h_audioldm_$(date +%Y%m%d_%H%M%S).log"
echo "logging to $LOG (tail -f $LOG to follow live, or just check it later)"

# Idempotent subset builder -- identical to run_train_10h.sh's, duplicated
# here rather than sourced so this script stays runnable standalone.
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

# eval_every=200 (vs run_train_10h.sh's 100): eval_step samples from the
# same weighted recipe as train_step, so validation batches now sometimes
# run the full audioldm reverse loop too -- doubled the interval rather
# than keeping 100, since run_train_10h.sh's "negligible next to train
# steps" reasoning no longer holds unchanged. Not measured yet; adjust once
# you see actual per-eval-call timing in the log.
# lambda_perc=0.0 / lambda_bit=2.0: same values run_train_10h.sh is using
# for the ongoing bit_loss-plateau investigation -- kept identical so this
# run isolates "does training against audioldm too change the audioldm
# eval number" rather than also changing the loss weighting at the same
# time. Revisit both once this run's own loss curves are in.
set +e
PYTHONUNBUFFERED=1 PYTHONPATH=src python3 -m audioseal_robust.train \
  recipe=sgmse_audioldm_mixed \
  attack.sgmse.checkpoint="$SGMSE_CHECKPOINT" \
  attack.audioldm.checkpoint="$AUDIOLDM_CHECKPOINT" \
  attack.audioldm.config="$AUDIOLDM_CONFIG" \
  data.train_dir="$TRAIN_SUBSET_DIR" \
  data.valid_dir="$VALID_SUBSET_DIR" \
  data.segment_duration="$SEGMENT_DURATION" \
  data.batch_size="$BATCH_SIZE" \
  eval_every=200 \
  lambda_perc=0.0 \
  lambda_bit=2.0 \
  device=auto \
  2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

echo "exit code: $STATUS" | tee -a "$LOG"
exit "$STATUS"
