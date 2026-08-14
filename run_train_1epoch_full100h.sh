#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# One epoch on the FULL train-clean-100 corpus (not the 10h subset), resuming
# from run4's epoch3 checkpoint instead of retraining from the pretrained
# baseline. See run_train_10h.sh for the recipe this mirrors (sgmse_mixed,
# lambda_bit=2.0, lambda_perc=0.0, batch_size=8 -- same OOM reasoning applies
# here, dataset size doesn't change per-step memory).
DATASET_ROOT="${DATASET_ROOT:-/data/datasets/LibriSpeech}"
TRAIN_DIR="${TRAIN_DIR:-$DATASET_ROOT/train-clean-100}"
SGMSE_CHECKPOINT="${SGMSE_CHECKPOINT:-/data/checkpoints/sgmse/sgmse_vb_pretrained.ckpt}"
RESUME_FROM="${RESUME_FROM:-checkpoints/run4_sgmse-mixed-gradclip_live_ema9j0h0/generator_epoch3.pth}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# Reuse the dev-clean validation subset already built by run_train_10h.sh --
# no need to rebuild it.
VALID_SUBSET_DIR="${VALID_SUBSET_DIR:-data/LibriSpeech/dev-clean_60min}"

if [ ! -f "$SGMSE_CHECKPOINT" ]; then
  echo "SGMSE_CHECKPOINT not found at $SGMSE_CHECKPOINT -- set SGMSE_CHECKPOINT to a real path" >&2
  exit 1
fi
if [ ! -d "$TRAIN_DIR" ]; then
  echo "TRAIN_DIR not found at $TRAIN_DIR -- set TRAIN_DIR to a real path" >&2
  exit 1
fi
if [ ! -f "$RESUME_FROM" ]; then
  echo "RESUME_FROM not found at $RESUME_FROM -- set RESUME_FROM to a real checkpoint path" >&2
  exit 1
fi

mkdir -p logs
LOG="logs/train_full100h_1ep_$(date +%Y%m%d_%H%M%S).log"
echo "logging to $LOG (tail -f $LOG to follow live, or just check it later)"

set +e
PYTHONUNBUFFERED=1 PYTHONPATH=src python3 -m audioseal_robust.train \
  recipe=sgmse_mixed \
  attack.sgmse.checkpoint="$SGMSE_CHECKPOINT" \
  data.train_dir="$TRAIN_DIR" \
  data.valid_dir="$VALID_SUBSET_DIR" \
  data.batch_size="$BATCH_SIZE" \
  epochs=1 \
  eval_every=100 \
  lambda_perc=0.0 \
  lambda_bit=2.0 \
  device=auto \
  generator.resume_from="$RESUME_FROM" \
  2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

echo "exit code: $STATUS" | tee -a "$LOG"
exit "$STATUS"
