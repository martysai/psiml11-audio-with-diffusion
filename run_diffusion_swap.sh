#!/usr/bin/env bash
# Swaps which diffusion model the generator trains against (AudioLDM/DiffErase
# vs SGMSE), then evaluates -- holding out whichever one wasn't trained on, to
# measure whether robustness generalized. See
# src/audioseal_robust/config/recipes.yaml for exactly what each recipe sets,
# and DiffEraseAttack/SGMSEAttack's docstrings in attacks.py for why one is
# always held out during training.
#
# Usage:
#   ./run_diffusion_swap.sh diff_erase   # train on AudioLDM, eval holds out SGMSE
#   ./run_diffusion_swap.sh sgmse        # train on SGMSE, eval holds out AudioLDM
#
# Extra args after the direction are forwarded to train.py as-is, e.g.:
#   ./run_diffusion_swap.sh sgmse epochs=5 data.batch_size=8
#
# Path defaults below match this server's known layout (see run_smoke_eval.sh
# for the same /data/... convention) -- override any of them via env vars,
# e.g. for a Colab/local run against examples/training_debug_colab.ipynb's
# data/LibriSpeech/... layout instead:
#   TRAIN_DIR=data/LibriSpeech/train-clean-100 \
#   VALID_DIR=data/LibriSpeech/dev-clean \
#   EVAL_DIR=data/LibriSpeech/test-clean \
#   DIFF_ERASE_CHECKPOINT=checkpoints/audioldm/data/checkpoints/audioldm-s-full \
#   ./run_diffusion_swap.sh sgmse
set -euo pipefail
cd "$(dirname "$0")"

DIRECTION="${1:?Usage: $0 <diff_erase|sgmse> [extra train.py overrides...]}"
shift

TRAIN_DIR="${TRAIN_DIR:-/data/datasets/LibriSpeech/train-clean-100}"
VALID_DIR="${VALID_DIR:-/data/datasets/LibriSpeech/dev-clean}"
EVAL_DIR="${EVAL_DIR:-/data/datasets/LibriSpeech/test-clean}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/audioseal_robust}"
DIFF_ERASE_CHECKPOINT="${DIFF_ERASE_CHECKPOINT:-/data/checkpoints/diff_erase_root/data/checkpoints/audioldm-full-s-v2.ckpt}"
DIFF_ERASE_CONFIG="${DIFF_ERASE_CONFIG:-src/audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml}"
SGMSE_CHECKPOINT="${SGMSE_CHECKPOINT:-checkpoints/sgmse/train_vb_29nqe0uh_epoch=115.ckpt}"

# Shared between train and eval calls: whichever attack is TRAINED needs the
# exact same checkpoint/config in both, so the eval numbers reflect the model
# that was actually optimized, not a different checkpoint of the same attack.
shared_args=()
# MBD is held-out-only everywhere (see attacks.py) and needs no local
# weights file (downloads its own from HF), so it's free to always include.
eval_extra_args=(attack.mbd.checkpoint=auto)

case "$DIRECTION" in
  diff_erase)
    train_recipe="diff_erase"
    eval_recipe="after_diff_erase_training"
    shared_args+=(attack.diff_erase.checkpoint="$DIFF_ERASE_CHECKPOINT" attack.diff_erase.config="$DIFF_ERASE_CONFIG")
    if [ -f "$SGMSE_CHECKPOINT" ]; then
      eval_extra_args+=(attack.sgmse.checkpoint="$SGMSE_CHECKPOINT")
    else
      echo "note: SGMSE_CHECKPOINT ($SGMSE_CHECKPOINT) not found -- held-out sgmse eval will be skipped, not fatal" >&2
    fi
    ;;
  sgmse)
    train_recipe="sgmse"
    eval_recipe="after_sgmse_training"
    if [ ! -f "$SGMSE_CHECKPOINT" ]; then
      echo "SGMSE_CHECKPOINT not found at $SGMSE_CHECKPOINT -- set SGMSE_CHECKPOINT to a real path" >&2
      exit 1
    fi
    shared_args+=(attack.sgmse.checkpoint="$SGMSE_CHECKPOINT")
    if [ -f "$DIFF_ERASE_CHECKPOINT" ]; then
      eval_extra_args+=(attack.diff_erase.checkpoint="$DIFF_ERASE_CHECKPOINT" attack.diff_erase.config="$DIFF_ERASE_CONFIG")
    else
      echo "note: DIFF_ERASE_CHECKPOINT ($DIFF_ERASE_CHECKPOINT) not found -- held-out diff_erase eval will be skipped, not fatal" >&2
    fi
    ;;
  *)
    echo "Unknown direction '$DIRECTION' -- expected diff_erase or sgmse" >&2
    exit 1
    ;;
esac

echo "=== [1/2] Training: recipe=$train_recipe ==="
PYTHONPATH=src python3 -m audioseal_robust.train \
  recipe="$train_recipe" \
  data.train_dir="$TRAIN_DIR" \
  data.valid_dir="$VALID_DIR" \
  checkpoint_dir="$CHECKPOINT_DIR" \
  device=cuda \
  "${shared_args[@]}" \
  "$@"

last_ckpt=$(ls -t "$CHECKPOINT_DIR"/generator_epoch*.pth 2>/dev/null | head -1)
if [ -z "$last_ckpt" ]; then
  echo "No checkpoint found in $CHECKPOINT_DIR after training -- something went wrong" >&2
  exit 1
fi

echo "=== [2/2] Evaluating: recipe=$eval_recipe, checkpoint=$last_ckpt ==="
MPLBACKEND=Agg PYTHONPATH=src python3 -m audioseal_robust.evaluate \
  recipe="$eval_recipe" \
  eval_dir="$EVAL_DIR" \
  label="swap_${DIRECTION}" \
  generator_checkpoint="$last_ckpt" \
  device=cuda \
  "${shared_args[@]}" \
  "${eval_extra_args[@]}"

echo "Done -- see eval_outputs/ for plots, mlflow/wandb run for full metrics."
