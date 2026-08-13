#!/usr/bin/env bash
# Swaps which diffusion model the generator trains against (AudioLDM
# vs SGMSE), then evaluates -- holding out whichever one wasn't trained on, to
# measure whether robustness generalized. See
# src/audioseal_robust/config/recipes.yaml for exactly what each recipe sets,
# and AudioLDMAttack/SGMSEAttack's docstrings in attacks.py for why one is
# always held out during training.
#
# Usage:
#   ./run_diffusion_swap.sh audioldm     # train on AudioLDM, eval holds out SGMSE
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
#   AUDIOLDM_CHECKPOINT=checkpoints/audioldm/data/checkpoints/audioldm-s-full \
#   ./run_diffusion_swap.sh sgmse
#
# Multi-GPU: set LAUNCHER to run both stages under torchrun instead of a
# single process (see docs/MULTI_GPU.md). data.batch_size stays PER GPU, so
# this raises the effective batch by the number of processes:
#   LAUNCHER="torchrun --standalone --nproc_per_node=4" ./run_diffusion_swap.sh audioldm
set -euo pipefail
cd "$(dirname "$0")"

DIRECTION="${1:?Usage: $0 <audioldm|sgmse> [extra train.py overrides...]}"
shift

# Split on whitespace into an array rather than interpolating $LAUNCHER bare:
# it carries arguments ("torchrun --standalone --nproc_per_node=4"), so it has
# to word-split, but an unquoted expansion would also glob and would trip
# `set -u` when unset.
read -ra launcher <<< "${LAUNCHER:-python3}"

TRAIN_DIR="${TRAIN_DIR:-/data/datasets/LibriSpeech/train-clean-100}"
VALID_DIR="${VALID_DIR:-/data/datasets/LibriSpeech/dev-clean}"
EVAL_DIR="${EVAL_DIR:-/data/datasets/LibriSpeech/test-clean}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/audioseal_robust}"
AUDIOLDM_CHECKPOINT="${AUDIOLDM_CHECKPOINT:-/data/checkpoints/audioldm_root/data/checkpoints/audioldm-full-s-v2.ckpt}"
AUDIOLDM_CONFIG="${AUDIOLDM_CONFIG:-src/audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml}"
SGMSE_CHECKPOINT="${SGMSE_CHECKPOINT:-checkpoints/sgmse/train_vb_29nqe0uh_epoch=115.ckpt}"

# Shared between train and eval calls: whichever attack is TRAINED needs the
# exact same checkpoint/config in both, so the eval numbers reflect the model
# that was actually optimized, not a different checkpoint of the same attack.
shared_args=()
# MBD is held-out-only everywhere (see attacks.py) and needs no local
# weights file (downloads its own from HF), so it's free to always include.
eval_extra_args=(attack.mbd.checkpoint=auto)

# "$@" reaches train.py only, so without this there is no way to override an
# evaluate.py-only setting -- tracking.project, n_eval_batches, headline_strength
# etc. Space-separated, applied after the recipe so it wins, e.g.:
#   EVAL_EXTRA_ARGS="tracking.project=my-wandb-proj n_eval_batches=2" \
#     ./run_diffusion_swap.sh audioldm epochs=1
if [ -n "${EVAL_EXTRA_ARGS:-}" ]; then
  read -ra _eval_overrides <<< "$EVAL_EXTRA_ARGS"
  eval_extra_args+=("${_eval_overrides[@]}")
fi

case "$DIRECTION" in
  audioldm)
    train_recipe="audioldm"
    eval_recipe="after_audioldm_training"
    shared_args+=(attack.audioldm.checkpoint="$AUDIOLDM_CHECKPOINT" attack.audioldm.config="$AUDIOLDM_CONFIG")
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
    if [ -f "$AUDIOLDM_CHECKPOINT" ]; then
      eval_extra_args+=(attack.audioldm.checkpoint="$AUDIOLDM_CHECKPOINT" attack.audioldm.config="$AUDIOLDM_CONFIG")
    else
      echo "note: AUDIOLDM_CHECKPOINT ($AUDIOLDM_CHECKPOINT) not found -- held-out audioldm eval will be skipped, not fatal" >&2
    fi
    ;;
  *)
    echo "Unknown direction '$DIRECTION' -- expected audioldm or sgmse" >&2
    exit 1
    ;;
esac

echo "=== [1/2] Training: recipe=$train_recipe (launcher: ${launcher[*]}) ==="
PYTHONPATH=src "${launcher[@]}" -m audioseal_robust.train \
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
MPLBACKEND=Agg PYTHONPATH=src "${launcher[@]}" -m audioseal_robust.evaluate \
  recipe="$eval_recipe" \
  eval_dir="$EVAL_DIR" \
  label="swap_${DIRECTION}" \
  generator_checkpoint="$last_ckpt" \
  device=cuda \
  "${shared_args[@]}" \
  "${eval_extra_args[@]}"

echo "Done -- see eval_outputs/ for plots, mlflow/wandb run for full metrics."
