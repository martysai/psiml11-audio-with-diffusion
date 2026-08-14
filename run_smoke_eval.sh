#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# audioldm only -- default eval_attacks=[identity,bigvgan,dac,audioldm] and
# held_out_attacks=[sgmse,mbd] would otherwise also run identity (and
# try + skip the rest for lack of a checkpoint/package). See
# run_full_eval_1h.sh's note on why the label carries "audioldm".
#
# tracking.backend left at its default (wandb) and n_curve_batches
# bumped to 20 -- same reasoning as run_full_eval_1h.sh, kept identical here
# so a smoke run and the real run are directly comparable in wandb, not
# just structurally similar.
#
# EVAL_DIR/segment_duration=10.24/save_row_artifacts: same "10.24s gap" fix
# as run_full_eval_1h.sh -- see that script's comment for the full reasoning
# (AudioLDM zero-pads shorter inputs with silence otherwise). Build
# test-clean-fixed on this machine first: `python3
# build_fixed_duration_eval_set.py /data/datasets/LibriSpeech/test-clean
# 10.24` (adjust the source path to this server's layout).
EVAL_DIR="${EVAL_DIR:-/data/datasets/LibriSpeech/test-clean-fixed}"

MPLBACKEND=Agg PYTHONPATH=src python3 -m audioseal_robust.evaluate \
  eval_dir="$EVAL_DIR" \
  label=smoke_audioldm \
  segment_duration=10.24 \
  n_eval_batches=2 \
  n_curve_batches=20 \
  device=cuda \
  eval_attacks=[] \
  held_out_attacks=[audioldm] \
  save_row_artifacts=true \
  attack.audioldm.checkpoint=/data/checkpoints/audioldm_root/data/checkpoints/audioldm-full-s-v2.ckpt \
  attack.audioldm.config=src/audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml

