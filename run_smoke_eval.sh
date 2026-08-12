#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

MPLBACKEND=Agg PYTHONPATH=src python3 -m audioseal_robust.evaluate \
  eval_dir=/data/datasets/LibriSpeech/test-clean \
  label=smoke_10 \
  batch_size=1 \
  n_eval_batches=10 \
  n_curve_batches=1 \
  device=cuda \
  eval_attacks=[] \
  held_out_attacks=[diff_erase] \
  tracking.backend=none \
  attack.diff_erase.checkpoint=/data/checkpoints/diff_erase_root/data/checkpoints/audioldm-full-s-v2.ckpt \
  attack.diff_erase.config=src/audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml