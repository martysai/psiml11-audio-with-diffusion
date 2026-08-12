#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

MPLBACKEND=Agg PYTHONPATH=src python3 -m audioseal_robust.evaluate \
  eval_dir=/data/datasets/LibriSpeech/test-clean \
  label=smoke \
  n_eval_batches=2 \
  device=cuda \
  tracking.backend=none \
  attack.diff_erase.checkpoint=/data/checkpoints/diff_erase_root/data/checkpoints/audioldm-full-s-v2.ckpt \
  attack.diff_erase.config=src/audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml \
  attack.mbd.checkpoint=auto

