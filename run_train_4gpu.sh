#!/usr/bin/env bash
# Generator-only robustness fine-tuning across 4 GPUs (DDP, one process per
# GPU). See docs/MULTI_GPU.md -- in particular, data.batch_size is PER GPU,
# so this trains at an effective batch of 4 x 16 = 64, and optim.lr is NOT
# rescaled for that automatically.
set -euo pipefail
cd "$(dirname "$0")"

NPROC="${NPROC:-4}"

PYTHONPATH=src torchrun --standalone --nproc_per_node="${NPROC}" \
  -m audioseal_robust.train \
  data.train_dir=/data/datasets/LibriSpeech/train-clean-100 \
  data.valid_dir=/data/datasets/LibriSpeech/dev-clean \
  data.batch_size=16 \
  data.num_workers=4 \
  eval_every=200 \
  device=cuda \
  tracking.backend=mlflow \
  "$@"
