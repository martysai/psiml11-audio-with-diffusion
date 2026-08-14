#!/usr/bin/env bash
# The run_full_eval_1h.sh evaluation spread over 4 GPUs (DDP, one process per
# GPU). n_eval_batches is a GLOBAL total split across ranks, so this measures
# exactly the same 150 batches as the single-GPU script -- roughly 4x faster,
# and the resulting numbers stay directly comparable to baselines recorded on
# 1xA100. See docs/MULTI_GPU.md.
#
# The attack is keyed `audioldm`, not `diff_erase`: the *method* is
# DiffErase's partial-noise + diffusion-regenerate, but the config key (and
# the checkpoint-path convention below) is named after the AudioLDM prior it
# runs on, which is the only checkpoint that exists for it -- run_full_eval_1h.sh
# passes the same two keys. `attack.diff_erase.*` is not in the schema, and
# OmegaConf rejects unknown keys outright, so a run using it would die at
# config-parse time before ever reaching a GPU.
set -euo pipefail
cd "$(dirname "$0")"

NPROC="${NPROC:-4}"

MPLBACKEND=Agg PYTHONPATH=src torchrun --standalone --nproc_per_node="${NPROC}" \
  -m audioseal_robust.evaluate \
  eval_dir=/data/datasets/LibriSpeech/test-clean \
  label=full_1h_4gpu \
  n_eval_batches=150 \
  device=cuda \
  tracking.backend=none \
  attack.audioldm.checkpoint=/data/checkpoints/audioldm_root/data/checkpoints/audioldm-full-s-v2.ckpt \
  attack.audioldm.config=src/audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml \
  attack.mbd.checkpoint=auto \
  "$@"
