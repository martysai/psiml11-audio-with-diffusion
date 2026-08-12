#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# diff_erase only -- default eval_attacks=[identity,bigvgan,dac,sgmse] and
# held_out_attacks=[diff_erase,mbd] would otherwise also run identity (and
# try + skip the rest for lack of a checkpoint/package). See
# run_full_eval_1h.sh's note on why the label carries "audioldm".
MPLBACKEND=Agg PYTHONPATH=src python3 -m audioseal_robust.evaluate \
  eval_dir=/data/datasets/LibriSpeech/test-clean \
  label=smoke_audioldm \
  n_eval_batches=2 \
  device=cuda \
  eval_attacks=[] \
  held_out_attacks=[diff_erase] \
  tracking.backend=none \
  attack.diff_erase.checkpoint=/data/checkpoints/diff_erase_root/data/checkpoints/audioldm-full-s-v2.ckpt \
  attack.diff_erase.config=src/audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml

