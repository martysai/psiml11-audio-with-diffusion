#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs
LOG="logs/full_eval_1h_$(date +%Y%m%d_%H%M%S).log"
echo "logging to $LOG (tail -f $LOG to follow live, or just check it later)"

# PYTHONUNBUFFERED: without it, python buffers stdout when it's not an
# interactive terminal (i.e. once piped into tee), so the log file would
# only update in big chunks instead of as things actually happen.
# tee (not just >): still shows output live in this terminal too, in case
# you *are* watching -- it just no longer requires it.
# pipefail+tee would otherwise mask python's exit code as tee's (always 0);
# capture it via PIPESTATUS and exit with it explicitly at the end instead --
# doing this (rather than just running the pipeline under `set -e` directly)
# is also what lets this script still log the exit code on failure instead
# of dying mid-pipeline before reaching that line.
set +e
# diff_erase only -- no identity/bigvgan/dac/sgmse, no mbd (default
# held_out_attacks is [diff_erase, mbd], override to drop mbd).
# batch_size=8 (default) * n_eval_batches=150 * segment_duration=3.0s (default)
# = 3600s = 1h of audio for the headline number.
#
# NOTE on the label: "diff_erase" is the attack *method* (partial-noise +
# diffusion-regenerate, per the DiffErase paper) -- attack.diff_erase.checkpoint
# below is a generic pretrained AudioLDM checkpoint standing in as its
# diffusion prior, NOT a checkpoint DiffErase's authors trained/released
# (they published none). Stamping that into the label so output filenames
# (full_1h_audioldm_confusion.png etc.) don't silently read as "the real
# DiffErase model" later -- swap this if a genuine DiffErase checkpoint ever
# replaces the one below.
PYTHONUNBUFFERED=1 MPLBACKEND=Agg PYTHONPATH=src python3 -m audioseal_robust.evaluate \
  eval_dir=/data/datasets/LibriSpeech/test-clean \
  label=full_1h_audioldm \
  n_eval_batches=150 \
  device=cuda \
  eval_attacks=[] \
  held_out_attacks=[diff_erase] \
  tracking.backend=none \
  attack.diff_erase.checkpoint=/data/checkpoints/diff_erase_root/data/checkpoints/audioldm-full-s-v2.ckpt \
  attack.diff_erase.config=src/audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml \
  2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

echo "exit code: $STATUS" | tee -a "$LOG"
exit "$STATUS"
