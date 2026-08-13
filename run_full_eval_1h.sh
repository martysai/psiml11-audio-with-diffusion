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
# audioldm only -- no identity/bigvgan/dac/sgmse, no mbd (default
# held_out_attacks is [sgmse, mbd], override to isolate audioldm).
# batch_size=8 (default) * n_eval_batches=150 * segment_duration=3.0s (default)
# = 3600s = 1h of audio for the headline number.
#
# NOTE on the label: the attack *method* here is partial-noise +
# diffusion-regenerate (as described in the DiffErase paper) -- attack.audioldm.checkpoint
# below is a pretrained AudioLDM checkpoint serving as its
# diffusion prior, NOT a checkpoint DiffErase's authors trained/released
# (they published none, which is why this attack is named after AudioLDM and
# not after them). Stamping that into the label so output filenames
# (full_1h_audioldm_confusion.png etc.) say what actually ran.
#
# tracking.backend left at its default (wandb) instead of "none" -- this is
# the real baseline number, meant to sit in the same wandb experiment as
# later post-fine-tuning runs for a same-dashboard comparison (see
# evaluate.py's module docstring).
#
# n_curve_batches=20 (up from the config default of 6): curve cost doesn't
# scale with n_eval_batches (see EvalConfig.n_curve_batches), so this only
# adds a couple minutes total, but roughly 3x's the sample count backing
# each t_star_grid point -- worth it for a number going in front of mentors.
PYTHONUNBUFFERED=1 MPLBACKEND=Agg PYTHONPATH=src python3 -m audioseal_robust.evaluate \
  eval_dir=/data/datasets/LibriSpeech/test-clean \
  label=full_1h_audioldm \
  n_eval_batches=150 \
  n_curve_batches=20 \
  device=cuda \
  eval_attacks=[] \
  held_out_attacks=[audioldm] \
  attack.audioldm.checkpoint=/data/checkpoints/audioldm_root/data/checkpoints/audioldm-full-s-v2.ckpt \
  attack.audioldm.config=src/audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml \
  2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

echo "exit code: $STATUS" | tee -a "$LOG"
exit "$STATUS"
