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
# EVAL_DIR: test-clean-fixed, NOT plain test-clean -- see the "10.24s gap"
# analysis this was built for (build_fixed_duration_eval_set.py). AudioLDM's
# own expected input is duration=10.24s (audioldm_original.yaml); feeding it
# our usual 3s segments means AudioLDMAttack zero-pads 7.24s of artificial
# silence onto every example (70% of what it actually processes) before
# running the VAE/UNet/vocoder pipeline -- a structurally different input
# than anything the pretrained checkpoint saw in training, made worse by the
# UNet's global self-attention (use_spatial_transformer: true) mixing that
# padding into the real-content region instead of leaving it isolated.
# test-clean-fixed holds only the ~586 test-clean files that are already
# >=10.24s, each cropped to exactly that length (deterministic, from the
# start) -- so AudioLDMAttack never pads at all. Build it on whichever
# machine runs this script: `python3 build_fixed_duration_eval_set.py
# /data/datasets/LibriSpeech/test-clean 10.24` (adjust the source path to
# this server's layout).
EVAL_DIR="${EVAL_DIR:-/data/datasets/LibriSpeech/test-clean-fixed}"

set +e
# audioldm only -- no identity/bigvgan/dac/sgmse, no mbd (default
# held_out_attacks is [sgmse, mbd], override to isolate audioldm).
# batch_size=8 (default) * n_eval_batches=70 * segment_duration=10.24s
# = 5734s (~1.59h) of audio for the headline number -- n_eval_batches capped
# at 70 (not the old 150) because test-clean-fixed only has 586 examples;
# 586/batch_size(8) = 73 full batches available, 70 leaves a little slack
# rather than exactly maxing it out.
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
#
# save_row_artifacts=true: writes every example's (x, x_wm, x_att) as .wav
# plus per-example metrics (bit_accuracy, presence_pos/neg, attack_sisnr) to
# CSV under output_dir/full_1h_audioldm_audioldm_rows/ -- see evaluate.py's
# _save_row_artifacts. Lets you actually listen to/inspect individual
# examples instead of only seeing aggregate numbers.
PYTHONUNBUFFERED=1 MPLBACKEND=Agg PYTHONPATH=src python3 -m audioseal_robust.evaluate \
  eval_dir="$EVAL_DIR" \
  label=full_1h_audioldm \
  segment_duration=10.24 \
  n_eval_batches=70 \
  n_curve_batches=20 \
  device=cuda \
  eval_attacks=[] \
  held_out_attacks=[audioldm] \
  save_row_artifacts=true \
  attack.audioldm.checkpoint=/data/checkpoints/audioldm_root/data/checkpoints/audioldm-full-s-v2.ckpt \
  attack.audioldm.config=src/audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml \
  2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

echo "exit code: $STATUS" | tee -a "$LOG"
exit "$STATUS"
