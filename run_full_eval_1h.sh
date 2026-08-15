#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Machine-specific overrides (gitignored): EVAL_DIR / AUDIOLDM_CKPT /
# PYTHON_BIN etc. pointing at wherever this box keeps its datasets, weights
# and interpreter. Keeps absolute local paths out of git while letting this
# script run unmodified on both the training server and a dev box.
if [ -f .env.local ]; then . ./.env.local; fi

# Everything below is env-var overridable (same convention as
# run_hopskipjump_eval.sh), defaulting to the training server's /data layout
# and falling back to this repo's own data/ dir when that mount isn't there.
if [ -z "${EVAL_DIR:-}" ]; then
  if [ -d /data/datasets/LibriSpeech/test-clean ]; then
    EVAL_DIR=/data/datasets/LibriSpeech/test-clean
  else
    EVAL_DIR=data/LibriSpeech/test-clean
  fi
fi
if [ ! -d "$EVAL_DIR" ]; then
  echo "EVAL_DIR not found: $EVAL_DIR -- set EVAL_DIR=<dir of wavs/flacs> (e.g. in .env.local)" >&2
  exit 1
fi

AUDIOLDM_CKPT="${AUDIOLDM_CKPT:-/data/checkpoints/audioldm_root/data/checkpoints/audioldm-full-s-v2.ckpt}"
if [ ! -f "$AUDIOLDM_CKPT" ]; then
  echo "AUDIOLDM_CKPT not found: $AUDIOLDM_CKPT -- set AUDIOLDM_CKPT=<weights_root>/data/checkpoints/audioldm-full-s-v2.ckpt (e.g. in .env.local)" >&2
  exit 1
fi

DEVICE="${DEVICE:-cuda}"
LABEL="${LABEL:-full_1h_audioldm}"
N_EVAL_BATCHES="${N_EVAL_BATCHES:-150}"
N_CURVE_BATCHES="${N_CURVE_BATCHES:-20}"
TRACKING_BACKEND="${TRACKING_BACKEND:-wandb}"
# wandb_mode=online needs a logged-in account; offline still records the run
# locally (syncable later with `wandb sync`) and never blocks on a login
# prompt, which would otherwise hang an unattended multi-hour run.
WANDB_MODE_ARG="${WANDB_MODE:-online}"

# AudioSealWM.get_watermark is torch.compile-decorated at import time and
# Inductor shells out to a C++ compiler; a plain Windows box has no MSVC
# cl.exe, so this must be set before audioseal is imported (audioseal's own
# escape hatch -- see tests/conftest.py's note). Never triggers on the Linux
# server, so that box keeps torch.compile and loses no perf.
if [[ "${OSTYPE:-}" == "msys" || "${OSTYPE:-}" == "cygwin" ]]; then
  export NO_TORCH_COMPILE=1
fi

# python3 is the training server's convention, but a Git-Bash-on-Windows box
# can have neither `python3` nor `python` on PATH (conda envs aren't
# activated there). PYTHON_BIN is word-split so it can name a conda env, e.g.
#   PYTHON_BIN="/d/miniconda3/Scripts/conda.exe run -n <env> --no-capture-output python"
# Going through `conda run` rather than the env's python.exe directly matters
# on Windows: activation is what puts the env's Library/bin (holding
# cudnn64_9.dll) on PATH, and without it every conv dies with "Invalid
# handle. Cannot load symbol cudnnGetVersion".
PYTHON_BIN="${PYTHON_BIN:-python3}"
read -r -a PYTHON_CMD <<< "$PYTHON_BIN" || true

mkdir -p logs
LOG="logs/full_eval_1h_$(date +%Y%m%d_%H%M%S).log"
echo "logging to $LOG (tail -f $LOG to follow live, or just check it later)"
echo "label=$LABEL device=$DEVICE eval_dir=$EVAL_DIR"
echo "checkpoint=$AUDIOLDM_CKPT"
echo "python=${PYTHON_CMD[*]}"

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
# tracking.backend defaults to wandb (overridable via TRACKING_BACKEND) --
# this is the real baseline number, meant to sit in the same wandb experiment
# as later post-fine-tuning runs for a same-dashboard comparison (see
# evaluate.py's module docstring). Set WANDB_MODE=offline on a box with no
# wandb login, or TRACKING_BACKEND=none to skip tracking entirely.
#
# n_curve_batches=20 (up from the config default of 6): curve cost doesn't
# scale with n_eval_batches (see EvalConfig.n_curve_batches), so this only
# adds a couple minutes total, but roughly 3x's the sample count backing
# each t_star_grid point -- worth it for a number going in front of mentors.
PYTHONUNBUFFERED=1 MPLBACKEND=Agg PYTHONPATH=src "${PYTHON_CMD[@]}" -m audioseal_robust.evaluate \
  eval_dir="$EVAL_DIR" \
  label="$LABEL" \
  n_eval_batches="$N_EVAL_BATCHES" \
  n_curve_batches="$N_CURVE_BATCHES" \
  device="$DEVICE" \
  eval_attacks=[] \
  held_out_attacks=[audioldm] \
  attack.audioldm.checkpoint="$AUDIOLDM_CKPT" \
  attack.audioldm.config=src/audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml \
  tracking.backend="$TRACKING_BACKEND" \
  tracking.wandb_mode="$WANDB_MODE_ARG" \
  "$@" \
  2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

echo "exit code: $STATUS" | tee -a "$LOG"
exit "$STATUS"
