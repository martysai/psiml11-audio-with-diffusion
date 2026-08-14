#!/usr/bin/env bash
# Evaluates HopSkipJumpAttack (AudioMarkBench's hard-label black-box evasion
# attack, see attacks.py:HopSkipJumpAttack) against a generator+detector
# pair -- either the stock/baseline AudioSeal checkpoint, or a fine-tuned
# one from train.py. Isolates hopskipjump (eval_attacks=[identity],
# held_out_attacks=[hopskipjump]) rather than running the full attack
# suite -- same "isolate the one attack you're testing" convention as
# run_smoke_eval.sh does for audioldm.
#
# Usage:
#   ./run_hopskipjump_eval.sh                              # baseline (stock AudioSeal)
#   ./run_hopskipjump_eval.sh /path/to/generator_epochN.pth # a fine-tuned checkpoint
#   ./run_hopskipjump_eval.sh /path/to/checkpoint.pth n_eval_batches=8   # + extra evaluate.py overrides
#
# All paths/knobs below are env-var overridable, e.g. for a real dataset
# instead of the single-file examples/ fallback:
#   EVAL_DIR=/data/datasets/LibriSpeech/test-clean ./run_hopskipjump_eval.sh
#
# Cheaper/faster query budget (confirmed by hand: ~18s vs. ~176s per
# eval-batch on this CPU box, at the cost of a less thorough per-example
# search -- fine for a quick orientation number, not for a final one):
#   HSJ_NUM_ITERATIONS=5 HSJ_INIT_NUM_EVALS=10 HSJ_MAX_NUM_EVALS=20 \
#   HSJ_INIT_MAX_TRIALS=50 ./run_hopskipjump_eval.sh
#
# device=cpu by default and NOT auto -- the point of this script (see the
# task this was written for) is to run alongside a training job that's
# already using the GPU, without competing for it. Override with DEVICE=cuda
# if you're deliberately running this when the GPU is free.
set -euo pipefail
cd "$(dirname "$0")"

# torch.compile (invoked inside AudioSeal's generator.get_watermark) needs a
# C/C++ compiler it can shell out to; on a plain Windows box (no MSVC "cl"
# installed) that fails with "InvalidCxxCompiler: Compiler: cl is not
# found" the moment embed_watermark runs -- confirmed by hand while building
# this script. $OSTYPE is "msys"/"cygwin" under Git Bash on Windows, which is
# how this repo's scripts get run there; on the real (Linux) training server
# this branch never triggers, so torch.compile stays on and nothing here
# costs that box any perf.
if [[ "${OSTYPE:-}" == "msys" || "${OSTYPE:-}" == "cygwin" ]]; then
  export TORCHDYNAMO_DISABLE=1
fi

# Every other script here hardcodes `python3` (the training server's
# convention). On a Git-Bash-on-Windows dev box, `python3` on PATH can
# resolve to a bare MSYS/mingw interpreter with none of this project's
# packages installed, while the one that actually has them is `python` --
# confirmed by hand while building this script (python3 raised
# "ModuleNotFoundError: No module named 'numpy'" here). Auto-detect instead
# of hardcoding either name, so this runs unmodified on both; override with
# PYTHON_BIN=... if neither guess is right for your setup.
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1 && python3 -c "import torch" >/dev/null 2>&1; then
    PYTHON_BIN=python3
  elif command -v python >/dev/null 2>&1 && python -c "import torch" >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "Neither python3 nor python has torch installed on PATH -- set PYTHON_BIN=... explicitly." >&2
    exit 1
  fi
fi

CHECKPOINT="${1:-}"
if [ $# -gt 0 ]; then shift; fi

# No LibriSpeech mount on a local dev box (see run_smoke_eval.sh's
# /data/datasets/... convention, which assumes the training server's
# layout). Prefers a small local subset (219 files / 3 speakers / 28MB,
# pulled by hand from the same openslr.org/resources/12/test-clean.tar.gz
# the server-side scripts use, wildcard-extracted to just 3 speaker dirs --
# `tar --wildcards -xzf test-clean.tar.gz 'LibriSpeech/test-clean/<id>/*'`)
# over examples/ (this repo's single test.wav, looped) -- real, if small,
# multi-speaker data beats a one-file smoke check whenever it's available.
# Falls back to examples/ if that subset was never fetched on this box.
LOCAL_SUBSET_DIR="data/LibriSpeech/test-clean_subset"
if [ -z "${EVAL_DIR:-}" ] && [ -d "$LOCAL_SUBSET_DIR" ]; then
  EVAL_DIR="$LOCAL_SUBSET_DIR"
fi
EVAL_DIR="${EVAL_DIR:-examples}"
if [ "$EVAL_DIR" = "examples" ]; then
  echo "note: EVAL_DIR defaults to examples/ (a single wav, looped) -- functional smoke check only, not a meaningful sample. Set EVAL_DIR=<dir of wavs>, or fetch $LOCAL_SUBSET_DIR, for a real run." >&2
fi

DEVICE="${DEVICE:-cpu}"
SEGMENT_DURATION="${SEGMENT_DURATION:-1.0}"
# WavDirDataset's DataLoader uses drop_last=True (see data.py), so with the
# single-file examples/ fallback (dataset length 1) any batch_size > 1
# yields ZERO batches -- confirmed by hand, it raised "Evaluation dataloader
# produced no batches". 4 is safe for the multi-file local subset/a real
# EVAL_DIR; only the examples/ fallback needs 1, so pick per-case.
if [ "$EVAL_DIR" = "examples" ]; then
  BATCH_SIZE="${BATCH_SIZE:-1}"
else
  BATCH_SIZE="${BATCH_SIZE:-4}"
fi
N_EVAL_BATCHES="${N_EVAL_BATCHES:-4}"
TRACKING_BACKEND="${TRACKING_BACKEND:-none}"

# HopSkipJumpAttack's query budget -- see HopSkipJumpAttackConfig in
# config.py for what each knob does. Bumped above the config schema's own
# (deliberately modest) defaults: a real trained AudioSeal detector's
# decision boundary is much harder for pure random-noise initialization to
# flip than the synthetic detector the unit tests use (confirmed by hand --
# the config default init_max_trials=100 mostly returned examples
# unperturbed, i.e. "no adversarial point found," against the real
# detector). Still far short of what a from-the-paper run would use --
# raise further via env vars if these still mostly fail to initialize.
HSJ_NUM_ITERATIONS="${HSJ_NUM_ITERATIONS:-25}"
HSJ_INIT_NUM_EVALS="${HSJ_INIT_NUM_EVALS:-20}"
HSJ_MAX_NUM_EVALS="${HSJ_MAX_NUM_EVALS:-100}"
HSJ_INIT_MAX_TRIALS="${HSJ_INIT_MAX_TRIALS:-300}"

if [ -n "$CHECKPOINT" ]; then
  if [ ! -f "$CHECKPOINT" ]; then
    echo "CHECKPOINT given but not found at $CHECKPOINT" >&2
    exit 1
  fi
  LABEL="${LABEL:-hopskipjump_$(basename "$CHECKPOINT" .pth)}"
  checkpoint_args=(generator_checkpoint="$CHECKPOINT")
else
  LABEL="${LABEL:-hopskipjump_baseline}"
  checkpoint_args=()
fi

mkdir -p logs
LOG="logs/hopskipjump_eval_${LABEL}_$(date +%Y%m%d_%H%M%S).log"
echo "logging to $LOG (tail -f $LOG to follow live, or just check it later)"
echo "label=$LABEL device=$DEVICE eval_dir=$EVAL_DIR checkpoint=${CHECKPOINT:-<baseline>}"

# PYTHONUNBUFFERED/tee/PIPESTATUS: see run_full_eval_1h.sh's comment on the
# same pattern.
set +e
PYTHONUNBUFFERED=1 MPLBACKEND=Agg PYTHONPATH=src "$PYTHON_BIN" -m audioseal_robust.evaluate \
  eval_dir="$EVAL_DIR" \
  label="$LABEL" \
  device="$DEVICE" \
  segment_duration="$SEGMENT_DURATION" \
  batch_size="$BATCH_SIZE" \
  n_eval_batches="$N_EVAL_BATCHES" \
  num_workers=0 \
  eval_attacks=[identity] \
  held_out_attacks=[hopskipjump] \
  attack.hopskipjump.checkpoint=auto \
  attack.hopskipjump.num_iterations="$HSJ_NUM_ITERATIONS" \
  attack.hopskipjump.init_num_evals="$HSJ_INIT_NUM_EVALS" \
  attack.hopskipjump.max_num_evals="$HSJ_MAX_NUM_EVALS" \
  attack.hopskipjump.init_max_trials="$HSJ_INIT_MAX_TRIALS" \
  compute_visqol=false \
  tracking.backend="$TRACKING_BACKEND" \
  "${checkpoint_args[@]}" \
  "$@" \
  2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

echo "exit code: $STATUS" | tee -a "$LOG"
exit "$STATUS"
