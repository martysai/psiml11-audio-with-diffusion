#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Evaluates data/generator_epoch18.pth (see
# psiml11-audio-with-diffusion/data/generator_epoch18.pth) against
# HopSkipJumpAttack (AudioMarkBench's hard-label black-box evasion attack,
# see attacks.py:HopSkipJumpAttack) on test-clean-fixed -- same
# checkpoint/dataset as run_eval_epoch18_audioldm_testclean_fixed.sh, attack
# swapped for hopskipjump. Isolates it (eval_attacks=[identity],
# held_out_attacks=[hopskipjump]) rather than running the full attack suite
# -- same convention as run_hopskipjump_eval.sh, which this is modeled on.
GENERATOR_CHECKPOINT="${GENERATOR_CHECKPOINT:-data/generator_epoch18.pth}"
EVAL_DIR="${EVAL_DIR:-/data/datasets/LibriSpeech/test-clean-fixed}"
LABEL="${LABEL:-epoch18_hsj_testclean_fixed}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DEVICE="${DEVICE:-cuda}"
# 1.0s (not 10.24s like the audioldm eval) -- that duration was specifically
# to match AudioLDM's own expected input length; HopSkipJump has no such
# requirement, and shorter segments keep its per-example query cost down
# (see the HSJ_* budget below). test-clean-fixed's files are all >=10.24s,
# so a 1.0s crop is still well within each file.
SEGMENT_DURATION="${SEGMENT_DURATION:-1.0}"
# 4, not audioldm's 8 -- HopSkipJumpAttack is a per-example iterative
# black-box search (see the HSJ_* query budget below), not a single forward
# pass, so it's far more expensive per example. Matches
# run_hopskipjump_eval.sh's own default for a real (non-examples/) eval_dir.
BATCH_SIZE="${BATCH_SIZE:-4}"
# 999999999, not a fixed headline-batch-count cap -- runs over the whole
# test-clean-fixed set (~586 examples, ~146 batches at batch_size=4) once
# instead of a small orientation sample: DataLoader's drop_last=True/
# shuffle=False/no-reiteration means the eval loop just ends naturally once
# the loader is exhausted -- same technique as run_full_eval_full_dataset.sh.
# HSJ is expensive per example (see the query budget below), so this is
# meant to run for a long time; override to a small number (e.g. 4) for a
# quick orientation check instead.
N_EVAL_BATCHES="${N_EVAL_BATCHES:-999999999}"

# HopSkipJumpAttack's query budget -- see HopSkipJumpAttackConfig in
# config.py for what each knob does. Bumped above the config schema's own
# (deliberately modest) defaults: a real trained AudioSeal detector's
# decision boundary is much harder for pure random-noise initialization to
# flip than the synthetic detector the unit tests use -- confirmed by hand
# in run_hopskipjump_eval.sh, the config default init_max_trials=100 mostly
# returned examples unperturbed ("no adversarial point found") against the
# real detector. Still far short of a from-the-paper run -- raise further
# via env vars if these still mostly fail to initialize.
HSJ_NUM_ITERATIONS="${HSJ_NUM_ITERATIONS:-25}"
HSJ_INIT_NUM_EVALS="${HSJ_INIT_NUM_EVALS:-20}"
HSJ_MAX_NUM_EVALS="${HSJ_MAX_NUM_EVALS:-100}"
HSJ_INIT_MAX_TRIALS="${HSJ_INIT_MAX_TRIALS:-300}"

mkdir -p logs
LOG="logs/eval_epoch18_hsj_testclean_fixed_$(date +%Y%m%d_%H%M%S).log"
echo "logging to $LOG (tail -f $LOG to follow live, or just check it later)"

# PYTHONUNBUFFERED/tee/PIPESTATUS: see run_full_eval_1h.sh's comment on the
# same pattern.
set +e
PYTHONUNBUFFERED=1 MPLBACKEND=Agg PYTHONPATH=src "$PYTHON_BIN" -m audioseal_robust.evaluate \
  eval_dir="$EVAL_DIR" \
  generator_checkpoint="$GENERATOR_CHECKPOINT" \
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
  2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

echo "exit code: $STATUS" | tee -a "$LOG"
exit "$STATUS"
