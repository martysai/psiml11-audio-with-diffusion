#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Evaluates data/generator_epoch18.pth (see
# psiml11-audio-with-diffusion/data/generator_epoch18.pth) against the
# audioldm attack, on test-clean -- same "isolate the one attack you're
# testing" convention as run_smoke_eval.sh/run_eval_finetuned.sh.
#
# EVAL_DIR points at test-clean-fixed (NOT plain test-clean), and
# segment_duration=10.24 matches AudioLDM's own expected input length
# (audioldm_original.yaml). Without both of these, AudioLDMAttack zero-pads
# every segment shorter than 10.24s up to that length before running the
# VAE/UNet/vocoder pipeline -- a structurally different input than anything
# the pretrained checkpoint saw in training (see build_fixed_duration_eval_set.py
# and run_full_eval_1h.sh's fix/audioldm-eval-duration-gap history for the
# full "10.24s gap" analysis). test-clean-fixed already exists under
# /data/datasets/LibriSpeech/ (built for the sgmse evals, e.g.
# run_eval_run3ep5_tstar.sh) -- no build step needed here.
GENERATOR_CHECKPOINT="${GENERATOR_CHECKPOINT:-data/generator_epoch18.pth}"
EVAL_DIR="${EVAL_DIR:-/data/datasets/LibriSpeech/test-clean-fixed}"
LABEL="${LABEL:-epoch18_audioldm_testclean_fixed}"
# generator_epoch18.pth was saved with inner_conv-wrapped conv keys, i.e.
# by a Python >=3.10 process (builder.py picks Moshi's SEANet -- inner_conv
# wrapped -- on 3.10+, Audiocraft's SEANet -- flat convs -- below that; see
# builder.py:25). load_generator_under_test (evaluate.py) now detects the
# right direction from the actual model/checkpoint keys rather than
# guessing from sys.version_info, so this loads correctly regardless of
# which Python trained vs. which Python is evaluating -- no PYTHON_BIN
# override needed for this reason anymore, but still overridable if the
# default `python3` on this machine lacks the needed deps.
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p logs
LOG="logs/eval_epoch18_audioldm_testclean_fixed_$(date +%Y%m%d_%H%M%S).log"
echo "logging to $LOG (tail -f $LOG to follow live, or just check it later)"

# n_eval_batches=70: test-clean-fixed holds ~586 examples (see
# build_fixed_duration_eval_set.py); 586/batch_size(8) = 73 full batches
# available, 70 leaves a little slack rather than exactly maxing it out --
# same cap run_eval_run3ep5_tstar.sh/the sgmse baseline use for this same
# fixed set.
#
# PYTHONUNBUFFERED/tee/PIPESTATUS: see run_full_eval_1h.sh's comment on the
# same pattern.
set +e
PYTHONUNBUFFERED=1 MPLBACKEND=Agg PYTHONPATH=src "$PYTHON_BIN" -m audioseal_robust.evaluate \
  eval_dir="$EVAL_DIR" \
  generator_checkpoint="$GENERATOR_CHECKPOINT" \
  label="$LABEL" \
  segment_duration=10.24 \
  n_eval_batches=70 \
  n_curve_batches=20 \
  device=cuda \
  eval_attacks=[] \
  held_out_attacks=[audioldm] \
  attack.audioldm.checkpoint=/data/checkpoints/diff_erase_root/data/checkpoints/audioldm-full-s-v2.ckpt \
  attack.audioldm.config=src/audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml \
  2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

echo "exit code: $STATUS" | tee -a "$LOG"
exit "$STATUS"
