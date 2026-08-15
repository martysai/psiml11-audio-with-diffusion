#!/usr/bin/env bash
# AzureML in-job wrapper around ../run_diffusion_swap.sh.
#
# run_diffusion_swap.sh is the real entry point and stays untouched -- it
# already takes every machine-specific path as an env var. This script's only
# jobs are the things that are specific to running it *inside an AML command
# job*:
#
#   1. Turn ${{inputs.*}}/${{outputs.*}} mount paths into those env vars.
#   2. Absorb the layout ambiguity of AML mounts (a registered folder may or
#      may not mount with its own top-level directory name preserved), so the
#      job doesn't fail on a path typo after a 10-minute image pull.
#   3. Run preflight.py, which hard-fails on a CPU-only box. device.py:36-40
#      only *warns* and falls back to CPU when cuda isn't available, and a
#      diffusion-backprop training loop on CPU looks healthy while being
#      ~100x too slow -- on a paid A100 that is the expensive failure mode.
#   4. Point HF/matplotlib/torch caches at writable scratch. The container's
#      HOME is not reliably writable, and AudioSeal's generator/detector plus
#      the mbd attack all pull weights from Hugging Face at runtime.
#   5. Own experiment tracking end to end (see "Tracking" below).
#   6. Copy eval_outputs/ to the output mount even when the run fails, so a
#      partial result is still recoverable.
#
# Tracking: the job YAML deliberately sets no tracking.* override. This script
# derives the whole tracking config from WANDB_* env vars instead, so there is
# exactly one place that decides it, and applies it to BOTH the train and the
# eval call -- otherwise eval would silently land in default_eval.yaml's
# separate `audioseal-robust-eval` project while training went to yours, and
# the eval numbers are the actual experimental result.
#
# Usage (from the repo root, which is where AML drops the snapshot):
#   bash azureml/aml_run.sh <audioldm|sgmse> \
#       --librispeech DIR --checkpoints DIR --artifacts DIR \
#       [-- train.py overrides...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  echo "usage: bash azureml/aml_run.sh <audioldm|sgmse|eval> --librispeech DIR --checkpoints DIR --artifacts DIR [--train-dir DIR] [--valid-dir DIR] [--eval-dir DIR] [--generators DIR] [--generator NAME] [--label NAME] [-- overrides...]" >&2
  exit 2
}

DIRECTION="${1:-}"
[ -n "$DIRECTION" ] || usage
shift

LIBRISPEECH=""
CHECKPOINTS=""
ARTIFACTS=""
# direction=eval only. --generators is the mounted model asset holding the
# fine-tuned .pth files; --generator selects one by filename within it. Left
# empty, evaluate.py's own default (the stock audioseal_wm_16bits card) stands,
# which is exactly the baseline arm.
GENERATORS=""
GENERATOR_NAME=""
LABEL_ARG=""
# Per-split overrides. These are CLI flags rather than environment variables
# because AzureML only expands ${{inputs.*}} inside a job's `command` -- set
# one in `environment_variables` and the container receives the literal
# string "${{inputs.trainfixed}}/...", which is exactly how
# <redacted-run> failed. Passing them here puts them in
# the command, where substitution happens.
TRAIN_DIR_ARG=""
VALID_DIR_ARG=""
EVAL_DIR_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --librispeech) LIBRISPEECH="${2:?--librispeech needs a value}"; shift 2 ;;
    --checkpoints) CHECKPOINTS="${2:?--checkpoints needs a value}"; shift 2 ;;
    --artifacts)   ARTIFACTS="${2:?--artifacts needs a value}";     shift 2 ;;
    --train-dir)   TRAIN_DIR_ARG="${2:?--train-dir needs a value}"; shift 2 ;;
    --valid-dir)   VALID_DIR_ARG="${2:?--valid-dir needs a value}"; shift 2 ;;
    --eval-dir)    EVAL_DIR_ARG="${2:?--eval-dir needs a value}";   shift 2 ;;
    --generators)  GENERATORS="${2:?--generators needs a value}";   shift 2 ;;
    --generator)   GENERATOR_NAME="${2:?--generator needs a value}"; shift 2 ;;
    --label)       LABEL_ARG="${2:?--label needs a value}";         shift 2 ;;
    --)            shift; break ;;
    *)             echo "unknown argument: $1" >&2; usage ;;
  esac
done
case "$DIRECTION" in
  eval)
    # No diffusion backbone is loaded, so --checkpoints is optional here.
    [ -n "$LIBRISPEECH" ] && [ -n "$ARTIFACTS" ] || usage ;;
  *)
    [ -n "$LIBRISPEECH" ] && [ -n "$CHECKPOINTS" ] && [ -n "$ARTIFACTS" ] || usage ;;
esac
# Everything still in "$@" is forwarded verbatim to train.py (or, for
# direction=eval, to evaluate.py).

# --- Absorb mount-layout ambiguity ------------------------------------------
# Registering a folder as a uri_folder/custom_model asset does not guarantee
# the folder's own name survives into the mount path, and the LibriSpeech
# tarballs additionally extract under a LibriSpeech/ parent. Rather than
# guess, look for the level that actually has the expected children.
descend_to() {  # descend_to <root> <expected-child> -- echoes the right level
  local root="$1" child="$2"
  if [ -d "$root/$child" ]; then echo "$root"; return 0; fi
  local sub
  for sub in "$root"/*/; do
    [ -d "${sub}${child}" ] && { echo "${sub%/}"; return 0; }
  done
  echo "$root"  # let preflight.py produce the readable error
}

DATA_ROOT="$(descend_to "$LIBRISPEECH" "train-clean-100")"
# An eval-only job may be handed a data asset that has no train-clean-100 at
# all (the eval split is the only one it reads), in which case descend_to
# returns the root unchanged -- which is right, but only if test-clean is
# there. Retry on the split this direction actually opens.
if [ "$DIRECTION" = "eval" ] && [ ! -d "$DATA_ROOT/test-clean" ]; then
  DATA_ROOT="$(descend_to "$LIBRISPEECH" "test-clean")"
fi
CKPT_ROOT="$(descend_to "${CHECKPOINTS:-$LIBRISPEECH}" "audioldm")"

# --- Discover the actual checkpoint filenames -------------------------------
# Not hardcoded: the AudioLDM release ships the latent checkpoint under
# several names across versions ("audioldm-s-full", "audioldm-full-s-v2.ckpt")
# and attacks.py only cares about the *directory* it sits in, not its name.
shopt -s nullglob
sgmse_cands=("$CKPT_ROOT"/sgmse/*.ckpt)
aldm_cands=("$CKPT_ROOT"/audioldm/data/checkpoints/audioldm-*)
shopt -u nullglob

SGMSE_CKPT="${SGMSE_CKPT_OVERRIDE:-${sgmse_cands[0]:-$CKPT_ROOT/sgmse/MISSING.ckpt}}"
ALDM_CKPT="${ALDM_CKPT_OVERRIDE:-${aldm_cands[0]:-$CKPT_ROOT/audioldm/data/checkpoints/MISSING}}"

# Absolute, because attacks.py:472 chdirs to the weights root before building
# the model. The config is read at attacks.py:436, i.e. before that chdir, so
# a relative path happens to work today -- an absolute one keeps working if
# that order ever changes.
ALDM_CONFIG="$REPO_ROOT/src/audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml"

# --- Resolve the generator under test (direction=eval) ----------------------
# Empty GENERATOR_NAME means the stock AudioSeal card, which evaluate.py's
# config already defaults to -- that is the baseline arm, not a missing value.
GENERATOR_PATH=""
if [ -n "$GENERATOR_NAME" ]; then
  [ -n "$GENERATORS" ] || { echo "--generator needs --generators (the mounted model asset)" >&2; exit 1; }
  # The asset is a flat folder of .pth files, so locate the file itself rather
  # than using descend_to, which keys off an expected child *directory*.
  GENERATOR_PATH="$(find "$GENERATORS" -name "$GENERATOR_NAME" -type f -print -quit 2>/dev/null || true)"
  if [ -z "$GENERATOR_PATH" ]; then
    echo "generator '$GENERATOR_NAME' not found under $GENERATORS" >&2
    echo "  available: $(find "$GENERATORS" -name '*.pth' -type f -printf '%f ' 2>/dev/null)" >&2
    exit 1
  fi
fi

# --- Writable scratch for caches --------------------------------------------
SCRATCH="${AZ_BATCHAI_JOB_TEMP_DIR:-${TMPDIR:-/tmp}}/audioseal-robust"
export HF_HOME="$SCRATCH/hf"
export TORCH_HOME="$SCRATCH/torch"
export MPLCONFIGDIR="$SCRATCH/mpl"
export XDG_CACHE_HOME="$SCRATCH/cache"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# --- Preflight (fails the job in seconds, not hours) ------------------------
preflight_args=(--direction "$DIRECTION" --data-root "$DATA_ROOT")
if [ "$DIRECTION" = "eval" ]; then
  [ -n "$GENERATOR_PATH" ] && preflight_args+=(--generator "$GENERATOR_PATH")
else
  preflight_args+=(
    --sgmse-checkpoint "$SGMSE_CKPT"
    --audioldm-checkpoint "$ALDM_CKPT"
    --audioldm-config "$ALDM_CONFIG"
  )
fi
PYTHONPATH="$REPO_ROOT/src" python3 azureml/preflight.py "${preflight_args[@]}"

# --- Wire run_diffusion_swap.sh's env-var contract --------------------------
# Each split can be overridden with --train-dir/--valid-dir/--eval-dir. That
# exists so a run can mix data sources -- specifically, training on a
# fixed-duration corpus (every file exactly 10.24s, so AudioLDMAttack's
# window is filled with real audio and never padded or tiled) while
# validation and evaluation stay on the stock dev-clean/test-clean, which
# keeps every reported number comparable with earlier runs. Unset, these
# resolve exactly as before.
export TRAIN_DIR="${TRAIN_DIR_ARG:-$DATA_ROOT/train-clean-100}"
export VALID_DIR="${VALID_DIR_ARG:-$DATA_ROOT/dev-clean}"
export EVAL_DIR="${EVAL_DIR_ARG:-$DATA_ROOT/test-clean}"

# preflight.py only sees --data-root, so an override points somewhere it
# never checked. Validate here instead: a typo or a mount that did not
# materialise should cost seconds, not the image pull plus model load it
# would otherwise take to surface as an empty-dataset error.
for _split in TRAIN VALID EVAL; do
  eval "_dir=\$${_split}_DIR"
  eval "_ovr=\$${_split}_DIR_ARG"
  [ -n "$_ovr" ] || continue
  # A literal "${{inputs.x}}" here means the caller put the override in the
  # job's environment_variables instead of its command: AzureML expands
  # those placeholders only in `command`. Name that explicitly rather than
  # reporting it as a missing directory.
  case "$_dir" in
    *'${{'*)
      echo "--${_split} override was not expanded by AzureML: $_dir" >&2
      echo "  \${{inputs.*}} is substituted in a job's 'command', not in 'environment_variables'." >&2
      exit 1 ;;
  esac
  [ -d "$_dir" ] || { echo "preflight: --${_split} override points at a missing directory: $_dir" >&2; exit 1; }
  # -print -quit stops at the first hit rather than walking a 20k-file tree.
  if [ -z "$(find "$_dir" \( -name '*.flac' -o -name '*.wav' \) -print -quit 2>/dev/null)" ]; then
    echo "preflight: --${_split} override has no .flac/.wav under it: $_dir" >&2
    exit 1
  fi
  echo "override: ${_split}_DIR -> $_dir"
done
export SGMSE_CHECKPOINT="$SGMSE_CKPT"
export AUDIOLDM_CHECKPOINT="$ALDM_CKPT"
export AUDIOLDM_CONFIG="$ALDM_CONFIG"
# Checkpoints go straight to the output mount rather than being copied at the
# end, so an epoch that completed before a crash/timeout is still retrievable.
export CHECKPOINT_DIR="$ARTIFACTS/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

# --- Tracking ---------------------------------------------------------------
# Backend selection, in priority order:
#
#   wandb   only when a key actually reached the container. Without this guard
#           a missing key makes wandb.init() block on an interactive login
#           prompt against a stdin that is never going to answer, i.e. the job
#           hangs until its timeout rather than failing -- strictly worse than
#           logging to stdout.
#   mlflow  the default inside any AzureML job. MLflow's tracking URI is
#           injected by the runtime, so metrics land on the Studio run with no
#           outbound network access at all. This is the ONLY correct choice on
#           a compliant cluster (managed compute): wandb is an external
#           endpoint, and tracking.py's WandbTracker uploads audio samples and
#           figures, not just scalars -- that is training data leaving the
#           compliance boundary, not just telemetry.
#   none    outside AML with no key: console logging.
#
# Set TRACKING_BACKEND explicitly to override the detection.
tracking_args=()
TRACKING_BACKEND="${TRACKING_BACKEND:-}"
if [ -z "$TRACKING_BACKEND" ]; then
  if [ -n "${WANDB_API_KEY:-}" ]; then
    TRACKING_BACKEND=wandb
  elif [ -n "${AZUREML_RUN_ID:-}" ]; then
    TRACKING_BACKEND=mlflow
  else
    TRACKING_BACKEND=none
  fi
fi

case "$TRACKING_BACKEND" in
  wandb)
    if [ -z "${WANDB_API_KEY:-}" ]; then
      echo "TRACKING_BACKEND=wandb but WANDB_API_KEY is unset -- wandb.init() would block on a login prompt" >&2
      exit 1
    fi
    WANDB_PROJECT="${WANDB_PROJECT:-audioseal-robust}"
    tracking_args+=(tracking.backend=wandb tracking.project="$WANDB_PROJECT" tracking.wandb_mode=online)
    # Cross-reference the two systems: AML's run id becomes the wandb run name,
    # so a wandb chart can be traced back to the AML job that produced it.
    if [ -n "${AZUREML_RUN_ID:-}" ]; then
      tracking_args+=(tracking.run_name="$AZUREML_RUN_ID")
    fi
    echo "tracking: wandb project=$WANDB_PROJECT entity=${WANDB_ENTITY:-<default>}"
    ;;
  mlflow)
    if [ -n "${WANDB_API_KEY:-}" ]; then
      echo "note: WANDB_API_KEY is set but TRACKING_BACKEND=mlflow -- not contacting wandb" >&2
    fi
    tracking_args+=(tracking.backend=mlflow)
    if [ -n "${AZUREML_RUN_ID:-}" ]; then
      tracking_args+=(tracking.run_name="$AZUREML_RUN_ID")
    fi
    echo "tracking: mlflow -> the AzureML run (no external egress)"
    ;;
  none)
    tracking_args+=(tracking.backend=none)
    echo "tracking: disabled -- falling back to console logging"
    ;;
  *)
    echo "Unknown TRACKING_BACKEND '$TRACKING_BACKEND' -- expected wandb, mlflow or none" >&2
    exit 1
    ;;
esac

# --- Launcher (single process vs. torchrun DDP) -----------------------------
# NPROC defaults to every visible GPU, so a 4xA100 node uses all four without
# the job YAML having to know the SKU. One process per GPU is the DDP
# convention; data.batch_size stays PER GPU, so the effective batch scales by
# NPROC and optim.lr is NOT rescaled automatically (see docs/MULTI_GPU.md).
if [ -z "${NPROC:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    # `|| true`, not `|| echo 1`: grep -c already prints "0" when it matches
    # nothing and *also* exits 1, so the fallback would append to it and leave
    # NPROC as the two-line string "0\n1" -- which then makes the -gt test below
    # die with "integer expression expected" at exactly the moment someone is
    # trying to work out why they only got one GPU.
    NPROC="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  fi
fi
case "${NPROC:-}" in
  '' | *[!0-9]*) NPROC=1 ;;
esac
if [ "$NPROC" -lt 1 ]; then
  NPROC=1
fi
if [ "$NPROC" -gt 1 ]; then
  # --standalone: single node, rendezvous over localhost. Multi-node would
  # need the AML-provided MASTER_ADDR/NODE_RANK instead.
  export LAUNCHER="torchrun --standalone --nproc_per_node=$NPROC"
else
  export LAUNCHER="python3"
fi

# The same tracking config has to reach evaluate.py, which "$@" does not
# reach; EVAL_EXTRA_ARGS (run_diffusion_swap.sh) is that channel. Anything the
# caller already put there is preserved and wins, since it lands last.
EVAL_EXTRA_ARGS="${tracking_args[*]} ${EVAL_EXTRA_ARGS:-}"
export EVAL_EXTRA_ARGS

# evaluate.py writes plots to ./eval_outputs relative to the CWD, and
# run_diffusion_swap.sh's own extra args go to train.py only -- so
# eval.output_dir can't be overridden from here. Copy on EXIT instead, which
# also catches the failure path.
collect() {
  local rc=$?
  if [ -d "$REPO_ROOT/eval_outputs" ]; then
    mkdir -p "$ARTIFACTS/eval_outputs"
    cp -r "$REPO_ROOT/eval_outputs/." "$ARTIFACTS/eval_outputs/" 2>/dev/null || true
    echo "collected eval_outputs/ -> \$ARTIFACTS/eval_outputs (exit=$rc)"
  fi
  return $rc
}
trap collect EXIT

echo "=== resolved paths ==========================================="
echo "direction            : $DIRECTION"
echo "TRAIN_DIR            : $TRAIN_DIR"
echo "VALID_DIR            : $VALID_DIR"
echo "EVAL_DIR             : $EVAL_DIR"
echo "SGMSE_CHECKPOINT     : $SGMSE_CHECKPOINT"
echo "AUDIOLDM_CHECKPOINT  : $AUDIOLDM_CHECKPOINT"
echo "AUDIOLDM_CONFIG      : $AUDIOLDM_CONFIG"
echo "CHECKPOINT_DIR       : $CHECKPOINT_DIR"
echo "EVAL_EXTRA_ARGS      : $EVAL_EXTRA_ARGS"
echo "tracking backend     : $TRACKING_BACKEND"
echo "launcher             : $LAUNCHER (NPROC=$NPROC)"
echo "train.py overrides   : $* ${tracking_args[*]}"
echo "=============================================================="

# --- direction=eval: no training, evaluate an existing generator ------------
# run_diffusion_swap.sh always trains first and then evaluates the checkpoint
# it just produced, so it cannot score an already-trained generator. Call
# evaluate.py directly instead, reusing everything above (mount resolution,
# preflight, caches, tracking, the collect trap).
if [ "$DIRECTION" = "eval" ]; then
  # $LAUNCHER carries arguments ("torchrun --standalone --nproc_per_node=4"),
  # so it has to word-split -- same treatment run_diffusion_swap.sh gives it.
  read -ra launcher_eval <<< "${LAUNCHER:-python3}"
  eval_args=(
    eval_dir="$EVAL_DIR"
    device=cuda
  )
  [ -n "$LABEL_ARG" ] && eval_args+=(label="$LABEL_ARG")
  # Unset = evaluate.py's default generator_checkpoint, the stock
  # audioseal_wm_16bits card. That IS the baseline arm.
  [ -n "$GENERATOR_PATH" ] && eval_args+=(generator_checkpoint="$GENERATOR_PATH")

  echo "generator under test : ${GENERATOR_PATH:-<stock audioseal_wm_16bits>}"
  echo "evaluate.py overrides: ${eval_args[*]} ${tracking_args[*]} $*"
  echo "=============================================================="

  # "$@" last so a config's overrides beat these defaults.
  MPLBACKEND=Agg PYTHONPATH=src "${launcher_eval[@]}" -m audioseal_robust.evaluate \
    "${eval_args[@]}" \
    "${tracking_args[@]}" \
    "$@"
  echo "Done -- see \$ARTIFACTS/eval_outputs for metrics, plots and per-sample rows."
  exit 0
fi

# `bash`, not ./ -- run_diffusion_swap.sh is mode 100644 in git (unlike
# run_smoke_eval.sh at 100755), so it is not executable in the snapshot.
# tracking_args go last so they beat anything the job YAML passed.
bash run_diffusion_swap.sh "$DIRECTION" "$@" "${tracking_args[@]}"
