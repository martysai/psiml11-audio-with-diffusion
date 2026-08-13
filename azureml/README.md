# Running the diffusion-swap experiment on AzureML

`run_diffusion_swap.sh` (repo root) is the entry point, unchanged. Everything
here just feeds it AzureML mount paths and guards the failure modes that are
expensive on rented GPU time.

```
azureml/
  environment.yaml              az ml environment create -f ...
  docker/Dockerfile             the image (build context = repo root)
  assets/librispeech.yaml       az ml data create   -f ...   ~6.8 GB
  assets/diffusion_checkpoints.yaml
                                az ml model create  -f ...   ~10.9 GB
  jobs/smoke_audioldm.yaml      az ml job create    -f ...   stage 1, minutes
  jobs/train_audioldm.yaml      az ml job create    -f ...   stage 2, hours
  jobs/train_sgmse.yaml         the mirror direction
  configs/*.yaml                inputs for aml_submit.py
  aml_submit.py                 config-driven submitter (SDK)
  aml_run.sh                    in-job wrapper (runs on the compute)
  preflight.py                  fail-fast checks (runs on the compute)
```

## One-time setup

The workspace this was set up against:

| | |
|---|---|
| subscription | `<redacted-subscription>` (Azure ML) |
| resource group | `gpu-resource-group` |
| workspace | `gpu-workspace` (<region>) |
| compute | `gpu-cluster` — `STANDARD_NC24ADS_A100_V4`, 1× A100 80GB |

`gpu-cluster`, not `gpu-cluster`/`gpu-cluster`: nothing in this project is distributed, so the
larger clusters would leave GPUs idle.

```bash
az login
az extension add -n ml
az account set --subscription <redacted-subscription>
az configure --defaults group=gpu-resource-group workspace=gpu-workspace

az ml data create -f azureml/assets/librispeech.yaml
az ml model create -f azureml/assets/diffusion_checkpoints.yaml
az ml environment create -f azureml/environment.yaml
```

Both asset YAMLs point at `azureml://datastores/workspaceblobstore/paths/psiml-assets/...`,
so `az ml data create` / `az ml model create` just register a pointer and return
instantly. The bytes get there separately, with AzCopy — see the header comment
in each asset YAML for the exact commands and, importantly, the concurrency
throttle they need.

Do not hand ~18 GB to `az ml data create` from a local `path:` and expect it to
finish. The AML uploader has **no resume**: a single connection reset restarts
from zero, so on anything but a datacenter link it can never converge. AzCopy
retries, resumes, and skips already-uploaded blobs with `--overwrite=false`.

## Submitting

**Run the smoke job first.** It is the same code, assets, compute and direction
as the real run with the step counts turned down — a few minutes of A100 to
find out whether the image builds, the mounts resolve, the AudioLDM tree is
intact and wandb is reachable, instead of discovering it hours in.

```bash
az ml job create -f azureml/jobs/smoke_audioldm.yaml \
  --set environment_variables.WANDB_API_KEY=$WANDB_API_KEY \
  --set environment_variables.WANDB_PROJECT=audioseal-robust \
  --set environment_variables.WANDB_ENTITY=<your-entity>

# only once that is green:
az ml job create -f azureml/jobs/train_audioldm.yaml --set ...same wandb flags...
az ml job create -f azureml/jobs/train_sgmse.yaml    --set ...same wandb flags...
```

or the config-driven submitter, which is the better fit once you start sweeping
overrides (`pip install azure-ai-ml azure-identity pyyaml` first):

```bash
python azureml/aml_submit.py --config azureml/configs/sgmse.yaml --dry-run
python azureml/aml_submit.py --config azureml/configs/sgmse.yaml --stream
python azureml/aml_submit.py --config azureml/configs/sgmse.yaml --set epochs=10
```

Run **both** directions. Each trains against one diffusion attack and holds the
other out; the comparison between them is the actual result (see
`src/audioseal_robust/config/recipes.yaml`).

### Sizing the full run

`train_audioldm.yaml`'s step budget is a placeholder. Take the seconds/step the
smoke job reports and set `epochs`/`updates_per_epoch` from it before submitting
stage 2 — see that file's BUDGET comment for why the config defaults (100k
steps) are not survivable inside a 12h limit, and why the 100h split behaves
differently from the 10h one here.

### Experiment tracking

Tracking is wired to an external wandb project and is driven entirely by env
vars — the job YAMLs deliberately set no `tracking.*` override, so `aml_run.sh`
is the single place that decides it:

| var | effect |
|---|---|
| `WANDB_API_KEY` | absent ⇒ falls back to console logging |
| `WANDB_PROJECT` | defaults to `audioseal-robust` |
| `WANDB_ENTITY` | read natively by the wandb SDK; needed for team projects |

The key is passed at submit time and never committed. Note it *is* then visible
in the job definition to anyone with workspace access, so rotate it if that
matters.

Two non-obvious details this handles for you:

- The AML run id becomes the wandb run name, so a chart traces back to the job.
- The same tracking config is forced onto **evaluate.py** too, via
  `EVAL_EXTRA_ARGS`. `default_eval.yaml` defaults to a *different* project
  (`audioseal-robust-eval`), and `run_diffusion_swap.sh` forwards `"$@"` to
  train.py only — so without that channel every experiment would split across
  two projects, with the eval half (the actual result) landing in the one you
  weren't watching.

## Outputs

Under the job's `artifacts` output:

- `checkpoints/generator_epoch{N}.pth` — written straight to the mount as each
  epoch finishes, so a timeout or crash still leaves the completed epochs.
- `eval_outputs/swap_{direction}_confusion.png`, `..._robustness_curve.png` —
  copied on exit, including the failure path.

Metrics go to your wandb project (see "Experiment tracking" above), or to stdout
and thus the run logs if no `WANDB_API_KEY` was passed.

## Things that will bite you

**The checkpoint tree layout is load-bearing.** `attacks.py:418-426` rejects any
AudioLDM checkpoint not at `<weights_root>/data/checkpoints/<file>`, then chdirs
to `<weights_root>` so the relative paths inside `audioldm_original.yaml` (its
VAE at line 56, CLAP at line 138) resolve. Register the model asset with
`audioldm/data/checkpoints/` intact — do not flatten it. `preflight.py` checks
this before anything expensive starts.

**`device=cuda` fails silently.** `device.py:36-40` only logs a warning and
returns CPU when CUDA isn't available, and `run_diffusion_swap.sh` passes
`device=cuda` unconditionally. A CPU-only compute target therefore produces a
job that trains, logs and checkpoints normally at ~1% speed. `preflight.py`
turns that into an immediate hard failure.

**The default budget is not a day of A100.** `config.py` defaults to
`epochs=100 x updates_per_epoch=1000` = 100k steps, each backpropping through a
full diffusion reverse loop. Note `updates_per_epoch` is a *cap*, not a target —
the inner loop also stops when the dataloader is exhausted. On the 10h subset
that happens first (~167 steps/epoch), so the run self-limits; on the full 100h
split it does not, and the cap is the only thing bounding the job. Raise
deliberately, and size it from the smoke run.

**wandb blocks rather than fails.** `wandb.init()` with no key prompts for an
interactive login, against a stdin that will never answer — the job hangs to its
timeout instead of erroring out. `aml_run.sh` guards on `WANDB_API_KEY` being
non-empty for exactly this reason. Don't remove that check.

**MLflow.** `tracking.py:84-85` calls `set_experiment()` then `start_run()`,
while AML has already started a run for the job. Untested here; the wandb path
is the supported one.

**`eval_every` needs `data.valid_dir`.** `train.py:303-305` raises if you set one
without the other. `aml_run.sh` always sets `VALID_DIR`, so overriding
`eval_every` is safe here — but note `eval_every` defaults to `0`, so setting
only `valid_dir` silently does nothing.

**Held-out attacks degrade, they don't fail.** `evaluate.py:120` catches
`NotImplementedError`/`FileNotFoundError`/`ModuleNotFoundError` at attack
construction and reports "skipped". Useful, but it also means a missing
checkpoint or a broken `audiocraft` install shows up as a quietly absent column
rather than an error — check the `plan:` line in the eval logs for how many
attacks actually ran.
