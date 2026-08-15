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
| subscription | Azure ML — resolved at submit time, never committed (this repo is public) |
| resource group | `gpu-resource-group` |
| workspace | `gpu-workspace` (<region>) |
| compute | `gpu-cluster` — `STANDARD_NC24ADS_A100_V4`, 1× A100 80GB |

`gpu-cluster`, not `gpu-cluster`/`gpu-cluster`: nothing in this project is distributed, so the
larger clusters would leave GPUs idle.

```bash
az login
az extension add -n ml
az account set --subscription <your-subscription-id>   # Azure ML
az configure --defaults group=gpu-resource-group workspace=gpu-workspace

# Everything below resolves the subscription from the CLI session above, so no
# subscription id is committed anywhere in this repo. To pin it explicitly
# instead, export AZUREML_SUBSCRIPTION_ID.

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

Tracking is driven entirely by env vars — the job YAMLs deliberately set no
`tracking.*` override, so `aml_run.sh` is the single place that decides it.
Resolution order is **wandb → mlflow → none**:

| condition | backend |
|---|---|
| `WANDB_API_KEY` is set | `wandb` |
| no key, but inside an AzureML job (`AZUREML_RUN_ID` set) | `mlflow` |
| neither | `none` (console logging) |

Set `TRACKING_BACKEND` explicitly to override. Asking for `wandb` without a key
is a fast, explicit failure rather than `wandb.init()` blocking forever on a
login prompt against a stdin that will never answer.

| var | effect |
|---|---|
| `WANDB_PROJECT` | defaults to `audioseal-robust` |
| `WANDB_ENTITY` | read natively by the wandb SDK; needed for team projects |

The key is passed at submit time and never committed. Note it *is* then visible
in the job definition to anyone with workspace access, so rotate it if that
matters.

**On managed compute, wandb is not an option.** managed compute clusters are
compliant compute, and wandb is an external endpoint — `tracking.py`'s
`WandbTracker` uploads audio samples and figures, not just scalars, so using it
there would move training data out of the compliance boundary. The managed-cluster job
YAML therefore sets no key at all and lands on `mlflow`, whose tracking URI the
AzureML runtime injects; metrics go to the Studio run with no outbound network.
Use wandb for playground runs only.

The `mlflow` path needs the **`azureml-mlflow`** plugin in the image, and it is
easy to miss: AzureML injects a tracking URI of the form
`azureml://<region>.api.azureml.ms/mlflow/v1.0/...`, and that `azureml://`
scheme is registered by the *plugin*, not by mlflow itself. Without it every
job dies at the first `mlflow.set_experiment()` with

```
UnsupportedModelRegistryStoreURIException: ... got unsupported URI 'azureml://...'
```

which names the *model registry* even though it is the tracking store that
failed to resolve. `import mlflow` succeeds either way, so a build check has to
perform the scheme lookup itself — the Dockerfile does. The plugin also caps
`mlflow<=3.13.0`, so mlflow is pinned in the image constraints; left open it
resolves to 3.15+ and `pip check` fails.

Tracking is never load-bearing: if the backend cannot be constructed, or starts
failing mid-run, `tracking.py` falls back to `ConsoleTracker` and training
continues (metrics keep going to the job log). A telemetry outage should not
kill a 24-hour GPU job.

Two non-obvious details this handles for you:

- The AML run id becomes the wandb/mlflow run name, so a chart traces back to
  the job.
- The same tracking config is forced onto **evaluate.py** too, via
  `EVAL_EXTRA_ARGS`. `default_eval.yaml` defaults to a *different* project
  (`audioseal-robust-eval`), and `run_diffusion_swap.sh` forwards `"$@"` to
  train.py only — so without that channel every experiment would split across
  two projects, with the eval half (the actual result) landing in the one you
  weren't watching.

### Running on managed compute (4× A100)

the workspace's `gpu-cluster` pool is shared and frequently saturated — the stage-1
smoke sat `Queued` for over an hour behind other users, with all 27 nodes busy
and no spot tier to fall back to. `as-managed-cluster-w3-vc` (<region>, same region as
the assets) had 32 A100 cores idle. A single A100 was also already hitting OOM
on this workload, so the managed-cluster path runs 4× A100 under `torchrun`/DDP.

```powershell
$rg = "gpu-resource-group"; $ws = "as-managed-cluster-w3-ws"

# one-time: point the compute workspace at the assets already staged in the workspace's blob
az ml datastore create -f azureml/datastores/project-assets.yaml `
  --resource-group $rg --workspace-name $ws

$ds = "azureml://datastores/project-assets/paths/psiml-assets"
az ml data create -f azureml/assets/librispeech.yaml `
  --set path="$ds/LibriSpeech" --resource-group $rg --workspace-name $ws
az ml data create -f azureml/assets/diffusion_checkpoints_cluster.yaml `
  --resource-group $rg --workspace-name $ws
az ml environment create -f azureml/environment.yaml `
  --resource-group $rg --workspace-name $ws

# then, per run -- via the wrapper, which fills in the subscription id the
# cluster YAMLs carry as a ${AZUREML_SUBSCRIPTION_ID} placeholder (a virtual
# cluster is targeted by full ARM id, and this repo is public). It already
# passes --resource-group/--workspace-name above; everything after the YAML
# path is forwarded to `az ml job create`.
python azureml/submit_job.py azureml/jobs/smoke_audioldm_cluster.yaml `
  --name "smoke-audioldm-cluster-$(Get-Date -Format 'MMdd-HHmmss')"

# the three HopSkipJump eval arms submit the same way:
python azureml/submit_job.py azureml/jobs/eval_hsj_baseline_cluster.yaml
python azureml/submit_job.py azureml/jobs/eval_hsj_sgmse_epoch3_cluster.yaml
python azureml/submit_job.py azureml/jobs/eval_hsj_audioldm_epoch18_cluster.yaml
```

Three things about this that are not obvious:

- **`job_tier` must be `standard`, never `premium`.** Premium costs
  materially more and is not ours to spend on this workload, however long
  standard waits for a node. The working managed-compute job this spec's shape was
  copied from happened to use premium; that was carried over by mistake once
  and cancelled. If you copy an existing job's `resources`/`queue_settings`
  block again, re-check this field.
- **The 18.8 GB is not copied.** The job identity `job-identity` already
  holds *Storage Blob Data Contributor* on `storage-account`, so a
  credential-less datastore reads the existing blobs in place.
- **The checkpoints register as a `uri_folder`, not a `custom_model`.**
  Registering a *model* asset makes the **workspace MSI** authenticate against
  the backing storage, and managed-cluster's MSI has no role on the playground
  account (`Datastore has no credentials, and workspace msi failed to
  authenticate storage account`). Granting one needs
  `roleAssignments/write` on `gpu-resource-group`. Data assets register by reference
  and skip that check; the mount at job time is identical, since that uses the
  job identity. Hence the separate
  `assets/diffusion_checkpoints_cluster.yaml`.
- **`n_eval_batches` and `n_curve_batches` must be ≥ the GPU count.** They are
  *global* totals that `evaluate.py` splits with `shard_size()`, so on 4 ranks
  the playground values (2 and 1) would leave ranks 2–3 with an empty shard and
  fail the run.

`aml_run.sh` reads the GPU count off `nvidia-smi` and sets
`torchrun --nproc_per_node=<n>` itself, so the SKU in `resources.instance_type`
is the only thing to change to scale. `data.batch_size` stays **per GPU**, so
the effective batch scales with the rank count and `optim.lr` is *not* rescaled
automatically — see `docs/MULTI_GPU.md`.


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

**MLflow.** `tracking.py` calls `set_experiment()` then `start_run()` while AML
has already started a run for the job. This is now the *default* path on
managed compute and is exercised there — `azureml-mlflow` reuses the ambient run
rather than creating a competing one, so metrics land on the job you submitted.
It needs that plugin in the image; see "Experiment tracking" above.

**Registering an environment does not build it.** `az ml environment create`
returns in well under a minute — it only registers the asset. The Docker build
happens on the *first job that uses that version*, takes ~20 minutes, and is
logged to that job's `azureml-logs/20_image_build_log.txt`, not to any
environment view. So a job that looks stuck in `Preparing` for 20 minutes on a
freshly registered version is usually just building. Two consequences: a build
failure surfaces as a *job* failure, and `environment: azureml:...@latest`
resolves at **submit** time — submit before registering and you silently pin the
old version.

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
