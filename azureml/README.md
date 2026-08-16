# Running the diffusion-swap experiment on Azure ML

`run_diffusion_swap.sh` remains the local entry point. The files in this
directory package the same training and evaluation commands as Azure ML jobs.
They intentionally contain no subscription, workspace, storage-account,
identity, or tenant-specific compute identifiers.

## Prerequisites

Install the Azure CLI ML extension and select a workspace:

```bash
az login
az extension add -n ml
az configure --defaults group=<resource-group> workspace=<workspace-name>
```

The SDK submitter also accepts workspace coordinates through environment
variables:

```bash
export AZUREML_SUBSCRIPTION_ID=<subscription-id>
export AZUREML_RESOURCE_GROUP=<resource-group>
export AZUREML_WORKSPACE_NAME=<workspace-name>
```

## Register assets and the environment

The job files expect these workspace assets:

| Asset | Definition |
|---|---|
| `librispeech-asr:1` | `assets/librispeech.yaml` |
| `psiml-diffusion-checkpoints:1` | `assets/diffusion_checkpoints.yaml` |
| `psiml-finetuned-generators:1` | `assets/finetuned_generators.yaml` |
| `librispeech-train-fixed-10s:1` | `assets/librispeech_train_fixed.yaml` |

Create only the assets needed by the job you plan to run:

```bash
az ml data create -f azureml/assets/librispeech.yaml
az ml model create -f azureml/assets/diffusion_checkpoints.yaml
az ml model create -f azureml/assets/finetuned_generators.yaml
az ml data create -f azureml/assets/librispeech_train_fixed.yaml
az ml environment create -f azureml/environment.yaml
```

The large checkpoint asset is normally staged in the workspace blob datastore
before registration. Its YAML header documents the required directory layout.

## Submit training jobs

The static YAML files use example compute names. Override them with a compute
target from your workspace:

```bash
az ml job create -f azureml/jobs/smoke_audioldm.yaml \
  --set compute=azureml:<single-gpu-compute>

az ml job create -f azureml/jobs/train_audioldm.yaml \
  --set compute=azureml:<single-gpu-compute>

az ml job create -f azureml/jobs/train_sgmse.yaml \
  --set compute=azureml:<single-gpu-compute>
```

Run the smoke job before a full AudioLDM training job. It exercises image
startup, asset access, CUDA, one real optimization path, checkpoint writing,
and the reduced evaluation stage.

For config-driven submissions and parameter sweeps:

```bash
pip install azure-ai-ml azure-identity pyyaml

python azureml/aml_submit.py \
  --config azureml/configs/audioldm.yaml \
  --compute <single-gpu-compute> \
  --dry-run

python azureml/aml_submit.py \
  --config azureml/configs/audioldm.yaml \
  --compute <single-gpu-compute> \
  --stream
```

`--set KEY=VALUE` adds an OmegaConf override such as `--set epochs=10`.

## Multi-GPU training

`smoke_audioldm_4gpu.yaml` and `train_audioldm_4gpu.yaml` target one compute
node with multiple visible GPUs. `aml_run.sh` reads the GPU count and launches
one `torchrun` worker per GPU.

```bash
az ml job create -f azureml/jobs/smoke_audioldm_4gpu.yaml \
  --set compute=azureml:<multi-gpu-compute>

az ml job create -f azureml/jobs/train_audioldm_4gpu.yaml \
  --set compute=azureml:<multi-gpu-compute>
```

`data.batch_size` is per GPU. Evaluation batch counts are global totals and
must be at least the number of workers so every rank receives work.

## Evaluation jobs

The `eval_hsj_*_cluster.yaml` jobs evaluate the stock and fine-tuned generators
with HopSkipJump. The `eval_fb_*_cluster.yaml` jobs run the fixed-budget attack
suite. Override each file's example compute name when submitting:

```bash
az ml job create -f azureml/jobs/eval_hsj_baseline_cluster.yaml \
  --set compute=azureml:<single-gpu-compute>

az ml job create -f azureml/jobs/eval_fb_baseline_cluster.yaml \
  --set compute=azureml:<single-gpu-compute>
```

The fine-tuned arms mount `psiml-finetuned-generators:1`; the baseline arms use
the stock AudioSeal card directly.

## Tracking

Tracking is selected by `azureml/aml_run.sh`:

| Condition | Backend |
|---|---|
| `TRACKING_BACKEND` is set | Explicit value |
| `WANDB_API_KEY` is set | Weights & Biases |
| Running inside Azure ML | MLflow attached to the job |
| Otherwise | Console output |

Use `TRACKING_BACKEND=mlflow` when experiment data must remain in the workspace.
The container installs `azureml-mlflow` so MLflow can resolve the runtime's
`azureml://` tracking URI.

## Outputs

Jobs write checkpoints, evaluation CSV files, plots, and optional row-level
audio artifacts below the mounted `artifacts` output. The output is configured
as `rw_mount`, so completed checkpoints survive a later timeout or failure.
