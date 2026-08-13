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
  jobs/train_sgmse.yaml         az ml job create    -f ...
  jobs/train_diff_erase.yaml
  configs/*.yaml                inputs for aml_submit.py
  aml_submit.py                 config-driven submitter (SDK)
  aml_run.sh                    in-job wrapper (runs on the compute)
  preflight.py                  fail-fast checks (runs on the compute)
```

## One-time setup

The workspace this was set up against:

| | |
|---|---|
| subscription | `5c9e4789-4852-4ffe-8551-d682affcbd74` (ASG Azure ML) |
| resource group | `playground-rg` |
| workspace | `as-playground-w3-ws` (westus3) |
| compute | `a100x1` — `STANDARD_NC24ADS_A100_V4`, 1× A100 80GB |

`a100x1`, not `a100x2`/`a100x4`: nothing in this project is distributed, so the
larger clusters would leave GPUs idle.

```bash
az login
az extension add -n ml
az account set --subscription 5c9e4789-4852-4ffe-8551-d682affcbd74
az configure --defaults group=playground-rg workspace=as-playground-w3-ws

az ml data create -f azureml/assets/librispeech.yaml
az ml model create -f azureml/assets/diffusion_checkpoints.yaml
az ml environment create -f azureml/environment.yaml
```

Both asset YAMLs carry a local `path:` for the machine the artifacts were
downloaded to. Uploading from elsewhere:

```bash
az ml data create -f azureml/assets/librispeech.yaml --path /some/other/LibriSpeech
```

## Submitting

Either the static YAML:

```bash
az ml job create -f azureml/jobs/train_sgmse.yaml
az ml job create -f azureml/jobs/train_diff_erase.yaml
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

## Outputs

Under the job's `artifacts` output:

- `checkpoints/generator_epoch{N}.pth` — written straight to the mount as each
  epoch finishes, so a timeout or crash still leaves the completed epochs.
- `eval_outputs/swap_{direction}_confusion.png`, `..._robustness_curve.png` —
  copied on exit, including the failure path.

Metrics go to stdout (`tracking.backend=none`) and so into the run logs.

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
full diffusion reverse loop. The shipped configs use ~600 steps. Raise
deliberately.

**MLflow.** `tracking.py:84-85` calls `set_experiment()` then `start_run()`,
while AML has already started a run for the job. The configs use
`tracking.backend=none` until that's been confirmed to coexist.

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
