# Multi-GPU training and evaluation

`src/audioseal_robust` runs on any number of GPUs through PyTorch
DistributedDataParallel (DDP), launched with `torchrun`. Nothing about the
single-GPU workflow changed: `python -m audioseal_robust.train ...` still
works exactly as before, and every distributed helper is a no-op at
world_size=1.

## Running

```bash
# training, 4 GPUs on one node
torchrun --standalone --nproc_per_node=4 -m audioseal_robust.train \
    data.train_dir=/data/datasets/LibriSpeech/train-clean-100 \
    data.valid_dir=/data/datasets/LibriSpeech/dev-clean \
    eval_every=200

# evaluation, 4 GPUs
torchrun --standalone --nproc_per_node=4 -m audioseal_robust.evaluate \
    eval_dir=/data/datasets/LibriSpeech/test-clean label=baseline

# throughput / wiring check on the new hardware (run this first)
torchrun --standalone --nproc_per_node=4 -m audioseal_robust.sanity_check --pretrained
```

Ready-made launchers: `run_train_4gpu.sh`, `run_full_eval_4gpu.sh`.

`--nproc_per_node` must equal the number of GPUs you want to use, one process
per GPU. Restrict which cards are used with `CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 ...`.

## What scales and what doesn't

This is the part worth reading before comparing any number against a
single-A100 baseline.

| Config field | Per GPU or global? | Effect of going 1 -> 4 GPUs |
| --- | --- | --- |
| `data.batch_size` | **per GPU** | effective batch is 4x larger |
| `batch_size` (eval) | **per GPU** | 4x more audio in flight at once |
| `data.num_workers`, `num_workers` | **per GPU** | 4x as many loader processes total |
| `n_eval_batches`, `n_curve_batches` | **global** | unchanged amount of audio, ~4x faster |
| `updates_per_epoch`, `epochs` | global (optimizer steps) | unchanged; each step sees 4x the data |

So a 4-GPU **training** run is not the same experiment as the 1-GPU run of
the same config -- it takes 4x as much data per optimizer step. A 4-GPU
**evaluation** run *is* the same experiment, deliberately: eval numbers have
to stay comparable to the baselines already recorded, so the batch counts are
split across ranks rather than multiplied by them.

### Epochs consume more than one pass over the data

`updates_per_epoch` is the real end of an epoch: an epoch runs exactly that
many optimizer steps, on any number of GPUs. Since each step now consumes
`world_size` batches, an epoch gets through proportionally more data --
`EpochBatchIterator` (in `train.py`) restarts the loader with a fresh shuffle
whenever it runs out.

This matters because the alternative is worse and silent. A
`DistributedSampler` hands each rank `len(dataset) / world_size` examples, so
the per-rank loader shrinks by the GPU count: LibriSpeech train-clean-100 at
`batch_size=16` gives 1783 batches on one GPU but 445 on each of four. Ending
the epoch when the loader runs dry would mean the same config runs 445-step
epochs instead of 1000-step ones on 4 GPUs -- 2.25x fewer optimizer steps
over the run, and checkpoints written twice as often -- with nothing in the
config or the logs saying so.

If you would rather have "one epoch = one pass over the data", set
`updates_per_epoch` to the per-rank loader length yourself
(`len(dataset) / world_size / batch_size`). The startup log prints how many
passes an epoch will make whenever it is more than one.

### Learning rate

`optim.lr` is **not** rescaled automatically for the larger effective batch.
Scaling it is a modelling decision, and silently changing the learning rate
based on how many GPUs happened to be free would make runs impossible to
compare. If you want the usual linear-scaling heuristic, ask for it
explicitly:

```bash
torchrun --standalone --nproc_per_node=4 -m audioseal_robust.train \
    data.train_dir=... optim.lr=2e-4     # 4 x the 5e-5 default
```

Note that this fine-tunes a pretrained generator with Adam at a small LR, the
regime where linear scaling is least reliable -- treat `sqrt(4)=2x` and `4x`
as things to try, not as a correction that must be applied.

## What the implementation actually does

Almost all of it lives in `src/audioseal_robust/distributed.py`. Four things
were not mechanical:

**1. DDP wraps `WatermarkEmbedder`, not the generator.** DDP synchronizes
gradients by hooking the wrapped module's `forward()`. This training loop
never calls the generator's `forward` -- it calls `get_watermark(x,
message=...)`. `DDP(generator).get_watermark(...)` resolves straight through
to the underlying module and skips DDP completely, so each rank would train
its own private copy with no allreduce, and the checkpoint saved by rank 0
would have seen a quarter of the data. This fails *silently*: the loss goes
down normally on every rank. `WatermarkEmbedder` (in `train.py`) exposes
`get_watermark` as `forward` so the call the loop makes is the call DDP
intercepts. `tests/test_distributed.py` verifies this on a real 2-process
run, with a negative control that asserts the bypassing version does diverge.

**2. Attack sampling is rank-synchronized.** `SampledReconstructionAttack`
draws a fresh branch every step, and the branches differ enormously in cost
(identity is free, sgmse is a 30-step diffusion sampler). Since DDP
synchronizes at every backward, unsynchronized draws make each step cost the
most expensive branch *any* rank drew -- with 4 ranks and a small sgmse
weight, that is nearly every step. Every rank builds a `random.Random` from
the same `cfg.seed` (`distributed.attack_sampling_rng`), so the branch
sequence matches without needing a collective. Everything else that is
sampled per step -- messages, target SNR, dataset crops, each attack's own
t* -- is deliberately rank-*dependent* (`distributed.seed_everything` seeds
with `seed + rank`), because identical draws on 4 ranks would give a 4x
bigger batch that is only 1x more diverse.

**3. TPR@FPR is computed on pooled scores, not averaged per rank.** Losses
are averages over examples, so averaging per-rank averages is exact
(`all_reduce_mean`). TPR@FPR and the confusion matrix are not: they are
measured against a threshold read off a *quantile* of the negative-score
distribution (`metrics._threshold_at_fpr`). A per-rank threshold estimated
from a quarter of the negatives against a 1% FPR budget is a different,
noisier operating point, and no averaging afterwards recovers the global one.
So `evaluate_attack` gathers the raw scores (`all_gather_scores`) and
computes the metric once over all of them. Peak memory reduces by **max**
across ranks, not mean -- the worst card is the one that OOMs.

**4. Skips have to be agreed on before any collective.** A stubbed attack
raises `NotImplementedError` and gets reported as "skipped" rather than
killing the run. Under DDP, one rank skipping while another proceeds means
the second blocks in an all_gather that the first will never reach -- a hang
that ends only at the 30-minute process-group timeout, with no useful error.
Both the construction-time skips (`_union_skipped`) and the forward-time ones
(inside `evaluate_attack`) are turned into a world-wide decision first.

Side effects are rank-0 only: experiment tracking (otherwise 4 MLflow runs
per training run, 3 of them holding a single shard's metrics), checkpoint
writes, plot files, and the stdout summary. Checkpoints are saved from the
unwrapped generator, so their `state_dict` keys carry no `module.` or
`generator.` prefix and load into `evaluate.py` unchanged -- a checkpoint
from a 4-GPU run is interchangeable with one from a 1-GPU run.

## Operational notes

- **Log lines are tagged with their rank** and non-zero ranks are quieted to
  WARNING, since all four processes share one terminal. Warnings and errors
  from every rank still come through.
- **`drop_last=True` everywhere** is load-bearing, not tidiness: it keeps
  every rank running the same number of steps on the same number of examples.
  A rank that runs out of batches early stops participating in the gradient
  allreduce and hangs the others.
- **`num_workers` is per rank.** A 4-GPU run with `num_workers=4` spawns 16
  loader processes; turn it down first if the box runs short on CPU or
  `/dev/shm`.
- **Too little data for the world size** raises immediately with a clear
  message rather than silently giving some rank zero batches.
- **`n_eval_batches` below the GPU count** leaves the high ranks idle (they
  still take part in every collective, so the result is correct -- just
  wasteful); a warning says so. This is worth knowing before running
  `run_smoke_eval.sh`-style configs (`n_eval_batches=2`) across 4 GPUs.
- **NCCL** is selected automatically on CUDA (gloo on CPU). If ranks hang at
  startup, `NCCL_DEBUG=INFO` and `export NCCL_P2P_DISABLE=1` are the usual
  first things to try.

## Tests

```bash
pytest tests/test_distributed.py                  # includes real 2-process runs
pytest tests/test_distributed.py -m "not slow"    # single-process logic only
```

The slow tests launch real multi-process workers over gloo on CPU
(`tests/ddp_worker_script.py`). They run anywhere, no GPU needed, and cover
the failure that no single-process test can see: gradients not actually being
synchronized.
