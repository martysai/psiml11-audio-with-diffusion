# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-GPU (DDP) tests for src/audioseal_robust.

Split in two halves:

  - Pure-logic tests (no process group): sharding arithmetic, DDP/wrapper
    unwrapping, the rank-shared attack RNG, rank-local device pinning, and
    the "every helper is a no-op at world_size=1" guarantee that lets the
    single-GPU path keep working unchanged.
  - Real two-process runs under `torch.distributed.run` over gloo (see
    ddp_worker_script.py). These cost a few seconds each but are the only
    way to catch the failure mode that matters -- gradients not actually
    being synchronized -- which is invisible from a single process because
    the loss goes down either way.

The two-process tests use gloo on CPU rather than NCCL on GPU so they run in
CI and on a laptop; what they exercise (wrapping, call routing, collectives,
checkpoint key names) is backend-independent.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from audioseal_robust.attacks import IdentityAttack, SampledReconstructionAttack
from audioseal_robust.device import resolve_device
from audioseal_robust.distributed import (
    DistEnv,
    all_gather_scores,
    all_gather_values,
    all_reduce_max,
    all_reduce_mean,
    attack_sampling_rng,
    shard_size,
    unwrap_module,
    wrap_ddp,
)
from audioseal_robust.train import EpochBatchIterator, WatermarkEmbedder, embed_watermark, unwrap_generator

sys.path.insert(0, str(Path(__file__).parent))
from ddp_worker_script import NBITS, tiny_generator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).parent / "ddp_worker_script.py"


# --------------------------------------------------------------------------
# Single-process logic
# --------------------------------------------------------------------------


def test_dist_env_defaults_to_single_process():
    env = DistEnv()
    assert not env.is_distributed
    assert env.is_main
    assert env.world_size == 1


def test_helpers_are_noops_when_not_distributed():
    """The single-GPU path must keep working with no process group at all --
    these would otherwise raise "default process group has not been
    initialized"."""
    env = DistEnv()
    device = torch.device("cpu")
    scores = torch.tensor([0.1, 0.2, 0.3])
    assert torch.equal(all_gather_scores(scores, env), scores)
    assert all_gather_values([1.0, 2.0], env) == [1.0, 2.0]
    assert all_reduce_mean({"loss": 3.0}, env, device) == {"loss": 3.0}
    assert all_reduce_max({"peak": 3.0}, env, device) == {"peak": 3.0}
    assert shard_size(20, env) == 20
    module = torch.nn.Linear(2, 2)
    assert wrap_ddp(module, env, device) is module


@pytest.mark.parametrize("total", [20, 7, 6, 3, 1, 0])
def test_shard_size_partitions_exactly(total):
    """Shards must sum to the global total (no work dropped, none done
    twice) and differ by at most one (no straggler rank doing a whole extra
    batch while three others wait at the next collective)."""
    world_size = 4
    shards = [
        shard_size(total, DistEnv(rank=r, world_size=world_size)) for r in range(world_size)
    ]
    assert sum(shards) == total
    assert max(shards) - min(shards) <= 1


def test_shard_size_is_identity_for_single_rank():
    assert shard_size(20, DistEnv(rank=0, world_size=1)) == 20


# --------------------------------------------------------------------------
# Model wrapping / unwrapping
# --------------------------------------------------------------------------


def test_watermark_embedder_forward_matches_get_watermark():
    """The whole point of WatermarkEmbedder is that DDP intercepts the call
    the training loop actually makes; it must not change what that call
    computes."""
    torch.manual_seed(0)
    generator = tiny_generator()
    x = torch.randn(2, 1, 2000)
    message = torch.randint(0, 2, (2, NBITS))

    direct = generator.get_watermark(x, message=message)
    through_embedder = WatermarkEmbedder(generator)(x, message)
    assert torch.allclose(direct, through_embedder)


def test_embed_watermark_accepts_raw_generator_and_embedder():
    """Single-GPU callers (sanity_check, tests) pass the generator itself;
    the DDP path passes a WatermarkEmbedder. Both must work, and produce the
    same thing given the same RNG state."""
    torch.manual_seed(0)
    generator = tiny_generator()
    x = torch.randn(4, 1, 2000)
    message = torch.randint(0, 2, (4, NBITS))

    torch.manual_seed(7)
    from_raw = embed_watermark(generator, x, message, 24.0, 36.0)
    torch.manual_seed(7)
    from_embedder = embed_watermark(WatermarkEmbedder(generator), x, message, 24.0, 36.0)
    assert torch.allclose(from_raw, from_embedder)


def test_unwrap_generator_strips_embedder_and_keeps_checkpoint_keys_clean():
    """Regression test for checkpoint compatibility: evaluate.py loads the
    saved state_dict straight into a bare AudioSealWM, so a `generator.` or
    `module.` prefix leaking in from the wrappers would make every checkpoint
    written by a multi-GPU run unloadable."""
    torch.manual_seed(0)
    generator = tiny_generator()
    embedder = WatermarkEmbedder(generator)

    assert unwrap_generator(embedder) is generator
    assert unwrap_generator(generator) is generator

    wrapped_keys = set(embedder.state_dict())
    unwrapped_keys = set(unwrap_generator(embedder).state_dict())
    assert any(k.startswith("generator.") for k in wrapped_keys), "wrapper should prefix, else test is vacuous"
    assert not any(k.startswith(("generator.", "module.")) for k in unwrapped_keys)
    assert unwrapped_keys == set(generator.state_dict())


def test_unwrap_module_passes_through_plain_modules():
    module = torch.nn.Linear(2, 2)
    assert unwrap_module(module) is module


# --------------------------------------------------------------------------
# Epoch length under sharding
# --------------------------------------------------------------------------


class _FakeSampler:
    def __init__(self):
        self.epochs_set = []

    def set_epoch(self, epoch):
        self.epochs_set.append(epoch)


@pytest.mark.parametrize(
    "loader_len,world_size,updates_per_epoch,expected",
    [
        # train-clean-100 at batch_size=16 -- 1783 batches on 1 GPU, 445 per
        # rank on 4. Same config, same 1000-step epoch either way.
        (1783, 1, 1000, 1000),
        (445, 4, 1000, 1000),
        # run_train_10h.sh's subset: ~167 batches on 1 GPU, ~41 per rank on 4.
        # The 1000-step cap is never reached, so it must NOT become a target.
        (167, 1, 1000, 167),
        (41, 4, 1000, 164),
        (1000, 1, 1000, 1000),
    ],
)
def test_epoch_length_is_the_same_on_any_gpu_count(loader_len, world_size, updates_per_epoch, expected):
    """Regression test: an epoch must be the same number of optimizer steps
    regardless of how many ranks the data was sharded over.

    Sharding across 4 ranks divides the per-rank loader length by 4, so a
    plain `for batch in dataloader` + break would end a train-clean-100 epoch
    at 445 steps instead of 1000 -- 2.25x fewer optimizer steps and twice as
    many checkpoints, from a config the user did not change. Cycling fixes
    that, but only for the sharded case: when one pass over the data is
    genuinely shorter than `updates_per_epoch` (the 10h subset rows), the
    epoch has to end there on every GPU count, or the cap silently turns into
    a target and a 16.7k-step run becomes a 100k-step one.
    """
    loader = [torch.tensor([float(i)]) for i in range(loader_len)]
    sampler = _FakeSampler()
    iterator = EpochBatchIterator(loader, sampler, updates_per_epoch, world_size)

    assert iterator.steps_per_epoch == expected
    for _ in range(3):  # several epochs, to catch state leaking between them
        assert len(list(iterator.epoch())) == expected


def test_single_process_never_replays_an_exhausted_loader():
    """The single-GPU path must behave exactly like the pre-DDP loop: stop
    when the data runs out, never restart it to reach `updates_per_epoch`.

    run_train_10h.sh depends on this. Its 10h subset exhausts at ~167
    steps/epoch, which is what bounds the run at ~16.7k steps; cycling it
    would run the configured 100 x 1000 = 100k steps instead, replaying the
    same 10h roughly six times over.
    """
    loader = [torch.tensor([float(i)]) for i in range(167)]
    sampler = _FakeSampler()
    iterator = EpochBatchIterator(loader, sampler, 1000, world_size=1)

    assert len(list(iterator.epoch())) == 167
    assert sampler.epochs_set == [0], "a single pass must not restart the loader"


def test_epoch_iterator_reshuffles_each_pass():
    """Each restart must call set_epoch with a NEW value, or every pass
    replays the identical order and the extra passes add no diversity."""
    loader = [torch.tensor([float(i)]) for i in range(10)]
    sampler = _FakeSampler()
    iterator = EpochBatchIterator(loader, sampler, 25, world_size=4)

    list(iterator.epoch())
    assert len(sampler.epochs_set) >= 3, "expected multiple passes over a 10-batch loader"
    assert sampler.epochs_set == sorted(set(sampler.epochs_set)), "set_epoch values must be distinct/increasing"


def test_epoch_iterator_works_without_a_sampler():
    """Single-GPU path: no DistributedSampler to set_epoch on, and the loader
    is the thing that ends the epoch."""
    loader = [torch.tensor([float(i)]) for i in range(4)]
    iterator = EpochBatchIterator(loader, None, 10)
    assert len(list(iterator.epoch())) == 4


def test_epoch_iterator_rejects_an_empty_loader():
    iterator = EpochBatchIterator([], None, 3)
    with pytest.raises(RuntimeError, match="no batches"):
        list(iterator.epoch())


# --------------------------------------------------------------------------
# Rank-shared attack sampling
# --------------------------------------------------------------------------


def test_attack_sampling_rng_agrees_across_ranks():
    """Every rank builds its own RNG from the same cfg.seed and must draw the
    same branch sequence -- that is what keeps a step's cost equal to the
    branch that was sampled, instead of the most expensive branch any of the
    4 ranks happened to draw."""
    attacks = {"identity": IdentityAttack(), "other": IdentityAttack()}
    weights = {"identity": 1.0, "other": 1.0}

    rank0 = SampledReconstructionAttack(attacks, weights, rng=attack_sampling_rng(1234))
    rank3 = SampledReconstructionAttack(attacks, weights, rng=attack_sampling_rng(1234))

    x = torch.randn(1, 1, 100)
    names0 = [rank0(x)[1] for _ in range(50)]
    names3 = [rank3(x)[1] for _ in range(50)]
    assert names0 == names3
    assert len(set(names0)) == 2, "both branches should still get sampled -- not a degenerate sequence"


def test_attack_sampling_rng_differs_across_seeds():
    attacks = {"identity": IdentityAttack(), "other": IdentityAttack()}
    weights = {"identity": 1.0, "other": 1.0}
    a = SampledReconstructionAttack(attacks, weights, rng=attack_sampling_rng(1))
    b = SampledReconstructionAttack(attacks, weights, rng=attack_sampling_rng(2))
    x = torch.randn(1, 1, 100)
    assert [a(x)[1] for _ in range(50)] != [b(x)[1] for _ in range(50)]


def test_sampled_attack_without_rng_keeps_using_global_random():
    """Backwards compatibility: the rng argument is optional and defaults to
    the previous behavior."""
    attack = SampledReconstructionAttack({"identity": IdentityAttack()}, {"identity": 1.0})
    x = torch.randn(1, 1, 100)
    assert attack(x)[1] == "identity"


# --------------------------------------------------------------------------
# Rank-local device pinning
# --------------------------------------------------------------------------


def test_resolve_device_ignores_local_rank_off_cuda():
    assert resolve_device("cpu", local_rank=3) == torch.device("cpu")


def test_resolve_device_without_local_rank_is_unchanged():
    """The pre-existing single-process signature must behave exactly as before."""
    assert resolve_device("cpu") == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_resolve_device_pins_local_rank():
    assert resolve_device("cuda", local_rank=0) == torch.device("cuda", 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_resolve_device_rejects_local_rank_beyond_visible_devices():
    with pytest.raises(RuntimeError, match="LOCAL_RANK"):
        resolve_device("cuda", local_rank=torch.cuda.device_count())


# --------------------------------------------------------------------------
# Real two-process runs
# --------------------------------------------------------------------------


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_workers(mode: str, out_dir: Path, nproc: int = 2) -> None:
    """Start `nproc` worker processes with exactly the environment torchrun
    provides (RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT) and
    wait for all of them.

    Launching the ranks directly rather than shelling out to `torchrun`
    keeps this runnable on any PyTorch build: torchrun's elastic agent
    creates its rendezvous TCPStore with libuv unconditionally, and the
    Windows CPU wheels this suite is developed against are built without it.
    The code under test only ever reads those five environment variables
    (see distributed.env_from_environment), so this exercises the same path
    a real `torchrun --nproc-per-node=4` takes.
    """
    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"), base_env.get("PYTHONPATH", "")])
    base_env["MASTER_ADDR"] = "127.0.0.1"
    base_env["MASTER_PORT"] = str(_free_port())
    base_env["WORLD_SIZE"] = str(nproc)

    procs = []
    logs = []
    for rank in range(nproc):
        worker_env = dict(base_env, RANK=str(rank), LOCAL_RANK=str(rank))
        # Log to files, not PIPEs: with a pipe, a rank that fills the ~64KB
        # buffer blocks on write until someone reads it, and this loop only
        # reads rank 0 first -- which would hang the whole test as soon as a
        # worker gets chatty. That failure looks exactly like the collective
        # deadlock these tests exist to catch, so it must not be possible.
        log_path = out_dir / f"worker_rank{rank}.log"
        logs.append(log_path)
        handle = log_path.open("w")
        procs.append(
            (
                subprocess.Popen(
                    [sys.executable, str(WORKER), "--mode", mode, "--out-dir", str(out_dir)],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=worker_env,
                ),
                handle,
            )
        )

    failures = []
    for rank, (proc, handle) in enumerate(procs):
        try:
            proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            proc.kill()
            failures.append(f"worker rank {rank} ({mode}) timed out -- likely a collective deadlock")
        handle.close()
        if proc.returncode != 0 and not failures:
            failures.append(
                f"worker rank {rank} ({mode}) exited {proc.returncode}:\n{logs[rank].read_text()}"
            )
    if failures:
        for proc, _ in procs:
            if proc.poll() is None:
                proc.kill()
        pytest.fail(failures[0])


@pytest.mark.slow
def test_ddp_actually_synchronizes_generator_gradients(tmp_path):
    """The core multi-GPU correctness property: after several steps on
    DIFFERENT data per rank, every rank must hold identical generator
    weights. If they don't, DDP never averaged the gradients and a 4-GPU run
    is really 4 independent runs, with rank 0's checkpoint trained on a
    quarter of the data -- a failure that shows no error and a perfectly
    normal-looking loss curve.
    """
    _run_workers("ddp_sync", tmp_path)

    rank0 = torch.load(tmp_path / "rank0.pt", weights_only=False)
    rank1 = torch.load(tmp_path / "rank1.pt", weights_only=False)

    assert rank0["state_dict"].keys() == rank1["state_dict"].keys()
    for key, value0 in rank0["state_dict"].items():
        assert torch.allclose(value0, rank1["state_dict"][key], atol=1e-6), (
            f"parameter {key} diverged across ranks -- gradients were not synchronized"
        )


@pytest.mark.slow
def test_bypassing_ddp_forward_diverges(tmp_path):
    """Negative control for the test above, and the regression test for
    WatermarkEmbedder's reason to exist: calling the generator's own
    `get_watermark` instead of going through the DDP-wrapped forward skips
    the gradient allreduce, and the ranks drift apart. If this ever starts
    passing (i.e. no divergence), the test above has stopped proving
    anything.
    """
    _run_workers("ddp_bypass", tmp_path)

    rank0 = torch.load(tmp_path / "rank0.pt", weights_only=False)
    rank1 = torch.load(tmp_path / "rank1.pt", weights_only=False)

    diverged = any(
        not torch.allclose(v, rank1["state_dict"][k], atol=1e-6)
        for k, v in rank0["state_dict"].items()
    )
    assert diverged, "expected ranks to diverge when DDP's forward is bypassed"


@pytest.mark.slow
def test_ddp_checkpoint_state_dict_has_no_wrapper_prefixes(tmp_path):
    """A DDP + WatermarkEmbedder state_dict would have every key prefixed
    `module.generator.`; evaluate.py's load_generator_under_test feeds these
    keys straight into a bare AudioSealWM."""
    _run_workers("ddp_sync", tmp_path)

    saved = torch.load(tmp_path / "rank0.pt", weights_only=False)
    assert saved["unwrap_module_type"] == "WatermarkEmbedder", "DDP wrapper should have been present"
    assert saved["is_raw_generator"] == "AudioSealWM"
    assert not any(k.startswith(("module.", "generator.")) for k in saved["state_dict"])


@pytest.mark.slow
def test_collectives_pool_across_ranks(tmp_path):
    _run_workers("collectives", tmp_path)

    rank0 = torch.load(tmp_path / "rank0.pt", weights_only=False)
    rank1 = torch.load(tmp_path / "rank1.pt", weights_only=False)

    # Both ranks must end up with the SAME pooled view -- evaluate.py relies
    # on this to compute one global TPR@FPR threshold rather than two.
    expected_scores = torch.tensor([0.0, 10.0, 11.0])
    for result in (rank0, rank1):
        assert result["world_size"] == 2
        assert torch.allclose(result["gathered_scores"], expected_scores), result["gathered_scores"]
        assert result["gathered_values"] == [0.0, 1.0, 1.0]
        # mean over ranks of {0, 1} = 0.5; the constant stays constant
        assert result["means"]["a"] == pytest.approx(0.5)
        assert result["means"]["b"] == pytest.approx(1.0)
        # peak memory reduces by MAX, not mean: the worst card is what OOMs
        assert result["maxes"]["peak"] == pytest.approx(1.0)

    # Global batch counts split, they do not multiply.
    assert rank0["shard_of_20"] + rank1["shard_of_20"] == 20
    assert rank0["shard_of_7"] + rank1["shard_of_7"] == 7


@pytest.mark.slow
def test_idle_rank_still_joins_collectives(tmp_path):
    """Regression test: with fewer global batches than ranks, the high ranks
    get a zero-size shard. They must still reach every collective and
    contribute nothing -- an empty shard is not a failure.

    Treating it as one made every rank raise, which run() catches per attack,
    so an entire evaluation would report "every attack skipped" and exit 0.
    `run_smoke_eval.sh` uses n_eval_batches=2, so `torchrun --nproc_per_node=4`
    on it hits exactly this.
    """
    _run_workers("idle_rank", tmp_path, nproc=3)

    results = [torch.load(tmp_path / f"rank{r}.pt", weights_only=False) for r in range(3)]

    # 2 global batches over 3 ranks -> [1, 1, 0]
    assert [r["local_shard"] for r in results] == [1, 1, 0]
    for result in results:
        # Every rank, including the idle one, sees the full pooled result.
        assert result["gathered_scores_numel"] == 6, "idle rank contributed or received the wrong amount"
        assert result["gathered_values"] == [1.0, 1.0]
