# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generator-only fine-tuning against sampled reconstruction attacks.

Per training step:
  x --[frozen-except-here G_theta]--> x_wm = x + scale * G_theta(x, m)
      (scale chosen per-example, see embed_watermark, so the perturbation
      lands at a target watermark SNR sampled uniformly per example --
      NOT whatever amplitude G_theta happens to produce unscaled)
  x_wm --[frozen, differentiable, randomly-sampled attack]--> x_att
  x_att --[frozen detector]--> presence p, decoded message m_hat
  total_loss = lambda_det * detection_loss(p, m_hat, m)
             + lambda_perc * perceptual_loss(x, x_wm)      # NOTE: x_wm, not x_att
  total_loss.backward()   # through detector and attack (frozen, not detached) into G_theta
  optimizer.step()        # optimizer only ever holds G_theta's parameters

Only the generator is ever placed in the optimizer -- see `build_optimizer`.
The detector and attack module are frozen via `requires_grad_(False)` (which
excludes them from backprop's *gradient accumulation* target, not from the
graph itself) -- see `build_detector` and `attacks.py` for why gradients
still reach `x_wm`.

Multi-GPU
---------
Run under torchrun, one process per GPU:

    torchrun --standalone --nproc_per_node=4 -m audioseal_robust.train \\
        data.train_dir=/path/to/wavs

`data.batch_size` is per GPU, so that command trains at an effective batch of
4 x batch_size. Plain `python -m audioseal_robust.train` still works exactly
as before -- every distributed helper degrades to a no-op at world_size=1
(see distributed.py).

Two things about this loop specifically needed handling to make DDP correct,
beyond wrapping the model:

  1. DDP only hooks the wrapped module's `forward()`, and this loop never
     calls the generator's forward -- it calls `get_watermark`. Calling
     `ddp.module.get_watermark` would bypass DDP entirely and silently train
     4 divergent generators that never exchange a gradient. Hence
     `WatermarkEmbedder`, whose `forward` *is* `get_watermark`, so DDP wraps
     something this loop actually calls.
  2. The attack branch is re-sampled every step, and the branches differ
     enormously in cost. That draw is shared across ranks -- see
     `distributed.attack_sampling_rng`.
"""

import contextlib
import itertools
import logging
import math
import os
import typing as tp
from datetime import datetime

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from audioseal import AudioSeal
from audioseal.models import AudioSealDetector, AudioSealWM

from .attacks import (
    BigVGANAttack,
    DACAttack,
    AudioLDMAttack,
    IdentityAttack,
    SGMSEAttack,
    SampledReconstructionAttack,
)
from .config import TrainConfig, load_config
from .data import build_dataloader
from .distributed import (
    DistEnv,
    all_reduce_mean,
    attack_sampling_rng,
    barrier,
    cleanup_distributed,
    configure_logging,
    gather_objects,
    init_distributed,
    seed_everything,
    unwrap_module,
    wrap_ddp,
)
from .losses import PsychoacousticMelLoss, detection_loss_components
from .tracking import ExperimentTracker, NullTracker, build_tracker

logger = logging.getLogger(__name__)


class WatermarkEmbedder(nn.Module):
    """Exposes `generator.get_watermark` as a plain `forward`, so that the
    generator can be wrapped in DistributedDataParallel.

    DDP installs its gradient-synchronization hooks by intercepting the
    wrapped module's `forward()` and nothing else. This training loop only
    ever calls `get_watermark(x, message=...)`, never `AudioSealWM.forward`,
    so `DDP(generator).get_watermark(...)` would resolve straight through to
    the underlying module, skip DDP completely, and leave each rank training
    its own private copy of the generator with no allreduce at all -- which
    fails silently: the loss still goes down on every rank, the checkpoint
    saved from rank 0 is just trained on 1/4 of the data.

    Wrapping this class instead means the call the loop makes IS the call
    DDP intercepts. `unwrap_generator` gets the real AudioSealWM back out
    for checkpointing and eval-mode toggles.
    """

    def __init__(self, generator: AudioSealWM):
        super().__init__()
        self.generator = generator

    def forward(self, x: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        return self.generator.get_watermark(x, message=message)


def unwrap_generator(module: nn.Module) -> AudioSealWM:
    """Peel DDP and `WatermarkEmbedder` off to get the raw AudioSealWM back.

    Used for anything that isn't the DDP-synchronized forward pass: saving a
    checkpoint (whose keys must match what `AudioSeal.load_generator` and
    evaluate.py expect -- neither a `module.` nor a `generator.` prefix), and
    `.train()`/`.eval()` toggles.
    """
    inner = unwrap_module(module)
    if isinstance(inner, WatermarkEmbedder):
        inner = inner.generator
    return tp.cast(AudioSealWM, inner)


def random_message(nbits: int, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, 2, (batch_size, nbits), device=device)


def _autocast(device: torch.device) -> tp.ContextManager[None]:
    """bf16 autocast for the forward pass -- halves activation memory (on
    top of attacks.py's activation checkpointing, which reduces step COUNT
    held live, not per-step size) and speeds up matmul-heavy ops, with none
    of fp16's overflow risk (bf16 keeps fp32's exponent range, just less
    mantissa) so no GradScaler is needed. Numerically sensitive ops
    (BCELoss inside detection_loss_components, log()/exp() in SGMSE's
    _spec_fwd/_spec_back) are auto-promoted back to fp32 by autocast's own
    op-casting policy, not something callers need to special-case.
    cuda/cpu only -- MPS's bf16 autocast support is inconsistent across
    torch versions, and this project's real training runs on cuda anyway."""
    if device.type in ("cuda", "cpu"):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _watermark_delta(embedder: nn.Module, x: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
    """delta = G_theta(x, m), accepting either a raw AudioSealWM (single-GPU
    path, tests, sanity_check) or a `WatermarkEmbedder`/DDP wrapper around
    one (the DDP path, where the call must go through `forward`)."""
    if isinstance(embedder, AudioSealWM):
        return embedder.get_watermark(x, message=message)
    return embedder(x, message)


def embed_watermark(
    embedder: nn.Module,
    x: torch.Tensor,
    message: torch.Tensor,
    snr_db_min: float,
    snr_db_max: float,
) -> torch.Tensor:
    """x_wm = x + scale * delta, where delta = generator.get_watermark(x, m)
    and `scale` is chosen per-example (sampled fresh every call) so that
    ||scale * delta|| lands at a target SNR (dB) relative to ||x||, drawn
    uniformly from [snr_db_min, snr_db_max] independently for each item in
    the batch -- rather than using whatever amplitude get_watermark()
    happens to produce unscaled. See TrainConfig.watermark_snr_db_{min,max}
    for how that range was picked.

    `embedder` is either the generator itself or a `WatermarkEmbedder`
    (possibly DDP-wrapped) around it -- see `_watermark_delta`. The scaling
    is deliberately left outside the DDP-wrapped module: it holds no
    parameters, so there is nothing for DDP to synchronize in it.

    x: (B, 1, T). Gradients flow through normally (delta is not detached),
    same as before this scaling was introduced.
    """
    delta = _watermark_delta(embedder, x, message)
    target_snr_db = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(snr_db_min, snr_db_max)
    x_norm = x.norm(dim=-1, keepdim=True)
    delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (x_norm / delta_norm) * (10 ** (-target_snr_db / 20))
    return x + scale * delta


def build_generator(cfg: TrainConfig, device: torch.device) -> AudioSealWM:
    """The only trainable component. Loaded from a pretrained checkpoint and
    left fully trainable -- this fine-tunes it further, it does not train
    from scratch."""
    generator = AudioSeal.load_generator(cfg.generator.checkpoint, nbits=cfg.nbits, device=device)
    if cfg.generator.resume_from:
        state = torch.load(cfg.generator.resume_from, map_location=device, weights_only=False)
        generator.load_state_dict(state["model"])
        logger.info("resumed generator weights from %s", cfg.generator.resume_from)
    generator.train()
    return generator


def build_detector(cfg: TrainConfig, device: torch.device) -> AudioSealDetector:
    """Frozen. Never placed in the optimizer (see build_optimizer); params
    also have requires_grad=False so no .grad ever accumulates on them even
    if someone mistakenly passes detector.parameters() somewhere.

    train() rather than eval(): train_step backprops through the detector
    (frozen but differentiable, see its comment) into the generator, and
    cudnn's LSTM (inside SEANet) only supports backward in training mode --
    eval() raises "cudnn RNN backward can only be called in training mode".
    Numerically harmless here since audioseal has no dropout/batchnorm
    anywhere, so train() vs eval() forward output is identical; freezing is
    entirely done via requires_grad_(False) above, not via this mode."""
    detector = AudioSeal.load_detector(cfg.detector.checkpoint, nbits=cfg.nbits, device=device)
    detector.train()
    for p in detector.parameters():
        p.requires_grad_(False)
    return detector


def build_attack(cfg: TrainConfig, device: torch.device) -> SampledReconstructionAttack:
    attacks: tp.Dict[str, nn.Module] = {
        "identity": IdentityAttack(),
        "bigvgan": BigVGANAttack(checkpoint=cfg.attack.bigvgan.checkpoint, sample_rate=cfg.sample_rate),
        "dac": DACAttack(checkpoint=cfg.attack.dac.checkpoint, sample_rate=cfg.sample_rate),
        "sgmse": SGMSEAttack(
            checkpoint=cfg.attack.sgmse.checkpoint,
            sample_rate=cfg.sample_rate,
            num_steps=cfg.attack.sgmse.num_steps,
        ),
        "audioldm": AudioLDMAttack(
            checkpoint=cfg.attack.audioldm.checkpoint,
            config=cfg.attack.audioldm.config,
            sample_rate=cfg.sample_rate,
            strength_max=cfg.attack.audioldm.strength_max,
        ),
    }
    weights = {
        "identity": cfg.attack.weights.identity,
        "bigvgan": cfg.attack.weights.bigvgan,
        "dac": cfg.attack.weights.dac,
        "sgmse": cfg.attack.weights.sgmse,
        "audioldm": cfg.attack.weights.audioldm,
    }
    # Rank-shared RNG: every rank must sample the same branch each step,
    # see distributed.attack_sampling_rng. On a single process this just
    # makes the branch sequence reproducible from cfg.seed.
    attack = SampledReconstructionAttack(attacks, weights, rng=attack_sampling_rng(cfg.seed))
    return attack.to(device)


def build_optimizer(generator: AudioSealWM, cfg: TrainConfig) -> torch.optim.Optimizer:
    # Only generator.parameters() -- this is the actual mechanism that keeps
    # the detector and attack module untrained, not just requires_grad.
    # Deliberately built from the *unwrapped* generator: DDP and
    # WatermarkEmbedder both reuse the very same Parameter objects, so this
    # optimizer still steps the replicated weights, while keeping the
    # invariant ("the optimizer holds exactly the generator's parameters")
    # literally checkable -- see tests/test_audioseal_robust.py.
    return torch.optim.Adam(
        generator.parameters(),
        lr=cfg.optim.lr,
        betas=tuple(cfg.optim.betas),
        weight_decay=cfg.optim.weight_decay,
    )


def build_perceptual_loss(cfg: TrainConfig, device: torch.device) -> PsychoacousticMelLoss:
    return PsychoacousticMelLoss(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.mel_loss.n_fft,
        hop_length=cfg.mel_loss.hop_length,
        win_length=cfg.mel_loss.win_length,
        n_mels=cfg.mel_loss.n_mels,
        f_min=cfg.mel_loss.f_min,
        f_max=cfg.mel_loss.f_max,
    ).to(device)


def _grad_norm(module: nn.Module) -> float:
    """L2 norm over all parameters of `module` that currently have a
    gradient. For a frozen module (requires_grad=False on every param, e.g.
    `attack` -- see build_attack/build_detector), no leaf tensor ever
    accumulates .grad regardless of what the backward graph passed through
    it, so this is always exactly 0.0 -- logging it every step is a running,
    real-training confirmation of that invariant (the one-shot version of
    this is tests/test_audioseal_robust.py::test_gradients_flow_to_generator_only's
    `assert all(p.grad is None for p in detector.parameters())`), not just a
    number that happens to be small."""
    total_sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().float().norm(2).item() ** 2
    return total_sq**0.5


def _normalize_grad_(module: nn.Module, target_norm: float, floor: float) -> float:
    """Rescale `module`'s gradients so their global L2 norm equals
    `target_norm`. Returns the norm *before* rescaling.

    Unlike clip_grad_norm_ this scales up as well as down, which is the point:
    it makes each step's contribution independent of which attack branch
    produced it (see OptimConfig.normalize_grad for the measurements that
    motivated it). The direction is untouched -- every gradient is multiplied
    by the same scalar.

    Under DDP this needs no extra synchronization: backward() has already
    all-reduced the gradients, so every rank computes an identical norm and
    therefore an identical scale, and the replicas stay bit-identical.
    """
    total = _grad_norm(module)
    if not math.isfinite(total):
        raise RuntimeError(
            f"non-finite gradient norm ({total}) -- refusing to rescale. This mirrors "
            "clip_grad_norm_(error_if_nonfinite=True); a NaN/Inf gradient means something "
            "upstream is broken rather than merely large."
        )
    # Below the floor there is no direction worth preserving, and scaling up to
    # target_norm would amplify numerical noise into a full-sized step.
    if total < floor:
        return total
    scale = target_norm / total
    for p in module.parameters():
        if p.grad is not None:
            p.grad.mul_(scale)
    return total


_BYTES_PER_GIB = float(1024**3)


class CudaMemoryProbe:
    """Per-step CUDA allocator instrumentation, split around the backward pass.

    This exists to answer "is the activation checkpointing in attacks.py
    actually buying anything?", which the usual dashboards cannot: wandb's
    "GPU Memory Allocated" system panels (and `nvidia-smi`) read NVML, i.e.
    the memory the *process* holds from the driver, which is PyTorch's
    caching allocator pool. That pool only ever grows to the run's
    high-water mark and then sits flat, so it looks identical whether
    checkpointing is on or off, and flat there is NOT evidence that
    checkpointing isn't happening. The numbers that actually move are the
    allocator's own:

      mem/activations_gib -- memory still live at the end of the forward
        pass that wasn't live before it, i.e. what autograd is holding for
        backward. This is precisely what checkpointing trades away, so it's
        the most direct signal: it should drop sharply once a checkpointed
        sampler loop (attacks.py's `_pc_step` / `_ddpm_step`) is in the
        graph, and scale with num_steps if checkpointing were off.
      mem/forward_peak_gib, mem/backward_peak_gib -- peak allocation within
        each half of the step. Uncheckpointed, the peak sits in the forward;
        checkpointed, the forward peak drops and the backward peak rises,
        since each segment's activations are recomputed there.
      mem/reserved_gib -- the allocator pool: the flat number the system
        panels show. Logged alongside so the gap against the allocated
        figures above is visible rather than mysterious.

    Every method only reads/resets allocator counters -- no device
    synchronisation, so this does not perturb the step time it sits next to.
    Off CUDA, or with `tracking.log_memory=false`, all methods are no-ops
    and no keys are emitted at all.
    """

    def __init__(self, device: torch.device, enabled: bool = True) -> None:
        self._enabled = enabled and device.type == "cuda"
        self._device = device
        self._allocated_before = 0
        self._metrics: tp.Dict[str, float] = {}

    def start(self) -> None:
        if not self._enabled:
            return
        # Peak counters are cumulative until explicitly reset, so without
        # this every step after the first would just re-report the run's
        # high-water mark -- a flat line, i.e. exactly the artifact this
        # probe exists to avoid.
        torch.cuda.reset_peak_memory_stats(self._device)
        self._allocated_before = torch.cuda.memory_allocated(self._device)

    def mark_forward_end(self) -> None:
        if not self._enabled:
            return
        allocated = torch.cuda.memory_allocated(self._device)
        self._metrics["mem/forward_peak_gib"] = (
            torch.cuda.max_memory_allocated(self._device) / _BYTES_PER_GIB
        )
        self._metrics["mem/activations_gib"] = (allocated - self._allocated_before) / _BYTES_PER_GIB
        torch.cuda.reset_peak_memory_stats(self._device)  # so the backward peak is measured on its own

    def mark_backward_end(self) -> None:
        if not self._enabled:
            return
        self._metrics["mem/backward_peak_gib"] = (
            torch.cuda.max_memory_allocated(self._device) / _BYTES_PER_GIB
        )
        self._metrics["mem/reserved_gib"] = torch.cuda.memory_reserved(self._device) / _BYTES_PER_GIB

    def metrics(self) -> tp.Dict[str, float]:
        return dict(self._metrics)


def _clip_grad_norm_per_sample(grad: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Clip each batch item's L2 norm without changing its direction."""
    if max_norm <= 0:
        raise ValueError("max_x_wm_grad_norm must be positive")

    norms = grad.detach().float().flatten(start_dim=1).norm(2, dim=1)
    finite_scales = (max_norm / norms.clamp_min(1e-12)).clamp(max=1.0)
    scales = torch.where(torch.isfinite(norms), finite_scales, torch.ones_like(norms))
    shape = (grad.shape[0],) + (1,) * (grad.ndim - 1)
    return grad * scales.to(dtype=grad.dtype).view(shape)


def _clip_and_capture_activation_grad(
    name: str,
    activation_grad_norms: tp.Dict[str, float],
    max_norm: tp.Optional[float],
) -> tp.Callable[[torch.Tensor], torch.Tensor]:
    """Record an activation's grad norm, then optionally clip it per sample."""

    def hook(grad: torch.Tensor) -> torch.Tensor:
        activation_grad_norms[name] = grad.detach().float().norm(2).item()
        if max_norm is None:
            return grad

        clipped = _clip_grad_norm_per_sample(grad, max_norm)
        activation_grad_norms[f"{name}_clipped"] = clipped.detach().float().norm(2).item()
        return clipped

    return hook


def train_step(
    embedder: nn.Module,
    detector: AudioSealDetector,
    attack: SampledReconstructionAttack,
    perceptual_loss_fn: PsychoacousticMelLoss,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
) -> tp.Dict[str, tp.Union[float, str]]:
    """`embedder` is the (possibly DDP-wrapped) `WatermarkEmbedder`. Under
    DDP the gradient allreduce happens inside `total_loss.backward()`, so
    every reported gradient norm below is already the synchronized,
    world-averaged one."""
    x = batch.to(device)  # clean audio, (B, 1, T), 16kHz
    message = random_message(cfg.nbits, x.size(0), device=x.device)

    memory = CudaMemoryProbe(device, enabled=cfg.tracking.log_memory)
    memory.start()

    # 1-4 under bf16 autocast (see _autocast's docstring): forward through
    # the generator, attack, and detector, plus loss computation. Backward
    # (below) stays outside -- autocast only needs to cover the forward pass,
    # and the hooks registered on x_wm/x_att fire during backward regardless
    # of which context the tensors were originally produced under.
    with _autocast(device):
        # 1. x_wm = x + scale * G_theta(x, m), scale set per-example to hit a
        #    randomly sampled target watermark SNR -- see embed_watermark.
        x_wm = embed_watermark(embedder, x, message, cfg.watermark_snr_db_min, cfg.watermark_snr_db_max)

        # Activation-gradient probes: register_hook fires during backward() with
        # the gradient AT that point in the graph, before it continues further
        # back. x_wm sits right at the generator/attack boundary -- its grad norm
        # is the actual signal reaching the generator after passing back through
        # the whole frozen detector+attack backbone, so if it's ~0 the backbone
        # is killing the gradient regardless of what grad_norm_generator (the
        # PARAMETER gradient, downstream of this) shows. x_att (attack/detector
        # boundary) isolates which half of the backbone is responsible: compare
        # against x_wm's norm -- a big drop across attack vs across detector
        # tells you which one is the vanishing point.
        activation_grad_norms: tp.Dict[str, float] = {}

        def _capture_activation_grad_norm(name: str) -> tp.Callable[[torch.Tensor], None]:
            def hook(grad: torch.Tensor) -> None:
                activation_grad_norms[name] = grad.detach().float().norm(2).item()

            return hook

        x_wm.register_hook(
            _clip_and_capture_activation_grad("x_wm", activation_grad_norms, cfg.optim.max_x_wm_grad_norm)
        )

        # 2. sampled reconstruction attack (frozen, graph stays connected).
        #    Same branch on every rank -- see build_attack.
        x_att, attack_name = attack(x_wm)
        x_att.register_hook(_capture_activation_grad_norm("x_att"))

        # 3. frozen detector: presence (B,2,T) softmax probs, message (B,nbits) sigmoid probs
        presence, m_hat = detector.forward(x_att)
        p = presence[:, 1, :].mean(dim=-1)  # presence prob per example, pooled over time

        # 4. losses
        det_loss, presence_loss, bit_loss = detection_loss_components(p, m_hat, message, bit_weight=cfg.lambda_bit)
        perc_loss = perceptual_loss_fn(x, x_wm)  # pre-attack, per spec
        total_loss = cfg.lambda_det * det_loss + cfg.lambda_perc * perc_loss

    # 5. backprop through detector + attack (frozen but differentiable) into G_theta
    memory.mark_forward_end()  # before zero_grad/backward frees or adds anything
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    memory.mark_backward_end()
    grad_norm_generator = _grad_norm(unwrap_generator(embedder))
    grad_norm_attack = _grad_norm(attack)  # expected: always 0.0, see _grad_norm's docstring
    if cfg.optim.normalize_grad:
        # Mutually exclusive with clipping by construction: rescaling to
        # exactly max_norm already satisfies the clip bound, so running
        # clip_grad_norm_ afterwards would be a no-op.
        if cfg.optim.max_norm is None:
            raise ValueError("optim.normalize_grad requires optim.max_norm (it is the target norm)")
        _normalize_grad_(
            unwrap_generator(embedder), cfg.optim.max_norm, cfg.optim.normalize_grad_floor
        )
        grad_scale = cfg.optim.max_norm / grad_norm_generator if grad_norm_generator else 1.0
    elif cfg.optim.max_norm is not None:
        torch.nn.utils.clip_grad_norm_(unwrap_generator(embedder).parameters(), cfg.optim.max_norm, error_if_nonfinite=True)
        grad_scale = min(1.0, cfg.optim.max_norm / grad_norm_generator) if grad_norm_generator else 1.0
    else:
        grad_scale = 1.0
    optimizer.step()

    return {
        "loss": total_loss.item(),
        "detection_loss": det_loss.item(),
        "presence_loss": presence_loss.item(),
        "bit_loss": bit_loss.item(),
        "perceptual_loss": perc_loss.item(),
        "presence_prob": p.mean().item(),
        "grad_norm_generator": grad_norm_generator,
        # The scalar actually applied to the generator's gradient before
        # optimizer.step(). Under plain clipping this is <1 on branches whose
        # gradients explode and 1.0 on the rest, which is exactly the silent
        # per-branch reweighting OptimConfig.normalize_grad exists to remove --
        # so logging it makes that imbalance visible per step instead of
        # something to reconstruct from a log afterwards.
        "grad_scale_applied": grad_scale,
        "grad_norm_attack": grad_norm_attack,
        "grad_norm_x_wm": activation_grad_norms.get("x_wm", 0.0),
        "grad_norm_x_wm_clipped": activation_grad_norms.get(
            "x_wm_clipped", activation_grad_norms.get("x_wm", 0.0)
        ),
        "grad_norm_x_att": activation_grad_norms.get("x_att", 0.0),
        "attack": attack_name,
        **memory.metrics(),  # empty off CUDA / with tracking.log_memory=false
    }


@torch.no_grad()
def eval_step(
    generator: AudioSealWM,
    detector: AudioSealDetector,
    attack: SampledReconstructionAttack,
    perceptual_loss_fn: PsychoacousticMelLoss,
    batch: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    attack_name: tp.Optional[str] = None,
) -> tp.Dict[str, tp.Union[float, str]]:
    """Same forward computation as train_step (same loss formula), just no
    backward/optimizer step -- for periodic train-vs-eval comparison during
    training (see TrainConfig.eval_every and DataConfig.valid_dir).

    `attack_name` pins the attack branch. The caller passes one explicitly and
    sweeps every branch, because a randomly sampled branch makes these numbers
    incomparable between eval points: each would measure a different task, and
    the resulting curve alternates between the branches' very different loss
    scales -- which reads as instability or regression rather than the
    branch-switching it actually is. See SampledReconstructionAttack.forward.

    Takes the RAW generator, not the DDP wrapper, on purpose: there is no
    backward here, so there is nothing for DDP to synchronize, and running a
    DDP forward that is never followed by a backward leaves its reducer
    expecting a gradient pass that never comes. Every rank still runs this
    (on its own shard of valid_dir, at the same steps) and the caller
    averages the results across ranks -- see `train`.
    """
    generator.eval()
    try:
        x = batch.to(device)
        message = random_message(cfg.nbits, x.size(0), device=x.device)
        with _autocast(device):
            x_wm = embed_watermark(generator, x, message, cfg.watermark_snr_db_min, cfg.watermark_snr_db_max)
            x_att, sampled_name = attack(x_wm, name=attack_name)
            presence, m_hat = detector.forward(x_att)
            p = presence[:, 1, :].mean(dim=-1)
            det_loss, presence_loss, bit_loss = detection_loss_components(p, m_hat, message, bit_weight=cfg.lambda_bit)
            perc_loss = perceptual_loss_fn(x, x_wm)
            total_loss = cfg.lambda_det * det_loss + cfg.lambda_perc * perc_loss
    finally:
        generator.train()  # train_step assumes the generator stays in train() mode

    return {
        "loss": total_loss.item(),
        "detection_loss": det_loss.item(),
        "presence_loss": presence_loss.item(),
        "bit_loss": bit_loss.item(),
        "perceptual_loss": perc_loss.item(),
        "presence_prob": p.mean().item(),
        "attack": sampled_name,
    }


def build_experiment_tracker(cfg: TrainConfig, env: DistEnv = DistEnv()) -> ExperimentTracker:
    """Only rank 0 gets a real tracker. Without this guard a 4-GPU run
    creates 4 MLflow/W&B runs per training run, three of them holding
    metrics from a single shard."""
    if not env.is_main:
        return NullTracker()
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    assert isinstance(config_dict, dict)
    config_dict["world_size"] = env.world_size
    config_dict["effective_batch_size"] = cfg.data.batch_size * env.world_size
    return build_tracker(
        backend=cfg.tracking.backend,
        project=cfg.tracking.project,
        run_name=cfg.tracking.run_name,
        config=config_dict,
        mlflow_tracking_uri=cfg.tracking.mlflow_tracking_uri,
        wandb_mode=cfg.tracking.wandb_mode,
    )


class EpochBatchIterator:
    """Yields one epoch of batches, restarting the dataloader with a fresh
    shuffle only when sharding left this rank's loader too short to cover the
    epoch on its own.

    An epoch is `min(updates_per_epoch, batches_per_pass * world_size)`
    optimizer steps, where `batches_per_pass` is the length of THIS rank's
    loader. Both halves of that minimum matter:

    - `updates_per_epoch` is the configured cap. On the full ~100h
      train-clean-100 at batch_size=16, a single GPU's loader holds 1783
      batches and the cap of 1000 is what ends the epoch.
    - `batches_per_pass * world_size` is how many steps one pass over the
      *dataset* supports, which does not depend on the GPU count. A
      DistributedSampler hands each rank `len(dataset) / world_size`
      examples, so the per-rank loader has 1/world_size as many batches --
      445 on each of four GPUs in the example above. Multiplying back by
      `world_size` is what keeps a 4-GPU epoch the same 1000 steps as the
      1-GPU epoch of the same config instead of 445 (2.25x fewer optimizer
      steps and checkpoints written twice as often, from an unchanged
      config). This is the only case in which this class cycles.

    On one process the two terms collapse to the pre-DDP loop -- `for batch
    in dataloader` with a break at `updates_per_epoch` -- so a single-GPU run
    still ends its epoch when the data runs out and the cap stays a cap
    rather than becoming a target. That distinction is load-bearing for
    run_train_10h.sh: its 10h subset exhausts at ~167 steps/epoch and is
    meant to stop there (~16.7k steps over 100 epochs), not to replay itself
    up to the config's 100k.

    The consequence of cycling, inherent to a `world_size`x larger effective
    batch rather than to this class: such an epoch consumes `world_size`
    passes over the data, since each step consumes `world_size` batches.
    `set_epoch` is called with a monotonically increasing pass index, so
    every pass is shuffled differently rather than replaying the same order.

    All ranks advance in lockstep -- `drop_last=True` gives them equal loader
    lengths, so they exhaust and restart on the same step.
    """

    def __init__(
        self,
        dataloader,
        sampler: tp.Optional[DistributedSampler],
        updates_per_epoch: int,
        world_size: int = 1,
    ):
        self._dataloader = dataloader
        self._sampler = sampler
        self._updates_per_epoch = updates_per_epoch
        self._world_size = max(1, world_size)
        self._pass_index = 0
        self._iterator: tp.Optional[tp.Iterator[torch.Tensor]] = None

    @property
    def batches_per_pass(self) -> int:
        """Batches in one pass over this rank's shard."""
        return len(self._dataloader)

    @property
    def steps_per_epoch(self) -> int:
        """Exact number of steps `epoch()` will yield, so the progress bar's
        total and the epoch itself cannot disagree."""
        return min(self._updates_per_epoch, self.batches_per_pass * self._world_size)

    def _start_pass(self) -> None:
        if self._sampler is not None:
            self._sampler.set_epoch(self._pass_index)
        self._pass_index += 1
        self._iterator = iter(self._dataloader)

    def epoch(self) -> tp.Iterator[torch.Tensor]:
        if self.batches_per_pass == 0:
            raise RuntimeError(
                "dataloader produced no batches -- not enough audio for this batch_size "
                f"(and world size {self._world_size})"
            )
        for _ in range(self.steps_per_epoch):
            if self._iterator is None:
                self._start_pass()
            try:
                yield next(tp.cast(tp.Iterator[torch.Tensor], self._iterator))
            except StopIteration:
                self._start_pass()
                try:
                    yield next(tp.cast(tp.Iterator[torch.Tensor], self._iterator))
                except StopIteration:
                    raise RuntimeError(
                        "dataloader produced no batches on a fresh pass -- not enough audio for "
                        "this batch_size (and world size)"
                    ) from None


def train(cfg: TrainConfig) -> None:
    env, device = init_distributed(cfg.device)
    # Rank-dependent seeding: identical seeds on every rank would make the
    # 4x larger effective batch only 1x more diverse. See seed_everything.
    seed_everything(cfg.seed, env)

    generator = build_generator(cfg, device)
    detector = build_detector(cfg, device)
    attack = build_attack(cfg, device)
    perceptual_loss_fn = build_perceptual_loss(cfg, device)
    optimizer = build_optimizer(generator, cfg)
    # DDP wraps the embedder, not the generator, because get_watermark (not
    # forward) is what this loop calls -- see WatermarkEmbedder.
    embedder = wrap_ddp(WatermarkEmbedder(generator), env, device)
    tracker = build_experiment_tracker(cfg, env)

    if env.is_distributed:
        logger.info(
            "distributed training: world_size=%d, batch_size=%d per rank -> effective batch %d",
            env.world_size, cfg.data.batch_size, cfg.data.batch_size * env.world_size,
        )

    dataloader, sampler = build_dataloader(
        cfg.data.train_dir,
        sample_rate=cfg.sample_rate,
        segment_duration=cfg.data.segment_duration,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        env=env,
    )

    # eval_every=0 (default) disables this entirely; a nonzero value needs
    # valid_dir set too, since there's nothing to evaluate on otherwise.
    valid_iter: tp.Optional[tp.Iterator[torch.Tensor]] = None
    if cfg.eval_every:
        if not cfg.data.valid_dir:
            raise ValueError("eval_every is set but data.valid_dir is not -- nothing to evaluate on")
        valid_dataloader, _ = build_dataloader(
            cfg.data.valid_dir,
            sample_rate=cfg.sample_rate,
            segment_duration=cfg.data.segment_duration,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            env=env,
        )
        # The valid sampler's set_epoch is deliberately never called: this is
        # a periodic train-vs-eval loss probe, and holding each rank to a
        # fixed shard keeps that comparison stable across steps rather than
        # adding reshuffling noise to it.
        valid_iter = itertools.cycle(valid_dataloader)  # valid set is usually far smaller than epochs*updates

    # Each run gets its own timestamped subfolder under cfg.checkpoint_dir so
    # that consecutive runs never overwrite each other's generator_epochN.pth
    # files (previously all runs shared the same flat directory).
    #
    # The timestamp is taken from rank 0 and shared, not computed per rank:
    # four independent datetime.now() calls can straddle a second boundary and
    # produce up to four different folder names. Only rank 0 writes
    # checkpoints today, so a split would be latent rather than fatal -- but
    # it would make the path rank-dependent, which anything added later
    # (resume, per-rank artifacts) would inherit as a bug. gather_objects
    # returns every rank's value in rank order, so [0] is rank 0's.
    run_checkpoint_dir = gather_objects(
        os.path.join(cfg.checkpoint_dir, datetime.now().strftime("%Y%m%d_%H%M%S")), env
    )[0]
    if env.is_main:
        os.makedirs(run_checkpoint_dir, exist_ok=True)
    barrier(env)  # no rank may reach the first torch.save before the dir exists

    batch_iterator = EpochBatchIterator(dataloader, sampler, cfg.updates_per_epoch, env.world_size)
    steps_per_epoch = batch_iterator.steps_per_epoch
    if steps_per_epoch < cfg.updates_per_epoch:
        logger.info(
            "epoch is %d steps, not updates_per_epoch=%d: one pass over the data supports only "
            "that many (this rank's loader holds %d batches at world_size=%d). "
            "updates_per_epoch is a cap, not a target -- see EpochBatchIterator",
            steps_per_epoch, cfg.updates_per_epoch, batch_iterator.batches_per_pass, env.world_size,
        )
    elif batch_iterator.batches_per_pass < steps_per_epoch:
        logger.info(
            "per-rank loader has %d batches but an epoch is %d steps, so each epoch makes ~%.2f "
            "passes over this rank's shard (see EpochBatchIterator -- the optimizer-step schedule "
            "matches the single-GPU run, the data is revisited)",
            batch_iterator.batches_per_pass, steps_per_epoch,
            steps_per_epoch / max(batch_iterator.batches_per_pass, 1),
        )

    step = 0
    try:
        for epoch in range(cfg.epochs):
            # Only rank 0 draws the bar: four ranks writing to one tty
            # interleave into unreadable output. EpochBatchIterator makes the
            # total exact, so the bar always ends where the epoch does.
            progress = tqdm(
                total=steps_per_epoch,
                desc=f"epoch {epoch}",
                unit="step",
                leave=False,
                disable=not env.is_main,
            )
            for batch in batch_iterator.epoch():
                metrics = train_step(embedder, detector, attack, perceptual_loss_fn, optimizer, batch, cfg, device)
                scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                # Every rank computed these on its own shard; log the
                # world-wide average, not rank 0's quarter of the batch.
                scalar_metrics = all_reduce_mean(scalar_metrics, env, device)
                tracker.log(scalar_metrics, step=step)
                progress.update(1)
                progress.set_postfix(loss=f"{metrics['loss']:.4f}", attack=metrics["attack"])
                if valid_iter is not None and step % cfg.eval_every == 0:
                    # Every branch, on the SAME batch, keyed by branch name.
                    # One batch for all of them so a difference between
                    # branches is attributable to the branch and not to the
                    # data; explicit names so each series is comparable across
                    # eval points (see eval_step's docstring).
                    #
                    # branch_names is the same list in the same order on every
                    # rank (it is derived from the configured attack weights),
                    # so the per-branch all_reduce_mean below is a collective
                    # every rank enters the same number of times, in the same
                    # order -- otherwise ranks would deadlock against each
                    # other until the NCCL timeout.
                    eval_batch = next(valid_iter)
                    eval_by_branch = {
                        name: eval_step(
                            unwrap_generator(embedder), detector, attack, perceptual_loss_fn,
                            eval_batch, cfg, device, attack_name=name,
                        )
                        for name in attack.branch_names
                    }
                    eval_scalar_metrics = {}
                    for name, eval_metrics in eval_by_branch.items():
                        reduced = all_reduce_mean(
                            {k: v for k, v in eval_metrics.items() if isinstance(v, (int, float))},
                            env, device,
                        )
                        eval_scalar_metrics.update({f"eval/{name}/{k}": v for k, v in reduced.items()})
                    tracker.log(eval_scalar_metrics, step=step)
                    if env.is_main and step % cfg.log_every == 0:
                        logger.info("epoch=%d step=%d eval=%s", epoch, step, eval_by_branch)
                if env.is_main and cfg.tracking.log_audio_every and step % cfg.tracking.log_audio_every == 0:
                    with torch.no_grad():
                        sample_x = batch[:1].to(device)
                        sample_message = random_message(cfg.nbits, 1, device)
                        # Unwrapped generator: this is a rank-0-only no-grad
                        # forward, and a DDP forward that only one rank makes
                        # (and never backwards) would desynchronize the group.
                        sample_x_wm = embed_watermark(
                            unwrap_generator(embedder), sample_x, sample_message,
                            cfg.watermark_snr_db_min, cfg.watermark_snr_db_max,
                        )
                        tracker.log_audio("x_wm_sample", sample_x_wm, cfg.sample_rate, step=step)
                if env.is_main and step % cfg.log_every == 0:
                    logger.info("epoch=%d step=%d %s", epoch, step, metrics)
                step += 1

            progress.close()
            if env.is_main:
                # unwrap_generator, not embedder.state_dict(): a DDP +
                # WatermarkEmbedder state_dict has every key prefixed
                # `module.generator.`, which evaluate.py's
                # load_generator_under_test could not load.
                ckpt_path = f"{run_checkpoint_dir}/generator_epoch{epoch}.pth"
                torch.save({"model": unwrap_generator(embedder).state_dict(), "xp.cfg": cfg}, ckpt_path)
                logger.info("saved checkpoint to %s", ckpt_path)
            barrier(env)  # keep ranks in lockstep across the epoch boundary
    finally:
        tracker.finish()
        cleanup_distributed(env)


def main() -> None:
    configure_logging()
    cfg = load_config()
    train(cfg)


if __name__ == "__main__":
    main()
