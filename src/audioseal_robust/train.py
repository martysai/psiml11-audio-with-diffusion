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
"""

import contextlib
import itertools
import logging
import os
import random
import typing as tp

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
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
from .device import resolve_device
from .losses import PsychoacousticMelLoss, detection_loss_components
from .tracking import ExperimentTracker, build_tracker

logger = logging.getLogger(__name__)


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


def embed_watermark(
    generator: AudioSealWM,
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

    x: (B, 1, T). Gradients flow through normally (delta is not detached),
    same as before this scaling was introduced.
    """
    delta = generator.get_watermark(x, message=message)
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
    attack = SampledReconstructionAttack(attacks, weights)
    return attack.to(device)


def build_optimizer(generator: AudioSealWM, cfg: TrainConfig) -> torch.optim.Optimizer:
    # Only generator.parameters() -- this is the actual mechanism that keeps
    # the detector and attack module untrained, not just requires_grad.
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


def train_step(
    generator: AudioSealWM,
    detector: AudioSealDetector,
    attack: SampledReconstructionAttack,
    perceptual_loss_fn: PsychoacousticMelLoss,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
) -> tp.Dict[str, tp.Union[float, str]]:
    x = batch.to(device)  # clean audio, (B, 1, T), 16kHz
    message = random_message(cfg.nbits, x.size(0), device=x.device)

    # 1-4 under bf16 autocast (see _autocast's docstring): forward through
    # the generator, attack, and detector, plus loss computation. Backward
    # (below) stays outside -- autocast only needs to cover the forward pass,
    # and the hooks registered on x_wm/x_att fire during backward regardless
    # of which context the tensors were originally produced under.
    with _autocast(device):
        # 1. x_wm = x + scale * G_theta(x, m), scale set per-example to hit a
        #    randomly sampled target watermark SNR -- see embed_watermark.
        x_wm = embed_watermark(generator, x, message, cfg.watermark_snr_db_min, cfg.watermark_snr_db_max)

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

        x_wm.register_hook(_capture_activation_grad_norm("x_wm"))

        # 2. sampled reconstruction attack (frozen, graph stays connected)
        x_att, attack_name = attack(x_wm)
        x_att.register_hook(_capture_activation_grad_norm("x_att"))

        # 3. frozen detector: presence (B,2,T) softmax probs, message (B,nbits) sigmoid probs
        presence, m_hat = detector.forward(x_att)
        p = presence[:, 1, :].mean(dim=-1)  # presence prob per example, pooled over time

        # 4. losses
        det_loss, presence_loss, bit_loss = detection_loss_components(p, m_hat, message)
        perc_loss = perceptual_loss_fn(x, x_wm)  # pre-attack, per spec
        total_loss = cfg.lambda_det * det_loss + cfg.lambda_perc * perc_loss

    # 5. backprop through detector + attack (frozen but differentiable) into G_theta
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm_generator = _grad_norm(generator)
    grad_norm_attack = _grad_norm(attack)  # expected: always 0.0, see _grad_norm's docstring
    if cfg.optim.max_norm is not None:
        torch.nn.utils.clip_grad_norm_(generator.parameters(), cfg.optim.max_norm)
    optimizer.step()

    return {
        "loss": total_loss.item(),
        "detection_loss": det_loss.item(),
        "presence_loss": presence_loss.item(),
        "bit_loss": bit_loss.item(),
        "perceptual_loss": perc_loss.item(),
        "presence_prob": p.mean().item(),
        "grad_norm_generator": grad_norm_generator,
        "grad_norm_attack": grad_norm_attack,
        "grad_norm_x_wm": activation_grad_norms.get("x_wm", 0.0),
        "grad_norm_x_att": activation_grad_norms.get("x_att", 0.0),
        "attack": attack_name,
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
) -> tp.Dict[str, tp.Union[float, str]]:
    """Same forward computation as train_step (same loss formula, same
    sampled-attack call), just no backward/optimizer step -- for periodic
    train-vs-eval loss comparison during training (see TrainConfig.eval_every
    and DataConfig.valid_dir)."""
    generator.eval()
    try:
        x = batch.to(device)
        message = random_message(cfg.nbits, x.size(0), device=x.device)
        with _autocast(device):
            x_wm = embed_watermark(generator, x, message, cfg.watermark_snr_db_min, cfg.watermark_snr_db_max)
            x_att, attack_name = attack(x_wm)
            presence, m_hat = detector.forward(x_att)
            p = presence[:, 1, :].mean(dim=-1)
            det_loss, presence_loss, bit_loss = detection_loss_components(p, m_hat, message)
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
        "attack": attack_name,
    }


def build_experiment_tracker(cfg: TrainConfig) -> ExperimentTracker:
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    assert isinstance(config_dict, dict)
    return build_tracker(
        backend=cfg.tracking.backend,
        project=cfg.tracking.project,
        run_name=cfg.tracking.run_name,
        config=config_dict,
        mlflow_tracking_uri=cfg.tracking.mlflow_tracking_uri,
        wandb_mode=cfg.tracking.wandb_mode,
    )


def train(cfg: TrainConfig) -> None:
    # All RNG sources actually touched by this pipeline: torch.manual_seed
    # alone only seeds the default CPU generator, not CUDA's per-device ones
    # (torch.cuda.manual_seed_all covers every visible GPU, harmless no-op if
    # none); numpy has its own independent RNG state (soundfile/librosa/scipy
    # in the sgmse/data path can draw from it); random.seed covers Python
    # stdlib (random_message doesn't use it, but attacks.py's AudioLDMAttack
    # None-strength sampling and SampledReconstructionAttack's attack-name
    # choice do).
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = resolve_device(cfg.device)

    generator = build_generator(cfg, device)
    detector = build_detector(cfg, device)
    attack = build_attack(cfg, device)
    perceptual_loss_fn = build_perceptual_loss(cfg, device)
    optimizer = build_optimizer(generator, cfg)
    tracker = build_experiment_tracker(cfg)

    dataloader = build_dataloader(
        cfg.data.train_dir,
        sample_rate=cfg.sample_rate,
        segment_duration=cfg.data.segment_duration,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # eval_every=0 (default) disables this entirely; a nonzero value needs
    # valid_dir set too, since there's nothing to evaluate on otherwise.
    valid_iter: tp.Optional[tp.Iterator[torch.Tensor]] = None
    if cfg.eval_every:
        if not cfg.data.valid_dir:
            raise ValueError("eval_every is set but data.valid_dir is not -- nothing to evaluate on")
        valid_dataloader = build_dataloader(
            cfg.data.valid_dir,
            sample_rate=cfg.sample_rate,
            segment_duration=cfg.data.segment_duration,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
        )
        valid_iter = itertools.cycle(valid_dataloader)  # valid set is usually far smaller than epochs*updates

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    step = 0
    try:
        for epoch in range(cfg.epochs):
            progress = tqdm(total=cfg.updates_per_epoch, desc=f"epoch {epoch}", unit="step", leave=False)
            # epoch_step (not the global `step`) drives the break below: `step`
            # never resets, so gating the break on step % updates_per_epoch
            # truncates epochs whenever the dataloader's own natural length
            # doesn't evenly divide updates_per_epoch (e.g. a small train_dir
            # with fewer batches/epoch than the cap) -- the global count drifts
            # out of phase with epoch boundaries and can cross a multiple of
            # updates_per_epoch mid-epoch, cutting it short well before its own
            # ~len(dataloader) steps. epoch_step ties the break to THIS epoch's
            # own progress instead, so every epoch runs the same
            # min(len(dataloader), updates_per_epoch) steps, consistently.
            epoch_step = 0
            for batch in dataloader:
                metrics = train_step(generator, detector, attack, perceptual_loss_fn, optimizer, batch, cfg, device)
                scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                tracker.log(scalar_metrics, step=step)
                progress.update(1)
                progress.set_postfix(loss=f"{metrics['loss']:.4f}", attack=metrics["attack"])
                if valid_iter is not None and step % cfg.eval_every == 0:
                    eval_metrics = eval_step(
                        generator, detector, attack, perceptual_loss_fn, next(valid_iter), cfg, device
                    )
                    eval_scalar_metrics = {
                        f"eval/{k}": v for k, v in eval_metrics.items() if isinstance(v, (int, float))
                    }
                    tracker.log(eval_scalar_metrics, step=step)
                    if step % cfg.log_every == 0:
                        logger.info("epoch=%d step=%d eval=%s", epoch, step, eval_metrics)
                if cfg.tracking.log_audio_every and step % cfg.tracking.log_audio_every == 0:
                    with torch.no_grad():
                        sample_x = batch[:1].to(device)
                        sample_message = random_message(cfg.nbits, 1, device)
                        sample_x_wm = embed_watermark(
                            generator, sample_x, sample_message, cfg.watermark_snr_db_min, cfg.watermark_snr_db_max
                        )
                        tracker.log_audio("x_wm_sample", sample_x_wm, cfg.sample_rate, step=step)
                if step % cfg.log_every == 0:
                    logger.info("epoch=%d step=%d %s", epoch, step, metrics)
                step += 1
                epoch_step += 1
                if epoch_step >= cfg.updates_per_epoch:
                    break

            progress.close()
            ckpt_path = f"{cfg.checkpoint_dir}/generator_epoch{epoch}.pth"
            torch.save({"model": generator.state_dict(), "xp.cfg": cfg}, ckpt_path)
            logger.info("saved checkpoint to %s", ckpt_path)
    finally:
        tracker.finish()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    train(cfg)


if __name__ == "__main__":
    main()
