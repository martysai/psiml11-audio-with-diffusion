# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generator-only fine-tuning against sampled reconstruction attacks.

Per training step:
  x --[frozen-except-here G_theta]--> x_wm = x + G_theta(x, m)
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

import logging
import os
import random
import typing as tp

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from audioseal import AudioSeal
from audioseal.models import AudioSealDetector, AudioSealWM

from .attacks import BigVGANAttack, DACAttack, IdentityAttack, SGMSEAttack, SampledReconstructionAttack
from .config import TrainConfig, load_config
from .data import build_dataloader
from .device import resolve_device
from .losses import PsychoacousticMelLoss, detection_loss
from .tracking import ExperimentTracker, build_tracker

logger = logging.getLogger(__name__)


def random_message(nbits: int, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, 2, (batch_size, nbits), device=device)


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
    if someone mistakenly passes detector.parameters() somewhere."""
    detector = AudioSeal.load_detector(cfg.detector.checkpoint, nbits=cfg.nbits, device=device)
    detector.eval()
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
    }
    weights = {
        "identity": cfg.attack.weights.identity,
        "bigvgan": cfg.attack.weights.bigvgan,
        "dac": cfg.attack.weights.dac,
        "sgmse": cfg.attack.weights.sgmse,
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

    # 1. x_wm = x + G_theta(x, m)
    watermark = generator.get_watermark(x, message=message)
    x_wm = x + watermark

    # 2. sampled reconstruction attack (frozen, graph stays connected)
    x_att, attack_name = attack(x_wm)

    # 3. frozen detector: presence (B,2,T) softmax probs, message (B,nbits) sigmoid probs
    presence, m_hat = detector.forward(x_att)
    p = presence[:, 1, :].mean(dim=-1)  # presence prob per example, pooled over time

    # 4. losses
    det_loss = detection_loss(p, m_hat, message)
    perc_loss = perceptual_loss_fn(x, x_wm)  # pre-attack, per spec
    total_loss = cfg.lambda_det * det_loss + cfg.lambda_perc * perc_loss

    # 5. backprop through detector + attack (frozen but differentiable) into G_theta
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    if cfg.optim.max_norm is not None:
        torch.nn.utils.clip_grad_norm_(generator.parameters(), cfg.optim.max_norm)
    optimizer.step()

    return {
        "loss": total_loss.item(),
        "detection_loss": det_loss.item(),
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
    torch.manual_seed(cfg.seed)
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

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    step = 0
    try:
        for epoch in range(cfg.epochs):
            for batch in dataloader:
                metrics = train_step(generator, detector, attack, perceptual_loss_fn, optimizer, batch, cfg, device)
                scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                tracker.log(scalar_metrics, step=step)
                if cfg.tracking.log_audio_every and step % cfg.tracking.log_audio_every == 0:
                    with torch.no_grad():
                        sample_x = batch[:1].to(device)
                        sample_wm = generator.get_watermark(sample_x, message=random_message(cfg.nbits, 1, device))
                        tracker.log_audio("x_wm_sample", sample_x + sample_wm, cfg.sample_rate, step=step)
                if step % cfg.log_every == 0:
                    logger.info("epoch=%d step=%d %s", epoch, step, metrics)
                step += 1
                if step % cfg.updates_per_epoch == 0:
                    break

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
