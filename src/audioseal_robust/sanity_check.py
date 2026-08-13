# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Sanity check for the training loop: does the loss actually go down, and
does it train at a reasonable speed on the resolved device (GPU if
available)?

This does NOT use a real dataset -- Ana's dataset isn't ready yet. It uses
a single fixed batch of synthetic (random noise) "audio" and deliberately
overfits it: a correct, correctly-wired training loop should be able to
drive the loss on one fixed batch down sharply within a few hundred steps,
regardless of whether the audio content is real speech. That makes this a
test of the training *mechanics* (does backprop actually reach the
generator, is the loss formula sane, is nothing accidentally frozen or
disconnected) and of *throughput*, not of watermark/audio quality -- swap
in `audioseal_robust.data.build_dataloader` against real audio once it's
available for an actual training run.

Use `--pretrained` to load the real facebook/audioseal checkpoints (needs
network access to Hugging Face) instead of a random init. This matters more
than it might sound: with a random-init generator+detector pair (i.e.
neither has ever learned to embed/detect a watermark), the perceptual loss's
log-mel term has a very steep gradient right at "silent" (small `x`, and
`log10(floor_level + x)` blows up as `x -> 0`), which can push the generator
to collapse its watermark amplitude to ~0 before the (initially weak,
poorly-conditioned) detection gradient gets any traction -- loss then goes
flat and stays flat. That's a real, reproducible local minimum, not a bug in
the wiring; it happens because there's nothing yet in a random-init detector
that meaningfully rewards a small nonzero perturbation. It is also not
representative of the actual use case here, which is fine-tuning an
*already pretrained, already-working* generator/detector pair -- with
`--pretrained`, the perturbation starts at a working nonzero operating
point and the detection gradient is strong and well-conditioned from step 0,
so this collapse doesn't happen. Prefer `--pretrained` when you have network
access; use plain (random-init) mode only to sanity-check mechanics/speed
offline, and consider raising `--lambda-det` there if you want it to escape
the collapse and actually converge too.

Usage:
    python -m audioseal_robust.sanity_check
    python -m audioseal_robust.sanity_check --device cuda --steps 500 --pretrained
"""

import argparse
import logging
import time

import torch

from .attacks import IdentityAttack, SampledReconstructionAttack
from .device import resolve_device
from .losses import PsychoacousticMelLoss, detection_loss
from .model_init import build_untrained_detector, build_untrained_generator
from .tracking import build_tracker

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seconds", type=float, default=1.0, help="synthetic audio segment length")
    p.add_argument("--nbits", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4, help="higher than the production default -- this is a fast overfit smoke test, not real training")
    p.add_argument("--lambda-det", type=float, default=1.0)
    p.add_argument("--lambda-perc", type=float, default=1.0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--pretrained",
        action="store_true",
        help="load real facebook/audioseal checkpoints from Hugging Face instead of random init (needs network)",
    )
    p.add_argument("--tracking-backend", type=str, default="wandb", choices=["mlflow", "wandb", "none"])
    p.add_argument("--tracking-project", type=str, default="audioseal-robust-sanity")
    p.add_argument(
        "--pass-threshold",
        type=float,
        default=0.5,
        help="PASS requires end-of-run loss < this fraction of the start-of-run loss",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    device = resolve_device(args.device)
    logger.info("resolved device: %s", device)

    if args.pretrained:
        from audioseal import AudioSeal

        generator = AudioSeal.load_generator("audioseal_wm_16bits", nbits=args.nbits, device=device)
        detector = AudioSeal.load_detector("audioseal_detector_16bits", nbits=args.nbits, device=device)
    else:
        generator = build_untrained_generator("audioseal_wm_16bits", nbits=args.nbits, device=device)
        detector = build_untrained_detector("audioseal_detector_16bits", nbits=args.nbits, device=device)

    generator.train()
    detector.eval()
    for p in detector.parameters():
        p.requires_grad_(False)

    attack = SampledReconstructionAttack({"identity": IdentityAttack()}, {"identity": 1.0}).to(device)
    perceptual_loss_fn = PsychoacousticMelLoss(sample_rate=16_000).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.9))

    tracker = build_tracker(args.tracking_backend, args.tracking_project, config=vars(args))

    torch.manual_seed(0)
    n_samples = int(args.seconds * 16_000)
    x = torch.randn(args.batch_size, 1, n_samples, device=device)
    message = torch.randint(0, 2, (args.batch_size, args.nbits), device=device)

    losses = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    for step in range(args.steps):
        watermark = generator.get_watermark(x, message=message)
        x_wm = x + watermark
        x_att, _ = attack(x_wm)
        presence, m_hat = detector.forward(x_att)
        p = presence[:, 1, :].mean(dim=-1)

        det_loss = detection_loss(p, m_hat, message.float())
        perc_loss = perceptual_loss_fn(x, x_wm)
        total_loss = args.lambda_det * det_loss + args.lambda_perc * perc_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        loss_value = total_loss.item()
        losses.append(loss_value)
        tracker.log(
            {
                "loss": loss_value,
                "detection_loss": det_loss.item(),
                "perceptual_loss": perc_loss.item(),
                "presence_prob": p.mean().item(),
            },
            step=step,
        )
        if step % 20 == 0:
            logger.info(
                "step=%d loss=%.4f det=%.4f perc=%.4f presence_prob=%.3f",
                step, loss_value, det_loss.item(), perc_loss.item(), p.mean().item(),
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    tracker.finish()

    k = max(1, len(losses) // 10)
    start_loss = sum(losses[:k]) / k
    end_loss = sum(losses[-k:]) / k
    steps_per_sec = args.steps / elapsed

    passed = end_loss < start_loss * args.pass_threshold

    print("\n=== Sanity check summary ===")
    print(f"device:              {device}")
    print(f"steps:                {args.steps}  batch_size: {args.batch_size}  segment: {args.seconds}s")
    print(f"start loss (first {k} steps avg): {start_loss:.4f}")
    print(f"end loss   (last {k} steps avg):  {end_loss:.4f}")
    print(f"wall time:            {elapsed:.1f}s  ({steps_per_sec:.2f} steps/sec)")
    print(f"result:               {'PASS' if passed else 'FAIL'} (need end_loss < {args.pass_threshold} * start_loss)")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
