# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Dump listenable/viewable audio for a handful of evaluated examples.

`evaluate.py` keeps only scores: every attacked waveform it produces is
discarded as soon as the detector has scored it. That makes the headline
numbers cheap to compute but impossible to sanity-check by ear -- an attack
reporting 27 dB SI-SNR and one reporting -49 dB both reduce to a single
number in the table, even though one leaves speech intact and the other
destroys it.

This module re-derives a small number of those examples and writes them out
as .wav files plus per-example figures, WITHOUT re-running a full evaluation
(a full HopSkipJump pass is ~1.7 h; six examples is ~1 min).

    python -m audioseal_robust.dump_audio eval_dir=... --n-examples 6
    python -m audioseal_robust.dump_audio eval_dir=... --attack identity

Everything except the dump-specific `--flags` uses the same `key=value`
override grammar as `evaluate.py`, so a dump can be pointed at exactly the
config a run used by copying its command line.

Reproducibility, precisely:

  * `clean` and `watermarked` are BIT-IDENTICAL to what the corresponding
    evaluation run scored, provided `seed`, `eval_dir`, `segment_duration`,
    `batch_size` and `generator_checkpoint` match. The dataset sorts its
    files and the loader does not shuffle, so example i is the same crop of
    the same file; the message and the SNR-targeting scale come off the same
    seeded RNG in the same order.
  * `attacked` is NOT bit-identical for a stochastic attack such as
    hopskipjump. Its random draws depend on how many examples preceded it,
    and this dumps a prefix rather than all 400. It is the same attack, at
    the same configuration, on the same input -- a representative sample,
    not a replay. The per-example CSV reports each example's own measured
    numbers so nothing here has to be taken on faith.
"""

import argparse
import csv
import logging
import random
import typing as tp
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audioseal import AudioSeal

from .config import load_eval_config
from .data import build_dataloader
from .device import resolve_device
from .evaluate import (
    _REFERENCE_POOL_BATCHES,
    _STRENGTH_AWARE_ATTACKS,
    build_eval_attacks,
    load_generator_under_test,
)
from .metrics import bit_accuracy, sisnr_score
from .train import embed_watermark, random_message

logger = logging.getLogger(__name__)

# Residuals sit ~30 dB below the signal (watermark) or lower, so writing
# them at their true amplitude produces a file that sounds like silence.
# They are peak-normalized to this level instead, and the gain applied is
# recorded per example so the loudness is never mistaken for the real one.
_RESIDUAL_PEAK = 0.7


def _detection_score(detector, wav: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """Same quantity `evaluate_attack` thresholds: the per-example mean over
    time of the detector's "watermarked" logit channel."""
    presence, message = detector.forward(wav)
    return presence[:, 1, :].mean(dim=-1), message


def _write_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> int:
    """Write a (1, T) or (T,) tensor as 16-bit PCM. Returns the number of
    samples that had to be clipped -- adding a watermark and then an attack
    perturbation can push a signal that was already near full scale past it,
    and silently clipping would be indistinguishable from attack damage."""
    audio = wav.detach().cpu().reshape(-1).float()
    n_clipped = int((audio.abs() > 1.0).sum().item())
    sf.write(str(path), audio.clamp(-1.0, 1.0).numpy(), sample_rate, subtype="PCM_16")
    return n_clipped


def _write_normalized_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> float:
    """Write a residual at an audible level. Returns the gain applied, in dB."""
    audio = wav.detach().cpu().reshape(-1).float()
    peak = audio.abs().max().item()
    if peak <= 0:
        sf.write(str(path), audio.numpy(), sample_rate, subtype="PCM_16")
        return 0.0
    gain = _RESIDUAL_PEAK / peak
    sf.write(str(path), (audio * gain).numpy(), sample_rate, subtype="PCM_16")
    return 20.0 * float(np.log10(gain))


def _plot_example(
    path: Path,
    panels: tp.List[tp.Tuple[str, torch.Tensor]],
    sample_rate: int,
) -> None:
    """Waveform + spectrogram per panel, sharing one figure.

    Waveforms are drawn on a shared y-scale so the watermark's amplitude can
    be compared against the attack's by eye; the residual panels are exempt
    (they would be invisible at that scale) and are labelled with their own
    peak instead.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signal_peak = max(
        float(w.detach().abs().max()) for name, w in panels if not name.startswith("residual")
    )
    fig, axes = plt.subplots(len(panels), 2, figsize=(13, 2.1 * len(panels)))
    for row, (name, wav) in enumerate(panels):
        audio = wav.detach().cpu().reshape(-1).float().numpy()
        t = np.arange(audio.size) / sample_rate

        ax = axes[row][0]
        ax.plot(t, audio, linewidth=0.5)
        if name.startswith("residual"):
            ax.set_ylabel(f"{name}\n(peak {np.abs(audio).max():.1e})", fontsize=8)
        else:
            ax.set_ylim(-signal_peak * 1.1, signal_peak * 1.1)
            ax.set_ylabel(name, fontsize=8)
        ax.set_xlim(0, t[-1] if t.size else 1)
        ax.tick_params(labelsize=7)
        if row < len(panels) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time (s)", fontsize=8)

        ax = axes[row][1]
        # NFFT=512 at 16 kHz -> 32 ms windows: fine enough to show the
        # attack's broadband noise floor against speech harmonics.
        ax.specgram(audio, NFFT=512, Fs=sample_rate, noverlap=384, cmap="magma")
        ax.tick_params(labelsize=7)
        if row < len(panels) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("Hz", fontsize=8)

    fig.suptitle(path.stem, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump clean/watermarked/attacked audio for a few evaluated examples.",
    )
    parser.add_argument("--n-examples", type=int, default=6,
                        help="how many examples to dump (default: 6)")
    parser.add_argument("--attack", default="hopskipjump",
                        help="attack to apply (default: hopskipjump)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="destination (default: <output_dir>/audio_<label>_<attack>)")
    parser.add_argument("--no-figures", action="store_true",
                        help="write .wav files only")
    args, overrides = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_eval_config(overrides)
    device = resolve_device(cfg.device)

    # Identical seeding to evaluate.run(), in the same order, so the clean
    # and watermarked signals below are the very ones that run scored.
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    generator = load_generator_under_test(cfg.generator_checkpoint, cfg.nbits, device)
    generator.eval()
    detector = AudioSeal.load_detector(cfg.detector_checkpoint, nbits=cfg.nbits, device=device)
    detector.eval()

    attacks, skipped = build_eval_attacks([args.attack], device, cfg)
    if args.attack in skipped:
        raise SystemExit(f"attack {args.attack!r} could not be constructed: {skipped[args.attack]}")
    attack = attacks[args.attack]

    dataloader = build_dataloader(
        cfg.eval_dir,
        sample_rate=cfg.sample_rate,
        segment_duration=cfg.segment_duration,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    # Mirrors evaluate.run()'s preparation exactly (same call order, same
    # RNG consumption) but stops as soon as enough examples exist. The
    # reference pool a query-based attack needs is still built from
    # _REFERENCE_POOL_BATCHES batches, matching the real run's pool size.
    n_batches = max(
        _REFERENCE_POOL_BATCHES,
        -(-args.n_examples // cfg.batch_size),  # ceil
    )
    prepared = []
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= n_batches:
            break
        x = batch.to(device)
        message = random_message(cfg.nbits, x.size(0), device=device)
        x_wm = embed_watermark(generator, x, message, cfg.watermark_snr_db, cfg.watermark_snr_db)
        prepared.append((x, x_wm, message))
    if not prepared:
        raise SystemExit(f"no batches produced from eval_dir={cfg.eval_dir!r}")

    bind_detector = getattr(attack, "bind_detector", None)
    if bind_detector is not None:
        bind_detector(detector)
    bind_reference_pool = getattr(attack, "bind_reference_pool", None)
    if bind_reference_pool is not None:
        pool = torch.cat(
            [
                torch.cat([x for x, _, _ in prepared[:_REFERENCE_POOL_BATCHES]]),
                torch.cat([x_wm for _, x_wm, _ in prepared[:_REFERENCE_POOL_BATCHES]]),
            ]
        )
        bind_reference_pool(pool)
        logger.info("bound a %d-waveform reference pool for query-based init", pool.shape[0])

    out_dir = args.out_dir or Path(cfg.output_dir) / f"audio_{cfg.label}_{args.attack}"
    out_dir.mkdir(parents=True, exist_ok=True)

    strength = cfg.headline_strength if args.attack in _STRENGTH_AWARE_ATTACKS else None
    rows = []
    dumped = 0
    for x, x_wm, message in prepared:
        if dumped >= args.n_examples:
            break
        take = min(x.size(0), args.n_examples - dumped)
        x, x_wm, message = x[:take], x_wm[:take], message[:take]

        logger.info("attacking examples %d-%d with %s", dumped, dumped + take - 1, args.attack)
        x_att = attack(x_wm, strength=strength)

        score_clean, _ = _detection_score(detector, x)
        score_wm, msg_wm = _detection_score(detector, x_wm)
        score_att, msg_att = _detection_score(detector, x_att)

        for i in range(take):
            idx = dumped + i
            stem = f"{idx:02d}"
            wm_residual = x_wm[i] - x[i]
            attack_delta = x_att[i] - x_wm[i]

            clipped = sum((
                _write_wav(out_dir / f"{stem}_1_clean.wav", x[i], cfg.sample_rate),
                _write_wav(out_dir / f"{stem}_2_watermarked.wav", x_wm[i], cfg.sample_rate),
                _write_wav(out_dir / f"{stem}_3_attacked.wav", x_att[i], cfg.sample_rate),
            ))
            wm_gain_db = _write_normalized_wav(
                out_dir / f"{stem}_4_watermark_residual_normalized.wav", wm_residual, cfg.sample_rate
            )
            att_gain_db = _write_normalized_wav(
                out_dir / f"{stem}_5_attack_perturbation_normalized.wav", attack_delta, cfg.sample_rate
            )

            if not args.no_figures:
                _plot_example(
                    out_dir / f"{stem}.png",
                    [
                        ("clean", x[i]),
                        ("watermarked", x_wm[i]),
                        ("attacked", x_att[i]),
                        ("residual: watermark", wm_residual),
                        ("residual: attack", attack_delta),
                    ],
                    cfg.sample_rate,
                )

            rows.append({
                "example": idx,
                "label": cfg.label,
                "attack": args.attack,
                "score_clean": float(score_clean[i]),
                "score_watermarked": float(score_wm[i]),
                "score_attacked": float(score_att[i]),
                "bit_acc_watermarked": bit_accuracy(msg_wm[i : i + 1], message[i : i + 1]),
                "bit_acc_attacked": bit_accuracy(msg_att[i : i + 1], message[i : i + 1]),
                "watermark_sisnr_db": sisnr_score(x[i : i + 1], x_wm[i : i + 1]),
                "attack_sisnr_db": sisnr_score(x_wm[i : i + 1], x_att[i : i + 1]),
                "watermark_residual_gain_db": round(wm_gain_db, 1),
                "attack_perturbation_gain_db": round(att_gain_db, 1),
                "clipped_samples": clipped,
            })
        dumped += take

    csv_path = out_dir / "per_example.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {dumped} examples to {out_dir}")
    print(f"  per-example metrics: {csv_path}")
    print(f"\n  {'ex':>2}  {'score_clean':>11}  {'score_wm':>9}  {'score_att':>9}  "
          f"{'bitacc_wm':>9}  {'bitacc_att':>10}  {'attack_snr':>10}")
    for r in rows:
        print(f"  {r['example']:>2}  {r['score_clean']:>11.4f}  {r['score_watermarked']:>9.4f}  "
              f"{r['score_attacked']:>9.4f}  {r['bit_acc_watermarked']:>9.4f}  "
              f"{r['bit_acc_attacked']:>10.4f}  {r['attack_sisnr_db']:>9.1f}dB")
    total_clipped = sum(r["clipped_samples"] for r in rows)
    if total_clipped:
        print(f"\n  note: {total_clipped} sample(s) clipped to [-1, 1] on write")


if __name__ == "__main__":
    main()
