# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Losses for generator-only robustness fine-tuning.

NOTE on overlap with AudioCraft's joint-training watermark solver
(`audiocraft/solvers/watermark.py` in the separate AudioCraft checkout, not
part of this project): that solver already computes analogous losses --
`wm_detection` (audiocraft/losses/wmloss.py:WMDetectionLoss, an NLLLoss over
the softmaxed 2-class presence head, which is mathematically a form of BCE)
and `wm_mb` (WMMbLoss, BCEWithLogitsLoss on the raw per-bit logits before
sigmoid) for detection, and `mel`/`msspec` (audiocraft/losses/specloss.py,
plain L1/multi-scale-L1+L2 log-mel distance, no psychoacoustic weighting) for
perceptual quality, combined either via a raw weighted sum (detection) or via
an adaptive gradient-norm balancer (perceptual). Do not additionally enable
those losses on the same training run as this module's `detection_loss` /
`PsychoacousticMelLoss` -- they measure the same thing and would double-count.
The two are not numerically interchangeable: this module's losses are computed
directly on `AudioSealDetector`'s already-activated outputs (softmax
probabilities / sigmoid probabilities), so they use `BCELoss` rather than
`BCEWithLogitsLoss` -- see `detection_loss` below.
"""

import contextlib
import typing as tp

import torch
import torch.nn as nn
import torchaudio


def _no_autocast(device: torch.device) -> tp.ContextManager[None]:
    """Force-disable autocast for an op that can't tolerate it (see
    detection_loss_components). cuda/cpu only, matching train.py's
    _autocast -- torch.autocast's device_type is validated even when
    enabled=False, and MPS's autocast support is inconsistent across torch
    versions, so skip it there rather than risk an unsupported device_type."""
    if device.type in ("cuda", "cpu"):
        return torch.autocast(device_type=device.type, enabled=False)
    return contextlib.nullcontext()


def detection_loss_components(
    p: torch.Tensor,
    m_hat: torch.Tensor,
    message: torch.Tensor,
    presence_target: float = 1.0,
    eps: float = 1e-6,
) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same computation as `detection_loss`, but also returns the two terms
    separately -- for logging (see train.py's train_step), where "all the
    losses" means presence and bit-message loss individually, not just their
    sum.

    Args: same as `detection_loss`.

    Returns:
        (presence_loss + bit_loss, presence_loss, bit_loss) -- all scalars.
    """
    # Forces fp32 regardless of any ambient bf16/fp16 autocast (see train.py's
    # _autocast): BCELoss/binary_cross_entropy raises rather than silently
    # casting under autocast -- "Many models use a sigmoid layer right before
    # the binary cross entropy layer... combine the two ... using
    # binary_cross_entropy_with_logits" -- but p/m_hat here are
    # AudioSealDetector's own already-activated outputs (frozen pretrained
    # model; see this module's docstring), not logits we control, so
    # switching to the *WithLogits variant isn't an option -- force the
    # precision this op actually requires instead.
    with _no_autocast(p.device):
        p = p.float()
        m_hat = m_hat.float()
        bce = nn.BCELoss()
        target_presence = torch.full_like(p, presence_target)
        presence_loss = bce(p.clamp(eps, 1 - eps), target_presence)

        bit_loss = bce(m_hat.clamp(eps, 1 - eps), message.float())

    return presence_loss + bit_loss, presence_loss, bit_loss


def detection_loss(
    p: torch.Tensor,
    m_hat: torch.Tensor,
    message: torch.Tensor,
    presence_target: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """BCE(presence_target, p) + BCE(m, m_hat), bit-wise.

    Args:
        p: presence probability, shape (B,). Already a probability in [0, 1]
            (e.g. mean-pooled over time from the detector's softmaxed
            presence head) -- NOT a logit.
        m_hat: decoded message probabilities, shape (B, nbits). Already a
            probability in [0, 1] (AudioSealDetector.decode_message applies
            sigmoid internally) -- NOT logits.
        message: ground-truth message bits, shape (B, nbits), values in {0, 1}.
        presence_target: target label for "watermark present" (1.0 for a
            positive example; pass 0.0 if you ever call this on unwatermarked
            audio).
        eps: clamp margin so BCELoss never sees exact 0/1 probabilities
            (which would produce -inf/NaN gradients).

    Returns:
        Scalar loss: presence term + message-bit term. Use
        `detection_loss_components` instead if you also want the two terms
        separately.
    """
    loss, _, _ = detection_loss_components(p, m_hat, message, presence_target, eps)
    return loss


def _absolute_threshold_of_hearing_db(freq_hz: torch.Tensor) -> torch.Tensor:
    """Terhardt (1979) approximation of the absolute threshold of hearing, in
    dB SPL, as a function of frequency in Hz. Lower threshold == quieter
    sounds are audible at that frequency == that frequency matters more
    perceptually.

    This is a *static, frequency-only* masking model: it does not account for
    simultaneous masking (loud content hiding nearby quiet content in the
    same frame) or temporal masking (pre/post-masking around transients). If
    you need those, look at a full model (e.g. MPEG-1 Psychoacoustic Model
    II); this is a deliberately lightweight proxy that's enough to
    down-weight inaudible/near-inaudible frequency bands (e.g. near-Nyquist
    content at 16kHz) in the mel loss.
    """
    f_khz = (freq_hz / 1000.0).clamp(min=1e-3)
    return (
        3.64 * f_khz.pow(-0.8)
        - 6.5 * torch.exp(-0.6 * (f_khz - 3.3).pow(2))
        + 1e-3 * f_khz.pow(4)
    )


class PsychoacousticMelLoss(nn.Module):
    """L1 distance between log-mel spectrograms of `x` and `x_wm`, with each
    mel bin weighted by an absolute-threshold-of-hearing curve so that
    perturbation energy placed in less-audible frequency bands is penalized
    less than energy placed in frequency bands where the ear is most
    sensitive (roughly 1-5kHz).

    Weights are normalized to mean 1 so the overall loss magnitude stays
    comparable to an unweighted mel-L1 loss (only the *relative* weighting
    across bins changes).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        f_min: float = 20.0,
        f_max: tp.Optional[float] = None,
        floor_level: float = 1e-5,
    ):
        super().__init__()
        f_max = f_max if f_max is not None else sample_rate / 2
        self.floor_level = floor_level
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=1.0,
            center=True,
        )

        mel_center_freqs = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )
        # melscale_fbanks returns (n_freqs, n_mels) triangular filters; take
        # each filter's peak (center) linear-frequency bin as its representative
        # frequency.
        freqs_hz = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        center_freqs = freqs_hz[mel_center_freqs.argmax(dim=0)]

        ath_db = _absolute_threshold_of_hearing_db(center_freqs)
        weight = 10 ** (-ath_db / 20.0)
        weight = weight / weight.mean()
        self.register_buffer("mel_weight", weight.view(1, -1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)
        self.mel.to(x.device)

        log_mel_x = torch.log10(self.floor_level + self.mel(x))
        log_mel_y = torch.log10(self.floor_level + self.mel(y))

        weighted_abs_diff = (log_mel_x - log_mel_y).abs() * self.mel_weight.to(x.device)
        return weighted_abs_diff.mean()
