# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for fitting short audio to AudioLDM's fixed 10.24s window.

`AudioLDMAttack` runs a backbone that consumes exactly 10.24s, so shorter
input has to be extended. It used to be zero-padded, which is catastrophic for
this model: the mel goes through log(clamp(m, min=1e-5)), so silence becomes a
constant -11.51 plateau. At segment_duration=2.0 that pinned 80.5% of the
spectrogram at the floor, and the UNet's attention spans all 256 latent frames
so the silence mixed into the real audio rather than staying in the padded
region. Observed consequence: attack_sisnr = -50 dB for AudioLDM against
-8.8 dB for MBD, a comparable diffusion resynthesis run at native length.

`_tile_or_crop` repeats the audio instead, keeping the statistics natural.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from audioseal_robust.attacks import _tile_or_crop  # noqa: E402

SR = 16000
TARGET = int(round(SR * 10.24))  # 163840, AudioLDM's window


def test_short_input_is_tiled_to_exactly_the_target():
    wav = torch.randn(2, SR * 2)  # 2s, the segment_duration we train at

    out = _tile_or_crop(wav, TARGET)

    assert out.shape == (2, TARGET)


def test_tiling_repeats_the_signal_rather_than_appending_silence():
    """The property that matters: no region of the output is digital silence."""
    wav = torch.randn(1, SR * 2) * 0.5

    out = _tile_or_crop(wav, TARGET)

    # The first tile is the original.
    assert torch.allclose(out[:, : SR * 2], wav)
    # The second tile repeats it.
    assert torch.allclose(out[:, SR * 2 : SR * 4], wav)
    # Nothing anywhere is a silent block.
    block = SR // 10
    for start in range(0, TARGET - block, block):
        assert out[:, start : start + block].abs().max() > 0, "found a silent region"


def test_long_input_is_cropped():
    wav = torch.randn(2, TARGET * 2)

    out = _tile_or_crop(wav, TARGET)

    assert out.shape == (2, TARGET)
    assert torch.allclose(out, wav[:, :TARGET])


def test_exact_length_is_returned_unchanged():
    wav = torch.randn(3, TARGET)
    assert torch.allclose(_tile_or_crop(wav, TARGET), wav)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty waveform"):
        _tile_or_crop(torch.zeros(1, 0), TARGET)


def test_is_differentiable_and_accumulates_over_tiles():
    """Gradients must reach the input through every tile -- that is the true
    Jacobian of tile-then-crop, and it is what keeps the attack usable in the
    training graph."""
    wav = torch.randn(1, 100, requires_grad=True)

    out = _tile_or_crop(wav, 250)  # 2 full tiles + half of a third
    out.sum().backward()

    assert wav.grad is not None
    # Samples 0..49 appear 3 times in the output, samples 50..99 twice.
    assert torch.allclose(wav.grad[0, :50], torch.full((50,), 3.0))
    assert torch.allclose(wav.grad[0, 50:], torch.full((50,), 2.0))


def test_tiling_avoids_the_mel_floor_that_zero_padding_creates():
    """The end-to-end reason this exists, checked on a real log-mel.

    Zero-padding drives most of the spectrogram to log(1e-5); tiling leaves
    none of it there.
    """
    torchaudio = pytest.importorskip("torchaudio")

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=1024, win_length=1024, hop_length=160,
        f_min=0.0, f_max=8000.0, n_mels=64, power=1.0,
    )
    floor = float(np.log(1e-5))

    def log_mel(w):
        return torch.log(torch.clamp(mel(w), min=1e-5))

    t = torch.arange(SR * 2) / SR
    # Broadband, like speech: a pure tone would leave most mel bins at the
    # floor even when tiled, which would test the signal rather than the fix.
    speech = sum((1.0 / k) * torch.sin(2 * np.pi * 220 * k * t) for k in range(1, 20))
    speech = (speech / speech.abs().max() * 0.5).unsqueeze(0)

    zero_padded = torch.nn.functional.pad(speech, (0, TARGET - speech.shape[-1]))
    tiled = _tile_or_crop(speech, TARGET)

    frac_at_floor = lambda w: (log_mel(w) <= floor + 1e-6).float().mean().item()  # noqa: E731

    assert frac_at_floor(zero_padded) > 0.5, "test setup: padding should hit the floor"
    assert frac_at_floor(tiled) < 0.01, "tiling must not produce a silence plateau"
    # And the distribution the model actually sees is no longer dragged down
    # toward the floor by a silent majority.
    assert log_mel(tiled).mean() > log_mel(zero_padded).mean() + 4
