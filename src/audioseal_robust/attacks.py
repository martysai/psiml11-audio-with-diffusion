# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Sampled reconstruction attack module.

Wraps a set of candidate "reconstruction attacks" (resynthesis pipelines that a
real adversary could run on a watermarked signal to try to wash out the
watermark) behind a single module that, on every forward call, samples ONE
attack and applies it.

All attacks here are *frozen but differentiable*: their own parameters are
excluded from training (`requires_grad_(False)`) and they are always kept in
`eval()` mode, but their forward computation is never wrapped in
`torch.no_grad()` and their output is never `.detach()`-ed. That distinction
matters because `requires_grad_(False)` only stops gradients from
*accumulating into the attack's own weights* -- it does not cut the
autograd graph. Gradients still flow *through* the frozen weights back to
whatever produced the attack's input (here, the generator's watermarked
signal `x_wm`), because autograd differentiates the forward computation
itself, not the `requires_grad` flag of the constants used in it.

Contrast this with the existing mp3/aac augmentations in AudioCraft's
watermarking solver (`audiocraft/utils/audio_effects.py`,
`apply_compression_skip_grad`), which run the codec on a `.detach()`-ed
copy of the input and reattach the *difference* with a straight-through
estimator. That is a workaround for compression codecs that have no
meaningful gradient (external, non-differentiable subprocess calls). Since
BigVGAN/DAC/SGMSE/AudioLDM are ordinary neural nets, we don't need that
workaround -- we can and do backprop through them for real (AudioLDM needs a
little more care to get there -- see that class's own docstring for why).
"""

import functools
import os
import random
import typing as tp
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


def _tile_or_crop(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    """Fit `wav` (B, T) to exactly `target_len` samples, tiling if it is short.

    AudioLDM's backbone consumes a fixed 10.24s window (`duration` in
    audioldm_original.yaml), so anything shorter has to be extended somehow.

    Tiling rather than zero-padding, because zero-padding this model is
    catastrophic rather than merely wasteful. Its mel goes through
    log(clamp(m, min=1e-5)), so digital silence becomes a constant -11.51
    plateau while speech sits near 0. At the 2.0s segments used for training
    that leaves 80.5% of the spectrogram pinned at the floor and a mel whose
    mean is ~-9.0, against ~+1.0 for natural audio -- far outside anything the
    pretrained VAE and UNet ever saw. Nor does it stay confined to the padded
    region: with use_spatial_transformer the UNet's attention spans all 256
    latent frames, so the silence mixes into the real-content region, and
    cropping the output afterwards cannot undo it.

    Measured, from a 4x A100 smoke evaluation: AudioLDM scored
    attack_sisnr = -50 dB -- the "attacked" audio essentially uncorrelated
    with its input -- while MBD, also a diffusion resynthesis but run at the
    audio's native length with no padding, scored -8.8 dB. A 41 dB gap between
    two comparable methods, from the padding alone.

    Relationship to the eval-side fix in this branch: building a
    fixed-duration eval set of files that are genuinely >= 10.24s is strictly
    better where it applies, since the model then sees real contiguous audio
    and nothing is synthesised at all. This covers the cases that cannot
    reach: the TRAINING path (train_step feeds whatever
    data.segment_duration produces, and shortening it is the point -- longer
    segments make detection easier for free, because the detector
    time-averages both presence and the message bits, see
    audioseal/models.py:decode_message), and any evaluation run against a
    dataset that has not been pre-curated, such as the AzureML mounts where
    `test-clean-fixed` does not exist.

    Tiling's own cost is repeated content and a seam at each wrap, which is
    far smaller than an 80% silence plateau. Gradients flow back through every
    tile onto the same input samples, which is simply the true Jacobian of
    tile-then-crop.
    """
    length = wav.shape[-1]
    if length == 0:
        raise ValueError("cannot fit an empty waveform to a target length")
    if length < target_len:
        wav = wav.repeat(1, target_len // length + 1)
    return wav[..., :target_len]


class IdentityAttack(nn.Module):
    """No-op attack: returns the input unchanged. Always available, has no
    parameters, and is trivially differentiable (gradient is the identity)."""

    def forward(self, x: torch.Tensor, strength: tp.Optional[float] = None) -> torch.Tensor:
        return x


class BigVGANAttack(nn.Module):
    """Resynthesis attack: extract a mel-spectrogram from `x_wm` and vocode it
    back to a waveform with a frozen, pretrained BigVGAN.

    TODO(setup): this backbone is not vendored or installed anywhere in this
    project. To enable it:
      1. `pip install bigvgan` (or vendor github.com/NVIDIA/BigVGAN directly).
      2. Pick a pretrained checkpoint, e.g. a `nvidia/bigvgan_v2_*` repo id on
         Hugging Face, and confirm its native sample rate -- BigVGAN
         checkpoints are commonly trained at 22.05/24/44.1kHz, not 16kHz, so
         you will likely need to resample around this attack (resample up
         before the mel extraction, resample back down to `sample_rate`
         after vocoding) or find/finetune a 16kHz checkpoint.
      3. Fill in `_load_backbone` below and remove the `NotImplementedError`.
    """

    def __init__(self, checkpoint: tp.Optional[str] = None, sample_rate: int = 16_000):
        super().__init__()
        self.sample_rate = sample_rate
        self.checkpoint = checkpoint
        self._backbone: tp.Optional[nn.Module] = None
        if checkpoint is not None:
            self._backbone = self._load_backbone(checkpoint)
            self._backbone.eval()
            for p in self._backbone.parameters():
                p.requires_grad_(False)

    def _load_backbone(self, checkpoint: str) -> nn.Module:
        raise NotImplementedError(
            "BigVGANAttack has no backbone loading implemented yet. Install "
            "BigVGAN and implement mel-extraction + vocoding here, then pass "
            "attack.bigvgan.checkpoint=<hf-repo-or-local-path> in the config."
        )

    def forward(self, x: torch.Tensor, strength: tp.Optional[float] = None) -> torch.Tensor:
        # strength: unused -- BigVGAN resynthesis isn't naturally parametrized
        # by a single "strength" knob the way diffusion purification is.
        if self._backbone is None:
            raise NotImplementedError(
                "BigVGANAttack was constructed without a checkpoint (see TODO "
                "in src/audioseal_robust/attacks.py). Either provide "
                "attack.bigvgan.checkpoint or set attack.weights.bigvgan=0 in "
                "the config to keep it disabled."
            )
        raise NotImplementedError("mel-extraction + BigVGAN vocoding forward pass TODO")


class DACAttack(nn.Module):
    """Resynthesis attack: encode/decode `x_wm` through a frozen, pretrained
    Descript Audio Codec (DAC).

    TODO(setup): not vendored/installed yet. To enable:
      1. `pip install descript-audio-codec`.
      2. Pick a pretrained model matching (or close to) 16kHz -- DAC ships
         16kHz, 24kHz and 44.1kHz variants; use `dac.utils.download(model_type="16khz")`
         or point `checkpoint` at a local `.pth`.
      3. Fill in `_load_backbone` below and remove the `NotImplementedError`.
    """

    def __init__(self, checkpoint: tp.Optional[str] = None, sample_rate: int = 16_000):
        super().__init__()
        self.sample_rate = sample_rate
        self.checkpoint = checkpoint
        self._backbone: tp.Optional[nn.Module] = None
        if checkpoint is not None:
            self._backbone = self._load_backbone(checkpoint)
            self._backbone.eval()
            for p in self._backbone.parameters():
                p.requires_grad_(False)

    def _load_backbone(self, checkpoint: str) -> nn.Module:
        raise NotImplementedError(
            "DACAttack has no backbone loading implemented yet. Install "
            "descript-audio-codec and implement encode/decode here, then pass "
            "attack.dac.checkpoint=<path> in the config."
        )

    def forward(self, x: torch.Tensor, strength: tp.Optional[float] = None) -> torch.Tensor:
        # strength: unused -- a codec encode/decode isn't naturally
        # parametrized by a "strength" knob (bitrate would be the closest
        # analogue; wire it up here if you want a bitrate-vs-robustness curve).
        if self._backbone is None:
            raise NotImplementedError(
                "DACAttack was constructed without a checkpoint (see TODO in "
                "src/audioseal_robust/attacks.py). Either provide "
                "attack.dac.checkpoint or set attack.weights.dac=0 in the "
                "config to keep it disabled."
            )
        raise NotImplementedError("DAC encode/decode forward pass TODO")


class SGMSEAttack(nn.Module):
    """A diffusion-based reconstruction (SGMSE, score-based generative model
    for speech enhancement) run on `x_wm`. In this project's current config,
    this is the held-out attack instead (see `held_out_attacks` in
    evaluate.py / EvalConfig in config.py) -- AudioLDMAttack is
    the one trained against, and SGMSE measures whether robustness
    generalizes to it. Its forward pass stays differentiable regardless (see
    below) in case it's re-enabled for training later -- being differentiable
    costs nothing extra under evaluate.py's outer `torch.no_grad()`.

    Structurally, SGMSE's forward pass is itself an iterative sampler
    (multiple reverse-diffusion steps), so backprop through it (were it
    trained against) means backprop through every step -- this can be
    memory-heavy and you may want to cap `num_steps` low during training, or
    backprop through only the last K steps, rather than the full
    inference-time step count. Its forward pass is NOT wrapped in no_grad --
    gradients can reach back through it into the generator (frozen params
    via requires_grad_(False), but the graph stays connected, same "frozen
    but differentiable" pattern as BigVGAN/DAC/AudioLDM).

    Wired to sp-uhh/sgmse (MIT licensed; vendored under src/sgmse/, see that
    directory's VENDORED.md), an OU-VE SDE score model for speech
    enhancement. Its mechanism is structurally different from
    AudioLDMAttack's DDPM: there's no discrete `q_sample`/`p_sample` step
    index -- the SDE runs in continuous time t in [t_eps, T=1], and the
    predictor/corrector sampler always starts from the SDE's own prior at
    t=T (centered on the input `y`, i.e. `x_wm` treated as "the noisy signal
    to enhance"). To get a `strength`-parametrized *partial* corruption
    (matching AudioLDMAttack's t* convention) rather than always running
    the model's default full enhancement, `forward` manually replicates
    `sgmse.sampling.get_pc_sampler`'s predictor-corrector loop (see that
    function for the reference this mirrors) but starts from
    `t_star = t_eps + strength * (T - t_eps)` instead of `T`, with the
    initial noisy state built from the SDE's own `marginal_prob(x0=Y, y=Y,
    t_star)` -- since x0=y=Y here (we only have one signal, not a genuine
    clean/noisy pair), the mean term collapses to exactly Y and this reduces
    to "add `std(t_star)`-scaled Gaussian noise to Y, then reverse-diffuse
    from there," the direct SDE analogue of AudioLDMAttack's
    `q_sample`-then-`p_sample` loop.

    `strength` (the t* robustness-curve axis, see evaluate.py): t*=0 means
    "no corruption" (~identity); t*=1 means starting from the SDE's full
    prior (maximum corruption before regeneration). Passing None samples a
    *separate* t* for every item in the batch, so one training step covers a
    spread of the curve rather than a single point; passing a float pins the
    whole batch to it, which is what evaluate.py's per-t* measurements need.
    NOTE: `t_star_grid`'s default values in config.py were calibrated against
    AudioLDM's DDPM (T=1000 discrete steps) using the mentor's timestep
    table -- SGMSE's noise schedule (OU-VE SDE, continuous t in [t_eps, 1])
    is a different process, so the same strength fractions likely correspond
    to a different *qualitative* corruption level here. Not yet recalibrated
    empirically for SGMSE specifically.
    """

    # Magnitude floor for the STFT amplitude transforms below. Only bites
    # where the spectrogram is essentially silent, and only on the gradient.
    _MAG_EPS = 1e-8

    def __init__(
        self,
        checkpoint: tp.Optional[str] = None,
        sample_rate: int = 16_000,
        num_steps: int = 30,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.checkpoint = checkpoint
        self.num_steps = num_steps
        # Replaced with a rank-shared generator by SampledReconstructionAttack;
        # the module default keeps standalone use working exactly as before.
        self._strength_rng: random.Random = tp.cast(random.Random, random)
        self._model: tp.Optional[nn.Module] = None
        # Overwritten from the checkpoint's own data module in _load_backbone;
        # these are SpecsDataModule's defaults.
        self._transform_type = "exponent"
        self._spec_abs_exponent = 0.5
        self._spec_factor = 0.15
        if checkpoint is not None:
            self._load_backbone(checkpoint)

    def _load_backbone(self, checkpoint: str) -> None:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"SGMSE checkpoint not found: {ckpt_path}")

        from sgmse.model import ScoreModel

        # weights_only=False: this checkpoint embeds a SpecsDataModule
        # instance (their data pipeline config, not just tensors), which
        # PyTorch >=2.6's default weights_only=True unpickler refuses as an
        # untrusted global. Fine to disable here -- it's a checkpoint we
        # downloaded directly from the paper authors' own release, not
        # arbitrary/untrusted input.
        model = ScoreModel.load_from_checkpoint(str(ckpt_path), map_location="cpu", weights_only=False)
        # .eval() is not just standard hygiene here -- ScoreModel overrides
        # train()/eval() to swap in EMA-smoothed weights on eval() (see
        # sgmse/model.py:ScoreModel.train), the SGMSE-native equivalent of
        # AudioLDMAttack's ema_scope. Skipping this would silently run the raw
        # (non-EMA) training weights instead.
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        data_module = model.data_module
        self._transform_type = data_module.transform_type
        self._spec_abs_exponent = float(data_module.spec_abs_exponent)
        self._spec_factor = float(data_module.spec_factor)

    def _magnitude(self, spec: torch.Tensor) -> torch.Tensor:
        """|spec|, but smooth at the origin: `torch.abs` on a complex tensor
        backprops z/|z|, which is NaN at z=0. This is 0 there instead."""
        return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + self._MAG_EPS**2)

    def _spec_fwd(self, spec: torch.Tensor) -> torch.Tensor:
        """`SpecsDataModule.spec_fwd`, rewritten so it can be backpropagated
        through. Upstream computes `|z|**e * exp(1j*angle(z))`, which is
        algebraically just `z * |z|**(e-1)` but, routed through abs()/angle(),
        has an unbounded gradient as |z| -> 0 (e = 0.5 by default) and a NaN
        one at exactly 0 -- i.e. on any silent STFT bin. Upstream never
        notices because its sampler runs entirely under torch.no_grad (see
        sgmse.sampling.get_pc_sampler); this attack is trained against, so it
        cannot."""
        if self._transform_type == "exponent":
            if self._spec_abs_exponent != 1:
                spec = spec * self._magnitude(spec).pow(self._spec_abs_exponent - 1)
            return spec * self._spec_factor
        if self._transform_type == "log":
            magnitude = self._magnitude(spec)
            return spec * (torch.log1p(magnitude) / magnitude) * self._spec_factor
        return spec

    def _spec_back(self, spec: torch.Tensor) -> torch.Tensor:
        """Inverse of `_spec_fwd`, rewritten the same way -- `to_audio` routes
        through the same abs()/angle() pair on the way out."""
        if self._transform_type == "exponent":
            spec = spec / self._spec_factor
            if self._spec_abs_exponent != 1:
                spec = spec * self._magnitude(spec).pow(1 / self._spec_abs_exponent - 1)
            return spec
        if self._transform_type == "log":
            spec = spec / self._spec_factor
            magnitude = self._magnitude(spec)
            return spec * (torch.expm1(magnitude) / magnitude)
        return spec

    def forward(self, x: torch.Tensor, strength: tp.Optional[float] = None) -> torch.Tensor:
        """`strength` is t* in [0, 1] (see class docstring). None means "draw
        one per example", which is what training should do; a float pins the
        whole batch to that t*, which is what evaluate.py's curve needs."""
        if self._model is None:
            raise NotImplementedError(
                "SGMSEAttack was constructed without a checkpoint (see TODO "
                "in src/audioseal_robust/attacks.py). Either provide "
                "attack.sgmse.checkpoint or set attack.weights.sgmse=0 in the "
                "config to keep it disabled."
            )

        from sgmse.sampling.correctors import CorrectorRegistry
        from sgmse.sampling.predictors import PredictorRegistry
        from sgmse.util.other import pad_spec

        device = x.device
        in_dtype = x.dtype

        # Forces fp32 regardless of any ambient bf16 autocast (see train.py's
        # _autocast): NCSN++'s upfirdn2d up/downsampling (backbones/ncsnpp_utils/
        # op/upfirdn2d.py) goes through a hand-written CUDA kernel on GPU (the
        # CPU path has a pure-PyTorch fallback, but that's not this path) that
        # only has a fp32 dispatch -- "upfirdn2d_cuda" not implemented for
        # 'BFloat16'. Activation checkpointing above is unaffected; this only
        # gives up the *dtype* half of the memory savings for this attack, not
        # the step-count half, which was the actual fix for the OOM.
        # enabled=False alone only stops NEW ops in this block from being cast
        # to bf16 -- it does not undo the dtype `x` already arrived in (bf16,
        # since it's the output of embed_watermark/attack's own bf16-autocast
        # forward in train_step). Cast explicitly, or the same op sees the
        # same bf16 tensor either way.
        with torch.autocast(device_type=device.type, enabled=False):
            x = x.float()
            batch_size = x.shape[0]
            sde = self._model.sde.copy()
            sde.N = self.num_steps
            eps = self._model.t_eps

            if strength is None:
                # Drawn from the rank-shared RNG rather than torch's global
                # generator, which distributed.seed_everything seeds per rank.
                # Unlike AudioLDM the step count here is fixed (num_steps), so
                # this is about comparability rather than cost: identical t*
                # across ranks means every rank corrupts to the same depth, so
                # the loss each contributes to the allreduce is measuring the
                # same difficulty. The per-example spread within the batch is
                # preserved -- each rank still sees different audio.
                strengths = torch.tensor(
                    [self._strength_rng.random() for _ in range(batch_size)],
                    device=device,
                )
            else:
                strengths = torch.full(
                    (batch_size,), float(min(max(strength, 0.0), 1.0)), device=device
                )
            t_star = eps + strengths * (sde.T - eps)  # (B,)

            orig_len = x.shape[-1]
            wav = x.squeeze(1)  # (B, T)
            # SGMSE's own enhance() normalizes by peak amplitude before STFT and
            # rescales back afterwards -- its score model was trained on
            # peak-normalized inputs.
            norm_factor = wav.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            wav = wav / norm_factor

            Y = self._spec_fwd(self._model._stft(wav)).unsqueeze(1)  # (B, 1, F, T_frames)
            Y = pad_spec(Y)

            # Forward-corrupt Y to t_star via the SDE's own marginal_prob. With
            # x0=y=Y (see class docstring), the mean term is exactly Y, so this
            # is "Y plus std(t_star)-scaled Gaussian noise" -- the SDE analogue
            # of AudioLDMAttack's q_sample.
            mean, std = sde.marginal_prob(Y, Y, t_star)
            xt = mean + std[:, None, None, None] * torch.randn_like(Y)
            xt_mean = xt

            predictor = PredictorRegistry.get_by_name("reverse_diffusion")(sde, self._model, probability_flow=False)
            corrector = CorrectorRegistry.get_by_name("ald")(sde, self._model, snr=0.5, n_steps=1)

            # Mirrors sgmse.sampling.get_pc_sampler's pc_sampler() loop exactly,
            # just starting from t_star instead of sde.T -- see that function
            # for the reference this replicates. Deliberately NOT wrapped in
            # torch.no_grad() (see class docstring: this attack must stay
            # differentiable for training).
            # (B, num_steps): row b is that example's own linspace(t_star[b], eps).
            ramp = torch.linspace(1.0, 0.0, self.num_steps, device=device)
            timesteps = eps + (t_star[:, None] - eps) * ramp[None, :]

            def _pc_step(
                xt: torch.Tensor, xt_mean: torch.Tensor, Y: torch.Tensor, vec_t: torch.Tensor, stepsize: torch.Tensor
            ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
                xt, xt_mean = corrector.update_fn(xt, Y, vec_t)
                xt, xt_mean = predictor.update_fn(xt, Y, vec_t, stepsize)
                return xt, xt_mean

            for i in range(self.num_steps):
                vec_t = timesteps[:, i]
                if i != self.num_steps - 1:
                    stepsize = vec_t - timesteps[:, i + 1]
                else:
                    stepsize = timesteps[:, -1]  # from eps to 0, as in the reference
                # Activation checkpointing: this loop is a differentiable
                # multi-step sampler (see class docstring's "can be memory-heavy"
                # note) -- without it, autograd holds every step's activations
                # (score-model forward, twice per step via corrector+predictor)
                # live simultaneously for backward, num_steps=30 by default. This
                # trades that for one recomputed forward per step during
                # backward instead. use_reentrant=False: the modern
                # checkpoint API, correctly threads gradients to Y (reused every
                # step, not just the loop-carried xt/xt_mean).
                xt, xt_mean = torch.utils.checkpoint.checkpoint(
                    _pc_step, xt, xt_mean, Y, vec_t, stepsize, use_reentrant=False
                )

            x_hat = self._model._istft(self._spec_back(xt_mean.squeeze(1)), orig_len)
            x_hat = x_hat * norm_factor
            return x_hat.unsqueeze(1).to(in_dtype)


class AudioLDMAttack(nn.Module):
    """The trained attack in this project variant: given nonzero weight in
    `AttackConfig.weights.audioldm` during training (see train.py), so the
    generator is fine-tuned directly against it. SGMSEAttack plays the
    held-out role instead here (see `held_out_attacks` in evaluate.py /
    EvalConfig in config.py) -- the generalization question this setup
    answers is "did robustness trained against AudioLDM-style
    latent-diffusion resynthesis transfer to a structurally different
    diffusion attack (SGMSE's OU-VE SDE) it never saw," rather than the
    reverse. Do not also give `sgmse` nonzero training weight unless you
    deliberately want to give up that held-out probe.

    HISTORY: this was called DiffEraseAttack until it was renamed, and was
    written against DiffErase-latent (github.com/DiffErase/Differase's
    "Audio Pirates"), whose *method* -- partial-noise then
    diffusion-regenerate -- is what this implements. It was never their
    model, though: that repo ships no pretrained checkpoint (its README is
    about training one), so there are no open weights to run and the code
    always pointed at a stock pretrained AudioLDM checkpoint instead. The
    name was making a promise the weights couldn't keep, hence `audioldm`.

    Forward-diffuses a mel-spectrogram of `x` up to timestep
    `strength * num_timesteps` (same t* convention as SGMSEAttack -- 0 = no
    corruption, 1 = full noise-then-regenerate), reverse-diffuses it back
    with the pretrained LatentDiffusion model, and vocodes the result back
    to a waveform with the accompanying HiFiGAN. Works on in-memory batched
    tensors so it slots into train.py's/evaluate.py's per-batch loops --
    see the "differentiability" paragraph below for the one place this
    implementation deliberately does NOT call through to the vendored
    library's own convenience methods.

    Model code (`audioldm_train`, MIT licensed) is vendored under
    `src/audioldm_train/` -- see that directory's VENDORED.md for provenance
    and its local changes. Only the actual *weights* (multi-GB, never belong
    in git) stay external: point `checkpoint` and `config` at a pretrained
    AudioLDM release on disk (e.g. `audioldm-s-full` from the AudioLDM
    authors' Zenodo record, plus the matching
    `audioldm_train/config/**/audioldm_original.yaml` vendored here). This
    attack stays disabled (constructing it without a checkpoint keeps
    raising NotImplementedError) until you do -- see AttackConfig.audioldm /
    EvalAttackConfig.audioldm in config.py.

    Differentiable, same "frozen but differentiable" pattern as
    BigVGAN/DAC/SGMSE (see module docstring): its own parameters are frozen
    via `requires_grad_(False)` and the forward pass here is NOT wrapped in
    `torch.no_grad()`. That alone is NOT sufficient for this model, though --
    confirmed by reading `audioldm_train/modules/latent_diffusion/ddpm.py`
    (the vendored `LatentDiffusion` class actually instantiated by the
    shipped `audioldm_original*.yaml` configs): `encode_first_stage`,
    `p_sample` (the conditional overload, i.e. the one taking `c`) and
    `decode_first_stage` are each individually `@torch.no_grad()`-locked (or
    wrap their body in `with torch.no_grad():`) *inside the vendored class
    itself*, independent of whatever context this forward() runs in. Calling
    them would silently detach the graph no matter what we do here. So this
    forward calls their own undecorated building blocks directly instead --
    `first_stage_model.encode`/`.decode` (exactly what
    `encode_first_stage`/`decode_first_stage` call internally) and
    `p_mean_variance` (what `p_sample` calls internally, then does its own
    un-decorated noise-injection algebra, inlined below) -- rather than the
    no_grad-locked wrappers. These primitives are confirmed differentiable
    independently of this project: `p_losses` (the model's own training-time
    loss, in the same vendored file) calls `apply_model` -- which
    `p_mean_variance` itself wraps -- directly, with no such decorator, since
    the original AudioLDM training pipeline backprops through it too. This
    is the same "reimplement the library's own sampler loop with its
    undecorated primitives instead of its no_grad-locked convenience
    wrapper" pattern SGMSEAttack already uses, for the same reason -- see
    that class. We do NOT patch the vendored ddpm.py itself to remove those
    decorators (that would be a second, riskier local modification beyond
    VENDORED.md's documented one, and isn't necessary since the undecorated
    primitives it calls are already reachable directly).

    Backpropagating through the reverse-diffusion loop means backprop
    through one U-Net forward per step -- expensive. Random `strength`
    sampling (`strength=None`) is therefore capped at `strength_max`
    (default 0.08) rather than drawn from the full [0, 1] range:
    `strength_max=0.08` caps `noise_timestep` at ~80 (out of a typical
    `num_timesteps=1000`), matching EvalConfig.t_star_grid's own calibrated
    "hard regime" ceiling (see that field's comment in config.py) rather
    than an arbitrary cutoff -- going further raises step count (and
    therefore memory) roughly linearly, well past the point that curve was
    calibrated against. evaluate.py still wraps every call in
    `torch.no_grad()` at the call site regardless of this attack's own
    differentiability, so eval's cost/behavior doesn't change -- only
    training exercises the backprop path.
    """

    def __init__(
        self,
        checkpoint: tp.Optional[str] = None,
        config: tp.Optional[str] = None,
        sample_rate: int = 16_000,
        strength_max: float = 0.08,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.checkpoint = checkpoint
        self.config_path = config
        self.strength_max = strength_max
        # Replaced with a rank-shared generator by SampledReconstructionAttack;
        # the module default keeps standalone use (tests, evaluate.py) working
        # exactly as before. See forward() for why sharing this matters.
        self._strength_rng: random.Random = tp.cast(random.Random, random)
        self._model: tp.Optional[nn.Module] = None
        self._vocoder: tp.Optional[nn.Module] = None
        self._stft: tp.Optional[nn.Module] = None
        self._duration: tp.Optional[float] = None
        self._model_sample_rate: tp.Optional[int] = None
        if checkpoint is not None:
            self._load_backbone(checkpoint)

    def _load_backbone(self, checkpoint: str) -> None:
        if self.config_path is None:
            raise NotImplementedError(
                "AudioLDMAttack needs `config` set alongside `checkpoint` "
                "-- see EvalAttackConfig.audioldm in src/audioseal_robust/config.py."
            )

        ckpt_path = Path(checkpoint).resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"AudioLDM checkpoint not found: {ckpt_path}")
        # get_vocoder() hardcodes a "data/checkpoints" *relative* path (same
        # convention first_stage_config.reload_from_ckpt uses in the yaml) --
        # `checkpoint` must live at <weights_root>/data/checkpoints/<file>
        # so we can cd into <weights_root> for that lookup to resolve.
        if ckpt_path.parent.name != "checkpoints" or ckpt_path.parent.parent.name != "data":
            raise ValueError(
                f"AudioLDM checkpoint {ckpt_path} must live at "
                "<weights_root>/data/checkpoints/<file> -- that's where "
                "get_vocoder() and the VAE's reload_from_ckpt look for the "
                "vocoder/VAE weights next to it (see AudioLDM's own "
                "data/checkpoints/ layout)."
            )
        weights_root = ckpt_path.parent.parent.parent

        import yaml

        from audioldm_train.utilities.audio.stft import TacotronSTFT
        from audioldm_train.utilities.model_util import get_vocoder, instantiate_from_config

        config_path = Path(self.config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"AudioLDM config not found: {config_path}")
        with open(config_path) as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)

        prep = model_config["preprocessing"]
        self._model_sample_rate = prep["audio"]["sampling_rate"]
        self._duration = prep["audio"]["duration"]
        n_mel_channels = prep["mel"]["n_mel_channels"]

        self._stft = TacotronSTFT(
            filter_length=prep["stft"]["filter_length"],
            hop_length=prep["stft"]["hop_length"],
            win_length=prep["stft"]["win_length"],
            n_mel_channels=n_mel_channels,
            sampling_rate=self._model_sample_rate,
            mel_fmin=prep["mel"]["mel_fmin"],
            mel_fmax=prep["mel"]["mel_fmax"],
        )

        # Disable the VAE's training-time perceptual loss before constructing
        # anything. AutoencoderKL.__init__ does `self.loss =
        # instantiate_from_config(lossconfig)` unconditionally, and the
        # configured LPIPSWithDiscriminator pulls in taming-transformers'
        # LPIPS, whose __init__ downloads a ~500MB vgg.pth to the *relative*
        # path "taming/modules/autoencoder/lpips/" -- i.e. into whatever the
        # CWD happens to be, which is `weights_root` thanks to the chdir
        # below, and which blows up with a read-only-filesystem OSError when
        # the weights live on a read-only mount (e.g. a container volume).
        # `self.loss` is only ever touched by training_step/validation_step/
        # configure_optimizers -- never by the encode/decode path this attack
        # uses -- so swapping it for a no-op is safe, and skips both the
        # download and the taming-transformers dependency entirely.
        first_stage_params = model_config["model"]["params"]["first_stage_config"]["params"]
        first_stage_params["lossconfig"] = {"target": "torch.nn.Identity"}

        original_cwd = os.getcwd()
        original_torch_load = torch.load
        os.chdir(weights_root)
        # AudioLDM's own model construction calls torch.load() in a
        # few places (e.g. get_vocoder, and AutoencoderKL's own internal
        # preview-vocoder in __init__) without map_location -- fine on the
        # GPU boxes those checkpoints were saved on, but on CPU-only/MPS
        # machines torch.load then tries to restore CUDA tensors and raises.
        # Patch the default rather than editing their source.
        torch.load = functools.partial(original_torch_load, map_location="cpu")
        try:
            model = instantiate_from_config(model_config["model"])
            state = original_torch_load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["state_dict"], strict=False)
            vocoder = get_vocoder(model_config, "cpu", n_mel_channels)
        finally:
            torch.load = original_torch_load
            os.chdir(original_cwd)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        vocoder.eval()
        for p in vocoder.parameters():
            p.requires_grad_(False)

        self._model = model
        self._vocoder = vocoder

    def forward(self, x: torch.Tensor, strength: tp.Optional[float] = None) -> torch.Tensor:
        """`strength` is t* in [0, 1] (see class docstring); None means
        "sample it randomly", capped at `self.strength_max` (see class
        docstring for why -- backprop-depth/memory, not a correctness bound:
        an explicitly-passed `strength` above `strength_max`, e.g. from
        evaluate.py's t_star_grid, is honored as-is, only clamped to
        [0, 1])."""
        if self._model is None:
            raise NotImplementedError(
                "AudioLDMAttack was constructed without a checkpoint (see "
                "class docstring in src/audioseal_robust/attacks.py). Either "
                "set attack.audioldm.{checkpoint,config} (training) / "
                "eval.audioldm.{checkpoint,config} (eval), or set "
                "attack.weights.audioldm=0 / remove audioldm from "
                "eval_attacks to skip it."
            )
        if strength is None:
            # self._strength_rng, not the global `random`: under DDP the global
            # RNG is seeded per rank (distributed.seed_everything uses
            # seed + rank), so each rank would draw a different t* and run a
            # different number of reverse-diffusion steps. DDP synchronises at
            # every backward, so the step costs whatever the deepest draw on
            # any rank costs -- with 4 ranks that is E[max of 4 uniforms] =
            # 0.8 * strength_max instead of 0.5, ~1.6x the expected work, paid
            # on every step. It also makes peak memory a function of the
            # unluckiest rank, and eval numbers less comparable. See
            # SampledReconstructionAttack, which injects a rank-shared RNG for
            # exactly the reason it already shares the branch draw.
            strength = self._strength_rng.random() * self.strength_max
        strength = float(min(max(strength, 0.0), 1.0))

        device = x.device
        orig_len = x.shape[-1]
        target_len = int(round(self._model_sample_rate * self._duration))

        wav = x.squeeze(1).clamp(-1.0, 1.0)  # (B, T)
        wav = _tile_or_crop(wav, target_len)

        mel, *_ = self._stft.mel_spectrogram(wav)  # (B, n_mel, T_frames)
        mel_input = mel.permute(0, 2, 1).unsqueeze(1)  # (B, 1, T_frames, n_mel)

        # Deliberately not using self._model.ema_scope() here: it swaps in
        # EMA-averaged weights for the duration of the block, but its
        # bookkeeping (LitEma.copy_to) asserts that every currently-frozen
        # parameter was ALSO frozen at EMA-construction time -- which
        # doesn't hold once we (or the caller, e.g. evaluate.py's
        # build_eval_attacks) freeze this model's parameters post-construction
        # for inference. Skipping it just means we use the checkpoint's raw
        # (non-EMA) weights directly, which is a normal, supported choice.
        #
        # NOTE: calling self._model.first_stage_model.encode(...) directly
        # rather than self._model.encode_first_stage(...) -- the latter is
        # @torch.no_grad()-locked inside the vendored LatentDiffusion class
        # itself (see class docstring's "differentiability" paragraph for
        # why this whole function bypasses that wrapper and two others).
        # This call is otherwise identical to what encode_first_stage does
        # internally.
        posterior = self._model.first_stage_model.encode(mel_input)
        z0 = self._model.get_first_stage_encoding(posterior)

        num_timesteps = self._model.num_timesteps
        # Valid timestep indices are [0, num_timesteps - 1] -- at
        # strength=1.0 the naive int(num_timesteps * strength) lands exactly
        # on num_timesteps, one past the end of the precomputed
        # noise-schedule buffers (sqrt_alphas_cumprod etc.), and q_sample's
        # index_select crashes.
        noise_timestep = min(int(num_timesteps * strength), num_timesteps - 1)
        t_noise = torch.full((z0.shape[0],), noise_timestep, device=device, dtype=torch.long)
        z = self._model.q_sample(x_start=z0, t=t_noise, noise=torch.randn_like(z0))

        # "Unconditional" here does NOT mean an empty cond dict -- the
        # UNet's FiLM layer (extra_film_condition_dim in the config)
        # asserts its conditioning input is never None whenever the
        # model was built with a cond_stage_config at all (see
        # ddpm.py:UNetModel.forward's `assert (y is not None) == ...`).
        # The correct null condition is each conditioner's own
        # classifier-free-guidance "empty" embedding
        # (get_unconditional_condition -- e.g. CLAP's embedding of an
        # empty-string prompt), same convention as Stable Diffusion's
        # empty-prompt embedding. Generic over whatever
        # cond_stage_config the loaded model was built with.
        cond: tp.Dict[str, tp.Any] = {
            key: self._model.cond_stage_models[i].get_unconditional_condition(z0.shape[0])
            for i, key in enumerate(self._model.conditioning_key)
        }
        clip_denoised = self._model.clip_denoised

        def _ddpm_step(z: torch.Tensor, t: int) -> torch.Tensor:
            # Inlined body of LatentDiffusion.p_sample (ddpm.py), minus its
            # @torch.no_grad() decorator -- see class docstring. p_mean_variance
            # itself is undecorated (confirmed differentiable: the model's own
            # p_losses calls apply_model, which this wraps, directly, with no
            # such decorator). noise_like(shape, device, repeat=False) (what
            # p_sample calls) is exactly torch.randn(shape, device=device) --
            # see audioldm_train/utilities/diffusion_util.py:noise_like.
            t_tensor = torch.full((z.shape[0],), t, device=device, dtype=torch.long)
            model_mean, _, model_log_variance = self._model.p_mean_variance(
                x=z, c=cond, t=t_tensor, clip_denoised=clip_denoised
            )
            noise = torch.randn_like(z)
            nonzero_mask = (
                (1 - (t_tensor == 0).float())
                .reshape(z.shape[0], *((1,) * (len(z.shape) - 1)))
                .contiguous()
            )
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        for t in range(noise_timestep, 0, -1):
            # Activation checkpointing -- same reasoning as SGMSEAttack.forward's
            # _pc_step: this is a differentiable reverse-diffusion loop (up to
            # noise_timestep <= num_timesteps*strength_max steps, each one a full
            # UNet forward through self._model), so autograd would otherwise hold
            # every step's activations live simultaneously for backward.
            # preserve_rng_state=True (checkpoint's default) reruns this step's
            # torch.randn_like(z) noise draw identically on recompute.
            z = torch.utils.checkpoint.checkpoint(_ddpm_step, z, t, use_reentrant=False)

        # NOTE: first_stage_model.decode(...) directly, not
        # self._model.decode_first_stage(...) -- same reason as the encode
        # call above (that wrapper is also @torch.no_grad()-locked). This is
        # otherwise identical to what decode_first_stage does internally.
        mel_out = self._model.first_stage_model.decode(z / self._model.scale_factor)
        if mel_out.shape[1] == 1:
            mel_out = mel_out.squeeze(1)  # (B, T_frames, n_mel)

        mel_for_vocoder = mel_out.permute(0, 2, 1)  # (B, n_mel, T_frames)
        wav_out = self._vocoder(mel_for_vocoder).squeeze(1)  # (B, T)

        if wav_out.shape[-1] < orig_len:
            wav_out = F.pad(wav_out, (0, orig_len - wav_out.shape[-1]))
        else:
            wav_out = wav_out[..., :orig_len]
        return wav_out.unsqueeze(1).to(x.dtype)


class MBDAttack(nn.Module):
    """Held-out attack (same status as SGMSEAttack in this project's current
    config -- eval only, NEVER given nonzero weight in AttackConfig.weights
    during training): Meta's
    MultiBand Diffusion (github.com/facebookresearch/audiocraft,
    `docs/MBD.md`), an EnCodec-conditioned diffusion decoder. Per the
    mentor's plan (2026-08-12 Notion doc), this is the intended replacement/
    complement for AudioLDMAttack's latent diffusion as "a diffusion model not
    trained specifically on watermark removal, but that can incidentally
    remove them due to the neural encoding embedded into the system":

        c = Q(E(x_wm))              -- EnCodec encoder + quantizer -> discrete codes
        x_hat = F_diffusion(c, z)   -- MBD's diffusion decoder, z ~ N(0, I)

    The new waveform keeps the information contained in `c` (speech
    content, speaker characteristics) but has different samples/phase/fine
    structure -- and, hopefully, less of the watermark, which the codec's
    quantization bottleneck has no reason to preserve.

    Runs natively at 24kHz; `regenerate()` handles the 16kHz<->24kHz
    resampling internally (see forward below), matching the pipeline in the
    mentor's plan (upsample -> EnCodec -> MBD -> downsample -> detector).

    `strength`: unlike AudioLDMAttack/SGMSEAttack, MBD has no continuous
    corruption-level knob -- the closest analogue is bitrate (fewer bits
    through the EnCodec bottleneck = more information thrown away = a
    stronger attack). `bandwidth` (1.5/3.0/6.0 kbps, MBD's only supported
    values) is fixed at construction rather than swept per-call, same
    reasoning as DACAttack's docstring for why `strength` goes unused here
    -- wire up a 3-point bitrate-vs-robustness curve if you want one.

    No local checkpoint needed: `get_mbd_24khz()` downloads its own
    pretrained weights from HF on first use. `checkpoint` here is just the
    usual enable/disable gate (matching every other attack's config
    pattern) -- set it to any non-None value (e.g. "auto") to turn this on.
    """

    def __init__(
        self,
        checkpoint: tp.Optional[str] = None,
        sample_rate: int = 16_000,
        bandwidth: float = 3.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.checkpoint = checkpoint
        self.bandwidth = bandwidth
        self._mbd: tp.Optional[tp.Any] = None
        if checkpoint is not None:
            self._load_backbone(checkpoint)

    def _load_backbone(self, checkpoint: str) -> None:
        from audiocraft.models import MultiBandDiffusion

        original_torch_load = torch.load
        # audiocraft.models.loaders._get_state_dict calls torch.load()
        # without weights_only=False; the checkpoint embeds an omegaconf
        # DictConfig (their model config), which PyTorch >=2.6's default
        # weights_only=True unpickler refuses as an untrusted global. Patch
        # the default rather than editing their source (same approach as
        # AudioLDMAttack's torch.load patch).
        torch.load = functools.partial(original_torch_load, weights_only=False)
        try:
            mbd = MultiBandDiffusion.get_mbd_24khz(bw=self.bandwidth)
        finally:
            torch.load = original_torch_load

        # mbd itself is a plain wrapper, not an nn.Module -- its component
        # models (mbd.codec_model, one per mbd.DPs[i].model, one per
        # frequency band) are NOT auto-registered as submodules of this
        # class, so the outer freeze/`.to(device)` dance in
        # evaluate.py:build_eval_attacks never reaches them. Freeze here
        # explicitly; forward() below re-syncs device on every call for the
        # same reason.
        for p in mbd.codec_model.parameters():
            p.requires_grad_(False)
        for dp in mbd.DPs:
            for p in dp.model.parameters():
                p.requires_grad_(False)
        self._mbd = mbd

    def forward(self, x: torch.Tensor, strength: tp.Optional[float] = None) -> torch.Tensor:
        if self._mbd is None:
            raise NotImplementedError(
                "MBDAttack was constructed without a checkpoint (see class "
                "docstring in src/audioseal_robust/attacks.py). Either "
                "provide attack.mbd.checkpoint (any non-None value -- MBD "
                "downloads its own weights) or remove mbd from "
                "eval.held_out_attacks to skip it."
            )
        current_device = next(self._mbd.codec_model.parameters()).device
        if current_device != x.device:
            self._mbd.codec_model.to(x.device)
            for dp in self._mbd.DPs:
                dp.model.to(x.device)
            self._mbd.device = x.device
        with torch.no_grad():
            out = self._mbd.regenerate(x, sample_rate=self.sample_rate)
        return out[..., : x.shape[-1]].to(x.dtype)


class SampledReconstructionAttack(nn.Module):
    """On every forward call, randomly samples ONE of the registered attacks
    (by name, weighted) and applies it. The chosen attack's parameters (if
    any) never receive gradients and are never touched by the optimizer, but
    the forward computation stays part of the autograd graph so gradients
    flow back through it to its input.

    Pass `rng` to control where the branch draw comes from. Under DDP the
    draw must agree across ranks (branch costs differ by orders of magnitude
    and DDP syncs at every backward, so a disagreement makes every step cost
    the most expensive branch any rank picked) -- see
    distributed.attack_sampling_rng.
    """

    def __init__(
        self,
        attacks: tp.Dict[str, nn.Module],
        weights: tp.Dict[str, float],
        rng: tp.Optional[random.Random] = None,
    ):
        super().__init__()
        assert set(attacks.keys()) == set(weights.keys()), (
            f"attacks {sorted(attacks.keys())} and weights "
            f"{sorted(weights.keys())} must name the same set of attacks"
        )
        assert any(w > 0 for w in weights.values()), "at least one attack weight must be > 0"

        self.attacks = nn.ModuleDict(attacks)
        for name, module in self.attacks.items():
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)

        self._names = [n for n, w in weights.items() if w > 0]
        self._sampling_weights = [weights[n] for n in self._names]
        # None -> the global `random` module, i.e. exactly the previous
        # behavior. Under DDP, train.py passes an identically-seeded
        # `random.Random` so every rank draws the SAME branch each step --
        # see distributed.attack_sampling_rng for why that has to be shared
        # when nothing else is.
        self._rng: random.Random = rng if rng is not None else tp.cast(random.Random, random)
        # Share that same generator with the branches' own strength sampling.
        # The branch draw was already synchronised across ranks; t* was not,
        # and for AudioLDM t* sets the number of reverse-diffusion steps, so an
        # unsynchronised draw makes every DDP step cost the deepest trajectory
        # any rank happened to draw (~1.6x the expected work at world_size=4)
        # and leaves peak memory hostage to the unluckiest rank. Injected here
        # rather than passed through every constructor so an attack added later
        # is covered by declaring the attribute.
        for module in self.attacks.values():
            if hasattr(module, "_strength_rng"):
                module._strength_rng = self._rng

    def train(self, mode: bool = True) -> "SampledReconstructionAttack":
        # Keep the outer module's `.training` flag consistent with the rest of
        # the model, but the frozen sub-attacks must always stay in eval()
        # regardless -- e.g. dropout/batchnorm inside a pretrained backbone
        # must not behave differently or update running stats during our
        # generator training.
        super().train(mode)
        for module in self.attacks.values():
            module.eval()
        return self

    @property
    def branch_names(self) -> tp.List[str]:
        """The attacks that can actually be sampled (weight > 0), in a stable
        order. Callers that need to evaluate every branch rather than a random
        one -- e.g. train.py's periodic validation -- iterate this."""
        return list(self._names)

    def forward(
        self,
        x_wm: torch.Tensor,
        strength: tp.Optional[float] = None,
        name: tp.Optional[str] = None,
    ) -> tp.Tuple[torch.Tensor, str]:
        """`strength`: passed through to whichever attack gets sampled (see
        e.g. SGMSEAttack's docstring for what it means there; ignored by
        attacks that don't use it). Leave it None during training -- each
        strength-aware attack should sample its own random t* per call in
        that case, which is what should give robustness across attack
        strengths rather than at a single fixed one.

        `name` forces a specific branch instead of sampling one. Two reasons
        it exists, both about measurement rather than training:

          * A sampled branch makes a metric incomparable across the steps it
            is logged at -- each point measures a different task. With
            identity and audioldm in one recipe the resulting curve alternates
            between two populations (loss ~0.8 vs ~5) and looks like
            instability, or like a regression, depending on which branch each
            point happened to draw.
          * Sampling here also consumes from the shared RNG, so evaluating
            shifts the branch sequence that training itself sees. Passing an
            explicit name draws nothing. That matters under DDP too: the
            shared RNG is seeded identically on every rank so all ranks walk
            the same branch sequence, and an eval that consumed from it would
            have to consume identically everywhere to keep them aligned.

        A weight-0 attack can still be named explicitly: the name selects a
        branch directly and bypasses the sampling weights entirely.
        """
        if name is None:
            name = self._rng.choices(self._names, weights=self._sampling_weights, k=1)[0]
        elif name not in self.attacks:
            raise KeyError(f"unknown attack {name!r} (have: {sorted(self.attacks)})")
        attack = self.attacks[name]
        # No torch.no_grad() and no .detach() here: see module docstring.
        x_att = attack(x_wm, strength=strength)
        return x_att, name
