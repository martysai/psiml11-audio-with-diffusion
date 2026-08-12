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
BigVGAN/DAC/SGMSE are ordinary neural nets, we don't need that workaround --
we can and do backprop through them for real.
"""

import functools
import os
import random
import typing as tp
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """The actual target attack for this project: a diffusion-based
    reconstruction (SGMSE, score-based generative model for speech
    enhancement) run on `x_wm`. This is the attack that motivates the whole
    fine-tuning setup, so it deserves the most care once wired up: unlike
    BigVGAN/DAC, SGMSE's forward pass is itself an iterative sampler
    (multiple reverse-diffusion steps), so backprop through it means
    backprop through every step -- this can be memory-heavy and you may want
    to cap `num_steps` low during training, or backprop through only the
    last K steps, rather than the full inference-time step count. Unlike
    DiffEraseAttack (held-out, eval-only, always under torch.no_grad()),
    this one's forward pass is NOT wrapped in no_grad -- it's meant to be
    trainable-against, so gradients must reach back through it into the
    generator (frozen params via requires_grad_(False), but the graph stays
    connected, same "frozen but differentiable" pattern as BigVGAN/DAC).

    Wired to sp-uhh/sgmse (MIT licensed; vendored under src/sgmse/, see that
    directory's VENDORED.md), an OU-VE SDE score model for speech
    enhancement. Its mechanism is structurally different from
    DiffEraseAttack's DDPM: there's no discrete `q_sample`/`p_sample` step
    index -- the SDE runs in continuous time t in [t_eps, T=1], and the
    predictor/corrector sampler always starts from the SDE's own prior at
    t=T (centered on the input `y`, i.e. `x_wm` treated as "the noisy signal
    to enhance"). To get a `strength`-parametrized *partial* corruption
    (matching DiffEraseAttack's t* convention) rather than always running
    the model's default full enhancement, `forward` manually replicates
    `sgmse.sampling.get_pc_sampler`'s predictor-corrector loop (see that
    function for the reference this mirrors) but starts from
    `t_star = t_eps + strength * (T - t_eps)` instead of `T`, with the
    initial noisy state built from the SDE's own `marginal_prob(x0=Y, y=Y,
    t_star)` -- since x0=y=Y here (we only have one signal, not a genuine
    clean/noisy pair), the mean term collapses to exactly Y and this reduces
    to "add `std(t_star)`-scaled Gaussian noise to Y, then reverse-diffuse
    from there," the direct SDE analogue of DiffEraseAttack's
    `q_sample`-then-`p_sample` loop.

    `strength` (the t* robustness-curve axis, see evaluate.py): t*=0 means
    "no corruption" (~identity); t*=1 means starting from the SDE's full
    prior (maximum corruption before regeneration). NOTE: `t_star_grid`'s
    default values in config.py were calibrated against DiffErase's DDPM
    (T=1000 discrete steps) using the mentor's timestep table -- SGMSE's
    noise schedule (OU-VE SDE, continuous t in [t_eps, 1]) is a different
    process, so the same strength fractions likely correspond to a
    different *qualitative* corruption level here. Not yet recalibrated
    empirically for SGMSE specifically.
    """

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
        self._model: tp.Optional[nn.Module] = None
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
        # DiffErase's ema_scope. Skipping this would silently run the raw
        # (non-EMA) training weights instead.
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model

    def forward(self, x: torch.Tensor, strength: tp.Optional[float] = None) -> torch.Tensor:
        """`strength` is t* in [0, 1] (see class docstring); None means
        "sample it randomly", which is what training should do."""
        if self._model is None:
            raise NotImplementedError(
                "SGMSEAttack was constructed without a checkpoint (see TODO "
                "in src/audioseal_robust/attacks.py). Either provide "
                "attack.sgmse.checkpoint or set attack.weights.sgmse=0 in the "
                "config to keep it disabled."
            )
        if strength is None:
            strength = random.random()
        strength = float(min(max(strength, 0.0), 1.0))

        from sgmse.sampling.correctors import CorrectorRegistry
        from sgmse.sampling.predictors import PredictorRegistry
        from sgmse.util.other import pad_spec

        device = x.device
        sde = self._model.sde.copy()
        sde.N = self.num_steps
        eps = self._model.t_eps
        t_star = eps + strength * (sde.T - eps)

        orig_len = x.shape[-1]
        wav = x.squeeze(1)  # (B, T)
        # SGMSE's own enhance() normalizes by peak amplitude before STFT and
        # rescales back afterwards -- its score model was trained on
        # peak-normalized inputs.
        norm_factor = wav.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        wav = wav / norm_factor

        Y = self._model._forward_transform(self._model._stft(wav)).unsqueeze(1)  # (B, 1, F, T_frames)
        Y = pad_spec(Y)

        # Forward-corrupt Y to t_star via the SDE's own marginal_prob. With
        # x0=y=Y (see class docstring), the mean term is exactly Y, so this
        # is "Y plus std(t_star)-scaled Gaussian noise" -- the SDE analogue
        # of DiffEraseAttack's q_sample.
        vec_t_star = torch.full((Y.shape[0],), t_star, device=device)
        mean, std = sde.marginal_prob(Y, Y, vec_t_star)
        xt = mean + std[:, None, None, None] * torch.randn_like(Y)
        xt_mean = xt

        predictor = PredictorRegistry.get_by_name("reverse_diffusion")(sde, self._model, probability_flow=False)
        corrector = CorrectorRegistry.get_by_name("ald")(sde, self._model, snr=0.5, n_steps=1)

        # Mirrors sgmse.sampling.get_pc_sampler's pc_sampler() loop exactly,
        # just starting from t_star instead of sde.T -- see that function
        # for the reference this replicates. Deliberately NOT wrapped in
        # torch.no_grad() (see class docstring: this attack must stay
        # differentiable for training).
        timesteps = torch.linspace(t_star, eps, self.num_steps, device=device)
        for i in range(self.num_steps):
            t = timesteps[i]
            stepsize = t - timesteps[i + 1] if i != self.num_steps - 1 else timesteps[-1]
            vec_t = torch.full((Y.shape[0],), t.item(), device=device)
            xt, xt_mean = corrector.update_fn(xt, Y, vec_t)
            xt, xt_mean = predictor.update_fn(xt, Y, vec_t, stepsize)

        x_hat = self._model.to_audio(xt_mean.squeeze(1), orig_len)
        x_hat = x_hat * norm_factor
        return x_hat.unsqueeze(1).to(x.dtype)


class DiffEraseAttack(nn.Module):
    """Held-out diffusion-based watermark-erasure attack: used ONLY for
    evaluation (see evaluate.py's `held_out_attacks`), NEVER given nonzero
    weight in `AttackConfig.weights` during training. The whole point of
    holding it out is to answer "did robustness generalize to an unseen
    diffusion attack, or did the generator just memorize the specific
    attack(s) it was trained against" -- if you add it to the training
    sampler too, you've destroyed the thing this is meant to measure.

    Wired to DiffErase-latent (github.com/DiffErase/Differase's "Audio
    Pirates" project, the AudioLDM-style latent-diffusion variant):
    forward-diffuses a mel-spectrogram of `x` up to timestep
    `strength * num_timesteps` (same t* convention as SGMSEAttack -- 0 = no
    corruption, 1 = full noise-then-regenerate), reverse-diffuses it back
    with the pretrained LatentDiffusion model, and vocodes the result back
    to a waveform with the accompanying HiFiGAN. This mirrors the reference
    `Differase/remove_differase-latent.py` script's `process_audio_batch`
    exactly, just working on in-memory batched tensors (so it slots into
    evaluate.py's per-batch loop) instead of reading/writing wav files.

    Model code (`audioldm_train`, MIT licensed) is vendored under
    `src/audioldm_train/` -- see that directory's VENDORED.md for provenance
    and the one intentional local change. Only the actual *weights*
    (multi-GB, never belongs in git) stay external: point `checkpoint` and
    `config` at files on disk (e.g. a checkout of the Differase repo's
    `DiffErase-latent/data/checkpoints/*.ckpt` and
    `DiffErase-latent/audioldm_train/config/**/*.yaml`) you've trained or
    otherwise obtained -- DiffErase-latent's own README is literally about
    *training* one, it ships none. This attack stays disabled (constructing
    it without a checkpoint keeps raising NotImplementedError) until you do
    -- see EvalAttackConfig.diff_erase in config.py.

    Differentiability isn't required here since eval never backprops (unlike
    SGMSEAttack, which trains against attacks and does need it) -- the whole
    reverse-diffusion loop runs under `torch.no_grad()`.
    """

    def __init__(
        self,
        checkpoint: tp.Optional[str] = None,
        config: tp.Optional[str] = None,
        sample_rate: int = 16_000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.checkpoint = checkpoint
        self.config_path = config
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
                "DiffEraseAttack needs `config` set alongside `checkpoint` "
                "-- see EvalAttackConfig.diff_erase in src/audioseal_robust/config.py."
            )

        ckpt_path = Path(checkpoint).resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"DiffErase-latent checkpoint not found: {ckpt_path}")
        # get_vocoder() hardcodes a "data/checkpoints" *relative* path (same
        # convention first_stage_config.reload_from_ckpt uses in the yaml) --
        # `checkpoint` must live at <weights_root>/data/checkpoints/<file>
        # so we can cd into <weights_root> for that lookup to resolve.
        if ckpt_path.parent.name != "checkpoints" or ckpt_path.parent.parent.name != "data":
            raise ValueError(
                f"DiffErase-latent checkpoint {ckpt_path} must live at "
                "<weights_root>/data/checkpoints/<file> -- that's where "
                "get_vocoder() and the VAE's reload_from_ckpt look for the "
                "vocoder/VAE weights next to it (see DiffErase-latent's own "
                "data/checkpoints/ layout)."
            )
        weights_root = ckpt_path.parent.parent.parent

        import yaml

        from audioldm_train.utilities.audio.stft import TacotronSTFT
        from audioldm_train.utilities.model_util import get_vocoder, instantiate_from_config

        config_path = Path(self.config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"DiffErase-latent config not found: {config_path}")
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

        original_cwd = os.getcwd()
        original_torch_load = torch.load
        os.chdir(weights_root)
        # DiffErase-latent's own model construction calls torch.load() in a
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
        "sample it randomly", matching SGMSEAttack's convention."""
        if self._model is None:
            raise NotImplementedError(
                "DiffEraseAttack was constructed without a checkpoint (see "
                "class docstring in src/audioseal_robust/attacks.py). Either "
                "set eval.diff_erase.{checkpoint,config} or remove diff_erase "
                "from eval.held_out_attacks to skip it."
            )
        if strength is None:
            strength = random.random()
        strength = float(min(max(strength, 0.0), 1.0))

        device = x.device
        orig_len = x.shape[-1]
        target_len = int(round(self._model_sample_rate * self._duration))

        wav = x.squeeze(1).clamp(-1.0, 1.0)  # (B, T)
        if wav.shape[-1] < target_len:
            wav = F.pad(wav, (0, target_len - wav.shape[-1]))
        else:
            wav = wav[..., :target_len]

        with torch.no_grad():
            mel, *_ = self._stft.mel_spectrogram(wav)  # (B, n_mel, T_frames)
            mel_input = mel.permute(0, 2, 1).unsqueeze(1)  # (B, 1, T_frames, n_mel)

            # Deliberately not using self._model.ema_scope() here: it swaps in
            # EMA-averaged weights for the duration of the block, but its
            # bookkeeping (LitEma.copy_to) asserts that every currently-frozen
            # parameter was ALSO frozen at EMA-construction time -- which
            # doesn't hold once we (or the caller, e.g. evaluate.py's
            # build_eval_attacks) freeze this model's parameters post-construction
            # for inference. Skipping it just means we use the checkpoint's raw
            # (non-EMA) weights directly, which is a normal, supported choice --
            # not required for correctness here regardless, since this whole
            # block already runs under torch.no_grad().
            posterior = self._model.encode_first_stage(mel_input)
            z0 = self._model.get_first_stage_encoding(posterior)

            num_timesteps = self._model.num_timesteps
            # Valid timestep indices are [0, num_timesteps - 1] -- at
            # strength=1.0 (the top of t_star_grid's default range) the naive
            # int(num_timesteps * strength) lands exactly on num_timesteps,
            # one past the end of the precomputed noise-schedule buffers
            # (sqrt_alphas_cumprod etc.), and q_sample's index_select crashes.
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
            for t in range(noise_timestep, 0, -1):
                t_tensor = torch.full((z.shape[0],), t, device=device, dtype=torch.long)
                z = self._model.p_sample(x=z, c=cond, t=t_tensor, clip_denoised=self._model.clip_denoised)

            mel_out = self._model.decode_first_stage(z)
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
    """Held-out attack (same status as DiffEraseAttack -- eval only, NEVER
    given nonzero weight in AttackConfig.weights during training): Meta's
    MultiBand Diffusion (github.com/facebookresearch/audiocraft,
    `docs/MBD.md`), an EnCodec-conditioned diffusion decoder. Per the
    mentor's plan (2026-08-12 Notion doc), this is the intended replacement/
    complement for DiffEraseAttack's AudioLDM as "a diffusion model not
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

    `strength`: unlike DiffEraseAttack/SGMSEAttack, MBD has no continuous
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
        # DiffEraseAttack's torch.load patch).
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
    """

    def __init__(self, attacks: tp.Dict[str, nn.Module], weights: tp.Dict[str, float]):
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

    def forward(
        self, x_wm: torch.Tensor, strength: tp.Optional[float] = None
    ) -> tp.Tuple[torch.Tensor, str]:
        """`strength`: passed through to whichever attack gets sampled (see
        e.g. SGMSEAttack's docstring for what it means there; ignored by
        attacks that don't use it). Leave it None during training -- each
        strength-aware attack should sample its own random t* per call in
        that case, which is what should give robustness across attack
        strengths rather than at a single fixed one."""
        name = random.choices(self._names, weights=self._sampling_weights, k=1)[0]
        attack = self.attacks[name]
        # No torch.no_grad() and no .detach() here: see module docstring.
        x_att = attack(x_wm, strength=strength)
        return x_att, name
