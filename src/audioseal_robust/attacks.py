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
import sys
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
    last K steps, rather than the full inference-time step count.

    TODO(setup): not vendored/installed yet. To enable:
      1. Get SGMSE source (e.g. github.com/sp-uhh/sgmse) and a pretrained
         checkpoint (their released checkpoints are usually 16kHz, which
         matches this pipeline).
      2. Fill in `_load_backbone` below with model construction + checkpoint
         loading, and implement `forward` to run the (differentiable) reverse
         SDE sampler for `num_steps` steps.
      3. Remove the `NotImplementedError`s.

    `strength` (the t* robustness-curve axis, see evaluate.py): SGMSE-style
    diffusion purification works by forward-diffusing the input up to some
    starting timestep t*, then reverse-diffusing (denoising) it back down --
    the further you push t* toward 1, the more the input is corrupted before
    regeneration, i.e. a *stronger* attack that erases more of the original
    signal (including the watermark) but also more of the content. t*=0
    means "no corruption" (should reduce to ~identity); t*=1 means "maximum
    corruption then full regeneration". During training (see attacks.py's
    SampledReconstructionAttack, called with strength=None from train.py),
    sample t* randomly per step instead of fixing it -- that's what should
    give robustness across attack strengths rather than just at whatever
    single t* you'd otherwise hardcode; the evaluation robustness curve
    (detection vs. t*) is the direct measurement of whether that worked.
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
        self._backbone: tp.Optional[nn.Module] = None
        if checkpoint is not None:
            self._backbone = self._load_backbone(checkpoint)
            self._backbone.eval()
            for p in self._backbone.parameters():
                p.requires_grad_(False)

    def _load_backbone(self, checkpoint: str) -> nn.Module:
        raise NotImplementedError(
            "SGMSEAttack has no backbone loading implemented yet. Install "
            "SGMSE and implement the score model + reverse SDE sampler here, "
            "then pass attack.sgmse.checkpoint=<path> in the config."
        )

    def forward(self, x: torch.Tensor, strength: tp.Optional[float] = None) -> torch.Tensor:
        """`strength` is t* in [0, 1] (see class docstring); None means
        "sample it randomly", which is what training should do."""
        if self._backbone is None:
            raise NotImplementedError(
                "SGMSEAttack was constructed without a checkpoint (see TODO "
                "in src/audioseal_robust/attacks.py). Either provide "
                "attack.sgmse.checkpoint or set attack.weights.sgmse=0 in the "
                "config to keep it disabled."
            )
        raise NotImplementedError("SGMSE reverse-diffusion forward pass TODO")


class DiffEraseAttack(nn.Module):
    """Held-out diffusion-based watermark-erasure attack: used ONLY for
    evaluation (see evaluate.py's `held_out_attacks`), NEVER given nonzero
    weight in `AttackConfig.weights` during training. The whole point of
    holding it out is to answer "did robustness generalize to an unseen
    diffusion attack, or did the generator just memorize the specific
    attack(s) it was trained against" -- if you add it to the training
    sampler too, you've destroyed the thing this is meant to measure.

    Wired to DiffErase-latent (github.com/<...>/Differase's "Audio Pirates"
    project, the AudioLDM-style latent-diffusion variant): forward-diffuses
    a mel-spectrogram of `x` up to timestep `strength * num_timesteps` (same
    t* convention as SGMSEAttack -- 0 = no corruption, 1 = full
    noise-then-regenerate), reverse-diffuses it back with the pretrained
    LatentDiffusion model, and vocodes the result back to a waveform with
    the accompanying HiFiGAN. This mirrors the reference
    `Differase/remove_differase-latent.py` script's `process_audio_batch`
    exactly, just working on in-memory batched tensors (so it slots into
    evaluate.py's per-batch loop) instead of reading/writing wav files.

    That repo is a separate checkout, not vendored here (same situation as
    the AudioCraft/Dora solver mentioned in config.py's module docstring),
    and ships no pretrained checkpoint of its own -- `DiffErase-latent`'s own
    README is literally about *training* one. So this attack stays disabled
    (constructing it without a checkpoint keeps raising NotImplementedError,
    same as before) until you point all three of `checkpoint`, `config`, and
    `differase_root` at a real checkout + a checkpoint you've trained or
    otherwise obtained -- see EvalAttackConfig.diff_erase in config.py.

    Differentiability isn't required here since eval never backprops (unlike
    SGMSEAttack, which trains against attacks and does need it) -- the whole
    reverse-diffusion loop runs under `torch.no_grad()`.
    """

    def __init__(
        self,
        checkpoint: tp.Optional[str] = None,
        config: tp.Optional[str] = None,
        differase_root: tp.Optional[str] = None,
        sample_rate: int = 16_000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.checkpoint = checkpoint
        self.config_path = config
        self.differase_root = differase_root
        self._model: tp.Optional[nn.Module] = None
        self._vocoder: tp.Optional[nn.Module] = None
        self._stft: tp.Optional[nn.Module] = None
        self._duration: tp.Optional[float] = None
        self._model_sample_rate: tp.Optional[int] = None
        if checkpoint is not None:
            self._load_backbone(checkpoint)

    def _load_backbone(self, checkpoint: str) -> None:
        if self.config_path is None or self.differase_root is None:
            raise NotImplementedError(
                "DiffEraseAttack needs `config` and `differase_root` set "
                "alongside `checkpoint` -- see EvalAttackConfig.diff_erase "
                "in src/audioseal_robust/config.py."
            )
        differase_root = Path(self.differase_root)
        latent_root = differase_root / "DiffErase-latent"
        if not latent_root.is_dir():
            raise FileNotFoundError(
                f"DiffErase-latent not found under differase_root={differase_root} "
                "-- point eval.diff_erase.differase_root at a checkout of the "
                "Differase repo (the parent of DiffErase-latent/, DiffErase-mel/)."
            )
        if str(latent_root) not in sys.path:
            sys.path.insert(0, str(latent_root))

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

        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"DiffErase-latent checkpoint not found: {ckpt_path}")

        original_cwd = os.getcwd()
        original_torch_load = torch.load
        os.chdir(latent_root)  # get_vocoder() hardcodes a "data/checkpoints" *relative* path
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
                "set eval.diff_erase.{checkpoint,config,differase_root} or "
                "remove diff_erase from eval.held_out_attacks to skip it."
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
            noise_timestep = int(num_timesteps * strength)
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
