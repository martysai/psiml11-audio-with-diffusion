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

import random
import typing as tp

import torch
import torch.nn as nn


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

    "DiffErase" here is a placeholder name for whatever specific
    diffusion-based erasure method/checkpoint you evaluate against --
    fill in `_load_backbone` once you've picked one (could be a different
    SGMSE checkpoint/config than the training attack, a DiffPure-style
    purifier, or another published diffusion watermark-removal method).
    Structurally identical to SGMSEAttack: same `strength` = t* convention
    (see SGMSEAttack's docstring), same frozen-but-differentiable wiring
    (differentiability isn't actually required here since eval never
    backprops, but keeping the same interface means evaluate.py can call
    every attack, including this one, uniformly).

    TODO(setup): pick the actual method/checkpoint, then fill in
    `_load_backbone` and `forward` the same way as SGMSEAttack.
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
            "DiffEraseAttack has no backbone loading implemented yet -- pick "
            "the specific held-out diffusion erasure method/checkpoint you "
            "want to evaluate generalization against, then implement this "
            "and pass eval.diff_erase.checkpoint=<path> in the eval config."
        )

    def forward(self, x: torch.Tensor, strength: tp.Optional[float] = None) -> torch.Tensor:
        if self._backbone is None:
            raise NotImplementedError(
                "DiffEraseAttack was constructed without a checkpoint (see "
                "TODO in src/audioseal_robust/attacks.py). Remove it from "
                "eval.held_out_attacks to skip it until then."
            )
        raise NotImplementedError("held-out diffusion erasure forward pass TODO")


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
