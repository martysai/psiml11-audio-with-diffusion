# Vendored: audioldm_train

Copied from `DiffErase-latent/audioldm_train` in the `Differase` repo
(github.com/DiffErase/Differase, "Audio Pirates: Black-box Audio Watermark
Removal via Diffusion Priors", arXiv:2605.30614), itself a fork of
`haoheliu/AudioLDM-training-finetuning`. MIT licensed (see `LICENSE` in this
directory).

Used by `audioseal_robust.attacks.DiffEraseAttack` to run inference with a
pretrained AudioLDM latent-diffusion model (VAE + UNet + HiFiGAN vocoder) as
the "diffusion prior" for the `diff_erase` held-out watermark-erasure attack
-- see that class's docstring for why (the paper explicitly states no
fine-tuning is applied; it uses an off-the-shelf pretrained checkpoint).

## Local modifications from upstream

`utilities/audio/stft.py`, `TacotronSTFT.mel_spectrogram` and
`STFT.transform`: make the mel front-end differentiable w.r.t. its input
waveform. `mel_spectrogram` did `magnitudes = magnitudes.data.to(...)`, and
`.data` detaches -- so the returned mel had `grad_fn is None` and no gradient
could reach the waveform, no matter what happened downstream. Under
`DiffEraseAttack`'s current eval-only `torch.no_grad()` path that is
invisible, but it silently breaks any attempt to *train* against this model
(the detection loss simply stops reaching the generator, leaving only the
perceptual term -- which drives watermark amplitude to zero rather than
raising an error). Removing `.data` changes no values.

Reconnecting the graph then exposes a second problem in `STFT.transform`:
`torch.sqrt(real**2 + imag**2)` has an infinite derivative at zero, i.e. on
every digitally-silent STFT bin, so the newly-live gradient would arrive as
NaN. A `1e-12` floor inside the sqrt fixes that and is invisible to the
output, since `dynamic_range_compression` already clamps at `1e-5` -- three
orders of magnitude above the `1e-6` magnitude the floor implies.

`phase`'s own `.data` is deliberately left alone: `atan2` is undefined at the
origin, `DiffEraseAttack` discards phase anyway, and only the mel is needed
differentiable.

`modules/latent_diffusion/ddpm.py`: removed `self.clap` (the
`CLAPAudioEmbeddingClassifierFreev2` instance built unconditionally in
`DDPM.__init__`, requiring a ~2.35GB checkpoint
`clap_music_speech_audioset_epoch_15_esc_89.98.pt`). Confirmed via `grep`
that it's used nowhere except `LatentDiffusion.generate_sample`'s
CLAP-similarity candidate re-ranking -- a training/eval-time convenience
never called from our inference path (`encode_first_stage` /
`get_first_stage_encoding` / `q_sample` / `p_sample` / `decode_first_stage`).
Removing it drops an entire checkpoint download for zero behavior change on
the path we actually use.

Do NOT otherwise diverge from upstream without re-verifying the specific
change against `attacks.py:DiffEraseAttack`'s actual call sequence --
several other "unused-looking" branches (e.g. the `film_clap_cond1`
conditioner, `cond_stage_models[0].get_unconditional_condition`) turned out
to be structurally required (the UNet's FiLM layer asserts its conditioning
input is never `None`) despite the model running fully unconditionally --
see attacks.py's inline comments before assuming similar branches elsewhere
are safe to cut.
