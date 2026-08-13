# Vendored: sgmse

Copied from `github.com/sp-uhh/sgmse` ("Speech Enhancement and Dereverberation
with Diffusion-based Generative Models" / "Analysing Diffusion-based
Generative Approaches versus Discriminative Approaches for Speech
Restoration"). MIT licensed (see `LICENSE` in this directory).

## Local modifications

`sdes.py`, `SDE.discretize`: batch-broadcast the `stepsize` argument so it may
be a per-example `(batch,)` tensor instead of only a scalar. `SGMSEAttack`
starts each item of a batch at its own `t*` (see that class's `strength`
handling), which makes every item's reverse-diffusion step size different;
upstream multiplies the `(batch, 1, F, T)` drift by the raw `stepsize`, which
only broadcasts when it is a scalar. A scalar `stepsize` still takes exactly
the original code path, so upstream's own samplers are unaffected. This also
happens to fix `get_ode_sampler`'s `denoise_update_fn`, which already passed a
`(batch,)` `vec_eps` as the stepsize and would have raised on any batch whose
size differed from its frame count.

Used by `audioseal_robust.attacks.SGMSEAttack` as one of the trainable
reconstruction attacks -- a real (not held-out) diffusion-based speech
enhancement model, run on watermarked audio to see if the watermark washes
out as "noise" the way real acoustic degradation would. See that class's
docstring for how its OU-VE SDE (continuous time, predictor/corrector
sampler) maps onto this project's `strength`/t* convention -- it's a
different mechanism from AudioLDMAttack's DDPM (src/audioldm_train/), not
just a different checkpoint.

Checkpoint used in this project: `train_vb_29nqe0uh_epoch=115.ckpt`
(VoiceBank-DEMAND, 16kHz enhancement), from the Google Drive links in the
upstream README -- not included here, multi-GB binary, supplied via
`attack.sgmse.checkpoint` in config.
