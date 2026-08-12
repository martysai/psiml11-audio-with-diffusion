# Vendored: sgmse

Copied from `github.com/sp-uhh/sgmse` ("Speech Enhancement and Dereverberation
with Diffusion-based Generative Models" / "Analysing Diffusion-based
Generative Approaches versus Discriminative Approaches for Speech
Restoration"). MIT licensed (see `LICENSE` in this directory). No local
modifications.

Used by `audioseal_robust.attacks.SGMSEAttack` as one of the trainable
reconstruction attacks -- a real (not held-out) diffusion-based speech
enhancement model, run on watermarked audio to see if the watermark washes
out as "noise" the way real acoustic degradation would. See that class's
docstring for how its OU-VE SDE (continuous time, predictor/corrector
sampler) maps onto this project's `strength`/t* convention -- it's a
different mechanism from DiffEraseAttack's DDPM (src/audioldm_train/), not
just a different checkpoint.

Checkpoint used in this project: `train_vb_29nqe0uh_epoch=115.ckpt`
(VoiceBank-DEMAND, 16kHz enhancement), from the Google Drive links in the
upstream README -- not included here, multi-GB binary, supplied via
`attack.sgmse.checkpoint` in config.
