# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generator-only fine-tuning of AudioSeal against sampled reconstruction
attacks (Identity / BigVGAN / DAC / AudioLDM), with the detector frozen.
SGMSE is held out for evaluation only (see evaluate.py's held_out_attacks) --
this project variant trains against AudioLDM-style latent-diffusion
resynthesis (AudioLDM) and measures whether robustness generalizes to
SGMSE's structurally different diffusion attack, rather than the reverse.

See train.py for the training loop, attacks.py for the attack module,
losses.py for the loss functions, and config.py for the config schema.
"""
