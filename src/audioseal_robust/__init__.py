# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generator-only fine-tuning of AudioSeal against sampled reconstruction
attacks (Identity / BigVGAN / DAC / SGMSE), with the detector frozen.

See train.py for the training loop, attacks.py for the attack module,
losses.py for the loss functions, and config.py for the config schema.
"""
