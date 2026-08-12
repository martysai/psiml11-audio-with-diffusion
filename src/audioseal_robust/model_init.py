# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build generator/detector with the real production architecture (same
hyperparameters as the shipped `audioseal_wm_16bits` / `audioseal_detector_16bits`
cards) but WITHOUT downloading pretrained weights.

This exists for the throughput sanity check (see sanity_check.py): we want
to measure steps/sec on the actual model size that will be trained, without
requiring network access to Hugging Face every time someone runs it. Real
training should still start from the pretrained checkpoint via
`AudioSeal.load_generator` / `AudioSeal.load_detector` (see train.py) --
this module intentionally reuses the same config-parsing code
(`audioseal.loader.AudioSeal.parse_config`) so the architecture is guaranteed
identical, it just skips the checkpoint download + state_dict load.
"""

import typing as tp

import torch
from omegaconf import OmegaConf

from audioseal.builder import (
    AudioSealDetectorConfig,
    AudioSealWMConfig,
    create_detector,
    create_generator,
)
from audioseal.loader import AudioSeal as AudioSealLoader
from audioseal.loader import load_local_model_config
from audioseal.models import AudioSealDetector, AudioSealWM


def _parse_local_card(card_name: str, config_type, nbits: tp.Optional[int]):
    raw = load_local_model_config(card_name)
    assert raw is not None, f"No local model card named {card_name!r} under src/audioseal/cards/"
    config_dict = OmegaConf.to_container(raw)
    assert isinstance(config_dict, dict)
    config_dict.pop("checkpoint", None)  # skip: we are not loading pretrained weights
    return AudioSealLoader.parse_config(config_dict, config_type=config_type, nbits=nbits)


def build_untrained_generator(
    card_name: str = "audioseal_wm_16bits",
    nbits: tp.Optional[int] = None,
    device: tp.Optional[torch.device] = None,
) -> AudioSealWM:
    model_config = _parse_local_card(card_name, AudioSealWMConfig, nbits)
    return create_generator(model_config, device=device)


def build_untrained_detector(
    card_name: str = "audioseal_detector_16bits",
    nbits: tp.Optional[int] = None,
    device: tp.Optional[torch.device] = None,
) -> AudioSealDetector:
    model_config = _parse_local_card(card_name, AudioSealDetectorConfig, nbits)
    return create_detector(model_config, device=device)
