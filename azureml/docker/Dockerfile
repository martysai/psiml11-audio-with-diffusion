# Build context is the repo root (see azureml/environment.yaml's build.path),
# so the requirements-*.txt files below are COPY-able.
#
# Why a Dockerfile and not a conda.yaml: audiocraft has to be installed with
# --no-deps *before* everything else (see requirements-mbd.txt's header --
# its closure drags in spacy -> thinc, whose Cython sources don't build
# against numpy>=2). A conda.yaml pip: block gives no control over install
# order or per-package flags, so it can't express that.
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# libsndfile1 is what `soundfile` binds to -- data.py:WavDirDataset reads
# LibriSpeech's .flac through it deliberately, to avoid torchaudio.load()'s
# torchcodec/FFmpeg path. ffmpeg is for PyAV (`av`), pulled in by audiocraft
# for the mbd held-out attack. build-essential covers any sdist without a
# cp310 wheel (`pesq` is the usual offender).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        git \
        build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# Pinned as a matched set, installed first so everything after it resolves
# against these versions rather than dragging in a different torch:
#   torch 2.4.0 <-> torchvision 0.19.0 <-> torchaudio 2.4.0 <-> xformers 0.0.27.post2
# xformers publishes one wheel per exact torch version and audiocraft imports
# it at module level, so a mismatch surfaces during *eval* -- i.e. an hour
# into a GPU run, after training already burned the budget.
#
# cu118, matching this base image's CUDA 11.8, and not cu121 -- for two
# independent reasons:
#
#   1. cu121 is now unbuildable at these pins. torch 2.4.x hard-pins
#      nvidia-cudnn-cu12==9.1.0.70, which has since been withdrawn from both
#      PyPI and the cu121 index (they now jump 9.0.0.312 -> 9.1.1.17). Nothing
#      can satisfy it. The cu118 index still carries nvidia-cudnn-cu11 at
#      exactly 9.1.0.70, so the same torch pin resolves there.
#   2. xformers' last true cu118 build is 0.0.27.post2+cu118. Later versions
#      ship a single wheel built against cu121, whose extension links
#      libcudart.so.12 and therefore cannot load against a cu118 torch. That
#      wheel is what pins torch to 2.4.0 rather than 2.4.1.
#
# xformers is installed HERE, from this same index, rather than being left to
# requirements-mbd.txt -- that later layer resolves against PyPI, which would
# hand back the cu121 build and reintroduce exactly the mismatch above.
#
# A100 is sm_80 and fully supported by CUDA 11.8, so nothing is lost here.
RUN python3 -m pip install \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        xformers==0.0.27.post2 \
        --index-url https://download.pytorch.org/whl/cu118

# Same pins restated as pip constraints so the (deliberately unpinned)
# requirements files below cannot silently upgrade torch out from under us.
# `==2.4.0` also matches the installed `2.4.0+cu118` -- PEP 440 ignores the
# local version segment here -- so these satisfy rather than trigger a
# reinstall from PyPI.
#
# transformers is pinned for a different reason than the torch stack, and it is
# not cosmetic: unpinned it resolves to 5.x, which requires a much newer torch
# than 2.4. When transformers decides torch is unavailable it does NOT fail
# cleanly -- transformers/integrations/tensor_parallel.py guards its `import
# torch` behind is_torch_available() but then subclasses torch.autograd.Function
# at module scope regardless, so the import dies with a bare
# `NameError: name 'torch' is not defined` that says nothing about versions.
# That is what killed all 4 ranks of <redacted-run>.
#
# 4.x is also the right major on API grounds: the vendored audioldm_train is
# 2023-era code importing RobertaTokenizer/T5EncoderModel/MT5EncoderModel,
# SpeechT5Processor + SpeechT5EncoderWithTextPrenet, GPT2/GPTJ, Bert and Bart.
# Bumping torch to satisfy 5.x instead would unravel the cu118 + xformers +
# audiocraft matching above, which is a far larger change for no benefit.
RUN printf '%s\n' \
        'torch==2.4.0' \
        'torchvision==0.19.0' \
        'torchaudio==2.4.0' \
        'xformers==0.0.27.post2' \
        'numpy<2' \
        'transformers==4.46.3' \
        > /tmp/constraints.txt

COPY requirements.txt \
     requirements-training.txt \
     requirements-audioldm.txt \
     requirements-mbd.txt \
     /tmp/reqs/

# torchcodec is dropped on purpose. requirements.txt carries it only for
# torchaudio.load(), which nothing under src/audioseal_robust calls (data.py
# uses soundfile instead -- see its comment at data.py:45), and its wheels
# pin an exact torch build that would fight the pin above.
RUN sed -i '/^torchcodec/d' /tmp/reqs/requirements.txt

# --no-deps, per requirements-mbd.txt's "INSTALL IN TWO STEPS" header. The
# packages audiocraft actually needs at import/run time are listed in
# requirements-mbd.txt and get installed normally in the next layer.
RUN python3 -m pip install --no-deps audiocraft

RUN python3 -m pip install -c /tmp/constraints.txt \
        -r /tmp/reqs/requirements.txt \
        -r /tmp/reqs/requirements-training.txt \
        -r /tmp/reqs/requirements-audioldm.txt \
        -r /tmp/reqs/requirements-mbd.txt

# Fail the *image build* rather than the GPU job if the stack is
# inconsistent.
#
# pip check is filtered rather than trusted wholesale. audiocraft is installed
# --no-deps on purpose (see requirements-mbd.txt), so it permanently reports
# its own unmet metadata: torch==2.1.0, torchvision==0.16.0, xformers<0.0.23,
# av==11.0.0, plus MusicGen-only extras (demucs, encodec, gradio, hydra-core,
# sentencepiece, torchtext) that the MBD import chain never touches. Honouring
# those pins would mean downgrading torch three minor versions for a held-out
# eval attack. Anything NOT from audiocraft is still fatal.
RUN conflicts="$(python3 -m pip check | grep -v '^audiocraft ' || true)"; \
    if [ -n "$conflicts" ]; then \
        echo "unexpected (non-audiocraft) dependency conflicts:" >&2; \
        echo "$conflicts" >&2; \
        exit 1; \
    fi; \
    echo "pip check: only the expected audiocraft --no-deps mismatches"

# Every import here is one this project actually performs: the audioldm_train
# chain for the audioldm attack, the sgmse chain, and
# audiocraft.models.MultiBandDiffusion for mbd (attacks.py:667).
#
# `import transformers` on its own is NOT a sufficient check and used to be all
# this did -- transformers is a _LazyModule, so the bare import succeeds even
# when the package is fundamentally broken against the installed torch, and the
# failure only surfaces once something pulls in modeling_utils. That is exactly
# how a transformers 5.x/torch 2.4 mismatch got past this build and instead
# killed a 4x A100 job at import time. So import the concrete model classes
# audioldm_train/conditional_models.py imports, and assert is_torch_available()
# directly for an error message that names the real problem.
#
# Deliberately `python3 -c` on one line rather than a heredoc: this Dockerfile
# can reach the daemon with CRLF endings, and a `<<'PY'` terminator arriving as
# "PY\r" never matches, so the heredoc swallows the rest of the build.
RUN python3 -c "\
import numpy, torch, torchaudio, torchvision, soundfile, librosa, scipy, pandas, h5py; \
import pytorch_lightning, torchlibrosa, transformers, timm, ftfy; \
import webdataset, braceexpand, wget, sklearn, regex, PIL, yaml, tqdm; \
import torch_ema, torchsde, torchinfo, pystoi; \
import torchmetrics, pesq, matplotlib, mlflow, wandb; \
import xformers, julius, num2words, av, flashy, spacy; \
from transformers.utils import is_torch_available; \
assert is_torch_available(), 'transformers ' + transformers.__version__ + ' does not consider torch ' + torch.__version__ + ' usable; it will NameError on torch at module scope'; \
from transformers import RobertaTokenizer, AutoTokenizer, T5EncoderModel, MT5EncoderModel; \
from transformers import SpeechT5Processor, GPT2Model, GPT2Tokenizer, GPT2Config, GPTJConfig, GPTJModel; \
from transformers import BertTokenizer, BertModel, BartTokenizer, BartModel, RobertaModel; \
from transformers.models.speecht5.modeling_speecht5 import SpeechT5EncoderWithTextPrenet; \
from transformers.tokenization_utils_base import BatchEncoding; \
from audiocraft.models import MultiBandDiffusion; \
assert numpy.__version__.startswith('1.'), 'numpy 2.x breaks this stack: ' + numpy.__version__; \
assert transformers.__version__.startswith('4.'), 'transformers must stay on 4.x for torch 2.4: ' + transformers.__version__; \
print('torch', torch.__version__, '| torchaudio', torchaudio.__version__, '| numpy', numpy.__version__, '| xformers', xformers.__version__, '| transformers', transformers.__version__)"
