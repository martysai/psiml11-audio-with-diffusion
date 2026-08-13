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
#   torch 2.4.1 <-> torchvision 0.19.1 <-> torchaudio 2.4.1 <-> xformers 0.0.28.post1
# xformers publishes one wheel per exact torch version and audiocraft imports
# it at module level, so a mismatch surfaces during *eval* -- i.e. an hour
# into a GPU run, after training already burned the budget.
RUN python3 -m pip install \
        torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121

# Same pins restated as pip constraints so the (deliberately unpinned)
# requirements files below cannot silently upgrade torch out from under us.
RUN printf '%s\n' \
        'torch==2.4.1' \
        'torchvision==0.19.1' \
        'torchaudio==2.4.1' \
        'xformers==0.0.28.post1' \
        'numpy<2' \
        > /tmp/constraints.txt

COPY requirements.txt \
     requirements-training.txt \
     requirements-diff-erase.txt \
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
        -r /tmp/reqs/requirements-diff-erase.txt \
        -r /tmp/reqs/requirements-mbd.txt

# Fail the *image build* rather than the GPU job if the stack is
# inconsistent. Every import below is one this project actually performs:
# the audioldm_train chain for diff_erase, the sgmse chain, and
# audiocraft.models.MultiBandDiffusion for mbd (attacks.py:667).
RUN python3 -m pip check && python3 - <<'PY'
import numpy, torch, torchaudio, torchvision
import soundfile, librosa, scipy, pandas, h5py
import pytorch_lightning, torchlibrosa, transformers, timm, ftfy
import webdataset, braceexpand, wget, sklearn, regex, PIL, yaml, tqdm
import torch_ema, torchsde, torchinfo, pystoi
import torchmetrics, pesq, matplotlib, mlflow, wandb
import xformers, julius, num2words, av, flashy, spacy
from audiocraft.models import MultiBandDiffusion  # attacks.py:667

assert numpy.__version__.startswith("1."), f"numpy 2.x breaks this stack: {numpy.__version__}"
print("torch", torch.__version__, "| torchaudio", torchaudio.__version__,
      "| numpy", numpy.__version__, "| xformers", xformers.__version__)
PY
