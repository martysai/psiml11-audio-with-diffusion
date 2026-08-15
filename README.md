# psiml11-audio-with-diffusion
Audio ML project with Diffusion models applications.

## Released models

Robustness fine-tuned AudioSeal generators, each hardened against a different
diffusion-based watermark-removal attack. Both are drop-in replacements for the
stock `audioseal_wm_16bits` generator and keep working with the **unmodified**
`audioseal_detector_16bits` detector.

| model | trained against |
| --- | --- |
| [`msaidov/audioseal-robust-audioldm-16bits`](https://huggingface.co/msaidov/audioseal-robust-audioldm-16bits) | AudioLDM latent-diffusion resynthesis |
| [`msaidov/audioseal-robust-sgmse-16bits`](https://huggingface.co/msaidov/audioseal-robust-sgmse-16bits) | SGMSE OU-VE SDE speech enhancement |

```python
from audioseal import AudioSeal
from huggingface_hub import hf_hub_download

generator = AudioSeal.load_generator(
    hf_hub_download("msaidov/audioseal-robust-audioldm-16bits", "generator.pth")
)
detector = AudioSeal.load_detector("audioseal_detector_16bits")
```

## Docs

- [docs/MULTI_GPU.md](docs/MULTI_GPU.md) -- running training and evaluation
  across multiple GPUs with `torchrun`, and which config fields are per-GPU
  vs. global.
- [docs/PUBLISHING.md](docs/PUBLISHING.md) -- turning a `train.py` checkpoint
  into one a stranger can load, and why the raw one is not publishable.
- [docs/TRAINING.md](docs/TRAINING.md) -- training a new AudioSeal
  watermarking model via AudioCraft/Dora.

