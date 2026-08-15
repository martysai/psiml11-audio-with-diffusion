# Publishing a fine-tuned generator

`train.py` writes checkpoints for *us*; a model hub needs a different file. This
page covers the one command in between, and why it is not optional.

## TL;DR

```bash
PYTHONPATH=src python -m audioseal_robust.export_checkpoint \
    checkpoints/<run>/generator_epoch18.pth \
    dist/generator.pth
```

Upload `dist/generator.pth`. Downstream, the entire usage is:

```python
from audioseal import AudioSeal
from huggingface_hub import hf_hub_download

generator = AudioSeal.load_generator(
    hf_hub_download("msaidov/audioseal-robust-audioldm-16bits", "generator.pth")
)
```

No `torch.load`, no `nbits=`, no state-dict surgery, and nothing installed
beyond stock `audioseal`.

## Why the raw checkpoint cannot be published

`train.py` saves `torch.save({"model": ..., "xp.cfg": cfg}, path)`. Two things
about that file only work inside this repo.

**1. `xp.cfg` pickles this repo's types by reference.** `cfg` comes from
`OmegaConf.structured(TrainConfig)`, so the pickle contains `GLOBAL` references
to `audioseal_robust.config.TrainConfig`, `GeneratorConfig`, `AttackConfig` and
eleven more. A user without this repo importable does not get a confusing model,
they get:

```
ModuleNotFoundError: No module named 'audioseal_robust'
```

straight out of `torch.load`, before touching a single weight. And installing the
repo does not fix it either: `AudioSeal.parse_model` reads `xp.cfg` when it is
present and hands it to `parse_config`, which does `assert "seanet" in config`.
A `TrainConfig` has no `seanet`, so the stock loader rejects the file outright.

**2. The conv keys are in the naming only half the world can read.**
`audioseal/builder.py` picks its SEANet by interpreter version: Moshi's on Python
≥ 3.10, whose convs sit under an extra `inner_conv` level, and Audiocraft's below
that, whose keys are flat (`....conv.conv.weight`). Upstream ships its own
checkpoints *flat* on purpose — that is the naming both builds can consume,
because `convert_state_dict_for_scriptable_model` up-converts flat → `inner_conv`
on new interpreters and no-ops on old ones. There is no path in the other
direction, so a checkpoint saved as `inner_conv` (i.e. anything we train on
Python ≥ 3.10) simply fails to load on Python < 3.10, with
`Missing key(s)`/`Unexpected key(s)` on every conv layer.

## What the exporter does about it

`audioseal_robust.export_checkpoint`:

- flattens `.inner_conv.` out of the state dict, so the published weights match
  upstream's convention and load on either side of the 3.10 split;
- replaces `xp.cfg` with the `audioseal_wm_16bits` architecture (`nbits`,
  `seanet`, `decoder`) as **plain dicts and lists**, which is what
  `parse_config` wants and costs the consumer no imports. The base model's
  `checkpoint:` download URL is dropped — it points at the *stock* weights and
  would be misleading inside a derived artifact;
- keeps the full training config, converted to plain containers, under an
  `"audioseal_robust"` key that the stock loader ignores. Nothing is lost, it is
  just out of the load path;
- verifies the result before letting you have it: the file is staged under a
  temporary name, reloaded through `AudioSeal.load_generator`, compared tensor
  by tensor against the source checkpoint and on a watermark forward pass, and
  only then renamed into place. A failed export leaves no file behind.

The result is structurally the same file upstream publishes as
`generator_base.pth`, so `AudioSeal.load_generator` needs no special-casing.

## Loading it back

Any of these work, and all of them go through the same stock code path:

```python
AudioSeal.load_generator("dist/generator.pth")                       # local
AudioSeal.load_generator(hf_hub_download(repo_id, "generator.pth"))  # cached HF
AudioSeal.load_generator("https://huggingface.co/<repo>/resolve/main/generator.pth")
```

The last one goes through `torch.hub.load_state_dict_from_url` and needs no
`huggingface_hub` install at all.

The detector is untouched by this project and is always the stock one:

```python
detector = AudioSeal.load_detector("audioseal_detector_16bits")
```

## Resuming, evaluating, and the naming split

The same 3.10 naming mismatch bites anywhere a checkpoint crosses machines, so
three call sites share one helper, `audioseal.loader.align_state_dict_to_model`,
which picks the direction from the model's and checkpoint's actual keys rather
than from `sys.version_info`:

- `AudioSeal.load_generator` / `load_detector` (so either naming loads),
- `evaluate.load_generator_under_test` (`generator_checkpoint=<path>`),
- `train.build_generator` (`generator.resume_from=<path>`).

Raw `generator_epochN.pth` files therefore still work as
`generator_checkpoint=` and `generator.resume_from=` inputs; they only need
exporting to be *published*, or to be usable as `generator.checkpoint=`.

## Checking your work

```bash
PYTHONPATH=src python -m pytest tests/test_export_checkpoint.py -v
```

`test_exported_checkpoint_pickles_no_project_types` is the one that matters: it
unpickles the exported file in a `python -I` subprocess that cannot import this
repo, which is exactly the position a downstream user is in.
