# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Config schema for generator-only robustness fine-tuning.

This project vendors a standalone training script rather than a Dora
solver (see discussion in the conversation that produced this module: the
Dora solver lives in a *separate* AudioCraft checkout on this machine, not in
this repo, and pulling in Dora's full env/SLURM/xp-management stack for a
single vendored script would be disproportionate). Config is instead handled
with OmegaConf directly: a structured dataclass schema (this file) merged
with `config/default.yaml` and then with CLI overrides
(`python -m audioseal_robust.train lambda_det=2.0 attack.weights.sgmse=1.0`),
which is the same override mechanism Dora/Hydra are themselves built on.

`lambda_det` and `lambda_perc` are ordinary config fields here -- see the
`total_loss` computation in `train.py` -- never hardcoded.
"""

import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class GeneratorConfig:
    # Model card name (see src/audioseal/cards/) or path/HF uri to a
    # checkpoint, passed to audioseal.AudioSeal.load_generator.
    checkpoint: str = "audioseal_wm_16bits"

    # Optional path to a train.py-saved checkpoint (a .pth with a "model"
    # key, e.g. checkpoints/.../generator_epochN.pth) to load ON TOP of
    # `checkpoint` above once the model is built. Kept separate from
    # `checkpoint` because that field goes through AudioSeal.load_generator's
    # parse_model/parse_config, which requires a full model config (asserts
    # "seanet" in config) that our saved dicts don't carry -- only the
    # architecture-defining pretrained checkpoint can go there. This just
    # does a plain load_state_dict for resuming fine-tuning from a later
    # epoch without retraining from the pretrained baseline.
    resume_from: tp.Optional[str] = None


@dataclass
class DetectorConfig:
    # Model card name or path/HF uri, passed to audioseal.AudioSeal.load_detector.
    # This model is always frozen (see train.py:build_detector) -- never
    # placed in the optimizer, never taken out of eval().
    checkpoint: str = "audioseal_detector_16bits"


@dataclass
class BigVGANAttackConfig:
    checkpoint: tp.Optional[str] = None  # TODO: fill in once BigVGAN is installed


@dataclass
class DACAttackConfig:
    checkpoint: tp.Optional[str] = None  # TODO: fill in once DAC is installed


@dataclass
class SGMSEAttackConfig:
    checkpoint: tp.Optional[str] = None  # TODO: fill in once SGMSE is installed
    num_steps: int = 30


@dataclass
class AudioLDMAttackConfig:
    """Wires AudioLDMAttack (attacks.py) to pretrained AudioLDM weights.
    The model *code* (audioldm_train, MIT licensed) is vendored under
    src/audioldm_train/ -- see that directory's VENDORED.md -- so only the
    *weights* need to be supplied externally (multi-GB, never belongs in
    git). Both fields must be set together, or left None (default) to keep
    the attack disabled.
    """

    # A full LatentDiffusion checkpoint (VAE + diffusion UNet), NOT the
    # VAE-only checkpoint -- must live at <weights_root>/data/checkpoints/<file>
    # (see attacks.py:AudioLDMAttack._load_backbone for why: get_vocoder()
    # and the VAE's reload_from_ckpt both hardcode that relative layout).
    # E.g. `audioldm-s-full` from the AudioLDM authors' Zenodo record,
    # placed at <weights_root>/data/checkpoints/audioldm-s-full alongside
    # the VAE/CLAP/HiFiGAN support bundle that config references by name.
    checkpoint: tp.Optional[str] = None
    # Matching audioldm_train config -- the vendored
    # config/2023_08_23_reproduce_audioldm/audioldm_original.yaml matches a
    # standard/small AudioLDM checkpoint's architecture). NOTE: every shipped
    # LatentDiffusion config in that repo is CLAP-conditioned, which pulls in
    # its own pretrained sub-checkpoint at model-construction time (e.g.
    # audioldm_original.yaml needs a `clap_htsat_tiny.pt` next to `checkpoint`,
    # under the same data/checkpoints/ dir) even though AudioLDMAttack runs
    # the model fully unconditionally -- confirmed empirically, see attacks.py.
    config: tp.Optional[str] = None
    # Caps the strength (t*) AudioLDMAttack samples when `strength=None` is
    # passed in (random.random() * strength_max otherwise) -- see
    # AudioLDMAttack.forward and its class docstring. Backprop through the
    # full DDPM reverse loop (up to num_timesteps, commonly ~1000, steps at
    # strength=1.0) is not memory-tractable during training; 0.08 matches
    # EvalConfig.t_star_grid's own calibrated "hard regime" ceiling (see that
    # field's comment below) rather than the naive full [0, 1] range other
    # attacks use. Only affects random sampling -- an explicit `strength`
    # (e.g. from evaluate.py's t_star_grid) is unaffected.
    strength_max: float = 0.08


@dataclass
class MBDAttackConfig:
    """Wires MBDAttack (attacks.py) -- Meta's MultiBand Diffusion, via the
    `audiocraft` pip package. Unlike every other attack here, no local
    weights file is needed: audiocraft downloads its own pretrained
    checkpoint from HF on first use."""

    # Enable/disable gate only (any non-None value, e.g. "auto", turns the
    # attack on) -- kept named `checkpoint` for consistency with the other
    # *AttackConfig classes and evaluate.py's uniform handling.
    checkpoint: tp.Optional[str] = None
    # EnCodec bitrate in kbps -- MBD only supports 1.5, 3.0, or 6.0. Lower =
    # more information discarded by the codec bottleneck = stronger attack.
    bandwidth: float = 3.0


@dataclass
class AttackWeights:
    # Sampling weight for each attack branch (need not sum to 1; normalized
    # internally). A weight of 0 disables that branch. Identity is the only
    # branch guaranteed to work without extra setup -- see attacks.py for
    # what's needed to enable the others.
    #
    # This project's current config trains against `audioldm`
    # and holds `sgmse` out for evaluation only (see EvalConfig in this file
    # / evaluate.py's held_out_attacks) -- do not also give `sgmse` nonzero
    # weight here unless you deliberately want to give up that held-out
    # generalization probe (see AudioLDMAttack's docstring in attacks.py).
    identity: float = 1.0
    bigvgan: float = 0.0
    dac: float = 0.0
    sgmse: float = 0.0
    audioldm: float = 0.0


@dataclass
class AttackConfig:
    weights: AttackWeights = field(default_factory=AttackWeights)
    bigvgan: BigVGANAttackConfig = field(default_factory=BigVGANAttackConfig)
    dac: DACAttackConfig = field(default_factory=DACAttackConfig)
    sgmse: SGMSEAttackConfig = field(default_factory=SGMSEAttackConfig)
    audioldm: AudioLDMAttackConfig = field(default_factory=AudioLDMAttackConfig)


@dataclass
class MelLossConfig:
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 20.0
    f_max: tp.Optional[float] = None


@dataclass
class OptimConfig:
    lr: float = 5e-5
    betas: tp.Tuple[float, float] = (0.5, 0.9)
    weight_decay: float = 0.0
    max_norm: tp.Optional[float] = 3.0
    max_x_wm_grad_norm: tp.Optional[float] = 1000.0
    # Rescale the generator's gradient to exactly `max_norm` every step,
    # instead of only shrinking it when it exceeds `max_norm`.
    #
    # This exists because plain clipping silently reweights a mixed-attack
    # recipe. The branches produce gradients of wildly different scale: on the
    # 4x A100 run <redacted-run> (recipe=audioldm_mixed, so
    # ~50/50 identity vs audioldm), the median parameter-gradient norm was
    #     identity 1.69   -- under max_norm=3.0, so 26/30 steps passed through
    #                        untouched
    #     audioldm 42.94  -- over it on 33/33 steps, scaled down 14.3x
    # because backprop through the diffusion chain amplifies the gradient ~252x
    # between x_att and x_wm (identity, a no-op, amplifies 1.0x). Clipping
    # therefore let identity steps through at full strength while shrinking
    # every audioldm step, leaving the optimizer on a ~93% identity diet -- a
    # task the pretrained checkpoint has already solved. Measured effect: over
    # 1550 steps the audioldm branch did not move at all (presence_prob
    # 0.0646 -> 0.0648, bit_loss 0.6977 -> 0.6912, both statistically
    # indistinguishable from flat), while identity drifted slightly.
    #
    # Adam does not rescue this. It is scale-invariant to a *uniform* rescale,
    # but its moment estimates are shared across steps, so a branch that is
    # consistently scaled down 14x relative to the other contributes
    # proportionally less to the running direction.
    #
    # Normalizing makes every step contribute an update of comparable size
    # regardless of which branch produced it. It discards gradient magnitude as
    # a signal -- but clipping already discarded it for 100% of audioldm steps,
    # so this trades an accidental, branch-dependent reweighting for a
    # deliberate, uniform one. Off by default: it changes the update rule for
    # every recipe, and single-attack runs do not have the imbalance it fixes.
    normalize_grad: bool = False
    # Below this gradient norm, normalization is skipped rather than amplifying
    # what is essentially numerical noise up to `max_norm`.
    normalize_grad_floor: float = 1e-6


@dataclass
class DataConfig:
    train_dir: str = "???"  # directory of 16kHz wav files, set on the CLI
    # None (default) disables in-training validation. When set together with
    # TrainConfig.eval_every, train.py cycles through this directory and logs
    # an "eval/*"-prefixed no-grad forward pass every eval_every steps -- see
    # train.py:eval_step. Should be audio the generator never trains on (same
    # "held-out" idea as EvalConfig.eval_dir), otherwise this just measures
    # training-set fit again under a different name.
    valid_dir: tp.Optional[str] = None
    segment_duration: float = 1.0  # seconds
    # PER GPU under torchrun (standard DDP convention): with
    # --nproc_per_node=4 this is 16 examples on each of 4 GPUs, i.e. an
    # effective batch of 64. The learning rate is NOT rescaled automatically
    # -- see docs/MULTI_GPU.md for why that is left as a deliberate choice
    # rather than a silent one.
    batch_size: int = 16
    num_workers: int = 4  # per rank: a 4-GPU run spawns 4 x this many loader processes


@dataclass
class TrackingConfig:
    # "wandb" (default: needs `pip install wandb` + login or wandb_mode=offline),
    # "mlflow" (local ./mlruns, no account/network needed -- used for Azure
    # job submissions, which have no wandb access), or "none" (console
    # logging only). See tracking.py.
    backend: str = "wandb"
    project: str = "audioseal-robust"
    run_name: tp.Optional[str] = None
    mlflow_tracking_uri: tp.Optional[str] = None  # None -> local ./mlruns
    wandb_mode: str = "online"  # "online" | "offline" | "disabled"
    log_audio_every: int = 0  # 0 disables; else log one x_wm sample every N steps
    # Per-step CUDA allocator metrics under the "mem/" prefix -- the only way
    # to see whether the diffusion attacks' activation checkpointing is
    # actually buying anything, since the wandb/nvidia-smi system panels
    # report reserved (pool) memory and stay flat either way. Costs a few
    # counter reads per step, no device sync. Ignored off CUDA. See
    # train.py:CudaMemoryProbe.
    log_memory: bool = True


@dataclass
class TrainConfig:
    sample_rate: int = 16_000
    nbits: int = 16
    device: str = "auto"  # "auto" (cuda > mps > cpu), or an explicit "cuda"/"mps"/"cpu"
    epochs: int = 100
    updates_per_epoch: int = 1000
    log_every: int = 50
    # 0 disables eval-loss logging entirely. When set, every eval_every steps
    # runs one no-grad forward pass over a batch from data.valid_dir (must
    # also be set) and logs it under the "eval/" prefix, alongside training
    # loss -- see train.py:eval_step.
    eval_every: int = 0
    checkpoint_dir: str = "./checkpoints/audioseal_robust"
    seed: int = 1234

    # Name of a bundle under config/recipes.yaml's `train:` section (e.g.
    # "audioldm" or "sgmse"), merged in on top of this schema/default.yaml
    # but below explicit CLI overrides -- see load_config and recipes.yaml's
    # own header comment for what these set and why. None (default): no
    # recipe applied, falls back to whatever attack.weights.* default.yaml
    # sets (identity-only, i.e. no attack fine-tuning at all).
    recipe: tp.Optional[str] = None

    # Loss weights -- exposed as config, see module docstring in losses.py
    # for exactly which existing AudioCraft solver losses these would
    # overlap with if that solver were used instead/also.
    lambda_det: float = 1.0
    lambda_perc: float = 1.0
    # Extra weight on bit_loss specifically, within detection_loss_components'
    # (presence_loss + lambda_bit * bit_loss) -- separate from lambda_det,
    # which scales the whole presence+bit sum together and so can't rebalance
    # the two against each other. Default 1.0 == old unweighted behavior
    # (equivalent to no bit_loss field at all). See run_train_10h.sh's
    # lambda_bit=2.0 override for why: presence_loss is the "easy" half of
    # detection_loss (the generator can win it just by embedding *something*
    # detectable, without correctly encoding bits), so it drops fast and
    # plateaus while bit_loss doesn't move (see the comic-snowball-9 run's
    # eval/bit_loss curve) -- weighting bit_loss higher pushes optimization
    # toward the actually-hard sub-problem instead of letting it settle for
    # the cheap presence-only win.
    lambda_bit: float = 1.0

    # Per-example watermark SNR (dB) is sampled uniformly from this range
    # every training step (see train.py:embed_watermark) rather than using
    # whatever amplitude generator.get_watermark() happens to produce
    # unscaled. Range per mentor's plan (2026-08-12 Notion doc); centered
    # near what we actually measured for stock/unscaled AudioSeal on our own
    # VCTK data (mean=30.65dB, std=2.73dB over 30 samples, p225) -- his
    # estimate landed almost exactly on our measured mean, so kept as-is
    # rather than recentered (see his own note: "measure stock AudioSeal on
    # your own data first and recenter if it lands somewhere else").
    watermark_snr_db_min: float = 24.0
    watermark_snr_db_max: float = 36.0

    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    mel_loss: MelLossConfig = field(default_factory=MelLossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


@dataclass
class EvalAttackConfig:
    """Per-attack extra settings for evaluate.py's attacks (checkpoints etc),
    separate from TrainConfig's AttackConfig since eval also runs held-out
    attacks (audioldm, mbd) that training must never see."""

    bigvgan: BigVGANAttackConfig = field(default_factory=BigVGANAttackConfig)
    dac: DACAttackConfig = field(default_factory=DACAttackConfig)
    sgmse: SGMSEAttackConfig = field(default_factory=SGMSEAttackConfig)
    audioldm: AudioLDMAttackConfig = field(default_factory=AudioLDMAttackConfig)
    mbd: MBDAttackConfig = field(default_factory=MBDAttackConfig)


@dataclass
class EvalConfig:
    """Config for evaluate.py -- deliberately separate from TrainConfig
    (rather than nested inside it) so baseline evaluation is runnable on its
    own from day 1, without needing an unrelated training data dir.

    Typical use:
        # day 1, before any fine-tuning:
        python -m audioseal_robust.evaluate eval_dir=/path/to/heldout/wavs label=baseline

        # after fine-tuning, same eval set, pointing at the checkpoint:
        python -m audioseal_robust.evaluate eval_dir=/path/to/heldout/wavs \\
            label=finetuned_epoch10 generator_checkpoint=./checkpoints/audioseal_robust/generator_epoch10.pth

    Both runs land in the same tracking project/experiment (see
    tracking.py), so "detection after attack went from X to Y at comparable
    PESQ" is a same-dashboard comparison, not eyeballed printouts.
    """

    sample_rate: int = 16_000
    nbits: int = 16
    device: str = "auto"
    seed: int = 1234

    # Name of a bundle under config/recipes.yaml's `eval:` section (e.g.
    # "after_audioldm_training" or "after_sgmse_training") -- same merge
    # position/semantics as TrainConfig.recipe (see load_eval_config). None
    # (default): no recipe applied, falls back to default_eval.yaml's plain
    # eval_attacks/held_out_attacks.
    recipe: tp.Optional[str] = None

    # "audioseal_wm_16bits" (vanilla pretrained -- the baseline) or a path to
    # a fine-tuned checkpoint saved by train.py (a .pth with a "model" key).
    generator_checkpoint: str = "audioseal_wm_16bits"
    detector_checkpoint: str = "audioseal_detector_16bits"
    label: str = "baseline"  # run label: "baseline", "finetuned_epoch10", ...

    eval_dir: str = "???"  # held-out wavs, never trained on
    segment_duration: float = 3.0
    # PER GPU under torchrun (standard DDP convention), so a 4-GPU run
    # processes 4 x batch_size examples at a time.
    batch_size: int = 8
    # GLOBAL batch count, split across ranks (see distributed.shard_size) --
    # deliberately NOT per GPU, unlike batch_size: this one controls how much
    # audio a reported number is measured on, so scaling it with the GPU
    # count would silently make 4-GPU results incomparable to the 1-GPU
    # baselines already recorded.
    n_eval_batches: int = 20
    # Fixed watermark SNR (dB) for eval -- a single number, not a range like
    # TrainConfig.watermark_snr_db_{min,max}, since eval wants one
    # reproducible operating point to report, not per-step variety. Without
    # this, evaluate.py embedded the watermark unscaled (raw
    # generator.get_watermark() amplitude), while train.py (see
    # train.py:embed_watermark) rescales it to a target SNR every step --
    # meaning eval was measuring detection at a watermark amplitude training
    # never actually optimized for. 30.0 = the midpoint of
    # TrainConfig's default [24, 36] range, and close to what we measured for
    # stock/unscaled AudioSeal on our own data (mean 30.65dB, see the
    # TrainConfig.watermark_snr_db_{min,max} note) -- so this also keeps the
    # stock-model baseline numbers close to what they were before this field
    # existed, rather than silently shifting them too.
    watermark_snr_db: float = 30.0
    # Batches per *robustness-curve* point (see evaluate.py's t_star_grid
    # loop), separate from n_eval_batches so the curve doesn't cost
    # len(t_star_grid)x the headline number. The curve is a shape ("where does
    # detection fall off along t*"), read off a plot; the headline TPR@FPR is
    # the number that gets quoted and compared across checkpoints, so that one
    # keeps the full n_eval_batches. Cheaper here = noisier per point, which
    # the curve can absorb.
    n_curve_batches: int = 6
    # DataLoader worker processes for the eval set. Anything >0 decodes and
    # resamples the next batch (see data.py:WavDirDataset.__getitem__ --
    # torchaudio.load + resample per item) while the GPU is busy on the
    # current one; at 0 that work is serialized with the forward passes on the
    # main thread, which shows up as a CPU stall between every batch.
    # PER RANK: a 4-GPU run spawns 4 x num_workers loader processes, so this
    # is the knob to turn down first if the box runs out of CPU or shared
    # memory.
    num_workers: int = 4

    # Where the confusion-matrix and robustness-curve PNGs (see plotting.py)
    # get written, one pair per run under f"{label}_*.png".
    output_dir: str = "./eval_outputs"

    # Per-example artifacts (x, x_wm, x_att as .wav + a CSV of per-row
    # metrics: bit_accuracy, presence_pos, presence_neg, attack_sisnr) under
    # f"{output_dir}/{label}_{attack_name}_rows/" -- one dir per attack,
    # every row from the HEADLINE pass only (not the robustness-curve sweep,
    # which would multiply this by len(t_star_grid) for numbers nobody
    # inspects per-example). Off by default: batch_size * n_eval_batches
    # examples * 3 wavs each adds up (e.g. 8*20*3 = 480 files at the config
    # defaults, 8*150*3=3600 for run_full_eval_1h.sh) and most runs only
    # need the aggregate metrics evaluate_attack already returns.
    save_row_artifacts: bool = False

    # Robustness (measured after the attack).
    fpr_target: float = 0.01  # TPR@FPR operating point
    # Robustness curve: detection vs. attack strength t* (diffusion
    # purification starting point -- 0 = no corruption, 1 = full
    # noise-then-regenerate). Only meaningful for attacks that implement a
    # `strength` axis (currently sgmse, audioldm) -- see attacks.py.
    # Fractions of num_timesteps (attack-agnostic -- each strength-aware
    # attack maps this through its own T, see SGMSEAttack/AudioLDMAttack
    # docstrings). Calibrated against the mentor's curriculum table (linear
    # schedule, T=1000): actual timesteps t*={0.3, 2, 6, 15, 40} correspond
    # to noise floors lambda_dB={50, 36, 30, 24, 18} -- 50dB is "barely an
    # attack" (warmup) and 18dB/t*=40 is already their "hard regime". The
    # OLD default here went up to strength=1.0 (t*=1000, the full schedule)
    # -- ~25x past their hard regime, and empirically destroys the audio
    # rather than measuring a realistic robustness curve. Recentered to
    # their actual range, plus one point (0.08 = t*=80) past their hard
    # regime to see where detection fully saturates.
    t_star_grid: tp.List[float] = field(
        default_factory=lambda: [0.0, 0.0003, 0.002, 0.006, 0.015, 0.04, 0.08]
    )
    # The single t* the HEADLINE robustness number is measured at, for the
    # strength-aware attacks (sgmse, audioldm -- the rest ignore strength
    # entirely). 0.04 = t*=40 out of T=1000 = the top of the mentor's
    # curriculum table, i.e. their "hard regime" (see t_star_grid above).
    #
    # Passing no strength at all (what evaluate.py used to do) is NOT a
    # neutral default -- it leaves strength=None, which makes each attack
    # sample its own t* ~ U[0, 1] per forward call (see
    # SGMSEAttack/AudioLDMAttack.forward). Two problems with that:
    #   1. Cost. AudioLDMAttack's reverse loop runs int(1000 * strength)
    #      sequential UNet steps, so U[0,1] averages ~500 steps/batch --
    #      ~12x this setting's 40, and still ~6x past t*=80 (t_star_grid's
    #      top point, itself already beyond the hard regime). We were paying
    #      that to measure a regime we don't even claim.
    #   2. Correctness. evaluate_attack runs positives and negatives through
    #      two separate forward calls, so they'd each draw a DIFFERENT random
    #      t*. The FPR threshold would then come from a differently-attacked
    #      population than the TPR measured against it, which is not a
    #      meaningful operating point.
    headline_strength: float = 0.04

    # Attacks the generator was (or will be) trained against.
    eval_attacks: tp.List[str] = field(default_factory=lambda: ["identity", "bigvgan", "dac", "audioldm"])
    # Held out on purpose: NEVER given nonzero weight in AttackConfig.weights
    # during training. This is the generalization probe -- "did robustness
    # transfer to an attack the generator never saw, or did it just
    # memorize the training attacks." `sgmse` is held out here (rather than
    # trained against, as in the original design) -- this project variant
    # trains on `audioldm` instead, so `sgmse` is the
    # generalization probe: does robustness against latent-diffusion
    # resynthesis transfer to SGMSE's structurally different OU-VE SDE
    # attack. `mbd` remains held out either way (never wired into
    # AttackConfig.weights at all).
    held_out_attacks: tp.List[str] = field(default_factory=lambda: ["sgmse", "mbd"])

    # Perceptual (watermarked vs. original, no attack).
    compute_pesq: bool = True
    compute_sisnr: bool = True
    compute_visqol: bool = False  # needs a separate ViSQOL binary build, see metrics.py

    attack: EvalAttackConfig = field(default_factory=EvalAttackConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


_DEFAULT_YAML = Path(__file__).parent / "config" / "default.yaml"
_DEFAULT_EVAL_YAML = Path(__file__).parent / "config" / "default_eval.yaml"
_RECIPES_YAML = Path(__file__).parent / "config" / "recipes.yaml"


def _load_recipe(section: str, name: str) -> tp.Any:
    """Look up recipes.yaml's `<section>.<name>` bundle (section is "train"
    or "eval") -- see that file's header comment and TrainConfig.recipe /
    EvalConfig.recipe for what this is and why. Raises with the available
    names on a typo rather than merging in nothing silently."""
    if not _RECIPES_YAML.exists():
        raise FileNotFoundError(f"recipe={name!r} requested but {_RECIPES_YAML} does not exist")
    recipes = OmegaConf.load(_RECIPES_YAML)
    section_cfg = recipes.get(section)
    if section_cfg is None or name not in section_cfg:
        available = sorted(section_cfg.keys()) if section_cfg is not None else []
        raise ValueError(f"Unknown {section} recipe {name!r}, expected one of {available}")
    return section_cfg[name]


def load_config(cli_args: tp.Optional[tp.List[str]] = None) -> TrainConfig:
    """Build the effective config: structured schema <- config/default.yaml
    <- recipe (if `recipe=<name>` is set, from either default.yaml or the
    CLI -- see TrainConfig.recipe / config/recipes.yaml) <- CLI overrides
    (`key=value`, dotlist form, e.g. `optim.lr=1e-4`). CLI overrides are
    merged in last specifically so they can still override anything a
    recipe set, without having to fork the recipe."""
    schema = OmegaConf.structured(TrainConfig)
    file_cfg = OmegaConf.load(_DEFAULT_YAML) if _DEFAULT_YAML.exists() else OmegaConf.create({})
    cli_cfg = OmegaConf.from_cli(cli_args)
    recipe_name = cli_cfg.get("recipe", None) or file_cfg.get("recipe", None)
    recipe_cfg = _load_recipe("train", recipe_name) if recipe_name else OmegaConf.create({})
    cfg = OmegaConf.merge(schema, file_cfg, recipe_cfg, cli_cfg)
    return tp.cast(TrainConfig, cfg)


def load_eval_config(cli_args: tp.Optional[tp.List[str]] = None) -> EvalConfig:
    """Same override mechanism as load_config (including the recipe step --
    see that function and EvalConfig.recipe / config/recipes.yaml), just
    against config/default_eval.yaml."""
    schema = OmegaConf.structured(EvalConfig)
    file_cfg = OmegaConf.load(_DEFAULT_EVAL_YAML) if _DEFAULT_EVAL_YAML.exists() else OmegaConf.create({})
    cli_cfg = OmegaConf.from_cli(cli_args)
    recipe_name = cli_cfg.get("recipe", None) or file_cfg.get("recipe", None)
    recipe_cfg = _load_recipe("eval", recipe_name) if recipe_name else OmegaConf.create({})
    cfg = OmegaConf.merge(schema, file_cfg, recipe_cfg, cli_cfg)
    return tp.cast(EvalConfig, cfg)