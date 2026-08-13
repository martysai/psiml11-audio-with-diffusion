# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Experiment tracking, pluggable between MLflow and Weights & Biases.

Default backend is Weights & Biases (`tracking.backend=wandb`) for hosted
dashboards and easy cross-device metric comparison; it needs `pip install
wandb` and either a logged-in account or `tracking.wandb_mode=offline` in the
caller's own environment -- neither is set up by this module.

MLflow is fully supported as an alternative (`tracking.backend=mlflow`),
using its local backend store (a `./mlflow.db` SQLite file with MLflow 3.x,
or a `./mlruns` directory with older MLflow -- either way, in the current
working directory unless `mlflow_tracking_uri` is set): no account, no
network call, no API key -- it works out of the box on a training box with
no internet egress, which is why Azure job submissions (no wandb access
there) pin this backend explicitly. Run `mlflow ui` in that directory to
browse it.

If the selected backend's package isn't installed, we fall back to a
console-only tracker rather than crashing the training run.
"""

import logging
import typing as tp
from abc import ABC, abstractmethod

import torch

logger = logging.getLogger(__name__)


def _flatten(d: tp.Mapping, parent_key: str = "", sep: str = ".") -> tp.Dict[str, tp.Any]:
    items: tp.Dict[str, tp.Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, tp.Mapping):
            items.update(_flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


class ExperimentTracker(ABC):
    @abstractmethod
    def log(self, metrics: tp.Dict[str, float], step: int) -> None:
        """Log scalar metrics at a given step."""

    def log_audio(self, key: str, wav: torch.Tensor, sample_rate: int, step: int) -> None:
        """Optional: log an audio sample. No-op unless a backend overrides it."""

    def log_figure(self, path: tp.Any) -> None:
        """Optional: log a saved plot (PNG path). No-op unless a backend
        overrides it -- the caller (evaluate.py) always has the file on disk
        regardless, this is just for it to also show up on the dashboard."""

    def finish(self) -> None:
        """Optional: flush/close the run."""


class ConsoleTracker(ExperimentTracker):
    """No external dependency, writes nothing to disk. Used when no tracking
    backend is configured, or as a fallback if the configured one isn't
    installed."""

    def log(self, metrics: tp.Dict[str, float], step: int) -> None:
        logger.info("step=%d %s", step, metrics)


class NullTracker(ExperimentTracker):
    """Discards everything. Used on non-zero ranks under DDP: a tracker is a
    process-external side effect, so letting all 4 ranks build a real one
    creates 4 runs per training run -- three of them holding metrics from a
    single shard. Unlike ConsoleTracker this stays silent, since those ranks
    are sharing rank 0's terminal."""

    def log(self, metrics: tp.Dict[str, float], step: int) -> None:
        pass


class MLflowTracker(ExperimentTracker):
    def __init__(
        self,
        experiment_name: str,
        run_name: tp.Optional[str],
        tracking_uri: tp.Optional[str],
        config: tp.Dict[str, tp.Any],
    ):
        import mlflow

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)
        flat_config = {k: str(v) for k, v in _flatten(config).items()}
        mlflow.log_params(flat_config)

    def log(self, metrics: tp.Dict[str, float], step: int) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_audio(self, key: str, wav: torch.Tensor, sample_rate: int, step: int) -> None:
        import tempfile
        from pathlib import Path

        import torchaudio

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{key}_step{step}.wav"
            torchaudio.save(str(path), wav.detach().cpu().reshape(1, -1), sample_rate)
            self._mlflow.log_artifact(str(path), artifact_path=f"audio/{key}")

    def log_figure(self, path: tp.Any) -> None:
        self._mlflow.log_artifact(str(path), artifact_path="plots")

    def finish(self) -> None:
        self._mlflow.end_run()


class WandbTracker(ExperimentTracker):
    def __init__(
        self,
        project: str,
        run_name: tp.Optional[str],
        mode: str,
        config: tp.Dict[str, tp.Any],
    ):
        import wandb

        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name, mode=mode, config=_flatten(config))

    def log(self, metrics: tp.Dict[str, float], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def log_audio(self, key: str, wav: torch.Tensor, sample_rate: int, step: int) -> None:
        self._wandb.log(
            {key: self._wandb.Audio(wav.detach().cpu().reshape(-1).numpy(), sample_rate=sample_rate)},
            step=step,
        )

    def log_figure(self, path: tp.Any) -> None:
        from pathlib import Path

        self._wandb.log({Path(path).stem: self._wandb.Image(str(path))})

    def finish(self) -> None:
        self._wandb.finish()


def build_tracker(
    backend: str,
    project: str,
    run_name: tp.Optional[str] = None,
    config: tp.Optional[tp.Dict[str, tp.Any]] = None,
    mlflow_tracking_uri: tp.Optional[str] = None,
    wandb_mode: str = "online",
) -> ExperimentTracker:
    config = config or {}

    if backend == "none":
        return ConsoleTracker()

    if backend == "mlflow":
        try:
            return MLflowTracker(project, run_name, mlflow_tracking_uri, config)
        except ImportError:
            logger.warning("mlflow not installed (`pip install mlflow`), falling back to console logging")
            return ConsoleTracker()

    if backend == "wandb":
        try:
            return WandbTracker(project, run_name, wandb_mode, config)
        except ImportError:
            logger.warning("wandb not installed (`pip install wandb`), falling back to console logging")
            return ConsoleTracker()

    raise ValueError(f"Unknown tracking backend: {backend!r} (expected 'mlflow', 'wandb', or 'none')")
