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

Inside an AzureML job that same backend behaves differently, and
deliberately: AzureML injects an `azureml://` tracking URI and has already
created a run, whose id arrives in MLFLOW_RUN_ID. MLflowTracker attaches to
that run instead of creating its own, which is what makes metrics appear in
the job's own Metrics tab rather than in a store on a node that is deleted
when the job ends. See MLflowTracker's docstring for why that path goes
through MlflowClient rather than MLflow's fluent API.

If the selected backend cannot be brought up -- not installed, misconfigured,
unreachable -- we fall back to a console-only tracker rather than crashing the
training run. See _tracker_or_console() for why that is deliberately broad.
"""

import logging
import os
import time
import typing as tp
from abc import ABC, abstractmethod

import torch

logger = logging.getLogger(__name__)


def _ambient_mlflow_run_id() -> tp.Optional[str]:
    """The run MLflow will resume from the environment, if any.

    AzureML starts an MLflow run for every job and exports its ID as
    MLFLOW_RUN_ID. MLflow's own `start_run()` picks that up and *resumes* it
    rather than creating a new run, which is exactly what we want -- metrics
    logged to it land in the job's Metrics tab in the Studio UI.

    This is read rather than acted on directly because the value is consumed
    (and deleted from os.environ) by `mlflow.start_run()` itself; see
    MLflowTracker.__init__ for what it changes.
    """
    return os.environ.get("MLFLOW_RUN_ID") or None


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
    """Logs through `MlflowClient` against an explicit run id, rather than
    through MLflow's fluent `start_run()`/`log_metrics()` API.

    That choice is what makes the AzureML path work. MLflow's fluent API keeps
    a process-global *active run stack*, and pushing AzureML's run onto it
    causes three separate problems:

      * `start_run()` refuses outright once `set_experiment()` has been called,
        because the experiment we ask for is not the one owning the
        environment's run --
            Cannot start run with ID <job> because active experiment ID does
            not match environment run ID
        AzureML owns that experiment (it comes from the job YAML), so ours
        never matched: every AzureML run lost its tracker at construction,
        `_tracker_or_console` degraded it to console logging, and the job's
        Metrics tab stayed empty.
      * fluent `start_run(run_name=...)` forwards the name to `update_run()`
        when resuming, silently *renaming* the AzureML job's run.
      * fluent registers `atexit.register(_safe_end_run)`, so an active run is
        terminated when the process exits. aml_run.sh drives training [1/2] and
        evaluation [2/2] as two processes through one AzureML run, so that
        would mark the job's run FINISHED as soon as training ended -- before
        the evaluation that produces the actual result had reported anything.
        Simply not calling `end_run()` ourselves is not enough to avoid this.

    Going through the client sidesteps all three: no global state is touched,
    the run is addressed by id, and its lifecycle stays with whoever created
    it.
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: tp.Optional[str],
        tracking_uri: tp.Optional[str],
        config: tp.Dict[str, tp.Any],
    ):
        import mlflow
        from mlflow.tracking import MlflowClient

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri) if tracking_uri else MlflowClient()

        # Inside an AzureML job the run already exists and its id arrives in
        # MLFLOW_RUN_ID; attaching to it is what puts metrics in that job's
        # Metrics tab. Anywhere else (a laptop, CI) we create and own one.
        ambient_run_id = _ambient_mlflow_run_id()
        self._owns_run = ambient_run_id is None
        if self._owns_run:
            experiment = self._client.get_experiment_by_name(experiment_name)
            experiment_id = (
                experiment.experiment_id
                if experiment is not None
                else self._client.create_experiment(experiment_name)
            )
            self._run_id = self._client.create_run(experiment_id, run_name=run_name).info.run_id
        else:
            self._run_id = ambient_run_id
            logger.info(
                "attached to the ambient MLflow run %s; AzureML owns its name, experiment "
                "and lifecycle, and metrics will appear in this job's Metrics tab",
                ambient_run_id,
            )
        self._log_params({k: str(v) for k, v in _flatten(config).items()})

    def _log_params(self, params: tp.Dict[str, str]) -> None:
        """Log config as params, never at the cost of the run's metrics.

        Two things make this fallible in a way a bare log_params is not, and
        both only bite on the attached path:

          * The two stages log their own config to the SAME run, so the second
            re-logs overlapping keys. MLflow treats params as write-once and
            rejects a *changed* value (an identical re-log is fine), and
            TrainConfig/EvalConfig genuinely disagree on some shared keys.
          * This runs in __init__, so an exception propagates to
            _tracker_or_console, which drops the whole tracker to console --
            costing exactly the metrics we came here to record. Params are
            secondary; losing them must not lose the metrics too.

        So conflicting keys are dropped (reported once, not per key) and any
        remaining failure is downgraded to a warning.
        """
        try:
            if not self._owns_run and params:
                existing = self._client.get_run(self._run_id).data.params
                conflicting = sorted(k for k, v in params.items() if existing.get(k, v) != v)
                if conflicting:
                    shown = ", ".join(conflicting[:5]) + ("..." if len(conflicting) > 5 else "")
                    logger.info(
                        "not re-logging %d param(s) an earlier stage already set to a different "
                        "value on this run (%s); MLflow params are write-once",
                        len(conflicting), shown,
                    )
                    params = {k: v for k, v in params.items() if k not in set(conflicting)}
            for key, value in params.items():
                self._client.log_param(self._run_id, key, value)
        except Exception:
            logger.warning(
                "could not log config params to MLflow -- continuing, metrics are unaffected",
                exc_info=True,
            )

    def log(self, metrics: tp.Dict[str, float], step: int) -> None:
        # One batched call rather than one request per key: this fires every
        # log_every steps for hours, with ~10 metrics a time.
        from mlflow.entities import Metric

        timestamp = int(time.time() * 1000)
        self._client.log_batch(
            self._run_id,
            metrics=[Metric(key, float(value), timestamp, step) for key, value in metrics.items()],
        )

    def log_audio(self, key: str, wav: torch.Tensor, sample_rate: int, step: int) -> None:
        import tempfile
        from pathlib import Path

        import torchaudio

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{key}_step{step}.wav"
            torchaudio.save(str(path), wav.detach().cpu().reshape(1, -1), sample_rate)
            self._client.log_artifact(self._run_id, str(path), artifact_path=f"audio/{key}")

    def log_figure(self, path: tp.Any) -> None:
        self._client.log_artifact(self._run_id, str(path), artifact_path="plots")

    def finish(self) -> None:
        # Only terminate a run we created. On the attached path the run belongs
        # to AzureML and the evaluation stage still has to report into it.
        if self._owns_run:
            self._client.set_terminated(self._run_id)


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


class _ResilientTracker(ExperimentTracker):
    """Delegates to a real tracker, surviving failures that happen *after*
    construction.

    _tracker_or_console covers the constructor; this covers the calls made
    during the run. log() fires every step for hours, so an outage in the
    tracking backend partway through an 11-hour job would otherwise discard
    everything computed up to that point -- the run dies at hour 6 holding a
    complete set of checkpoints and no evaluation.

    On the first failure it degrades to ConsoleTracker permanently rather than
    just swallowing the error, so metrics keep being emitted to the job log
    instead of silently stopping. That also bounds the noise: one stack trace,
    not one per step.
    """

    def __init__(self, inner: ExperimentTracker):
        self._inner = inner

    def _guard(self, method: str, *args: tp.Any) -> None:
        try:
            getattr(self._inner, method)(*args)
        except Exception:
            logger.warning(
                "%s.%s() failed -- degrading to console logging for the rest of the run "
                "so metrics are not lost. Training is unaffected.",
                type(self._inner).__name__,
                method,
                exc_info=True,
            )
            self._inner = ConsoleTracker()
            getattr(self._inner, method)(*args)

    def log(self, metrics: tp.Dict[str, float], step: int) -> None:
        self._guard("log", metrics, step)

    def log_audio(self, key: str, wav: torch.Tensor, sample_rate: int, step: int) -> None:
        self._guard("log_audio", key, wav, sample_rate, step)

    def log_figure(self, path: tp.Any) -> None:
        self._guard("log_figure", path)

    def finish(self) -> None:
        self._guard("finish")


def _tracker_or_console(
    backend: str, install_hint: str, factory: tp.Callable[[], ExperimentTracker]
) -> ExperimentTracker:
    """Build a tracker, degrading to console logging if it cannot be created.

    The except clause is deliberately broad. A tracker records what a run did;
    it is not part of what a run *does*. Letting its constructor propagate means
    a logging problem takes down the training job it was only supposed to be
    observing -- which is exactly what happened here: a missing mlflow plugin
    (azureml-mlflow, which registers the `azureml://` URI scheme that AzureML
    injects into every job) killed two 4x A100 runs at step 0, before a single
    batch was processed, with 8 GPU-hours of queueing in front of each.

    Failing here costs the run its dashboard. It must not also cost the run its
    GPUs -- especially since nothing is actually lost: ConsoleTracker puts every
    metric in the job log, and evaluate.py writes the plots and metric JSON that
    are the real deliverable to eval_outputs/ regardless of the tracker.

    Unknown-backend stays fatal, in build_tracker: that is a typo in the config
    rather than a sick dependency, it is free to detect, and silently logging to
    a console the user did not ask for would just hide it.
    """
    try:
        return _ResilientTracker(factory())
    except ImportError:
        logger.warning(
            "%s is not installed (%s) -- falling back to console logging", backend, install_hint
        )
    except Exception:
        logger.warning(
            "could not initialise the %s tracker -- falling back to console logging. "
            "Training continues; metrics go to stdout and eval_outputs/ only.",
            backend,
            exc_info=True,
        )
    return ConsoleTracker()


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
        return _tracker_or_console(
            "mlflow",
            "pip install mlflow",
            lambda: MLflowTracker(project, run_name, mlflow_tracking_uri, config),
        )

    if backend == "wandb":
        return _tracker_or_console(
            "wandb",
            "pip install wandb",
            lambda: WandbTracker(project, run_name, wandb_mode, config),
        )

    raise ValueError(f"Unknown tracking backend: {backend!r} (expected 'mlflow', 'wandb', or 'none')")
