# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for MLflowTracker inside an AzureML job.

AzureML creates an MLflow run for every job and exports its ID as
MLFLOW_RUN_ID. `mlflow.start_run()` resumes that run, which is what makes
metrics show up in the job's Metrics tab.

The tracker used to call `mlflow.set_experiment(<our name>)` first. That sets
MLflow's module-level `_active_experiment_id`, and `start_run()` then refuses
when it does not match the experiment owning the environment's run:

    Cannot start run with ID <job> because active experiment ID does not match
    environment run ID

AzureML owns that experiment, so ours never matched: every AzureML run lost its
tracker at construction, `_tracker_or_console` degraded it to console logging,
and the Metrics tab stayed empty for the entire job.

These tests pin down the attached path (the run is joined, not recreated,
renamed, moved, or ended) and that the standalone path still owns its run.
"""

import os
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

pytest.importorskip("torch")
mlflow = pytest.importorskip("mlflow")

from audioseal_robust.tracking import MLflowTracker  # noqa: E402


@pytest.fixture()
def tracking_uri(tmp_path):
    """A private sqlite-backed MLflow store (the file store is deprecated)."""
    return "sqlite:///" + str(tmp_path / "mlflow.db").replace(os.sep, "/")


@pytest.fixture()
def clean_mlflow_state(monkeypatch):
    """MLflow keeps active run/experiment in module-level state, so tests
    would otherwise leak into each other."""
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    while mlflow.active_run() is not None:
        mlflow.end_run()
    yield
    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.tracking.fluent._active_experiment_id = None


@pytest.fixture()
def azureml_run(tracking_uri, clean_mlflow_state, monkeypatch):
    """Stand in for AzureML: it owns both the experiment and the run, and
    exports the run id the same way."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri)
    experiment_id = client.create_experiment(f"azureml-owned-{uuid.uuid4().hex[:8]}")
    run = client.create_run(experiment_id, run_name="train-audioldm-cluster-0814")
    monkeypatch.setenv("MLFLOW_RUN_ID", run.info.run_id)
    return client, experiment_id, run


# The job YAML's experiment_name is AzureML's business; ours is deliberately
# different, which is precisely what used to trigger the failure.
OUR_EXPERIMENT = "audioseal-robust-diffusion-swap"


def test_attaches_to_the_ambient_azureml_run(azureml_run, tracking_uri):
    client, experiment_id, run = azureml_run

    tracker = MLflowTracker(OUR_EXPERIMENT, "some-other-name", tracking_uri, {"optim": {"lr": 1e-5}})
    tracker.log({"loss": 3.0, "bit_loss": 0.693}, step=0)
    tracker.finish()

    after = client.get_run(run.info.run_id)
    assert after.info.run_id == run.info.run_id, "should join AzureML's run, not create one"
    assert after.info.experiment_id == experiment_id, "must not move the run to our experiment"
    assert after.data.metrics["loss"] == pytest.approx(3.0)


def test_does_not_rename_the_ambient_run(azureml_run, tracking_uri):
    """start_run() forwards run_name to update_run() when resuming, so passing
    ours would rename the AzureML job's run in the Studio UI."""
    client, _, run = azureml_run

    MLflowTracker(OUR_EXPERIMENT, "a-different-run-name", tracking_uri, {}).finish()

    after = client.get_run(run.info.run_id)
    assert after.data.tags.get("mlflow.runName") == "train-audioldm-cluster-0814"


def test_finish_does_not_end_a_run_it_does_not_own(azureml_run, tracking_uri):
    """aml_run.sh drives training [1/2] and evaluation [2/2] as two processes
    through one AzureML run. Ending it after training would mark the job
    finished while the evaluation -- the actual result -- was still running."""
    client, _, run = azureml_run

    MLflowTracker(OUR_EXPERIMENT, None, tracking_uri, {}).finish()

    assert client.get_run(run.info.run_id).info.status == "RUNNING"


def test_second_stage_survives_conflicting_params(azureml_run, tracking_uri):
    """Both stages log their own config to the same run. MLflow treats params
    as write-once, and TrainConfig/EvalConfig disagree on some shared keys --
    that must not cost the second stage its metrics."""
    client, _, run = azureml_run

    MLflowTracker(OUR_EXPERIMENT, None, tracking_uri, {"batch_size": 4, "seed": 1234}).finish()
    # Same keys, different values, as the evaluation stage would send.
    second = MLflowTracker(OUR_EXPERIMENT, None, tracking_uri, {"batch_size": 8, "seed": 1234})
    second.log({"eval_loss": 1.5}, step=0)
    second.finish()

    after = client.get_run(run.info.run_id)
    assert after.data.metrics["eval_loss"] == pytest.approx(1.5), "metrics must survive param conflicts"
    assert after.data.params["batch_size"] == "4", "first writer wins; MLflow params are write-once"
    assert after.data.params["seed"] == "1234", "non-conflicting params still logged"


def test_standalone_run_is_created_and_ended(tracking_uri, clean_mlflow_state):
    """Without MLFLOW_RUN_ID (a laptop, or CI) the tracker still owns its run:
    it creates one in our experiment and ends it on finish()."""
    mlflow.set_tracking_uri(tracking_uri)

    tracker = MLflowTracker(OUR_EXPERIMENT, "local-run", tracking_uri, {"optim": {"lr": 5e-5}})
    run_id = tracker._run_id
    tracker.log({"loss": 2.0}, step=0)
    tracker.finish()

    client = mlflow.tracking.MlflowClient(tracking_uri)
    after = client.get_run(run_id)
    assert after.info.status == "FINISHED", "a run we started must be ended"
    assert after.data.tags.get("mlflow.runName") == "local-run"
    assert client.get_experiment(after.info.experiment_id).name == OUR_EXPERIMENT


def test_ambient_run_survives_process_exit(azureml_run, tracking_uri):
    """The decisive one, and the reason this tracker uses MlflowClient instead
    of MLflow's fluent API.

    Fluent registers `atexit.register(_safe_end_run)`, which terminates
    whatever run is on its active-run stack when the interpreter shuts down --
    so merely declining to call `end_run()` does NOT keep AzureML's run alive.
    Verified: pushing the ambient run onto the fluent stack leaves it FINISHED
    after the process exits, which in the real job means training [1/2] closes
    the run before evaluation [2/2] reports the result.

    Only observable across a real process boundary, hence the subprocess.
    """
    client, _, run = azureml_run

    child = textwrap.dedent(
        f"""
        import os, sys
        os.environ["MLFLOW_RUN_ID"] = {run.info.run_id!r}
        sys.path.insert(0, {str(Path(__file__).resolve().parent.parent / "src")!r})
        from audioseal_robust.tracking import MLflowTracker
        t = MLflowTracker("our-experiment", "our-name", {tracking_uri!r}, {{"lr": 1e-5}})
        t.log({{"loss": 1.0}}, step=0)
        t.finish()
        """
    )
    result = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True)
    assert result.returncode == 0, f"child failed: {result.stderr[-800:]}"

    after = client.get_run(run.info.run_id)
    assert after.data.metrics["loss"] == pytest.approx(1.0)
    assert after.info.status == "RUNNING", (
        "the ambient run was terminated when the process exited -- the evaluation stage "
        "would report into a FINISHED run"
    )
