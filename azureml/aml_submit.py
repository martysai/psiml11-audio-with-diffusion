#!/usr/bin/env python3
"""Config-driven AzureML submitter for the diffusion-swap experiment.

One YAML per experiment direction under azureml/configs/, one submit call:

    python azureml/aml_submit.py --config azureml/configs/sgmse.yaml
    python azureml/aml_submit.py --config azureml/configs/sgmse.yaml --dry-run
    python azureml/aml_submit.py --config azureml/configs/sgmse.yaml --stream

Why this exists alongside azureml/jobs/*.yaml, which `az ml job create` can
already submit on its own: the static YAMLs are the source of truth for *how*
a job is shaped, but sweeping the interesting axis here (epochs,
segment_duration, batch_size, attack.sgmse.num_steps, and eventually
attack.*.strength_max) means re-emitting the same job with different train.py
overrides. Doing that from a config dict beats hand-editing YAML per run, and
keeps a submitted run traceable back to the exact config file that produced
it -- see `--tag-config`, on by default.

Requires the SDK on the submitting machine:

    pip install azure-ai-ml azure-identity pyyaml

If you'd rather not install anything, `az ml job create -f
azureml/jobs/train_sgmse.yaml` does the same thing with the CLI's own
defaults; see azureml/README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


def die(msg: str) -> NoReturn:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load_config(path: Path) -> Dict[str, Any]:
    import yaml

    if not path.is_file():
        die(f"config not found: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        die(f"config {path} did not parse to a mapping")

    for required in ("direction", "compute", "environment", "inputs"):
        if required not in cfg:
            die(f"config {path} is missing required key: {required}")
    if cfg["direction"] not in ("sgmse", "audioldm", "eval"):
        die(f"direction must be 'sgmse', 'audioldm' or 'eval', got {cfg['direction']!r}")
    if cfg["direction"] == "eval" and cfg.get("generator") and not cfg["inputs"].get("generators"):
        die(f"config {path} sets generator={cfg['generator']!r} but no inputs.generators asset to find it in")
    return cfg


def build_overrides(overrides: Optional[Dict[str, Any]]) -> List[str]:
    """dict -> the `key=value` argv train.py's OmegaConf CLI parser expects.

    Values go through str() rather than repr()/json: OmegaConf parses the
    right-hand side itself, and `True`/`3`/`0.08` all round-trip correctly,
    while a quoted "3" would land as a string and fail the structured
    schema's int field.
    """
    if not overrides:
        return []
    out = []
    for key, value in overrides.items():
        if isinstance(value, bool):
            value = str(value).lower()
        out.append(f"{key}={value}")
    return out


def _az(*args: str) -> str:
    """Run an `az` command, returning stripped stdout ('' if az is unavailable)."""
    exe = shutil.which("az") or shutil.which("az.cmd")
    if not exe:
        return ""
    try:
        out = subprocess.run(
            [exe, *args], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _az_subscription_id() -> str:
    return _az("account", "show", "--query", "id", "-o", "tsv")


def _az_default(name: str) -> str:
    raw = _az("configure", "--list-defaults", "-o", "json")
    if not raw:
        return ""
    try:
        for entry in json.loads(raw):
            if entry.get("name") == name:
                return str(entry.get("value") or "")
    except (ValueError, AttributeError, TypeError):
        pass
    return ""


def make_client(cfg: Dict[str, Any]):
    from azure.ai.ml import MLClient
    from azure.identity import AzureCliCredential, DefaultAzureCredential

    # AzureCliCredential first: this repo's workflow is `az login` on a laptop,
    # and DefaultAzureCredential's earlier links (env vars, managed identity)
    # can silently pick a different, wrong identity on a corp machine.
    try:
        credential = AzureCliCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception:
        credential = DefaultAzureCredential()

    ws = dict(cfg.get("workspace") or {})
    # Environment, then the Azure CLI, then config.json. This repo is public, so
    # the shipped configs deliberately carry no subscription id and we resolve
    # it at submit time instead of committing it.
    for key, env in (
        ("subscription_id", "AZUREML_SUBSCRIPTION_ID"),
        ("resource_group_name", "AZUREML_RESOURCE_GROUP"),
        ("workspace_name", "AZUREML_WORKSPACE_NAME"),
    ):
        if not ws.get(key) and os.environ.get(env):
            ws[key] = os.environ[env]
    for key, source in (
        ("subscription_id", _az_subscription_id),
        ("resource_group_name", lambda: _az_default("group")),
        ("workspace_name", lambda: _az_default("workspace")),
    ):
        if not ws.get(key):
            ws[key] = source()

    if all(ws.get(k) for k in ("subscription_id", "resource_group_name", "workspace_name")):
        return MLClient(
            credential,
            subscription_id=ws["subscription_id"],
            resource_group_name=ws["resource_group_name"],
            workspace_name=ws["workspace_name"],
        )
    # Last resort: a config.json somewhere up the tree.
    try:
        return MLClient.from_config(credential)
    except Exception as exc:
        missing = [k for k in ("subscription_id", "resource_group_name", "workspace_name") if not ws.get(k)]
        die(
            f"could not resolve the workspace (missing: {', '.join(missing)}); "
            f"MLClient.from_config() also failed ({exc}). Either fill in the "
            "`workspace:` block, export AZUREML_SUBSCRIPTION_ID / "
            "AZUREML_RESOURCE_GROUP / AZUREML_WORKSPACE_NAME, or run: "
            "az login && az configure --defaults group=<rg> workspace=<ws>"
        )


def build_command(cfg: Dict[str, Any]) -> str:
    """The `bash azureml/aml_run.sh ...` line the job runs.

    Deliberately free of any azure.ai.ml import so `--dry-run` keeps working
    on a machine with nothing installed -- which is most of the value of
    having a dry run at all.
    """
    direction = cfg["direction"]
    inputs_cfg = cfg["inputs"]

    parts = [
        "bash azureml/aml_run.sh",
        direction,
        "--librispeech ${{inputs.librispeech}}",
    ]
    # The 10.9 GB diffusion-backbone asset is required to train against either
    # attack, but an eval-only job running identity/hopskipjump loads neither
    # backbone -- so mounting it there would just add startup latency.
    if inputs_cfg.get("checkpoints"):
        parts.append("--checkpoints ${{inputs.checkpoints}}")
    elif direction != "eval":
        die(f"direction={direction} trains a generator and needs inputs.checkpoints")

    if inputs_cfg.get("generators"):
        parts.append("--generators ${{inputs.generators}}")

    parts.append("--artifacts ${{outputs.artifacts}}")
    # Omitted = the stock AudioSeal card, i.e. the baseline arm.
    if cfg.get("generator"):
        parts.append(f"--generator {cfg['generator']}")
    if cfg.get("label"):
        parts.append(f"--label {cfg['label']}")
    parts += ["--", *build_overrides(cfg.get("overrides"))]
    return " ".join(parts)


def build_job(cfg: Dict[str, Any]):
    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.constants import AssetTypes, InputOutputModes

    direction = cfg["direction"]
    inputs_cfg = cfg["inputs"]

    job_inputs = {
        "librispeech": Input(
            type=AssetTypes.URI_FOLDER,
            path=inputs_cfg["librispeech"],
            # DOWNLOAD, not RO_MOUNT: data.py:36 walks the tree once per
            # audio extension and train.py builds one dataset per split, so
            # a mount pays ~6 recursive blob listings before step 1 and blob
            # latency on every __getitem__ after. Keep in sync with
            # azureml/jobs/*.yaml.
            mode=InputOutputModes.DOWNLOAD,
        ),
    }
    for name in ("checkpoints", "generators"):
        if inputs_cfg.get(name):
            job_inputs[name] = Input(
                type=AssetTypes.CUSTOM_MODEL,
                path=inputs_cfg[name],
                mode=InputOutputModes.RO_MOUNT,
            )
    job = command(
        code=str(REPO_ROOT),
        command=build_command(cfg),
        environment=cfg["environment"],
        compute=cfg["compute"],
        experiment_name=cfg.get("experiment_name", "audioseal-robust-diffusion-swap"),
        display_name=cfg.get("display_name", f"swap-{direction}"),
        description=cfg.get("description"),
        inputs=job_inputs,
        outputs={
            "artifacts": Output(type=AssetTypes.URI_FOLDER, mode=InputOutputModes.RW_MOUNT),
        },
        environment_variables={"PYTHONUNBUFFERED": "1", **(cfg.get("environment_variables") or {})},
        tags={"direction": direction, **(cfg.get("tags") or {})},
    )

    timeout_minutes = cfg.get("timeout_minutes")
    if timeout_minutes:
        job.set_limits(timeout=int(timeout_minutes) * 60)
    return job


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="e.g. azureml/configs/sgmse.yaml")
    ap.add_argument("--compute", help="override the config's compute target")
    ap.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra train.py override, repeatable (e.g. --set epochs=10)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the resolved command and exit")
    ap.add_argument("--stream", action="store_true", help="tail the job's logs until it finishes")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.compute:
        cfg["compute"] = args.compute
    for kv in args.sets:
        if "=" not in kv:
            die(f"--set expects KEY=VALUE, got {kv!r}")
        key, value = kv.split("=", 1)
        cfg.setdefault("overrides", {})[key] = value

    cfg.setdefault("tags", {})["submitted_from_config"] = args.config.name

    if args.dry_run:
        print(f"compute     : {cfg['compute']}")
        print(f"environment : {cfg['environment']}")
        print(f"inputs      : {cfg['inputs']}")
        print(f"code        : {REPO_ROOT}")
        print(f"command     : {build_command(cfg)}")
        return

    client = make_client(cfg)
    submitted = client.jobs.create_or_update(build_job(cfg))
    print(f"submitted: {submitted.name}")
    print(f"studio   : {submitted.studio_url}")

    if args.stream:
        client.jobs.stream(submitted.name)


if __name__ == "__main__":
    main()
