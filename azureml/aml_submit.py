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
    if cfg["direction"] not in ("sgmse", "audioldm"):
        die(f"direction must be 'sgmse' or 'audioldm', got {cfg['direction']!r}")
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

    ws = cfg.get("workspace") or {}
    if all(ws.get(k) for k in ("subscription_id", "resource_group_name", "workspace_name")):
        return MLClient(
            credential,
            subscription_id=ws["subscription_id"],
            resource_group_name=ws["resource_group_name"],
            workspace_name=ws["workspace_name"],
        )
    # Fall back to whatever `az configure --defaults` / config.json already
    # points at, so the config file doesn't have to hardcode a workspace.
    try:
        return MLClient.from_config(credential)
    except Exception as exc:
        die(
            "no workspace in the config file and MLClient.from_config() failed "
            f"({exc}). Either fill in the `workspace:` block or run: "
            "az configure --defaults group=<rg> workspace=<ws>"
        )


def build_job(cfg: Dict[str, Any]):
    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.constants import AssetTypes, InputOutputModes

    direction = cfg["direction"]
    inputs_cfg = cfg["inputs"]

    parts = [
        "bash azureml/aml_run.sh",
        direction,
        "--librispeech ${{inputs.librispeech}}",
        "--checkpoints ${{inputs.checkpoints}}",
        "--artifacts ${{outputs.artifacts}}",
        "--",
        *build_overrides(cfg.get("overrides")),
    ]

    job = command(
        code=str(REPO_ROOT),
        command=" ".join(parts),
        environment=cfg["environment"],
        compute=cfg["compute"],
        experiment_name=cfg.get("experiment_name", "audioseal-robust-diffusion-swap"),
        display_name=cfg.get("display_name", f"swap-{direction}"),
        description=cfg.get("description"),
        inputs={
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
            "checkpoints": Input(
                type=AssetTypes.CUSTOM_MODEL,
                path=inputs_cfg["checkpoints"],
                mode=InputOutputModes.RO_MOUNT,
            ),
        },
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
        direction = cfg["direction"]
        overrides = " ".join(build_overrides(cfg.get("overrides")))
        print(f"compute     : {cfg['compute']}")
        print(f"environment : {cfg['environment']}")
        print(f"inputs      : {cfg['inputs']}")
        print(f"code        : {REPO_ROOT}")
        print(
            "command     : bash azureml/aml_run.sh "
            f"{direction} --librispeech <mount> --checkpoints <mount> "
            f"--artifacts <mount> -- {overrides}"
        )
        return

    client = make_client(cfg)
    submitted = client.jobs.create_or_update(build_job(cfg))
    print(f"submitted: {submitted.name}")
    print(f"studio   : {submitted.studio_url}")

    if args.stream:
        client.jobs.stream(submitted.name)


if __name__ == "__main__":
    main()
