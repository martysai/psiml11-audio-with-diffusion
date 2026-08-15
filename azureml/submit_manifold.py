#!/usr/bin/env python3
"""Submit an azureml/jobs/*_manifold.yaml with the subscription filled in.

This repo is PUBLIC, so no subscription id is committed. The Singularity
("manifold") job YAMLs still need one, because a virtual cluster is targeted
by full ARM id rather than by compute name -- `compute:` and
`_AZUREML_SINGULARITY_JOB_UAI` both carry one. They ship with a literal
``${AZUREML_SUBSCRIPTION_ID}`` placeholder instead.

`az ml job create` does no variable expansion, so this wrapper does it:

    python azureml/submit_manifold.py azureml/jobs/eval_hsj_baseline_manifold.yaml
    python azureml/submit_manifold.py azureml/jobs/eval_hsj_baseline_manifold.yaml --dry-run

Anything after the YAML path is forwarded verbatim to `az ml job create`, so
`--set`, `--stream`, `--web` and friends keep working:

    python azureml/submit_manifold.py azureml/jobs/train_audioldm_manifold.yaml \
        --set display_name=retry-2

The subscription is resolved from $AZUREML_SUBSCRIPTION_ID (the same variable
aml_submit.py honours), falling back to the active `az account show`. The
resolved YAML is written next to the original and deleted afterwards: `code:
../..` is interpreted relative to the YAML's own directory, so it cannot be
staged through a temp dir without silently changing which tree gets uploaded.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aml_submit import _az_subscription_id, die  # noqa: E402

PLACEHOLDER = "${AZUREML_SUBSCRIPTION_ID}"
# Manifold jobs live in their own workspace, not the playground one the rest
# of the repo defaults to; passing these explicitly keeps the command correct
# regardless of what `az configure --defaults` currently points at.
DEFAULT_TARGET = ["--resource-group", "manifold3-rg", "--workspace-name", "as-manifold3-w3-ws"]


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        raise SystemExit(0 if argv else 2)

    job_yaml = Path(argv[0]).resolve()
    passthrough = argv[1:]
    dry_run = "--dry-run" in passthrough
    passthrough = [a for a in passthrough if a != "--dry-run"]

    if not job_yaml.is_file():
        die(f"no such job file: {job_yaml}")

    text = job_yaml.read_text(encoding="utf-8")
    if PLACEHOLDER not in text:
        die(
            f"{job_yaml.name} contains no {PLACEHOLDER}; submit it directly with "
            "`az ml job create -f ...` instead of through this wrapper."
        )

    subscription = os.environ.get("AZUREML_SUBSCRIPTION_ID") or _az_subscription_id()
    if not subscription:
        die(
            "could not resolve a subscription id. Run `az login`, or export "
            "AZUREML_SUBSCRIPTION_ID=<id>."
        )

    # Written beside the original so the relative `code:` path still resolves.
    resolved = job_yaml.with_name(f".{job_yaml.stem}.resolved.yaml")
    resolved.write_text(text.replace(PLACEHOLDER, subscription), encoding="utf-8")

    # Only ever echo the tail: the whole point of the placeholder is that the
    # full id does not end up in logs, terminal scrollback or CI output.
    print(f"resolved {PLACEHOLDER} -> subscription ...{subscription[-6:]}", file=sys.stderr)

    target = [] if any(a in passthrough for a in ("--resource-group", "-g")) else DEFAULT_TARGET
    cmd = ["az", "ml", "job", "create", "-f", str(resolved), *target, *passthrough]

    try:
        if dry_run:
            print("[dry-run] " + " ".join(cmd))
            print(f"[dry-run] resolved yaml kept at {resolved}")
            return
        raise SystemExit(subprocess.call(cmd))
    finally:
        if not dry_run:
            resolved.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
