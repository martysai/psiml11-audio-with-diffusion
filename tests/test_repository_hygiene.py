# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

_INTERNAL_LITERALS = (
    "manifold" + "3",
    "as-" + "manifold",
    "as-" + "playground",
    "as-" + "shared",
    "as" + "playground",
    "playground" + "-rg",
    "shared" + "-rg",
    "psiml_assets_" + "playground",
    "_AZUREML_" + "SINGULARITY_JOB_UAI",
    "Singularity" + ".NC",
    "ASG " + "Azure ML",
    "westus" + "3",
    "maratsaidov-" + "microsoft",
    "@" + "microsoft.com",
    "Microsoft." + "MachineLearningServices",
    "_manifold" + ".yaml",
    "submit_" + "manifold.py",
)

_PERSONAL_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+"
    r"|/(?:Users|home)/[^/\s\"']+"
    r"|/mnt/(?:fast|data|home)/[^/\s\"']+)",
    re.IGNORECASE,
)
_DATED_RUN_NAME = re.compile(
    r"\b(?:train|smoke|eval|hsj|fb)-[a-z0-9-]*-"
    r"(?:0[1-9]|1[0-2])(?:0[1-9]|[12][0-9]|3[01])-"
    r"[0-2][0-9][0-5][0-9](?:[0-5][0-9])?\b",
    re.IGNORECASE,
)
_TENANT_COMPUTE_NAME = re.compile(r"\ba100x[0-9]+\b", re.IGNORECASE)
_GUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_CLOUD_IDENTIFIER = re.compile(
    r"(?:/subscriptions/|subscription[_ -]?id|tenant[_ -]?id|client[_ -]?id)"
    r"[^\n]{0,120}"
    + _GUID.pattern,
    re.IGNORECASE,
)
_DOCUMENT_SUFFIXES = {".md", ".yaml", ".yml", ".sh", ".ps1"}


def _repository_files():
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        path = REPO_ROOT / raw_path.decode("utf-8")
        if path.is_file():
            yield path


def _read_text(path: Path):
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def test_repository_contains_no_internal_infrastructure_metadata():
    findings = []
    for path in _repository_files():
        text = _read_text(path)
        if text is None:
            continue

        relative = path.relative_to(REPO_ROOT).as_posix()
        lowered = text.lower()
        for token in _INTERNAL_LITERALS:
            if token.lower() in lowered:
                findings.append(f"{relative}: contains {token!r}")

        if _PERSONAL_PATH.search(text):
            findings.append(f"{relative}: contains an absolute user path")
        if _DATED_RUN_NAME.search(text):
            findings.append(f"{relative}: contains a dated experiment run name")
        if _TENANT_COMPUTE_NAME.search(text):
            findings.append(f"{relative}: contains a tenant-specific compute name")
        if path.suffix.lower() != ".ipynb" and _GUID.search(text):
            findings.append(f"{relative}: contains a GUID")
        if _CLOUD_IDENTIFIER.search(text):
            findings.append(f"{relative}: contains a cloud resource identifier")
        if "manifold" in path.name.lower() or "singularity" in path.name.lower():
            findings.append(f"{relative}: internal compute name appears in filename")
        if path.suffix.lower() in _DOCUMENT_SUFFIXES:
            for platform_name in ("manifold", "singularity"):
                if re.search(rf"\b{platform_name}\b", text, re.IGNORECASE):
                    findings.append(
                        f"{relative}: contains internal compute name "
                        f"{platform_name!r}"
                    )

    assert not findings, "\n" + "\n".join(sorted(findings))
