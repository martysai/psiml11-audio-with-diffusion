# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Plot renderers for evaluate.py's results dict, saved as PNGs rather than
just printed/logged as numbers:

  - `plot_confusion_matrices`: one 2x2 "message preserved?" confusion matrix
    (TP/FN/FP/TN, see metrics.confusion_counts) per attack, with its F1 in
    the title.
  - `plot_robustness_curve`: detection (TPR@FPR) vs. attack strength t*, one
    line per strength-aware attack (sgmse, audioldm).

Both import matplotlib lazily so evaluate.py stays importable without it if
you only need the numeric results. Return None (and log nothing) if there's
nothing to plot yet -- e.g. all attacks are still stubbed NotImplementedError
skips, which is the day-1 state until backbones are wired up (see
attacks.py).
"""

import logging
import typing as tp
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_confusion_matrices(results: tp.Dict[str, tp.Any], out_path: Path) -> tp.Optional[Path]:
    attacks = {name: r for name, r in results.get("attacks", {}).items() if "confusion" in r}
    if not attacks:
        logger.info("no attack produced a confusion matrix yet (all skipped?), nothing to plot")
        return None

    import matplotlib.pyplot as plt
    import numpy as np

    n = len(attacks)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    for ax, (name, r) in zip(axes[0], attacks.items()):
        c = r["confusion"]
        matrix = np.array([[c["tp"], c["fn"]], [c["fp"], c["tn"]]])
        ax.imshow(matrix, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["detected", "not detected"])
        ax.set_yticklabels(["watermarked", "clean"])
        vmax = matrix.max()
        for i in range(2):
            for j in range(2):
                color = "white" if matrix[i, j] > vmax / 2 else "black"
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=14)
        ax.set_title(f"{name} ({r.get('tag')})\nF1={r['f1']:.3f}")

    fig.suptitle(f"Message preserved? -- {results.get('label')}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_robustness_curve(results: tp.Dict[str, tp.Any], out_path: Path) -> tp.Optional[Path]:
    curves = {
        name: r["robustness_curve"] for name, r in results.get("attacks", {}).items() if r.get("robustness_curve")
    }
    if not curves:
        logger.info("no attack produced a robustness curve yet (sgmse/audioldm still stubs?), nothing to plot")
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, curve in curves.items():
        t_stars = [p["t_star"] for p in curve]
        tprs = [p["tpr_at_fpr"] for p in curve]
        ax.plot(t_stars, tprs, marker="o", label=name)

    ax.set_xlabel("t* (attack strength)")
    ax.set_ylabel("TPR@FPR")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Detection vs. attack strength -- {results.get('label')}")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_roc_curves(results: tp.Dict[str, tp.Any], out_path: Path) -> tp.Optional[Path]:
    """Plot ROC curves (TPR vs FPR) for attacks that produced score pools.

    Expects each attack to have `roc_curve` with `fpr` and `tpr` lists and
    optionally `roc_auc` scalar in its metrics.
    """
    curves = {}
    for name, r in results.get("attacks", {}).items():
        rc = r.get("roc_curve")
        if rc and rc.get("fpr") and rc.get("tpr"):
            curves[name] = {"fpr": rc["fpr"], "tpr": rc["tpr"], "auc": r.get("roc_auc")}

    if not curves:
        logger.info("no ROC data available to plot yet, nothing to plot")
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for name, rc in curves.items():
        fpr = rc["fpr"]
        tpr = rc["tpr"]
        label = name
        if rc.get("auc") is not None:
            label = f"{name} (AUC={rc['auc']:.3f})"
        ax.plot(fpr, tpr, label=label)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_title(f"ROC curves -- {results.get('label')}")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
