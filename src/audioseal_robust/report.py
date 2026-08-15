# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Aggregate every evaluation run in an output directory into one report.

`evaluate.py` writes per-run artifacts (`<label>_metrics.csv`,
`<label>_curve.csv`, `<label>_results.json` and the PNGs). This module joins
them into a single Markdown report with a cross-run comparison table, a
comparison plot, and -- most importantly -- an automatically derived caveat
list, so a number that *looks* like robustness but isn't can't be quoted
without the reason showing up next to it.

    python -m audioseal_robust.report
    python -m audioseal_robust.report --output-dir ./eval_outputs --out REPORT.md

Caveats are derived from the columns `evaluate.py` already records rather
than being written by hand:

  * `fpr_supported=False` -- the negative sample was too thin to resolve
    `fpr_target`, so `tpr_at_fpr` degenerated into "fraction of positives
    above the maximum negative score" and is a high-variance lower bound.
  * `attack_failure_rate > 0` -- the attack returned some examples
    unperturbed. Those still detect, so they inflate the apparent
    robustness of the watermark; `tpr_at_fpr_attacked` is the honest
    number over perturbed-only examples.
  * `attack_sisnr` at or below 0 dB -- the "attack" destroyed the signal
    rather than removing the watermark. Defeating a detector with audio
    that no longer resembles the input proves nothing.

Any `NOTES.md` in the output directory is included verbatim, which is where
run provenance that the CSVs cannot know about belongs (e.g. a run whose
numbers were recovered from a log written before CSV export existed).
"""

import argparse
import csv
import math
import typing as tp
from pathlib import Path

# Below this SI-SNR the attack has degraded the audio so far that beating
# the detector is meaningless -- 0 dB means the error is as large as the
# signal itself.
DEGENERATE_SNR_DB = 0.0


def _to_float(value: tp.Optional[str]) -> tp.Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_bool(value: tp.Optional[str]) -> tp.Optional[bool]:
    if value is None or value == "":
        return None
    return value.strip().lower() in ("true", "1", "yes")


def _fmt(value: tp.Optional[float], spec: str = ".3f", dash: str = "--") -> str:
    return dash if value is None else format(value, spec)


def collect_rows(output_dir: Path, suffix: str) -> tp.List[tp.Dict[str, str]]:
    """Read every `*_<suffix>.csv` in `output_dir`, newest file last."""
    rows: tp.List[tp.Dict[str, str]] = []
    for path in sorted(output_dir.glob(f"*_{suffix}.csv"), key=lambda p: p.stat().st_mtime):
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row["_source_csv"] = path.name
                rows.append(row)
    return rows


def derive_caveats(row: tp.Dict[str, str]) -> tp.List[str]:
    """Caveats implied by one metrics row. Empty means the row is quotable."""
    caveats = []
    label, attack = row.get("label", "?"), row.get("attack", "?")

    if _to_bool(row.get("fpr_supported")) is False:
        n_neg = row.get("n_negatives") or "?"
        resolution = _fmt(_to_float(row.get("fpr_resolution")))
        target = _to_float(row.get("fpr_target"))
        needed = row.get("min_negatives_for_target")
        if not needed and target:
            needed = str(math.ceil(1.0 / target))
        need = f" (need >= {needed})" if needed else ""
        caveats.append(
            f"`{label}` / `{attack}`: only {n_neg} negatives -- cannot resolve FPR below "
            f"{resolution}{need}. `tpr_at_fpr` is a high-variance lower bound, "
            f"not a {row.get('fpr_target', '?')} operating point."
        )

    failure_rate = _to_float(row.get("attack_failure_rate"))
    if failure_rate is not None and failure_rate > 0:
        attacked = _to_float(row.get("tpr_at_fpr_attacked"))
        extra = "" if attacked is None else f" `tpr_at_fpr_attacked`={attacked:.3f} over perturbed-only examples."
        caveats.append(
            f"`{label}` / `{attack}`: the attack left {failure_rate:.1%} of examples unperturbed "
            f"({row.get('n_attack_failures', '?')} of them). Unperturbed watermarked audio still "
            f"detects, so it inflates apparent robustness.{extra}"
        )

    snr = _to_float(row.get("attack_sisnr"))
    if snr is not None and snr <= DEGENERATE_SNR_DB:
        caveats.append(
            f"`{label}` / `{attack}`: attack SI-SNR is {snr:.1f} dB -- the attack destroyed the "
            f"audio rather than removing the watermark. Evading a detector with audio this far "
            f"from the input does not demonstrate a watermark weakness."
        )
    return caveats


def plot_comparison(rows: tp.List[tp.Dict[str, str]], out_path: Path) -> tp.Optional[Path]:
    """Grouped bars of tpr@fpr and bit accuracy per run/attack.

    Detection metrics and SI-SNR share no axis, so SI-SNR is annotated on
    each bar instead of plotted -- it is the number that says whether a low
    tpr@fpr was earned or simply bought by wrecking the audio.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plotted = [r for r in rows if _to_float(r.get("tpr_at_fpr")) is not None]
    if not plotted:
        return None

    names = [f"{r.get('label', '?')}\n{r.get('attack', '?')}" for r in plotted]
    tpr = [_to_float(r.get("tpr_at_fpr")) or 0.0 for r in plotted]
    bit_acc = [_to_float(r.get("bit_accuracy")) or 0.0 for r in plotted]

    x = np.arange(len(plotted))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7.0, 1.9 * len(plotted)), 4.8))
    bars_tpr = ax.bar(x - width / 2, tpr, width, label="TPR@FPR")
    ax.bar(x + width / 2, bit_acc, width, label="bit accuracy")

    # 0.5 is chance for bit accuracy: at or below it the message carries no
    # recoverable information, which is the real "watermark is gone" line.
    ax.axhline(0.5, linestyle=":", linewidth=1, color="grey")
    ax.annotate("chance (bit acc)", xy=(0.995, 0.5), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=7, color="grey")

    for rect, row in zip(bars_tpr, plotted):
        snr = _to_float(row.get("attack_sisnr"))
        if snr is None:
            continue
        ax.annotate(f"{snr:.0f} dB", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score")
    ax.set_title("Detection vs. attack (bar labels: attack SI-SNR)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_markdown(
    output_dir: Path,
    rows: tp.List[tp.Dict[str, str]],
    curve_rows: tp.List[tp.Dict[str, str]],
    comparison_png: tp.Optional[Path],
) -> str:
    lines: tp.List[str] = ["# Watermark robustness evaluation report", ""]

    notes = output_dir / "NOTES.md"
    if notes.is_file():
        lines += [notes.read_text(encoding="utf-8").strip(), ""]

    if not rows:
        lines += ["No `*_metrics.csv` found -- run `audioseal_robust.evaluate` first.", ""]
        return "\n".join(lines)

    lines += ["## Results", "",
              "| run | attack | tag | bit acc | TPR@FPR | TPR@FPR (perturbed) | F1 | attack SI-SNR | n pos/neg | FPR ok | time (s) |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|"]
    for r in rows:
        supported = _to_bool(r.get("fpr_supported"))
        lines.append(
            "| {label} | {attack} | {tag} | {bit} | {tpr} | {tpr_att} | {f1} | {snr} | {npos}/{nneg} | {ok} | {secs} |".format(
                label=r.get("label", "?"), attack=r.get("attack", "?"), tag=r.get("tag", ""),
                bit=_fmt(_to_float(r.get("bit_accuracy"))),
                tpr=_fmt(_to_float(r.get("tpr_at_fpr"))),
                tpr_att=_fmt(_to_float(r.get("tpr_at_fpr_attacked"))),
                f1=_fmt(_to_float(r.get("f1"))),
                snr=_fmt(_to_float(r.get("attack_sisnr")), ".1f"),
                npos=r.get("n_positives", "?"), nneg=r.get("n_negatives", "?"),
                ok={True: "yes", False: "**no**", None: "--"}[supported],
                secs=_fmt(_to_float(r.get("seconds")), ".0f"),
            )
        )
    lines.append("")

    lines += ["## Confusion matrices", "",
              "| run | attack | watermarked detected | watermarked missed | clean detected | clean missed |",
              "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r.get('label', '?')} | {r.get('attack', '?')} | {r.get('tp', '--')} | "
            f"{r.get('fn', '--')} | {r.get('fp', '--')} | {r.get('tn', '--')} |"
        )
    lines.append("")

    caveats = [c for r in rows for c in derive_caveats(r)]
    lines += ["## Caveats", ""]
    lines += ([f"- {c}" for c in caveats] if caveats
              else ["- None: every run resolved its FPR target, perturbed every example, "
                    "and kept attack SI-SNR above 0 dB."])
    lines.append("")

    if curve_rows:
        lines += ["## Robustness curves", "",
                  "| run | attack | t* | bit acc | TPR@FPR | attack SI-SNR |", "|---|---|---:|---:|---:|---:|"]
        for r in curve_rows:
            lines.append(
                f"| {r.get('label', '?')} | {r.get('attack', '?')} | "
                f"{_fmt(_to_float(r.get('t_star')), '.4f')} | {_fmt(_to_float(r.get('bit_accuracy')))} | "
                f"{_fmt(_to_float(r.get('tpr_at_fpr')))} | {_fmt(_to_float(r.get('attack_sisnr')), '.1f')} |"
            )
        lines.append("")

    lines += ["## Run configuration", "",
              "| run | device | eval_dir | segment (s) | batch | perceptual SI-SNR | perceptual PESQ |",
              "|---|---|---|---:|---:|---:|---:|"]
    for label in dict.fromkeys(r.get("label", "?") for r in rows):
        r = next(x for x in rows if x.get("label") == label)
        lines.append(
            f"| {label} | {r.get('device', '--')} | `{r.get('eval_dir', '--')}` | "
            f"{r.get('segment_duration', '--')} | {r.get('batch_size', '--')} | "
            f"{_fmt(_to_float(r.get('perceptual_sisnr')), '.2f')} | "
            f"{_fmt(_to_float(r.get('perceptual_pesq')), '.2f')} |"
        )
    lines.append("")

    lines += ["## Figures", ""]
    if comparison_png is not None:
        lines += ["### Cross-run comparison", "", f"![comparison]({comparison_png.name})", ""]
    for png in sorted(output_dir.glob("*.png")):
        if comparison_png is not None and png.name == comparison_png.name:
            continue
        lines += [f"### {png.stem}", "", f"![{png.stem}]({png.name})", ""]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default="./eval_outputs", type=Path,
                        help="directory evaluate.py wrote its artifacts to")
    parser.add_argument("--out", default=None, type=Path,
                        help="report path (default: <output-dir>/REPORT.md)")
    parser.add_argument("--include", default=None, nargs="*",
                        help="only include runs whose label is in this list")
    args = parser.parse_args()

    output_dir = args.output_dir
    rows = collect_rows(output_dir, "metrics")
    curve_rows = collect_rows(output_dir, "curve")
    if args.include:
        rows = [r for r in rows if r.get("label") in args.include]
        curve_rows = [r for r in curve_rows if r.get("label") in args.include]

    comparison_png = plot_comparison(rows, output_dir / "comparison.png") if rows else None
    report_path = args.out or (output_dir / "REPORT.md")
    report_path.write_text(render_markdown(output_dir, rows, curve_rows, comparison_png), encoding="utf-8")
    print(f"wrote {report_path} ({len(rows)} attack rows, {len(curve_rows)} curve points)")
    if comparison_png is not None:
        print(f"wrote {comparison_png}")


if __name__ == "__main__":
    main()
