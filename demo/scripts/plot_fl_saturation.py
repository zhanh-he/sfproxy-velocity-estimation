#!/usr/bin/env python3
"""Render the FL input-level / velocity-102 diagnostic as a slide-sized figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "demo/analysis/guitar_recovery/fl_20s_candidate_screen.csv"
DEFAULT_OUT = ROOT / "demo/analysis/figures/fl_velocity102_vs_audio_level"
HIGHLIGHTS = {("5xh4c", 140), ("mwC4c", 0), ("yfC4c", 160),
              ("cDC4c", 40), ("LDC4c", 80), ("WWh4c", 100)}
ORIGINAL = {("QDC4c", 40), ("QDC4c", 80)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output stem for SVG and PNG")
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No screened FL windows")
    x = np.asarray([float(row["audio_rms_int16"]) for row in rows])
    y = 100 * np.asarray([float(row["sfproxy_at_102"]) for row in rows])
    r = float(np.corrcoef(x, y)[0, 1])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.titlesize": 23, "axes.labelsize": 14,
                         "svg.fonttype": "path"})
    fig, ax = plt.subplots(figsize=(14.4, 8.1), dpi=160)
    fig.patch.set_facecolor("#fbfcfd")
    ax.set_facecolor("#fbfcfd")
    others = [row for row in rows if (row["piece"], int(row["start_s"])) not in HIGHLIGHTS | ORIGINAL]
    ax.scatter([float(row["audio_rms_int16"]) for row in others],
               [100 * float(row["sfproxy_at_102"]) for row in others],
               s=20, color="#9bb0bf", alpha=.48, edgecolors="none",
               label=f"Other windows (n={len(others)})")
    for markers, color, marker, label, size in [
        (HIGHLIGHTS, "#137c80", "D", "New review candidates (n=6)", 78),
        (ORIGINAL, "#cf694e", "*", "Original QDC4c excerpts (n=2)", 170),
    ]:
        chosen = [row for row in rows if (row["piece"], int(row["start_s"])) in markers]
        ax.scatter([float(row["audio_rms_int16"]) for row in chosen],
                   [100 * float(row["sfproxy_at_102"]) for row in chosen],
                   s=size, marker=marker, color=color, edgecolors="white",
                   linewidths=.8, zorder=4, label=label)
    for row, offset in [(next(row for row in rows if row["piece"] == "QDC4c" and row["start_s"] == "40"), (12, 10)),
                        (next(row for row in rows if row["piece"] == "QDC4c" and row["start_s"] == "80"), (12, -18))]:
        ax.annotate(f"QDC4c {row['start_s']}–{int(row['start_s'])+20} s",
                    (float(row["audio_rms_int16"]), 100 * float(row["sfproxy_at_102"])),
                    xytext=offset, textcoords="offset points", color="#9e402c", fontsize=11,
                    fontweight="bold")
    ax.set(xlabel="Original recording level · waveform RMS (int16 units)",
           ylabel="Diff-SFProxy notes at MIDI velocity 102 (%)", xlim=(0, max(x)*1.12), ylim=(-2, 101))
    ax.grid(color="#d8e0e5", alpha=.8, linewidth=.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#9eacb6")
    ax.spines["left"].set_color("#9eacb6")
    ax.tick_params(colors="#536879")
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    fig.suptitle("FL recording level and velocity-102 clustering", x=.085, y=.98,
                 ha="left", fontweight="bold", color="#182838")
    fig.text(.085, .91, f"{len(rows)} non-overlapping 20 s windows · 79 recordings · ≥50 score notes per window · Pearson r = {r:.3f}",
             color="#536879", fontsize=13)
    fig.text(.085, .018, "Authentic input gain for all plotted windows. Correlation is descriptive; the paper-selected FL checkpoint remains unverified.",
             color="#536879", fontsize=10)
    fig.subplots_adjust(left=.095, right=.965, top=.85, bottom=.15)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        fig.savefig(args.out.with_suffix(f".{extension}"), facecolor=fig.get_facecolor())
    svg = args.out.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8")
    plt.close(fig)
    print(f"{args.out}: {len(rows)} windows, r={r:.3f}")


if __name__ == "__main__":
    main()
