"""Build presentation SVGs and site JSON from the camera-ready CSV tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "docs" / "assets"
COLORS = {"Flat velocity": "#8a95a7", "VeloEst": "#d9b869", "Diff-Synth": "#ec876c", "Diff-SFProxy": "#55c8b6"}


def rows(name: str):
    with (ROOT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def svg_header(title: str, subtitle: str):
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720" role="img" aria-label="' + escape(title) + '">',
        '<rect width="1280" height="720" fill="#101a29"/>',
        f'<text x="76" y="82" fill="#f7f3e9" font-family="Arial,sans-serif" font-size="38" font-weight="700">{escape(title)}</text>',
        f'<text x="76" y="122" fill="#aeb9c9" font-family="Arial,sans-serif" font-size="19">{escape(subtitle)}</text>',
    ]


def text(x, y, value, color="#f7f3e9", size=18, weight=400):
    return f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}">{escape(str(value))}</text>'


def guitar_figure(data):
    selected = [r for r in data if (r["method"], r["variant"]) in {
        ("Flat velocity", "64"), ("VeloEst", "zero-shot"),
        ("Diff-Synth", "5 s"), ("Diff-SFProxy", "5 s")
    }]
    parts = svg_header("Guitar dynamics: Diff-SFProxy leads", "Pearson r on rendered vs real Bark-scale loudness  |  ISMIR 2026 Table 2")
    for panel, (label, prefix) in enumerate([("GAPS test", "gaps"), ("François Leduc", "fl")]):
        x0 = 75 + panel * 620
        parts.append(text(x0, 192, label, size=26, weight=700))
        for tick in [0, 0.25, 0.5, 0.75, 1]:
            x = x0 + 222 + tick * 302
            parts.append(f'<line x1="{x}" y1="225" x2="{x}" y2="585" stroke="#344255" stroke-width="1"/>')
            parts.append(text(x - 8, 610, f"{tick:g}", color="#93a2b4", size=14))
        for i, row in enumerate(selected):
            y = 245 + i * 87
            value = float(row[f"{prefix}_r_bssl"])
            color = COLORS[row["method"]]
            parts.append(text(x0, y + 26, row["method"], size=18, weight=600))
            parts.append(f'<rect x="{x0 + 222}" y="{y}" width="{302 * value:.1f}" height="38" rx="8" fill="{color}"/>')
            parts.append(text(x0 + 232 + 302 * value, y + 27, f"{value:.3f}", size=18, weight=700))
    parts.append(text(76, 666, "5 s backend for both adaptation methods. Guitar velocity MAE is unavailable because labels are absent.", color="#aeb9c9", size=16))
    parts.append("</svg>")
    (OUT / "guitar_results.svg").write_text("\n".join(parts) + "\n")


def recovery_figure(data):
    parts = svg_header("Proxy gradients recover velocity", "Velocity MAE reduction after inversion  |  ISMIR 2026 Table 1")
    for group, instrument in enumerate(["Piano", "Guitar"]):
        x0 = 75 + group * 620
        parts.append(text(x0, 190, instrument, size=27, weight=700))
        for i, row in enumerate(r for r in data if r["instrument"] == instrument):
            y = 238 + i * 112
            value = float(row["stress_gain_pct"])
            is_default = row["segment_s"] == "5"
            color = "#55c8b6" if is_default else "#687c94"
            parts.append(text(x0, y + 23, row["segment_s"] + " s", size=21, weight=700 if is_default else 400))
            parts.append(f'<rect x="{x0 + 100}" y="{y}" width="{value * 5.2:.1f}" height="38" rx="8" fill="{color}"/>')
            parts.append(text(x0 + 110 + value * 5.2, y + 27, f"{value:.1f}%", size=20, weight=700))
            parts.append(text(x0 + 100, y + 68, f"stress MAE {row['stress_init_mae']} → {row['stress_recovered_mae']}", color="#aeb9c9", size=17))
    parts.append(text(76, 666, "5 s is the default: longer 10 s segments add cost without meaningful recovery gain.", color="#aeb9c9", size=17))
    parts.append("</svg>")
    (OUT / "recovery.svg").write_text("\n".join(parts) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paper = rows("paper_results.csv")
    recovery = rows("proxy_recovery.csv")
    guitar_figure(paper)
    recovery_figure(recovery)
    (OUT / "paper_results.json").write_text(json.dumps({"evaluation": paper, "recovery": recovery}, indent=2) + "\n")


if __name__ == "__main__":
    main()
