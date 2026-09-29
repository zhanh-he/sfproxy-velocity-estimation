#!/usr/bin/env python3
"""Apply one constant gain per demo MP3 for comfortable A/B listening.

This only changes the audio preview gain. It does not change MIDI, relative
note dynamics, or any reported velocity metric. Requires ffmpeg and ffprobe.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path


DOCS = Path(__file__).resolve().parents[1] / "docs"


def loudness(path: Path) -> tuple[float, float]:
    result = subprocess.run(["ffmpeg", "-nostats", "-i", str(path), "-filter_complex",
                             "ebur128=peak=true", "-f", "null", "-"],
                            capture_output=True, text=True, check=True)
    integrated = re.findall(r"I:\s*(-?\d+\.\d+) LUFS", result.stderr)
    peaks = re.findall(r"Peak:\s*(-?\d+\.\d+) dBFS", result.stderr)
    if not integrated or not peaks:
        raise ValueError(f"Could not measure loudness: {path}")
    return float(integrated[-1]), float(peaks[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-lufs", type=float, default=-24.0)
    parser.add_argument("--max-peak-dbfs", type=float, default=-1.0)
    parser.add_argument("--report", type=Path, default=DOCS.parent / "analysis/listening_gain_report.csv")
    args = parser.parse_args()
    cases = json.loads((DOCS / "assets/cases.json").read_text())["cases"]
    report = []
    for case in cases:
        for method in case["available"]:
            path = DOCS / case["assetBase"] / f"{method}.mp3"
            before, peak_before = loudness(path)
            gain = 0.0 if abs(args.target_lufs - before) <= 0.6 else min(
                args.target_lufs - before, args.max_peak_dbfs - peak_before)
            if abs(gain) >= 0.05:
                output = path.with_name(path.stem + ".gain-tmp.mp3")
                try:
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path),
                                    "-filter:a", f"volume={gain:.3f}dB", "-ac", "1",
                                    "-ar", "22050", "-b:a", "160k", str(output)], check=True)
                    output.replace(path)
                finally:
                    output.unlink(missing_ok=True)
            after, peak_after = loudness(path)
            report.append({"case_id": case["id"], "method": method,
                           "original_lufs": before, "original_peak_dbfs": peak_before,
                           "applied_gain_db": round(gain, 3),
                           "output_lufs": after, "output_peak_dbfs": peak_after})
            print(f"{case['id']}/{method}: {before:.1f} → {after:.1f} LUFS, gain {gain:+.1f} dB")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(report)


if __name__ == "__main__":
    main()
