#!/usr/bin/env python3
"""Score the exact 20-second MP3 excerpts served by the listening site.

Requires the research analysis dependencies (torch, torchaudio, librosa). The
audio correlations use the paper's BSSL evaluator with its sone / Ntot path;
MAE is measured on aligned MIDI velocities and is only defined for piano.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path


DEMO = Path(__file__).resolve().parents[1]
DOCS = DEMO / "docs"
MANIFEST = DOCS / "assets/cases.json"
REPORT = DEMO / "analysis/listening_case_scores.csv"
METHODS = ("reference", "flat64", "diffsynth", "sfproxy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", action="append", help="Score only selected cases")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--write-manifest", action="store_true", help="Store rounded card scores in cases.json")
    args = parser.parse_args()

    sys.path.insert(0, str(DEMO / "analysis/src"))
    from data_analysis.evaluation.bssl_eval import evaluate_bssl_pair

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    chosen = set(args.case_id or (case["id"] for case in manifest["cases"]))
    unknown = chosen - {case["id"] for case in manifest["cases"]}
    if unknown:
        parser.error(f"Unknown cases: {', '.join(sorted(unknown))}")

    rows = []
    for case in manifest["cases"]:
        if case["id"] not in chosen:
            continue
        notes = json.loads((DOCS / case["notesUrl"]).read_text(encoding="utf-8"))["notes"]
        reference = notes["reference"]
        audio_dir = DOCS / case["assetBase"]
        reference_audio = audio_dir / "reference.mp3"
        card_scores = {}
        for method in METHODS:
            if method not in case["available"]:
                continue
            if method == "reference":
                bssl, bstl = 1.0, 1.0
            else:
                result = evaluate_bssl_pair(
                    audio_dir / f"{method}.mp3", reference_audio,
                    sample_rate=22050, frames_per_second=50, fft_size=1024,
                    bssl_mode="sone", num_samples=2048, normalization="zscore",
                    device=args.device, pearson_only=True,
                )["summary"]
                bssl = result["bssl_pearson_correlation"]
                bstl = result["ntot_pearson_correlation"]
            if not all(math.isfinite(v) and -1.00001 <= v <= 1.00001 for v in (bssl, bstl)):
                raise ValueError(f"Invalid audio correlation for {case['id']}/{method}: {bssl}, {bstl}")
            score = {"bssl": round(bssl, 3), "bstl": round(bstl, 3)}
            if case["velocityGroundTruth"]:
                if len(reference) != len(notes[method]):
                    raise ValueError(f"Unmatched note count: {case['id']}/{method}")
                score["mae"] = round(statistics.mean(
                    abs(original["v"] - predicted["v"])
                    for original, predicted in zip(reference, notes[method])
                ), 3)
            card_scores[method] = score
            rows.append({"case_id": case["id"], "method": method,
                         "mae": score.get("mae", ""), "bssl": score["bssl"], "bstl": score["bstl"]})
            print(f"{case['id']}/{method}: {score}", flush=True)
        case["scores"] = card_scores

    if args.case_id and REPORT.is_file():
        with REPORT.open(newline="", encoding="utf-8") as stream:
            rows.extend(row for row in csv.DictReader(stream) if row["case_id"] not in chosen)
    order = {(case["id"], method): (i, j)
             for i, case in enumerate(manifest["cases"]) for j, method in enumerate(METHODS)}
    rows = [row for row in rows if (row["case_id"], row["method"]) in order]
    rows.sort(key=lambda row: order[(row["case_id"], row["method"])])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case_id", "method", "mae", "bssl", "bstl"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if args.write_manifest:
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
