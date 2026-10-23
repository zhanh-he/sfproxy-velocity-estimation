#!/usr/bin/env python3
"""Check that every advertised public listening case has matched 20 s assets."""

from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
import subprocess
from pathlib import Path


DOCS = Path(__file__).resolve().parents[1] / "docs"
METHODS = ("reference", "flat64", "diffsynth", "sfproxy")


def main() -> None:
    manifest = json.loads((DOCS / "assets/cases.json").read_text())
    ids = set()
    total_audio = 0
    expected_scores = {}
    for case in manifest["cases"]:
        case_id = case["id"]
        if case_id in ids:
            raise ValueError(f"Duplicate case ID: {case_id}")
        ids.add(case_id)
        available = case["available"]
        if len(set(available)) != len(available) or any(method not in METHODS for method in available):
            raise ValueError(f"Invalid methods in {case_id}: {available}")
        if not available:
            if case.get("notesUrl"):
                raise ValueError(f"{case_id} has note data but no published methods")
            print(f"{case_id}: reserved slot")
            continue
        if "reference" not in available or "flat64" not in available:
            raise ValueError(f"{case_id} has predictions without a reference and Flat 64")
        notes_path = DOCS / case["notesUrl"]
        payload = json.loads(notes_path.read_text())
        if payload["duration_seconds"] != 20:
            raise ValueError(f"{case_id}: expected exactly 20 seconds")
        if payload.get("case_id") != case_id and case_id != "maestro-scriabin-60":
            raise ValueError(f"{case_id}: note JSON belongs to {payload.get('case_id')}")
        reference = payload["notes"]["reference"]
        if set(case.get("scores", {})) != set(available):
            raise ValueError(f"{case_id}: missing or unexpected card scores")
        for method in available:
            base = DOCS / case["assetBase"] / method
            for ext in ("mp3", "mid"):
                path = base.with_suffix(f".{ext}")
                if not path.is_file():
                    raise FileNotFoundError(path)
            notes = payload["notes"][method]
            if len(notes) != len(reference):
                raise ValueError(f"{case_id}/{method}: note count differs")
            for original, item in zip(reference, notes):
                if (original["p"], original["s"], original["e"]) != (item["p"], item["s"], item["e"]):
                    raise ValueError(f"{case_id}/{method}: note identity or timing differs")
                if not 1 <= item["v"] <= 127:
                    raise ValueError(f"{case_id}/{method}: velocity out of MIDI range")
            if payload.get("stats", {}).get(method, {}).get("mae_20s") is not None:
                measured = statistics.mean(abs(a["v"] - b["v"]) for a, b in zip(reference, notes))
                if abs(measured - payload["stats"][method]["mae_20s"]) > 0.001:
                    raise ValueError(f"{case_id}/{method}: displayed MAE differs from note data")
            score = case["scores"][method]
            for metric in ("bssl", "bstl"):
                value = score.get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1:
                    raise ValueError(f"{case_id}/{method}: invalid {metric} card score")
            if case["velocityGroundTruth"]:
                measured = statistics.mean(abs(a["v"] - b["v"]) for a, b in zip(reference, notes))
                if abs(measured - score.get("mae", float("inf"))) > 0.001:
                    raise ValueError(f"{case_id}/{method}: card MAE differs from aligned MIDI")
            elif "mae" in score:
                raise ValueError(f"{case_id}/{method}: guitar has no verified velocity MAE")
            expected_scores[(case_id, method)] = score
            if shutil.which("ffprobe"):
                result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                         "-of", "default=noprint_wrappers=1:nokey=1", str(base.with_suffix(".mp3"))],
                                        capture_output=True, text=True, check=True)
                if abs(float(result.stdout) - 20.0) > 0.1:
                    raise ValueError(f"{case_id}/{method}: audio is not 20 seconds")
            total_audio += 1
        print(f"{case_id}: {len(reference)} aligned notes, {len(available)} audio methods")
    report = DOCS.parent / "analysis/listening_case_scores.csv"
    with report.open(newline="", encoding="utf-8") as stream:
        score_rows = list(csv.DictReader(stream))
    if len(score_rows) != len(expected_scores):
        raise ValueError("Card score report length differs from the manifest")
    for row in score_rows:
        key = (row["case_id"], row["method"])
        score = expected_scores.pop(key, None)
        if score is None or any(float(row[metric]) != score[metric] for metric in ("bssl", "bstl")):
            raise ValueError(f"Card score CSV differs from manifest for {key}")
        mae_matches = (float(row["mae"]) == score["mae"]) if "mae" in score else (row["mae"] == "")
        if not mae_matches:
            raise ValueError(f"Card MAE CSV differs from manifest for {key}")
    if expected_scores:
        raise ValueError(f"Card score CSV omits {len(expected_scores)} methods")
    print(f"Validated {len(ids)} cases and {total_audio} 20-second audio files")


if __name__ == "__main__":
    main()
