#!/usr/bin/env python3
"""Build one 20-second, aligned listening case from research-workspace files.

The inputs are a reference recording/MIDI pair and, when available, saved
Diff-Synth and Diff-SFProxy prediction MIDIs for the same performance. The
renderer is the SFZ instrument used for that dataset. Missing predictions
stay missing; this script never substitutes another checkpoint for a paper run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import pretty_midi

from export_maestro_demo import clipped_midi, notes_in_window


def match_notes(reference: list[dict], predicted: list[dict], label: str) -> list[dict]:
    if len(reference) != len(predicted):
        raise ValueError(f"{label}: {len(predicted)} notes, expected {len(reference)}")
    remaining = list(predicted)
    matched = []
    for index, note in enumerate(reference):
        candidates = [item for item in remaining if item["p"] == note["p"]]
        if not candidates:
            raise ValueError(f"{label}: no match for reference note {index}")
        item = min(candidates, key=lambda x: abs(x["s"] - note["s"]) + abs(x["e"] - note["e"]))
        if abs(item["s"] - note["s"]) > 0.01 or abs(item["e"] - note["e"]) > 0.01:
            raise ValueError(f"{label}: timing mismatch for reference note {index}: {note} vs {item}")
        remaining.remove(item)
        matched.append({**note, "v": item["v"]})
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-audio", required=True, type=Path)
    parser.add_argument("--reference-midi", required=True, type=Path)
    parser.add_argument("--diffsynth-midi", type=Path)
    parser.add_argument("--sfproxy-midi", type=Path)
    parser.add_argument("--sfz", required=True, type=Path)
    parser.add_argument("--sfizz-render", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--velocity-ground-truth", action="store_true")
    args = parser.parse_args()

    if args.start < 0 or args.duration != 20.0:
        parser.error("Cases must use a nonnegative start and exactly 20 seconds")
    sources = {"reference": args.reference_midi}
    for label, path in (("diffsynth", args.diffsynth_midi), ("sfproxy", args.sfproxy_midi)):
        if path is not None:
            sources[label] = path
    for path in (args.reference_audio, args.sfz, args.sfizz_render, *sources.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    midis = {label: pretty_midi.PrettyMIDI(str(path)) for label, path in sources.items()}
    flat = pretty_midi.PrettyMIDI(str(args.reference_midi))
    for instrument in flat.instruments:
        for note in instrument.notes:
            note.velocity = 64
    midis["flat64"] = flat
    reference = notes_in_window(midis["reference"], args.start, args.duration)
    if not reference:
        raise ValueError("Chosen 20-second window has no score notes")
    note_sets = {"reference": reference}
    for label, midi in midis.items():
        if label != "reference":
            note_sets[label] = match_notes(reference, notes_in_window(midi, args.start, args.duration), label)
        clipped_midi(midi, args.start, args.duration).write(str(out / f"{label}.mid"))

    ffmpeg = args.ffmpeg
    subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", str(args.start),
                    "-i", str(args.reference_audio), "-t", "20", "-ac", "1",
                    "-ar", "22050", "-b:a", "128k", str(out / "reference.mp3")], check=True)
    for label in ("flat64", "diffsynth", "sfproxy"):
        if label not in midis:
            continue
        wav = out / f"{label}.wav"
        try:
            subprocess.run([str(args.sfizz_render), "--sfz", str(args.sfz), "--midi",
                            str(out / f"{label}.mid"), "--wav", str(wav),
                            "--samplerate", "44100", "--blocksize", "1024",
                            "--polyphony", "256", "--quality", "3"],
                           check=True, cwd=args.sfz.parent)
            subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(wav),
                            "-t", "20", "-ac", "1", "-ar", "22050", "-b:a", "128k",
                            str(out / f"{label}.mp3")], check=True)
        finally:
            wav.unlink(missing_ok=True)

    stats = {}
    if args.velocity_ground_truth:
        for label, notes in note_sets.items():
            if label != "reference":
                stats[label] = {"mae_20s": round(statistics.mean(
                    abs(a["v"] - b["v"]) for a, b in zip(reference, notes)), 3)}
    payload = {
        "dataset": args.dataset,
        "case_id": args.case_id,
        "title": args.title,
        "source_start_seconds": args.start,
        "duration_seconds": args.duration,
        "velocity_ground_truth": args.velocity_ground_truth,
        "soundfont": args.sfz.name,
        "notes": note_sets,
        "stats": stats,
    }
    (out / "notes.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {args.case_id}: {len(reference)} aligned notes over 20 seconds; methods={list(note_sets)}")


if __name__ == "__main__":
    main()
