#!/usr/bin/env python3
"""Update the listening demo from the two released VeloEst checkpoints.

Pass full-piece MIDIs produced by scripts/infer_compare.py. The existing
demo_notes.json defines the exact excerpt and matched score. Either checkpoint
MIDI can be updated separately by providing only its corresponding argument.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pretty_midi

from export_maestro_demo import clipped_midi, notes_in_window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--veloest-midi", type=Path)
    parser.add_argument("--diffsfproxy-midi", type=Path)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--sfz", type=Path, required=True)
    parser.add_argument("--sfizz-render", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    sources = {key: value for key, value in
               (("veloest", args.veloest_midi), ("sfproxy", args.diffsfproxy_midi)) if value is not None}
    if not sources:
        parser.error("Provide --veloest-midi, --diffsfproxy-midi, or both")

    assets = args.assets.expanduser().resolve()
    notes_path = assets / "demo_notes.json"
    payload = json.loads(notes_path.read_text())
    start = float(payload["start_seconds"])
    duration = float(payload["duration_seconds"])
    reference = payload["notes"]["reference"]
    sfz = args.sfz.expanduser().resolve()
    renderer = args.sfizz_render.expanduser().resolve()
    for label, midi_path in sources.items():
        midi_path = midi_path.expanduser().resolve()
        source = pretty_midi.PrettyMIDI(str(midi_path))
        predicted = notes_in_window(source, start, duration)
        if len(predicted) != len(reference):
            raise ValueError(f"{label} note count differs from reference: {len(predicted)} vs {len(reference)}")
        # Reconstructed HDF5 MIDI may differ by a few milliseconds. Match by
        # pitch and nearest onset, then use reference coordinates in the chart.
        remaining = list(predicted)
        aligned = []
        for index, actual in enumerate(reference):
            candidates = [item for item in remaining if item["p"] == actual["p"]]
            if not candidates:
                raise ValueError(f"No {label} note matches reference index {index}: {actual}")
            estimate = min(candidates, key=lambda item: abs(item["s"] - actual["s"]) + abs(item["e"] - actual["e"]))
            if abs(estimate["s"] - actual["s"]) > 0.01 or abs(estimate["e"] - actual["e"]) > 0.01:
                raise ValueError(f"{label} note timing differs at index {index}: {actual} vs {estimate}")
            remaining.remove(estimate)
            aligned.append({"p": actual["p"], "s": actual["s"], "e": actual["e"], "v": estimate["v"]})

        clipped_midi(source, start, duration).write(str(assets / f"{label}.mid"))
        wav = assets / f"{label}_full.wav"
        try:
            subprocess.run([str(renderer), "--sfz", str(sfz), "--midi", str(midi_path),
                            "--wav", str(wav), "--samplerate", "44100", "--blocksize", "1024",
                            "--polyphony", "256", "--quality", "3"], check=True, cwd=sfz.parent)
            subprocess.run([args.ffmpeg, "-v", "error", "-y", "-ss", str(start), "-i", str(wav),
                            "-t", str(duration), "-ac", "1", "-ar", "22050", "-b:a", "128k",
                            str(assets / f"{label}.mp3")], check=True)
        finally:
            wav.unlink(missing_ok=True)
        payload["notes"][label] = aligned
        print(f"Updated {label}: {len(aligned)} aligned notes and {duration:g} s audio")

    notes_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
