"""Export a matched MAESTRO excerpt from saved predictions and renders.

Run on a machine with the MAESTRO files, model predictions, SoundFont, sfizz_render,
ffmpeg, and pretty_midi installed. This script does not perform model inference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pretty_midi


def notes_in_window(midi: pretty_midi.PrettyMIDI, start: float, duration: float):
    end = start + duration
    notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            if note.end <= start or note.start >= end:
                continue
            notes.append(
                {
                    "p": note.pitch,
                    "s": round(max(note.start, start) - start, 3),
                    "e": round(min(note.end, end) - start, 3),
                    "v": note.velocity,
                }
            )
    return sorted(notes, key=lambda n: (n["s"], n["p"]))


def clipped_midi(source: pretty_midi.PrettyMIDI, start: float, duration: float):
    result = pretty_midi.PrettyMIDI(initial_tempo=120)
    end = start + duration
    for instrument in source.instruments:
        target = pretty_midi.Instrument(instrument.program, instrument.is_drum, instrument.name)
        for note in instrument.notes:
            if note.end <= start or note.start >= end:
                continue
            target.notes.append(
                pretty_midi.Note(
                    note.velocity,
                    note.pitch,
                    max(note.start, start) - start,
                    min(note.end, end) - start,
                )
            )
        for cc in instrument.control_changes:
            if start <= cc.time < end:
                target.control_changes.append(
                    pretty_midi.ControlChange(cc.number, cc.value, cc.time - start)
                )
        if target.notes:
            result.instruments.append(target)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--synth-midi-root", type=Path, required=True)
    parser.add_argument("--synth-render-root", type=Path, required=True)
    parser.add_argument("--proxy-render-root", type=Path, required=True)
    parser.add_argument("--soundfont", type=Path, required=True)
    parser.add_argument("--sfizz-render", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--item-index", type=int, default=0)
    parser.add_argument("--start", type=float, default=60)
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    items = json.loads(args.manifest.read_text())["items"]
    item = items[args.item_index]
    key = item["key"]
    relative = Path(item["pred_midi"]).relative_to(
        Path(json.loads(args.manifest.read_text())["pred_midi_dir"])
    )
    stem = key.replace("/", "_")
    source_midi = args.dataset_root / (key + ".midi")
    source_audio = args.dataset_root / (key + ".wav")
    midi_paths = {
        "reference": source_midi,
        "diffsynth": args.synth_midi_root / relative,
        "sfproxy": Path(item["pred_midi"]),
    }
    wav_paths = {
        "reference": source_audio,
        "diffsynth": args.synth_render_root / stem / (stem + ".pred.wav"),
        "sfproxy": args.proxy_render_root / stem / (stem + ".pred.wav"),
    }
    for path in [*midi_paths.values(), *wav_paths.values(), args.soundfont, args.sfizz_render]:
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output.mkdir(parents=True, exist_ok=True)
    midis = {name: pretty_midi.PrettyMIDI(str(path)) for name, path in midi_paths.items()}
    flat = pretty_midi.PrettyMIDI(str(source_midi))
    for instrument in flat.instruments:
        for note in instrument.notes:
            note.velocity = 64
    flat_full_path = args.output / "flat_full.mid"
    flat.write(str(flat_full_path))
    flat_wav = args.output / "flat_full.wav"
    subprocess.run(
        [str(args.sfizz_render), "--sfz", str(args.soundfont), "--midi", str(flat_full_path),
         "--wav", str(flat_wav), "--samplerate", "44100", "--blocksize", "1024",
         "--polyphony", "256", "--quality", "3"],
        check=True,
        cwd=args.soundfont.parent,
    )
    midis["flat64"] = flat
    wav_paths["flat64"] = flat_wav
    for name, midi in midis.items():
        clipped_midi(midi, args.start, args.duration).write(str(args.output / f"{name}.mid"))
    note_data = {
        "dataset": "MAESTRO v3 test",
        "key": key,
        "start_seconds": args.start,
        "duration_seconds": args.duration,
        "soundfont": args.soundfont.name,
        "notes": {
            name: notes_in_window(midi, args.start, args.duration)
            for name, midi in midis.items()
        },
    }
    (args.output / "demo_notes.json").write_text(json.dumps(note_data, indent=2) + "\n")
    for name, source in wav_paths.items():
        subprocess.run(
            [args.ffmpeg, "-v", "error", "-y", "-ss", str(args.start), "-i", str(source),
             "-t", str(args.duration), "-ac", "1", "-ar", "22050", "-b:a", "128k",
             str(args.output / f"{name}.mp3")],
            check=True,
        )
    flat_full_path.unlink()
    flat_wav.unlink()
    print(f"Exported {key} at {args.start:g}-{args.start + args.duration:g}s to {args.output}")


if __name__ == "__main__":
    main()
