#!/usr/bin/env python3
"""Compare the released VeloEst checkpoints on one aligned audio/MIDI pair.

The score supplies note pitch and timing. Only note velocities are changed in
the output MIDIs; programs, tempo, and control changes are preserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = {
    "veloest": ROOT / "score_hpt/checkpoints/veloest_onset_only_120k.pth",
    "diffsfproxy": ROOT / "score_hpt/checkpoints/veloest_diffsfproxy_piano_5s_120k.pth",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, required=True, help="Performance audio (WAV, FLAC, or MP3).")
    p.add_argument("--midi", type=Path, required=True, help="Score aligned to the audio.")
    p.add_argument("--out", type=Path, required=True, help="Directory for predictions and figures.")
    p.add_argument("--veloest-ckpt", type=Path, default=DEFAULT_WEIGHTS["veloest"])
    p.add_argument("--diffsynth-ckpt", type=Path, help="Optional adapted Diff-Synth VeloEst checkpoint.")
    p.add_argument("--diffsfproxy-ckpt", type=Path, default=DEFAULT_WEIGHTS["diffsfproxy"])
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--velocity-method", choices=("onset_only", "max_frame"), default="onset_only")
    p.add_argument("--reference-velocities", action="store_true", help="The input MIDI has true note velocities; report MAE.")
    p.add_argument("--plot-start", type=float, default=0.0, help="Start time of the SVG piano-roll excerpt in seconds.")
    p.add_argument("--plot-seconds", type=float, default=20.0, help="Length of the SVG excerpt in seconds.")
    p.add_argument("--sfz", type=Path, help="Optional complete SFZ instrument for rendering all three output MIDIs.")
    p.add_argument("--sfizz-render", help="sfizz_render executable; otherwise use SFIZZ_RENDER_BIN or PATH.")
    p.add_argument("--ffmpeg", default="ffmpeg", help="Used to encode MP3 when --sfz is supplied.")
    return p


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def velocity_color(value: int) -> str:
    stops = [(1, (85, 49, 133)), (32, (153, 58, 138)), (64, (218, 88, 102)),
             (96, (247, 160, 76)), (127, (255, 224, 119))]
    value = max(1, min(127, int(value)))
    for (low, a), (high, b) in zip(stops, stops[1:]):
        if value <= high:
            t = (value - low) / (high - low)
            return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))
    return "#ffe077"


def make_svg(rows: list[dict], out: Path, start: float, seconds: float, reference: bool, metrics: dict) -> None:
    """Slide-sized piano-roll comparison with a common pitch/time scale."""
    end = start + seconds
    visible = [r for r in rows if r["start_s"] < end and r["end_s"] > start]
    width, left, right = 1600, 172, 58
    panel_top, panel_height, panel_gap = 130, 177, 33
    lo = max(21, min((r["pitch"] for r in visible), default=48) - 2)
    hi = min(108, max((r["pitch"] for r in visible), default=84) + 2)
    pitch_range = max(1, hi - lo + 1)
    plot_width = width - left - right
    panels = [("Flat 64", "flat64", "#a8b6c5"),
              ("VeloEst", "veloest", "#e8c779")]
    if "diffsynth" in rows[0]:
        panels.append(("VeloEst + Diff-Synth", "diffsynth", "#ec876c"))
    panels.append(("VeloEst + Diff-SFProxy", "diffsfproxy", "#69d7c3"))
    shift = (len(panels) - 3) * (panel_height + panel_gap)
    height = 860 + shift
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Aligned MIDI velocities from {len(panels)} approaches">',
             '<rect width="100%" height="100%" fill="#0b1421"/>',
             '<text x="58" y="58" fill="#f7f2e7" font-family="Arial,sans-serif" font-size="31" font-weight="700">Same notes, different velocities</text>',
             f'<text x="58" y="88" fill="#aabacc" font-family="Arial,sans-serif" font-size="17">{len(visible)} notes · {start:g}–{end:g} s · identical pitch and onset across methods</text>']
    for index, (title, field, accent) in enumerate(panels):
        top = panel_top + index * (panel_height + panel_gap)
        parts.append(f'<rect x="48" y="{top}" width="1504" height="{panel_height}" rx="13" fill="#152237" stroke="#31435a"/>')
        parts.append(f'<text x="67" y="{top+39}" fill="{accent}" font-family="Arial,sans-serif" font-size="21" font-weight="700">{html.escape(title)}</text>')
        stat = metrics.get(field, {})
        label = f'MAE {stat["mae"]:.1f}' if reference and stat.get("mae") is not None else f'mean {stat["mean"]:.1f}'
        parts.append(f'<text x="67" y="{top+66}" fill="#aabacc" font-family="Arial,sans-serif" font-size="15">{label} / 127</text>')
        for tick in range(math.ceil(start / 5) * 5, math.floor(end / 5) * 5 + 1, 5):
            x = left + (tick - start) / seconds * plot_width
            parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top+18}" y2="{top+149}" stroke="#31435a" stroke-width="1"/>')
            if index == len(panels) - 1:
                parts.append(f'<text x="{x-8:.1f}" y="{top+167}" fill="#92a3b6" font-family="Arial,sans-serif" font-size="13">{tick:g}s</text>')
        for pitch in range(math.ceil(lo / 12) * 12, hi + 1, 12):
            y = top + 23 + (hi - pitch) / pitch_range * 125
            parts.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#263a50"/>')
        for row in visible:
            x0 = left + (max(start, row["start_s"]) - start) / seconds * plot_width
            x1 = left + (min(end, row["end_s"]) - start) / seconds * plot_width
            y = top + 21 + (hi - row["pitch"]) / pitch_range * 125
            color = velocity_color(row[field])
            parts.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(2, x1-x0):.1f}" height="5.5" rx="1.5" fill="{color}"/>')
    parts.append(f'<text x="58" y="{804+shift}" fill="#aabacc" font-family="Arial,sans-serif" font-size="15">Velocity 1–127</text>')
    for value in (1, 32, 64, 96, 127):
        x = 215 + (value - 1) / 126 * 485
        parts.append(f'<rect x="{x:.1f}" y="{786+shift}" width="44" height="14" rx="3" fill="{velocity_color(value)}"/>')
        parts.append(f'<text x="{x+12:.1f}" y="{823+shift}" fill="#aabacc" font-family="Arial,sans-serif" font-size="13">{value}</text>')
    parts.append(f'<text x="58" y="{849+shift}" fill="#7f96aa" font-family="Arial,sans-serif" font-size="13">Bars show note pitch, onset, duration, and predicted velocity. MAE appears only when the input MIDI is declared ground truth.</text>')
    parts.append('</svg>')
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def renderer_binary(explicit: str | None) -> str:
    binary = explicit or os.environ.get("SFIZZ_RENDER_BIN") or shutil.which("sfizz_render") or shutil.which("sfizz-render")
    if not binary:
        raise FileNotFoundError("sfizz_render not found. Set --sfizz-render or SFIZZ_RENDER_BIN.")
    return str(Path(binary).expanduser().resolve())


def render_audio(sfz: Path, binary: str, ffmpeg: str, out: Path, labels: tuple[str, ...]) -> None:
    sfz = sfz.expanduser().resolve()
    if not sfz.is_file():
        raise FileNotFoundError(f"SFZ instrument not found: {sfz}")
    if not shutil.which(ffmpeg) and not Path(ffmpeg).is_file():
        raise FileNotFoundError(f"ffmpeg not found: {ffmpeg}")
    for label in labels:
        wav = out / f"{label}.wav"
        subprocess.run([binary, "--sfz", str(sfz), "--midi", str(out / f"{label}.mid"),
                        "--wav", str(wav), "--samplerate", "44100", "--blocksize", "1024",
                        "--polyphony", "256", "--quality", "3"], check=True, cwd=sfz.parent)
        subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(wav), "-ac", "1",
                        "-ar", "22050", "-b:a", "128k", str(out / f"{label}.mp3")], check=True)


def run(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT / "score_hpt/pytorch"))
    from direct_invension.common import compose_cfg, extract_sorted_notes, replace_note_velocities
    from inference import VeloTranscription
    from utilities import load_mono_audio

    audio_path, midi_path = args.audio.expanduser().resolve(), args.midi.expanduser().resolve()
    weights = {"veloest": args.veloest_ckpt.expanduser().resolve()}
    if args.diffsynth_ckpt:
        weights["diffsynth"] = args.diffsynth_ckpt.expanduser().resolve()
    weights["diffsfproxy"] = args.diffsfproxy_ckpt.expanduser().resolve()
    for path in (audio_path, midi_path, *weights.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.plot_start < 0 or args.plot_seconds <= 0:
        raise ValueError("--plot-start must be >= 0 and --plot-seconds must be > 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    cfg = compose_cfg(["model.type=hpt", "score_informed.method=note_editor",
                       "model.input2=onset", "model.input3=null",
                       f"exp.cuda={'true' if use_cuda else 'false'}"], job_name="infer_compare")
    notes = extract_sorted_notes(midi_path)
    if not notes:
        raise ValueError("Input MIDI has no notes")
    audio = load_mono_audio(audio_path, sample_rate=int(cfg.feature.sample_rate))
    duration = len(audio) / int(cfg.feature.sample_rate)
    if duration <= 0:
        raise ValueError("Input audio is empty")
    if max(note.onset for note in notes) > duration + 0.05:
        raise ValueError("MIDI contains note onsets after the audio ends; supply an aligned pair")

    fps = int(cfg.feature.frames_per_second)
    begin = int(cfg.feature.begin_note)
    classes = int(cfg.feature.classes_num)
    onset_roll = np.zeros((int(round(duration * fps)) + 1, classes), dtype=np.float32)
    for note in notes:
        pitch_index = note.pitch - begin
        if 0 <= pitch_index < classes and note.onset < duration:
            onset_roll[min(int(round(note.onset * fps)), len(onset_roll)-1), pitch_index] = 1.0

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    predictions = {}
    for label, path in weights.items():
        model = VeloTranscription(checkpoint_path=str(path), cfg=cfg)
        roll = np.asarray(model.transcribe(audio, input2=onset_roll)["output_dict"]["velocity_output"])
        velocities = []
        for note in notes:
            pitch_index = note.pitch - begin
            frame = min(int(round(note.onset * fps)), len(roll)-1)
            if not (0 <= pitch_index < classes) or note.onset >= duration:
                velocities.append(note.velocity)
                continue
            if args.velocity_method == "max_frame":
                stop = min(len(roll), max(frame + 1, int(round(note.offset * fps))))
                value = float(np.max(roll[frame:stop, pitch_index]))
            else:
                value = float(roll[frame, pitch_index])
            velocities.append(int(np.clip(round(value * int(cfg.feature.velocity_scale)), 1, 127)))
        predictions[label] = velocities
        replace_note_velocities(midi_path, velocities, out / f"{label}.mid")
        del model
        if use_cuda:
            torch.cuda.empty_cache()

    flat = [64] * len(notes)
    replace_note_velocities(midi_path, flat, out / "flat64.mid")
    rows = [{"start_s": round(n.onset, 6), "end_s": round(n.offset, 6), "pitch": n.pitch,
             "input_velocity": n.velocity, "flat64": 64,
             **{label: values[i] for label, values in predictions.items()}}
            for i, n in enumerate(notes)]
    with (out / "notes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metrics = {}
    for label, values in (("flat64", flat), *predictions.items()):
        metrics[label] = {"mean": round(float(np.mean(values)), 3)}
        if args.reference_velocities:
            metrics[label]["mae"] = round(float(np.mean([abs(v - n.velocity) for v, n in zip(values, notes)])), 3)
    make_svg(rows, out / "comparison.svg", args.plot_start, args.plot_seconds,
             args.reference_velocities, metrics)
    summary = {
        "audio": str(audio_path), "midi": str(midi_path), "duration_seconds": round(duration, 3),
        "notes": len(notes), "device": "cuda" if use_cuda else "cpu",
        "model": "HPT + onset-assisted note_editor", "velocity_method": args.velocity_method,
        "reference_velocities": args.reference_velocities,
        "checkpoints": {k: {"path": str(v), "sha256": file_hash(v)} for k, v in weights.items()},
        "metrics": metrics,
        "outputs": {name: name for name in ("flat64.mid", *(f"{label}.mid" for label in weights), "notes.csv", "comparison.svg")},
    }
    if args.sfz:
        labels = ("flat64", *weights)
        render_audio(args.sfz, renderer_binary(args.sfizz_render), args.ffmpeg, out, labels)
        summary["rendered_sfz"] = str(args.sfz.expanduser().resolve())
        summary["outputs"].update({f"{name}.{ext}": f"{name}.{ext}" for name in
                                   labels for ext in ("wav", "mp3")})
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
