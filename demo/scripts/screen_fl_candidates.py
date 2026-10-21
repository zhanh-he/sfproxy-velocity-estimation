#!/usr/bin/env python3
"""Screen FL HDF5 performances for 20 s velocity-distribution anomalies.

Run in the Score-HPT research environment on the machine holding the original
HDF5 cache and the recovered 5 s guitar checkpoints. Each model is loaded once.
This uses the same onset-assisted HPT configuration and onset-frame velocity
selection as scripts/infer_compare.py. It does not evaluate against invented
guitar velocity labels or replace the paper's full-dataset metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("diffsynth", "sfproxy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": round(float(np.mean(values)), 3),
        f"{prefix}_std": round(float(np.std(values)), 3),
        f"{prefix}_p10": round(float(np.percentile(values, 10)), 3),
        f"{prefix}_p90": round(float(np.percentile(values, 90)), 3),
        f"{prefix}_at_102": round(float(np.mean(values == 102)), 4),
        f"{prefix}_ge_100": round(float(np.mean(values >= 100)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-dir", type=Path, required=True)
    parser.add_argument("--diffsynth-ckpt", type=Path, required=True)
    parser.add_argument("--sfproxy-ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--piece", action="append", help="Limit to these HDF5 stems for a smoke test")
    parser.add_argument("--min-notes", type=int, default=50)
    parser.add_argument("--audio-gain", type=float, default=1.0,
                        help="Diagnostic input gain; use 1.0 for authentic demo screening")
    args = parser.parse_args()
    if not 0 < args.audio_gain <= 1:
        parser.error("--audio-gain must be in (0, 1]; published demos use 1.0")

    sys.path.insert(0, str(ROOT / "score_hpt/pytorch"))
    from direct_invension.common import compose_cfg
    from inference import VeloTranscription
    from utilities import int16_to_float32, original_score_events

    cfg = compose_cfg([
        "model.type=hpt", "score_informed.method=note_editor",
        "model.input2=onset", "model.input3=null", "exp.cuda=true",
    ], job_name="screen_fl_candidates")
    weights = {"diffsynth": args.diffsynth_ckpt.resolve(), "sfproxy": args.sfproxy_ckpt.resolve()}
    for label, path in weights.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
    models = {label: VeloTranscription(checkpoint_path=str(path), cfg=cfg) for label, path in weights.items()}
    names = set(args.piece or ())
    files = sorted(args.h5_dir.expanduser().glob("*.h5"))
    if names:
        files = [path for path in files if path.stem in names]
        if {path.stem for path in files} != names:
            raise FileNotFoundError(f"Unfound pieces: {sorted(names - {path.stem for path in files})}")
    if not files:
        raise ValueError("No FL HDF5 files found")

    fps = int(cfg.feature.frames_per_second)
    sr = int(cfg.feature.sample_rate)
    begin = int(cfg.feature.begin_note)
    classes = int(cfg.feature.classes_num)
    scale = int(cfg.feature.velocity_scale)
    rows: list[dict] = []
    for item_no, path in enumerate(files, 1):
        with h5py.File(path, "r") as cache:
            waveform = np.asarray(cache["waveform"], dtype=np.int16)
            times = np.asarray(cache["midi_event_time"], dtype=float)
            events = [event.decode("utf-8") if isinstance(event, bytes) else str(event)
                      for event in cache["midi_event"][:]]
        duration = len(waveform) / sr
        notes, _ = original_score_events(cfg, times, events, duration)
        if not notes:
            continue
        audio = int16_to_float32(waveform) * args.audio_gain
        onset_roll = np.zeros((int(round(duration * fps)) + 1, classes), dtype=np.float32)
        onset_times = np.asarray([float(note["onset_time"]) for note in notes])
        pitches = np.asarray([int(note["midi_note"]) for note in notes])
        indices = np.rint(onset_times * fps).astype(int)
        pitch_indices = pitches - begin
        valid = (pitch_indices >= 0) & (pitch_indices < classes) & (onset_times < duration)
        onset_roll[np.minimum(indices[valid], len(onset_roll) - 1), pitch_indices[valid]] = 1.0

        velocities = {}
        for label, model in models.items():
            roll = np.asarray(model.transcribe(audio, input2=onset_roll)["output_dict"]["velocity_output"])
            picked = np.asarray([roll[min(indices[i], len(roll) - 1), pitch_indices[i]]
                                 if valid[i] else float(note.get("velocity", 64)) / scale
                                 for i, note in enumerate(notes)])
            velocities[label] = np.clip(np.rint(picked * scale), 1, 127).astype(np.int16)

        for start in range(0, int(duration - 20) + 1, 20):
            mask = (onset_times >= start) & (onset_times < start + 20)
            count = int(np.count_nonzero(mask))
            if count < args.min_notes:
                continue
            segment = waveform[start * sr:(start + 20) * sr].astype(np.float64)
            row = {"piece": path.stem, "start_s": start, "end_s": start + 20,
                   "duration_s": round(duration, 3), "notes": count,
                   "audio_rms_int16": round(float(np.sqrt(np.mean(segment * segment))), 3)}
            for label in METHODS:
                row.update(summarize(velocities[label][mask], label))
            rows.append(row)
        print(f"{item_no}/{len(files)} {path.stem}: {len(notes)} notes, {len(rows)} windows", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    args.out.with_suffix(".meta.json").write_text(json.dumps({
        "source": str(args.h5_dir.resolve()), "pieces": len(files), "windows": len(rows),
        "model": "HPT + onset-assisted note_editor", "velocity_method": "onset_only",
        "audio_gain": args.audio_gain,
        "sample_rate": sr, "frames_per_second": fps,
        "weights": {label: {"path": str(path), "sha256": sha256(path)} for label, path in weights.items()},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
