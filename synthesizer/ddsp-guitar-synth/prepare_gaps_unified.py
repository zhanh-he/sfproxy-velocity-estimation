#!/usr/bin/env python3
"""Prepare GAPS DDSP-Guitar-Synth datasets under the repo contract."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from mido import MidiFile
from tqdm import tqdm


OPEN_STRING_PITCHES = (40, 45, 50, 55, 59, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare GAPS DDSP-Guitar-Synth datasets under the unified repo contract."
    )
    parser.add_argument("--gaps_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--frame_rate", type=float, default=100.0)
    parser.add_argument("--segment_seconds", type=float, default=10.0)
    parser.add_argument("--max_fret", type=int, default=24)
    return parser.parse_args()


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"id", "split", "audio_path", "midi_path"}
    if not rows:
        raise ValueError(f"No rows found in metadata file: {path}")
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Metadata file is missing columns: {sorted(missing)}")
    return rows


def normalize_split(raw: str) -> str:
    split = str(raw or "").strip().lower()
    if split in {"", "validate", "validation", "valid", "val"}:
        return "validation"
    if split in {"train", "test"}:
        return split
    raise ValueError(f"Unsupported split '{raw}'")


def load_midi_notes(path: Path) -> list[dict[str, float | int]]:
    midi = MidiFile(str(path))
    time_s = 0.0
    active: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
    notes: list[dict[str, float | int]] = []

    for msg in midi:
        time_s += float(msg.time)
        if msg.type == "note_on" and int(msg.velocity) > 0:
            key = (int(getattr(msg, "channel", 0)), int(msg.note))
            active[key].append((time_s, int(msg.velocity)))
            continue
        if msg.type not in {"note_off", "note_on"}:
            continue

        key = (int(getattr(msg, "channel", 0)), int(msg.note))
        if key not in active or not active[key]:
            continue
        onset_s, velocity = active[key].pop()
        if not active[key]:
            del active[key]
        if time_s <= onset_s:
            continue
        notes.append(
            {
                "onset_s": float(onset_s),
                "offset_s": float(time_s),
                "pitch": int(msg.note),
                "velocity": int(velocity),
            }
        )

    for (_, pitch), pending in active.items():
        for onset_s, velocity in pending:
            notes.append(
                {
                    "onset_s": float(onset_s),
                    "offset_s": float(max(time_s, onset_s + 1e-3)),
                    "pitch": int(pitch),
                    "velocity": int(velocity),
                }
            )

    notes.sort(key=lambda note: (float(note["onset_s"]), -int(note["pitch"]), float(note["offset_s"])))
    return notes


def load_audio_ffmpeg(path: Path, sample_rate: int) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg failed for {path}: {stderr or 'unknown error'}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"Decoded empty audio from {path}")
    return audio


def candidate_strings(pitch: int, max_fret: int) -> list[int]:
    return [
        string_idx
        for string_idx, open_pitch in enumerate(OPEN_STRING_PITCHES)
        if open_pitch <= pitch <= open_pitch + max_fret
    ]


def build_conditioning(
    notes: list[dict[str, float | int]],
    *,
    frame_rate: float,
    frames_total: int,
    max_fret: int,
) -> np.ndarray:
    conditioning = np.zeros((frames_total, 6, 2), dtype=np.float32)
    active_until = np.zeros(6, dtype=np.int64)
    notes_by_onset: dict[int, list[dict[str, int]]] = defaultdict(list)

    for note in notes:
        onset = int(round(float(note["onset_s"]) * frame_rate))
        if onset >= frames_total:
            continue
        offset = int(round(float(note["offset_s"]) * frame_rate))
        offset = max(onset + 1, min(frames_total, offset))
        notes_by_onset[onset].append(
            {
                "onset": onset,
                "offset": offset,
                "pitch": int(note["pitch"]),
                "velocity": int(note["velocity"]),
            }
        )

    for onset in sorted(notes_by_onset):
        used_strings: set[int] = set()
        group = sorted(notes_by_onset[onset], key=lambda note: (-note["pitch"], -note["offset"]))
        for note in group:
            candidates = [idx for idx in candidate_strings(note["pitch"], max_fret) if idx not in used_strings]
            if not candidates:
                continue
            free_candidates = [idx for idx in candidates if active_until[idx] <= onset]
            string_idx = max(free_candidates) if free_candidates else min(candidates, key=lambda idx: (active_until[idx], -idx))
            conditioning[note["onset"] : note["offset"], string_idx, 0] = float(note["pitch"])
            conditioning[note["onset"], string_idx, 1] = float(note["velocity"])
            active_until[string_idx] = note["offset"]
            used_strings.add(string_idx)

    return conditioning


def compute_starts(total_length: int, chunk_size: int, hop_size: int) -> list[int]:
    if total_length < chunk_size:
        return []
    return list(range(0, total_length - chunk_size + 1, hop_size))


def chunk_track(audio: np.ndarray, conditioning: np.ndarray, *, sample_rate: int, frame_rate: float, segment_seconds: float):
    audio_chunk_size = int(round(segment_seconds * sample_rate))
    frame_chunk_size = int(round(segment_seconds * frame_rate))
    if audio_chunk_size <= 0 or frame_chunk_size <= 0:
        raise ValueError("segment_seconds must be positive")

    audio_starts = compute_starts(audio.shape[0], audio_chunk_size, audio_chunk_size)
    frame_starts = compute_starts(conditioning.shape[0], frame_chunk_size, frame_chunk_size)
    item_count = min(len(audio_starts), len(frame_starts))
    if item_count <= 0:
        return None

    audio_chunked = np.stack(
        [audio[start : start + audio_chunk_size] for start in audio_starts[:item_count]],
        axis=0,
    )
    conditioning_chunked = np.stack(
        [conditioning[start : start + frame_chunk_size] for start in frame_starts[:item_count]],
        axis=0,
    )
    return conditioning_chunked.astype(np.float32), audio_chunked.astype(np.float32)


def build_split_dataset(
    rows: list[dict[str, str]],
    *,
    root: Path,
    sample_rate: int,
    frame_rate: float,
    segment_seconds: float,
    max_fret: int,
    split_name: str,
):
    conditioning_items: list[np.ndarray] = []
    audio_items: list[np.ndarray] = []
    track_count = 0

    for row in tqdm(rows, total=len(rows), desc=f"Preparing {split_name}", unit="track"):
        audio_path = root / row["audio_path"]
        midi_path = root / row["midi_path"]
        audio = load_audio_ffmpeg(audio_path, sample_rate)
        frames_total = max(1, int(round(audio.shape[0] * float(frame_rate) / float(sample_rate))))
        conditioning = build_conditioning(
            load_midi_notes(midi_path),
            frame_rate=frame_rate,
            frames_total=frames_total,
            max_fret=max_fret,
        )
        chunked = chunk_track(
            audio,
            conditioning,
            sample_rate=sample_rate,
            frame_rate=frame_rate,
            segment_seconds=segment_seconds,
        )
        if chunked is None:
            continue
        conditioning_chunked, audio_chunked = chunked
        conditioning_items.append(conditioning_chunked)
        audio_items.append(audio_chunked)
        track_count += 1

    if conditioning_items:
        conditioning = np.concatenate(conditioning_items, axis=0)
        audio = np.concatenate(audio_items, axis=0)
    else:
        conditioning = np.zeros(
            (0, int(round(segment_seconds * frame_rate)), 6, 2),
            dtype=np.float32,
        )
        audio = np.zeros((0, int(round(segment_seconds * sample_rate))), dtype=np.float32)

    data = {
        "conditioning": conditioning,
        "mic_audio": audio,
        "mix_audio": audio,
    }
    stats = {
        "track_count": int(track_count),
        "item_count": int(conditioning.shape[0]),
    }
    return data, stats


def save_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)


def main() -> None:
    args = parse_args()
    root = args.gaps_path.expanduser().resolve()
    rows = read_metadata(root / "gaps_metadata_with_splits.csv")
    train_rows = [row for row in rows if normalize_split(row.get("split", "")) == "train"]
    val_rows = [row for row in rows if normalize_split(row.get("split", "")) == "validation"]

    train_data, train_stats = build_split_dataset(
        train_rows,
        root=root,
        sample_rate=args.sample_rate,
        frame_rate=args.frame_rate,
        segment_seconds=args.segment_seconds,
        max_fret=args.max_fret,
        split_name="train",
    )
    val_data, val_stats = build_split_dataset(
        val_rows,
        root=root,
        sample_rate=args.sample_rate,
        frame_rate=args.frame_rate,
        segment_seconds=args.segment_seconds,
        max_fret=args.max_fret,
        split_name="validation",
    )

    seg_tag = str(args.segment_seconds).replace(".0", "")
    train_path = args.output_dir / f"train_gaps_midi_{seg_tag}s.npz"
    val_path = args.output_dir / f"val_gaps_midi_{seg_tag}s.npz"
    save_npz(train_path, train_data)
    save_npz(val_path, val_data)

    meta = {
        "dataset": "GAPS",
        "gaps_path": str(root),
        "sample_rate": int(args.sample_rate),
        "frame_rate": float(args.frame_rate),
        "segment_seconds": float(args.segment_seconds),
        "max_fret": int(args.max_fret),
        "train_dataset_path": str(train_path),
        "val_dataset_path": str(val_path),
        "train_track_count": train_stats["track_count"],
        "train_item_count": train_stats["item_count"],
        "val_track_count": val_stats["track_count"],
        "val_item_count": val_stats["item_count"],
    }
    meta_path = args.output_dir / "prepare_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
