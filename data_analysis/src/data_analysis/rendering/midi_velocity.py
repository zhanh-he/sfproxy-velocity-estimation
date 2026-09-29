from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class MidiVelocityFlattenResult:
    in_midi: Path
    out_midi: Path
    flat_velocity: int
    notes_modified: int


def write_flat_velocity_midi(
    in_midi: str | Path,
    out_midi: str | Path,
    *,
    flat_velocity: int = 64,
    channels: Optional[Sequence[int]] = None,
) -> MidiVelocityFlattenResult:
    """Create a copy of a MIDI file with NOTE_ON velocities flattened.

    Rules:
      - Only NOTE_ON with velocity > 0 are replaced.
      - NOTE_ON with velocity == 0 (used as NOTE_OFF in some files) is kept.

    Args:
        in_midi: input .mid/.midi path
        out_midi: output midi path
        flat_velocity: value in [1, 127]
        channels: if provided, only modify these MIDI channels.

    Returns:
        MidiVelocityFlattenResult
    """
    from mido import MidiFile

    in_midi = Path(in_midi)
    out_midi = Path(out_midi)
    if not in_midi.exists():
        raise FileNotFoundError(f"MIDI not found: {in_midi}")

    if flat_velocity <= 0 or flat_velocity > 127:
        raise ValueError("flat_velocity must be in [1, 127]")

    ch_set = set(int(c) for c in channels) if channels is not None else None

    mid = MidiFile(str(in_midi))
    notes_modified = 0

    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on":
                if msg.velocity and msg.velocity > 0:
                    if ch_set is None or getattr(msg, "channel", None) in ch_set:
                        msg.velocity = int(flat_velocity)
                        notes_modified += 1

    out_midi.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(out_midi))

    return MidiVelocityFlattenResult(
        in_midi=in_midi,
        out_midi=out_midi,
        flat_velocity=int(flat_velocity),
        notes_modified=int(notes_modified),
    )
