from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_repo_imports() -> None:
    candidates = [
        repo_root(),
        repo_root() / "score_hpt",
        repo_root() / "data_analysis" / "src",
        repo_root() / "synth-proxy" / "src",
    ]
    for path in candidates:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def slugify(text: str, *, max_len: int = 120) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    token = token.strip("._-") or "item"
    return token[:max_len]


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def dump_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    return path


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_dataset_type(dataset_type: str) -> str:
    return str(dataset_type).strip().lower()


def resolve_dataset_split(split_or_scope: str) -> str:
    token = str(split_or_scope or "test").strip().lower()
    if token == "full":
        return "all"
    if token == "valid":
        return "validation"
    return token or "test"


def resolve_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (repo_root() / "score_hpt" / path).resolve()
    return path


def require_path(value: Any, *, field_name: str) -> Path:
    token = str(value).strip()
    if not token:
        raise ValueError(f"{field_name} must be set.")
    return resolve_path(token)


def compose_cfg(overrides: list[str], *, job_name: str):
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(config_path="../config", job_name=job_name, version_base=None):
        return compose(config_name="config", overrides=overrides)


def resolve_dataset_dir(cfg, dataset_type: str) -> Path:
    field_name = f"{dataset_type}_dir"
    if not hasattr(cfg.dataset, field_name):
        raise KeyError(f"dataset.{field_name} is not defined in config.yaml")
    return require_path(getattr(cfg.dataset, field_name), field_name=f"dataset.{field_name}")


def validate_hop_contract(*, fps: float, hop_size: int, route_name: str) -> None:
    if abs(float(fps) - 100.0) < 1e-6 and int(hop_size) != 221:
        raise ValueError(
            f"{route_name} expects backend.supervision.hop_size=221 when fps=100, got hop_size={hop_size}."
        )


@dataclass(frozen=True)
class SortedMidiNote:
    onset: float
    offset: float
    pitch: int
    velocity: int
    instrument_index: int
    note_index: int


def _import_pretty_midi():
    try:
        import pretty_midi  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "pretty_midi is required for MIDI velocity manipulation. Install it with `pip install pretty_midi`."
        ) from exc
    return pretty_midi


def load_pretty_midi(midi_path: str | Path):
    pretty_midi = _import_pretty_midi()
    midi_path = Path(midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI not found: {midi_path}")
    return pretty_midi.PrettyMIDI(str(midi_path))


def _sorted_note_refs(pm) -> List[Tuple[Tuple[float, int, float, int, int], object]]:
    refs: List[Tuple[Tuple[float, int, float, int, int], object]] = []
    for inst_idx, inst in enumerate(pm.instruments):
        for note_idx, note in enumerate(inst.notes):
            key = (
                round(float(note.start), 6),
                int(note.pitch),
                round(float(note.end), 6),
                int(inst_idx),
                int(note_idx),
            )
            refs.append((key, note))
    refs.sort(key=lambda item: item[0])
    return refs


def extract_sorted_notes(midi_path: str | Path) -> List[SortedMidiNote]:
    pm = load_pretty_midi(midi_path)
    out: List[SortedMidiNote] = []
    for key, note in _sorted_note_refs(pm):
        out.append(
            SortedMidiNote(
                onset=float(note.start),
                offset=float(note.end),
                pitch=int(note.pitch),
                velocity=int(note.velocity),
                instrument_index=int(key[3]),
                note_index=int(key[4]),
            )
        )
    return out


def replace_note_velocities(
    in_midi: str | Path,
    velocities: Sequence[int] | np.ndarray,
    out_midi: str | Path,
) -> Path:
    pm = load_pretty_midi(in_midi)
    refs = _sorted_note_refs(pm)
    vel = np.asarray(velocities, dtype=np.float64).reshape(-1)
    if vel.size != len(refs):
        raise ValueError(
            f"Velocity count mismatch: got {vel.size}, expected {len(refs)} notes from {in_midi}"
        )
    for (_, note), value in zip(refs, vel):
        note.velocity = int(np.clip(round(float(value)), 1, 127))

    out_midi = Path(out_midi)
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_midi))
    return out_midi


def write_flat_velocity_copy(
    in_midi: str | Path,
    out_midi: str | Path,
    *,
    flat_velocity: int = 64,
) -> Path:
    notes = extract_sorted_notes(in_midi)
    velocity = int(np.clip(round(flat_velocity), 1, 127))
    velocities = np.full(len(notes), velocity, dtype=np.int64)
    return replace_note_velocities(in_midi, velocities, out_midi)
