from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from direct_invension.common import (
    dump_json,
    ensure_repo_imports,
    extract_sorted_notes,
    resolve_dataset_split,
    write_flat_velocity_copy,
)

from tqdm.auto import tqdm


def _build_dataset_scan_helpers():
    ensure_repo_imports()
    from data_analysis.cli._dataset_utils import load_maestro_audio_map, resolve_real_audio, scan_midis  # type: ignore

    return scan_midis, load_maestro_audio_map, resolve_real_audio


def _relative_key(dataset_dir: Path, midi_path: Path) -> str:
    try:
        return str(midi_path.relative_to(dataset_dir).with_suffix(""))
    except Exception:
        return midi_path.stem


def _flat_out_path(out_dir: Path, dataset_dir: Path, midi_path: Path, *, flat_velocity: int) -> Path:
    try:
        rel = midi_path.relative_to(dataset_dir)
    except Exception:
        rel = Path(midi_path.name)
    return (out_dir / "pred_midis" / rel).with_suffix(f".flat{int(flat_velocity)}.mid")


def predict_flat_dataset(
    *,
    dataset_type: str,
    dataset_dir: str | Path,
    out_dir: str | Path,
    flat_velocity: int = 64,
    split: str = "test",
    maps_pianos: str = "both",
    skip_existing: bool = True,
    max_items: Optional[int] = None,
) -> Dict[str, object]:
    scan_midis, load_maestro_audio_map, resolve_real_audio = _build_dataset_scan_helpers()

    dataset_dir = Path(dataset_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_split = resolve_dataset_split(split)
    midi_files = scan_midis(dataset_type, dataset_dir, split=scan_split, maps_pianos=maps_pianos)
    if max_items is not None:
        midi_files = midi_files[: int(max_items)]
    maestro_audio_map = load_maestro_audio_map(dataset_type, dataset_dir, split=scan_split)

    label = f"flat_velocity_{int(flat_velocity)}"
    items: List[Dict[str, object]] = []
    file_summaries: List[Dict[str, object]] = []

    for midi_path in tqdm(
        midi_files,
        desc=f"Flat infer [{dataset_type}]",
        unit="file",
        dynamic_ncols=True,
    ):
        midi_path = Path(midi_path).resolve()
        real_audio = resolve_real_audio(dataset_type, dataset_dir, midi_path, maestro_audio_map)
        pred_midi_path = _flat_out_path(out_dir, dataset_dir, midi_path, flat_velocity=int(flat_velocity))
        skipped_existing = bool(skip_existing and pred_midi_path.exists())
        if not skipped_existing:
            write_flat_velocity_copy(midi_path, pred_midi_path, flat_velocity=int(flat_velocity))

        key = _relative_key(dataset_dir, midi_path)
        num_notes = len(extract_sorted_notes(midi_path))
        items.append(
            {
                "key": key,
                "label": label,
                "gt_midi": str(midi_path),
                "pred_midi": str(pred_midi_path),
                "real_audio": str(real_audio),
            }
        )
        file_summaries.append(
            {
                "key": key,
                "midi_path": str(midi_path),
                "pred_midi_path": str(pred_midi_path),
                "real_audio": str(real_audio),
                "flat_velocity": int(flat_velocity),
                "num_notes": int(num_notes),
                "skipped_existing": skipped_existing,
            }
        )

    manifest = {
        "format_version": 1,
        "label": label,
        "dataset_type": str(dataset_type),
        "dataset_dir": str(dataset_dir),
        "split": scan_split,
        "requested_split": split,
        "maps_pianos": maps_pianos,
        "flat_velocity": int(flat_velocity),
        "items": items,
    }
    manifest_path = dump_json(out_dir / f"flat{int(flat_velocity)}_manifest.json", manifest)
    file_summary_path = dump_json(out_dir / f"flat{int(flat_velocity)}_predictions.json", {"files": file_summaries})

    return {
        "dataset_type": dataset_type,
        "dataset_dir": str(dataset_dir),
        "num_items": len(items),
        "out_dir": str(out_dir),
        "pred_midi_dir": str(out_dir / "pred_midis"),
        "flat_velocity": int(flat_velocity),
        "manifest_path": str(manifest_path),
        "file_summary_path": str(file_summary_path),
    }



if __name__ == "__main__":
    raise SystemExit("flat_infer.py is an implementation module. Use flat_eval_job.py for the supported flat-velocity evaluation job.")
