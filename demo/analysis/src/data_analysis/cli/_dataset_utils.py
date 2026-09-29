from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


SMD_EXCLUDED = {"Beethoven_WoO080_001_20081107-SMD"}
FRANCOISLEDUC_ALIASES = {"francoisleduc"}
GAPS_ALIASES = {"gaps"}


def _normalize_split(split: str) -> str:
    split = (split or "all").lower()
    if split == "valid":
        split = "validation"
    if split in ("all", "train", "validation", "test"):
        return split
    raise ValueError(f"Unsupported split: {split}")


def _normalize_maps_pianos(maps_pianos: str) -> List[str]:
    token = (maps_pianos or "both").lower()
    if token in ("both", "all"):
        return ["ENSTDkCl", "ENSTDkAm"]
    if token in ("enstdkcl", "cl"):
        return ["ENSTDkCl"]
    if token in ("enstdkam", "am"):
        return ["ENSTDkAm"]
    raise ValueError(f"Unsupported maps_pianos: {maps_pianos}")


def normalize_dataset_type(dataset_type: str) -> str:
    dataset_type = (dataset_type or "").strip().lower()
    if dataset_type in FRANCOISLEDUC_ALIASES:
        return "francoisleduc"
    if dataset_type in GAPS_ALIASES:
        return "gaps"
    return dataset_type


def _resolve_maestro_root_and_csv(dataset_dir: Path) -> tuple[Path, Path]:
    direct_csv = dataset_dir / "maestro-v3.0.0.csv"
    if direct_csv.exists():
        return dataset_dir, direct_csv
    nested_csv = dataset_dir / "MAESTRO_v3.0.0" / "maestro-v3.0.0.csv"
    if nested_csv.exists():
        return nested_csv.parent, nested_csv
    for p in dataset_dir.rglob("maestro-v3.0.0.csv"):
        return p.parent, p
    raise FileNotFoundError(
        f"MAESTRO metadata not found under: {dataset_dir}. "
        "Expected maestro-v3.0.0.csv"
    )


def _read_francoisleduc_metadata(dataset_dir: Path) -> List[Dict[str, str]]:
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"FrancoisLeduc metadata not found under: {dataset_dir}. "
            "Expected metadata.csv"
        )
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_gaps_metadata(dataset_dir: Path) -> List[Dict[str, str]]:
    metadata_path = dataset_dir / "gaps_metadata_with_splits.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"GAPS metadata not found under: {dataset_dir}. "
            "Expected gaps_metadata_with_splits.csv"
        )
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_gaps_split(split: str) -> str:
    token = (split or "").strip().lower()
    if token in ("", "valid", "validate", "validation", "val"):
        return "validation"
    if token in ("train", "test"):
        return token
    raise ValueError(f"Unsupported GAPS split: {split}")


def scan_midis(
    dataset_type: str,
    dataset_dir: Path,
    *,
    split: str = "all",
    maps_pianos: str = "both",
) -> List[Path]:
    dataset_type = normalize_dataset_type(dataset_type)
    if dataset_type == "smd":
        return [p for p in sorted(dataset_dir.glob("*.mid")) if p.stem not in SMD_EXCLUDED]
    if dataset_type == "maps":
        mids: List[Path] = []
        for piano in _normalize_maps_pianos(maps_pianos):
            mids.extend(sorted((dataset_dir / piano / "MUS").glob("*.mid")))
        return mids
    if dataset_type == "maestro":
        split = _normalize_split(split)
        maestro_root, csv_path = _resolve_maestro_root_and_csv(dataset_dir)
        mids: List[Path] = []
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split != "all" and row.get("split", "").lower() != split:
                    continue
                p = maestro_root / row["midi_filename"]
                if p.exists():
                    mids.append(p)
        return mids
    if dataset_type == "francoisleduc":
        split = _normalize_split(split)
        mids: List[Path] = []
        for row in _read_francoisleduc_metadata(dataset_dir):
            if split != "all" and row.get("split", "").lower() != split:
                continue
            p = dataset_dir / row["midi_filename"]
            if p.exists():
                mids.append(p)
        return mids
    if dataset_type == "gaps":
        split = _normalize_split(split)
        mids: List[Path] = []
        for row in _read_gaps_metadata(dataset_dir):
            row_split = _normalize_gaps_split(row.get("split", ""))
            if split != "all" and row_split != split:
                continue
            p = dataset_dir / row["midi_path"]
            if p.exists():
                mids.append(p)
        return mids
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def build_item_out_dir(out_dir: Path, dataset_dir: Path, midi_path: Path) -> Path:
    rel = midi_path.relative_to(dataset_dir)
    return out_dir / rel.parent


def build_item_result_path(results_dir: Path, dataset_dir: Path, midi_path: Path) -> Path:
    rel = midi_path.relative_to(dataset_dir)
    return (results_dir / rel).with_suffix(".json")


def load_maestro_audio_map(
    dataset_type: str,
    dataset_dir: Path,
    *,
    split: str = "all",
) -> Dict[str, Path]:
    dataset_type = normalize_dataset_type(dataset_type)
    if dataset_type == "francoisleduc":
        split = _normalize_split(split)
        out: Dict[str, Path] = {}
        for row in _read_francoisleduc_metadata(dataset_dir):
            if split != "all" and row.get("split", "").lower() != split:
                continue
            out[row["midi_filename"]] = dataset_dir / row["audio_filename"]
        return out
    if dataset_type == "gaps":
        split = _normalize_split(split)
        out: Dict[str, Path] = {}
        for row in _read_gaps_metadata(dataset_dir):
            row_split = _normalize_gaps_split(row.get("split", ""))
            if split != "all" and row_split != split:
                continue
            out[row["midi_path"]] = dataset_dir / row["audio_path"]
        return out
    if dataset_type != "maestro":
        return {}
    split = _normalize_split(split)
    maestro_root, csv_path = _resolve_maestro_root_and_csv(dataset_dir)
    out: Dict[str, Path] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split != "all" and row.get("split", "").lower() != split:
                continue
            out[row["midi_filename"]] = maestro_root / row["audio_filename"]
    return out


def resolve_real_audio(
    dataset_type: str,
    dataset_dir: Path,
    midi_path: Path,
    maestro_audio_map: Dict[str, Path],
) -> Path:
    dataset_type = normalize_dataset_type(dataset_type)
    if dataset_type == "smd":
        p = midi_path.with_suffix(".mp3")
        if not p.exists():
            raise FileNotFoundError(f"SMD real audio not found for {midi_path}: {p}")
        return p
    if dataset_type == "maps":
        p = midi_path.with_suffix(".wav")
        if not p.exists():
            raise FileNotFoundError(f"MAPS real audio not found for {midi_path}: {p}")
        return p
    if dataset_type == "maestro":
        maestro_root, _ = _resolve_maestro_root_and_csv(dataset_dir)
        key = str(midi_path.relative_to(maestro_root))
        if key not in maestro_audio_map:
            raise FileNotFoundError(f"MAESTRO audio mapping not found for MIDI: {key}")
        p = maestro_audio_map[key]
        if not p.exists():
            raise FileNotFoundError(f"MAESTRO real audio missing: {p}")
        return p
    if dataset_type == "francoisleduc":
        key = str(midi_path.relative_to(dataset_dir))
        if key not in maestro_audio_map:
            raise FileNotFoundError(
                f"FrancoisLeduc audio mapping not found for MIDI: {key}"
            )
        p = maestro_audio_map[key]
        if not p.exists():
            raise FileNotFoundError(f"FrancoisLeduc real audio missing: {p}")
        return p
    if dataset_type == "gaps":
        key = str(midi_path.relative_to(dataset_dir))
        if key not in maestro_audio_map:
            raise FileNotFoundError(f"GAPS audio mapping not found for MIDI: {key}")
        p = maestro_audio_map[key]
        if not p.exists():
            raise FileNotFoundError(f"GAPS real audio missing: {p}")
        return p
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def mean_or_nan(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def json_text(payload: Dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json_text(payload) + "\n", encoding="utf-8")


def save_and_print_json(
    payload: Dict[str, object],
    *,
    json_out: Optional[str | Path] = None,
    mute_output: bool = False,
    print_obj: Optional[Dict[str, object]] = None,
) -> None:
    if json_out:
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(out_path, payload)
    if not mute_output:
        shown = print_obj if print_obj is not None else payload
        print(json_text(shown))


def pluck(item: Dict[str, object], path: Sequence[str]) -> object:
    cur: object = item
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(".".join(path))
        cur = cur[key]
    return cur


def collect_ok_metric(ok_items: List[Dict[str, object]], path: Sequence[str]) -> List[float]:
    values: List[float] = []
    for item in ok_items:
        try:
            values.append(float(pluck(item, path)))
        except (KeyError, TypeError, ValueError):
            continue
    return values
