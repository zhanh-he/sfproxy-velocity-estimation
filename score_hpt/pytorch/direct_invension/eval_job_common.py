from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from direct_invension.common import normalize_dataset_type, repo_root, resolve_dataset_split


_DATASET_DIRS = {
    "maestro": Path("Dataset/maestro-v3.0.0"),
    "smd": Path("Dataset/SMD"),
    "maps": Path("Dataset/MAPS"),
    "gaps": Path("Dataset/GAPS"),
    "francoisleduc": Path("Dataset/FrancoisLeducGuitarDataset"),
}

_INSTRUMENT_PATHS = {
    "piano": Path("202604_midiproxy_data/soundfont/SalamanderGrandPiano/SalamanderGrandPianoV3.sfz"),
    "guitar": Path("202604_midiproxy_data/soundfont/SpanishClassicalGuitar/SpanishClassicalGuitar-20190618.sfz"),
}


@dataclass(frozen=True)
class EvalJobContext:
    dataset_type: str
    requested_split: str
    effective_split: str
    resolved_split: str
    max_items: int | None
    instrument_key: str
    dataset_dir: Path
    instrument_path: Path
    workspace_root: Path


def supported_datasets() -> tuple[str, ...]:
    return tuple(_DATASET_DIRS)


def supported_instruments() -> tuple[str, ...]:
    return tuple(_INSTRUMENT_PATHS)


def add_common_eval_job_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--dataset",
        required=True,
        choices=supported_datasets(),
        help="Dataset type.",
    )
    parser.add_argument(
        "--instrument",
        required=True,
        choices=supported_instruments(),
        help="Instrument soundfont key.",
    )
    parser.add_argument(
        "--compute_velo_mae",
        action="store_true",
        help="Enable velocity MAE computation.",
    )
    parser.add_argument(
        "--eval_scope",
        required=True,
        choices=("test", "full", "one"),
        help="Evaluation scope. 'full' maps to split='all'; 'one' runs only one item for fast debugging.",
    )
    return parser


def shared_root() -> Path:
    return repo_root().parent.resolve()


def require_existing_path(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def resolve_dataset_dir(dataset_type: str) -> Path:
    if dataset_type not in _DATASET_DIRS:
        raise KeyError(f"Unsupported dataset_type: {dataset_type}")
    return require_existing_path(shared_root() / _DATASET_DIRS[dataset_type], label="Dataset directory")


def resolve_instrument_path(instrument_key: str) -> Path:
    if instrument_key not in _INSTRUMENT_PATHS:
        raise KeyError(f"Unsupported instrument_key: {instrument_key}")
    return require_existing_path(shared_root() / _INSTRUMENT_PATHS[instrument_key], label="Instrument path")


def resolve_workspace_root() -> Path:
    return require_existing_path(
        shared_root() / "202604_midiproxy_data" / "score_hpt" / "workspaces",
        label="Workspace root",
    )


def resolve_eval_job_context(*, dataset: str, eval_scope: str, instrument: str) -> EvalJobContext:
    dataset_type = normalize_dataset_type(dataset)
    requested_split = str(eval_scope).strip().lower()
    effective_split = "test" if requested_split == "one" else requested_split
    max_items = 1 if requested_split == "one" else None
    resolved_split = resolve_dataset_split(effective_split)
    instrument_key = str(instrument).strip().lower()
    return EvalJobContext(
        dataset_type=dataset_type,
        requested_split=requested_split,
        effective_split=effective_split,
        resolved_split=resolved_split,
        max_items=max_items,
        instrument_key=instrument_key,
        dataset_dir=resolve_dataset_dir(dataset_type),
        instrument_path=resolve_instrument_path(instrument_key),
        workspace_root=resolve_workspace_root(),
    )


def write_result_summary(
    *,
    eval_out_dir: str | Path,
    metadata: Mapping[str, object],
    summary_text: str,
) -> Path:
    eval_out_dir = Path(eval_out_dir)
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = eval_out_dir / "result_summary.txt"
    txt_path.write_text(
        "\n".join(
            [
                "Metadata",
                *(f"{key} = {value}" for key, value in metadata.items()),
                "",
                summary_text,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return txt_path
