from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from direct_invension.common import (
    compose_cfg,
    dump_json,
    ensure_repo_imports,
    normalize_dataset_type,
    resolve_dataset_dir,
    resolve_dataset_split,
    resolve_path,
)
from inference import VeloTranscription, resolve_checkpoint
from utilities import (
    load_mono_audio,
    original_score_events,
    pick_velocity_from_roll,
    prepare_aux_rolls,
    read_midi,
    select_condition_roll,
    write_events_to_midi,
)


def _build_dataset_scan_helpers():
    ensure_repo_imports()
    from data_analysis.cli._dataset_utils import load_maestro_audio_map, resolve_real_audio, scan_midis  # type: ignore

    return scan_midis, load_maestro_audio_map, resolve_real_audio


def _relative_key(dataset_dir: Path, midi_path: Path) -> str:
    try:
        return str(midi_path.relative_to(dataset_dir).with_suffix(""))
    except Exception:
        return midi_path.stem


def _route2_out_path(out_dir: Path, dataset_dir: Path, midi_path: Path) -> Path:
    try:
        rel = midi_path.relative_to(dataset_dir)
    except Exception:
        rel = Path(midi_path.name)
    return (out_dir / "pred_midis" / rel).with_suffix(".route2.mid")


def _decode_midi_events(midi_dict) -> List[str]:
    return [
        msg.decode() if isinstance(msg, (bytes, bytearray)) else str(msg)
        for msg in midi_dict["midi_event"]
    ]


def _build_model_inputs(cfg, target_dict):
    input2 = select_condition_roll(target_dict, cfg.model.input2)
    input3 = select_condition_roll(target_dict, cfg.model.input3)
    return input2, input3


def predict_route2_pair(
    *,
    cfg,
    transcriber: VeloTranscription,
    audio_path: str | Path,
    midi_path: str | Path,
    output_midi_path: str | Path,
    velocity_method: str,
) -> Dict[str, object]:
    audio_path = Path(audio_path)
    midi_path = Path(midi_path)
    output_midi_path = Path(output_midi_path)

    audio = load_mono_audio(audio_path, sample_rate=cfg.feature.sample_rate)
    midi_dict = read_midi(str(midi_path), dataset=cfg.dataset.test_set)
    midi_events_time = np.asarray(midi_dict["midi_event_time"], dtype=float)
    midi_events = _decode_midi_events(midi_dict)
    duration = float(len(audio) / cfg.feature.sample_rate)

    target_dict, _, _ = prepare_aux_rolls(cfg, midi_events_time, midi_events, duration)
    input2, input3 = _build_model_inputs(cfg, target_dict)
    transcribed = transcriber.transcribe(audio, input2=input2, input3=input3)
    velocity_roll = np.asarray(transcribed["output_dict"]["velocity_output"], dtype=np.float32)

    note_events, pedal_events = original_score_events(cfg, midi_events_time, midi_events, duration)
    pick_velocity_from_roll(note_events, velocity_roll, cfg, strategy=velocity_method)

    output_midi_path.parent.mkdir(parents=True, exist_ok=True)
    write_events_to_midi(0.0, note_events, pedal_events, str(output_midi_path))

    return {
        "audio_path": str(audio_path),
        "midi_path": str(midi_path),
        "output_midi_path": str(output_midi_path),
        "num_notes": int(len(note_events)),
        "duration_sec": duration,
        "velocity_method": str(velocity_method),
    }


def predict_route2_dataset(
    *,
    cfg,
    checkpoint_path: str | Path,
    out_dir: str | Path,
    dataset_type: str,
    dataset_dir: str | Path,
    split: str = "test",
    maps_pianos: str = "both",
    velocity_method: str = "onset_only",
    skip_existing: bool = True,
    max_items: int | None = None,
    label: str = "route2",
    manifest_name: str | None = None,
    file_summary_name: str | None = None,
) -> Dict[str, object]:
    scan_midis, load_maestro_audio_map, resolve_real_audio = _build_dataset_scan_helpers()

    dataset_dir = Path(dataset_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path).resolve()

    transcriber = VeloTranscription(checkpoint_path=str(checkpoint_path), cfg=cfg)
    requested_split = str(split)
    scan_split = resolve_dataset_split(requested_split)
    midi_files = scan_midis(dataset_type, dataset_dir, split=scan_split, maps_pianos=maps_pianos)
    if max_items is not None:
        midi_files = midi_files[: max(0, int(max_items))]
    maestro_audio_map = load_maestro_audio_map(dataset_type, dataset_dir, split=scan_split)

    items: List[Dict[str, object]] = []
    file_summaries: List[Dict[str, object]] = []

    for midi_path in tqdm(
        midi_files,
        desc=f"Route2 infer [{dataset_type}]",
        unit="file",
        dynamic_ncols=True,
    ):
        midi_path = Path(midi_path).resolve()
        real_audio = Path(resolve_real_audio(dataset_type, dataset_dir, midi_path, maestro_audio_map)).resolve()
        pred_midi_path = _route2_out_path(out_dir, dataset_dir, midi_path)
        skipped_existing = bool(skip_existing and pred_midi_path.exists())

        if skipped_existing:
            pair_payload = {
                "audio_path": str(real_audio),
                "midi_path": str(midi_path),
                "output_midi_path": str(pred_midi_path),
                "num_notes": None,
                "duration_sec": None,
                "velocity_method": str(velocity_method),
                "skipped_existing": True,
            }
        else:
            pair_payload = predict_route2_pair(
                cfg=cfg,
                transcriber=transcriber,
                audio_path=real_audio,
                midi_path=midi_path,
                output_midi_path=pred_midi_path,
                velocity_method=velocity_method,
            )
            pair_payload["skipped_existing"] = False

        key = _relative_key(dataset_dir, midi_path)
        items.append(
            {
                "key": key,
                "label": str(label),
                "gt_midi": str(midi_path),
                "pred_midi": str(pred_midi_path),
                "real_audio": str(real_audio),
            }
        )
        file_summaries.append({"key": key, **pair_payload})

    manifest = {
        "format_version": 1,
        "label": str(label),
        "dataset_type": str(dataset_type),
        "dataset_dir": str(dataset_dir),
        "split": scan_split,
        "requested_split": requested_split,
        "maps_pianos": maps_pianos,
        "checkpoint_path": str(checkpoint_path),
        "velocity_method": str(velocity_method),
        "items": items,
    }
    manifest_file = manifest_name or f"{label}_manifest.json"
    summary_file = file_summary_name or f"{label}_predictions.json"
    manifest_path = dump_json(out_dir / manifest_file, manifest)
    file_summary_path = dump_json(out_dir / summary_file, {"files": file_summaries})

    return {
        "label": str(label),
        "dataset_type": dataset_type,
        "dataset_dir": str(dataset_dir),
        "num_items": len(items),
        "out_dir": str(out_dir),
        "pred_midi_dir": str(out_dir / "pred_midis"),
        "checkpoint_path": str(checkpoint_path),
        "split": scan_split,
        "requested_split": requested_split,
        "velocity_method": str(velocity_method),
        "manifest_path": str(manifest_path),
        "file_summary_path": str(file_summary_path),
    }


def main() -> None:
    from omegaconf import OmegaConf

    cfg = compose_cfg(sys.argv[1:], job_name="route2_infer")
    dataset_type = normalize_dataset_type(cfg.dataset.test_set)
    dataset_dir = resolve_dataset_dir(cfg, dataset_type)
    infer_cfg = cfg.route2.infer
    out_dir = resolve_path(infer_cfg.out_dir)
    checkpoint_path = resolve_checkpoint(cfg, str(getattr(infer_cfg, "checkpoint_path", "") or "").strip() or None)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    payload = predict_route2_dataset(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        out_dir=out_dir,
        dataset_type=dataset_type,
        dataset_dir=dataset_dir,
        split=str(infer_cfg.split),
        maps_pianos=str(infer_cfg.maps_pianos),
        velocity_method=str(infer_cfg.velocity_method),
        skip_existing=not bool(infer_cfg.overwrite),
    )

    run_payload = {
        "dataset_type": dataset_type,
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "checkpoint_path": str(checkpoint_path),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "result": payload,
    }
    dump_json(out_dir / "route2_infer_run.json", run_payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
