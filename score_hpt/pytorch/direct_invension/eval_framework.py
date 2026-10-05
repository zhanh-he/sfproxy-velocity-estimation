from __future__ import annotations

from dataclasses import dataclass
import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from direct_invension.common import (
    dump_json,
    ensure_repo_imports,
    extract_sorted_notes,
    load_json,
    slugify,
    SortedMidiNote,
)


@dataclass(frozen=True)
class VelocityAlignmentResult:
    num_gt_notes: int
    num_pred_notes: int
    num_matched_notes: int
    mae: float
    matched_in_exact_order: bool
    unmatched_gt: int
    unmatched_pred: int

def _direct_order_match(
    gt_notes: Sequence[SortedMidiNote],
    pred_notes: Sequence[SortedMidiNote],
    *,
    onset_tolerance_s: float,
    offset_tolerance_s: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if len(gt_notes) != len(pred_notes):
        return None
    gt_vel = np.zeros(len(gt_notes), dtype=np.float64)
    pred_vel = np.zeros(len(pred_notes), dtype=np.float64)
    for idx, (gt, pred) in enumerate(zip(gt_notes, pred_notes)):
        if int(gt.pitch) != int(pred.pitch):
            return None
        if abs(float(gt.onset) - float(pred.onset)) > onset_tolerance_s:
            return None
        if abs((float(gt.offset) - float(gt.onset)) - (float(pred.offset) - float(pred.onset))) > offset_tolerance_s:
            return None
        gt_vel[idx] = float(gt.velocity)
        pred_vel[idx] = float(pred.velocity)
    return gt_vel, pred_vel


def _greedy_pitch_onset_match(
    gt_notes: Sequence[SortedMidiNote],
    pred_notes: Sequence[SortedMidiNote],
    *,
    onset_tolerance_s: float,
    offset_tolerance_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    pred_by_pitch: Dict[int, List[int]] = {}
    for idx, note in enumerate(pred_notes):
        pred_by_pitch.setdefault(int(note.pitch), []).append(idx)

    used = np.zeros(len(pred_notes), dtype=bool)
    gt_vel: List[float] = []
    pred_vel: List[float] = []
    for gt in gt_notes:
        candidates = pred_by_pitch.get(int(gt.pitch), [])
        best_idx = None
        best_score = None
        gt_dur = float(gt.offset) - float(gt.onset)
        for idx in candidates:
            if used[idx]:
                continue
            pred = pred_notes[idx]
            onset_diff = abs(float(gt.onset) - float(pred.onset))
            if onset_diff > onset_tolerance_s:
                continue
            dur_diff = abs(gt_dur - (float(pred.offset) - float(pred.onset)))
            if dur_diff > offset_tolerance_s:
                continue
            score = (onset_diff, dur_diff, abs(float(gt.offset) - float(pred.offset)))
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            continue
        used[best_idx] = True
        gt_vel.append(float(gt.velocity))
        pred_vel.append(float(pred_notes[best_idx].velocity))
    return np.asarray(gt_vel, dtype=np.float64), np.asarray(pred_vel, dtype=np.float64)


def align_note_velocities(
    gt_midi: str | Path,
    pred_midi: str | Path,
    *,
    onset_tolerance_s: float = 0.03,
    offset_tolerance_s: float = 0.08,
) -> VelocityAlignmentResult:
    gt_notes = extract_sorted_notes(gt_midi)
    pred_notes = extract_sorted_notes(pred_midi)

    direct = _direct_order_match(
        gt_notes,
        pred_notes,
        onset_tolerance_s=onset_tolerance_s,
        offset_tolerance_s=offset_tolerance_s,
    )
    matched_in_exact_order = direct is not None
    if direct is None:
        gt_vel, pred_vel = _greedy_pitch_onset_match(
            gt_notes,
            pred_notes,
            onset_tolerance_s=onset_tolerance_s,
            offset_tolerance_s=offset_tolerance_s,
        )
    else:
        gt_vel, pred_vel = direct

    if gt_vel.size == 0 or pred_vel.size == 0:
        matched = 0
        mae = float("nan")
    else:
        matched = int(min(gt_vel.size, pred_vel.size))
        mae = float(np.mean(np.abs(gt_vel[:matched] - pred_vel[:matched])))

    return VelocityAlignmentResult(
        num_gt_notes=int(len(gt_notes)),
        num_pred_notes=int(len(pred_notes)),
        num_matched_notes=int(matched),
        mae=mae,
        matched_in_exact_order=matched_in_exact_order,
        unmatched_gt=int(max(0, len(gt_notes) - matched)),
        unmatched_pred=int(max(0, len(pred_notes) - matched)),
    )


@dataclass(frozen=True)
class PredictionItem:
    key: str
    gt_midi: str
    pred_midi: str
    real_audio: Optional[str] = None
    label: Optional[str] = None
    feature_json: Optional[str] = None


_METRIC_FIELD_NAMES = ("pearson_correlation", "mean_absolute_error", "cosine_similarity", "spearman_correlation")


def _import_dataset_helpers():
    ensure_repo_imports()
    from data_analysis.cli._dataset_utils import (  # type: ignore
        load_maestro_audio_map,
        resolve_real_audio,
        scan_midis,
    )
    return scan_midis, load_maestro_audio_map, resolve_real_audio


def _import_bssl_eval():
    ensure_repo_imports()
    from data_analysis.evaluation.bssl_eval import evaluate_bssl_pair  # type: ignore
    return evaluate_bssl_pair


def _import_renderers():
    ensure_repo_imports()
    from data_analysis.rendering.fluidsynth import render_midi_with_sf2_fluidsynth  # type: ignore
    from data_analysis.rendering.sfizz import render_midi_with_sfz_sfizz  # type: ignore
    return render_midi_with_sf2_fluidsynth, render_midi_with_sfz_sfizz


def _load_manifest(manifest: str | Path | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(manifest, Mapping):
        return dict(manifest)
    return load_json(manifest)


def _choose_backend(instrument_path: Path, backend: str) -> str:
    backend = str(backend or "auto").lower()
    if backend not in {"auto", "fluidsynth", "sfizz"}:
        raise ValueError("backend must be one of auto | fluidsynth | sfizz")
    if backend != "auto":
        return backend
    ext = instrument_path.suffix.lower()
    if ext == ".sf2":
        return "fluidsynth"
    if ext == ".sfz":
        return "sfizz"
    raise ValueError(
        f"Unsupported instrument extension: {instrument_path.suffix}. Use .sf2/.sfz or pass backend explicitly."
    )


def render_midi_to_audio(
    *,
    midi_path: str | Path,
    instrument_path: str | Path,
    wav_path: str | Path,
    sample_rate: int = 44100,
    backend: str = "auto",
    skip_existing: bool = True,
    fluidsynth_gain: float = 0.5,
    fluidsynth_extra_tail_sec: float = 2.0,
    sfizz_block_size: int = 1024,
    sfizz_polyphony: int = 256,
    sfizz_quality: int = 3,
) -> Dict[str, Any]:
    midi_path = Path(midi_path)
    instrument_path = Path(instrument_path)
    wav_path = Path(wav_path)
    if skip_existing and wav_path.exists():
        return {
            "status": "skipped_existing",
            "midi_path": str(midi_path),
            "instrument_path": str(instrument_path),
            "wav_path": str(wav_path),
            "backend": _choose_backend(instrument_path, backend),
        }

    render_midi_with_sf2_fluidsynth, render_midi_with_sfz_sfizz = _import_renderers()
    chosen = _choose_backend(instrument_path, backend)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    if chosen == "fluidsynth":
        extra = render_midi_with_sf2_fluidsynth(
            midi_path=midi_path,
            sf2_path=instrument_path,
            wav_path=wav_path,
            sample_rate=int(sample_rate),
            gain=float(fluidsynth_gain),
            extra_tail_sec=float(fluidsynth_extra_tail_sec),
        )
    else:
        extra = render_midi_with_sfz_sfizz(
            midi_path=midi_path,
            sfz_path=instrument_path,
            wav_path=wav_path,
            sample_rate=int(sample_rate),
            block_size=int(sfizz_block_size),
            polyphony=int(sfizz_polyphony),
            quality=int(sfizz_quality),
        )
    return {
        "status": "rendered",
        "backend": chosen,
        "wav_path": str(wav_path),
        "extra": extra,
    }


def _extract_pair_metrics(eval_payload: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not eval_payload:
        return out

    bssl_raw = (
        eval_payload.get("bssl", {}).get("flattened_raw_metrics")
        or eval_payload.get("bssl", {}).get("flattened_normalized_metrics")
        or {}
    )
    bstl_raw = (
        eval_payload.get("ntot", {}).get("curve_metrics_raw")
        or eval_payload.get("ntot", {}).get("curve_metrics")
        or {}
    )
    for src_name, src_payload in (("bssl", bssl_raw), ("bstl", bstl_raw)):
        metric_aliases = {
            "pearson_correlation": "pearson_correlation",
            "mean_absolute_error": "mean_absolute_error",
            "cosine_similarity": "cosine_sim",
            "spearman_correlation": "spearman_correlation",
        }
        for output_metric, source_metric in metric_aliases.items():
            if source_metric in src_payload:
                out[f"{src_name}_{output_metric}"] = float(src_payload[source_metric])
    return out


_RE_SUFFIX = re.compile(r"(?:_pred|_gt|_direct|_route\d+|\.pred|\.gt|\.direct|\.route\d+|\.flat\d+)$", re.IGNORECASE)
_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
_MIDI_EXTENSIONS = (".mid", ".midi")


def _normalize_stem(stem: str) -> str:
    text = str(stem)
    prev = None
    while prev != text:
        prev = text
        text = _RE_SUFFIX.sub("", text)
    return text


def _index_paths_with_suffixes(base_dir: Path, suffixes: Sequence[str]) -> Dict[str, List[Path]]:
    suffix_set = {str(s).lower() for s in suffixes}
    index: Dict[str, List[Path]] = {}
    for path in sorted(base_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffix_set:
            continue
        resolved = path.resolve()
        rel_no_suffix = str(path.relative_to(base_dir).with_suffix(""))
        for key in {
            rel_no_suffix,
            _normalize_stem(rel_no_suffix),
            path.stem,
            _normalize_stem(path.stem),
        }:
            index.setdefault(key, []).append(resolved)
    return index


def _index_prediction_midis(pred_midi_dir: Path) -> Dict[str, List[Path]]:
    return _index_paths_with_suffixes(pred_midi_dir, _MIDI_EXTENSIONS)


def _path_key_candidates(base_dir: Path, path: Path) -> List[str]:
    try:
        rel_no_suffix = str(path.relative_to(base_dir).with_suffix(""))
    except Exception:
        rel_no_suffix = path.stem
    return [
        rel_no_suffix,
        _normalize_stem(rel_no_suffix),
        path.stem,
        _normalize_stem(path.stem),
    ]


def _find_first_index_match(index: Mapping[str, Sequence[Path]], key_candidates: Sequence[str]) -> Optional[Path]:
    for key in key_candidates:
        matches = index.get(key) or []
        if matches:
            return Path(matches[0]).resolve()
    return None


def build_dataset_prediction_manifest(
    *,
    dataset_type: str,
    dataset_dir: str | Path,
    pred_midi_dir: str | Path,
    label: str,
    split: str = "test",
    maps_pianos: str = "both",
    max_items: Optional[int] = None,
    manifest_out: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Create a manifest by matching GT MIDIs in a dataset with predicted MIDIs in a folder."""
    scan_midis, load_maestro_audio_map, resolve_real_audio = _import_dataset_helpers()

    dataset_dir = Path(dataset_dir).resolve()
    pred_midi_dir = Path(pred_midi_dir).resolve()
    midi_files = scan_midis(dataset_type, dataset_dir, split=split, maps_pianos=maps_pianos)
    if max_items is not None:
        midi_files = midi_files[: max(0, int(max_items))]
    maestro_audio_map = load_maestro_audio_map(dataset_type, dataset_dir, split=split)
    pred_index = _index_prediction_midis(pred_midi_dir)

    items: List[Dict[str, Any]] = []
    missing: List[str] = []
    for gt_midi in tqdm(
        midi_files,
        desc=f"Build manifest [{dataset_type}]",
        unit="file",
        dynamic_ncols=True,
    ):
        gt_midi = Path(gt_midi).resolve()
        try:
            key = str(gt_midi.relative_to(dataset_dir).with_suffix(""))
        except Exception:
            key = gt_midi.stem
        stem_candidates = _path_key_candidates(dataset_dir, gt_midi)
        pred_path = _find_first_index_match(pred_index, stem_candidates)
        if pred_path is None:
            missing.append(str(gt_midi))
            continue
        real_audio = resolve_real_audio(dataset_type, dataset_dir, gt_midi, maestro_audio_map)
        items.append(
            {
                "key": key,
                "label": label,
                "gt_midi": str(gt_midi),
                "pred_midi": str(pred_path),
                "real_audio": str(real_audio),
            }
        )

    manifest = {
        "format_version": 1,
        "label": label,
        "dataset_type": dataset_type,
        "dataset_dir": str(dataset_dir),
        "pred_midi_dir": str(pred_midi_dir),
        "split": split,
        "maps_pianos": maps_pianos,
        "items": items,
        "missing_gt_midis": missing,
    }
    if manifest_out is not None:
        dump_json(manifest_out, manifest)
    return manifest


def build_folder_prediction_manifest(
    *,
    gt_midi_dir: str | Path,
    pred_midi_dir: str | Path,
    label: str,
    reference_audio_dir: Optional[str | Path] = None,
    max_items: Optional[int] = None,
    manifest_out: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Create a manifest by matching predicted MIDIs against a GT MIDI folder."""
    gt_midi_dir = Path(gt_midi_dir).resolve()
    pred_midi_dir = Path(pred_midi_dir).resolve()
    reference_audio_path = Path(reference_audio_dir).resolve() if reference_audio_dir is not None else None

    gt_midis: List[Path] = []
    for suffix in _MIDI_EXTENSIONS:
        gt_midis.extend(sorted(gt_midi_dir.rglob(f"*{suffix}")))
    # Deduplicate in case both .mid and .midi globs overlap unexpectedly.
    gt_midis = sorted({p.resolve() for p in gt_midis})
    if max_items is not None:
        gt_midis = gt_midis[: max(0, int(max_items))]

    pred_index = _index_prediction_midis(pred_midi_dir)
    audio_index = (
        _index_paths_with_suffixes(reference_audio_path, _AUDIO_EXTENSIONS)
        if reference_audio_path is not None
        else {}
    )

    items: List[Dict[str, Any]] = []
    missing_gt_midis: List[str] = []
    missing_reference_audio: List[str] = []
    for gt_midi in tqdm(
        gt_midis,
        desc="Build folder manifest",
        unit="file",
        dynamic_ncols=True,
    ):
        key_candidates = _path_key_candidates(gt_midi_dir, gt_midi)
        pred_path = _find_first_index_match(pred_index, key_candidates)
        if pred_path is None:
            missing_gt_midis.append(str(gt_midi))
            continue

        item: Dict[str, Any] = {
            "key": key_candidates[0],
            "label": label,
            "gt_midi": str(gt_midi),
            "pred_midi": str(pred_path),
        }
        if reference_audio_path is not None:
            reference_audio = _find_first_index_match(audio_index, key_candidates)
            if reference_audio is not None:
                item["real_audio"] = str(reference_audio)
            else:
                missing_reference_audio.append(str(gt_midi))
        items.append(item)

    manifest = {
        "format_version": 1,
        "label": label,
        "gt_midi_dir": str(gt_midi_dir),
        "pred_midi_dir": str(pred_midi_dir),
        "reference_audio_dir": str(reference_audio_path) if reference_audio_path is not None else None,
        "items": items,
        "missing_gt_midis": missing_gt_midis,
        "missing_reference_audio": missing_reference_audio,
    }
    if manifest_out is not None:
        dump_json(manifest_out, manifest)
    return manifest


def attach_reference_audio_from_folder(
    manifest: str | Path | Mapping[str, Any],
    *,
    gt_root: str | Path,
    reference_audio_dir: str | Path,
    manifest_out: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Attach audio references from a folder to an existing manifest."""
    payload = _load_manifest(manifest)
    gt_root = Path(gt_root).resolve()
    reference_audio_dir = Path(reference_audio_dir).resolve()
    audio_index = _index_paths_with_suffixes(reference_audio_dir, _AUDIO_EXTENSIONS)

    items: List[Dict[str, Any]] = []
    missing_reference_audio: List[str] = []
    for item in tqdm(
        payload.get("items", []),
        desc="Attach reference audio",
        unit="file",
        dynamic_ncols=True,
    ):
        updated = dict(item)
        gt_midi = Path(str(item["gt_midi"])).resolve()
        reference_audio = _find_first_index_match(audio_index, _path_key_candidates(gt_root, gt_midi))
        if reference_audio is None:
            updated.pop("real_audio", None)
            missing_reference_audio.append(str(gt_midi))
        else:
            updated["real_audio"] = str(reference_audio)
        items.append(updated)

    merged = dict(payload)
    merged["items"] = items
    merged["reference_audio_dir"] = str(reference_audio_dir)
    merged["missing_reference_audio"] = missing_reference_audio
    if manifest_out is not None:
        dump_json(manifest_out, merged)
    return merged


def _summary_row_from_result(summary: Mapping[str, Any], *, label: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"label": label}
    for key, value in summary.items():
        row[key] = value
    return row


def evaluation_results_to_dataframe(*results: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for result in results:
        label = str(result.get("label") or result.get("summary", {}).get("label") or "eval")
        rows.append(_summary_row_from_result(result.get("summary", {}), label=label))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _maybe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except Exception:
        return None
    if np.isnan(f):
        return None
    return f


def _mean_metrics(results: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in keys:
        vals = []
        for item in results:
            v = _maybe_float(item.get(key))
            if v is not None:
                vals.append(v)
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def evaluate_prediction_item(
    item: Mapping[str, Any],
    *,
    instrument_path: str | Path,
    out_dir: str | Path,
    render_sr: int = 44100,
    eval_sr: int = 22050,
    frames_per_second: float = 50.0,
    fft_size: int = 1024,
    bssl_mode: str = "sone",
    num_samples: int = 2048,
    normalization: str = "zscore",
    device: Optional[str] = None,
    backend: str = "auto",
    skip_existing_render: bool = True,
    compute_velocity_mae: bool = True,
    compute_synth_gt_metrics: bool = True,
    score_progress=None,
) -> Dict[str, Any]:
    evaluate_bssl_pair = _import_bssl_eval()

    key = str(item.get("key") or Path(str(item["gt_midi"])).stem)
    safe_key = slugify(key)
    out_dir = Path(out_dir)
    item_render_dir = out_dir / "renders" / safe_key
    item_render_dir.mkdir(parents=True, exist_ok=True)

    gt_midi = Path(str(item["gt_midi"]))
    pred_midi = Path(str(item["pred_midi"]))
    real_audio = Path(str(item["real_audio"])) if item.get("real_audio") else None

    pred_syn_wav = item_render_dir / f"{safe_key}.pred.wav"

    gt_syn_wav = item_render_dir / f"{safe_key}.gt.wav"
    gt_render = None
    if compute_synth_gt_metrics:
        gt_render = render_midi_to_audio(
            midi_path=gt_midi,
            instrument_path=instrument_path,
            wav_path=gt_syn_wav,
            sample_rate=int(render_sr),
            backend=backend,
            skip_existing=skip_existing_render,
        )
    pred_render = render_midi_to_audio(
        midi_path=pred_midi,
        instrument_path=instrument_path,
        wav_path=pred_syn_wav,
        sample_rate=int(render_sr),
        backend=backend,
        skip_existing=skip_existing_render,
    )

    velocity_alignment = align_note_velocities(gt_midi, pred_midi) if compute_velocity_mae else None

    synth_ref = None
    if compute_synth_gt_metrics:
        synth_ref = evaluate_bssl_pair(
            pred_wav=pred_syn_wav,
            gt_wav=gt_syn_wav,
            sample_rate=int(eval_sr),
            frames_per_second=float(frames_per_second),
            fft_size=int(fft_size),
            bssl_mode=bssl_mode,
            num_samples=int(num_samples),
            normalization=normalization,
            device=device,
        )
        if score_progress is not None:
            score_progress.update(1)
    real_ref = None
    if real_audio is not None:
        real_ref = evaluate_bssl_pair(
            pred_wav=pred_syn_wav,
            gt_wav=real_audio,
            sample_rate=int(eval_sr),
            frames_per_second=float(frames_per_second),
            fft_size=int(fft_size),
            bssl_mode=bssl_mode,
            num_samples=int(num_samples),
            normalization=normalization,
            device=device,
        )
        if score_progress is not None:
            score_progress.update(1)

    summary: Dict[str, Any] = {"key": key}
    if velocity_alignment is not None:
        summary.update(
            {
                "velocity_mae": float(velocity_alignment.mae),
                "num_gt_notes": int(velocity_alignment.num_gt_notes),
                "num_pred_notes": int(velocity_alignment.num_pred_notes),
                "num_matched_notes": int(velocity_alignment.num_matched_notes),
            }
        )
    for prefix, payload in (("synth_ref", synth_ref), ("real_ref", real_ref)):
        if payload is None:
            continue
        metrics = _extract_pair_metrics(payload)
        for metric_name, metric_value in metrics.items():
            summary[f"{prefix}_{metric_name}"] = float(metric_value)

    return {
        "status": "ok",
        "key": key,
        "label": item.get("label"),
        "gt_midi": str(gt_midi),
        "pred_midi": str(pred_midi),
        "real_audio": str(real_audio) if real_audio else None,
        "render": {
            "gt": gt_render,
            "pred": pred_render,
        },
        "velocity": velocity_alignment.__dict__ if velocity_alignment is not None else None,
        "synth_ref": synth_ref,
        "real_ref": real_ref,
        "summary": summary,
    }


def evaluate_prediction_manifest(
    manifest: str | Path | Mapping[str, Any],
    *,
    instrument_path: str | Path,
    out_dir: str | Path,
    render_sr: int = 44100,
    eval_sr: int = 22050,
    frames_per_second: float = 50.0,
    fft_size: int = 1024,
    bssl_mode: str = "sone",
    num_samples: int = 2048,
    normalization: str = "zscore",
    device: Optional[str] = None,
    backend: str = "auto",
    skip_existing_render: bool = True,
    fail_fast: bool = False,
    compute_velocity_mae: bool = True,
    compute_synth_gt_metrics: bool = True,
) -> Dict[str, Any]:
    payload = _load_manifest(manifest)
    items = payload.get("items", [])
    label = str(payload.get("label") or "evaluation")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_file_dir = out_dir / "per_file_results"
    per_file_dir.mkdir(parents=True, exist_ok=True)
    score_steps = sum((1 if compute_synth_gt_metrics else 0) + (1 if item.get("real_audio") else 0) for item in items)
    score_progress = (
        tqdm(
            total=score_steps,
            desc="Score",
            unit="pair",
            dynamic_ncols=True,
        )
        if score_steps > 0
        else None
    )

    results: List[Dict[str, Any]] = []
    ok_count = 0
    fail_count = 0
    try:
        for item in tqdm(
            items,
            desc="Evaluate",
            unit="file",
            dynamic_ncols=True,
        ):
            item_score_steps = (1 if compute_synth_gt_metrics else 0) + (1 if item.get("real_audio") else 0)
            item_score_before = int(score_progress.n) if score_progress is not None else 0
            try:
                res = evaluate_prediction_item(
                    item,
                    instrument_path=instrument_path,
                    out_dir=out_dir,
                    render_sr=int(render_sr),
                    eval_sr=int(eval_sr),
                    frames_per_second=float(frames_per_second),
                    fft_size=int(fft_size),
                    bssl_mode=bssl_mode,
                    num_samples=int(num_samples),
                    normalization=normalization,
                    device=device,
                    backend=backend,
                    skip_existing_render=skip_existing_render,
                    compute_velocity_mae=compute_velocity_mae,
                    compute_synth_gt_metrics=compute_synth_gt_metrics,
                    score_progress=score_progress,
                )
                ok_count += 1
            except Exception as exc:  # noqa: BLE001
                res = {
                    "status": "error",
                    "key": item.get("key"),
                    "label": item.get("label"),
                    "error": repr(exc),
                    "gt_midi": item.get("gt_midi"),
                    "pred_midi": item.get("pred_midi"),
                }
                if score_progress is not None:
                    completed = int(score_progress.n) - item_score_before
                    remaining = max(0, int(item_score_steps) - int(completed))
                    if remaining:
                        score_progress.update(remaining)
                fail_count += 1
                if fail_fast:
                    raise
            results.append(res)
            key = slugify(str(item.get("key") or len(results)))
            dump_json(per_file_dir / f"{key}.json", res)
    finally:
        if score_progress is not None:
            score_progress.close()

    ok_results = [r for r in results if r.get("status") == "ok"]
    summary_fields = [
        "real_ref_bssl_pearson_correlation",
        "real_ref_bssl_mean_absolute_error",
        "real_ref_bssl_cosine_similarity",
        "real_ref_bssl_spearman_correlation",
        "real_ref_bstl_pearson_correlation",
        "real_ref_bstl_mean_absolute_error",
        "real_ref_bstl_cosine_similarity",
        "real_ref_bstl_spearman_correlation",
    ]
    if compute_synth_gt_metrics:
        summary_fields = [
            "synth_ref_bssl_pearson_correlation",
            "synth_ref_bssl_mean_absolute_error",
            "synth_ref_bssl_cosine_similarity",
            "synth_ref_bssl_spearman_correlation",
            "synth_ref_bstl_pearson_correlation",
            "synth_ref_bstl_mean_absolute_error",
            "synth_ref_bstl_cosine_similarity",
            "synth_ref_bstl_spearman_correlation",
            *summary_fields,
        ]
    if compute_velocity_mae:
        summary_fields = ["velocity_mae", *summary_fields]
    summary_values = _mean_metrics([r.get("summary", {}) for r in ok_results], summary_fields)
    summary = {
        "label": label,
        "num_items": int(len(items)),
        "num_ok": int(ok_count),
        "num_fail": int(fail_count),
        "velocity_mae_enabled": bool(compute_velocity_mae),
        **summary_values,
    }
    final_payload = {
        "label": label,
        "manifest": payload,
        "summary": summary,
        "results": results,
        "per_file_results_dir": str(per_file_dir),
    }
    dump_json(out_dir / f"{slugify(label)}_evaluation.json", final_payload)
    return final_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shared audio-domain evaluation for Route I / II / III / IV.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_manifest = subparsers.add_parser("manifest", help="Evaluate one manifest JSON.")
    p_manifest.add_argument("--manifest", required=True)
    p_manifest.add_argument("--instrument", required=True)
    p_manifest.add_argument("--out_dir", required=True)
    p_manifest.add_argument("--render_sr", type=int, default=44100)
    p_manifest.add_argument("--eval_sr", type=int, default=22050)
    p_manifest.add_argument("--fps", type=float, default=50.0)
    p_manifest.add_argument("--fft", type=int, default=1024)
    p_manifest.add_argument("--bssl_mode", default="sone", choices=["sone", "bark"])
    p_manifest.add_argument("--num_samples", type=int, default=2048)
    p_manifest.add_argument("--norm", default="zscore", choices=["zscore", "minmax", "none"])
    p_manifest.add_argument("--device", default=None)
    p_manifest.add_argument("--backend", default="auto", choices=["auto", "fluidsynth", "sfizz"])
    p_manifest.add_argument("--overwrite_render", action="store_true")

    p_dataset = subparsers.add_parser("dataset", help="Build a manifest from dataset + pred MIDI dir, then evaluate.")
    p_dataset.add_argument("--dataset_type", required=True, choices=["smd", "maestro", "maps", "francoisleduc", "gaps"])
    p_dataset.add_argument("--dataset_dir", required=True)
    p_dataset.add_argument("--pred_midi_dir", required=True)
    p_dataset.add_argument("--label", required=True)
    p_dataset.add_argument("--instrument", required=True)
    p_dataset.add_argument("--out_dir", required=True)
    p_dataset.add_argument("--split", default="test")
    p_dataset.add_argument("--maps_pianos", default="both")
    p_dataset.add_argument("--render_sr", type=int, default=44100)
    p_dataset.add_argument("--eval_sr", type=int, default=22050)
    p_dataset.add_argument("--fps", type=float, default=50.0)
    p_dataset.add_argument("--fft", type=int, default=1024)
    p_dataset.add_argument("--bssl_mode", default="sone", choices=["sone", "bark"])
    p_dataset.add_argument("--num_samples", type=int, default=2048)
    p_dataset.add_argument("--norm", default="zscore", choices=["zscore", "minmax", "none"])
    p_dataset.add_argument("--device", default=None)
    p_dataset.add_argument("--backend", default="auto", choices=["auto", "fluidsynth", "sfizz"])
    p_dataset.add_argument("--overwrite_render", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.cmd == "manifest":
        payload = evaluate_prediction_manifest(
            args.manifest,
            instrument_path=args.instrument,
            out_dir=args.out_dir,
            render_sr=args.render_sr,
            eval_sr=args.eval_sr,
            frames_per_second=args.fps,
            fft_size=args.fft,
            bssl_mode=args.bssl_mode,
            num_samples=args.num_samples,
            normalization=args.norm,
            device=args.device,
            backend=args.backend,
            skip_existing_render=not args.overwrite_render,
        )
        print(payload["summary"])
        return

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / f"{slugify(args.label)}_manifest.json"
    manifest = build_dataset_prediction_manifest(
        dataset_type=args.dataset_type,
        dataset_dir=args.dataset_dir,
        pred_midi_dir=args.pred_midi_dir,
        label=args.label,
        split=args.split,
        maps_pianos=args.maps_pianos,
        manifest_out=manifest_path,
    )
    payload = evaluate_prediction_manifest(
        manifest,
        instrument_path=args.instrument,
        out_dir=out_dir,
        render_sr=args.render_sr,
        eval_sr=args.eval_sr,
        frames_per_second=args.fps,
        fft_size=args.fft,
        bssl_mode=args.bssl_mode,
        num_samples=args.num_samples,
        normalization=args.norm,
        device=args.device,
        backend=args.backend,
        skip_existing_render=not args.overwrite_render,
    )
    print(payload["summary"])


if __name__ == "__main__":
    main()
