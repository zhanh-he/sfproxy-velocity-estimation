from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from direct_invension.common import (
    dump_json,
    require_path,
    resolve_dataset_dir,
    resolve_dataset_split,
    resolve_path,
    slugify,
    validate_hop_contract,
)
from direct_invension.eval_framework import (
    attach_reference_audio_from_folder,
    build_dataset_prediction_manifest,
    build_folder_prediction_manifest,
    evaluate_prediction_manifest,
)


def strip_audio_references(manifest: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(manifest)
    items = []
    for item in payload.get("items", []):
        updated = dict(item)
        updated.pop("real_audio", None)
        items.append(updated)
    payload["items"] = items
    payload["reference_audio_dir"] = None
    payload["missing_reference_audio"] = []
    return payload


def fmt_metric(value: object) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:
        return "NaN"
    return f"{v:.6f}"


def _fmt_row(values: Sequence[str]) -> str:
    return "  " + "  ".join(f"{value:<10}" for value in values)


def append_metric_block(lines: list[str], summary: Dict[str, Any], title: str, prefix: str) -> None:
    lines.append(title)
    lines.append(_fmt_row(("pearson", "cosine", "mae", "spearman")))
    values = (
        fmt_metric(summary.get(f"{prefix}_pearson_correlation")),
        fmt_metric(summary.get(f"{prefix}_cosine_similarity")),
        fmt_metric(summary.get(f"{prefix}_mean_absolute_error")),
        fmt_metric(summary.get(f"{prefix}_spearman_correlation")),
    )
    lines.append(_fmt_row(values))


def _compute_synth_gt_metrics(config_prefix: str, eval_cfg=None) -> bool:
    # If compute_synth_gt_metrics is explicitly set in config, use that value
    if eval_cfg is not None and hasattr(eval_cfg, "compute_synth_gt_metrics"):
        return bool(getattr(eval_cfg, "compute_synth_gt_metrics"))
    # Otherwise, use default behavior: disable for Route III/IV
    return not str(config_prefix).startswith(("route3.", "route4."))


def format_evaluation_summary(
    *,
    dataset_type: str,
    instrument_path: Path,
    prediction_dir: Path,
    out_dir: Path,
    manifest_path: Path,
    summary: Dict[str, Any],
    extra_lines: Optional[Sequence[str]] = None,
) -> str:
    lines = [
        "Summary",
        f"  dataset: {dataset_type}",
        f"  label: {summary.get('label', '?')}",
        f"  num_items/ok/fail: {summary.get('num_items', '?')}/{summary.get('num_ok', '?')}/{summary.get('num_fail', '?')}",
        f"  instrument: {instrument_path}",
        f"  prediction_dir: {prediction_dir}",
        f"  manifest: {manifest_path}",
        f"  out_dir: {out_dir}",
    ]
    if extra_lines:
        lines.extend(str(line) for line in extra_lines)
    if bool(summary.get("velocity_mae_enabled", True)):
        lines.append(f"velocity_mae (0-127): {fmt_metric(summary.get('velocity_mae'))}")
    else:
        lines.append("velocity_mae (0-127): disabled")

    if any(str(key).startswith("synth_ref_") for key in summary):
        lines.append("")
        append_metric_block(lines, summary, "synth_gt_vs_synth_pred_bssl", "synth_ref_bssl")
        append_metric_block(lines, summary, "synth_gt_vs_synth_pred_bstl", "synth_ref_bstl")
    lines.append("")
    append_metric_block(lines, summary, "real_vs_synth_pred_bssl", "real_ref_bssl")
    append_metric_block(lines, summary, "real_vs_synth_pred_bstl", "real_ref_bstl")
    return "\n".join(lines)


def run_evaluation(
    *,
    cfg,
    dataset_type: str,
    eval_cfg,
    config_prefix: str,
    route_name: str,
    run_json_name: str,
    max_items: Optional[int] = None,
    extra_run_payload: Optional[Mapping[str, Any]] = None,
    extra_summary_lines: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    out_dir = resolve_path(eval_cfg.out_dir)
    pred_midi_dir = require_path(eval_cfg.pred_midi_dir, field_name=f"{config_prefix}.pred_midi_dir")
    instrument_path = require_path(eval_cfg.instrument_path, field_name=f"{config_prefix}.instrument_path")
    manifest_mode = str(eval_cfg.manifest_mode).lower()
    audio_reference_mode = str(eval_cfg.audio_reference_mode).lower()
    label = str(eval_cfg.label)
    requested_split = str(eval_cfg.split)
    resolved_split = resolve_dataset_split(requested_split)
    compute_velocity_mae = bool(getattr(eval_cfg, "compute_velocity_mae", True))
    compute_synth_gt_metrics = _compute_synth_gt_metrics(config_prefix, eval_cfg)

    validate_hop_contract(
        fps=float(eval_cfg.frames_per_second),
        hop_size=int(cfg.backend.supervision.hop_size),
        route_name=route_name,
    )

    if not pred_midi_dir.exists():
        raise FileNotFoundError(f"Prediction MIDI folder does not exist: {pred_midi_dir}")
    if not instrument_path.exists():
        raise FileNotFoundError(f"Instrument path does not exist: {instrument_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{slugify(label)}_manifest.json"

    if manifest_mode == "dataset":
        dataset_dir = resolve_dataset_dir(cfg, dataset_type)
        manifest = build_dataset_prediction_manifest(
            dataset_type=dataset_type,
            dataset_dir=dataset_dir,
            pred_midi_dir=pred_midi_dir,
            label=label,
            split=resolved_split,
            maps_pianos=str(eval_cfg.maps_pianos),
            max_items=max_items,
        )
        if audio_reference_mode == "folder":
            reference_audio_dir = require_path(
                eval_cfg.reference_audio_dir,
                field_name=f"{config_prefix}.reference_audio_dir",
            )
            if not reference_audio_dir.exists():
                raise FileNotFoundError(
                    f"Reference audio folder does not exist: {reference_audio_dir}"
                )
            manifest = attach_reference_audio_from_folder(
                manifest,
                gt_root=dataset_dir,
                reference_audio_dir=reference_audio_dir,
            )
        elif audio_reference_mode == "none":
            manifest = strip_audio_references(manifest)
        elif audio_reference_mode != "dataset":
            raise ValueError(f"{config_prefix}.audio_reference_mode must be one of dataset | folder | none")
    elif manifest_mode == "folder":
        if audio_reference_mode == "dataset":
            raise ValueError("audio_reference_mode=dataset is only valid when manifest_mode=dataset.")
        gt_midi_dir = require_path(eval_cfg.gt_midi_dir, field_name=f"{config_prefix}.gt_midi_dir")
        if not gt_midi_dir.exists():
            raise FileNotFoundError(f"GT MIDI folder does not exist: {gt_midi_dir}")
        reference_audio_dir = None
        if audio_reference_mode == "folder":
            reference_audio_dir = require_path(
                eval_cfg.reference_audio_dir,
                field_name=f"{config_prefix}.reference_audio_dir",
            )
            if not reference_audio_dir.exists():
                raise FileNotFoundError(
                    f"Reference audio folder does not exist: {reference_audio_dir}"
                )
        elif audio_reference_mode != "none":
            raise ValueError(f"{config_prefix}.audio_reference_mode must be one of dataset | folder | none")
        manifest = build_folder_prediction_manifest(
            gt_midi_dir=gt_midi_dir,
            pred_midi_dir=pred_midi_dir,
            label=label,
            reference_audio_dir=reference_audio_dir,
            max_items=max_items,
        )
    else:
        raise ValueError(f"{config_prefix}.manifest_mode must be dataset or folder")

    manifest["manifest_mode"] = manifest_mode
    manifest["audio_reference_mode"] = audio_reference_mode
    dump_json(manifest_path, manifest)

    payload = evaluate_prediction_manifest(
        manifest,
        instrument_path=instrument_path,
        out_dir=out_dir,
        render_sr=int(eval_cfg.render_sr),
        eval_sr=int(eval_cfg.eval_sr),
        frames_per_second=float(eval_cfg.frames_per_second),
        fft_size=int(eval_cfg.fft_size),
        bssl_mode=str(eval_cfg.bssl_mode),
        num_samples=int(eval_cfg.num_samples),
        normalization=str(eval_cfg.normalization),
        device="cuda",
        backend=str(eval_cfg.backend),
        skip_existing_render=not bool(eval_cfg.overwrite_render),
        compute_velocity_mae=compute_velocity_mae,
        compute_synth_gt_metrics=compute_synth_gt_metrics,
    )

    run_payload: Dict[str, Any] = {
        "dataset_type": dataset_type,
        "manifest_path": str(manifest_path),
        "prediction_dir": str(pred_midi_dir),
        "instrument_path": str(instrument_path),
        "requested_split": requested_split,
        "resolved_split": resolved_split,
        "compute_velocity_mae": compute_velocity_mae,
        "max_items": int(max_items) if max_items is not None else None,
        "summary": payload["summary"],
        "per_file_results_dir": payload["per_file_results_dir"],
    }
    if extra_run_payload:
        run_payload.update(dict(extra_run_payload))
    dump_json(out_dir / run_json_name, run_payload)

    summary_text = format_evaluation_summary(
        dataset_type=dataset_type,
        instrument_path=instrument_path,
        prediction_dir=pred_midi_dir,
        out_dir=out_dir,
        manifest_path=manifest_path,
        summary=payload["summary"],
        extra_lines=extra_summary_lines,
    )
    return {
        "manifest": manifest,
        "payload": payload,
        "run_payload": run_payload,
        "summary_text": summary_text,
    }
