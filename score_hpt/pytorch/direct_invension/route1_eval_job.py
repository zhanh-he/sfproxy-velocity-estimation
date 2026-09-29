from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from direct_invension.common import compose_cfg, slugify, validate_hop_contract
from direct_invension.eval_framework import evaluation_results_to_dataframe
from direct_invension.eval_job_common import (
    add_common_eval_job_args,
    require_existing_path,
    resolve_eval_job_context,
    write_result_summary,
)
from direct_invension.eval_runner import run_evaluation
from direct_invension.route1_infer import predict_route1_dataset, resolve_route1_stats_json


@dataclass(frozen=True)
class Route1EvalJobRequest:
    dataset: str
    eval_scope: str
    compute_velo_mae: bool
    instrument: str


@dataclass(frozen=True)
class Route1EvalJobResult:
    dataset_type: str
    requested_split: str
    effective_split: str
    resolved_split: str
    compute_velocity_mae: bool
    instrument_key: str
    dataset_dir: Path
    instrument_path: Path
    stats_json: Path
    infer_out_dir: Path
    eval_out_dir: Path
    summary_df: pd.DataFrame
    summary_text: str
    txt_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Route I direct inversion + evaluation and save summary report.")
    return add_common_eval_job_args(parser)


def _request_from_args(args: argparse.Namespace) -> Route1EvalJobRequest:
    return Route1EvalJobRequest(
        dataset=args.dataset,
        eval_scope=args.eval_scope,
        compute_velo_mae=bool(args.compute_velo_mae),
        instrument=args.instrument,
    )


def _build_overrides(
    *,
    dataset_type: str,
    effective_split: str,
    compute_velocity_mae: bool,
    instrument_path: Path,
    infer_out_dir: Path,
    eval_out_dir: Path,
    pred_midi_dir: Path,
    route_label: str,
) -> list[str]:
    return [
        f"dataset.test_set={dataset_type}",
        f"route1.infer.out_dir={infer_out_dir}",
        f"route1.infer.split={effective_split}",
        "route1.infer.maps_pianos=both",
        f"route1.eval.pred_midi_dir={pred_midi_dir}",
        f"route1.eval.out_dir={eval_out_dir}",
        f"route1.eval.instrument_path={instrument_path}",
        f"route1.eval.split={effective_split}",
        "route1.eval.maps_pianos=both",
        f"route1.eval.compute_velocity_mae={'true' if compute_velocity_mae else 'false'}",
        f"route1.eval.label={route_label}",
    ]


def run_route1_eval_job(request: Route1EvalJobRequest) -> Route1EvalJobResult:
    ctx = resolve_eval_job_context(
        dataset=request.dataset,
        eval_scope=request.eval_scope,
        instrument=request.instrument,
    )

    run_slug = slugify(f"{ctx.dataset_type}_{ctx.requested_split}_{ctx.instrument_key}")
    route_label = slugify(f"route1_{run_slug}")
    infer_out_dir = ctx.workspace_root / "route1" / run_slug
    eval_out_dir = ctx.workspace_root / "route1_eval" / run_slug
    pred_midi_dir = infer_out_dir / "pred_midis"

    cfg = compose_cfg(
        _build_overrides(
            dataset_type=ctx.dataset_type,
            effective_split=ctx.effective_split,
            compute_velocity_mae=bool(request.compute_velo_mae),
            instrument_path=ctx.instrument_path,
            infer_out_dir=infer_out_dir,
            eval_out_dir=eval_out_dir,
            pred_midi_dir=pred_midi_dir,
            route_label=route_label,
        ),
        job_name="route1_eval_job",
    )
    stats_json = require_existing_path(resolve_route1_stats_json(cfg, ctx.dataset_type), label="Route I stats JSON")

    validate_hop_contract(
        fps=float(cfg.feature.frames_per_second),
        hop_size=int(cfg.route1.infer.note_hop),
        route_name="Route I eval",
    )

    predict_route1_dataset(
        dataset_type=ctx.dataset_type,
        dataset_dir=ctx.dataset_dir,
        out_dir=infer_out_dir,
        dataset_stats_json=stats_json,
        split=ctx.effective_split,
        maps_pianos="both",
        flat_velocity=int(cfg.route1.infer.flat_velocity),
        mapping_mode=str(cfg.route1.infer.mapping_mode),
        harmonic_weight=float(cfg.route1.infer.harmonic_weight),
        flux_weight=float(cfg.route1.infer.flux_weight),
        fallback_velocity=int(cfg.route1.infer.fallback_velocity),
        note_sample_rate=int(cfg.route1.infer.note_sample_rate),
        note_fft_size=int(cfg.route1.infer.note_fft_size),
        note_hop=int(cfg.route1.infer.note_hop),
        harmonic_count=int(cfg.route1.infer.harmonic_count),
        band_bins=int(cfg.route1.infer.band_bins),
        energy_window_s=float(cfg.route1.infer.energy_window_s),
        onset_pre_s=float(cfg.route1.infer.onset_pre_s),
        onset_post_s=float(cfg.route1.infer.onset_post_s),
        skip_existing=not bool(cfg.route1.infer.overwrite),
        max_items=ctx.max_items,
    )

    eval_result = run_evaluation(
        cfg=cfg,
        dataset_type=ctx.dataset_type,
        eval_cfg=cfg.route1.eval,
        config_prefix="route1.eval",
        route_name="Route I eval",
        run_json_name="route1_eval_run.json",
        max_items=ctx.max_items,
        extra_run_payload={
            "stats_json": str(stats_json),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        extra_summary_lines=[
            f"  requested_split: {ctx.requested_split}",
            f"  effective_split: {ctx.effective_split}",
            f"  resolved_split: {ctx.resolved_split}",
            f"  compute_velocity_mae: {bool(request.compute_velo_mae)}",
            f"  stats_json: {stats_json}",
            "  bstl_source: raw ntot metrics from data_analysis.evaluation.bssl_eval",
        ],
    )
    summary_df = evaluation_results_to_dataframe(eval_result["payload"])
    summary_text = eval_result["summary_text"]
    txt_path = write_result_summary(
        eval_out_dir=eval_out_dir,
        metadata={
            "DATASET_TYPE": ctx.dataset_type,
            "EVAL_SCOPE": ctx.requested_split,
            "EFFECTIVE_SPLIT": ctx.effective_split,
            "RESOLVED_SPLIT": ctx.resolved_split,
            "COMPUTE_VELOCITY_MAE": bool(request.compute_velo_mae),
            "INSTRUMENT_KEY": ctx.instrument_key,
            "STATS_JSON": stats_json,
        },
        summary_text=summary_text,
    )
    return Route1EvalJobResult(
        dataset_type=ctx.dataset_type,
        requested_split=ctx.requested_split,
        effective_split=ctx.effective_split,
        resolved_split=ctx.resolved_split,
        compute_velocity_mae=bool(request.compute_velo_mae),
        instrument_key=ctx.instrument_key,
        dataset_dir=ctx.dataset_dir,
        instrument_path=ctx.instrument_path,
        stats_json=stats_json,
        infer_out_dir=infer_out_dir,
        eval_out_dir=eval_out_dir,
        summary_df=summary_df,
        summary_text=summary_text,
        txt_path=txt_path,
    )


def main() -> None:
    result = run_route1_eval_job(_request_from_args(build_parser().parse_args()))
    print(result.summary_text)
    print("")
    print(f"TXT report: {result.txt_path}")


if __name__ == "__main__":
    main()
