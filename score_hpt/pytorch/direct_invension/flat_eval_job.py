from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from direct_invension.common import compose_cfg, slugify
from direct_invension.eval_framework import evaluation_results_to_dataframe
from direct_invension.eval_job_common import (
    add_common_eval_job_args,
    resolve_eval_job_context,
    write_result_summary,
)
from direct_invension.eval_runner import run_evaluation
from direct_invension.flat_infer import predict_flat_dataset


@dataclass(frozen=True)
class FlatEvalJobRequest:
    dataset: str
    eval_scope: str
    compute_velo_mae: bool
    instrument: str
    flat_velo: int


@dataclass(frozen=True)
class FlatEvalJobResult:
    dataset_type: str
    requested_split: str
    effective_split: str
    resolved_split: str
    compute_velocity_mae: bool
    instrument_key: str
    dataset_dir: Path
    instrument_path: Path
    flat_velocity: int
    infer_out_dir: Path
    eval_out_dir: Path
    summary_df: pd.DataFrame
    summary_text: str
    txt_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run flat-velocity baseline inference + evaluation and save summary report.")
    add_common_eval_job_args(parser)
    parser.add_argument("--flat_velo", required=True, type=int, help="Fixed MIDI velocity for all notes.")
    return parser


def _request_from_args(args: argparse.Namespace) -> FlatEvalJobRequest:
    return FlatEvalJobRequest(
        dataset=args.dataset,
        eval_scope=args.eval_scope,
        compute_velo_mae=bool(args.compute_velo_mae),
        instrument=args.instrument,
        flat_velo=int(args.flat_velo),
    )


def _build_overrides(
    *,
    dataset_type: str,
    effective_split: str,
    compute_velocity_mae: bool,
    instrument_path: Path,
    flat_velocity: int,
    infer_out_dir: Path,
    eval_out_dir: Path,
    pred_midi_dir: Path,
    route_label: str,
) -> list[str]:
    return [
        f"dataset.test_set={dataset_type}",
        f"flat.infer.out_dir={infer_out_dir}",
        f"flat.infer.split={effective_split}",
        "flat.infer.maps_pianos=both",
        f"flat.infer.flat_velocity={int(flat_velocity)}",
        f"flat.eval.pred_midi_dir={pred_midi_dir}",
        f"flat.eval.out_dir={eval_out_dir}",
        f"flat.eval.instrument_path={instrument_path}",
        f"flat.eval.split={effective_split}",
        "flat.eval.maps_pianos=both",
        f"flat.eval.compute_velocity_mae={'true' if compute_velocity_mae else 'false'}",
        f"flat.eval.label={route_label}",
    ]


def run_flat_eval_job(request: FlatEvalJobRequest) -> FlatEvalJobResult:
    ctx = resolve_eval_job_context(
        dataset=request.dataset,
        eval_scope=request.eval_scope,
        instrument=request.instrument,
    )
    flat_velocity = int(request.flat_velo)

    run_slug = slugify(f"{ctx.dataset_type}_{ctx.requested_split}_{ctx.instrument_key}_flat{flat_velocity}")
    route_label = slugify(f"flat_velocity_{flat_velocity}_{ctx.dataset_type}_{ctx.requested_split}_{ctx.instrument_key}")
    infer_out_dir = ctx.workspace_root / "flat" / run_slug
    eval_out_dir = ctx.workspace_root / "flat_eval" / run_slug
    pred_midi_dir = infer_out_dir / "pred_midis"

    cfg = compose_cfg(
        _build_overrides(
            dataset_type=ctx.dataset_type,
            effective_split=ctx.effective_split,
            compute_velocity_mae=bool(request.compute_velo_mae),
            instrument_path=ctx.instrument_path,
            flat_velocity=flat_velocity,
            infer_out_dir=infer_out_dir,
            eval_out_dir=eval_out_dir,
            pred_midi_dir=pred_midi_dir,
            route_label=route_label,
        ),
        job_name="flat_eval_job",
    )

    predict_flat_dataset(
        dataset_type=ctx.dataset_type,
        dataset_dir=ctx.dataset_dir,
        out_dir=infer_out_dir,
        flat_velocity=flat_velocity,
        split=ctx.effective_split,
        maps_pianos="both",
        skip_existing=not bool(cfg.flat.infer.overwrite),
        max_items=ctx.max_items,
    )

    eval_result = run_evaluation(
        cfg=cfg,
        dataset_type=ctx.dataset_type,
        eval_cfg=cfg.flat.eval,
        config_prefix="flat.eval",
        route_name="Flat velocity eval",
        run_json_name="flat_eval_run.json",
        max_items=ctx.max_items,
        extra_run_payload={
            "flat_velocity": flat_velocity,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        extra_summary_lines=[
            f"  requested_split: {ctx.requested_split}",
            f"  effective_split: {ctx.effective_split}",
            f"  resolved_split: {ctx.resolved_split}",
            f"  compute_velocity_mae: {bool(request.compute_velo_mae)}",
            f"  flat_velocity: {flat_velocity}",
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
            "FLAT_VELOCITY": flat_velocity,
        },
        summary_text=summary_text,
    )
    return FlatEvalJobResult(
        dataset_type=ctx.dataset_type,
        requested_split=ctx.requested_split,
        effective_split=ctx.effective_split,
        resolved_split=ctx.resolved_split,
        compute_velocity_mae=bool(request.compute_velo_mae),
        instrument_key=ctx.instrument_key,
        dataset_dir=ctx.dataset_dir,
        instrument_path=ctx.instrument_path,
        flat_velocity=flat_velocity,
        infer_out_dir=infer_out_dir,
        eval_out_dir=eval_out_dir,
        summary_df=summary_df,
        summary_text=summary_text,
        txt_path=txt_path,
    )


def main() -> None:
    result = run_flat_eval_job(_request_from_args(build_parser().parse_args()))
    print(result.summary_text)
    print("")
    print(f"TXT report: {result.txt_path}")


if __name__ == "__main__":
    main()
