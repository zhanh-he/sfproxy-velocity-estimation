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
    require_existing_path,
    resolve_eval_job_context,
    write_result_summary,
)
from direct_invension.eval_runner import run_evaluation
from direct_invension.route34_eval_support import checkpoint_model_overrides
from direct_invension.route3_infer import predict_route3_dataset


@dataclass(frozen=True)
class Route3EvalJobRequest:
    ckpt_path: str | Path
    dataset: str
    eval_scope: str
    compute_velo_mae: bool
    instrument: str


@dataclass(frozen=True)
class Route3EvalJobResult:
    checkpoint_path: Path
    dataset_type: str
    requested_split: str
    effective_split: str
    resolved_split: str
    compute_velocity_mae: bool
    instrument_key: str
    dataset_dir: Path
    instrument_path: Path
    infer_out_dir: Path
    eval_out_dir: Path
    summary_df: pd.DataFrame
    summary_text: str
    txt_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Route III inference + evaluation and save summary report.")
    add_common_eval_job_args(parser)
    parser.add_argument("--ckpt_path", required=True, help="Checkpoint path for Route III inference.")
    return parser


def _request_from_args(args: argparse.Namespace) -> Route3EvalJobRequest:
    return Route3EvalJobRequest(
        ckpt_path=args.ckpt_path,
        dataset=args.dataset,
        eval_scope=args.eval_scope,
        compute_velo_mae=bool(args.compute_velo_mae),
        instrument=args.instrument,
    )


def _build_overrides(
    *,
    checkpoint_path: Path,
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
        *checkpoint_model_overrides(checkpoint_path),
        f"dataset.test_set={dataset_type}",
        f"route3.infer.checkpoint_path={checkpoint_path}",
        f"route3.infer.out_dir={infer_out_dir}",
        f"route3.infer.split={effective_split}",
        "route3.infer.maps_pianos=both",
        f"route3.eval.pred_midi_dir={pred_midi_dir}",
        f"route3.eval.out_dir={eval_out_dir}",
        f"route3.eval.instrument_path={instrument_path}",
        f"route3.eval.split={effective_split}",
        "route3.eval.maps_pianos=both",
        f"route3.eval.compute_velocity_mae={'true' if compute_velocity_mae else 'false'}",
        f"route3.eval.label={route_label}",
    ]


def run_route3_eval_job(request: Route3EvalJobRequest) -> Route3EvalJobResult:
    checkpoint_path = require_existing_path(request.ckpt_path, label="Checkpoint")
    ctx = resolve_eval_job_context(
        dataset=request.dataset,
        eval_scope=request.eval_scope,
        instrument=request.instrument,
    )

    ckpt_token = slugify(f"{checkpoint_path.parent.name}_{checkpoint_path.stem}")
    run_slug = slugify(f"{ctx.dataset_type}_{ctx.requested_split}_{ctx.instrument_key}")
    route_label = slugify(f"route3_{run_slug}_{ckpt_token}")
    infer_out_dir = ctx.workspace_root / "route3" / run_slug / ckpt_token
    eval_out_dir = ctx.workspace_root / "route3_eval" / run_slug / ckpt_token
    pred_midi_dir = infer_out_dir / "pred_midis"

    cfg = compose_cfg(
        _build_overrides(
            checkpoint_path=checkpoint_path,
            dataset_type=ctx.dataset_type,
            effective_split=ctx.effective_split,
            compute_velocity_mae=bool(request.compute_velo_mae),
            instrument_path=ctx.instrument_path,
            infer_out_dir=infer_out_dir,
            eval_out_dir=eval_out_dir,
            pred_midi_dir=pred_midi_dir,
            route_label=route_label,
        ),
        job_name="route3_eval_job",
    )

    predict_route3_dataset(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        out_dir=infer_out_dir,
        dataset_type=ctx.dataset_type,
        dataset_dir=ctx.dataset_dir,
        split=ctx.effective_split,
        maps_pianos="both",
        velocity_method=str(cfg.route3.infer.velocity_method),
        skip_existing=not bool(cfg.route3.infer.overwrite),
        max_items=ctx.max_items,
    )

    eval_result = run_evaluation(
        cfg=cfg,
        dataset_type=ctx.dataset_type,
        eval_cfg=cfg.route3.eval,
        config_prefix="route3.eval",
        route_name="Route III eval",
        run_json_name="route3_eval_run.json",
        max_items=ctx.max_items,
        extra_run_payload={"config": OmegaConf.to_container(cfg, resolve=True)},
        extra_summary_lines=[
            f"  requested_split: {ctx.requested_split}",
            f"  effective_split: {ctx.effective_split}",
            f"  resolved_split: {ctx.resolved_split}",
            f"  compute_velocity_mae: {bool(request.compute_velo_mae)}",
            "  bstl_source: raw ntot metrics from data_analysis.evaluation.bssl_eval",
        ],
    )
    summary_df = evaluation_results_to_dataframe(eval_result["payload"])
    summary_text = eval_result["summary_text"]
    txt_path = write_result_summary(
        eval_out_dir=eval_out_dir,
        metadata={
            "CHECKPOINT_PATH": checkpoint_path,
            "DATASET_TYPE": ctx.dataset_type,
            "EVAL_SCOPE": ctx.requested_split,
            "EFFECTIVE_SPLIT": ctx.effective_split,
            "RESOLVED_SPLIT": ctx.resolved_split,
            "COMPUTE_VELOCITY_MAE": bool(request.compute_velo_mae),
            "INSTRUMENT_KEY": ctx.instrument_key,
        },
        summary_text=summary_text,
    )
    return Route3EvalJobResult(
        checkpoint_path=checkpoint_path,
        dataset_type=ctx.dataset_type,
        requested_split=ctx.requested_split,
        effective_split=ctx.effective_split,
        resolved_split=ctx.resolved_split,
        compute_velocity_mae=bool(request.compute_velo_mae),
        instrument_key=ctx.instrument_key,
        dataset_dir=ctx.dataset_dir,
        instrument_path=ctx.instrument_path,
        infer_out_dir=infer_out_dir,
        eval_out_dir=eval_out_dir,
        summary_df=summary_df,
        summary_text=summary_text,
        txt_path=txt_path,
    )


def main() -> None:
    result = run_route3_eval_job(_request_from_args(build_parser().parse_args()))
    print(result.summary_text)
    print("")
    print(f"TXT report: {result.txt_path}")


if __name__ == "__main__":
    main()
