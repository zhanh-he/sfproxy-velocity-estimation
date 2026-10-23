from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from direct_invension.common import compose_cfg, normalize_dataset_type, repo_root, resolve_dataset_split, slugify
from direct_invension.eval_framework import evaluation_results_to_dataframe
from direct_invension.eval_runner import run_evaluation
from direct_invension.route34_eval_support import checkpoint_model_overrides
from direct_invension.route4_infer import predict_route4_dataset


_DATASET_DIRS = {
    "maestro": Path("Dataset/maestro-v3.0.0"),
    "smd": Path("Dataset/SMD"),
    "gaps": Path("Dataset/GAPS"),
    "francoisleduc": Path("Dataset/FrancoisLeducGuitarDataset"),
}

_INSTRUMENT_PATHS = {
    "piano": Path("202604_midiproxy_data/soundfont/SalamanderGrandPiano/SalamanderGrandPianoV3.sfz"),
    "guitar": Path("202604_midiproxy_data/soundfont/SpanishClassicalGuitar/SpanishClassicalGuitar-20190618.sfz"),
}


@dataclass(frozen=True)
class Route4EvalJobRequest:
    ckpt_path: str | Path
    dataset: str
    eval_scope: str
    compute_velo_mae: bool
    instrument: str


@dataclass(frozen=True)
class Route4EvalJobResult:
    checkpoint_path: Path
    dataset_type: str
    requested_split: str
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
    parser = argparse.ArgumentParser(description="Run Route IV inference + evaluation and save summary report.")
    parser.add_argument("--ckpt_path", required=True, help="Checkpoint path for Route IV inference.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("maestro", "smd", "gaps", "francoisleduc"),
        help="Dataset type.",
    )
    parser.add_argument(
        "--eval_scope",
        required=True,
        choices=("test", "full", "one"),
        help="Evaluation scope. 'full' maps to split='all'; 'one' runs only one item for fast debugging.",
    )
    parser.add_argument("--compute_velo_mae", action="store_true", help="Enable velocity MAE computation.")
    parser.add_argument(
        "--instrument",
        required=True,
        choices=("piano", "guitar"),
        help="Instrument soundfont key.",
    )
    return parser


def _request_from_args(args: argparse.Namespace) -> Route4EvalJobRequest:
    return Route4EvalJobRequest(
        ckpt_path=args.ckpt_path,
        dataset=args.dataset,
        eval_scope=args.eval_scope,
        compute_velo_mae=bool(args.compute_velo_mae),
        instrument=args.instrument,
    )


def _shared_root() -> Path:
    return repo_root().parent.resolve()


def _require_existing_path(path: Path, *, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _resolve_dataset_dir(dataset_type: str) -> Path:
    if dataset_type not in _DATASET_DIRS:
        raise KeyError(f"Unsupported Route IV dataset_type: {dataset_type}")
    return _require_existing_path(_shared_root() / _DATASET_DIRS[dataset_type], label="Dataset directory")


def _resolve_instrument_path(instrument_key: str) -> Path:
    if instrument_key not in _INSTRUMENT_PATHS:
        raise KeyError(f"Unsupported Route IV instrument_key: {instrument_key}")
    return _require_existing_path(_shared_root() / _INSTRUMENT_PATHS[instrument_key], label="Instrument path")


def _resolve_workspace_root() -> Path:
    return _require_existing_path(
        _shared_root() / "202604_midiproxy_data" / "score_hpt" / "workspaces",
        label="Workspace root",
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
        f"route4.infer.checkpoint_path={checkpoint_path}",
        f"route4.infer.out_dir={infer_out_dir}",
        f"route4.infer.split={effective_split}",
        "route4.infer.maps_pianos=both",
        f"route4.eval.pred_midi_dir={pred_midi_dir}",
        f"route4.eval.out_dir={eval_out_dir}",
        f"route4.eval.instrument_path={instrument_path}",
        f"route4.eval.split={effective_split}",
        "route4.eval.maps_pianos=both",
        f"route4.eval.compute_velocity_mae={'true' if compute_velocity_mae else 'false'}",
        f"route4.eval.label={route_label}",
    ]


def _write_reports(
    *,
    checkpoint_path: Path,
    dataset_type: str,
    requested_split: str,
    resolved_split: str,
    compute_velocity_mae: bool,
    instrument_key: str,
    infer_out_dir: Path,
    eval_out_dir: Path,
    summary_text: str,
) -> Path:
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = eval_out_dir / "result_summary.txt"

    txt_path.write_text(
        "\n".join(
            [
                "Metadata",
                f"CHECKPOINT_PATH = {checkpoint_path}",
                f"DATASET_TYPE = {dataset_type}",
                f"EVAL_SCOPE = {requested_split}",
                f"COMPUTE_VELOCITY_MAE = {compute_velocity_mae}",
                f"INSTRUMENT_KEY = {instrument_key}",
                "",
                summary_text,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return txt_path


def run_route4_eval_job(request: Route4EvalJobRequest) -> Route4EvalJobResult:
    checkpoint_path = _require_existing_path(Path(request.ckpt_path), label="Checkpoint")
    dataset_type = normalize_dataset_type(request.dataset)
    requested_split = str(request.eval_scope).strip().lower()
    effective_split = "test" if requested_split == "one" else requested_split
    max_items = 1 if requested_split == "one" else None
    resolved_split = resolve_dataset_split(effective_split)
    instrument_key = str(request.instrument).strip().lower()
    dataset_dir = _resolve_dataset_dir(dataset_type)
    instrument_path = _resolve_instrument_path(instrument_key)
    workspace_root = _resolve_workspace_root()

    ckpt_token = slugify(f"{checkpoint_path.parent.name}_{checkpoint_path.stem}")
    run_slug = slugify(f"{dataset_type}_{requested_split}_{instrument_key}")
    route_label = slugify(f"route4_{run_slug}_{ckpt_token}")
    infer_out_dir = workspace_root / "route4" / run_slug / ckpt_token
    eval_out_dir = workspace_root / "route4_eval" / run_slug / ckpt_token
    pred_midi_dir = infer_out_dir / "pred_midis"

    cfg = compose_cfg(
        _build_overrides(
            checkpoint_path=checkpoint_path,
            dataset_type=dataset_type,
            effective_split=effective_split,
            compute_velocity_mae=bool(request.compute_velo_mae),
            instrument_path=instrument_path,
            infer_out_dir=infer_out_dir,
            eval_out_dir=eval_out_dir,
            pred_midi_dir=pred_midi_dir,
            route_label=route_label,
        ),
        job_name="route4_eval_job",
    )

    predict_route4_dataset(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        out_dir=infer_out_dir,
        dataset_type=dataset_type,
        dataset_dir=dataset_dir,
        split=effective_split,
        maps_pianos="both",
        velocity_method=str(cfg.route4.infer.velocity_method),
        skip_existing=not bool(cfg.route4.infer.overwrite),
        max_items=max_items,
    )

    eval_result = run_evaluation(
        cfg=cfg,
        dataset_type=dataset_type,
        eval_cfg=cfg.route4.eval,
        config_prefix="route4.eval",
        route_name="Route IV eval",
        run_json_name="route4_eval_run.json",
        max_items=max_items,
        extra_run_payload={"config": OmegaConf.to_container(cfg, resolve=True)},
        extra_summary_lines=[
            f"  requested_split: {requested_split}",
            f"  effective_split: {effective_split}",
            f"  resolved_split: {resolved_split}",
            f"  compute_velocity_mae: {bool(request.compute_velo_mae)}",
            "  bstl_source: raw ntot metrics from data_analysis.evaluation.bssl_eval",
        ],
    )
    summary_df = evaluation_results_to_dataframe(eval_result["payload"])
    summary_text = eval_result["summary_text"]
    txt_path = _write_reports(
        checkpoint_path=checkpoint_path,
        dataset_type=dataset_type,
        requested_split=requested_split,
        resolved_split=resolved_split,
        compute_velocity_mae=bool(request.compute_velo_mae),
        instrument_key=instrument_key,
        infer_out_dir=infer_out_dir,
        eval_out_dir=eval_out_dir,
        summary_text=summary_text,
    )
    return Route4EvalJobResult(
        checkpoint_path=checkpoint_path,
        dataset_type=dataset_type,
        requested_split=requested_split,
        resolved_split=resolved_split,
        compute_velocity_mae=bool(request.compute_velo_mae),
        instrument_key=instrument_key,
        dataset_dir=dataset_dir,
        instrument_path=instrument_path,
        infer_out_dir=infer_out_dir,
        eval_out_dir=eval_out_dir,
        summary_df=summary_df,
        summary_text=summary_text,
        txt_path=txt_path,
    )


def main() -> None:
    result = run_route4_eval_job(_request_from_args(build_parser().parse_args()))
    print(result.summary_text)
    print("")
    print(f"TXT report: {result.txt_path}")


if __name__ == "__main__":
    main()
