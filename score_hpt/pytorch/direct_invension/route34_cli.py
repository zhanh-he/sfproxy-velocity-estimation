from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from omegaconf import OmegaConf

from direct_invension.common import (
    compose_cfg,
    dump_json,
    normalize_dataset_type,
    resolve_dataset_dir,
    resolve_dataset_split,
    resolve_path,
)
from direct_invension.eval_runner import run_evaluation
from direct_invension.route34_eval_support import checkpoint_model_overrides


def predict_route_dataset_via_route2(
    *,
    route_name: str,
    predict_route2_fn: Callable[..., Mapping[str, Any]],
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
) -> Mapping[str, Any]:
    return predict_route2_fn(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        out_dir=out_dir,
        dataset_type=dataset_type,
        dataset_dir=dataset_dir,
        split=resolve_dataset_split(split),
        maps_pianos=maps_pianos,
        velocity_method=velocity_method,
        skip_existing=skip_existing,
        max_items=max_items,
        label=route_name,
        manifest_name=f"{route_name}_manifest.json",
        file_summary_name=f"{route_name}_predictions.json",
    )


def run_route_infer_main(
    *,
    argv: Sequence[str],
    route_name: str,
    predict_fn: Callable[..., Mapping[str, Any]],
) -> None:
    from inference import resolve_checkpoint

    raw_cfg = compose_cfg(list(argv), job_name=f"{route_name}_infer")
    infer_cfg = getattr(getattr(raw_cfg, route_name), "infer")
    checkpoint_override = str(getattr(infer_cfg, "checkpoint_path", "") or "").strip() or None
    checkpoint_path = resolve_checkpoint(raw_cfg, checkpoint_override)

    # Route III / IV checkpoints may encode model family + score-conditioning in the
    # checkpoint folder name. Re-compose with those inferred overrides before loading.
    cfg = compose_cfg(
        [*checkpoint_model_overrides(checkpoint_path), *list(argv)],
        job_name=f"{route_name}_infer",
    )
    dataset_type = normalize_dataset_type(cfg.dataset.test_set)
    dataset_dir = resolve_dataset_dir(cfg, dataset_type)
    infer_cfg = getattr(getattr(cfg, route_name), "infer")
    out_dir = resolve_path(infer_cfg.out_dir)
    split = resolve_dataset_split(str(infer_cfg.split))

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    payload = predict_fn(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        out_dir=out_dir,
        dataset_type=dataset_type,
        dataset_dir=dataset_dir,
        split=split,
        maps_pianos=str(infer_cfg.maps_pianos),
        velocity_method=str(infer_cfg.velocity_method),
        skip_existing=not bool(infer_cfg.overwrite),
    )

    run_payload = {
        "dataset_type": dataset_type,
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "checkpoint_path": str(checkpoint_path),
        "requested_split": str(infer_cfg.split),
        "resolved_split": split,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "result": dict(payload),
    }
    dump_json(out_dir / f"{route_name}_infer_run.json", run_payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_route_evaluate_main(
    *,
    argv: Sequence[str],
    route_name: str,
    route_title: str,
) -> None:
    cfg = compose_cfg(list(argv), job_name=f"{route_name}_evaluate")
    dataset_type = normalize_dataset_type(cfg.dataset.test_set)
    eval_cfg = getattr(getattr(cfg, route_name), "eval")
    requested_split = str(eval_cfg.split)
    resolved_split = resolve_dataset_split(requested_split)

    result = run_evaluation(
        cfg=cfg,
        dataset_type=dataset_type,
        eval_cfg=eval_cfg,
        config_prefix=f"{route_name}.eval",
        route_name=route_title,
        run_json_name=f"{route_name}_eval_run.json",
        extra_run_payload={"config": OmegaConf.to_container(cfg, resolve=True)},
        extra_summary_lines=[
            f"  requested_split: {requested_split}",
            f"  resolved_split: {resolved_split}",
            f"  compute_velocity_mae: {bool(eval_cfg.compute_velocity_mae)}",
        ],
    )
    print(result["summary_text"])
