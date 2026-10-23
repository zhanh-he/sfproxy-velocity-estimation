from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def normalize_route_name(route_name: str) -> str:
    token = str(route_name or "").strip().lower()
    if token not in {"route3", "route4"}:
        raise ValueError("route_name must be one of route3 | route4")
    return token


def checkpoint_model_overrides(checkpoint_path: str | Path) -> list[str]:
    checkpoint_path = Path(checkpoint_path)
    model_tag = checkpoint_path.parent.name.lower()

    if "note_editor" in model_tag or "score_note_editor" in model_tag:
        score_method = "note_editor"
        input2 = "onset"
    else:
        score_method = "direct"
        input2 = "null"

    return [
        "model.type=hpt",
        f"score_informed.method={score_method}",
        f"model.input2={input2}",
        "model.input3=null",
    ]


def predict_route3_dataset(**kwargs) -> Dict[str, Any]:
    from direct_invension.route3_infer import predict_route3_dataset as _predict_route3_dataset

    return _predict_route3_dataset(**kwargs)


def predict_route4_dataset(**kwargs) -> Dict[str, Any]:
    from direct_invension.route4_infer import predict_route4_dataset as _predict_route4_dataset

    return _predict_route4_dataset(**kwargs)


def infer_route_dataset(
    *,
    route_name: str,
    cfg,
    checkpoint_path: str | Path,
    out_dir: str | Path,
    dataset_type: str,
    dataset_dir: str | Path,
    split: str = "test",
    maps_pianos: str = "both",
    velocity_method: str = "onset_only",
    overwrite: bool = False,
) -> Dict[str, Any]:
    route_name = normalize_route_name(route_name)
    predictor = predict_route3_dataset if route_name == "route3" else predict_route4_dataset
    return predictor(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        out_dir=out_dir,
        dataset_type=dataset_type,
        dataset_dir=dataset_dir,
        split=split,
        maps_pianos=maps_pianos,
        velocity_method=velocity_method,
        skip_existing=not bool(overwrite),
    )
