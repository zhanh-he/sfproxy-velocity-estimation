from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from direct_invension.route34_cli import predict_route_dataset_via_route2, run_route_infer_main


def predict_route2_dataset(**kwargs) -> Dict[str, object]:
    from direct_invension.route2_infer import predict_route2_dataset as _predict_route2_dataset

    return _predict_route2_dataset(**kwargs)


def predict_route3_dataset(
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
) -> Dict[str, object]:
    return dict(
        predict_route_dataset_via_route2(
            route_name="route3",
            predict_route2_fn=predict_route2_dataset,
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            out_dir=out_dir,
            dataset_type=dataset_type,
            dataset_dir=dataset_dir,
            split=split,
            maps_pianos=maps_pianos,
            velocity_method=velocity_method,
            skip_existing=skip_existing,
            max_items=max_items,
        )
    )


def main() -> None:
    run_route_infer_main(
        argv=sys.argv[1:],
        route_name="route3",
        predict_fn=predict_route3_dataset,
    )


if __name__ == "__main__":
    main()
