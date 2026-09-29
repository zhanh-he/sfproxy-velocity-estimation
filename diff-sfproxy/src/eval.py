from __future__ import annotations

from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import hydra
from omegaconf import DictConfig

from eval_velocity_recovery import run_eval_velocity_recovery


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    run_eval_velocity_recovery(cfg)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
