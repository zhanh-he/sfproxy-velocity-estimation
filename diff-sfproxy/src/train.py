# pylint: disable=W1203
"""Training script for the SoundFont neural proxy.

Run example:
  python src/train.py --config-name train

"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

# Allow running this file directly with `python src/train.py`
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


from typing import Any, Dict, List

import hydra
import lightning as L
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

import common.hydra  # noqa: F401
from common.hydra import instantiate_callbacks
from common.logging import RankedLogger
from data.datasets_pkl import NoteProxyDatasetPkl

log = RankedLogger(__name__, rank_zero_only=True)


def _configure_default_logging() -> None:
    # Keep warnings/errors, but suppress startup INFO chatter in notebook runs.
    logging.getLogger(__name__).setLevel(logging.WARNING)
    logging.getLogger("common.hydra").setLevel(logging.WARNING)


def _log_run_config(logger: Logger | None, cfg: DictConfig) -> None:
    if logger is None:
        return
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg_dict, dict):
        logger.log_hyperparams(cfg_dict)


def _reset_output_dir(dir_path: Path) -> None:
    if not dir_path.exists():
        return
    for child in dir_path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _resolve_dataset_dir(dataset_cfg: DictConfig, data_dir: str, task_name: str) -> Path:
    explicit_path = dataset_cfg.get("path")
    if explicit_path:
        explicit_path = Path(str(explicit_path))
        if (explicit_path / "configs.pkl").is_file():
            return explicit_path
        export_dirs = sorted(
            [p for p in explicit_path.glob("export_*") if p.is_dir()],
            key=lambda p: p.name,
        )
        if export_dirs:
            return export_dirs[-1]
        if explicit_path.is_dir():
            return explicit_path

    instrument_name = dataset_cfg.get("instrument_name")
    if not instrument_name:
        raise ValueError("dataset path could not be resolved: set dataset.*.path or dataset.*.instrument_name")

    instrument_root = Path(str(data_dir)) / str(instrument_name)
    split = dataset_cfg.get("split")
    if split:
        tagged_dir = instrument_root / str(task_name) / str(split)
        if tagged_dir.is_dir():
            return tagged_dir

    export_dirs = sorted(
        [p for p in instrument_root.glob("export_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not export_dirs:
        raise ValueError(f"no exported dataset found under {instrument_root}")
    return export_dirs[-1]


def train(cfg: DictConfig) -> Dict[str, Any]:
    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg.seed))

    output_dir = Path(str(cfg.paths.output_dir))
    if bool(cfg.get("reset_output_dir", False)):
        _reset_output_dir(output_dir)

    train_path = _resolve_dataset_dir(cfg.dataset.train, cfg.paths.data_dir, cfg.task_name)
    log.info(f"Instantiating training dataset: {train_path}")
    train_dataset = NoteProxyDatasetPkl(train_path)
    if float(cfg.dataset.train.dataset_size) < 1.0:
        frac = float(cfg.dataset.train.dataset_size)
        log.info(f"Using {frac * 100:.1f}% of the dataset")
        dataset_size = max(1, int(frac * len(train_dataset)))
        rnd_indices = torch.multinomial(torch.ones(len(train_dataset)), dataset_size, replacement=False)
        train_dataset = Subset(train_dataset, rnd_indices)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=int(cfg.dataset.train.loader.batch_size),
        shuffle=True,
        num_workers=int(cfg.dataset.train.loader.num_workers),
        drop_last=False,
    )

    val_path = _resolve_dataset_dir(cfg.dataset.val, cfg.paths.data_dir, cfg.task_name)
    log.info(f"Instantiating validation dataset: {val_path}")
    val_dataset = NoteProxyDatasetPkl(val_path)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=int(cfg.dataset.val.loader.batch_size),
        shuffle=False,
        num_workers=int(cfg.dataset.val.loader.num_workers),
        drop_last=False,
    )

    # Instantiate model
    log.info(f"Instantiating proxy model <{cfg.model.cfg._target_}>")
    model_nn = hydra.utils.instantiate(cfg.model.cfg)

    log.info(f"Instantiating Lightning module <{cfg.solver._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.solver, model=model_nn)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] | None = instantiate_callbacks(cfg.get("callbacks"))

    logger: Logger | None = None
    if not cfg.get("logger"):
        log.warning("No logger configs found. Skipping logger.")
    elif cfg.logger.get("wandb"):
        log.info(f"Instantiating logger <{cfg.logger.wandb._target_}>")
        logger = hydra.utils.instantiate(cfg.logger.wandb)
        _log_run_config(logger, cfg)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        log.info(f"Resuming from ckpt_path={ckpt_path}")

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    metrics_dict = trainer.callback_metrics
    log.info(f"Metrics: {metrics_dict}")

    return metrics_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    _configure_default_logging()
    train(cfg)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
