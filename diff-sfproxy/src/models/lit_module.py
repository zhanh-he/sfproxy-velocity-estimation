from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F

from common.logging import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Huber loss averaged over valid tokens."""
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")
    if mask.ndim != 2:
        raise ValueError("mask must be (B, N)")

    w = mask.to(pred.dtype).unsqueeze(-1)
    denom = torch.clamp(w.sum(), min=1.0)
    loss = F.smooth_l1_loss(pred * w, target * w, reduction="sum") / denom
    return loss


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.to(pred.dtype).unsqueeze(-1)
    denom = torch.clamp(w.sum(), min=1.0)
    err = torch.abs(pred - target) * w
    return err.sum() / denom


class NoteProxyLitModule(LightningModule):
    """Lightning module for note proxy training."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        optimizer: torch.optim.Optimizer = torch.optim.AdamW,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        seg_loss_weight: float = 0.05,
    ):
        super().__init__()
        self.model = model
        self.lr = float(lr)
        self.optimizer_cls = optimizer
        self.scheduler_cls = scheduler
        self.scheduler_config = scheduler_config
        self.seg_loss_weight = float(seg_loss_weight)

    def forward(self, pitch: torch.Tensor, cont: torch.Tensor, mask: torch.Tensor):
        return self.model(pitch=pitch, cont=cont, mask=mask)

    def _step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pitch = batch["pitch"].to(torch.long)
        cont = batch["cont"].to(torch.float32)
        mask = batch["mask"].to(torch.bool)
        target_note = batch["target_note"].to(torch.float32)

        pred_note, pred_seg = self(pitch, cont, mask)

        loss_note = masked_smooth_l1(pred_note, target_note, mask)
        mae_note = masked_mae(pred_note, target_note, mask)

        loss = loss_note

        metrics = {
            "loss_note": loss_note.detach(),
            "mae_note": mae_note.detach(),
        }

        if pred_seg is not None and "target_seg" in batch:
            target_seg = batch["target_seg"].to(torch.float32)
            loss_seg = F.smooth_l1_loss(pred_seg, target_seg)
            loss = loss + self.seg_loss_weight * loss_seg
            metrics["loss_seg"] = loss_seg.detach()

        metrics["loss"] = loss.detach()

        return loss, metrics

    def training_step(self, batch: Any, batch_idx: int):
        loss, metrics = self._step(batch)
        self.log("train/loss", metrics["loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/note_mae", metrics["mae_note"], prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        _, metrics = self._step(batch)
        self.log("val/loss", metrics["loss"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/note_mae", metrics["mae_note"], prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = self.optimizer_cls(self.parameters(), lr=self.lr)

        if self.scheduler_cls is None:
            return {"optimizer": opt}

        sched = self.scheduler_cls(optimizer=opt)
        cfg = self.scheduler_config or {"interval": "step", "frequency": 1}
        cfg = dict(cfg)
        cfg["scheduler"] = sched
        return {"optimizer": opt, "lr_scheduler": cfg}
