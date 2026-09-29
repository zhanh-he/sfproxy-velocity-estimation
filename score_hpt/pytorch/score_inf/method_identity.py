from __future__ import annotations
import torch
import torch.nn as nn

from .registry import register_score_inf
from .io_types import AcousticIO, CondIO


@register_score_inf("direct")
class DirectOutput(nn.Module):
    """
    No-op score-informed module (direct passthrough).
    Returns the base velocity prediction unchanged.
    """

    def __init__(self):
        super().__init__()

    def forward(self, acoustic: AcousticIO, cond: CondIO):
        vel0 = acoustic.vel if acoustic.vel is not None else torch.sigmoid(acoustic.vel_logits)
        vel_corr = vel0
        return {"vel_corr": vel_corr, "delta": None, "debug": {"vel0": vel0}}
