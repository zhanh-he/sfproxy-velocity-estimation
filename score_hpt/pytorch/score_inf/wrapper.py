# pytorch/score_inf/wrapper.py
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

from .io_types import dict_to_acoustic, dict_to_cond

class ScoreInfWrapper(nn.Module):
    """
    base_adapter(audio, ...) -> acoustic dict: {vel/vel_logits}
    post(acoustic_io, cond_io) -> {"vel_corr": ...}
    """
    def __init__(self, base_adapter: nn.Module, post: nn.Module, freeze_base: bool = True):
        super().__init__()
        self.base_adapter = base_adapter
        self.post = post
        self.freeze_base = freeze_base

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_base:
            self.base_adapter.eval()
        return self

    @staticmethod
    def _align_time(acoustic: "AcousticIO", cond: "CondIO") -> None:
        tensors = []

        def collect(t: torch.Tensor) -> None:
            if torch.is_tensor(t) and t.dim() >= 2:
                tensors.append(t)

        collect(acoustic.vel)
        collect(acoustic.vel_logits)
        collect(cond.onset)
        collect(cond.frame)
        collect(cond.exframe)

        for v in (acoustic.extra or {}).values():
            collect(v)
        for v in (cond.extra or {}).values():
            collect(v)

        if not tensors:
            return

        min_len = min(t.size(1) for t in tensors)

        def trim(t: torch.Tensor) -> torch.Tensor:
            if torch.is_tensor(t) and t.dim() >= 2 and t.size(1) != min_len:
                return t[:, :min_len]
            return t

        acoustic.vel = trim(acoustic.vel)
        acoustic.vel_logits = trim(acoustic.vel_logits)
        cond.onset = trim(cond.onset)
        cond.frame = trim(cond.frame)
        cond.exframe = trim(cond.exframe)
        acoustic.extra = {k: trim(v) for k, v in (acoustic.extra or {}).items()}
        cond.extra = {k: trim(v) for k, v in (cond.extra or {}).items()}

    def forward(
        self,
        audio: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        *base_inputs,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.freeze_base:
            with torch.no_grad():
                acoustic_dict = self.base_adapter(audio, *base_inputs, **kwargs)
        else:
            acoustic_dict = self.base_adapter(audio, *base_inputs, **kwargs)

        acoustic = dict_to_acoustic(acoustic_dict)
        cond_io = dict_to_cond(cond)
        self._align_time(acoustic, cond_io)

        out = self.post(acoustic, cond_io)
        return out
