# pytorch/score_inf/io_types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch

Tensor = torch.Tensor

@dataclass
class AcousticIO:
    vel: Optional[Tensor] = None         # (B,T,88) in [0,1]
    vel_logits: Optional[Tensor] = None  # (B,T,88) logits
    extra: Dict[str, Tensor] = field(default_factory=dict)

@dataclass
class CondIO:
    """
    GT conditioning only (no pedal, no dur):
      onset:   (B,T,88) 0/1 (optional)
      frame:   (B,T,88) 0/1
      exframe: (B,T,88) 0/1
    """
    onset: Optional[Tensor] = None
    frame: Optional[Tensor] = None
    exframe: Optional[Tensor] = None
    extra: Dict[str, Tensor] = field(default_factory=dict)

def dict_to_acoustic(d: Dict) -> AcousticIO:
    return AcousticIO(
        vel=d.get("vel", None),
        vel_logits=d.get("vel_logits", None),
        extra=d.get("extra", {}) or {},
    )

def dict_to_cond(d: Dict) -> CondIO:
    return CondIO(
        onset=d.get("onset", None),
        frame=d.get("frame", None),
        exframe=d.get("exframe", None),
        extra={k: v for k, v in d.items() if k not in {"onset", "frame", "exframe"}},
    )
