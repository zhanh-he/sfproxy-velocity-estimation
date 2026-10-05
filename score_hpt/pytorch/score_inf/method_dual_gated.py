from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

from .registry import register_score_inf
from .io_types import AcousticIO, CondIO
from .utils import safe_logit

class _TinyConv(nn.Module):
    def __init__(self, in_ch: int, hid: int, n_blocks: int = 4):
        super().__init__()
        self.inp = nn.Conv2d(in_ch, hid, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hid, hid, 3, padding=1, groups=hid, bias=False),
                nn.GroupNorm(8 if hid >= 8 else 1, hid),
                nn.SiLU(),
                nn.Conv2d(hid, hid, 1, bias=False),
            )
            for _ in range(n_blocks)
        ])
        self.out = nn.Conv2d(hid, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for blk in self.blocks:
            h = h + 0.1 * blk(h)
        return self.out(h).squeeze(1)  # (B,T,88)

def _stack_cond(cond: CondIO, keys: List[str]) -> torch.Tensor:
    maps = []
    for k in keys:
        maps.append(getattr(cond, k))
    return torch.stack(maps, dim=1)

@register_score_inf("dual_gated")
class DualGated(nn.Module):
    """
    Velocity-only dual-expert + gate:
      v_ac = vel0 + alpha*tanh(delta_ac(vel_logits + cond))
      v_sc = sigmoid(sc_expert(cond))   # pure GT prior from cond
      w    = sigmoid(gate(vel_logits + cond))
      v    = w*v_ac + (1-w)*v_sc
    """
    def __init__(
        self,
        cond_keys: List[str] = ["onset", "frame", "exframe"],
        alpha: float = 0.2,
        hid: int = 48,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.cond_keys = cond_keys
        self.alpha = alpha

        # acoustic expert input: 1 (vel_logits) + Cc (cond)
        self.ac_expert = _TinyConv(in_ch=1 + len(cond_keys), hid=hid, n_blocks=n_blocks)
        # score/cond expert input: Cc
        self.sc_expert = _TinyConv(in_ch=len(cond_keys), hid=hid, n_blocks=n_blocks)
        # gate input: 1 + Cc
        self.gate = _TinyConv(in_ch=1 + len(cond_keys), hid=hid, n_blocks=n_blocks)

    def forward(self, acoustic: AcousticIO, cond: CondIO):
        vel0 = acoustic.vel if acoustic.vel is not None else torch.sigmoid(acoustic.vel_logits)
        vel_logits = acoustic.vel_logits if acoustic.vel_logits is not None else safe_logit(vel0)

        cond_map = _stack_cond(cond, self.cond_keys)  # (B,Cc,T,88)
        vel_logit_map = vel_logits.unsqueeze(1)       # (B,1,T,88)

        x_ac = torch.cat([vel_logit_map, cond_map], dim=1)  # (B,1+Cc,T,88)

        delta_ac = self.ac_expert(x_ac)
        v_ac = torch.clamp(vel0 + self.alpha * torch.tanh(delta_ac), 0.0, 1.0)

        v_sc = torch.sigmoid(self.sc_expert(cond_map))
        w = torch.sigmoid(self.gate(x_ac))

        vel_corr = torch.clamp(w * v_ac + (1.0 - w) * v_sc, 0.0, 1.0)

        return {"vel_corr": vel_corr, "delta": delta_ac, "debug": {"w": w, "v_ac": v_ac, "v_sc": v_sc}}
