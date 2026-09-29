from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn

from .registry import register_score_inf
from .io_types import AcousticIO, CondIO
from .utils import safe_logit

class _FiLMResBlock(nn.Module):
    def __init__(self, channels: int, cond_channels: int, dilation_t: int = 1, groups: int = 8):
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=(3,3),
            padding=(dilation_t, 1),
            dilation=(dilation_t, 1),
            groups=channels,
            bias=False
        )
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=max(1, min(groups, channels)), num_channels=channels)
        self.act = nn.SiLU()
        self.film = nn.Conv2d(cond_channels, 2 * channels, kernel_size=1, bias=True)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, cond_map: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.pw(y)

        gb = self.film(cond_map)
        gamma, beta = torch.chunk(gb, 2, dim=1)
        y = y * (1.0 + gamma) + beta
        return x + self.res_scale * y

def _stack_cond(cond: CondIO, keys: List[str]) -> torch.Tensor:
    # return (B,C,T,88)
    maps = []
    for k in keys:
        maps.append(getattr(cond, k))
    return torch.stack(maps, dim=1)

@register_score_inf("scrr")
class SCRR(nn.Module):
    """
    Velocity-only SCRR:
    input: acoustic vel_logits (or derived from vel)
    conditioning: cond maps (GT)
    output: vel_corr = clamp(vel0 + alpha*tanh(delta))
    """
    def __init__(
        self,
        in_feats: List[str] = ["vel_logits"],                 # vel/vel_logits or acoustic.extra keys
        cond_keys: List[str] = ["onset", "frame", "exframe"], # required cond maps
        hidden: int = 48,
        n_blocks: int = 8,
        dilations_t: Optional[List[int]] = None,
        alpha: float = 0.2,
        norm_groups: int = 8,
    ):
        super().__init__()
        if not in_feats:
            raise ValueError("SCRR in_feats must be a non-empty list.")
        self.in_feats = in_feats
        self.cond_keys = cond_keys
        self.alpha = alpha

        self.in_channels = len(in_feats)
        self.cond_channels = len(cond_keys)

        if dilations_t is None:
            base = [1,2,4,8]
            dilations_t = (base * ((n_blocks + len(base) - 1)//len(base)))[:n_blocks]
        assert len(dilations_t) == n_blocks

        self.conv_in = nn.Conv2d(self.in_channels, hidden, kernel_size=1, bias=True)
        self.blocks = nn.ModuleList([
            _FiLMResBlock(hidden, self.cond_channels, dilation_t=d, groups=norm_groups)
            for d in dilations_t
        ])
        self.conv_out = nn.Conv2d(hidden, 1, kernel_size=1, bias=True)

    def _get_feat(self, acoustic: AcousticIO, name: str) -> torch.Tensor:
        if name == "vel":
            if acoustic.vel is None:
                if acoustic.vel_logits is None:
                    raise KeyError("acoustic needs vel or vel_logits")
                return torch.sigmoid(acoustic.vel_logits)
            return acoustic.vel
        if name == "vel_logits":
            if acoustic.vel_logits is not None:
                return acoustic.vel_logits
            if acoustic.vel is None:
                raise KeyError("acoustic needs vel or vel_logits")
            return safe_logit(acoustic.vel)
        if name in acoustic.extra:
            return acoustic.extra[name]
        raise KeyError(name)

    def forward(self, acoustic: AcousticIO, cond: CondIO):
        xs = [self._get_feat(acoustic, k) for k in self.in_feats]
        x = torch.stack(xs, dim=1)  # (B,C,T,88)

        cond_map = _stack_cond(cond, self.cond_keys)  # (B,Cc,T,88)

        h = self.conv_in(x)
        for blk in self.blocks:
            h = blk(h, cond_map)

        delta = self.conv_out(h).squeeze(1)  # (B,T,88)

        vel0 = self._get_feat(acoustic, "vel")
        vel_corr = torch.clamp(vel0 + self.alpha * torch.tanh(delta), 0.0, 1.0)

        return {"vel_corr": vel_corr, "delta": delta, "debug": {"vel0": vel0}}
