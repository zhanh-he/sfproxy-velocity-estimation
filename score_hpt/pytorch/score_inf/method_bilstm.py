from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

from .registry import register_score_inf
from .io_types import AcousticIO, CondIO
from .utils import safe_logit


@register_score_inf("bilstm")
class ScoreInformedBiLSTM(nn.Module):
    """
    Velocity-only + GT cond (onset/frame/exframe) BiLSTM correction.
    """

    def __init__(
        self,
        in_feats: List[str] = ["vel_logits"],            # "vel" / "vel_logits" / acoustic.extra keys
        cond_keys: List[str] = ["onset", "frame", "exframe"],
        hidden: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        mode: str = "direct",                            # "direct" or "residual"
        alpha: float = 0.2,
    ):
        super().__init__()
        assert mode in ("direct", "residual")

        self.in_feats = in_feats
        self.cond_keys = cond_keys
        self.hidden = hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.mode = mode
        self.alpha = alpha

        n_ch = len(in_feats) + len(cond_keys)
        input_size = 88 * n_ch

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden * 2, 88)

    def _vel(self, acoustic: AcousticIO) -> torch.Tensor:
        if acoustic.vel is not None:
            return acoustic.vel
        if acoustic.vel_logits is not None:
            return torch.sigmoid(acoustic.vel_logits)
        raise KeyError("acoustic must provide vel or vel_logits")

    def _vel_logits(self, acoustic: AcousticIO) -> torch.Tensor:
        if acoustic.vel_logits is not None:
            return acoustic.vel_logits
        if acoustic.vel is not None:
            return safe_logit(acoustic.vel)
        raise KeyError("acoustic must provide vel or vel_logits")

    def _get_feat(self, acoustic: AcousticIO, name: str) -> torch.Tensor:
        if name == "vel":
            return self._vel(acoustic)
        if name == "vel_logits":
            return self._vel_logits(acoustic)
        if name in acoustic.extra:
            return acoustic.extra[name]
        raise KeyError(f"Unknown acoustic feature '{name}'")

    def _get_cond_map(self, cond: CondIO, name: str) -> torch.Tensor:
        return getattr(cond, name)

    def forward(self, acoustic: AcousticIO, cond: CondIO):
        feats = [self._get_feat(acoustic, k) for k in self.in_feats]
        feats += [self._get_cond_map(cond, k) for k in self.cond_keys]
        x = torch.cat(feats, dim=-1)  # (B,T,88*(...))

        h, _ = self.rnn(x)
        out = self.fc(h)

        if self.mode == "direct":
            vel_corr = torch.sigmoid(out)
            delta = None
            logits = out
        else:
            vel0 = self._vel(acoustic)
            delta = out
            vel_corr = torch.clamp(vel0 + self.alpha * torch.tanh(delta), 0.0, 1.0)
            logits = None

        return {"vel_corr": vel_corr, "delta": delta, "logits": logits}
