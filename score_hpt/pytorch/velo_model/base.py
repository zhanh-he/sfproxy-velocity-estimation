# pytorch/adapters/base.py
from __future__ import annotations

from typing import Any, Dict, Optional, Callable, Type
import torch
import torch.nn as nn

from score_inf.utils import safe_logit

AcousticDict = Dict[str, Any]


# -------------------------
# Shape helpers
# -------------------------
def ensure_btp88(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Ensure x is (B, T, 88). Accepts:
      - (B, T, 88)
      - (B, 88, T) -> transpose
      - (B, T, 88, 1) or (B, T, 1, 88) -> squeeze then fix
    """
    if x is None:
        return x
    if not torch.is_tensor(x):
        raise TypeError(f"{name}: expected torch.Tensor, got {type(x)}")

    if x.dim() == 4 and (x.shape[-1] == 1 or x.shape[-2] == 1):
        x = x.squeeze(-1) if x.shape[-1] == 1 else x.squeeze(-2)

    if x.dim() != 3:
        raise ValueError(f"{name}: expected 3D tensor, got shape={tuple(x.shape)}")

    b, a, c = x.shape
    if c == 88:
        return x
    if a == 88:
        return x.transpose(1, 2).contiguous()
    raise ValueError(f"{name}: cannot infer (B,T,88) from shape={tuple(x.shape)}")


def clamp_prob(x: torch.Tensor, name: str) -> torch.Tensor:
    x = ensure_btp88(x, name)
    return torch.clamp(x, 0.0, 1.0)


def fill_vel_and_logits(acoustic: AcousticDict) -> AcousticDict:
    """
    Ensure acoustic contains BOTH:
      - vel: prob in [0,1]
      - vel_logits: logits
    by deriving one from the other if missing.
    """
    vel = acoustic.get("vel", None)
    vel_logits = acoustic.get("vel_logits", None)

    if vel is None and vel_logits is None:
        raise KeyError("Adapter output must include at least one of: 'vel' or 'vel_logits'.")

    if vel is not None:
        acoustic["vel"] = clamp_prob(vel, "vel")

    if vel_logits is not None:
        acoustic["vel_logits"] = ensure_btp88(vel_logits, "vel_logits")

    if acoustic.get("vel") is None:
        acoustic["vel"] = torch.sigmoid(acoustic["vel_logits"])
    if acoustic.get("vel_logits") is None:
        acoustic["vel_logits"] = safe_logit(acoustic["vel"])

    return acoustic


# -------------------------
# Base adapter
# -------------------------
class BaseAdapter(nn.Module):
    """
    Adapter wraps a base acoustic model and returns unified velocity-only acoustic dict.

    Required:
      - vel OR vel_logits: (B,T,88)

    Optional:
      - extra: dict[str, Tensor]
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor, *args, **kwargs) -> AcousticDict:  # pragma: no cover
        raise NotImplementedError

    def _finalize(self, acoustic: AcousticDict) -> AcousticDict:
        if "extra" not in acoustic or acoustic["extra"] is None:
            acoustic["extra"] = {}
        acoustic = fill_vel_and_logits(acoustic)
        return acoustic


# -------------------------
# Registry + builder
# -------------------------
ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {}


def register_adapter(name: str) -> Callable[[Type[BaseAdapter]], Type[BaseAdapter]]:
    def deco(cls: Type[BaseAdapter]) -> Type[BaseAdapter]:
        if name in ADAPTER_REGISTRY:
            raise KeyError(f"Adapter '{name}' already registered.")
        ADAPTER_REGISTRY[name] = cls
        cls._adapter_name = name
        return cls
    return deco


def build_adapter(adapter_cfg: Dict[str, Any], model: Optional[nn.Module] = None, cfg: Optional[Any] = None) -> BaseAdapter:
    if "type" not in adapter_cfg:
        raise KeyError("adapter config must contain 'type'.")

    name = adapter_cfg["type"]
    params = adapter_cfg.get("params", {}) or {}

    if name not in ADAPTER_REGISTRY:
        raise KeyError(f"Unknown adapter type '{name}'. Available: {list(ADAPTER_REGISTRY.keys())}")

    cls = ADAPTER_REGISTRY[name]
    try:
        return cls(model=model, cfg=cfg, **params)
    except TypeError:
        # Backward compatibility: adapter may not accept cfg
        return cls(model=model, **params)
