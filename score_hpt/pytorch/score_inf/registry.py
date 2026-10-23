from __future__ import annotations
from typing import Callable, Dict, Type
import torch.nn as nn

SCORE_INF_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_score_inf(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def deco(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in SCORE_INF_REGISTRY:
            raise KeyError(f"Score-inf method '{name}' already registered.")
        SCORE_INF_REGISTRY[name] = cls
        cls._score_inf_name = name
        return cls
    return deco

def build_score_inf(method: str, params: dict) -> nn.Module:
    if method not in SCORE_INF_REGISTRY:
        raise KeyError(
            f"Unknown score-inf method '{method}'. "
            f"Available: {list(SCORE_INF_REGISTRY.keys())}"
        )
    return SCORE_INF_REGISTRY[method](**params)