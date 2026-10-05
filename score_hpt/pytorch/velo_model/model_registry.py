from __future__ import annotations

from typing import Dict, Type
import torch.nn as nn

from .hpt_adapter import Single_Velocity_HPT
from .dynest_adapter import DynestAudioCNN
from .hppnet_adapter import HPPNet_SP

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "hpt": Single_Velocity_HPT,
    "hpt_pretrained": Single_Velocity_HPT,
    "dynest": DynestAudioCNN,
    "hppnet": HPPNet_SP,
}


def build_model(cfg) -> nn.Module:
    model_type = cfg.model.type
    if model_type not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown base model type '{cfg.model.type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type](cfg)
