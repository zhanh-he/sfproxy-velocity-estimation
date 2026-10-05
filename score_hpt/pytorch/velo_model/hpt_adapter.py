from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from feature_extractor import get_feature_extractor_and_bins
from score_inf.utils import safe_logit

from .base import BaseAdapter, ensure_btp88, register_adapter
from .pretrained_utils import (
    resolve_frontend_pretrained_checkpoint,
    select_prefixed_substate,
    unwrap_checkpoint_state_dict,
)


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_gru(rnn):
    """Initialize GRU weights and biases for better convergence."""

    def _concat_init(tensor, inits):
        fan_in = tensor.shape[0] // len(inits)
        for i, fn in enumerate(inits):
            fn(tensor[i * fan_in : (i + 1) * fan_in])

    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
        bound = math.sqrt(3 / fan_in)
        nn.init.uniform_(tensor, -bound, bound)

    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, f"weight_ih_l{i}"),
            [_inner_uniform, _inner_uniform, _inner_uniform],
        )
        nn.init.constant_(getattr(rnn, f"bias_ih_l{i}"), 0.0)

        _concat_init(
            getattr(rnn, f"weight_hh_l{i}"),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_],
        )
        nn.init.constant_(getattr(rnn, f"bias_hh_l{i}"), 0.0)


class HPTConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels, momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum)
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)

    def forward(self, input):
        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(1, 2))
        return x


class HPT2020Module(nn.Module):
    def __init__(self, classes_num, input_shape, momentum):
        super().__init__()
        self.conv_block1 = HPTConvBlock(
            in_channels=1,
            out_channels=48,
            momentum=momentum,
        )
        self.conv_block2 = HPTConvBlock(
            in_channels=48,
            out_channels=64,
            momentum=momentum,
        )
        self.conv_block3 = HPTConvBlock(
            in_channels=64,
            out_channels=96,
            momentum=momentum,
        )
        self.conv_block4 = HPTConvBlock(
            in_channels=96,
            out_channels=128,
            momentum=momentum,
        )
        # Auto Calculate midfeat
        with torch.no_grad():
            dummy = torch.zeros((1, 1, 1000, input_shape))
            x = self.conv_block1(dummy)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            x = self.conv_block4(x)
            midfeat = x.shape[1] * x.shape[3]

        self.fc5 = nn.Linear(in_features=midfeat, out_features=768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)
        self.gru = nn.GRU(
            input_size=768,
            hidden_size=256,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )
        self.fc = nn.Linear(in_features=512, out_features=classes_num, bias=True)
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, input):
        x = self.conv_block1(input)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training)
        (x, _) = self.gru(x)
        x = F.dropout(x, p=0.5, training=self.training)
        output = torch.sigmoid(self.fc(x))
        return output


class Single_Velocity_HPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        sample_rate = cfg.feature.sample_rate
        fft_size = cfg.feature.fft_size
        frames_per_second = cfg.feature.frames_per_second
        audio_feature = cfg.feature.audio_feature
        classes_num = cfg.feature.classes_num
        momentum = 0.01
        self.feature_extractor, self.FRE = get_feature_extractor_and_bins(
            audio_feature,
            sample_rate,
            fft_size,
            frames_per_second,
        )
        self.bn0 = nn.BatchNorm2d(self.FRE, momentum)
        self.velocity_model = HPT2020Module(classes_num, self.FRE, momentum)
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)

    def forward(self, input):
        x = self.feature_extractor(input)
        x = x.unsqueeze(3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        est_velocity = self.velocity_model(x)
        vel_logits = safe_logit(est_velocity)
        return {"velocity_output": est_velocity, "velocity_logits": vel_logits}


def _load_hpt_pretrained_weights(model: nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = unwrap_checkpoint_state_dict(checkpoint)
    model_state = select_prefixed_substate(
        state_dict,
        model.state_dict().keys(),
        prefixes=("base_adapter.model.", "model.", ""),
    )
    model.load_state_dict(model_state, strict=True)


def _maybe_load_hpt_pretrained_weights(model: nn.Module, cfg, model_label: str = "hpt") -> None:
    checkpoint_path = resolve_frontend_pretrained_checkpoint(
        getattr(cfg, "model", None),
        model_label=model_label,
        required=False,
    )
    if checkpoint_path is not None:
        _load_hpt_pretrained_weights(model, checkpoint_path)


@dataclass
class HPTKeySpec:
    dict_keys: Optional[Dict[str, str]] = None


@register_adapter("hpt")
class HPTAdapter(BaseAdapter):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        cfg=None,
        keyspec: Optional[Dict[str, Any]] = None,
        keep_extra: bool = True,
    ):
        if model is None:
            if cfg is None:
                raise ValueError("HPTAdapter: cfg is required when model is None.")
            model = Single_Velocity_HPT(cfg)
            _maybe_load_hpt_pretrained_weights(model, cfg, model_label="hpt")
        super().__init__(model=model)
        ks = keyspec or {}
        self.keyspec = HPTKeySpec(dict_keys=ks.get("dict_keys", None))
        self.keep_extra = keep_extra

    def _forced_key(self, unified: str) -> Optional[str]:
        if self.keyspec.dict_keys and unified in self.keyspec.dict_keys:
            return self.keyspec.dict_keys[unified]
        return None

    def _from_dict(self, out: Dict[str, Any]) -> Dict[str, Any]:
        vel_k = self._forced_key("vel") or "velocity_output"
        vel_logit_k = self._forced_key("vel_logits") or "velocity_logits"

        def get_tensor(k: Optional[str]) -> Optional[torch.Tensor]:
            if k is None:
                return None
            v = out.get(k, None)
            return v if torch.is_tensor(v) else None

        vel_raw = get_tensor(vel_k)
        vel_logit_raw = get_tensor(vel_logit_k)
        if vel_raw is None and vel_logit_raw is None:
            available = sorted(out.keys())
            raise KeyError(
                "HPTAdapter expected at least one velocity tensor, but none found. "
                f"Checked keys: vel='{vel_k}', vel_logits='{vel_logit_k}'. "
                f"Available keys: {available}"
            )

        vel = None if vel_raw is None else torch.clamp(ensure_btp88(vel_raw, "vel"), 0.0, 1.0)
        vel_logits = None if vel_logit_raw is None else ensure_btp88(vel_logit_raw, "vel_logits")

        extra: Dict[str, torch.Tensor] = {}
        if self.keep_extra:
            used = {k for k in [vel_k, vel_logit_k] if k}
            for k, v in out.items():
                if k in used:
                    continue
                if torch.is_tensor(v):
                    extra[k] = v

        return {"vel": vel, "vel_logits": vel_logits, "extra": extra}

    def forward(self, audio: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        out = self.model(audio, *args, **kwargs)
        return self._finalize(self._from_dict(out))


@register_adapter("hpt_pretrained")
class HPTPretrainedAdapter(HPTAdapter):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        cfg=None,
        keyspec: Optional[Dict[str, Any]] = None,
        keep_extra: bool = True,
    ):
        super().__init__(model=model, cfg=cfg, keyspec=keyspec, keep_extra=keep_extra)
