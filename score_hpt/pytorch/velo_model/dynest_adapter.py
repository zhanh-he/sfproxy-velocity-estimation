from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_extractor import get_feature_extractor_and_bins

from .base import BaseAdapter, ensure_btp88, register_adapter


class Block(nn.Module):
    """Two-layer residual CNN block with channel matching skip."""

    def __init__(self, inp: int, out: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inp)
        self.conv1 = nn.Conv2d(inp, out, (3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, (3, 1), padding=(1, 0))
        self.bn3 = nn.BatchNorm2d(out)
        self.skip = nn.Conv2d(inp, out, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.bn1(x))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn3(out)
        return out + self.skip(x)


class SelfAttention(nn.Module):
    """Squeezed temporal self-attention over feature maps."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.squeeze(-1).transpose(-1, -2)
        res, _ = self.att(seq, seq, seq)
        res = res.transpose(-1, -2).unsqueeze(-1)
        return res + x


class SYMultiScaleAttnFlex(nn.Module):
    """Multi-scale CNN encoder with attention and configurable strides."""

    def __init__(self, freq_bins: int, latent_dim: int, temporal_strides=None):
        super().__init__()
        if temporal_strides is None:
            s1, s2 = 3, 3
        elif isinstance(temporal_strides, Sequence) and not isinstance(
            temporal_strides, (str, bytes)
        ):
            s1 = int(temporal_strides[0]) if len(temporal_strides) >= 1 else 1
            s2 = int(temporal_strides[1]) if len(temporal_strides) >= 2 else s1
        else:
            s1 = s2 = int(temporal_strides)
        s1 = max(1, s1)
        s2 = max(1, s2)

        fs = (3, 1)
        ps = (1, 0)

        self.bn0 = nn.BatchNorm2d(freq_bins)
        self.conv1 = nn.Conv2d(freq_bins, freq_bins, (1, 1))
        self.maxpool012 = nn.MaxPool2d((s1, 1), (s1, 1))
        self.conv02 = nn.Conv2d(freq_bins, freq_bins, (1, 1))

        self.block11 = Block(freq_bins, freq_bins * 2)
        self.block12 = Block(freq_bins, freq_bins * 2)

        self.maxpool112 = nn.MaxPool2d((s1, 1), (s1, 1))
        self.dropout12 = nn.Dropout(p=0.2)
        self.maxpool123 = nn.MaxPool2d((s2, 1), (s2, 1))
        self.dropout123 = nn.Dropout(p=0.2)
        self.us121 = nn.ConvTranspose2d(
            freq_bins * 2, freq_bins * 2, kernel_size=(s1, 1), stride=(s1, 1)
        )

        self.conv21 = nn.Conv2d(freq_bins * 2, freq_bins * 2, (1, 2))
        self.conv22 = nn.Conv2d(freq_bins * 2, freq_bins * 2, (1, 2))
        self.conv23 = nn.Conv2d(freq_bins * 2, freq_bins * 2, (1, 1))

        self.block21 = Block(freq_bins * 2, freq_bins * 3)
        self.block22 = Block(freq_bins * 2, freq_bins * 3)
        self.block23 = Block(freq_bins * 2, freq_bins * 3)

        self.self_att23 = SelfAttention(freq_bins * 3, 1)
        self.bn23 = nn.BatchNorm2d(freq_bins * 3)

        self.maxpool212 = nn.MaxPool2d((s2, 1), (s2, 1))
        self.maxpool223 = nn.MaxPool2d((s2, 1), (s2, 1))
        self.dropout22 = nn.Dropout(p=0.2)
        self.dropout23 = nn.Dropout(p=0.2)
        self.us221 = nn.ConvTranspose2d(
            freq_bins * 3, freq_bins * 3, kernel_size=(s2, 1), stride=(s2, 1)
        )
        self.us232 = nn.ConvTranspose2d(
            freq_bins * 3, freq_bins * 3, kernel_size=(s2, 1), stride=(s2, 1)
        )

        self.conv31 = nn.Conv2d(freq_bins * 3, freq_bins * 3, (1, 2))
        self.conv32 = nn.Conv2d(freq_bins * 3, freq_bins * 3, (1, 3))
        self.conv33 = nn.Conv2d(freq_bins * 3, freq_bins * 3, (1, 2))

        self.block31 = Block(freq_bins * 3, freq_bins * 3)
        self.block32 = Block(freq_bins * 3, freq_bins * 3)
        self.block33 = Block(freq_bins * 3, freq_bins * 3)

        self.bn31 = nn.BatchNorm2d(freq_bins * 3)
        self.bn32 = nn.BatchNorm2d(freq_bins * 3)
        self.bn33 = nn.BatchNorm2d(freq_bins * 3)
        self.relu31 = nn.ReLU(inplace=True)
        self.relu32 = nn.ReLU(inplace=True)
        self.relu33 = nn.ReLU(inplace=True)

        self.maxpool312 = nn.MaxPool2d((s1, 1), (s1, 1))
        self.dropout312 = nn.Dropout(p=0.2)
        self.us321 = nn.ConvTranspose2d(
            freq_bins * 3, freq_bins * 3, kernel_size=(s1, 1), stride=(s1, 1)
        )
        self.us332 = nn.ConvTranspose2d(
            freq_bins * 3, freq_bins * 3, kernel_size=(s2, 1), stride=(s2, 1)
        )

        self.self_att = SelfAttention(freq_bins * 3, 1)
        self.bn4 = nn.BatchNorm2d(freq_bins * 3)

        self.conv41 = nn.Conv2d(freq_bins * 3, freq_bins * 3, (1, 2))
        self.conv42 = nn.Conv2d(freq_bins * 3, freq_bins * 3, (1, 3))

        self.block41 = Block(freq_bins * 3, freq_bins * 2)
        self.block42 = Block(freq_bins * 3, freq_bins * 2)

        self.us421 = nn.ConvTranspose2d(
            freq_bins * 2, freq_bins * 2, kernel_size=(s1, 1), stride=(s1, 1)
        )

        self.conv51 = nn.Conv2d(freq_bins * 2, freq_bins * 2, (1, 2))
        self.block51 = Block(freq_bins * 2, freq_bins)

        self.bn51 = nn.BatchNorm2d(freq_bins)
        self.relu51 = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(freq_bins, latent_dim, fs, padding=ps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.bn0(x)
        x02 = self.maxpool012(x0)
        x02 = self.block12(self.conv02(x02))

        x1 = self.conv1(x0)
        x11 = self.block11(x1)

        x11_n = torch.cat([x11, self.us121(x02, output_size=x11.shape)], dim=-1)

        x12 = self.dropout12(self.maxpool112(x11))
        x12_n = torch.cat([x02, x12], dim=-1)
        x13 = self.dropout123(self.maxpool123(x02))

        x21 = self.block21(self.conv21(x11_n))
        x22 = self.block22(self.conv22(x12_n))
        x23 = self.block23(self.conv23(x13))

        x21_n = torch.cat([x21, self.us221(x22, output_size=x21.shape)], dim=-1)
        x22_n = torch.cat(
            [x22, self.dropout22(self.maxpool212(x21)), self.us232(x23, output_size=x22.shape)],
            dim=-1,
        )
        x23 = self.bn23(self.self_att23(x23))
        x23_n = torch.cat([x23, self.dropout23(self.maxpool223(x22))], dim=-1)

        x31 = self.relu31(self.bn31(self.block31(self.conv31(x21_n))))
        x32 = self.relu32(self.bn32(self.block32(self.conv32(x22_n))))
        x33 = self.relu33(self.bn33(self.block33(self.conv33(x23_n))))
        x33 = self.bn4(self.self_att(x33))

        x31_n = torch.cat([x31, self.us321(x32, output_size=x31.shape)], dim=-1)
        x32_n = torch.cat(
            [x32, self.us332(x33, output_size=x32.shape), self.dropout312(self.maxpool312(x31))],
            dim=-1,
        )

        x41 = self.block41(self.conv41(x31_n))
        x42 = self.block42(self.conv42(x32_n))

        x51_n = torch.cat([x41, self.us421(x42, output_size=x41.shape)], dim=-1)
        x51 = self.relu51(self.bn51(self.block51(self.conv51(x51_n))))

        return self.conv_out(x51)


class DynestAudioCNN(nn.Module):
    """Single-input AudioCNN tailored for predicting MIDI velocities."""

    def __init__(self, cfg):
        super().__init__()
        sr = cfg.feature.sample_rate
        fft_size = cfg.feature.fft_size
        fps = cfg.feature.frames_per_second
        audio_feature = cfg.feature.audio_feature
        classes_num = cfg.feature.classes_num

        self.feature_extractor, self.freq_bins = get_feature_extractor_and_bins(
            audio_feature,
            sr,
            fft_size,
            fps,
        )
        self.bn0 = nn.BatchNorm2d(self.freq_bins, momentum=0.01)
        self.adapt_conv = None

        temporal_cfg = getattr(cfg, "dynest", None) or getattr(cfg, "cnn", None) or cfg.model
        temporal_strides = getattr(
            temporal_cfg,
            "temporal_scale",
            getattr(temporal_cfg, "temporal_strides", None),
        )

        self.encoder = SYMultiScaleAttnFlex(
            self.freq_bins,
            classes_num,
            temporal_strides=temporal_strides,
        )

    def _ensure_channel_dim(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.freq_bins and x.shape[1] != self.freq_bins:
            x = x.transpose(1, 2)

        n_channels = x.shape[1]
        if n_channels == self.freq_bins:
            return x

        if n_channels == 1 and self.freq_bins > 1:
            return x.repeat(1, self.freq_bins, 1)

        if self.freq_bins == 1 and n_channels > 1:
            return x.mean(dim=1, keepdim=True)

        if (
            self.adapt_conv is None
            or self.adapt_conv.in_channels != n_channels
            or self.adapt_conv.out_channels != self.freq_bins
        ):
            self.adapt_conv = nn.Conv1d(
                n_channels,
                self.freq_bins,
                kernel_size=1,
                bias=False,
            ).to(x.device)
        return self.adapt_conv(x)

    def forward(self, waveform: torch.Tensor, target_len: Optional[int] = None):
        features = self.feature_extractor(waveform)
        if features.dim() == 2:
            features = features.unsqueeze(0)
        if features.dim() != 3:
            raise RuntimeError(
                f"Unexpected feature shape {tuple(features.shape)}; expected 3 dims [B, F, T]"
            )

        features = self._ensure_channel_dim(features)
        features = features.unsqueeze(3)
        features = self.bn0(features)
        logits = self.encoder(features)
        logits = logits.squeeze(-1).transpose(1, 2)

        if target_len is not None:
            logits = logits[:, :target_len, :]

        velocity = torch.sigmoid(logits)
        return {"velocity_output": velocity, "velocity_logits": logits}


@dataclass
class DynestKeySpec:
    dict_keys: Optional[Dict[str, str]] = None


@register_adapter("dynest")
class DynestAdapter(BaseAdapter):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        cfg=None,
        keyspec: Optional[Dict[str, Any]] = None,
        keep_extra: bool = True,
    ):
        if model is None:
            if cfg is None:
                raise ValueError("DynestAdapter: cfg is required when model is None.")
            model = DynestAudioCNN(cfg)
        super().__init__(model=model)
        ks = keyspec or {}
        self.keyspec = DynestKeySpec(dict_keys=ks.get("dict_keys", None))
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
                "DynestAdapter expected at least one velocity tensor, but none found. "
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
