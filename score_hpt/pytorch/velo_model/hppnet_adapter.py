from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_extractor import get_feature_extractor_and_bins
from score_inf.utils import safe_logit

from .base import BaseAdapter, ensure_btp88, register_adapter

DEFAULT_MODEL_SIZE = 128


class BiLSTM(nn.Module):
    """Bidirectional LSTM with chunked inference copied from HPPNet."""

    inference_chunk_length = 512

    def __init__(self, input_features: int, recurrent_features: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_features,
            recurrent_features,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.rnn(x)[0]

        batch_size, sequence_length, _ = x.shape
        hidden_size = self.rnn.hidden_size
        num_directions = 2 if self.rnn.bidirectional else 1

        h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
        c = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
        output = torch.zeros(
            batch_size,
            sequence_length,
            num_directions * hidden_size,
            device=x.device,
        )

        slices = range(0, sequence_length, self.inference_chunk_length)
        for start in slices:
            end = start + self.inference_chunk_length
            output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

        if self.rnn.bidirectional:
            h.fill_(0)
            c.fill_(0)
            for start in reversed(slices):
                end = start + self.inference_chunk_length
                result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                output[:, start:end, hidden_size:] = result[:, :, hidden_size:]
        return output


class FreqGroupLSTM(nn.Module):
    """Apply an LSTM independently on every frequency bin."""

    def __init__(self, channel_in: int, channel_out: int, lstm_size: int) -> None:
        super().__init__()
        self.channel_out = channel_out
        self.lstm = BiLSTM(channel_in, lstm_size // 2)
        self.linear = nn.Linear(lstm_size, channel_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T, F]
        b, _, t, freq = x.size()
        x = torch.permute(x, [0, 3, 2, 1]).reshape(b * freq, t, -1)
        x = self.linear(self.lstm(x))
        x = x.reshape(b, freq, t, self.channel_out)
        x = torch.permute(x, [0, 3, 2, 1])
        return torch.sigmoid(x)


class HarmonicDilatedConv(nn.Module):
    """Stack of dilated convolutions tuned to harmonic partials."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        dilations = [48, 76, 96, 111, 124, 135, 144, 152]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    c_in,
                    c_out,
                    kernel_size=[1, 3],
                    padding="same",
                    dilation=[1, d],
                )
                for d in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = sum(conv(x) for conv in self.convs)
        return torch.relu(out)


class CNNTrunk(nn.Module):
    """Shared convolutional backbone from HPPNet."""

    def __init__(self, c_in: int = 1, c_har: int = 16, embedding: int = 128) -> None:
        super().__init__()
        self.block_1 = self._conv_block(c_in, c_har, kernel_size=7)
        self.block_2 = self._conv_block(c_har, c_har, kernel_size=7)
        self.block_2_5 = self._conv_block(c_har, c_har, kernel_size=7)
        self.conv_3 = HarmonicDilatedConv(c_har, embedding)
        self.block_4 = self._conv_block(
            embedding,
            embedding,
            pool_size=[1, 4],
            dilation=[1, 48],
        )
        self.block_5 = self._conv_block(embedding, embedding, dilation=[1, 12])
        self.block_6 = self._conv_block(embedding, embedding, kernel_size=[5, 1])
        self.block_7 = self._conv_block(embedding, embedding, kernel_size=[5, 1])
        self.block_8 = self._conv_block(embedding, embedding, kernel_size=[5, 1])

    @staticmethod
    def _conv_block(
        channel_in: int,
        channel_out: int,
        kernel_size: Iterable[int] = (1, 3),
        pool_size: Tuple[int, int] | None = None,
        dilation: Iterable[int] = (1, 1),
    ) -> nn.Sequential:
        layers = [
            nn.Conv2d(
                channel_in,
                channel_out,
                kernel_size=kernel_size,
                padding="same",
                dilation=dilation,
            ),
            nn.ReLU(),
        ]
        if pool_size:
            layers.append(nn.MaxPool2d(pool_size))
        layers.append(nn.InstanceNorm2d(channel_out))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_2_5(x)
        x = self.conv_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)
        x = self.block_8(x)
        return x


class SubNet(nn.Module):
    """Single-head subnet used for velocity prediction."""

    def __init__(self, model_size: int, head_names: Iterable[str]) -> None:
        super().__init__()
        self.trunk = CNNTrunk(c_in=1, c_har=16, embedding=model_size)
        self.heads = nn.ModuleDict(
            {name: FreqGroupLSTM(model_size, 1, model_size) for name in head_names}
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.trunk(x)
        outputs = {}
        for name, head in self.heads.items():
            logits = head(features)
            outputs[name] = torch.clamp(logits, min=1e-7, max=1 - 1e-7)
        return outputs


class HPPNet_SP(nn.Module):
    """Velocity-only adaptation of HPPNet (Kim et al., ISMIR 2024)."""

    def __init__(self, cfg) -> None:
        super().__init__()
        sample_rate = cfg.feature.sample_rate
        frames_per_second = cfg.feature.frames_per_second
        feature_type = getattr(cfg.feature, "audio_feature", "cqt")
        fft_size = cfg.feature.fft_size
        self.frontend, _ = get_feature_extractor_and_bins(
            feature_type,
            sample_rate,
            fft_size,
            frames_per_second,
        )

        model_size = int(getattr(cfg.model, "hppnet_model_size", DEFAULT_MODEL_SIZE))
        self.velocity_subnet = SubNet(model_size, head_names=("velocity",))

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        spec = self.frontend(waveform)
        spec_db = spec.permute(0, 2, 1).unsqueeze(1)

        outputs = self.velocity_subnet(spec_db)
        velocity = outputs["velocity"].squeeze(1)  # [B, T, pitch_bins]
        if velocity.size(-1) != 88:
            velocity = F.interpolate(
                velocity.unsqueeze(1),
                size=(velocity.size(-2), 88),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        safe_logits = safe_logit(velocity)
        return {"velocity_output": velocity, "velocity_logits": safe_logits}


@dataclass
class HPPNetKeySpec:
    dict_keys: Optional[Dict[str, str]] = None


@register_adapter("hppnet")
class HPPNetAdapter(BaseAdapter):
    """HPPNet velocity adapter. Output: {vel, vel_logits, extra}."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        cfg=None,
        keyspec: Optional[Dict[str, Any]] = None,
        keep_extra: bool = True,
    ):
        if model is None:
            if cfg is None:
                raise ValueError("HPPNetAdapter: cfg is required when model is None.")
            model = HPPNet_SP(cfg)
        super().__init__(model=model)
        ks = keyspec or {}
        self.keyspec = HPPNetKeySpec(dict_keys=ks.get("dict_keys", None))
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
                "HPPNetAdapter expected at least one velocity tensor, but none found. "
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
