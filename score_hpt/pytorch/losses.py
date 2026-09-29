from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


VELOCITY_SCALE = 128.0
DEFAULT_PIANO_SSM_FFT_BASE = (192, 384, 768, 1526, 3072, 6144, 12288)


# -----------------------------------------------------------------------------
# Velocity supervision losses (Route II main path, Route III / IV auxiliary)
# -----------------------------------------------------------------------------


def _get_velocity_target(target_dict):
    return target_dict["velocity_roll"] / VELOCITY_SCALE



def _get_velocity_pred(output_dict):
    if "vel_corr" in output_dict:
        return output_dict["vel_corr"]
    if "velocity_output" in output_dict:
        return output_dict["velocity_output"]
    raise KeyError("velocity prediction not found in output_dict (expected vel_corr or velocity_output)")



def has_supervised_velocity_target(target_dict):
    if target_dict is None:
        return False
    if "velocity_roll" not in target_dict:
        return False
    if target_dict["velocity_roll"] is None:
        return False
    if "has_velocity_target" in target_dict:
        flag = target_dict["has_velocity_target"]
        if torch.is_tensor(flag):
            return bool(torch.any(flag > 0))
        return bool(flag)
    return True



def _align_time_dim(*tensors):
    if not tensors:
        return tensors
    min_steps = min(tensor.size(1) for tensor in tensors)
    if all(tensor.size(1) == min_steps for tensor in tensors):
        return tensors
    return tuple(tensor[:, :min_steps] for tensor in tensors)



def _masked_mean(values, mask):
    values, mask = _align_time_dim(values, mask)
    mask = mask.to(values.dtype)
    denom = torch.sum(mask).clamp_min(1e-8)
    return torch.sum(values * mask) / denom



def _masked_bce(output, target, mask):
    output, target, mask = _align_time_dim(output, target, mask)
    output = torch.clamp(output, 1e-7, 1.0 - 1e-7)
    matrix = F.binary_cross_entropy(output, target, reduction="none")
    return _masked_mean(matrix, mask)



def _masked_mse(output, target, mask):
    output, target, mask = _align_time_dim(output, target, mask)
    return _masked_mean((output - target) ** 2, mask)



def _masked_l1(output, target, mask):
    output, target, mask = _align_time_dim(output, target, mask)
    return _masked_mean(torch.abs(output - target), mask)


############ Velocity-supervised losses (used when GT velocity is available) ############


def _velocity_pointwise_loss(output_dict, target_dict, mask_key, pointwise_loss):
    pred = _get_velocity_pred(output_dict)
    target = _get_velocity_target(target_dict)
    return pointwise_loss(pred, target, target_dict[mask_key])



def velocity_bce(cfg, output_dict, target_dict, cond_dict=None):
    """Velocity regression loss using BCE on onset positions (HPT)."""
    return _velocity_pointwise_loss(output_dict, target_dict, "onset_roll", _masked_bce)



def velocity_mse(cfg, output_dict, target_dict, cond_dict=None):
    """Velocity regression loss using MSE on onset positions (ONF)."""
    return _velocity_pointwise_loss(output_dict, target_dict, "onset_roll", _masked_mse)



def kim_velocity_bce_l1(cfg, output_dict, target_dict, cond_dict=None):
    """BCE + L1 hybrid loss proposed by Kim et al. (ISMIR 2024). Route II default."""
    theta = cfg.loss.kim_loss_alpha
    pred = _get_velocity_pred(output_dict)
    onset_target = _get_velocity_target(target_dict)
    bce_loss = _masked_bce(pred, onset_target, target_dict["frame_roll"])
    l1_loss = _masked_l1(pred, onset_target, target_dict["onset_roll"])
    return theta * bce_loss + (1 - theta) * l1_loss


############ Auxiliary regularizers (Route III / IV) ############


def velocity_prior_loss(cfg, output_dict, target_dict=None):
    """
    Weak anti-collapse prior for proxy-only or proxy-heavy training.

    L = (mean(v) - mu)^2 + relu(var_min - var(v))

    By default the prior is evaluated on onset positions only.
    """
    weight = float(getattr(cfg.loss, "velocity_prior_weight", 0.0) or 0.0)
    pred = _get_velocity_pred(output_dict)
    if weight <= 0:
        return pred.new_tensor(0.0)

    use_onsets_only = bool(getattr(cfg.loss, "velocity_prior_onset_only", True))
    if use_onsets_only and target_dict is not None and "onset_roll" in target_dict:
        mask = target_dict["onset_roll"]
        pred, mask = _align_time_dim(pred, mask)
        values = pred[mask > 0]
    else:
        values = pred.reshape(-1)

    if values.numel() == 0:
        return pred.new_tensor(0.0)

    prior_mean = float(getattr(cfg.loss, "velocity_prior_mean", 0.5))
    prior_min_var = float(getattr(cfg.loss, "velocity_prior_min_var", 0.01))
    mean_term = (values.mean() - prior_mean) ** 2
    var_term = F.relu(pred.new_tensor(prior_min_var) - values.var(unbiased=False))
    return mean_term + var_term


def velocity_saturation_loss(cfg, output_dict, target_dict=None):
    """Penalize saturated velocity predictions near the upper boundary."""
    weight = float(getattr(cfg.loss, "velocity_saturation_weight", 0.0) or 0.0)
    pred = _get_velocity_pred(output_dict)
    if weight <= 0:
        return pred.new_tensor(0.0)

    use_onsets_only = bool(getattr(cfg.loss, "velocity_saturation_onset_only", True))
    if use_onsets_only and target_dict is not None and "onset_roll" in target_dict:
        mask = target_dict["onset_roll"]
        pred, mask = _align_time_dim(pred, mask)
        values = pred[mask > 0]
    else:
        values = pred.reshape(-1)

    if values.numel() == 0:
        return pred.new_tensor(0.0)

    threshold = float(getattr(cfg.loss, "velocity_saturation_threshold", 0.95))
    excess = F.relu(values - pred.new_tensor(threshold))
    return excess.pow(2).mean()


# -----------------------------------------------------------------------------
# Backend audio losses for Route III (Diff-Synth = frozen DDSP renderer)
# -----------------------------------------------------------------------------


def _mean_difference(
    target: torch.Tensor,
    value: torch.Tensor,
    loss_type: str = "L1",
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Common reduction used by the Piano-SSM spectral loss."""
    difference = target - value
    weights = 1.0 if weights is None else weights
    loss_type = str(loss_type).upper()

    if loss_type == "L1":
        return torch.mean(torch.abs(difference * weights))
    if loss_type == "L2":
        return torch.mean((difference ** 2) * weights)
    if loss_type == "COSINE":
        target_flat = target.reshape(target.size(0), -1)
        value_flat = value.reshape(value.size(0), -1)
        cosine_loss = 1.0 - F.cosine_similarity(target_flat, value_flat, dim=-1)
        if torch.is_tensor(weights):
            weights = weights.reshape(weights.size(0), -1).mean(dim=-1)
        return torch.mean(cosine_loss * weights)
    raise ValueError(f"Invalid loss_type: {loss_type}")



def _canonicalize_audio(audio: torch.Tensor) -> torch.Tensor:
    """Convert audio to mono batch-major shape [B, samples]."""
    if not torch.is_tensor(audio):
        raise TypeError(f"Expected a torch.Tensor audio input, got {type(audio)!r}")

    if audio.dim() == 1:
        return audio.unsqueeze(0)
    if audio.dim() == 2:
        return audio
    if audio.dim() != 3:
        raise ValueError(f"Expected audio tensor with 1, 2, or 3 dims, got shape {tuple(audio.shape)}")

    if audio.size(1) == 1:
        return audio.squeeze(1)
    if audio.size(-1) == 1:
        return audio.squeeze(-1)
    if audio.size(1) <= 4:
        return audio.mean(dim=1)
    if audio.size(-1) <= 4:
        return audio.mean(dim=-1)

    raise ValueError(f"Cannot infer mono axis for audio tensor with shape {tuple(audio.shape)}")



def _align_audio_pair(pred_audio: torch.Tensor, target_audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_audio = _canonicalize_audio(pred_audio)
    target_audio = _canonicalize_audio(target_audio)
    min_len = min(pred_audio.size(-1), target_audio.size(-1))
    return pred_audio[..., :min_len], target_audio[..., :min_len]



def _get_audio_loss_cfg(cfg, section_name: Optional[str] = None):
    audio_cfg = getattr(getattr(cfg, "backend", None), "audio_loss", None)
    if audio_cfg is None:
        return None
    if section_name is None:
        return audio_cfg
    nested = getattr(audio_cfg, section_name, None)
    return nested if nested is not None else audio_cfg



def _piano_ssm_default_fft_sizes(sample_rate: int) -> Tuple[int, ...]:
    """Replicate the Piano-SSM repo default scaling for the spectral loss."""
    sr = max(1, int(sample_rate))
    values = [max(2, int(sr / 48000.0 * base)) for base in DEFAULT_PIANO_SSM_FFT_BASE]
    deduped: list[int] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)



def get_audio_loss_name(cfg) -> str:
    """Return the canonical Route III backend audio-loss name."""
    audio_cfg = getattr(getattr(cfg, "backend", None), "audio_loss", None)
    raw_name = None
    if audio_cfg is not None:
        raw_name = getattr(audio_cfg, "type", None)
    name = str(raw_name or "piano_ssm_spectral_plus_log_rms").strip().lower()
    if name not in AUDIO_LOSS_TYPES:
        available = ", ".join(sorted(AUDIO_LOSS_TYPES.keys()))
        raise ValueError(f"Unknown backend.audio_loss.type: {name!r}. Available: {available}")
    return name


class _BaseAudioLoss(nn.Module):
    """Shared utilities for Route III proxy audio losses."""

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = float(eps)
        self._window_cache: Dict[Tuple[int, str, str], torch.Tensor] = {}

    def _window(self, win_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(win_length), str(device), str(dtype))
        if key not in self._window_cache:
            self._window_cache[key] = torch.hann_window(int(win_length), device=device, dtype=dtype)
        return self._window_cache[key]

    def _audio_stats(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "audio_rms_pred": pred_audio.pow(2).mean().sqrt().detach(),
            "audio_rms_target": target_audio.pow(2).mean().sqrt().detach(),
        }


class GlobalLogRMSLoss(_BaseAudioLoss):
    """
    Clip-level loudness loss on log-RMS (dB-RMS by default).

    Used as the small auxiliary term inside piano_ssm_spectral_plus_log_rms.
    """

    def __init__(
        self,
        loss_type: str = "L1",
        db_scale: bool = True,
        eps: float = 1e-7,
    ) -> None:
        super().__init__(eps=eps)
        self.loss_type = str(loss_type).upper()
        self.db_scale = bool(db_scale)

    def _log_rms(self, audio: torch.Tensor) -> torch.Tensor:
        rms = audio.pow(2).mean(dim=-1).clamp_min(self.eps).sqrt()
        if self.db_scale:
            return 20.0 * torch.log10(rms)
        return torch.log(rms)

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor):
        pred_audio, target_audio = _align_audio_pair(pred_audio, target_audio)
        pred_log_rms = self._log_rms(pred_audio)
        target_log_rms = self._log_rms(target_audio)
        loss = _mean_difference(target_log_rms, pred_log_rms, loss_type=self.loss_type)

        stats = {
            "log_rms_loss": loss.detach(),
            "log_rms_pred": pred_log_rms.mean().detach(),
            "log_rms_target": target_log_rms.mean().detach(),
        }
        stats.update(self._audio_stats(pred_audio, target_audio))
        return loss, stats


class CompositeAudioLoss(nn.Module):
    """Weighted sum of named audio losses."""

    def __init__(self, named_losses):
        super().__init__()
        self.loss_modules = nn.ModuleDict({name: module for name, module, _ in named_losses})
        self.loss_weights = {name: float(weight) for name, _, weight in named_losses}

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor):
        pred_audio, target_audio = _align_audio_pair(pred_audio, target_audio)
        total_loss = pred_audio.new_tensor(0.0)
        stats: Dict[str, torch.Tensor] = {}

        for name, module in self.loss_modules.items():
            weight = self.loss_weights[name]
            if weight <= 0:
                continue
            sub_loss, sub_stats = module(pred_audio, target_audio)
            weighted_loss = weight * sub_loss
            total_loss = total_loss + weighted_loss
            stats[f"{name}_raw"] = sub_loss.detach()
            stats[f"{name}_weighted"] = weighted_loss.detach()
            for key, value in sub_stats.items():
                if torch.is_tensor(value):
                    stats[f"{name}_{key}"] = value.detach()
                else:
                    stats[f"{name}_{key}"] = value

        return total_loss, stats


class PianoSSMSpectralLoss(_BaseAudioLoss):
    """
    Piano-SSM / DDSP-style multi-resolution spectral loss.

    Mirrors the Piano-SSM repository implementation of SpectralLoss and keeps
    the overlap-based STFT padding behaviour intact.
    """

    def __init__(
        self,
        fft_sizes: Optional[Iterable[int]] = None,
        loss_type: str = "L1",
        mag_weight: float = 1.0,
        logmag_weight: float = 0.0,
        overlap: float = 0.75,
        sample_rate: Optional[int] = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(eps=eps)
        if fft_sizes is None:
            if sample_rate is None:
                fft_sizes = (2048, 1024, 512, 256, 128, 64)
            else:
                fft_sizes = _piano_ssm_default_fft_sizes(int(sample_rate))
        self.fft_sizes = tuple(int(size) for size in fft_sizes)
        self.loss_type = str(loss_type).upper()
        self.mag_weight = float(mag_weight)
        self.logmag_weight = float(logmag_weight)
        self.overlap = float(overlap)

    def compute_mag(self, audio: torch.Tensor, size: int) -> torch.Tensor:
        audio = _canonicalize_audio(audio)
        frame_length = max(2, int(size))
        frame_step = max(1, int(round(frame_length * (1.0 - self.overlap))))
        window = self._window(frame_length, audio.device, audio.dtype)

        total_frames = max(1, (audio.size(-1) + frame_step - 1) // frame_step)
        padded_length = (total_frames - 1) * frame_step + frame_length
        pad_size = max(0, padded_length - audio.size(-1))
        audio = F.pad(audio, (0, pad_size))

        stft_output = torch.stft(
            audio,
            n_fft=frame_length,
            hop_length=frame_step,
            win_length=frame_length,
            window=window,
            center=False,
            pad_mode="constant",
            return_complex=True,
        )
        return stft_output.abs().permute(0, 2, 1).clamp_min(self.eps)

    def mean_difference(
        self,
        target: torch.Tensor,
        value: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _mean_difference(target, value, loss_type=self.loss_type, weights=weights)

    def safe_log(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=self.eps))

    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        epoch: int = 0,
        weights: Optional[torch.Tensor] = None,
    ):
        del epoch  # kept for interface compatibility
        pred_audio, target_audio = _align_audio_pair(pred_audio, target_audio)

        total_loss = pred_audio.new_tensor(0.0)
        stats: Dict[str, torch.Tensor] = {}

        for size in self.fft_sizes:
            target_mag = self.compute_mag(target_audio, size)
            pred_mag = self.compute_mag(pred_audio, size)

            if self.mag_weight > 0:
                total_loss = total_loss + self.mag_weight * self.mean_difference(target_mag, pred_mag, weights=weights)

            if self.logmag_weight > 0:
                target_logmag = self.safe_log(target_mag)
                pred_logmag = self.safe_log(pred_mag)
                total_loss = total_loss + self.logmag_weight * self.mean_difference(target_logmag, pred_logmag, weights=weights)

        stats.update(self._audio_stats(pred_audio, target_audio))
        return total_loss, stats


class PianoSSMSpectralPlusLogRMSLoss(CompositeAudioLoss):
    """
    Route III default proxy loss.

    Main term: Piano-SSM multi-resolution spectral loss (sourced from DDSP 2020.).
    Auxiliary term: clip-level log-RMS loudness.

    Keeps strong spectral supervision and adds a small loudness bias.
    """

    def __init__(
        self,
        spectral_loss: PianoSSMSpectralLoss,
        log_rms_loss: GlobalLogRMSLoss,
        spectral_weight: float = 1.0,
        log_rms_weight: float = 0.05,
    ) -> None:
        super().__init__([
            ("spectral", spectral_loss, spectral_weight),
            ("log_rms", log_rms_loss, log_rms_weight),
        ])


AUDIO_LOSS_TYPES = {
    "piano_ssm_spectral": PianoSSMSpectralLoss,
    "piano_ssm_spectral_plus_log_rms": PianoSSMSpectralPlusLogRMSLoss,
}



def _build_piano_ssm_spectral_loss_from_cfg(cfg, sample_rate: int) -> PianoSSMSpectralLoss:
    audio_cfg = _get_audio_loss_cfg(cfg, "piano_ssm_spectral")
    fft_sizes = getattr(audio_cfg, "fft_sizes", None)
    if fft_sizes is not None:
        fft_sizes = tuple(int(v) for v in fft_sizes)
        if len(fft_sizes) == 0:
            fft_sizes = None
    return PianoSSMSpectralLoss(
        fft_sizes=fft_sizes,
        loss_type=str(getattr(audio_cfg, "loss_type", "L1")),
        mag_weight=float(getattr(audio_cfg, "mag_weight", 1.0)),
        logmag_weight=float(getattr(audio_cfg, "logmag_weight", 1.0)),
        overlap=float(getattr(audio_cfg, "overlap", 0.75)),
        sample_rate=sample_rate,
        eps=float(getattr(audio_cfg, "eps", 1e-5)),
    )


def _build_log_rms_aux_loss_from_cfg(cfg) -> GlobalLogRMSLoss:
    audio_cfg = _get_audio_loss_cfg(cfg, "piano_ssm_spectral_plus_log_rms")
    return GlobalLogRMSLoss(
        loss_type=str(getattr(audio_cfg, "loss_type", "L1")),
        db_scale=bool(getattr(audio_cfg, "db_scale", True)),
        eps=float(getattr(audio_cfg, "eps", 1e-7)),
    )


def build_audio_loss(cfg, sample_rate_override=None, frame_rate_override=None):
    """Build the Route III backend audio loss selected by `backend.audio_loss.type`."""
    del frame_rate_override  # no longer used; kept for interface compatibility
    audio_loss_name = get_audio_loss_name(cfg)
    if sample_rate_override is not None:
        sample_rate = int(sample_rate_override)
    else:
        sample_rate = int(
            getattr(getattr(cfg, "backend", None), "sample_rate", getattr(getattr(cfg, "feature", None), "sample_rate", 16000))
        )

    if audio_loss_name == "piano_ssm_spectral":
        return _build_piano_ssm_spectral_loss_from_cfg(cfg, sample_rate=sample_rate)

    if audio_loss_name == "piano_ssm_spectral_plus_log_rms":
        audio_cfg = _get_audio_loss_cfg(cfg, "piano_ssm_spectral_plus_log_rms")
        spectral_loss = _build_piano_ssm_spectral_loss_from_cfg(cfg, sample_rate=sample_rate)
        log_rms_loss = _build_log_rms_aux_loss_from_cfg(cfg)
        return PianoSSMSpectralPlusLogRMSLoss(
            spectral_loss=spectral_loss,
            log_rms_loss=log_rms_loss,
            spectral_weight=float(getattr(audio_cfg, "spectral_weight", 1.0)),
            log_rms_weight=float(getattr(audio_cfg, "log_rms_weight", 0.05)),
        )

    available = ", ".join(sorted(AUDIO_LOSS_TYPES.keys()))
    raise ValueError(f"Unknown backend.audio_loss.type: {audio_loss_name!r}. Available: {available}")


# -----------------------------------------------------------------------------
# Loss function selector for velocity supervision
# -----------------------------------------------------------------------------


def get_loss_func(cfg, loss_type=None):
    """
    Return a callable with unified signature:
      fn(cfg, output_dict, target_dict, cond_dict=None) -> loss

    Selection order:
    - explicit loss_type if provided
    - cfg.loss.loss_type
    """
    selected_loss_type = loss_type if loss_type is not None else cfg.loss.loss_type
    loss_map = {
        "velocity_bce": velocity_bce,
        "velocity_mse": velocity_mse,
        "kim_bce_l1": kim_velocity_bce_l1,
    }
    if selected_loss_type in loss_map:
        return loss_map[selected_loss_type]

    raise ValueError(f"Incorrect loss_type: {selected_loss_type!r}")
