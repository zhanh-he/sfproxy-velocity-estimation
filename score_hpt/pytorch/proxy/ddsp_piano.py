from __future__ import annotations

from pathlib import Path
import importlib
import sys
from typing import Dict, Optional

import torch

from pytorch_utils import move_data_to_device
from .common import (
    align_time_dim,
    choose_crop_bounds,
    crop_audio,
    crop_roll_time_first,
    derive_channel_onsets,
    resample_roll_btp,
    resample_waveform,
    resolve_backend_segment_seconds,
    stable_voice_assignment,
)


class DDSPPianoProxy:
    """Frozen DDSP-Piano renderer driven by predicted onset velocities."""

    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.src_sample_rate = int(cfg.feature.sample_rate)
        self.src_frames_per_second = float(cfg.feature.frames_per_second)
        self.segment_seconds = float(cfg.feature.segment_seconds)
        self.crop_mode = str(getattr(cfg.backend, "crop_mode", "random"))

        diffsynth_piano_cfg = getattr(cfg.backend, "diffsynth_piano", None)
        self.native_sample_rate = int(getattr(diffsynth_piano_cfg, "native_sample_rate", 16000) or 16000)
        self.native_frame_rate = int(getattr(diffsynth_piano_cfg, "native_frame_rate", 250) or 250)
        self.native_segment_seconds = float(getattr(diffsynth_piano_cfg, "native_segment_seconds", 3.0) or 3.0)
        self.crop_seconds = resolve_backend_segment_seconds(
            cfg,
            backend_cfg=diffsynth_piano_cfg,
            backend_default=self.native_segment_seconds,
            total_segment_seconds=self.segment_seconds,
        )
        self.sample_rate = int(getattr(diffsynth_piano_cfg, "sample_rate", self.native_sample_rate) or self.native_sample_rate)
        self.frame_rate = int(getattr(diffsynth_piano_cfg, "frame_rate", self.native_frame_rate) or self.native_frame_rate)
        self.n_synths = int(getattr(cfg.backend.diffsynth_piano, "n_synths", 16))
        self.n_substrings = int(getattr(cfg.backend.diffsynth_piano, "n_substrings", 2))
        self.n_piano_models = int(getattr(cfg.backend.diffsynth_piano, "n_piano_models", 10))
        self.piano_model_index = int(getattr(cfg.backend.diffsynth_piano, "piano_model_index", 0))
        self.begin_note = int(cfg.feature.begin_note)
        self.frame_key = str(getattr(cfg.backend.diffsynth_piano.score_keys, "frame", "frame_roll"))
        self.onset_key = str(getattr(cfg.backend.diffsynth_piano.score_keys, "onset", "onset_roll"))
        self.pedal_key = str(getattr(cfg.backend.diffsynth_piano.score_keys, "pedal", "pedal_frame_roll"))
        self.duration = float(min(self.crop_seconds if self.crop_seconds > 0 else self.native_segment_seconds, self.segment_seconds))

        proxy_root = self._resolve_proxy_root()
        self.model = self._build_model(proxy_root)
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        # DDSP-Piano contains cuDNN RNN modules. We still need gradients from
        # backend audio loss to flow back into vel_pred, so keep the frozen
        # renderer in training mode during forward/backward.
        self.model.train()

        midi_values = torch.arange(self.begin_note, self.begin_note + 88, device=self.device, dtype=torch.float32)
        self.midi_values = midi_values.view(1, 1, -1)

    def _resolve_proxy_root(self) -> Path:
        explicit = str(getattr(getattr(self.cfg.backend, 'diffsynth', None), 'project_root', '') or '').strip()
        if explicit:
            root = Path(explicit).expanduser().resolve()
        else:
            generic = str(getattr(self.cfg.backend, 'project_root', '') or '').strip()
            if generic:
                root = Path(generic).expanduser().resolve()
            else:
                root = Path(__file__).resolve().parents[3] / 'synthesizer' / 'ddsp-piano-pytorch'
        return root

    def _build_model(self, proxy_root: Path):
        if str(proxy_root) not in sys.path:
            sys.path.insert(0, str(proxy_root))
        get_model = importlib.import_module("ddsp_piano.default_model").get_model

        model = get_model(
            inference=False,
            duration=self.duration,
            n_synths=self.n_synths,
            n_substrings=self.n_substrings,
            n_piano_models=self.n_piano_models,
            frame_rate=self.frame_rate,
            sample_rate=self.sample_rate,
        )

        checkpoint_path = Path(str(getattr(self.cfg.backend, "checkpoint", "") or "")).expanduser().resolve()
        state = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, torch.nn.Module):
            model = state
        else:
            if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            model.load_state_dict(state, strict=True)
        return model

    def _get_piano_model_index(self, batch_data_dict, batch_size: int) -> torch.Tensor:
        batch_key = getattr(self.cfg.backend.diffsynth_piano, "piano_model_batch_key", "")
        batch_key = str(batch_key or "").strip()
        if batch_key and batch_key in batch_data_dict:
            indices = move_data_to_device(batch_data_dict[batch_key], self.device).long()
            return indices.view(-1)
        return torch.full((batch_size,), self.piano_model_index, device=self.device, dtype=torch.long)

    def _crop_batch(self, batch_data_dict, audio: torch.Tensor, vel_pred: torch.Tensor, random_state=None):
        frame_roll = move_data_to_device(batch_data_dict[self.frame_key], self.device).float()
        onset_roll = move_data_to_device(batch_data_dict[self.onset_key], self.device).float()
        pedal_roll = move_data_to_device(batch_data_dict[self.pedal_key], self.device).float()
        vel_pred, frame_roll, onset_roll = align_time_dim(vel_pred, frame_roll, onset_roll)
        pedal_roll = pedal_roll[:, : vel_pred.size(1)]

        start_sec, crop_sec = choose_crop_bounds(
            total_seconds=self.segment_seconds,
            crop_seconds=self.crop_seconds,
            mode=self.crop_mode,
            random_state=random_state,
        )
        audio = crop_audio(audio, self.src_sample_rate, start_sec, crop_sec)
        vel_pred = crop_roll_time_first(vel_pred, self.src_frames_per_second, start_sec, crop_sec, include_endpoint=True)
        frame_roll = crop_roll_time_first(frame_roll, self.src_frames_per_second, start_sec, crop_sec, include_endpoint=True)
        onset_roll = crop_roll_time_first(onset_roll, self.src_frames_per_second, start_sec, crop_sec, include_endpoint=True)
        pedal_roll = crop_roll_time_first(pedal_roll.unsqueeze(-1), self.src_frames_per_second, start_sec, crop_sec, include_endpoint=True).squeeze(-1)
        return audio, vel_pred, frame_roll, onset_roll, pedal_roll

    def _build_conditioning(self, frame_roll: torch.Tensor, onset_roll: torch.Tensor, vel_pred: torch.Tensor):
        target_frames = max(1, int(round(self.duration * self.frame_rate)))
        active_pitch = frame_roll * self.midi_values
        onset_velocity = onset_roll * vel_pred
        active_pitch = resample_roll_btp(active_pitch, target_frames, mode="nearest")
        onset_velocity = resample_roll_btp(onset_velocity, target_frames, mode="nearest")

        batch_size = active_pitch.size(0)
        conditioning = torch.zeros(
            batch_size,
            target_frames,
            self.n_synths,
            2,
            device=self.device,
            dtype=vel_pred.dtype,
        )

        for batch_idx in range(batch_size):
            assigned_pitch_np, assigned_index_np = stable_voice_assignment(
                active_pitch[batch_idx].detach().cpu().numpy(),
                self.n_synths,
            )
            assigned_pitch = torch.as_tensor(assigned_pitch_np, device=self.device, dtype=vel_pred.dtype)
            assigned_index = torch.as_tensor(assigned_index_np, device=self.device, dtype=torch.long)
            safe_index = torch.clamp(assigned_index, min=0)
            channel_onset = derive_channel_onsets(assigned_pitch)
            gathered_velocity = torch.gather(onset_velocity[batch_idx], dim=1, index=safe_index)
            gathered_velocity = gathered_velocity * (assigned_index >= 0).to(gathered_velocity.dtype)
            gathered_velocity = gathered_velocity * channel_onset

            conditioning[batch_idx, :, :, 0] = assigned_pitch
            conditioning[batch_idx, :, :, 1] = gathered_velocity

        return conditioning

    def _build_pedal(self, pedal_roll: torch.Tensor) -> torch.Tensor:
        target_frames = max(1, int(round(self.duration * self.frame_rate)))
        pedal = resample_roll_btp(pedal_roll.unsqueeze(-1), target_frames, mode="nearest").squeeze(-1)
        pedal4 = torch.zeros(
            pedal.size(0),
            pedal.size(1),
            4,
            device=self.device,
            dtype=pedal.dtype,
        )
        pedal4[..., 0] = pedal
        return pedal4

    def render(self, batch_data_dict, audio: torch.Tensor, vel_pred: torch.Tensor, random_state=None) -> Dict[str, torch.Tensor]:
        audio, vel_pred, frame_roll, onset_roll, pedal_roll = self._crop_batch(
            batch_data_dict,
            audio,
            vel_pred,
            random_state=random_state,
        )
        conditioning = self._build_conditioning(frame_roll, onset_roll, vel_pred)
        pedal = self._build_pedal(pedal_roll)
        piano_model = self._get_piano_model_index(batch_data_dict, conditioning.size(0))

        proxy_audio, _, _ = self.model(conditioning, pedal, piano_model)
        target_audio = resample_waveform(audio, self.src_sample_rate, self.sample_rate)
        min_len = min(proxy_audio.size(-1), target_audio.size(-1))
        proxy_audio = proxy_audio[..., :min_len]
        target_audio = target_audio[..., :min_len]

        return {
            "rendered_audio": proxy_audio,
            "reference_audio": target_audio,
            "proxy_audio": proxy_audio,
            "target_audio": target_audio,
            "sample_rate": int(self.sample_rate),
            "stats": {
                "proxy_frames": torch.tensor(float(conditioning.size(1)), device=self.device),
                "proxy_polyphony": (conditioning[..., 0] > 0).float().sum(dim=-1).mean().detach(),
                "renderer_sample_rate": torch.tensor(float(self.sample_rate), device=self.device),
                "renderer_frame_rate": torch.tensor(float(self.frame_rate), device=self.device),
                "renderer_segment_seconds": torch.tensor(float(self.duration), device=self.device),
                "native_segment_seconds": torch.tensor(float(self.native_segment_seconds), device=self.device),
            },
        }
