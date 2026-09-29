from __future__ import annotations

from pathlib import Path
import importlib
import math
import sys
from typing import Dict

import torch

from .common import (
    build_guitar_synth_inputs_from_note_events,
    choose_crop_bounds,
    crop_audio,
    crop_roll_time_first,
    crop_roll_time_last,
    resample_roll_bct,
    resample_roll_btp,
    resample_waveform,
    resolve_backend_segment_seconds,
)


class DDSPGuitarSynthProxy:
    """Frozen DDSP-Guitar-Synth renderer driven by predicted onset velocities."""

    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.src_sample_rate = int(cfg.feature.sample_rate)
        self.src_frames_per_second = float(cfg.feature.frames_per_second)
        self.segment_seconds = float(cfg.feature.segment_seconds)
        self.crop_mode = str(getattr(cfg.backend, 'crop_mode', 'random') or 'random')
        self.begin_note = int(getattr(cfg.feature, 'begin_note', 21) or 21)
        default_velocity_scale = float(getattr(cfg.feature, 'velocity_scale', 128) or 128)
        self.velocity_midi_max = max(1.0, default_velocity_scale - 1.0)

        diffsynth_cfg = getattr(cfg.backend, 'diffsynth_guitar', None)
        self.configured_sample_rate = int(getattr(diffsynth_cfg, 'sample_rate', 22050) or 22050)
        self.configured_frame_rate = float(getattr(diffsynth_cfg, 'frame_rate', 100.0) or 100.0)
        self.configured_hop_size = int(getattr(diffsynth_cfg, 'hop_size', 0) or 0)
        self.configured_n_fft = int(getattr(diffsynth_cfg, 'n_fft', 2048) or 2048)
        self.max_fret = int(getattr(diffsynth_cfg, 'max_fret', 24) or 24)
        self.native_sample_rate = int(getattr(diffsynth_cfg, 'native_sample_rate', self.configured_sample_rate) or self.configured_sample_rate)
        self.native_frame_rate = float(getattr(diffsynth_cfg, 'native_frame_rate', self.configured_frame_rate) or self.configured_frame_rate)
        self.native_segment_seconds = float(getattr(diffsynth_cfg, 'native_segment_seconds', 10.0) or 10.0)
        self.crop_seconds = resolve_backend_segment_seconds(
            cfg,
            backend_cfg=diffsynth_cfg,
            backend_default=self.native_segment_seconds,
            total_segment_seconds=self.segment_seconds,
        )

        proxy_root = self._resolve_proxy_root()
        self.model = self._build_model(proxy_root)
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        # DDSP-Guitar-Synth contains GRU layers. We still need gradients to flow
        # from backend audio loss back into vel_pred, so cuDNN RNNs must run in
        # training mode even though backend parameters stay frozen.
        self.model.train()

    @staticmethod
    def _round_half_up(x: float) -> int:
        return int(math.floor(float(x) + 0.5))

    def _derive_hop_length(self, sample_rate: int, frame_rate: float) -> int:
        return max(1, self._round_half_up(float(sample_rate) / max(float(frame_rate), 1e-6)))

    def _resolve_proxy_root(self) -> Path:
        explicit = str(getattr(getattr(self.cfg.backend, 'diffsynth', None), 'project_root', '') or '').strip()
        if explicit:
            root = Path(explicit).expanduser().resolve()
        else:
            generic = str(getattr(self.cfg.backend, 'project_root', '') or '').strip()
            if generic:
                root = Path(generic).expanduser().resolve()
            else:
                root = Path(__file__).resolve().parents[3] / 'synthesizer' / 'ddsp-guitar-synth'
        return root

    def _read_checkpoint_config(self, checkpoint_path: Path) -> dict:
        state = torch.load(checkpoint_path, map_location='cpu')
        if not isinstance(state, dict):
            return {}
        renderer_config = state.get('renderer_config', {})
        return renderer_config if isinstance(renderer_config, dict) else {}

    def _build_model(self, proxy_root: Path):
        if str(proxy_root) not in sys.path:
            sys.path.insert(0, str(proxy_root))
        importlib.invalidate_caches()
        MidiSynth = importlib.import_module('midi_synth.midi_synth').MidiSynth

        checkpoint_path = Path(str(getattr(self.cfg.backend, 'checkpoint', '') or '')).expanduser().resolve()
        renderer_config = self._read_checkpoint_config(checkpoint_path)
        self.sample_rate = int(renderer_config.get('sample_rate', self.configured_sample_rate) or self.configured_sample_rate)
        target_frame_rate = float(renderer_config.get('target_frame_rate', self.configured_frame_rate) or self.configured_frame_rate)
        hop_length = int(renderer_config.get('hop_length', self.configured_hop_size or self._derive_hop_length(self.sample_rate, target_frame_rate)) or 0)
        if hop_length <= 0:
            hop_length = self._derive_hop_length(self.sample_rate, target_frame_rate)
        self.hop_length = int(hop_length)
        self.frame_rate = float(renderer_config.get('effective_frame_rate', self.sample_rate / float(self.hop_length)))
        self.n_fft = int(renderer_config.get('n_fft', self.configured_n_fft) or self.configured_n_fft)
        self.native_sample_rate = int(self.sample_rate)
        self.native_frame_rate = float(self.frame_rate)
        self.native_segment_seconds = float(renderer_config.get('segment_seconds', self.native_segment_seconds) or self.native_segment_seconds)
        self.crop_seconds = min(float(self.crop_seconds), float(self.segment_seconds))

        model = MidiSynth(sr=self.sample_rate, hop_length=self.hop_length, reverb_length=self.sample_rate)
        state = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state, strict=True)
        return model

    @staticmethod
    def _derive_onsets(midi_pitch: torch.Tensor) -> torch.Tensor:
        active = (midi_pitch > 0).to(midi_pitch.dtype)
        onsets = torch.zeros_like(active)
        onsets[..., 0] = active[..., 0]
        if midi_pitch.size(-1) > 1:
            changed = midi_pitch[..., 1:] != midi_pitch[..., :-1]
            onsets[..., 1:] = ((midi_pitch[..., 1:] > 0) & changed).to(midi_pitch.dtype)
        return onsets

    def _crop_batch(self, batch_data_dict, audio: torch.Tensor, vel_pred: torch.Tensor, random_state=None):
        synth_input_frame_rate = self.src_frames_per_second
        aligned_note_events = batch_data_dict['aligned_note_events']
        derived = build_guitar_synth_inputs_from_note_events(
            aligned_note_events,
            frame_rate=synth_input_frame_rate,
            segment_seconds=self.segment_seconds,
            max_fret=self.max_fret,
            device=self.device,
        )
        midi_pitch = derived['midi_pitch']
        midi_onsets = derived['midi_onsets']
        midi_activity = derived['midi_activity']

        start_sec, crop_sec = choose_crop_bounds(
            total_seconds=self.segment_seconds,
            crop_seconds=self.crop_seconds,
            mode=self.crop_mode,
            random_state=random_state,
        )
        audio = crop_audio(audio, self.src_sample_rate, start_sec, crop_sec)
        vel_pred = crop_roll_time_first(vel_pred, self.src_frames_per_second, start_sec, crop_sec, include_endpoint=True)
        midi_pitch = crop_roll_time_last(midi_pitch, synth_input_frame_rate, start_sec, crop_sec, include_endpoint=False)
        if midi_onsets is not None:
            midi_onsets = crop_roll_time_last(midi_onsets, synth_input_frame_rate, start_sec, crop_sec, include_endpoint=False)
        if midi_activity is not None:
            midi_activity = crop_roll_time_last(midi_activity, synth_input_frame_rate, start_sec, crop_sec, include_endpoint=False)
        return audio, vel_pred, midi_pitch, midi_onsets, midi_activity, synth_input_frame_rate

    def _resample_synth_inputs(
        self,
        midi_pitch: torch.Tensor,
        midi_onsets: Optional[torch.Tensor],
        midi_activity: Optional[torch.Tensor],
        synth_input_frame_rate: float,
    ):
        if abs(float(synth_input_frame_rate) - float(self.frame_rate)) < 1e-6:
            return midi_pitch, midi_onsets, midi_activity

        target_frames = max(1, int(round(midi_pitch.size(-1) * float(self.frame_rate) / float(synth_input_frame_rate))))
        midi_pitch = resample_roll_bct(midi_pitch, target_frames=target_frames, mode='nearest')
        if midi_onsets is not None:
            midi_onsets = (resample_roll_bct(midi_onsets, target_frames=target_frames, mode='nearest') > 0.5).to(midi_pitch.dtype)
        if midi_activity is not None:
            midi_activity = (resample_roll_bct(midi_activity, target_frames=target_frames, mode='nearest') > 0.5).to(midi_pitch.dtype)
        return midi_pitch, midi_onsets, midi_activity

    def render(self, batch_data_dict, audio: torch.Tensor, vel_pred: torch.Tensor, random_state=None) -> Dict[str, torch.Tensor]:
        audio, vel_pred, midi_pitch, midi_onsets, midi_activity, synth_input_frame_rate = self._crop_batch(
            batch_data_dict,
            audio,
            vel_pred,
            random_state=random_state,
        )

        midi_pitch, midi_onsets, midi_activity = self._resample_synth_inputs(
            midi_pitch=midi_pitch,
            midi_onsets=midi_onsets,
            midi_activity=midi_activity,
            synth_input_frame_rate=synth_input_frame_rate,
        )

        target_frames = midi_pitch.size(-1)
        pred_proxy = resample_roll_btp(vel_pred, target_frames, mode='nearest')
        pred_proxy = pred_proxy.unsqueeze(1).expand(-1, midi_pitch.size(1), -1, -1)

        pitch_idx = torch.clamp(midi_pitch.long() - self.begin_note, min=0, max=87)
        gathered_velocity = torch.gather(pred_proxy, dim=-1, index=pitch_idx.unsqueeze(-1)).squeeze(-1)
        active_mask = (midi_pitch > 0).to(gathered_velocity.dtype)
        if midi_activity is not None:
            active_mask = midi_activity.to(gathered_velocity.dtype)
        if midi_onsets is None:
            midi_onsets = self._derive_onsets(midi_pitch)
        else:
            midi_onsets = midi_onsets.to(gathered_velocity.dtype)

        midi_velocity = gathered_velocity * self.velocity_midi_max
        onset_velocity = midi_velocity * active_mask * midi_onsets

        conditioning = torch.stack(
            [midi_pitch.transpose(1, 2), onset_velocity.transpose(1, 2)],
            dim=-1,
        )
        model_inputs = conditioning.to(self.device)

        target_audio = resample_waveform(audio, self.src_sample_rate, self.sample_rate)
        outputs = self.model(model_inputs)
        proxy_audio = outputs['audio']
        if proxy_audio.dim() == 3 and proxy_audio.size(1) == 1:
            proxy_audio = proxy_audio.squeeze(1)

        min_len = min(proxy_audio.size(-1), target_audio.size(-1))
        proxy_audio = proxy_audio[..., :min_len]
        target_audio = target_audio[..., :min_len]

        return {
            'rendered_audio': proxy_audio,
            'reference_audio': target_audio,
            'proxy_audio': proxy_audio,
            'target_audio': target_audio,
            'sample_rate': int(self.sample_rate),
            'stats': {
                'proxy_frames': torch.tensor(float(target_frames), device=self.device),
                'proxy_polyphony': active_mask.sum(dim=1).mean().detach(),
                'synth_input_source_fps': torch.tensor(float(synth_input_frame_rate), device=self.device),
                'renderer_sample_rate': torch.tensor(float(self.sample_rate), device=self.device),
                'renderer_frame_rate': torch.tensor(float(self.frame_rate), device=self.device),
                'renderer_segment_seconds': torch.tensor(float(self.crop_seconds), device=self.device),
                'renderer_hop_length': torch.tensor(float(self.hop_length), device=self.device),
                'renderer_n_fft': torch.tensor(float(self.n_fft), device=self.device),
                'native_segment_seconds': torch.tensor(float(self.native_segment_seconds), device=self.device),
            },
        }
