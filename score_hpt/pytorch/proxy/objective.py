from __future__ import annotations

from typing import Dict
import numpy as np
import torch

from .audio_losses import build_audio_loss, get_audio_loss_name
from .common import resample_waveform, resolve_supervision_frame_rate, resolve_supervision_sample_rate
from .naming import normalize_backend_type


class DisabledProxyObjective:
    enabled = False

    def compute(self, batch_data_dict, audio, vel_pred, iteration: int) -> Dict[str, torch.Tensor]:
        return {}


class AudioProxyObjective:
    enabled = True

    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.random_state = np.random.RandomState(cfg.exp.random_seed)
        self.warmup_iterations = int(getattr(cfg.backend, 'warmup_iterations', 0) or 0)

        self.audio_loss_name = get_audio_loss_name(cfg)
        self.supervision_sample_rate = int(resolve_supervision_sample_rate(cfg))
        self.supervision_frame_rate = float(resolve_supervision_frame_rate(cfg))

        backend_type = normalize_backend_type(getattr(cfg.backend, 'type', 'diffsynth_piano'))
        from .ddsp_piano import DDSPPianoProxy
        from .ddsp_guitar_synth import DDSPGuitarSynthProxy

        renderer_cls = {
            'diffsynth_piano': DDSPPianoProxy,
            'diffsynth_guitar': DDSPGuitarSynthProxy,
        }[backend_type]
        self.renderer = renderer_cls(cfg, device)

        self.audio_loss = build_audio_loss(
            cfg,
            sample_rate_override=self.supervision_sample_rate,
            frame_rate_override=self.supervision_frame_rate,
        )

    def compute(self, batch_data_dict, audio, vel_pred, iteration: int) -> Dict[str, torch.Tensor]:
        if iteration < self.warmup_iterations:
            return {}

        render_out = self.renderer.render(
            batch_data_dict=batch_data_dict,
            audio=audio,
            vel_pred=vel_pred,
            random_state=self.random_state,
        )
        rendered_audio = render_out.get('rendered_audio', render_out['proxy_audio'])
        reference_audio = render_out.get('reference_audio', render_out['target_audio'])
        renderer_sample_rate = int(render_out.get('sample_rate', getattr(self.renderer, 'sample_rate', self.supervision_sample_rate)))
        if renderer_sample_rate != self.supervision_sample_rate:
            rendered_audio = resample_waveform(rendered_audio, renderer_sample_rate, self.supervision_sample_rate)
            reference_audio = resample_waveform(reference_audio, renderer_sample_rate, self.supervision_sample_rate)
        proxy_loss, audio_stats = self.audio_loss(rendered_audio, reference_audio)

        stats: Dict[str, torch.Tensor] = {
            'proxy_loss': proxy_loss,
            'loss_sample_rate': torch.tensor(float(self.supervision_sample_rate), device=self.device),
            'loss_frame_rate': torch.tensor(float(self.supervision_frame_rate), device=self.device),
            **audio_stats,
        }
        for key, value in render_out.get('stats', {}).items():
            stats[key] = value
        return stats


def build_proxy_objective(cfg, device: torch.device):
    enabled = bool(getattr(cfg.backend, 'enabled', False))
    backend_weight = float(getattr(cfg.loss, 'backend_weight', 0.0) or 0.0)
    if not enabled or backend_weight <= 0:
        return DisabledProxyObjective()

    backend_type = normalize_backend_type(getattr(cfg.backend, 'type', 'diffsynth_piano'))
    if backend_type == 'diffproxy':
        from .sfproxy import SFProxyObjective
        return SFProxyObjective(cfg, device)
    return AudioProxyObjective(cfg, device)
