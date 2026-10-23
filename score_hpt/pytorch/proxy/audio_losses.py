"""Compatibility wrapper for Route III proxy audio losses.

The actual implementations live in `pytorch/losses.py` so that all loss
selection stays in one place. This module re-exports the audio losses for the
proxy package.
"""

from losses import (
    AUDIO_LOSS_TYPES,
    CompositeAudioLoss,
    GlobalLogRMSLoss,
    PianoSSMSpectralLoss,
    PianoSSMSpectralPlusLogRMSLoss,
    build_audio_loss,
    get_audio_loss_name,
)

__all__ = [
    "AUDIO_LOSS_TYPES",
    "CompositeAudioLoss",
    "GlobalLogRMSLoss",
    "PianoSSMSpectralLoss",
    "PianoSSMSpectralPlusLogRMSLoss",
    "build_audio_loss",
    "get_audio_loss_name",
]
