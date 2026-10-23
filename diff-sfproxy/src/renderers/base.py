from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np


@dataclass(frozen=True)
class NoteEvent:
    """A single note event inside a fixed-length segment.

    Attributes
    pitch: MIDI pitch in [0, 127]
    onset_s: onset time in seconds
    dur_s: duration in seconds
    velocity_01: continuous velocity in [0, 1]
    """

    pitch: int
    onset_s: float
    dur_s: float
    velocity_01: float


class BaseRenderer(Protocol):
    """Abstract renderer interface.

    Implementations should be deterministic for the same input.
    """

    def render_segment(self, notes: List[NoteEvent], sr: int, seg_len_s: float) -> np.ndarray:
        """Render a segment.

        Args:
            notes: list of NoteEvent
            sr: sample rate
            seg_len_s: segment length in seconds

        Returns:
            audio: mono float32 waveform with shape (sr * seg_len_s,)
        """

        ...
