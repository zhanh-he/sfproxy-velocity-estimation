from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from data.note_samplers import BaseNoteSampler
from renderers.base import NoteEvent
from renderers.fluidsynth_sf2 import FluidSynthConfig, FluidSynthSF2Renderer
from renderers.sfizz_sfz import SfizzConfig, SfizzSFZRenderer


@dataclass
class InstrumentSpec:
    """Instrument-file specification for SF2/SFZ rendering."""

    name: str
    path: str = ""
    backend: str = "auto"  # auto | fluidsynth | sfizz
    bank: int = 0
    program: int = 0
    polyphony: int = 64
    gain_db: float = -6.0
    sfizz_block_size: int = 1024
    sfizz_quality: int = 3
    sfizz_use_eot: bool = False

    # Convenience fields for export-time configs (not used by the renderer itself)
    sr: int = 32000
    seg_len_s: float = 2.0
    nmax: int = 64

    @property
    def instrument_path(self) -> str:
        path = str(self.path or "").strip()
        if not path:
            raise ValueError("InstrumentSpec requires `path`")
        return str(Path(path).expanduser())

    @property
    def instrument_ext(self) -> str:
        return Path(self.instrument_path).suffix.lower()

    @property
    def render_backend(self) -> str:
        backend = str(self.backend or "auto").strip().lower()
        if backend == "auto":
            ext = self.instrument_ext
            if ext == ".sf2":
                return "fluidsynth"
            if ext == ".sfz":
                return "sfizz"
            raise ValueError(f"Unsupported instrument extension: {ext}")
        if backend not in {"fluidsynth", "sfizz"}:
            raise ValueError(f"Unsupported backend: {backend}")
        return backend


class RenderedInstrumentDataset(Dataset):
    """Online dataset that samples note events and renders audio from an instrument file.

    This dataset is intended for *offline export* only.
    Training should use the exported pkl dataset.

    Returned tensors are padded to a fixed Nmax.

    Returned tuple:
      pitch: (Nmax,) int64
      cont: (Nmax, 3) float32  [onset_s, dur_s, vel_01]
      mask: (Nmax,) bool
      audio: (1, T) float32
      idx: int (seed index used)
    """

    MAX_SEED_VALUE = 2**64 - 1
    OFFSET_COEFFICIENT = 100_000_000_000

    def __init__(
        self,
        instrument: InstrumentSpec,
        sampler: BaseNoteSampler,
        dataset_size: int,
        seed_offset: int,
        sr: int,
        seg_len_s: float,
        nmax: int,
        rms_range: Tuple[float, float] = (0.0, 10.0),
        peak_abs_max: float = 0.999,
        max_tries: int = 20,
    ):
        super().__init__()
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if seed_offset < 0:
            raise ValueError("seed_offset must be non-negative")

        if seed_offset * self.OFFSET_COEFFICIENT + dataset_size > self.MAX_SEED_VALUE:
            raise ValueError("seed_offset too large")

        self.instrument = instrument
        self.sampler = sampler

        self.dataset_size = int(dataset_size)
        self.seed_offset = int(seed_offset * self.OFFSET_COEFFICIENT)

        self.sr = int(sr)
        self.seg_len_s = float(seg_len_s)
        self.nmax = int(nmax)

        self.rms_range = rms_range
        self.peak_abs_max = float(peak_abs_max)
        self.max_tries = int(max_tries)

        self.renderer: Optional[object] = None

    def __len__(self) -> int:
        return self.dataset_size

    def _instantiate_renderer(self) -> None:
        backend = self.instrument.render_backend
        if backend == "fluidsynth":
            cfg = FluidSynthConfig(
                sf2_path=str(self.instrument.instrument_path),
                bank=int(self.instrument.bank),
                program=int(self.instrument.program),
                polyphony=int(self.instrument.polyphony),
                gain_db=float(self.instrument.gain_db),
                allow_cli_fallback=True,
            )
            self.renderer = FluidSynthSF2Renderer(cfg)
            return

        if backend == "sfizz":
            cfg = SfizzConfig(
                sfz_path=str(self.instrument.instrument_path),
                block_size=int(self.instrument.sfizz_block_size),
                polyphony=int(self.instrument.polyphony),
                quality=int(self.instrument.sfizz_quality),
                use_eot=bool(self.instrument.sfizz_use_eot),
            )
            self.renderer = SfizzSFZRenderer(cfg)
            return

        raise ValueError(f"Unsupported instrument backend: {backend}")

    def _notes_to_padded(self, notes: List[NoteEvent]):
        # pad / truncate to Nmax
        n = min(len(notes), self.nmax)
        pitch = torch.zeros((self.nmax,), dtype=torch.long)
        cont = torch.zeros((self.nmax, 3), dtype=torch.float32)
        mask = torch.zeros((self.nmax,), dtype=torch.bool)
        for i in range(n):
            ne = notes[i]
            pitch[i] = int(ne.pitch)
            cont[i, 0] = float(ne.onset_s)
            cont[i, 1] = float(ne.dur_s)
            cont[i, 2] = float(ne.velocity_01)
            mask[i] = True
        return pitch, cont, mask

    @staticmethod
    def _rms(x: torch.Tensor) -> float:
        return float(torch.sqrt(torch.mean(x * x)).item())

    def __getitem__(self, idx: int):
        if self.renderer is None:
            self._instantiate_renderer()

        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.seed_offset + int(idx))

        last_pitch = None
        last_cont = None
        last_mask = None
        last_audio = None

        # Rejection sampling based on RMS/peak.
        for _ in range(max(1, self.max_tries)):
            notes = self.sampler.sample(rng)
            pitch, cont, mask = self._notes_to_padded(notes)

            audio_np = self.renderer.render_segment(notes, sr=self.sr, seg_len_s=self.seg_len_s)
            audio = torch.from_numpy(audio_np).to(torch.float32).unsqueeze(0)

            rms = self._rms(audio)
            peak = float(torch.max(torch.abs(audio)).item())

            last_pitch, last_cont, last_mask, last_audio = pitch, cont, mask, audio

            if self.rms_range[0] <= rms <= self.rms_range[1] and peak <= self.peak_abs_max:
                return pitch, cont, mask, audio, int(idx)

        # If all tries fail, return the last sample for debugging.
        return last_pitch, last_cont, last_mask, last_audio, int(idx)
