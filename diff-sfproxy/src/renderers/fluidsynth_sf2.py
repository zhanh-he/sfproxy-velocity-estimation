from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from renderers.base import NoteEvent


@dataclass
class FluidSynthConfig:
    """Configuration for FluidSynth rendering."""

    sf2_path: str
    bank: int = 0
    program: int = 0
    polyphony: int = 64
    gain_db: float = -6.0
    disable_reverb: bool = True
    disable_chorus: bool = True
    channel: int = 0
    # flush and fadeout in seconds
    flush_s: float = 0.01
    fadeout_s: float = 0.01
    # if True, fall back to fluidsynth CLI when python binding is unavailable
    allow_cli_fallback: bool = True


class FluidSynthSF2Renderer:
    """Render SF2 instruments using FluidSynth.

    This renderer tries to use the Python binding (pyfluidsynth / fluidsynth).
    If unavailable and allow_cli_fallback=True, it falls back to the `fluidsynth` CLI.

    Notes:
    - Rendering is deterministic only if the underlying synth settings are deterministic.
    - Reverb/chorus are disabled by default to reduce non-deterministic tails.
    """

    def __init__(self, cfg: FluidSynthConfig):
        self.cfg = cfg
        self._sf2_path = str(Path(cfg.sf2_path))
        if not Path(self._sf2_path).exists():
            raise FileNotFoundError(f"SF2 not found: {self._sf2_path}")

        self._synth = None
        self._sfid = None
        self._binding_ok = False

    @staticmethod
    def _db_to_gain_linear(gain_db: float) -> float:
        return float(10.0 ** (gain_db / 20.0))

    @staticmethod
    def _to_midi_velocity(velocity_01: float) -> int:
        v = int(round(float(velocity_01) * 127.0))
        return int(np.clip(v, 1, 127))

    def _ensure_binding(self, sr: int) -> None:
        """Initialize python binding synth if possible."""
        if self._synth is not None:
            return

        try:
            import fluidsynth  # type: ignore
        except Exception:
            self._binding_ok = False
            return

        try:
            # Typical pyfluidsynth signature supports samplerate kwarg.
            synth = fluidsynth.Synth(samplerate=sr)
        except Exception:
            try:
                synth = fluidsynth.Synth()
            except Exception:
                self._binding_ok = False
                return

        # Apply settings when possible.
        gain = self._db_to_gain_linear(self.cfg.gain_db)
        try:
            # Some bindings expose `setting`.
            synth.setting("synth.gain", gain)
        except Exception:
            try:
                synth.gain = gain
            except Exception:
                pass

        try:
            synth.setting("synth.polyphony", int(self.cfg.polyphony))
        except Exception:
            pass

        if self.cfg.disable_reverb:
            try:
                synth.setting("synth.reverb.active", 0)
            except Exception:
                try:
                    synth.set_reverb_on(False)
                except Exception:
                    pass

        if self.cfg.disable_chorus:
            try:
                synth.setting("synth.chorus.active", 0)
            except Exception:
                try:
                    synth.set_chorus_on(False)
                except Exception:
                    pass

        # Load soundfont and select program.
        try:
            sfid = synth.sfload(self._sf2_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load SF2 via python binding: {self._sf2_path}") from e

        try:
            synth.program_select(self.cfg.channel, sfid, int(self.cfg.bank), int(self.cfg.program))
        except Exception as e:
            raise RuntimeError(
                f"Failed to select program bank={self.cfg.bank} program={self.cfg.program}"
            ) from e

        self._synth = synth
        self._sfid = sfid
        self._binding_ok = True

    def _all_notes_off(self) -> None:
        """Send 'all notes off' and 'all sound off' CCs."""
        if self._synth is None:
            return
        ch = int(self.cfg.channel)
        try:
            # all sound off
            self._synth.cc(ch, 120, 0)
        except Exception:
            pass
        try:
            # all notes off
            self._synth.cc(ch, 123, 0)
        except Exception:
            pass

    def render_segment(self, notes: List[NoteEvent], sr: int, seg_len_s: float) -> np.ndarray:
        """Render a fixed-length segment."""
        self._ensure_binding(sr)

        if self._binding_ok:
            return self._render_with_binding(notes, sr, seg_len_s)

        if not self.cfg.allow_cli_fallback:
            raise RuntimeError(
                "FluidSynth python binding is not available and CLI fallback is disabled. "
                "Install pyfluidsynth/fluidsynth or enable CLI fallback."
            )

        return self._render_with_cli(notes, sr, seg_len_s)

    def _render_with_binding(self, notes: List[NoteEvent], sr: int, seg_len_s: float) -> np.ndarray:
        assert self._synth is not None

        num_samples = int(round(sr * float(seg_len_s)))
        if num_samples <= 0:
            raise ValueError(f"Invalid seg_len_s={seg_len_s}")

        # Flush: stop tails and clear internal state.
        self._all_notes_off()
        flush_n = int(round(sr * float(self.cfg.flush_s)))
        if flush_n > 0:
            _ = self._synth.get_samples(flush_n)

        # Build event list: (sample_index, is_on, pitch, velocity)
        events: List[tuple[int, bool, int, int]] = []
        for n in notes:
            onset = int(round(float(n.onset_s) * sr))
            dur = int(round(float(n.dur_s) * sr))
            if dur <= 0:
                continue
            if onset >= num_samples:
                continue
            pitch = int(np.clip(int(n.pitch), 0, 127))
            vel = self._to_midi_velocity(n.velocity_01)
            off = min(onset + dur, num_samples)
            events.append((onset, True, pitch, vel))
            events.append((off, False, pitch, 0))

        events.sort(key=lambda x: x[0])

        # Render sequentially by generating samples between events.
        out = np.zeros((num_samples, 2), dtype=np.float32)
        cur = 0
        ch = int(self.cfg.channel)

        def _pull(nframes: int, write_pos: int) -> int:
            if nframes <= 0:
                return 0
            raw = self._synth.get_samples(nframes)
            raw = np.asarray(raw)
            if raw.ndim == 1:
                raw = raw.reshape(-1, 2)
            if raw.shape[0] != nframes:
                raw = raw[:nframes]
            out[write_pos : write_pos + raw.shape[0]] = raw.astype(np.float32, copy=False)
            return raw.shape[0]

        for sample_idx, is_on, pitch, vel in events:
            sample_idx = int(np.clip(sample_idx, 0, num_samples))
            if sample_idx > cur:
                n = sample_idx - cur
                written = _pull(n, cur)
                cur += written

            # Apply event at exact boundary.
            if is_on:
                try:
                    self._synth.noteon(ch, pitch, vel)
                except Exception:
                    pass
            else:
                try:
                    self._synth.noteoff(ch, pitch)
                except Exception:
                    pass

        if cur < num_samples:
            _ = _pull(num_samples - cur, cur)

        # Convert to mono.
        mono = out.mean(axis=1)

        # Fadeout.
        fade_n = int(round(sr * float(self.cfg.fadeout_s)))
        if fade_n > 1 and fade_n < mono.shape[0]:
            w = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
            mono[-fade_n:] *= w

        return mono.astype(np.float32, copy=False)

    def _render_with_cli(self, notes: List[NoteEvent], sr: int, seg_len_s: float) -> np.ndarray:
        """CLI fallback using `fluidsynth` binary."""
        if shutil.which("fluidsynth") is None:
            raise RuntimeError(
                "fluidsynth CLI not found. Install fluidsynth or install python binding."
            )

        # Lazy import: only required for CLI fallback.
        try:
            import mido  # type: ignore
        except Exception as e:
            raise RuntimeError("mido is required for CLI fallback rendering") from e

        num_samples = int(round(sr * float(seg_len_s)))

        gain = self._db_to_gain_linear(self.cfg.gain_db)

        tempo = mido.bpm2tempo(120)
        ticks_per_beat = 480

        # Build event list in ticks.
        ev: List[tuple[float, bool, int, int]] = []
        for n in notes:
            if n.dur_s <= 0:
                continue
            ev.append((float(n.onset_s), True, int(n.pitch), self._to_midi_velocity(n.velocity_01)))
            ev.append((float(n.onset_s) + float(n.dur_s), False, int(n.pitch), 0))
        ev.sort(key=lambda x: x[0])

        with tempfile.TemporaryDirectory(prefix="sfproxy_fs_") as td:
            td_path = Path(td)
            midi_path = td_path / "seg.mid"
            wav_path = td_path / "seg.wav"

            mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
            track = mido.MidiTrack()
            mid.tracks.append(track)
            track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

            last_t = 0.0
            last_tick = 0

            for t_s, is_on, pitch, vel in ev:
                t_s = max(0.0, min(float(seg_len_s), float(t_s)))
                tick = int(round(mido.second2tick(t_s, ticks_per_beat, tempo)))
                delta = max(0, tick - last_tick)
                last_tick = tick
                last_t = t_s

                if is_on:
                    track.append(mido.Message("note_on", note=int(pitch), velocity=int(vel), time=delta))
                else:
                    track.append(mido.Message("note_off", note=int(pitch), velocity=0, time=delta))

            # Ensure track length equals seg_len_s.
            end_tick = int(round(mido.second2tick(float(seg_len_s), ticks_per_beat, tempo)))
            delta_end = max(0, end_tick - last_tick)
            track.append(mido.MetaMessage("end_of_track", time=delta_end))

            mid.save(str(midi_path))

            cmd = [
                "fluidsynth",
                "-ni",
                "-r",
                str(int(sr)),
                "-F",
                str(wav_path),
                "-o",
                "synth.reverb.active=0" if self.cfg.disable_reverb else "synth.reverb.active=1",
                "-o",
                "synth.chorus.active=0" if self.cfg.disable_chorus else "synth.chorus.active=1",
                "-o",
                f"synth.polyphony={int(self.cfg.polyphony)}",
                "-g",
                str(float(gain)),
                str(self._sf2_path),
                str(midi_path),
            ]

            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Read wav.
            try:
                from scipy.io import wavfile

                sr_read, audio = wavfile.read(str(wav_path))
            except Exception as e:
                raise RuntimeError("Failed to read fluidsynth output wav") from e

            if sr_read != sr:
                # No resampling in Phase-1. Fail fast.
                raise RuntimeError(f"Unexpected sample rate from fluidsynth: {sr_read} != {sr}")

            audio = np.asarray(audio)
            if audio.ndim == 1:
                audio = audio[:, None]

            # Convert to float32 [-1, 1].
            if np.issubdtype(audio.dtype, np.integer):
                max_val = float(np.iinfo(audio.dtype).max)
                audio_f = audio.astype(np.float32) / max_val
            else:
                audio_f = audio.astype(np.float32)

            # Mono
            mono = audio_f.mean(axis=1)

            # Trim/pad.
            if mono.shape[0] < num_samples:
                pad = np.zeros((num_samples - mono.shape[0],), dtype=np.float32)
                mono = np.concatenate([mono, pad], axis=0)
            elif mono.shape[0] > num_samples:
                mono = mono[:num_samples]

            # Fadeout
            fade_n = int(round(sr * float(self.cfg.fadeout_s)))
            if fade_n > 1 and fade_n < mono.shape[0]:
                w = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
                mono[-fade_n:] *= w

            return mono.astype(np.float32, copy=False)
