from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from renderers.base import NoteEvent


@dataclass(frozen=True)
class SfizzRenderCaps:
    binary: str
    supports_polyphony: bool
    supports_quality: bool
    supports_log: bool
    supports_track: bool
    supports_oversampling: bool


def find_sfizz_render_binary() -> str:
    """Return the sfizz offline renderer binary path."""
    env_bin = os.environ.get("SFIZZ_RENDER_BIN")
    if env_bin:
        candidate = Path(env_bin).expanduser()
        if candidate.exists() and os.access(str(candidate), os.X_OK):
            return str(candidate)

    for name in ("sfizz_render", "sfizz-render"):
        candidate = shutil.which(name)
        if candidate:
            return candidate

    repo_root = Path(__file__).resolve().parents[3]
    for candidate in (
        repo_root / "sfizz/build/library/bin/sfizz_render",
        repo_root / "sfizz/build/library/bin/sfizz-render",
    ):
        if candidate.exists() and os.access(str(candidate), os.X_OK):
            return str(candidate)

    raise FileNotFoundError(
        "sfizz_render not found. Tried SFIZZ_RENDER_BIN, PATH "
        "(`sfizz_render`/`sfizz-render`), and local build path "
        f"{repo_root / 'sfizz/build/library/bin/sfizz_render'}."
    )


def _detect_caps(binary: str) -> SfizzRenderCaps:
    try:
        proc = subprocess.run(
            [binary, "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        txt = proc.stdout or ""
    except Exception:
        txt = ""

    def has(token: str) -> bool:
        return token in txt

    return SfizzRenderCaps(
        binary=binary,
        supports_polyphony=has("--polyphony") or has("-p,"),
        supports_quality=has("--quality") or has("-q,"),
        supports_log=has("--log"),
        supports_track=has("--track") or has("-t,"),
        supports_oversampling=has("--oversampling"),
    )


@dataclass
class SfizzConfig:
    sfz_path: str
    block_size: int = 1024
    polyphony: int = 256
    quality: int = 3
    track: Optional[int] = None
    oversampling: Optional[int] = None
    use_eot: bool = False
    fadeout_s: float = 0.01


def _best_temp_root() -> str:
    for candidate in ("/dev/shm", tempfile.gettempdir()):
        if candidate and os.path.isdir(candidate) and os.access(candidate, os.W_OK | os.X_OK):
            return candidate
    return tempfile.gettempdir()


class SfizzSFZRenderer:
    """Render SFZ instruments via `sfizz_render`."""

    def __init__(self, cfg: SfizzConfig):
        self.cfg = cfg
        self._sfz_path = str(Path(cfg.sfz_path))
        if not Path(self._sfz_path).exists():
            raise FileNotFoundError(f"SFZ not found: {self._sfz_path}")
        self._binary = find_sfizz_render_binary()
        self._caps = _detect_caps(self._binary)
        self._work_dir = Path(tempfile.mkdtemp(prefix="sfproxy_sfz_", dir=_best_temp_root()))
        self._midi_path = self._work_dir / "seg.mid"
        self._wav_path = self._work_dir / "seg.wav"
        self._cleanup = weakref.finalize(self, shutil.rmtree, self._work_dir, True)

    @staticmethod
    def _to_midi_velocity(velocity_01: float) -> int:
        value = int(round(float(velocity_01) * 127.0))
        return int(np.clip(value, 1, 127))

    def render_segment(self, notes: List[NoteEvent], sr: int, seg_len_s: float) -> np.ndarray:
        try:
            import mido  # type: ignore
        except Exception as e:
            raise RuntimeError("mido is required for SFZ rendering") from e

        try:
            from scipy.io import wavfile
        except Exception as e:
            raise RuntimeError("scipy is required to read sfizz_render output wav") from e

        num_samples = int(round(sr * float(seg_len_s)))
        tempo = mido.bpm2tempo(120)
        ticks_per_beat = 480

        events: List[tuple[float, bool, int, int]] = []
        for note in notes:
            if note.dur_s <= 0:
                continue
            events.append((float(note.onset_s), True, int(note.pitch), self._to_midi_velocity(note.velocity_01)))
            events.append((float(note.onset_s) + float(note.dur_s), False, int(note.pitch), 0))
        events.sort(key=lambda x: x[0])

        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

        last_tick = 0
        for t_s, is_on, pitch, vel in events:
            t_s = max(0.0, min(float(seg_len_s), float(t_s)))
            tick = int(round(mido.second2tick(t_s, ticks_per_beat, tempo)))
            delta = max(0, tick - last_tick)
            last_tick = tick
            if is_on:
                track.append(mido.Message("note_on", note=int(pitch), velocity=int(vel), time=delta))
            else:
                track.append(mido.Message("note_off", note=int(pitch), velocity=0, time=delta))

        end_tick = int(round(mido.second2tick(float(seg_len_s), ticks_per_beat, tempo)))
        delta_end = max(0, end_tick - last_tick)
        track.append(mido.MetaMessage("end_of_track", time=delta_end))
        mid.save(str(self._midi_path))

        if self._wav_path.exists():
            self._wav_path.unlink()

        cmd = [
            self._binary,
            "--sfz",
            str(self._sfz_path),
            "--midi",
            str(self._midi_path),
            "--wav",
            str(self._wav_path),
            "--samplerate",
            str(int(sr)),
            "--blocksize",
            str(int(self.cfg.block_size)),
        ]
        if self._caps.supports_polyphony:
            cmd += ["--polyphony", str(int(self.cfg.polyphony))]
        if self._caps.supports_quality:
            cmd += ["--quality", str(int(self.cfg.quality))]
        if self._caps.supports_track and self.cfg.track is not None:
            cmd += ["--track", str(int(self.cfg.track))]
        if self._caps.supports_oversampling and self.cfg.oversampling is not None:
            cmd += ["--oversampling", str(int(self.cfg.oversampling))]
        if self.cfg.use_eot:
            cmd += ["--use-eot"]

        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(Path(self._sfz_path).parent),
        )
        if proc.returncode != 0:
            proc = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path(self._sfz_path).parent),
            )
            stdout = proc.stdout[-4000:] if proc.stdout else ""
            raise RuntimeError(f"sfizz_render failed with code {proc.returncode}: {stdout}")

        sr_read, audio = wavfile.read(str(self._wav_path))
        if sr_read != sr:
            raise RuntimeError(f"Unexpected sample rate from sfizz_render: {sr_read} != {sr}")

        audio = np.asarray(audio)
        if audio.ndim == 1:
            audio = audio[:, None]

        if np.issubdtype(audio.dtype, np.integer):
            max_val = float(np.iinfo(audio.dtype).max)
            audio_f = audio.astype(np.float32) / max_val
        else:
            audio_f = audio.astype(np.float32)

        mono = audio_f.mean(axis=1)

        if mono.shape[0] < num_samples:
            pad = np.zeros((num_samples - mono.shape[0],), dtype=np.float32)
            mono = np.concatenate([mono, pad], axis=0)
        elif mono.shape[0] > num_samples:
            mono = mono[:num_samples]

        fade_n = int(round(sr * float(self.cfg.fadeout_s)))
        if fade_n > 1 and fade_n < mono.shape[0]:
            weights = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
            mono[-fade_n:] *= weights

        return mono.astype(np.float32, copy=False)
