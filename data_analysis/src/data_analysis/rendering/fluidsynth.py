from __future__ import annotations

import os
import wave
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np


@dataclass(frozen=True)
class FluidSynthRenderInfo:
    sf2_path: Path
    midi_path: Path
    wav_path: Path
    sample_rate: int
    gain: float
    extra_tail_sec: float


def _float_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == np.int16:
        return x
    x = x.astype(np.float32, copy=False)
    mx = float(np.max(np.abs(x))) if x.size else 0.0
    if mx <= 1.01:
        x = x * 32767.0
    x = np.clip(x, -32768.0, 32767.0)
    return x.astype(np.int16)


def render_midi_with_sf2_fluidsynth(
    *,
    midi_path: str | Path,
    sf2_path: str | Path,
    wav_path: str | Path,
    sample_rate: int = 44100,
    gain: float = 0.5,
    extra_tail_sec: float = 2.0,
    disable_reverb_chorus: bool = True,
) -> Dict[str, object]:
    """Render MIDI with an SF2 SoundFont using pyfluidsynth (headless).

    This backend is NOT compatible with .sfz; use sfizz_render for SFZ.
    """
    import mido
    import fluidsynth

    midi_path = Path(midi_path)
    sf2_path = Path(sf2_path)
    wav_path = Path(wav_path)

    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI not found: {midi_path}")
    if not sf2_path.exists():
        raise FileNotFoundError(f"SF2 not found: {sf2_path}")

    wav_path.parent.mkdir(parents=True, exist_ok=True)

    synth = fluidsynth.Synth(samplerate=float(sample_rate), gain=float(gain))
    # Offline rendering path: use get_samples() directly and do not start
    # realtime audio/midi drivers (avoids driver-related stderr noise).

    if disable_reverb_chorus:
        # Keep this section intentionally no-op to avoid build-dependent
        # warnings from low-level chorus/reverb parameter validation.
        pass

    sfid = synth.sfload(str(sf2_path))
    preset_exists_cache: Dict[int, bool] = {}
    ignored_program_changes: list[Dict[str, int]] = []

    def preset_exists(program: int) -> bool:
        key = int(program)
        if key not in preset_exists_cache:
            try:
                preset_exists_cache[key] = synth.sfpreset_name(sfid, 0, key) is not None
            except Exception:
                preset_exists_cache[key] = False
        return preset_exists_cache[key]

    for ch in range(16):
        try:
            if preset_exists(0):
                synth.program_select(ch, sfid, 0, 0)
        except Exception:
            pass

    mid = mido.MidiFile(str(midi_path))

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        def write_audio(seconds: float) -> None:
            if seconds <= 0:
                return
            nframes = int(round(seconds * sample_rate))
            if nframes <= 0:
                return
            samples = np.asarray(synth.get_samples(nframes))
            if samples.ndim != 1:
                samples = samples.reshape(-1)
            if samples.size % 2 != 0:
                samples = np.pad(samples, (0, 1))
            wf.writeframes(_float_to_int16(samples).tobytes())

        for msg in mid:
            write_audio(float(msg.time))
            if msg.is_meta:
                continue

            t = msg.type
            ch = getattr(msg, "channel", 0)
            try:
                if t == "note_on":
                    vel = int(msg.velocity)
                    if vel == 0:
                        synth.noteoff(ch, int(msg.note))
                    else:
                        synth.noteon(ch, int(msg.note), vel)
                elif t == "note_off":
                    synth.noteoff(ch, int(msg.note))
                elif t == "program_change":
                    program = int(msg.program)
                    if preset_exists(program):
                        synth.program_select(ch, sfid, 0, program)
                    else:
                        ignored_program_changes.append({
                            "channel": int(ch),
                            "program": int(program),
                        })
                elif t == "control_change":
                    synth.cc(ch, int(msg.control), int(msg.value))
                elif t == "pitchwheel":
                    pb = int(msg.pitch) + 8192
                    pb = max(0, min(16383, pb))
                    synth.pitch_bend(ch, pb)
            except Exception:
                pass

        write_audio(float(extra_tail_sec))

    try:
        synth.delete()
    except Exception:
        pass

    info = FluidSynthRenderInfo(
        sf2_path=sf2_path,
        midi_path=midi_path,
        wav_path=wav_path,
        sample_rate=int(sample_rate),
        gain=float(gain),
        extra_tail_sec=float(extra_tail_sec),
    )
    return {
        "backend": "fluidsynth",
        "info": info.__dict__,
        "ignored_program_changes": ignored_program_changes,
        "wav_path": str(wav_path),
    }
