from __future__ import annotations

"""Probe a piano SoundFont by rendering single-note pitch sweeps.

This module is designed for the SoundFont visualisation requested in the repo:
- x-axis: MIDI pitch
- y-axis: extracted note-level metric
- z-axis: MIDI velocity (encoded by line color)

Implementation notes
--------------------
1. We generate one MIDI file per velocity.
2. Each MIDI file contains a sequential sweep over the requested pitches.
3. We render the MIDI with either SF2 (FluidSynth) or SFZ (sfizz).
4. We extract note-wise metrics from the rendered WAV:
   - Bark dB curve  -> peak of frame-wise Bark-band average dB
   - Ntot sone curve -> peak total loudness (sone)
5. We save one CSV per velocity, plus one merged CSV for plotting.

The heavy audio dependencies are imported lazily so that `--help` still works
in environments where torchaudio / fluidsynth are not yet available.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


BLACK_KEY_CLASSES = {1, 3, 6, 8, 10}


@dataclass(frozen=True)
class SoundfontProbeConfig:
    instrument_path: str
    out_dir: str
    backend: str = "auto"
    render_sr: int = 44100
    eval_sr: int = 22050
    pitch_min: int = 21
    pitch_max: int = 108
    velocity_min: int = 10
    velocity_max: int = 110
    velocity_step: int = 2
    highlight_velocity_step: int = 10
    note_duration_sec: float = 2.2
    analysis_duration_sec: float = 2.0
    inter_note_gap_sec: float = 0.2
    initial_silence_sec: float = 0.1
    final_tail_sec: float = 1.0
    fft_size: int = 1024
    frames_per_second: float = 50.0
    db_max: float = 96.0
    outer_ear: str = "terhardt"
    device: Optional[str] = None
    channel: int = 0
    program: int = 0
    sfizz_block_size: int = 1024
    sfizz_polyphony: int = 256
    sfizz_quality: int = 3
    fluidsynth_gain: float = 0.5
    keep_wav: bool = False
    keep_midi: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class ProbeRunResult:
    config: SoundfontProbeConfig
    csv_path: Path
    summary_json_path: Path
    raw_dir: Path
    dataframe: pd.DataFrame


@dataclass(frozen=True)
class PitchSweepScheduleItem:
    pitch: int
    velocity: int
    pitch_name: str
    is_black_key: bool
    note_on_sec: float
    note_off_sec: float
    analysis_start_sec: float
    analysis_end_sec: float


def midi_to_pitch_name(pitch: int) -> str:
    """Return scientific pitch notation, e.g. 60 -> C4."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = int(pitch // 12) - 1
    return f"{names[int(pitch % 12)]}{octave}"


def is_black_key(pitch: int) -> bool:
    return int(pitch % 12) in BLACK_KEY_CLASSES


def build_pitch_list(pitch_min: int, pitch_max: int) -> List[int]:
    if pitch_min > pitch_max:
        raise ValueError("pitch_min must be <= pitch_max")
    return list(range(int(pitch_min), int(pitch_max) + 1))


def build_velocity_list(velocity_min: int, velocity_max: int, velocity_step: int) -> List[int]:
    if velocity_min <= 0 or velocity_max > 127:
        raise ValueError("velocity range must stay inside MIDI [1, 127]")
    if velocity_step <= 0:
        raise ValueError("velocity_step must be positive")
    if velocity_min > velocity_max:
        raise ValueError("velocity_min must be <= velocity_max")
    return list(range(int(velocity_min), int(velocity_max) + 1, int(velocity_step)))


def build_highlight_velocity_list(
    velocity_min: int,
    velocity_max: int,
    *,
    highlight_step: int = 10,
) -> List[int]:
    if highlight_step <= 0:
        raise ValueError("highlight_step must be positive")
    first = ((int(velocity_min) + highlight_step - 1) // highlight_step) * highlight_step
    if first > velocity_max:
        return []
    return list(range(first, int(velocity_max) + 1, int(highlight_step)))


def _choose_backend(instrument_path: Path, backend: str) -> str:
    backend = str(backend).lower()
    if backend not in {"auto", "fluidsynth", "sfizz"}:
        raise ValueError("backend must be one of: auto, fluidsynth, sfizz")
    if backend != "auto":
        return backend
    ext = instrument_path.suffix.lower()
    if ext == ".sf2":
        return "fluidsynth"
    if ext == ".sfz":
        return "sfizz"
    raise ValueError(f"Unsupported instrument extension: {instrument_path.suffix}")


def create_pitch_sweep_midi(
    *,
    midi_path: str | Path,
    pitches: Sequence[int],
    velocity: int,
    note_duration_sec: float = 2.2,
    analysis_duration_sec: float = 2.0,
    inter_note_gap_sec: float = 0.2,
    initial_silence_sec: float = 0.1,
    final_tail_sec: float = 1.0,
    channel: int = 0,
    program: int = 0,
    tempo_bpm: float = 120.0,
    ticks_per_beat: int = 480,
) -> pd.DataFrame:
    """Create one MIDI file that sweeps through all requested pitches.

    One note is active at a time. The analysis window is always anchored at the
    note onset, which keeps the per-note extraction simple and reproducible.
    """
    if note_duration_sec <= 0:
        raise ValueError("note_duration_sec must be positive")
    if analysis_duration_sec <= 0:
        raise ValueError("analysis_duration_sec must be positive")
    if analysis_duration_sec > note_duration_sec + inter_note_gap_sec:
        raise ValueError("analysis_duration_sec is too long for the note/gap layout")

    midi_path = Path(midi_path)
    midi_path.parent.mkdir(parents=True, exist_ok=True)

    import mido

    mid = mido.MidiFile(ticks_per_beat=int(ticks_per_beat))
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(float(tempo_bpm))
    track.append(mido.MetaMessage("set_tempo", tempo=int(tempo), time=0))
    track.append(mido.Message("program_change", channel=int(channel), program=int(program), time=0))

    def sec_to_ticks(seconds: float) -> int:
        return int(round(mido.second2tick(float(seconds), mid.ticks_per_beat, tempo)))

    schedule: List[PitchSweepScheduleItem] = []
    current_sec = float(initial_silence_sec)
    pending_delta_sec = float(initial_silence_sec)

    for pitch in pitches:
        track.append(
            mido.Message(
                "note_on",
                note=int(pitch),
                velocity=int(velocity),
                channel=int(channel),
                time=sec_to_ticks(pending_delta_sec),
            )
        )
        track.append(
            mido.Message(
                "note_off",
                note=int(pitch),
                velocity=0,
                channel=int(channel),
                time=sec_to_ticks(note_duration_sec),
            )
        )

        schedule.append(
            PitchSweepScheduleItem(
                pitch=int(pitch),
                velocity=int(velocity),
                pitch_name=midi_to_pitch_name(int(pitch)),
                is_black_key=is_black_key(int(pitch)),
                note_on_sec=float(current_sec),
                note_off_sec=float(current_sec + note_duration_sec),
                analysis_start_sec=float(current_sec),
                analysis_end_sec=float(current_sec + analysis_duration_sec),
            )
        )

        current_sec = float(current_sec + note_duration_sec + inter_note_gap_sec)
        pending_delta_sec = float(inter_note_gap_sec)

    track.append(mido.MetaMessage("end_of_track", time=sec_to_ticks(final_tail_sec)))
    mid.save(str(midi_path))

    return pd.DataFrame([asdict(x) for x in schedule])


def render_pitch_sweep_midi(
    *,
    midi_path: str | Path,
    instrument_path: str | Path,
    wav_path: str | Path,
    backend: str = "auto",
    sample_rate: int = 44100,
    sfizz_block_size: int = 1024,
    sfizz_polyphony: int = 256,
    sfizz_quality: int = 3,
    fluidsynth_gain: float = 0.5,
) -> Dict[str, object]:
    """Render the sweep MIDI with the requested instrument backend."""
    midi_path = Path(midi_path)
    instrument_path = Path(instrument_path)
    wav_path = Path(wav_path)
    chosen = _choose_backend(instrument_path, backend)

    if chosen == "fluidsynth":
        from .fluidsynth import render_midi_with_sf2_fluidsynth

        return render_midi_with_sf2_fluidsynth(
            midi_path=midi_path,
            sf2_path=instrument_path,
            wav_path=wav_path,
            sample_rate=int(sample_rate),
            gain=float(fluidsynth_gain),
            extra_tail_sec=0.0,
        )

    from .sfizz import render_midi_with_sfz_sfizz

    return render_midi_with_sfz_sfizz(
        midi_path=midi_path,
        sfz_path=instrument_path,
        wav_path=wav_path,
        sample_rate=int(sample_rate),
        block_size=int(sfizz_block_size),
        polyphony=int(sfizz_polyphony),
        quality=int(sfizz_quality),
    )


def _load_audio_mono(
    wav_path: str | Path,
    *,
    target_sample_rate: int,
    device: Optional[str] = None,
):
    import torch
    import torchaudio

    dev = torch.device(device) if device else torch.device("cpu")
    waveform, sample_rate = torchaudio.load(str(wav_path))
    waveform = waveform.mean(dim=0, keepdim=True)
    if int(sample_rate) != int(target_sample_rate):
        waveform = torchaudio.functional.resample(waveform, int(sample_rate), int(target_sample_rate))
        sample_rate = int(target_sample_rate)
    return waveform.to(dev), int(sample_rate), dev


def _extract_feature_tracks(
    wav_path: str | Path,
    *,
    eval_sr: int,
    frames_per_second: float,
    fft_size: int,
    db_max: float,
    outer_ear: str,
    device: Optional[str] = None,
) -> Dict[str, object]:
    import torch

    from .feature_extractor import PsychoFeatureExtractor

    waveform, sample_rate, dev = _load_audio_mono(
        wav_path,
        target_sample_rate=int(eval_sr),
        device=device,
    )

    bark_extractor = PsychoFeatureExtractor(
        sample_rate=int(sample_rate),
        fft_size=int(fft_size),
        frames_per_second=float(frames_per_second),
        db_max=float(db_max),
        outer_ear=outer_ear,
        return_mode="bark",
    ).to(dev)

    ntot_extractor = PsychoFeatureExtractor(
        sample_rate=int(sample_rate),
        fft_size=int(fft_size),
        frames_per_second=float(frames_per_second),
        db_max=float(db_max),
        outer_ear=outer_ear,
        return_mode="ntot",
    ).to(dev)

    with torch.no_grad():
        bark = bark_extractor(waveform).squeeze(0).cpu().numpy().astype(np.float64)  # (C, F)
        ntot = ntot_extractor(waveform).squeeze(0).cpu().numpy().astype(np.float64)  # (F,)

    hop_duration = float(bark_extractor.hop_size) / float(bark_extractor.sample_rate)
    frame_times = np.arange(bark.shape[1], dtype=np.float64) * hop_duration
    waveform_np = waveform.squeeze(0).cpu().numpy().astype(np.float64)

    return {
        "waveform": waveform_np,
        "sample_rate": int(sample_rate),
        "bark": bark,
        "ntot": ntot,
        "frame_times": frame_times,
        "hop_duration": hop_duration,
    }


def extract_note_metrics_from_wav(
    *,
    wav_path: str | Path,
    schedule_df: pd.DataFrame,
    eval_sr: int = 22050,
    frames_per_second: float = 50.0,
    fft_size: int = 1024,
    db_max: float = 96.0,
    outer_ear: str = "terhardt",
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Extract note-wise Bark dB and Ntot sone peaks from one rendered WAV.

    Definitions used in the output CSV
    ----------------------------------
    - bark_peak_db_avg:
        Peak over time of the frame-wise mean Bark-band level.
        We subtract `db_max` so the y-axis behaves like a dBFS-style negative scale.
    - ntot_peak_sone:
        Peak of the total loudness curve in sones.
    - waveform_peak_dbfs / waveform_rms_dbfs:
        Direct time-domain references. These are not used in the main figure,
        but they are useful for sanity-checking a SoundFont.
    """
    feature_tracks = _extract_feature_tracks(
        wav_path,
        eval_sr=int(eval_sr),
        frames_per_second=float(frames_per_second),
        fft_size=int(fft_size),
        db_max=float(db_max),
        outer_ear=outer_ear,
        device=device,
    )

    waveform = np.asarray(feature_tracks["waveform"], dtype=np.float64)
    sample_rate = int(feature_tracks["sample_rate"])
    bark = np.asarray(feature_tracks["bark"], dtype=np.float64)  # (C, F)
    ntot = np.asarray(feature_tracks["ntot"], dtype=np.float64)  # (F,)
    frame_times = np.asarray(feature_tracks["frame_times"], dtype=np.float64)

    eps = 1e-12
    rows: List[Dict[str, object]] = []
    for item in schedule_df.to_dict(orient="records"):
        start_sec = float(item["analysis_start_sec"])
        end_sec = float(item["analysis_end_sec"])
        mask = (frame_times >= start_sec) & (frame_times < end_sec)

        bark_peak_db_avg = float("nan")
        bark_mean_db_avg = float("nan")
        ntot_peak_sone = float("nan")
        ntot_mean_sone = float("nan")
        num_frames = int(np.sum(mask))

        if num_frames > 0:
            bark_seg = bark[:, mask]
            bark_frame_mean_db = bark_seg.mean(axis=0) - float(db_max)
            bark_peak_db_avg = float(np.max(bark_frame_mean_db))
            bark_mean_db_avg = float(np.mean(bark_frame_mean_db))

            ntot_seg = ntot[mask]
            ntot_peak_sone = float(np.max(ntot_seg))
            ntot_mean_sone = float(np.mean(ntot_seg))

        s0 = max(0, int(round(start_sec * sample_rate)))
        s1 = min(int(waveform.shape[-1]), int(round(end_sec * sample_rate)))
        segment = waveform[s0:s1]
        if segment.size:
            peak = float(np.max(np.abs(segment)))
            rms = float(np.sqrt(np.mean(np.square(segment))))
            waveform_peak_dbfs = float(20.0 * np.log10(max(peak, eps)))
            waveform_rms_dbfs = float(20.0 * np.log10(max(rms, eps)))
        else:
            waveform_peak_dbfs = float("nan")
            waveform_rms_dbfs = float("nan")

        rows.append(
            {
                **item,
                "bark_peak_db_avg": bark_peak_db_avg,
                "bark_mean_db_avg": bark_mean_db_avg,
                "ntot_peak_sone": ntot_peak_sone,
                "ntot_mean_sone": ntot_mean_sone,
                "waveform_peak_dbfs": waveform_peak_dbfs,
                "waveform_rms_dbfs": waveform_rms_dbfs,
                "analysis_num_frames": int(num_frames),
            }
        )

    return pd.DataFrame(rows)




def _to_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)


def probe_soundfont(config: SoundfontProbeConfig) -> ProbeRunResult:
    """Main driver.

    This function is resume-friendly at the velocity level. Each velocity gets its
    own CSV under `out_dir/raw`. The merged CSV is written after the loop.
    """
    instrument_path = Path(config.instrument_path).expanduser().resolve()
    if not instrument_path.exists():
        raise FileNotFoundError(f"Instrument not found: {instrument_path}")

    chosen_backend = _choose_backend(instrument_path, config.backend)
    pitches = build_pitch_list(config.pitch_min, config.pitch_max)
    velocities = build_velocity_list(config.velocity_min, config.velocity_max, config.velocity_step)
    highlight_velocities = build_highlight_velocity_list(
        config.velocity_min,
        config.velocity_max,
        highlight_step=config.highlight_velocity_step,
    )

    out_dir = Path(config.out_dir).expanduser().resolve()
    raw_dir = out_dir / "raw"
    midi_dir = out_dir / "midis"
    wav_dir = out_dir / "wavs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if config.keep_midi:
        midi_dir.mkdir(parents=True, exist_ok=True)
    if config.keep_wav:
        wav_dir.mkdir(parents=True, exist_ok=True)

    all_frames: List[pd.DataFrame] = []
    for velocity in velocities:
        raw_csv_path = raw_dir / f"velocity_{int(velocity):03d}.csv"
        if raw_csv_path.exists() and not config.overwrite:
            cached_df = pd.read_csv(raw_csv_path)
            all_frames.append(cached_df)
            print(f"[probe_soundfont] reuse cached CSV for velocity={velocity}: {raw_csv_path}")
            continue

        midi_path = midi_dir / f"pitch_sweep_v{int(velocity):03d}.mid"
        wav_path = wav_dir / f"pitch_sweep_v{int(velocity):03d}.wav"
        if not config.keep_wav:
            wav_path.parent.mkdir(parents=True, exist_ok=True)
        if not config.keep_midi:
            midi_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[probe_soundfont] render velocity={velocity}")
        schedule_df = create_pitch_sweep_midi(
            midi_path=midi_path,
            pitches=pitches,
            velocity=int(velocity),
            note_duration_sec=float(config.note_duration_sec),
            analysis_duration_sec=float(config.analysis_duration_sec),
            inter_note_gap_sec=float(config.inter_note_gap_sec),
            initial_silence_sec=float(config.initial_silence_sec),
            final_tail_sec=float(config.final_tail_sec),
            channel=int(config.channel),
            program=int(config.program),
        )

        render_info = render_pitch_sweep_midi(
            midi_path=midi_path,
            instrument_path=instrument_path,
            wav_path=wav_path,
            backend=chosen_backend,
            sample_rate=int(config.render_sr),
            sfizz_block_size=int(config.sfizz_block_size),
            sfizz_polyphony=int(config.sfizz_polyphony),
            sfizz_quality=int(config.sfizz_quality),
            fluidsynth_gain=float(config.fluidsynth_gain),
        )

        velocity_df = extract_note_metrics_from_wav(
            wav_path=wav_path,
            schedule_df=schedule_df,
            eval_sr=int(config.eval_sr),
            frames_per_second=float(config.frames_per_second),
            fft_size=int(config.fft_size),
            db_max=float(config.db_max),
            outer_ear=config.outer_ear,
            device=config.device,
        )

        velocity_df["instrument_name"] = instrument_path.stem
        velocity_df["instrument_path"] = str(instrument_path)
        velocity_df["backend"] = chosen_backend
        velocity_df["render_sr"] = int(config.render_sr)
        velocity_df["eval_sr"] = int(config.eval_sr)
        velocity_df["frames_per_second"] = float(config.frames_per_second)
        velocity_df["fft_size"] = int(config.fft_size)
        velocity_df["db_max"] = float(config.db_max)
        velocity_df["outer_ear"] = config.outer_ear
        velocity_df["highlight_velocity"] = velocity_df["velocity"].astype(int).isin(highlight_velocities)
        velocity_df["midi_path"] = str(midi_path)
        velocity_df["wav_path"] = str(wav_path)
        velocity_df["render_backend_info"] = json.dumps(_to_jsonable(render_info), ensure_ascii=False)
        velocity_df.to_csv(raw_csv_path, index=False)
        all_frames.append(velocity_df)

        if not config.keep_wav and wav_path.exists():
            wav_path.unlink()
        if not config.keep_midi and midi_path.exists():
            midi_path.unlink()

    if not all_frames:
        raise RuntimeError("No data was generated. Please check the configuration.")

    merged_df = pd.concat(all_frames, ignore_index=True)
    merged_df = merged_df.sort_values(["velocity", "pitch"]).reset_index(drop=True)

    csv_path = out_dir / "soundfont_probe_metrics.csv"
    summary_json_path = out_dir / "soundfont_probe_summary.json"
    merged_df.to_csv(csv_path, index=False)

    summary = {
        "config": asdict(config),
        "instrument_name": instrument_path.stem,
        "instrument_path": str(instrument_path),
        "backend": chosen_backend,
        "num_records": int(len(merged_df)),
        "num_pitches": int(len(pitches)),
        "num_velocities": int(len(velocities)),
        "pitch_min": int(min(pitches)),
        "pitch_max": int(max(pitches)),
        "velocity_min": int(min(velocities)),
        "velocity_max": int(max(velocities)),
        "highlight_velocities": [int(v) for v in highlight_velocities],
        "csv_path": str(csv_path),
        "raw_dir": str(raw_dir),
        "stats": {
            "bark_peak_db_avg": {
                "min": float(np.nanmin(merged_df["bark_peak_db_avg"].to_numpy(dtype=float))),
                "max": float(np.nanmax(merged_df["bark_peak_db_avg"].to_numpy(dtype=float))),
            },
            "ntot_peak_sone": {
                "min": float(np.nanmin(merged_df["ntot_peak_sone"].to_numpy(dtype=float))),
                "max": float(np.nanmax(merged_df["ntot_peak_sone"].to_numpy(dtype=float))),
            },
        },
    }
    _write_json(summary_json_path, summary)

    return ProbeRunResult(
        config=config,
        csv_path=csv_path,
        summary_json_path=summary_json_path,
        raw_dir=raw_dir,
        dataframe=merged_df,
    )


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Render a piano SoundFont pitch/velocity sweep and save note-wise metrics.")
    p.add_argument("instrument", help="Path to .sf2 or .sfz piano instrument")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for CSVs and temporary assets")
    p.add_argument("--backend", type=str, default="auto", choices=["auto", "fluidsynth", "sfizz"])
    p.add_argument("--render_sr", type=int, default=44100)
    p.add_argument("--eval_sr", type=int, default=22050)
    p.add_argument("--pitch_min", type=int, default=21)
    p.add_argument("--pitch_max", type=int, default=108)
    p.add_argument("--velocity_min", type=int, default=10)
    p.add_argument("--velocity_max", type=int, default=110)
    p.add_argument("--velocity_step", type=int, default=2)
    p.add_argument("--highlight_velocity_step", type=int, default=10)
    p.add_argument("--note_duration", type=float, default=2.2, help="Seconds between note_on and note_off")
    p.add_argument("--analysis_duration", type=float, default=2.0, help="Seconds used to measure the note")
    p.add_argument("--inter_note_gap", type=float, default=0.2)
    p.add_argument("--initial_silence", type=float, default=0.1)
    p.add_argument("--final_tail", type=float, default=1.0)
    p.add_argument("--fft_size", type=int, default=1024)
    p.add_argument("--fps", type=float, default=50.0, help="frames_per_second for psychoacoustic features")
    p.add_argument("--db_max", type=float, default=96.0)
    p.add_argument("--outer_ear", type=str, default="terhardt")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--program", type=int, default=0)
    p.add_argument("--sfizz_block_size", type=int, default=1024)
    p.add_argument("--sfizz_polyphony", type=int, default=256)
    p.add_argument("--sfizz_quality", type=int, default=3)
    p.add_argument("--fluidsynth_gain", type=float, default=0.5)
    p.add_argument("--keep_wav", action="store_true")
    p.add_argument("--keep_midi", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _main() -> None:
    args = _parse_args()
    config = SoundfontProbeConfig(
        instrument_path=args.instrument,
        out_dir=args.out_dir,
        backend=args.backend,
        render_sr=int(args.render_sr),
        eval_sr=int(args.eval_sr),
        pitch_min=int(args.pitch_min),
        pitch_max=int(args.pitch_max),
        velocity_min=int(args.velocity_min),
        velocity_max=int(args.velocity_max),
        velocity_step=int(args.velocity_step),
        highlight_velocity_step=int(args.highlight_velocity_step),
        note_duration_sec=float(args.note_duration),
        analysis_duration_sec=float(args.analysis_duration),
        inter_note_gap_sec=float(args.inter_note_gap),
        initial_silence_sec=float(args.initial_silence),
        final_tail_sec=float(args.final_tail),
        fft_size=int(args.fft_size),
        frames_per_second=float(args.fps),
        db_max=float(args.db_max),
        outer_ear=args.outer_ear,
        device=args.device,
        channel=int(args.channel),
        program=int(args.program),
        sfizz_block_size=int(args.sfizz_block_size),
        sfizz_polyphony=int(args.sfizz_polyphony),
        sfizz_quality=int(args.sfizz_quality),
        fluidsynth_gain=float(args.fluidsynth_gain),
        keep_wav=bool(args.keep_wav),
        keep_midi=bool(args.keep_midi),
        overwrite=bool(args.overwrite),
    )
    result = probe_soundfont(config)
    print(json.dumps({
        "csv_path": str(result.csv_path),
        "summary_json_path": str(result.summary_json_path),
        "num_records": int(len(result.dataframe)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
