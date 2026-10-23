from __future__ import annotations

"""Layer-1 style evaluation: dynamics observables on note events.

This follows the project docs' recommendation to keep **dynamics-focused note-wise
features** as the primary evaluation, and treat embedding/BSSL as secondary.

Use case (MIDI-known scenario):
- You have the MIDI notes.
- You have two rendered audios (gt vs predicted).
- Extract note-wise harmonic energy + simple onset flux proxies.
- Compare the two sets of note-wise measurements.

This is intentionally lightweight and renderer-agnostic.
"""

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import librosa
import numpy as np
import torch
import pretty_midi


@dataclass(frozen=True)
class NoteEvent:
    start: float
    end: float
    pitch: int


def load_note_events(midi_path: str | Path) -> List[NoteEvent]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes: List[NoteEvent] = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append(NoteEvent(float(n.start), float(n.end), int(n.pitch)))
    notes.sort(key=lambda x: x.start)
    return notes


def _hz_from_midi(pitch: int) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def _load_audio_mono(wav_path: str | Path, target_sr: int, device: torch.device) -> torch.Tensor:
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio not found: {wav_path}")
    audio, sr = librosa.load(str(wav_path), sr=None, mono=True)
    if int(sr) != int(target_sr):
        audio = librosa.resample(audio, orig_sr=int(sr), target_sr=int(target_sr))
    y = torch.as_tensor(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    return y.to(device)


def compute_mag_stft(
    wav_path: str | Path,
    *,
    sample_rate: int,
    fft_size: int = 2048,
    frames_per_second: float = 100.0,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute magnitude STFT.

    Returns:
        times: (F,) seconds
        freqs: (K,) Hz
        mag: (F, K) magnitude
    """
    dev = torch.device(device) if device else torch.device("cpu")
    y = _load_audio_mono(wav_path, int(sample_rate), dev)  # (1,T)
    hop = int(round(sample_rate / float(frames_per_second)))
    window = torch.hann_window(int(fft_size), device=dev)

    with torch.no_grad():
        spec = torch.stft(
            y.squeeze(0),
            n_fft=int(fft_size),
            hop_length=int(hop),
            win_length=int(fft_size),
            window=window,
            center=True,
            return_complex=True,
        )  # (K, F)
        mag = spec.abs().transpose(0, 1).cpu().numpy().astype(np.float64)  # (F,K)

    freqs = np.linspace(0.0, sample_rate / 2.0, int(fft_size) // 2 + 1, dtype=np.float64)
    times = np.arange(mag.shape[0], dtype=np.float64) * (hop / float(sample_rate))
    return times, freqs, mag


def extract_note_harmonic_energy(
    *,
    times: np.ndarray,
    freqs: np.ndarray,
    mag: np.ndarray,
    notes: Iterable[NoteEvent],
    harmonics: int = 6,
    bandwidth_hz: float = 15.0,
    onset_window: Tuple[float, float] = (-0.02, 0.08),
    log_compress: bool = True,
) -> np.ndarray:
    """Pitch-conditioned harmonic energy per note.

    For each note, average spectral magnitude in a short onset window around
    f0 and a few harmonics.

    Returns:
        energies: (N,) float
    """
    times = np.asarray(times, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    mag = np.asarray(mag, dtype=np.float64)

    if mag.ndim != 2:
        raise ValueError("mag must be (F,K)")

    F, K = mag.shape
    out: List[float] = []

    # Precompute for fast bin search
    for note in notes:
        t0 = note.start + float(onset_window[0])
        t1 = note.start + float(onset_window[1])
        if t1 <= 0:
            out.append(float("nan"))
            continue
        i0 = int(np.searchsorted(times, max(t0, 0.0), side="left"))
        i1 = int(np.searchsorted(times, t1, side="right"))
        i0 = max(0, min(i0, F))
        i1 = max(0, min(i1, F))
        if i1 <= i0:
            out.append(float("nan"))
            continue

        f0 = _hz_from_midi(int(note.pitch))
        # sum energy over harmonics
        e = 0.0
        for h in range(1, int(harmonics) + 1):
            fh = f0 * h
            if fh >= freqs[-1]:
                break
            # find bins within bandwidth
            lo = fh - bandwidth_hz
            hi = fh + bandwidth_hz
            k0 = int(np.searchsorted(freqs, lo, side="left"))
            k1 = int(np.searchsorted(freqs, hi, side="right"))
            k0 = max(0, min(k0, K))
            k1 = max(0, min(k1, K))
            if k1 <= k0:
                continue
            e += float(np.mean(mag[i0:i1, k0:k1]))

        if log_compress:
            e = float(np.log1p(max(e, 0.0)))
        out.append(e)

    return np.asarray(out, dtype=np.float64)


def extract_note_onset_flux(
    *,
    times: np.ndarray,
    freqs: np.ndarray,
    mag: np.ndarray,
    notes: Iterable[NoteEvent],
    band_hz: Tuple[float, float] = (30.0, 8000.0),
    onset_window: Tuple[float, float] = (-0.02, 0.10),
) -> np.ndarray:
    """A simple onset flux proxy per note.

    We compute spectral flux in a broad band within a short window around onset.
    This is not a full onset detector; it's a cheap dynamics observable.

    Returns:
        flux: (N,)
    """
    times = np.asarray(times, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    mag = np.asarray(mag, dtype=np.float64)

    F, K = mag.shape
    k0 = int(np.searchsorted(freqs, float(band_hz[0]), side="left"))
    k1 = int(np.searchsorted(freqs, float(band_hz[1]), side="right"))
    k0 = max(0, min(k0, K))
    k1 = max(0, min(k1, K))

    out: List[float] = []
    for note in notes:
        t0 = note.start + float(onset_window[0])
        t1 = note.start + float(onset_window[1])
        i0 = int(np.searchsorted(times, max(t0, 0.0), side="left"))
        i1 = int(np.searchsorted(times, t1, side="right"))
        i0 = max(0, min(i0, F))
        i1 = max(0, min(i1, F))
        if i1 - i0 < 2:
            out.append(float("nan"))
            continue

        X = mag[i0:i1, k0:k1]
        dX = np.diff(X, axis=0)
        flux = float(np.sum(np.maximum(dX, 0.0))) / float(dX.size)
        out.append(flux)

    return np.asarray(out, dtype=np.float64)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = ~(np.isnan(a) | np.isnan(b))
    if np.sum(mask) < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _safe_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = ~(np.isnan(a) | np.isnan(b))
    if np.sum(mask) < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def evaluate_note_dynamics_observables(
    *,
    midi_path: str | Path,
    pred_wav: str | Path,
    gt_wav: str | Path,
    sample_rate: int,
    fft_size: int = 2048,
    frames_per_second: float = 100.0,
    harmonics: int = 6,
    device: Optional[str] = None,
    pearson_only: bool = False,
) -> Dict[str, object]:
    """Compare two audios via note-wise harmonic energy + onset flux.

    Returns a dict of summary metrics (corr + MAE) for both observables.
    """
    notes = load_note_events(midi_path)

    t_p, f_p, m_p = compute_mag_stft(
        pred_wav,
        sample_rate=sample_rate,
        fft_size=fft_size,
        frames_per_second=frames_per_second,
        device=device,
    )
    t_g, f_g, m_g = compute_mag_stft(
        gt_wav,
        sample_rate=sample_rate,
        fft_size=fft_size,
        frames_per_second=frames_per_second,
        device=device,
    )

    # energies
    e_p = extract_note_harmonic_energy(times=t_p, freqs=f_p, mag=m_p, notes=notes, harmonics=harmonics)
    e_g = extract_note_harmonic_energy(times=t_g, freqs=f_g, mag=m_g, notes=notes, harmonics=harmonics)

    # flux
    x_p = extract_note_onset_flux(times=t_p, freqs=f_p, mag=m_p, notes=notes)
    x_g = extract_note_onset_flux(times=t_g, freqs=f_g, mag=m_g, notes=notes)

    def mae(a: np.ndarray, b: np.ndarray) -> float:
        mask = ~(np.isnan(a) | np.isnan(b))
        if not np.any(mask):
            return float("nan")
        return float(np.mean(np.abs(a[mask] - b[mask])))

    harmonic_pearson = _safe_corr(e_p, e_g)
    onset_pearson = _safe_corr(x_p, x_g)
    harmonic_metrics = {
        "pearson": harmonic_pearson,
    }
    onset_metrics = {
        "pearson": onset_pearson,
    }
    if not pearson_only:
        harmonic_metrics.update(
            {
                "cosine_sim": _safe_cosine_sim(e_p, e_g),
                "mae": mae(e_p, e_g),
            }
        )
        onset_metrics.update(
            {
                "cosine_sim": _safe_cosine_sim(x_p, x_g),
                "mae": mae(x_p, x_g),
            }
        )

    res = {
        "config": {
            "sample_rate": int(sample_rate),
            "fft_size": int(fft_size),
            "frames_per_second": float(frames_per_second),
            "harmonics": int(harmonics),
            "device": device or "cpu",
            "pearson_only": bool(pearson_only),
        },
        "counts": {
            "num_notes": int(len(notes)),
        },
        "harmonic_energy": harmonic_metrics,
        "onset_flux": onset_metrics,
    }
    return res


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layer-1 note-wise dynamics observables evaluation")
    parser.add_argument("midi", help="MIDI file (notes source)")
    parser.add_argument("pred_wav", help="predicted WAV")
    parser.add_argument("gt_wav", help="ground-truth WAV")
    parser.add_argument("sample_rate", type=int, help="analysis sampling rate")
    parser.add_argument("--fft", type=int, default=2048)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--harmonics", type=int, default=6)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    res = evaluate_note_dynamics_observables(
        midi_path=args.midi,
        pred_wav=args.pred_wav,
        gt_wav=args.gt_wav,
        sample_rate=args.sample_rate,
        fft_size=args.fft,
        frames_per_second=args.fps,
        harmonics=args.harmonics,
        device=args.device,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
