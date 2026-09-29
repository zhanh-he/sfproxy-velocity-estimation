from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class DynamicsFeatureConfig:
    """Config for note-wise dynamics features."""

    n_fft: int = 2048
    hop: int = 221
    win_length: Optional[int] = None

    harmonic_count: int = 5
    band_bins: int = 1

    # Window for harmonic energy after onset.
    energy_window_s: float = 0.12

    # Window for onset flux around onset.
    onset_pre_s: float = 0.02
    onset_post_s: float = 0.08

    # Output
    include_f0_ratio: bool = False


def _stft_mag(audio: torch.Tensor, cfg: DynamicsFeatureConfig) -> torch.Tensor:
    """Compute magnitude spectrogram.

    Args:
        audio: (T,) float32

    Returns:
        mag: (F, TT)
    """

    if audio.ndim != 1:
        raise ValueError("audio must be 1D")

    win_len = int(cfg.win_length) if cfg.win_length is not None else int(cfg.n_fft)
    window = torch.hann_window(win_len, device=audio.device)

    spec = torch.stft(
        audio,
        n_fft=int(cfg.n_fft),
        hop_length=int(cfg.hop),
        win_length=win_len,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = spec.abs()
    return mag


def _frame_times(num_frames: int, sr: int, hop: int) -> torch.Tensor:
    # Frame index corresponds to hop-centered time when center=True.
    # We approximate by hop/sr.
    return torch.arange(num_frames, dtype=torch.float32) * (float(hop) / float(sr))


def extract_note_features_padded(
    audio: torch.Tensor,
    pitch: torch.Tensor,
    cont: torch.Tensor,
    mask: torch.Tensor,
    sr: int,
    seg_len_s: float,
    cfg: DynamicsFeatureConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract padded note-wise features.

    Args:
        audio: (T,) float32
        pitch: (Nmax,) int64
        cont: (Nmax, 3) float32 [onset_s, dur_s, vel_01]
        mask: (Nmax,) bool

    Returns:
        feats: (Nmax, D) float32
        mask_out: same as input mask
    """

    if audio.ndim != 1:
        audio = audio.view(-1)

    nmax = int(pitch.shape[0])

    # STFT magnitude
    mag = _stft_mag(audio, cfg)  # (F, TT)
    f_bins, t_bins = mag.shape

    times = _frame_times(t_bins, sr=sr, hop=cfg.hop).to(audio.device)
    freq_res = float(sr) / float(cfg.n_fft)

    # Prepare output
    d = 2 + (1 if cfg.include_f0_ratio else 0)
    feats = torch.zeros((nmax, d), dtype=torch.float32, device=audio.device)

    # Precompute per-frame spectral flux for onset windows
    # flux_t is defined on frames 1..TT-1
    if t_bins >= 2:
        diff = mag[:, 1:] - mag[:, :-1]
        flux_t = torch.relu(diff).sum(dim=0)  # (TT-1,)
        # Pad one element at start to align with frame indices
        flux_t = torch.cat([torch.zeros((1,), device=audio.device), flux_t], dim=0)  # (TT,)
    else:
        flux_t = torch.zeros((t_bins,), device=audio.device)

    # Iterate notes
    for i in range(nmax):
        if not bool(mask[i].item()):
            continue

        p = int(pitch[i].item())
        onset_s = float(cont[i, 0].item())
        dur_s = float(cont[i, 1].item())

        if onset_s < 0.0 or onset_s >= float(seg_len_s):
            continue

        f0 = 440.0 * (2.0 ** ((p - 69) / 12.0))

        # Harmonic energy window
        e0 = onset_s
        e1 = min(float(seg_len_s), onset_s + float(cfg.energy_window_s))
        if e1 <= e0:
            e1 = min(float(seg_len_s), onset_s + 0.02)

        # Frame indices
        idx0 = int(torch.searchsorted(times, torch.tensor(e0, device=times.device)).item())
        idx1 = int(torch.searchsorted(times, torch.tensor(e1, device=times.device)).item())
        idx0 = max(0, min(idx0, t_bins - 1))
        idx1 = max(idx0 + 1, min(idx1, t_bins))

        # Harmonic bins
        harm_bins = []
        for h in range(1, int(cfg.harmonic_count) + 1):
            fh = f0 * float(h)
            if fh >= float(sr) / 2.0:
                break
            b = int(round(fh / freq_res))
            if 0 <= b < f_bins:
                harm_bins.append(b)

        if len(harm_bins) == 0:
            continue

        # Sum energy around each harmonic bin with +/- band
        band = int(cfg.band_bins)
        energy = 0.0
        f0_energy = 0.0
        for j, b in enumerate(harm_bins):
            lo = max(0, b - band)
            hi = min(f_bins, b + band + 1)
            # squared magnitude approximates energy
            e = (mag[lo:hi, idx0:idx1] ** 2).mean().item()
            energy += float(e)
            if j == 0:
                f0_energy = float(e)

        energy = max(0.0, energy)
        # log compression
        feats[i, 0] = math.log1p(energy)

        # Onset flux window
        o0 = max(0.0, onset_s - float(cfg.onset_pre_s))
        o1 = min(float(seg_len_s), onset_s + float(cfg.onset_post_s))
        j0 = int(torch.searchsorted(times, torch.tensor(o0, device=times.device)).item())
        j1 = int(torch.searchsorted(times, torch.tensor(o1, device=times.device)).item())
        j0 = max(0, min(j0, t_bins - 1))
        j1 = max(j0 + 1, min(j1, t_bins))

        onset_flux = float(flux_t[j0:j1].mean().item()) if (j1 > j0) else 0.0
        feats[i, 1] = math.log1p(max(0.0, onset_flux))

        if cfg.include_f0_ratio:
            ratio = 0.0
            if energy > 0:
                ratio = float(f0_energy / energy)
            feats[i, 2] = float(ratio)

    return feats, mask
