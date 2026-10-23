from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .bssl_eval import evaluate_bssl_pair
from .note_dynamics import evaluate_note_dynamics_observables


def evaluate_real_reference(
    *,
    midi_path: str | Path,
    real_wav: str | Path,
    gt_syn_wav: str | Path,
    pred_syn_wav: str | Path,
    sample_rate: int,
    frames_per_second: float = 50.0,
    fft_size: int = 1024,
    bssl_mode: str = "sone",
    num_samples: int = 2048,
    normalization: str = "zscore",
    device: Optional[str] = None,
    ntot_plot_real_gt_path: Optional[str | Path] = None,
    ntot_plot_real_pred_path: Optional[str | Path] = None,
    ntot_plot_gt_pred_path: Optional[str | Path] = None,
    run_note_dynamics: bool = True,
    note_fft_size: int = 2048,
    note_frames_per_second: float = 100.0,
    note_harmonics: int = 6,
    pearson_only: bool = False,
) -> Dict[str, object]:
    """Advanced evaluation with real audio as the only reference.

    Returns three metric groups:
      - real_vs_gt_syn
      - real_vs_pred_syn
      - gt_vs_pred_syn
    """
    midi_path = Path(midi_path)
    real_wav = Path(real_wav)
    gt_syn_wav = Path(gt_syn_wav)
    pred_syn_wav = Path(pred_syn_wav)

    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI not found: {midi_path}")
    if not real_wav.exists():
        raise FileNotFoundError(f"Real audio not found: {real_wav}")
    if not gt_syn_wav.exists():
        raise FileNotFoundError(f"GT synthetic audio not found: {gt_syn_wav}")
    if not pred_syn_wav.exists():
        raise FileNotFoundError(f"Pred synthetic audio not found: {pred_syn_wav}")

    real_vs_gt = evaluate_bssl_pair(
        pred_wav=gt_syn_wav,
        gt_wav=real_wav,
        sample_rate=sample_rate,
        frames_per_second=frames_per_second,
        fft_size=fft_size,
        bssl_mode=bssl_mode,
        num_samples=num_samples,
        normalization=normalization,
        device=device,
        ntot_plot_path=ntot_plot_real_gt_path,
        pearson_only=pearson_only,
    )

    real_vs_pred = evaluate_bssl_pair(
        pred_wav=pred_syn_wav,
        gt_wav=real_wav,
        sample_rate=sample_rate,
        frames_per_second=frames_per_second,
        fft_size=fft_size,
        bssl_mode=bssl_mode,
        num_samples=num_samples,
        normalization=normalization,
        device=device,
        ntot_plot_path=ntot_plot_real_pred_path,
        pearson_only=pearson_only,
    )

    gt_vs_pred = evaluate_bssl_pair(
        pred_wav=pred_syn_wav,
        gt_wav=gt_syn_wav,
        sample_rate=sample_rate,
        frames_per_second=frames_per_second,
        fft_size=fft_size,
        bssl_mode=bssl_mode,
        num_samples=num_samples,
        normalization=normalization,
        device=device,
        ntot_plot_path=ntot_plot_gt_pred_path,
        pearson_only=pearson_only,
    )

    real_vs_gt_note = None
    real_vs_pred_note = None
    gt_vs_pred_note = None
    if run_note_dynamics:
        real_vs_gt_note = evaluate_note_dynamics_observables(
            midi_path=midi_path,
            pred_wav=gt_syn_wav,
            gt_wav=real_wav,
            sample_rate=sample_rate,
            fft_size=note_fft_size,
            frames_per_second=note_frames_per_second,
            harmonics=note_harmonics,
            device=device,
            pearson_only=pearson_only,
        )
        real_vs_pred_note = evaluate_note_dynamics_observables(
            midi_path=midi_path,
            pred_wav=pred_syn_wav,
            gt_wav=real_wav,
            sample_rate=sample_rate,
            fft_size=note_fft_size,
            frames_per_second=note_frames_per_second,
            harmonics=note_harmonics,
            device=device,
            pearson_only=pearson_only,
        )
        gt_vs_pred_note = evaluate_note_dynamics_observables(
            midi_path=midi_path,
            pred_wav=pred_syn_wav,
            gt_wav=gt_syn_wav,
            sample_rate=sample_rate,
            fft_size=note_fft_size,
            frames_per_second=note_frames_per_second,
            harmonics=note_harmonics,
            device=device,
            pearson_only=pearson_only,
        )

    summary: Dict[str, float] = {}
    for side, payload in (
        ("real_vs_gt", real_vs_gt),
        ("real_vs_pred", real_vs_pred),
        ("gt_vs_pred", gt_vs_pred),
    ):
        summary[f"{side}_bssl_pearson"] = float(payload["summary"]["bssl_pearson_correlation"])
        summary[f"{side}_ntot_pearson"] = float(payload["summary"]["ntot_pearson_correlation"])
        if not pearson_only:
            summary[f"{side}_bssl_cosine_sim"] = float(payload["summary"]["bssl_cosine_sim"])
            summary[f"{side}_bssl_mae"] = float(payload["summary"]["bssl_mean_absolute_error"])
            summary[f"{side}_ntot_cosine_sim"] = float(payload["summary"]["ntot_cosine_sim"])
            summary[f"{side}_ntot_mae"] = float(payload["summary"]["ntot_mean_absolute_error"])

    out: Dict[str, object] = {
        "reference": {
            "midi_path": str(midi_path),
            "real_wav": str(real_wav),
            "gt_syn_wav": str(gt_syn_wav),
            "pred_syn_wav": str(pred_syn_wav),
        },
        "real_vs_gt_syn": real_vs_gt,
        "real_vs_pred_syn": real_vs_pred,
        "gt_vs_pred_syn": gt_vs_pred,
        "summary": summary,
    }
    if real_vs_gt_note is not None and real_vs_pred_note is not None and gt_vs_pred_note is not None:
        out["real_vs_gt_note_dynamics"] = real_vs_gt_note
        out["real_vs_pred_note_dynamics"] = real_vs_pred_note
        out["gt_vs_pred_note_dynamics"] = gt_vs_pred_note
        summary = out["summary"]  # type: ignore[assignment]
        assert isinstance(summary, dict)
        note_groups = ("harmonic_energy", "onset_flux")
        for side, note in (
            ("real_vs_gt", real_vs_gt_note),
            ("real_vs_pred", real_vs_pred_note),
            ("gt_vs_pred", gt_vs_pred_note),
        ):
            for group in note_groups:
                summary[f"{side}_{group}_pearson"] = float(note[group]["pearson"])  # type: ignore[index]
                if not pearson_only:
                    summary[f"{side}_{group}_cosine_sim"] = float(note[group]["cosine_sim"])  # type: ignore[index]
                    summary[f"{side}_{group}_mae"] = float(note[group]["mae"])  # type: ignore[index]
    return out
