from __future__ import annotations

"""Visualize a piano SoundFont response over pitch and velocity.

Expected workflow
-----------------
1. Probe the SoundFont by rendering one pitch sweep per velocity.
2. Save note-wise metrics to CSV.
3. Plot two panels:
   - Bark dB summary vs pitch
   - Ntot sone peak vs pitch

The plotting code is intentionally independent from the audio backend so that
once the CSV exists, you can re-style or re-plot the figure quickly.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd

from .soundfont_probe import (
    SoundfontProbeConfig,
    build_highlight_velocity_list,
    midi_to_pitch_name,
    probe_soundfont,
)


BLACK_KEY_CLASSES = {1, 3, 6, 8, 10}

__all__ = [
    "load_soundfont_probe_csv",
    "draw_piano_keyboard",
    "plot_soundfont_response_figure",
    "run_probe_and_plot",
    "build_argument_parser",
    "main",
]


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "pitch",
        "velocity",
        "bark_peak_db_avg",
        "ntot_peak_sone",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {sorted(missing)}")
    out = df.copy()
    out["pitch"] = out["pitch"].astype(int)
    out["velocity"] = out["velocity"].astype(int)
    out = out.sort_values(["velocity", "pitch"]).reset_index(drop=True)
    return out


def load_soundfont_probe_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return _prepare_dataframe(pd.read_csv(csv_path))


def _is_black_key(pitch: int) -> bool:
    return int(pitch % 12) in BLACK_KEY_CLASSES


def _pitch_ticks(pitch_min: int, pitch_max: int) -> Tuple[List[int], List[str]]:
    ticks: List[int] = []
    labels: List[str] = []
    for pitch in range(int(pitch_min), int(pitch_max) + 1):
        if pitch % 12 == 0:
            ticks.append(int(pitch))
            labels.append(f"{midi_to_pitch_name(pitch)}\n({pitch})")
    return ticks, labels


def draw_piano_keyboard(ax: plt.Axes, pitch_min: int, pitch_max: int) -> None:
    """Draw a simple keyboard strip aligned to MIDI semitone positions."""
    ax.set_xlim(float(pitch_min) - 0.5, float(pitch_max) + 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_facecolor("white")

    for pitch in range(int(pitch_min), int(pitch_max) + 1):
        x0 = float(pitch) - 0.5
        ax.add_patch(
            patches.Rectangle(
                (x0, 0.0),
                1.0,
                1.0,
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
                zorder=1,
            )
        )

    for pitch in range(int(pitch_min), int(pitch_max) + 1):
        if not _is_black_key(pitch):
            continue
        x0 = float(pitch) - 0.30
        ax.add_patch(
            patches.Rectangle(
                (x0, 0.35),
                0.60,
                0.65,
                facecolor="black",
                edgecolor="black",
                linewidth=0.8,
                zorder=2,
            )
        )

    ticks, labels = _pitch_ticks(pitch_min, pitch_max)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=2)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#555555")
    ax.set_xlabel("Pitch", fontsize=15)


def _velocity_norm(df: pd.DataFrame) -> mpl.colors.Normalize:
    vmin = int(df["velocity"].min())
    vmax = int(df["velocity"].max())
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax)


def _panel_y_limits(values: np.ndarray, mode: str) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)

    if mode == "db":
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        lo = 6.0 * np.floor((vmin - 1.0) / 6.0)
        hi = min(0.0, 6.0 * np.ceil((vmax + 1.0) / 6.0))
        if hi <= lo:
            hi = lo + 6.0
        return lo, hi

    vmax = float(np.max(values))
    hi = max(1.0, np.ceil(vmax * 1.08))
    return 0.0, hi


def _style_main_axis(ax: plt.Axes, ylabel: str, ylim: Tuple[float, float]) -> None:
    ax.set_facecolor("white")
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(*ylim)
    ax.grid(True, which="major", axis="both", linestyle=(0, (1.2, 3.2)), color="#9c9c9c", alpha=0.7)
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", labelsize=12)
    for side in ax.spines:
        ax.spines[side].set_color("#666666")


def _plot_metric_panel(
    *,
    ax: plt.Axes,
    kax: plt.Axes,
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    highlight_velocities: Sequence[int],
    pitch_min: int,
    pitch_max: int,
    label_pitch: Optional[int] = None,
) -> None:
    pitches = np.arange(int(pitch_min), int(pitch_max) + 1)
    label_pitch = int(label_pitch) if label_pitch is not None else int(np.median(pitches))

    all_values: List[float] = []
    velocities = sorted(int(v) for v in df["velocity"].dropna().unique())
    for velocity in velocities:
        sub = df[df["velocity"] == int(velocity)].sort_values("pitch")
        if sub.empty:
            continue
        x = sub["pitch"].to_numpy(dtype=np.float64)
        y = sub[metric_col].to_numpy(dtype=np.float64)
        all_values.extend(y[np.isfinite(y)].tolist())
        ax.plot(
            x,
            y,
            color=cmap(norm(float(velocity))),
            linewidth=1.0,
            alpha=0.9,
            zorder=2,
        )

    for velocity in highlight_velocities:
        sub = df[df["velocity"] == int(velocity)].sort_values("pitch")
        if sub.empty:
            continue
        x = sub["pitch"].to_numpy(dtype=np.float64)
        y = sub[metric_col].to_numpy(dtype=np.float64)
        ax.plot(x, y, color="black", linewidth=1.0, alpha=0.75, zorder=3)

        valid = np.isfinite(y)
        if not np.any(valid):
            continue
        x_valid = x[valid]
        y_valid = y[valid]
        idx = int(np.argmin(np.abs(x_valid - float(label_pitch))))
        ax.text(
            float(x_valid[idx]) + 0.25,
            float(y_valid[idx]),
            f"{int(velocity)}",
            fontsize=12,
            color="#2f2f2f",
            alpha=0.75,
            va="center",
            ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.35, pad=0.2),
            zorder=4,
        )

    ax.set_xlim(float(pitch_min) - 0.5, float(pitch_max) + 0.5)
    ylim_mode = "db" if "db" in metric_col else "sone"
    ylim = _panel_y_limits(np.asarray(all_values, dtype=np.float64), mode=ylim_mode)
    _style_main_axis(ax, ylabel=ylabel, ylim=ylim)
    ax.set_title(title, fontsize=19, weight="bold", pad=24)

    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.028,
        pad=0.015,
    )
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("MIDI velocity", fontsize=15)

    draw_piano_keyboard(kax, pitch_min=pitch_min, pitch_max=pitch_max)


def plot_soundfont_response_figure(
    df: pd.DataFrame,
    *,
    output_path: Optional[str | Path] = None,
    instrument_name: Optional[str] = None,
    title: Optional[str] = None,
    pitch_min: Optional[int] = None,
    pitch_max: Optional[int] = None,
    highlight_velocity_step: int = 10,
    db_metric_col: str = "bark_peak_db_avg",
    sone_metric_col: str = "ntot_peak_sone",
    figsize: Tuple[float, float] = (16.0, 10.5),
) -> plt.Figure:
    df = _prepare_dataframe(df)
    pitch_min = int(df["pitch"].min()) if pitch_min is None else int(pitch_min)
    pitch_max = int(df["pitch"].max()) if pitch_max is None else int(pitch_max)
    instrument_name = instrument_name or str(df.get("instrument_name", pd.Series(["SoundFont"])).iloc[0])
    title = title or instrument_name

    highlight_velocities = build_highlight_velocity_list(
        int(df["velocity"].min()),
        int(df["velocity"].max()),
        highlight_step=int(highlight_velocity_step),
    )

    cmap = mpl.cm.get_cmap("YlOrRd")
    norm = _velocity_norm(df)

    fig = plt.figure(figsize=figsize, facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.36)

    top = outer[0].subgridspec(2, 1, height_ratios=[12, 2], hspace=0.0)
    bot = outer[1].subgridspec(2, 1, height_ratios=[12, 2], hspace=0.0)

    ax_db = fig.add_subplot(top[0])
    kax_db = fig.add_subplot(top[1], sharex=ax_db)
    ax_sone = fig.add_subplot(bot[0])
    kax_sone = fig.add_subplot(bot[1], sharex=ax_sone)

    _plot_metric_panel(
        ax=ax_db,
        kax=kax_db,
        df=df,
        metric_col=db_metric_col,
        ylabel="Peak sound level (dB avg)",
        title="Sound Intensity (dB) – Pitch – MIDI Velocity",
        cmap=cmap,
        norm=norm,
        highlight_velocities=highlight_velocities,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        label_pitch=max(pitch_min, min(pitch_max, 59)),
    )

    _plot_metric_panel(
        ax=ax_sone,
        kax=kax_sone,
        df=df,
        metric_col=sone_metric_col,
        ylabel="Peak loudness (sone)",
        title="Sones – Pitch – MIDI Velocity",
        cmap=cmap,
        norm=norm,
        highlight_velocities=highlight_velocities,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        label_pitch=max(pitch_min, min(pitch_max, 59)),
    )

    fig.suptitle(title, fontsize=25, weight="bold", y=0.98)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def run_probe_and_plot(
    *,
    instrument_path: str | Path,
    out_dir: str | Path,
    backend: str = "auto",
    render_sr: int = 44100,
    eval_sr: int = 22050,
    pitch_min: int = 21,
    pitch_max: int = 108,
    velocity_min: int = 10,
    velocity_max: int = 110,
    velocity_step: int = 2,
    highlight_velocity_step: int = 10,
    note_duration_sec: float = 2.2,
    analysis_duration_sec: float = 2.0,
    inter_note_gap_sec: float = 0.2,
    initial_silence_sec: float = 0.1,
    final_tail_sec: float = 1.0,
    fft_size: int = 1024,
    frames_per_second: float = 50.0,
    db_max: float = 96.0,
    outer_ear: str = "terhardt",
    device: Optional[str] = None,
    sfizz_block_size: int = 1024,
    sfizz_polyphony: int = 256,
    sfizz_quality: int = 3,
    fluidsynth_gain: float = 0.5,
    keep_wav: bool = False,
    keep_midi: bool = True,
    overwrite: bool = False,
    figure_name: Optional[str] = None,
) -> dict:
    out_dir = Path(out_dir).expanduser().resolve()
    config = SoundfontProbeConfig(
        instrument_path=str(instrument_path),
        out_dir=str(out_dir),
        backend=backend,
        render_sr=int(render_sr),
        eval_sr=int(eval_sr),
        pitch_min=int(pitch_min),
        pitch_max=int(pitch_max),
        velocity_min=int(velocity_min),
        velocity_max=int(velocity_max),
        velocity_step=int(velocity_step),
        highlight_velocity_step=int(highlight_velocity_step),
        note_duration_sec=float(note_duration_sec),
        analysis_duration_sec=float(analysis_duration_sec),
        inter_note_gap_sec=float(inter_note_gap_sec),
        initial_silence_sec=float(initial_silence_sec),
        final_tail_sec=float(final_tail_sec),
        fft_size=int(fft_size),
        frames_per_second=float(frames_per_second),
        db_max=float(db_max),
        outer_ear=outer_ear,
        device=device,
        sfizz_block_size=int(sfizz_block_size),
        sfizz_polyphony=int(sfizz_polyphony),
        sfizz_quality=int(sfizz_quality),
        fluidsynth_gain=float(fluidsynth_gain),
        keep_wav=bool(keep_wav),
        keep_midi=bool(keep_midi),
        overwrite=bool(overwrite),
    )

    probe_result = probe_soundfont(config)
    figure_name = figure_name or f"{Path(instrument_path).stem}_soundfont_response.png"
    figure_path = out_dir / figure_name
    fig = plot_soundfont_response_figure(
        probe_result.dataframe,
        output_path=figure_path,
        instrument_name=Path(instrument_path).stem,
        title=Path(instrument_path).stem,
        highlight_velocity_step=int(highlight_velocity_step),
    )
    plt.close(fig)

    payload = {
        "config": asdict(config),
        "csv_path": str(probe_result.csv_path),
        "summary_json_path": str(probe_result.summary_json_path),
        "figure_path": str(figure_path),
        "num_records": int(len(probe_result.dataframe)),
    }
    result_json_path = out_dir / "soundfont_visualization_result.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    payload["result_json_path"] = str(result_json_path)
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render + plot piano SoundFont loudness curves across pitch and velocity.")
    p.add_argument("--instrument", type=str, default=None, help="Path to .sf2 or .sfz instrument")
    p.add_argument("--csv_path", type=str, default=None, help="Use an existing CSV instead of re-rendering")
    p.add_argument("--out_dir", type=str, default=None, help="Output directory")
    p.add_argument("--backend", type=str, default="auto", choices=["auto", "fluidsynth", "sfizz"])
    p.add_argument("--render_sr", type=int, default=44100)
    p.add_argument("--eval_sr", type=int, default=22050)
    p.add_argument("--pitch_min", type=int, default=21)
    p.add_argument("--pitch_max", type=int, default=108)
    p.add_argument("--velocity_min", type=int, default=10)
    p.add_argument("--velocity_max", type=int, default=110)
    p.add_argument("--velocity_step", type=int, default=2)
    p.add_argument("--highlight_velocity_step", type=int, default=10)
    p.add_argument("--note_duration", type=float, default=2.2)
    p.add_argument("--analysis_duration", type=float, default=2.0)
    p.add_argument("--inter_note_gap", type=float, default=0.2)
    p.add_argument("--initial_silence", type=float, default=0.1)
    p.add_argument("--final_tail", type=float, default=1.0)
    p.add_argument("--fft_size", type=int, default=1024)
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--db_max", type=float, default=96.0)
    p.add_argument("--outer_ear", type=str, default="terhardt")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--sfizz_block_size", type=int, default=1024)
    p.add_argument("--sfizz_polyphony", type=int, default=256)
    p.add_argument("--sfizz_quality", type=int, default=3)
    p.add_argument("--fluidsynth_gain", type=float, default=0.5)
    p.add_argument("--keep_wav", action="store_true")
    p.add_argument("--keep_midi", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--figure_name", type=str, default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argument_parser().parse_args(argv)

    if args.csv_path is None and args.instrument is None:
        raise ValueError("You must provide either --csv_path or --instrument")

    if args.csv_path is not None:
        csv_path = Path(args.csv_path).expanduser().resolve()
        df = load_soundfont_probe_csv(csv_path)
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else csv_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        instrument_name = args.title or str(df.get("instrument_name", pd.Series([csv_path.stem])).iloc[0])
        figure_name = args.figure_name or f"{instrument_name}_soundfont_response.png"
        figure_path = out_dir / figure_name
        fig = plot_soundfont_response_figure(
            df,
            output_path=figure_path,
            instrument_name=instrument_name,
            title=instrument_name,
            pitch_min=args.pitch_min,
            pitch_max=args.pitch_max,
            highlight_velocity_step=args.highlight_velocity_step,
        )
        plt.close(fig)
        print(json.dumps({
            "csv_path": str(csv_path),
            "figure_path": str(figure_path),
        }, ensure_ascii=False, indent=2))
        return

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else Path.cwd() / "soundfont_probe_output"
    result = run_probe_and_plot(
        instrument_path=args.instrument,
        out_dir=out_dir,
        backend=args.backend,
        render_sr=args.render_sr,
        eval_sr=args.eval_sr,
        pitch_min=args.pitch_min,
        pitch_max=args.pitch_max,
        velocity_min=args.velocity_min,
        velocity_max=args.velocity_max,
        velocity_step=args.velocity_step,
        highlight_velocity_step=args.highlight_velocity_step,
        note_duration_sec=args.note_duration,
        analysis_duration_sec=args.analysis_duration,
        inter_note_gap_sec=args.inter_note_gap,
        initial_silence_sec=args.initial_silence,
        final_tail_sec=args.final_tail,
        fft_size=args.fft_size,
        frames_per_second=args.fps,
        db_max=args.db_max,
        outer_ear=args.outer_ear,
        device=args.device,
        sfizz_block_size=args.sfizz_block_size,
        sfizz_polyphony=args.sfizz_polyphony,
        sfizz_quality=args.sfizz_quality,
        fluidsynth_gain=args.fluidsynth_gain,
        keep_wav=args.keep_wav,
        keep_midi=args.keep_midi,
        overwrite=args.overwrite,
        figure_name=args.figure_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
