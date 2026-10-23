from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi


MIDI_VALUE_RANGE = 128


@dataclass(frozen=True)
class MidiNote:
    start: float
    end: float
    pitch: int
    velocity: int


@dataclass
class MidiDatasetStats:
    dataset_name: str
    midi_files: list[Path]
    pitch_counts: np.ndarray
    velocity_counts: np.ndarray
    velocity_boundaries_01: tuple[float, float]
    duration_hist_counts: np.ndarray
    duration_bin_edges: np.ndarray
    ioi_hist_counts: np.ndarray
    ioi_bin_edges: np.ndarray
    chord_size_values: np.ndarray
    chord_size_counts: np.ndarray
    note_count: int
    onset_cluster_count: int
    max_polyphony: int
    errors: list[tuple[Path, str]]

    @property
    def missing_pitches(self) -> list[int]:
        return np.flatnonzero(self.pitch_counts == 0).tolist()

    @property
    def missing_velocities(self) -> list[int]:
        return np.flatnonzero(self.velocity_counts == 0).tolist()


def discover_midi_files(dataset_root: Path | str, patterns: list[str] | None = None) -> list[Path]:
    root = Path(dataset_root)
    midi_files = [path for pattern in (patterns or ["**/*.mid", "**/*.midi"]) for path in root.glob(pattern)]
    return sorted({path.resolve() for path in midi_files if path.is_file()})


def _safe_dataset_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return slug or "dataset"


def _load_notes(midi_path: Path) -> list[MidiNote]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [
        MidiNote(float(note.start), float(note.end), int(note.pitch), int(note.velocity))
        for instrument in pm.instruments
        for note in instrument.notes
    ]
    return sorted(notes, key=lambda note: (note.start, note.pitch, note.end))


def _group_onsets(notes: list[MidiNote], tolerance_s: float) -> list[list[MidiNote]]:
    if not notes:
        return []
    groups = [[notes[0]]]
    anchor = notes[0].start
    for note in notes[1:]:
        if note.start - anchor <= tolerance_s:
            groups[-1].append(note)
        else:
            groups.append([note])
            anchor = note.start
    return groups


def _max_polyphony(notes: list[MidiNote]) -> int:
    events = sorted(
        [(note.start, 1) for note in notes] + [(note.end, -1) for note in notes],
        key=lambda item: (item[0], item[1]),
    )
    active = max_active = 0
    for _, delta in events:
        active += delta
        max_active = max(max_active, active)
    return max_active


def _hist(values: list[float], upper: float, bin_size: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(0.0, upper + bin_size, bin_size, dtype=np.float64)
    return np.histogram(np.asarray(values, dtype=np.float64), bins=edges)[0].astype(np.int64), edges


def analyze_midi_files(
    midi_files: list[Path],
    *,
    dataset_name: str,
    onset_group_tolerance_s: float,
    duration_max: float,
    duration_bin_size: float,
    ioi_max: float,
    ioi_bin_size: float,
) -> MidiDatasetStats:
    pitch_counts = np.zeros(MIDI_VALUE_RANGE, dtype=np.int64)
    velocity_counts = np.zeros(MIDI_VALUE_RANGE, dtype=np.int64)
    durations: list[float] = []
    iois: list[float] = []
    chord_sizes: list[int] = []
    velocity_values_01: list[float] = []
    note_count = 0
    onset_cluster_count = 0
    max_polyphony = 0
    errors: list[tuple[Path, str]] = []

    for midi_path in midi_files:
        try:
            notes = _load_notes(midi_path)
        except Exception as exc:  # noqa: BLE001
            errors.append((midi_path, str(exc)))
            continue
        max_polyphony = max(max_polyphony, _max_polyphony(notes))
        note_count += len(notes)

        for note in notes:
            pitch_counts[note.pitch] += 1
            velocity_counts[note.velocity] += 1
            velocity_values_01.append(note.velocity / 127.0)
            durations.append(max(0.01, note.end - note.start))

        groups = _group_onsets(notes, onset_group_tolerance_s)
        onset_cluster_count += len(groups)
        chord_sizes.extend(len(group) for group in groups)
        iois.extend(curr[0].start - prev[0].start for prev, curr in zip(groups[:-1], groups[1:]))

    duration_hist_counts, duration_bin_edges = _hist(durations, duration_max, duration_bin_size)
    ioi_hist_counts, ioi_bin_edges = _hist(iois, ioi_max, ioi_bin_size)
    if note_count == 0:
        raise ValueError("No readable MIDI notes found under the provided root/patterns.")

    chord_size_values, chord_size_counts = np.unique(np.asarray(chord_sizes, dtype=np.int64), return_counts=True)
    velocity_boundaries_01 = tuple(float(v) for v in np.quantile(np.asarray(velocity_values_01), [0.33, 0.66]))

    return MidiDatasetStats(
        dataset_name=dataset_name,
        midi_files=midi_files,
        pitch_counts=pitch_counts,
        velocity_counts=velocity_counts,
        velocity_boundaries_01=velocity_boundaries_01,
        duration_hist_counts=duration_hist_counts,
        duration_bin_edges=duration_bin_edges,
        ioi_hist_counts=ioi_hist_counts,
        ioi_bin_edges=ioi_bin_edges,
        chord_size_values=chord_size_values,
        chord_size_counts=chord_size_counts,
        note_count=note_count,
        onset_cluster_count=onset_cluster_count,
        max_polyphony=max_polyphony,
        errors=errors,
    )


def _plot_counts(
    x: np.ndarray,
    counts: np.ndarray,
    *,
    xlabel: str,
    title: str,
    output_path: Path,
    dpi: int,
    widths: np.ndarray | None = None,
    color: str = "tab:blue",
    xlim: tuple[float, float] | None = None,
) -> plt.Figure:
    percentages = counts / counts.sum() * 100.0
    fig, ax_left = plt.subplots(figsize=(6, 2))
    if widths is None:
        ax_left.bar(x, counts, color=color, alpha=0.8, label="Total count")
    else:
        ax_left.bar(x, counts, width=widths, color=color, alpha=0.8, label="Total count")
    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel("Total count")
    if xlim is not None:
        ax_left.set_xlim(*xlim)

    ax_right = ax_left.twinx()
    ax_right.plot(x, percentages, color="tab:red", linewidth=1.5, label="Percentage")
    ax_right.set_ylabel("Percentage (%)")

    ax_left.set_title(title)
    ax_left.grid(True, axis="y", alpha=0.25)
    handles, labels = [], []
    for ax in (ax_left, ax_right):
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    ax_left.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return fig


def midi_stats_to_dict(stats: MidiDatasetStats, *, config: dict) -> dict:
    pitch_values = np.flatnonzero(stats.pitch_counts)
    velocity_values = np.flatnonzero(stats.velocity_counts)
    return {
        "format_version": 2,
        "dataset": stats.dataset_name,
        "midi_file_count": len(stats.midi_files),
        "note_count": stats.note_count,
        "onset_cluster_count": stats.onset_cluster_count,
        "max_polyphony": stats.max_polyphony,
        "pitch_range_observed": [int(pitch_values.min()), int(pitch_values.max())],
        "velocity_range_observed": [int(velocity_values.min()), int(velocity_values.max())],
        "pitches": {"values": pitch_values.tolist(), "counts": stats.pitch_counts[pitch_values].tolist()},
        "durations": {
            "bin_edges": stats.duration_bin_edges.tolist(),
            "hist_counts": stats.duration_hist_counts.tolist(),
        },
        "iois": {
            "bin_edges": stats.ioi_bin_edges.tolist(),
            "hist_counts": stats.ioi_hist_counts.tolist(),
        },
        "chord_sizes": {
            "values": stats.chord_size_values.tolist(),
            "counts": stats.chord_size_counts.tolist(),
        },
        "velocities_01": {
            "values": (velocity_values.astype(np.float64) / 127.0).tolist(),
            "counts": stats.velocity_counts[velocity_values].tolist(),
        },
        "velocity_boundaries_01": list(stats.velocity_boundaries_01),
        "config": config,
        "errors": [(str(path), message) for path, message in stats.errors],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan one MIDI dataset, plot distributions, and export sampler stats.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--patterns", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/midi_dataset_stats"))
    parser.add_argument("--json-out-dir", type=Path, default=Path("stats/midi_sampler"))
    parser.add_argument("--onset-group-tolerance-ms", type=float, default=20.0)
    parser.add_argument("--duration-max", type=float, default=4.0)
    parser.add_argument("--duration-bin-size", type=float, default=0.02)
    parser.add_argument("--ioi-max", type=float, default=2.0)
    parser.add_argument("--ioi-bin-size", type=float, default=0.02)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no-show", action="store_true")
    return parser


def main(args: argparse.Namespace) -> int:
    midi_files = discover_midi_files(args.root, args.patterns)
    stats = analyze_midi_files(
        midi_files,
        dataset_name=args.dataset,
        onset_group_tolerance_s=args.onset_group_tolerance_ms / 1000.0,
        duration_max=args.duration_max,
        duration_bin_size=args.duration_bin_size,
        ioi_max=args.ioi_max,
        ioi_bin_size=args.ioi_bin_size,
    )
    slug = _safe_dataset_slug(args.dataset)

    figures = [
        _plot_counts(
            np.arange(MIDI_VALUE_RANGE),
            stats.pitch_counts,
            xlabel="MIDI pitch",
            title=f"{args.dataset} MIDI Pitch Distribution",
            output_path=args.output_dir / f"{slug}_pitch.png",
            dpi=args.dpi,
            xlim=(-0.5, MIDI_VALUE_RANGE - 0.5),
        ),
        _plot_counts(
            np.arange(MIDI_VALUE_RANGE),
            stats.velocity_counts,
            xlabel="MIDI velocity",
            title=f"{args.dataset} MIDI Velocity Distribution",
            output_path=args.output_dir / f"{slug}_velocity.png",
            dpi=args.dpi,
            xlim=(-0.5, MIDI_VALUE_RANGE - 0.5),
        ),
        _plot_counts(
            0.5 * (stats.duration_bin_edges[:-1] + stats.duration_bin_edges[1:]),
            stats.duration_hist_counts,
            xlabel="Duration (s)",
            title=f"{args.dataset} Note Duration Distribution",
            output_path=args.output_dir / f"{slug}_duration.png",
            dpi=args.dpi,
            widths=np.diff(stats.duration_bin_edges),
        ),
        _plot_counts(
            0.5 * (stats.ioi_bin_edges[:-1] + stats.ioi_bin_edges[1:]),
            stats.ioi_hist_counts,
            xlabel="IOI (s)",
            title=f"{args.dataset} Onset IOI Distribution",
            output_path=args.output_dir / f"{slug}_ioi.png",
            dpi=args.dpi,
            widths=np.diff(stats.ioi_bin_edges),
        ),
        _plot_counts(
            stats.chord_size_values,
            stats.chord_size_counts,
            xlabel="Chord size",
            title=f"{args.dataset} Chord Size Distribution",
            output_path=args.output_dir / f"{slug}_chord_size.png",
            dpi=args.dpi,
            color="tab:purple",
        ),
    ]

    payload = midi_stats_to_dict(
        stats,
        config={
            "onset_group_tolerance_ms": args.onset_group_tolerance_ms,
            "duration_max": args.duration_max,
            "duration_bin_size": args.duration_bin_size,
            "ioi_max": args.ioi_max,
            "ioi_bin_size": args.ioi_bin_size,
        },
    )
    args.json_out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.json_out_dir / f"{slug}_sampler.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Scanned {args.dataset}: {len(stats.midi_files)} MIDI files, total notes = {stats.note_count:,}")
    if stats.errors:
        print(f"Skipped corrupt/unreadable MIDI files: {len(stats.errors)}")
    print(f"Missing pitches ({len(stats.missing_pitches)}): {stats.missing_pitches}")
    print(f"Missing velocities ({len(stats.missing_velocities)}): {stats.missing_velocities}")
    print(f"Saved: {json_path}")

    if not args.no_show:
        plt.show()
    for fig in figures:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(build_arg_parser().parse_args()))
