from __future__ import annotations

import argparse
import json
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from .midi_velocity import write_flat_velocity_midi


@dataclass(frozen=True)
class RenderPairResult:
    midi_path: Path
    instrument_path: Path
    gt_wav: Path
    pred_wav: Path
    flat_velocity: int
    backend: str
    extra: Dict[str, object]


def _sanitize_token(text: str, max_len: int = 40) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    token = token.strip("._-") or "inst"
    return token[:max_len]


def build_render_file_paths(
    *,
    midi_path: str | Path,
    instrument_path: str | Path,
    out_dir: str | Path,
    flat_velocity: int,
) -> Tuple[Path, Path, Path, str]:
    """Build non-colliding output paths for gt/pred/flat-midi.

    Files are grouped under an instrument-specific subfolder:
      out_dir/<instrument_stem>-<instrument_path_hash8>/

    File stem is MIDI stem only.
    """
    midi_path = Path(midi_path)
    instrument_path = Path(instrument_path)
    out_dir = Path(out_dir)

    midi_base = _sanitize_token(midi_path.stem, max_len=80)
    inst_base = _sanitize_token(instrument_path.stem, max_len=40)
    inst_hash = hashlib.sha1(str(instrument_path.resolve()).encode("utf-8")).hexdigest()[:8]
    inst_tag = f"{inst_base}-{inst_hash}"
    render_dir = out_dir / inst_tag
    base = midi_base

    gt_wav = render_dir / f"{base}.gt.wav"
    pred_wav = render_dir / f"{base}.flat{int(flat_velocity)}.wav"
    flat_midi = render_dir / f"{base}.flat{int(flat_velocity)}.mid"
    return gt_wav, pred_wav, flat_midi, base


def _choose_backend(instrument_path: Path, backend: str) -> str:
    backend = backend.lower()
    if backend not in ("auto", "sfizz", "fluidsynth"):
        raise ValueError("backend must be one of: auto, sfizz, fluidsynth")

    if backend != "auto":
        return backend

    ext = instrument_path.suffix.lower()
    if ext == ".sfz":
        return "sfizz"
    if ext == ".sf2":
        return "fluidsynth"
    raise ValueError(
        f"Unknown instrument extension: {instrument_path.suffix}. "
        "Use .sfz (sfizz) or .sf2 (fluidsynth), or pass backend explicitly."
    )


def render_midi_two_versions(
    midi_path: str | Path,
    instrument_path: str | Path,
    *,
    out_dir: str | Path,
    sample_rate: int = 44100,
    flat_velocity: int = 64,
    backend: str = "auto",
    # sfizz options
    sfizz_block_size: int = 1024,
    sfizz_polyphony: int = 256,
    sfizz_quality: int = 3,
    sfizz_use_eot: bool = False,
    # fluidsynth options
    fluidsynth_gain: float = 0.5,
    fluidsynth_extra_tail_sec: float = 2.0,
) -> RenderPairResult:
    """Render the same MIDI into two WAV files.

    - Ground-truth: original MIDI velocities
    - Predicted baseline: NOTE_ON velocities flattened to `flat_velocity`

    The backend is selected based on instrument file extension:
      - .sfz -> sfizz_render CLI
      - .sf2 -> pyfluidsynth

    Returns:
        RenderPairResult
    """
    midi_path = Path(midi_path)
    instrument_path = Path(instrument_path)
    out_dir = Path(out_dir)

    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI not found: {midi_path}")
    if not instrument_path.exists():
        raise FileNotFoundError(f"Instrument not found: {instrument_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_wav, pred_wav, flat_midi, _ = build_render_file_paths(
        midi_path=midi_path,
        instrument_path=instrument_path,
        out_dir=out_dir,
        flat_velocity=int(flat_velocity),
    )

    # Create flattened MIDI next to outputs for reproducibility.
    gt_wav.parent.mkdir(parents=True, exist_ok=True)
    flatten_result = write_flat_velocity_midi(midi_path, flat_midi, flat_velocity=int(flat_velocity))

    chosen = _choose_backend(instrument_path, backend)

    extra: Dict[str, object] = {
        "flatten": flatten_result.__dict__,
    }

    if chosen == "sfizz":
        from .sfizz import render_midi_with_sfz_sfizz

        if instrument_path.suffix.lower() != ".sfz":
            raise ValueError(
                f"backend=sfizz requires .sfz instrument, got {instrument_path}"
            )

        extra["gt"] = render_midi_with_sfz_sfizz(
            midi_path=midi_path,
            sfz_path=instrument_path,
            wav_path=gt_wav,
            sample_rate=sample_rate,
            block_size=sfizz_block_size,
            polyphony=sfizz_polyphony,
            quality=sfizz_quality,
            use_eot=sfizz_use_eot,
        )
        extra["pred"] = render_midi_with_sfz_sfizz(
            midi_path=flat_midi,
            sfz_path=instrument_path,
            wav_path=pred_wav,
            sample_rate=sample_rate,
            block_size=sfizz_block_size,
            polyphony=sfizz_polyphony,
            quality=sfizz_quality,
            use_eot=sfizz_use_eot,
        )

    elif chosen == "fluidsynth":
        from .fluidsynth import render_midi_with_sf2_fluidsynth

        if instrument_path.suffix.lower() != ".sf2":
            raise ValueError(
                f"backend=fluidsynth requires .sf2 instrument, got {instrument_path}. "
                "If you only have .sfz, use backend=sfizz and install sfizz_render."
            )

        extra["gt"] = render_midi_with_sf2_fluidsynth(
            midi_path=midi_path,
            sf2_path=instrument_path,
            wav_path=gt_wav,
            sample_rate=sample_rate,
            gain=fluidsynth_gain,
            extra_tail_sec=fluidsynth_extra_tail_sec,
        )
        extra["pred"] = render_midi_with_sf2_fluidsynth(
            midi_path=flat_midi,
            sf2_path=instrument_path,
            wav_path=pred_wav,
            sample_rate=sample_rate,
            gain=fluidsynth_gain,
            extra_tail_sec=fluidsynth_extra_tail_sec,
        )
    else:
        raise RuntimeError(f"Unexpected backend: {chosen}")

    return RenderPairResult(
        midi_path=midi_path,
        instrument_path=instrument_path,
        gt_wav=gt_wav,
        pred_wav=pred_wav,
        flat_velocity=int(flat_velocity),
        backend=chosen,
        extra=extra,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render one MIDI into two WAVs: original velocity vs flattened velocity."
    )
    parser.add_argument("midi", help="arg1: MIDI file")
    parser.add_argument(
        "--instrument",
        required=True,
        help="Path to instrument file: .sfz (sfizz) or .sf2 (fluidsynth)",
    )
    parser.add_argument("--out_dir", default="./renders", help="Output directory")
    parser.add_argument("--sr", type=int, default=44100, help="Render sample rate")
    parser.add_argument("--flat_velocity", type=int, default=64, help="Flattened velocity")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "sfizz", "fluidsynth"], help="Renderer backend")
    parser.add_argument("--sfizz_block", type=int, default=1024)
    parser.add_argument("--sfizz_poly", type=int, default=256)
    parser.add_argument("--sfizz_quality", type=int, default=3)
    parser.add_argument("--sfizz_use_eot", action="store_true")
    parser.add_argument("--fluidsynth_gain", type=float, default=0.5)
    parser.add_argument("--fluidsynth_tail", type=float, default=2.0)
    parser.add_argument("--json_out", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--mute_output", action="store_true", help="Do not print JSON to stdout")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    res = render_midi_two_versions(
        args.midi,
        args.instrument,
        out_dir=args.out_dir,
        sample_rate=args.sr,
        flat_velocity=args.flat_velocity,
        backend=args.backend,
        sfizz_block_size=args.sfizz_block,
        sfizz_polyphony=args.sfizz_poly,
        sfizz_quality=args.sfizz_quality,
        sfizz_use_eot=args.sfizz_use_eot,
        fluidsynth_gain=args.fluidsynth_gain,
        fluidsynth_extra_tail_sec=args.fluidsynth_tail,
    )
    payload = json.dumps(
        {
            "gt_wav": str(res.gt_wav),
            "pred_wav": str(res.pred_wav),
            "backend": res.backend,
            "extra": res.extra,
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    if not args.mute_output:
        print(payload)


if __name__ == "__main__":
    main()
