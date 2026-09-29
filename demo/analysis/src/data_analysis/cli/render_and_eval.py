from __future__ import annotations

import argparse
from pathlib import Path

from data_analysis.evaluation.bssl_eval import evaluate_bssl_pair
from data_analysis.evaluation.note_dynamics import evaluate_note_dynamics_observables
from data_analysis.rendering.render_pair import render_midi_two_versions
from ._dataset_utils import save_and_print_json


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render MIDI twice (gt vs flat velocity) and evaluate with Layer-1 + Layer-3 metrics."
    )
    p.add_argument("midi", help="arg1: MIDI file")
    p.add_argument(
        "--instrument",
        required=True,
        help="Path to instrument file: .sfz (sfizz) or .sf2 (fluidsynth)",
    )
    p.add_argument("--out_dir", default="./renders")
    p.add_argument("--render_sr", type=int, default=44100)
    p.add_argument("--eval_sr", type=int, default=22050)
    p.add_argument("--flat_velocity", type=int, default=64)
    p.add_argument("--backend", type=str, default="auto", choices=["auto", "sfizz", "fluidsynth"])

    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--fft", type=int, default=1024)
    p.add_argument("--bssl_mode", type=str, default="sone", choices=["sone", "bark"])
    p.add_argument("--num_samples", type=int, default=2048)
    p.add_argument("--norm", type=str, default="zscore", choices=["zscore", "minmax", "none"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ntot_plot", action="store_true", help="Save pred-vs-gt ntot plot PNG")
    p.add_argument("--ntot_plot_path", type=str, default=None, help="Optional PNG path for ntot comparison plot")
    p.add_argument("--skip_note_dynamics", action="store_true", help="Skip Layer-1 note-wise dynamics metrics")
    p.add_argument("--note_fps", type=float, default=100.0)
    p.add_argument("--note_fft", type=int, default=2048)
    p.add_argument("--note_harmonics", type=int, default=6)
    p.add_argument("--mute_render_output", action="store_true", help="Hide render block in final JSON")
    p.add_argument("--mute_eval_output", action="store_true", help="Hide eval block in final JSON")
    p.add_argument("--json_out", type=str, default=None, help="Optional JSON output path")
    p.add_argument("--mute_output", action="store_true", help="Do not print JSON to stdout")

    args = p.parse_args()

    render_res = render_midi_two_versions(
        args.midi,
        args.instrument,
        out_dir=args.out_dir,
        sample_rate=args.render_sr,
        flat_velocity=args.flat_velocity,
        backend=args.backend,
    )

    plot_prefix = Path(render_res.pred_wav).name
    if ".flat" in plot_prefix:
        plot_prefix = plot_prefix.split(".flat", 1)[0]
    default_plot = Path(render_res.pred_wav).parent / f"{plot_prefix}.ntot.compare.png"
    ntot_plot_path = None
    if args.ntot_plot:
        ntot_plot_path = Path(args.ntot_plot_path) if args.ntot_plot_path else default_plot

    eval_res = evaluate_bssl_pair(
        pred_wav=render_res.pred_wav,
        gt_wav=render_res.gt_wav,
        sample_rate=args.eval_sr,
        frames_per_second=args.fps,
        fft_size=args.fft,
        bssl_mode=args.bssl_mode,
        num_samples=args.num_samples,
        normalization=args.norm,
        device=args.device,
        ntot_plot_path=ntot_plot_path,
    )
    note_res = None
    if not args.skip_note_dynamics:
        note_res = evaluate_note_dynamics_observables(
            midi_path=args.midi,
            pred_wav=render_res.pred_wav,
            gt_wav=render_res.gt_wav,
            sample_rate=args.eval_sr,
            fft_size=args.note_fft,
            frames_per_second=args.note_fps,
            harmonics=args.note_harmonics,
            device=args.device,
        )

    out = {}
    if not args.mute_render_output:
        out["render"] = {
            "gt_wav": str(render_res.gt_wav),
            "pred_wav": str(render_res.pred_wav),
            "backend": render_res.backend,
            "extra": render_res.extra,
        }
    if not args.mute_eval_output:
        out["eval"] = eval_res
        if note_res is not None:
            out["note_dynamics"] = note_res

    save_and_print_json(
        out,
        json_out=args.json_out,
        mute_output=args.mute_output,
    )


if __name__ == "__main__":
    main()
