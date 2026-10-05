from __future__ import annotations

import argparse
from pathlib import Path

from data_analysis.evaluation.adv_eval import evaluate_real_reference
from data_analysis.rendering.render_pair import render_midi_two_versions
from ._dataset_utils import save_and_print_json


def _infer_real_audio_from_midi(midi_path: Path) -> Path:
    mp3 = midi_path.with_suffix(".mp3")
    if mp3.exists():
        return mp3
    wav = midi_path.with_suffix(".wav")
    if wav.exists():
        return wav
    raise FileNotFoundError(
        f"Cannot infer real audio for {midi_path}. Tried: {mp3} and {wav}. "
        "Pass --real_audio explicitly."
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Advanced single-file evaluation: real(reference) vs gt_syn and real(reference) vs pred_syn."
    )
    p.add_argument("midi", help="MIDI file used to synthesize gt/pred")
    p.add_argument("--real_audio", type=str, default=None, help="Real recording path (.mp3/.wav). If omitted, infer from MIDI stem.")
    p.add_argument("--instrument", required=True, help="Path to instrument file: .sfz or .sf2")
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
    p.add_argument("--skip_note_dynamics", action="store_true", help="Skip Layer-1 note-wise dynamics metrics")
    p.add_argument("--note_fps", type=float, default=100.0)
    p.add_argument("--note_fft", type=int, default=2048)
    p.add_argument("--note_harmonics", type=int, default=6)
    p.add_argument("--pearson_only", action="store_true", help="Only compute/report Pearson metrics in the exported summaries")
    p.add_argument("--ntot_plot", action="store_true", help="Save two ntot plots: real-vs-gt and real-vs-pred")
    p.add_argument("--json_out", type=str, default=None)
    p.add_argument("--mute_output", action="store_true")

    args = p.parse_args()

    midi_path = Path(args.midi).resolve()
    real_audio = Path(args.real_audio).resolve() if args.real_audio else _infer_real_audio_from_midi(midi_path)

    render_res = render_midi_two_versions(
        midi_path,
        args.instrument,
        out_dir=args.out_dir,
        sample_rate=args.render_sr,
        flat_velocity=args.flat_velocity,
        backend=args.backend,
    )

    ntot_real_gt = None
    ntot_real_pred = None
    ntot_gt_pred = None
    if args.ntot_plot:
        stem = Path(render_res.pred_wav).name
        if ".flat" in stem:
            stem = stem.split(".flat", 1)[0]
        out_dir = Path(render_res.pred_wav).parent
        ntot_real_gt = out_dir / f"{stem}.real_vs_gt.ntot.png"
        ntot_real_pred = out_dir / f"{stem}.real_vs_pred.ntot.png"
        ntot_gt_pred = out_dir / f"{stem}.gt_vs_pred.ntot.png"

    adv_eval = evaluate_real_reference(
        midi_path=midi_path,
        real_wav=real_audio,
        gt_syn_wav=render_res.gt_wav,
        pred_syn_wav=render_res.pred_wav,
        sample_rate=args.eval_sr,
        frames_per_second=args.fps,
        fft_size=args.fft,
        bssl_mode=args.bssl_mode,
        num_samples=args.num_samples,
        normalization=args.norm,
        device=args.device,
        ntot_plot_real_gt_path=ntot_real_gt,
        ntot_plot_real_pred_path=ntot_real_pred,
        ntot_plot_gt_pred_path=ntot_gt_pred,
        run_note_dynamics=not args.skip_note_dynamics,
        note_fft_size=args.note_fft,
        note_frames_per_second=args.note_fps,
        note_harmonics=args.note_harmonics,
        pearson_only=args.pearson_only,
    )

    out = {
        "render": {
            "gt_wav": str(render_res.gt_wav),
            "pred_wav": str(render_res.pred_wav),
            "backend": render_res.backend,
            "extra": render_res.extra,
        },
        "adv_eval": adv_eval,
    }

    save_and_print_json(
        out,
        json_out=args.json_out,
        mute_output=args.mute_output,
    )


if __name__ == "__main__":
    main()
