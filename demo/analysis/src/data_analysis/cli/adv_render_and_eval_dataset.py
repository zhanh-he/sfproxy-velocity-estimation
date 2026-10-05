from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from data_analysis.evaluation.adv_eval import evaluate_real_reference
from data_analysis.rendering.render_pair import build_render_file_paths, render_midi_two_versions
from ._dataset_utils import (
    scan_midis,
    build_item_out_dir,
    build_item_result_path,
    load_maestro_audio_map,
    normalize_dataset_type,
    resolve_real_audio,
    save_json,
    collect_ok_metric,
    json_text,
    mean_or_nan,
)


def _fmt_metric(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _append_four_block(
    lines: List[str],
    summary: Dict[str, object],
    title: str,
    prefixes: List[tuple[str, str]],
    *,
    pearson_only: bool,
) -> None:
    lines.append("")
    lines.append(title)
    lines.append("           " + "  ".join(f"{label:<20}" for label, _ in prefixes))
    for label, suffix in (("pearson", "pearson"), ("cosine ", "cosine_sim"), ("mae    ", "mae")):
        if pearson_only and suffix != "pearson":
            continue
        if suffix != "pearson" and not any(f"{prefix}_{suffix}" in summary for _, prefix in prefixes):
            continue
        values = "  ".join(f"{_fmt_metric(summary.get(f'{prefix}_{suffix}')):<20}" for _, prefix in prefixes)
        lines.append(f"  {label}: {values}")


def _format_summary(summary: Dict[str, object]) -> str:
    lines = [
        "Summary",
        f"  dataset: {summary.get('dataset_type', '?')}",
        f"  scanned/ok/fail/skip_render: {summary.get('num_scanned', '?')}/{summary.get('num_ok', '?')}/{summary.get('num_fail', '?')}/{summary.get('num_skip_render', '?')}",
        f"  dataset_dir: {summary.get('dataset_dir', '?')}",
        f"  instrument: {summary.get('instrument', '?')}",
    ]
    if summary.get("split") is not None:
        lines.append(f"  split: {summary['split']}")
    if summary.get("maps_pianos") is not None:
        lines.append(f"  maps_pianos: {summary['maps_pianos']}")

    has_note = "mean_real_vs_gt_harmonic_energy_pearson" in summary
    pearson_only = "mean_real_vs_gt_bssl_cosine_sim" not in summary

    for side in ("real_vs_gt", "real_vs_pred", "gt_vs_pred"):
        prefixes: List[tuple[str, str]] = [
            ("BSSL", f"mean_{side}_bssl"),
            ("NTOT", f"mean_{side}_ntot"),
        ]
        if has_note:
            prefixes.extend([
                ("Note Harmonic Energy", f"mean_{side}_harmonic_energy"),
                ("Note Onset Flux", f"mean_{side}_onset_flux"),
            ])
        _append_four_block(lines, summary, side, prefixes, pearson_only=pearson_only)

    lines.append("")
    lines.append(f"per_file_results_dir: {summary.get('per_file_results_dir', '?')}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Advanced dataset evaluation with real audio as reference: real-vs-gt_syn and real-vs-pred_syn."
    )
    p.add_argument(
        "--dataset_type",
        required=True,
        choices=["smd", "maestro", "maps", "francoisleduc", "gaps"],
    )
    p.add_argument("--dataset_dir", required=True)
    p.add_argument(
        "--split",
        type=str,
        default="test",
        help="MAESTRO/FrancoisLeduc/GAPS only: train/valid/validation/test/all (default: test)",
    )
    p.add_argument("--maps_pianos", type=str, default="both", choices=["both", "ENSTDkCl", "ENSTDkAm"])
    p.add_argument("--instrument", required=True, help=".sfz or .sf2 instrument path")
    p.add_argument("--out_dir", default="./adv_renders_dataset")

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

    p.add_argument("--ntot_plot", action="store_true")
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--skip_existing", action="store_true", help="Skip rendering when rendered WAVs already exist; evaluation still runs")
    p.add_argument("--fail_fast", action="store_true")
    p.add_argument("--mute_output", action="store_true")
    p.add_argument("--print_full_results", action="store_true", help="Print full payload including per-item results")
    p.add_argument("--json_out", type=str, default=None)

    args = p.parse_args()

    dataset_type = normalize_dataset_type(args.dataset_type)
    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_file_dir = out_dir / "per_file_results"
    per_file_dir.mkdir(parents=True, exist_ok=True)

    midi_files = scan_midis(
        dataset_type,
        dataset_dir,
        split=args.split,
        maps_pianos=args.maps_pianos,
    )
    if not midi_files:
        raise RuntimeError(
            f"No MIDI files found for dataset_type={dataset_type}, split={args.split}, "
            f"maps_pianos={args.maps_pianos}, dataset_dir={dataset_dir}"
        )
    if args.max_files is not None:
        midi_files = midi_files[: max(0, int(args.max_files))]

    maestro_audio_map = load_maestro_audio_map(dataset_type, dataset_dir, split=args.split)

    results: List[Dict[str, object]] = []
    num_ok = 0
    num_fail = 0
    num_skip = 0

    loop = tqdm(midi_files, desc=dataset_type.upper(), unit="track")
    for midi_path in loop:
        midi_path = midi_path.resolve()
        item_out_dir = build_item_out_dir(out_dir, dataset_dir, midi_path)
        item_out_dir.mkdir(parents=True, exist_ok=True)
        item_json_path = build_item_result_path(per_file_dir, dataset_dir, midi_path)
        item_json_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            real_audio = resolve_real_audio(dataset_type, dataset_dir, midi_path, maestro_audio_map)
            gt_wav, pred_wav, _, _ = build_render_file_paths(
                midi_path=midi_path,
                instrument_path=args.instrument,
                out_dir=item_out_dir,
                flat_velocity=int(args.flat_velocity),
            )
            render_backend = args.backend
            render_extra: Dict[str, object] = {}

            if args.skip_existing and gt_wav.exists() and pred_wav.exists():
                num_skip += 1
                render_extra["render_skipped"] = True
            else:
                render_res = render_midi_two_versions(
                    midi_path,
                    args.instrument,
                    out_dir=item_out_dir,
                    sample_rate=args.render_sr,
                    flat_velocity=args.flat_velocity,
                    backend=args.backend,
                )
                gt_wav = render_res.gt_wav
                pred_wav = render_res.pred_wav
                render_backend = render_res.backend
                render_extra = render_res.extra

            ntot_real_gt = None
            ntot_real_pred = None
            ntot_gt_pred = None
            if args.ntot_plot:
                plot_prefix = gt_wav.name
                if plot_prefix.endswith(".gt.wav"):
                    plot_prefix = plot_prefix[: -len(".gt.wav")]
                ntot_real_gt = gt_wav.parent / f"{plot_prefix}.real_vs_gt.ntot.png"
                ntot_real_pred = gt_wav.parent / f"{plot_prefix}.real_vs_pred.ntot.png"
                ntot_gt_pred = gt_wav.parent / f"{plot_prefix}.gt_vs_pred.ntot.png"

            adv_eval = evaluate_real_reference(
                midi_path=midi_path,
                real_wav=real_audio,
                gt_syn_wav=gt_wav,
                pred_syn_wav=pred_wav,
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

            item = {
                "status": "ok",
                "midi_path": str(midi_path),
                "real_audio": str(real_audio),
                "render": {
                    "gt_wav": str(gt_wav),
                    "pred_wav": str(pred_wav),
                    "backend": render_backend,
                    "extra": render_extra,
                },
                "adv_eval": adv_eval,
            }
            save_json(item_json_path, item)
            results.append(item)
            num_ok += 1
            loop.set_postfix(ok=num_ok, fail=num_fail, skip=num_skip)

        except Exception as e:
            item = {
                "status": "error",
                "midi_path": str(midi_path),
                "error": repr(e),
            }
            save_json(item_json_path, item)
            results.append(item)
            num_fail += 1
            loop.set_postfix(ok=num_ok, fail=num_fail, skip=num_skip)
            if args.fail_fast:
                raise

    ok_items = [r for r in results if isinstance(r, dict) and r.get("status") == "ok" and "adv_eval" in r]
    metric_paths: Dict[str, tuple[str, str, str]] = {}
    for side in ("real_vs_gt", "real_vs_pred", "gt_vs_pred"):
        for group in ("bssl", "ntot"):
            metric_paths[f"mean_{side}_{group}_pearson"] = ("adv_eval", "summary", f"{side}_{group}_pearson")
            if not args.pearson_only:
                metric_paths[f"mean_{side}_{group}_cosine_sim"] = ("adv_eval", "summary", f"{side}_{group}_cosine_sim")
                metric_paths[f"mean_{side}_{group}_mae"] = ("adv_eval", "summary", f"{side}_{group}_mae")
    if not args.skip_note_dynamics:
        for side in ("real_vs_gt", "real_vs_pred", "gt_vs_pred"):
            for group in ("harmonic_energy", "onset_flux"):
                metric_paths[f"mean_{side}_{group}_pearson"] = ("adv_eval", "summary", f"{side}_{group}_pearson")
                if not args.pearson_only:
                    metric_paths[f"mean_{side}_{group}_cosine_sim"] = ("adv_eval", "summary", f"{side}_{group}_cosine_sim")
                    metric_paths[f"mean_{side}_{group}_mae"] = ("adv_eval", "summary", f"{side}_{group}_mae")
    metric_means = {
        name: mean_or_nan(collect_ok_metric(ok_items, path))
        for name, path in metric_paths.items()
    }

    summary = {
        "dataset_type": dataset_type,
        "split": args.split if dataset_type in {"maestro", "francoisleduc", "gaps"} else None,
        "maps_pianos": args.maps_pianos if dataset_type == "maps" else None,
        "dataset_dir": str(dataset_dir),
        "instrument": str(Path(args.instrument).resolve()),
        "num_scanned": int(len(midi_files)),
        "num_ok": int(len(ok_items)),
        "num_ok_run": int(num_ok),
        "num_fail": int(num_fail),
        "num_skip_render": int(num_skip),
        **metric_means,
        "per_file_results_dir": str(per_file_dir),
    }

    out = {"summary": summary, "results": results}
    json_out = Path(args.json_out) if args.json_out else (out_dir / f"{dataset_type}_adv_batch_eval.json")
    save_json(json_out, out)
    if not args.mute_output:
        if args.print_full_results:
            print(json_text(out))
        else:
            print(_format_summary(summary))


if __name__ == "__main__":
    main()
