#!/usr/bin/env python3
"""Rebuild selected FL demo candidates from original HDF5 audio and guitar weights.

Run this on the research machine with the 5 s guitar checkpoints, SFZ, and
sfizz renderer. Each piece is inferred once; its 20 s excerpt is then clipped
with the same builder used for the other listening cases. No prediction is
recalibrated for the website.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", action="append", required=True, metavar="PIECE:START",
                        help="FL HDF5 stem and 20 s window start; repeat for each candidate")
    parser.add_argument("--h5-dir", type=Path, required=True)
    parser.add_argument("--veloest-ckpt", type=Path, required=True)
    parser.add_argument("--diffsynth-ckpt", type=Path, required=True)
    parser.add_argument("--sfproxy-ckpt", type=Path, required=True)
    parser.add_argument("--sfz", type=Path, required=True)
    parser.add_argument("--sfizz-render", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "demo/docs")
    args = parser.parse_args()

    selections: dict[str, list[int]] = {}
    for selection in args.selection:
        piece, delimiter, start = selection.rpartition(":")
        if not delimiter or not piece or not start.isdigit() or int(start) < 0:
            parser.error(f"Expected PIECE:START, got {selection!r}")
        selections.setdefault(piece, []).append(int(start))
    for piece, starts in selections.items():
        source = args.h5_dir / f"{piece}.h5"
        if not source.is_file():
            raise FileNotFoundError(source)
        folder = args.work_dir / piece
        pair = folder / "source"
        inference = folder / "inference"
        subprocess.run([sys.executable, str(ROOT / "demo/scripts/export_pair_from_h5.py"),
                        str(source), str(pair)], check=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/infer_compare.py"),
                        "--audio", str(pair.with_suffix(".wav")),
                        "--midi", str(pair.with_suffix(".mid")),
                        "--out", str(inference),
                        "--veloest-ckpt", str(args.veloest_ckpt),
                        "--diffsynth-ckpt", str(args.diffsynth_ckpt),
                        "--diffsfproxy-ckpt", str(args.sfproxy_ckpt),
                        "--device", "cuda", "--velocity-method", "onset_only"], check=True)
        for start in starts:
            case_id = f"fl-{piece.lower()}-{start}"
            out = args.docs_dir / "assets/cases" / case_id
            subprocess.run([sys.executable, str(ROOT / "demo/scripts/build_listening_case.py"),
                            "--reference-audio", str(pair.with_suffix(".wav")),
                            "--reference-midi", str(pair.with_suffix(".mid")),
                            "--diffsynth-midi", str(inference / "diffsynth.mid"),
                            "--sfproxy-midi", str(inference / "diffsfproxy.mid"),
                            "--sfz", str(args.sfz), "--sfizz-render", str(args.sfizz_render),
                            "--out", str(out), "--dataset", "FL", "--case-id", case_id,
                            "--title", f"François Leduc · guitar performance {piece}",
                            "--start", str(start)], check=True)
            print(f"Ready: {case_id}", flush=True)


if __name__ == "__main__":
    main()
