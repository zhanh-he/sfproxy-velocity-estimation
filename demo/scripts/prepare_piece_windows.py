#!/usr/bin/env python3
"""Build 20 s listening candidates from an existing full-piece inference pair.

The caller supplies the exact prediction MIDIs and soundfont; this script does
not infer or modify velocities. Use the scoring script before publication.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-audio', 'reference-midi', 'diffsynth-midi',
                 'sfproxy-midi', 'sfz', 'sfizz-render', 'out-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--case-prefix', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--start', action='append', required=True, type=int)
    parser.add_argument('--velocity-ground-truth', action='store_true')
    args = parser.parse_args()
    builder = ROOT / 'demo/scripts/build_listening_case.py'
    for start in args.start:
        case_id = f'{args.case_prefix}-{start}'
        command = [sys.executable, str(builder),
                   '--reference-audio', str(args.reference_audio),
                   '--reference-midi', str(args.reference_midi),
                   '--diffsynth-midi', str(args.diffsynth_midi),
                   '--sfproxy-midi', str(args.sfproxy_midi),
                   '--sfz', str(args.sfz),
                   '--sfizz-render', str(args.sfizz_render),
                   '--out', str(args.out_dir / case_id),
                   '--dataset', args.dataset, '--case-id', case_id,
                   '--title', args.title, '--start', str(start)]
        if args.velocity_ground_truth:
            command.append('--velocity-ground-truth')
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
