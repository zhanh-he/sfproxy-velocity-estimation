#!/usr/bin/env python3
"""List aligned, non-overlapping 20 s demo windows from full-piece MIDI files.

For piano, report note-velocity MAE against the reference score. This is a
screening aid, not a replacement for audio rendering and BSSL/BSTL scoring.
"""

from __future__ import annotations

import argparse
import statistics

import pretty_midi


def notes(path):
    midi = pretty_midi.PrettyMIDI(path)
    return sorted((n.start, n.end, n.pitch, n.velocity)
                  for instrument in midi.instruments for n in instrument.notes)


def window(items, start):
    return [n for n in items if start <= n[0] < start + 20]


def match(ref, pred):
    remaining = list(pred)
    velocities = []
    for source in ref:
        eligible = [item for item in remaining if item[2] == source[2]]
        if not eligible:
            return None
        item = min(eligible, key=lambda n: abs(n[0]-source[0]) + abs(n[1]-source[1]))
        if abs(item[0]-source[0]) > 0.01 or abs(item[1]-source[1]) > 0.01:
            return None
        velocities.append(item[3])
        remaining.remove(item)
    return velocities if not remaining else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--diffsynth', required=True)
    parser.add_argument('--sfproxy', required=True)
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--piano', action='store_true')
    args = parser.parse_args()
    ref, ds, sf = [notes(path) for path in (args.reference, args.diffsynth, args.sfproxy)]
    print('start,notes,aligned,flat_mae,diffsynth_mae,sfproxy_mae')
    for start in range(0, int(args.duration - 20) + 1, 20):
        actual = window(ref, start)
        a, b = match(actual, window(ds, start)), match(actual, window(sf, start))
        if args.piano and a is not None and b is not None and actual:
            target = [n[3] for n in actual]
            mae = [statistics.mean(abs(x-y) for x,y in zip(target, values))
                   for values in ([64]*len(target), a, b)]
            fields = [f'{value:.3f}' for value in mae]
        else:
            fields = ['', '', '']
        print(','.join(map(str, [start, len(actual), int(a is not None and b is not None), *fields])))


if __name__ == '__main__':
    main()
