#!/usr/bin/env python3
"""Prepare GuitarSet-style DDSP-Guitar-Synth datasets for unified runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

def split_tracks(loader, test_player_id: str, mode: str) -> Dict[str, List[str]]:
    splits = loader.split_dataset_by_player_id(test_player_id)
    if mode in {'solo', 'comp'}:
        splits = loader.filter_splits_by_version(splits, version=mode)
    return splits


def pick_split_key(splits: dict, prefix: str) -> str:
    for key in splits:
        if key.startswith(prefix):
            return key
    raise KeyError(f"Could not find split starting with '{prefix}'. Available: {list(splits.keys())}")


def save_npz(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare GuitarSet DDSP-Guitar-Synth datasets under the unified repo contract.')
    parser.add_argument('--guitarset_path', type=str, required=True, help='Path passed to mirdata for GuitarSet.')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--sample_rate', type=int, default=22050)
    parser.add_argument('--frame_rate', type=float, default=100.0)
    parser.add_argument('--segment_seconds', type=float, default=10.0)
    parser.add_argument('--test_player_id', type=str, default='00')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'solo', 'comp'])
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--use_crepe_f0_labels', action='store_true')
    args = parser.parse_args()

    from data.guitarset_loader import GuitarSetLoader

    loader = GuitarSetLoader(
        data_home=args.guitarset_path,
        download=args.download,
        use_crepe_f0_labels=args.use_crepe_f0_labels,
    )
    mode = None if args.mode == 'all' else args.mode
    splits = split_tracks(loader, args.test_player_id, mode if mode is not None else 'all')
    train_key = pick_split_key(splits, 'train_player-')
    val_key = pick_split_key(splits, 'test_player-')

    train_ids = splits[train_key]
    val_ids = splits[val_key]

    train_data = loader.create_midi_conditioning_dataset(
        train_ids,
        sr=args.sample_rate,
        frame_rate=args.frame_rate,
        item_dur=args.segment_seconds,
    )
    val_data = loader.create_midi_conditioning_dataset(
        val_ids,
        sr=args.sample_rate,
        frame_rate=args.frame_rate,
        item_dur=args.segment_seconds,
    )

    mode_tag = '' if args.mode == 'all' else f'_{args.mode}'
    seg_tag = str(args.segment_seconds).replace('.0', '')
    train_path = args.output_dir / f'train_gset_midi{mode_tag}_{seg_tag}s.npz'
    val_path = args.output_dir / f'val_gset_midi{mode_tag}_{seg_tag}s.npz'
    save_npz(train_path, train_data)
    save_npz(val_path, val_data)

    meta = {
        'guitarset_path': args.guitarset_path,
        'sample_rate': int(args.sample_rate),
        'frame_rate': float(args.frame_rate),
        'segment_seconds': float(args.segment_seconds),
        'test_player_id': str(args.test_player_id),
        'mode': args.mode,
        'use_crepe_f0_labels': bool(args.use_crepe_f0_labels),
        'train_key': train_key,
        'val_key': val_key,
        'train_track_count': len(train_ids),
        'val_track_count': len(val_ids),
        'train_item_count': int(train_data['conditioning'].shape[0]),
        'val_item_count': int(val_data['conditioning'].shape[0]),
        'train_dataset_path': str(train_path),
        'val_dataset_path': str(val_path),
    }
    meta_path = args.output_dir / 'prepare_metadata.json'
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
