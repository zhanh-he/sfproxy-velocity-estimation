import argparse
import json
import os
import shutil

from ddsp_piano.utils.dataset import preprocess_split


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess the MAESTRO dataset into cached numpy segments.'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'validation', 'test'],
        choices=['train', 'validation', 'test'],
        help='Dataset splits to preprocess.'
    )
    parser.add_argument('--segment_duration', type=float, default=10.0,
                        help='Segment duration in seconds. Default is the repo unified contract.')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap ratio between consecutive segments.')
    parser.add_argument('--sample_rate', type=int, default=22050,
                        help='Audio sample rate used during preprocessing.')
    parser.add_argument('--frame_rate', type=int, default=100,
                        help='Conditioning frame rate used during preprocessing.')
    parser.add_argument('--max_polyphony', type=int, default=16,
                        help='Maximum number of simultaneous notes modeled.')
    parser.add_argument('maestro_path', type=str,
                        help='Path to the MAESTRO dataset directory.')
    parser.add_argument('cache_root', type=str,
                        help='Directory where the cached numpy files will be stored.')
    return parser.parse_args()


def _write_cache_contract(args) -> None:
    os.makedirs(args.cache_root, exist_ok=True)
    contract = {
        'dataset': 'MAESTRO-v3.0.0',
        'segment_duration': float(args.segment_duration),
        'sample_rate': int(args.sample_rate),
        'frame_rate': int(args.frame_rate),
        'overlap': float(args.overlap),
        'max_polyphony': int(args.max_polyphony),
        'splits': list(args.splits),
    }
    contract_path = os.path.join(args.cache_root, 'cache_config.json')
    with open(contract_path, 'w', encoding='utf-8') as f:
        json.dump(contract, f, indent=2)
    print(f'[*] Wrote cache contract to {contract_path}')


def main():
    args = parse_args()
    for split in args.splits:
        print(f'[*] Preprocessing split: {split}')
        preprocess_split(
            dataset_dir=args.maestro_path,
            cache_root=args.cache_root,
            split=split,
            max_polyphony=args.max_polyphony,
            segment_duration=args.segment_duration,
            overlap=args.overlap,
            sample_rate=args.sample_rate,
            frame_rate=args.frame_rate
        )

    metadata_src = os.path.join(args.maestro_path, 'maestro-v3.0.0.csv')
    if os.path.exists(metadata_src):
        os.makedirs(args.cache_root, exist_ok=True)
        metadata_dst = os.path.join(args.cache_root, 'maestro-v3.0.0.csv')
        shutil.copy2(metadata_src, metadata_dst)
        print(f'[*] Copied metadata to {metadata_dst}')
    else:
        print('[!] Warning: maestro-v3.0.0.csv not found; metadata was not copied.')

    _write_cache_contract(args)


if __name__ == '__main__':
    main()
