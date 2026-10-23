from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from omegaconf import OmegaConf
import torch

REQUIRED_FILES = [
    "inputs_pitch.pkl",
    "inputs_cont.pkl",
    "inputs_mask.pkl",
    "targets_note.pkl",
]
OPTIONAL_FILES = ["targets_seg.pkl"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose multiple exported SFProxy datasets into one dataset.")
    parser.add_argument("--input-dirs", nargs="+", required=True, help="Input dataset directories.")
    parser.add_argument("--weights", nargs="+", type=float, help="Mixture weights for each input dir.")
    parser.add_argument("--names", nargs="*", default=None, help="Optional component names.")
    parser.add_argument("--out-dir", required=True, help="Output dataset directory.")
    parser.add_argument("--total-size", type=int, default=None, help="Number of samples in the composed dataset.")
    parser.add_argument("--config-file", type=str, default="", help="Optional YAML config file that defines sampler_weights_v2.")
    parser.add_argument("--preset", type=str, default="", help="Preset name inside sampler_weights_v2 when using --config-file.")
    parser.add_argument("--seed", type=int, default=86, help="Random seed for deterministic composition.")
    parser.add_argument("--preset-name", type=str, default="mixed_v2", help="Logical preset name stored in configs.pkl.")
    parser.add_argument(
        "--allow-replacement",
        action="store_true",
        help="Allow sampling with replacement when a component dataset is smaller than its requested count.",
    )
    return parser.parse_args()


def _load_recipe(config_file: str, preset: str) -> tuple[list[str], list[float]]:
    if not config_file or not preset:
        raise ValueError("Both --config-file and --preset are required when loading weights from config")

    cfg = OmegaConf.load(config_file)
    mapping = OmegaConf.to_container(cfg.sampler_weights_v2[preset], resolve=True)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Could not load sampler_weights_v2.{preset} from {config_file}")

    names = [str(name) for name in mapping.keys()]
    weights = [float(mapping[name]) for name in mapping.keys()]
    return names, weights


def _allocate_counts(total_size: int, weights: Sequence[float]) -> list[int]:
    if total_size <= 0:
        raise ValueError("total_size must be positive")
    if not weights:
        raise ValueError("weights must be non-empty")
    weights_t = torch.tensor(weights, dtype=torch.float64)
    if torch.any(weights_t < 0):
        raise ValueError("weights must be non-negative")
    if float(weights_t.sum()) <= 0:
        raise ValueError("weights must sum to a positive value")
    probs = weights_t / weights_t.sum()
    raw = probs * int(total_size)
    counts = torch.floor(raw).to(torch.long)
    remainder = int(total_size - int(counts.sum().item()))
    if remainder > 0:
        frac = raw - counts.to(raw.dtype)
        order = torch.argsort(frac, descending=True)
        for idx in order[:remainder].tolist():
            counts[idx] += 1
    return [int(v) for v in counts.tolist()]


def _check_dir(dataset_dir: Path) -> None:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    for name in ["configs.pkl", *REQUIRED_FILES]:
        path = dataset_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")


def _load_configs(dataset_dir: Path) -> dict:
    with open(dataset_dir / "configs.pkl", "rb") as f:
        cfg = torch.load(f, map_location="cpu")
    if not isinstance(cfg, dict):
        raise ValueError(f"configs.pkl must contain a dict: {dataset_dir}")
    return cfg


def _check_compatibility(configs: Sequence[dict], dirs: Sequence[Path]) -> None:
    keys = ["instrument", "sr", "seg_len_s", "nmax", "d_note", "d_seg"]
    base = configs[0]
    for cfg, dataset_dir in zip(configs[1:], dirs[1:]):
        for key in keys:
            if cfg.get(key) != base.get(key):
                raise ValueError(
                    f"Incompatible field '{key}' between {dirs[0]} ({base.get(key)}) and {dataset_dir} ({cfg.get(key)})"
                )


def _sample_indices(size: int, count: int, rng: torch.Generator, allow_replacement: bool) -> torch.Tensor:
    if count <= 0:
        return torch.empty((0,), dtype=torch.long)
    if count <= size:
        return torch.randperm(size, generator=rng)[:count]
    if not allow_replacement:
        raise ValueError(
            f"Requested {count} samples from dataset of size {size}. Use --allow-replacement to permit reuse."
        )
    return torch.randint(low=0, high=size, size=(count,), generator=rng, dtype=torch.long)


def _load_tensor(dataset_dir: Path, filename: str) -> torch.Tensor | None:
    path = dataset_dir / filename
    if not path.is_file():
        return None
    return torch.load(str(path), map_location="cpu")


def _concat_selected(dataset_dirs: Sequence[Path], counts: Sequence[int], rng: torch.Generator, allow_replacement: bool) -> dict[str, torch.Tensor | None]:
    buckets: dict[str, list[torch.Tensor]] = {name: [] for name in REQUIRED_FILES + OPTIONAL_FILES}
    component_meta: list[dict] = []
    for dataset_dir, count in zip(dataset_dirs, counts):
        cfg = _load_configs(dataset_dir)
        size = int(cfg["dataset_size"])
        indices = _sample_indices(size=size, count=int(count), rng=rng, allow_replacement=allow_replacement)
        component_meta.append({
            "path": str(dataset_dir),
            "requested_count": int(count),
            "source_size": int(size),
        })
        if count <= 0:
            continue
        for filename in REQUIRED_FILES + OPTIONAL_FILES:
            tensor = _load_tensor(dataset_dir, filename)
            if tensor is None:
                continue
            buckets[filename].append(tensor.index_select(0, indices))

    merged: dict[str, torch.Tensor | None] = {}
    for filename, parts in buckets.items():
        merged[filename] = torch.cat(parts, dim=0) if parts else None

    total = sum(int(c) for c in counts)
    perm = torch.randperm(total, generator=rng) if total > 0 else torch.empty((0,), dtype=torch.long)
    for filename, tensor in list(merged.items()):
        if tensor is not None:
            merged[filename] = tensor.index_select(0, perm)

    merged["__component_meta__"] = component_meta  # type: ignore[index]
    return merged


def main() -> None:
    args = _parse_args()
    dataset_dirs = [Path(p).resolve() for p in args.input_dirs]
    names = list(args.names or [])

    if args.weights is None:
        if not args.config_file:
            raise ValueError("Provide --weights or use --config-file with --preset")
        recipe_names, recipe_weights = _load_recipe(args.config_file, args.preset)
        if names and len(names) != len(recipe_names):
            raise ValueError("--names must match the configured recipe length")
        names = names or recipe_names
        weights = recipe_weights
    else:
        weights = [float(w) for w in args.weights]

    if len(dataset_dirs) != len(weights):
        raise ValueError("--input-dirs and --weights must have the same length")
    if names and len(names) != len(dataset_dirs):
        raise ValueError("--names must be omitted or have the same length as --input-dirs")
    if not names:
        names = [p.parent.name for p in dataset_dirs]

    for dataset_dir in dataset_dirs:
        _check_dir(dataset_dir)

    configs = [_load_configs(dataset_dir) for dataset_dir in dataset_dirs]
    _check_compatibility(configs, dataset_dirs)

    total_size = int(args.total_size) if args.total_size is not None else int(configs[0]["dataset_size"])
    counts = _allocate_counts(total_size, weights)
    rng = torch.Generator().manual_seed(int(args.seed))
    merged = _concat_selected(dataset_dirs, counts, rng, bool(args.allow_replacement))

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = dict(configs[0])
    composed_sampler = {
        "type": "composed",
        "preset": str(args.preset_name),
        "components": [
            {
                "name": str(name),
                "weight": float(weight),
                "count": int(count),
                "source_dir": str(dataset_dir),
            }
            for name, weight, count, dataset_dir in zip(names, weights, counts, dataset_dirs)
        ],
    }
    base_cfg.update(
        {
            "dataset_size": int(total_size),
            "start_index": 0,
            "end_index": int(total_size),
            "sampler_preset": str(args.preset_name),
            "sampler": composed_sampler,
            "compose_seed": int(args.seed),
        }
    )
    with open(out_dir / "configs.pkl", "wb") as f:
        torch.save(base_cfg, f)

    for filename in REQUIRED_FILES + OPTIONAL_FILES:
        tensor = merged.get(filename)
        if tensor is None:
            continue
        torch.save(tensor, out_dir / filename)

    summary_lines = [
        f"Wrote composed dataset to {out_dir}",
        f"preset={args.preset_name}",
        f"total_size={total_size}",
    ]
    for name, count, weight, dataset_dir in zip(names, counts, weights, dataset_dirs):
        summary_lines.append(f"  - {name}: count={count}, weight={weight:.4f}, dir={dataset_dir}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
