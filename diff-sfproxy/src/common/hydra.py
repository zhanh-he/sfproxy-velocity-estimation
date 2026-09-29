"""Hydra helpers used by the sfproxy entrypoints."""

from __future__ import annotations

import math
from typing import Callable, List

import hydra
from lightning import Callback
from omegaconf import DictConfig, OmegaConf

from common.logging import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


DEFAULT_SAMPLER_WEIGHTS = {
    "boundary_v2": {"boundary": 1.0, "coverage": 0.0, "stress": 0.0},
    "coverage_v2": {"boundary": 0.0, "coverage": 1.0, "stress": 0.0},
    "stress_v2": {"boundary": 0.0, "coverage": 0.0, "stress": 1.0},
    "mixed_v2": {"boundary": 0.3, "coverage": 0.5, "stress": 0.2},
}


def _register_resolver(name: str, fn: Callable) -> None:
    if OmegaConf.has_resolver(name):
        return
    OmegaConf.register_new_resolver(name, fn)


def _resolve_seg_tag(seg_len_s: float) -> str:
    value = float(seg_len_s)
    if value.is_integer():
        return f"{int(value)}s"
    return f"{value:g}".replace(".", "p") + "s"


def _resolve_split_value(split: str, train_value, val_value):
    split_name = str(split).strip().lower()
    if split_name == "train":
        return train_value
    if split_name == "val":
        return val_value
    raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'val'.")


def _resolve_sampler_weight(sampler_preset: str, component: str):
    preset = str(sampler_preset).strip()
    name = str(component).strip()
    if preset in DEFAULT_SAMPLER_WEIGHTS:
        weights = DEFAULT_SAMPLER_WEIGHTS[preset]
        if name not in weights:
            raise KeyError(f"Component '{name}' not in preset '{preset}': available={list(weights)}")
        return weights[name]
    if preset in {"coverage_v1", "coverage_shared_legacy"}:
        return ""
    raise ValueError(f"Unsupported sampler_preset '{preset}'.")


def _format_mix_value(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace(".", "p")


def _resolve_sampler_mix_tag(boundary, coverage, stress) -> str:
    values = [boundary, coverage, stress]
    if all(value in (None, "") for value in values):
        return ""
    if any(value in (None, "") for value in values):
        raise ValueError("sampler_mix requires boundary, coverage, and stress.")

    return "_".join(
        [
            f"b{_format_mix_value(float(boundary))}",
            f"c{_format_mix_value(float(coverage))}",
            f"s{_format_mix_value(float(stress))}",
        ]
    )


def _resolve_sampler_tag(sampler_preset: str, sampler_mix_tag: str) -> str:
    preset = str(sampler_preset).strip()
    mix_tag = str(sampler_mix_tag).strip()
    if not mix_tag:
        return preset
    return f"{preset}_{mix_tag}"


def register_resolvers() -> None:
    _register_resolver("mul", lambda x, y: x * y)
    _register_resolver("start", lambda total_num_steps, start_ratio: int(start_ratio * total_num_steps))
    _register_resolver("warm", lambda total_num_steps, warmup_ratio: int(warmup_ratio * total_num_steps))
    _register_resolver(
        "total_num_steps",
        lambda num_epochs, dataset_size, batch_size: num_epochs * dataset_size // batch_size,
    )
    _register_resolver("div_int", lambda x, y: int(round(float(x) / float(y))))
    _register_resolver(
        "hop_from_sr_fps",
        lambda sample_rate, frame_rate: int(math.floor(float(sample_rate) / float(frame_rate) + 0.5)),
    )
    _register_resolver("seg_tag", _resolve_seg_tag)
    _register_resolver("split_value", _resolve_split_value)
    _register_resolver("sampler_weight", _resolve_sampler_weight)
    _register_resolver("sampler_mix_tag", _resolve_sampler_mix_tag)
    _register_resolver("sampler_tag", _resolve_sampler_tag)


def instantiate_callbacks(callbacks_cfg: DictConfig | None) -> List[Callback] | None:
    """Instantiate Lightning callbacks from a Hydra config."""
    if not callbacks_cfg:
        log.info("No callback configs found. Skipping...")
        return None
    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig")

    callbacks: List[Callback] = []
    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


register_resolvers()
