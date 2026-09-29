from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch


def get_frontend_pretrained_value(model_cfg) -> str:
    return str(getattr(model_cfg, "frontend_pretrained", "") or "").strip()


def get_frontend_pretrained_mode(model_cfg) -> str:
    return str(getattr(model_cfg, "frontend_pretrained_mode", "scratch") or "scratch").strip()


def resolve_pretrained_checkpoint(raw_path: str, model_label: str, required: bool) -> Optional[Path]:
    text = str(raw_path or "").strip()
    return Path(text).expanduser().resolve() if text else None


def get_route2_piano_pretrained_subdir(model_cfg) -> str:
    model_type = str(getattr(model_cfg, "type", "") or "").strip().lower()
    input2 = str(getattr(model_cfg, "input2", "") or "").strip().lower()
    if model_type == "hpt" and input2 == "onset":
        return "hpt+onset+score_note_editor"
    return "hpt"


def find_latest_iteration_checkpoint(checkpoint_dir: Path, model_label: str) -> Path:
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    iteration_files = []
    for path in checkpoint_dir.glob("*_iterations.pth"):
        match = re.search(r"(\d+)_iterations\.pth$", path.name)
        if match:
            iteration_files.append((int(match.group(1)), path.resolve()))

    iteration_files.sort(key=lambda item: item[0])
    return iteration_files[-1][1]


def resolve_frontend_pretrained_checkpoint(model_cfg, model_label: str, required: bool) -> Optional[Path]:
    mode = get_frontend_pretrained_mode(model_cfg)
    if mode == "route2_piano_specific":
        raw_path = get_frontend_pretrained_value(model_cfg)
        return resolve_pretrained_checkpoint(raw_path, model_label=model_label, required=required)
    if mode == "route2_piano_auto":
        subdir = get_route2_piano_pretrained_subdir(model_cfg)
        return find_latest_iteration_checkpoint(Path("pretrained_checkpoints") / subdir, model_label=model_label)
    return None


def unwrap_checkpoint_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    state = checkpoint
    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]

    cleaned = {}
    for key, value in state.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        cleaned[new_key] = value
    return cleaned


def select_prefixed_substate(
    state_dict: Dict[str, torch.Tensor],
    model_keys: Iterable[str],
    prefixes: Iterable[str],
) -> Dict[str, torch.Tensor]:
    required_keys = list(model_keys)
    required_set = set(required_keys)

    for prefix in prefixes:
        if prefix:
            candidate = {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
        else:
            candidate = dict(state_dict)

        if required_set.issubset(candidate.keys()):
            return {key: candidate[key] for key in required_keys}

    return {key: state_dict[key] for key in required_keys if key in state_dict}


def load_score_wrapper_state(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = unwrap_checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
