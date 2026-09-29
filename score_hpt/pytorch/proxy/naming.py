from __future__ import annotations

from typing import Dict


_CANONICAL_TO_DISPLAY: Dict[str, str] = {
    "diffsynth_piano": "DiffSynth-Piano",
    "diffsynth_guitar": "DiffSynth-Guitar",
    "diffproxy": "DiffProxy",
}

_CANONICAL_TO_RUN_TAG: Dict[str, str] = {
    "diffsynth_piano": "diffsynth_piano",
    "diffsynth_guitar": "diffsynth_guitar",
    "diffproxy": "diffproxy",
}


def normalize_backend_type(raw_backend_type: str | None) -> str:
    """Normalize the configured backend id."""
    name = str(raw_backend_type or "diffsynth_piano").strip().lower()
    name = name.replace('-', '_').replace('/', '_')
    name = name or "diffsynth_piano"
    if name not in _CANONICAL_TO_DISPLAY:
        raise ValueError(
            f"Unknown backend type: {raw_backend_type!r}. "
            f"Supported: {', '.join(_CANONICAL_TO_DISPLAY)}"
        )
    return name


def backend_display_name(raw_backend_type: str | None) -> str:
    canonical = normalize_backend_type(raw_backend_type)
    return _CANONICAL_TO_DISPLAY.get(canonical, str(raw_backend_type or canonical))


def backend_run_tag(raw_backend_type: str | None) -> str:
    canonical = normalize_backend_type(raw_backend_type)
    return _CANONICAL_TO_RUN_TAG.get(canonical, canonical)


def is_diffproxy_backend(raw_backend_type: str | None) -> bool:
    return normalize_backend_type(raw_backend_type) == "diffproxy"


def is_diffsynth_backend(raw_backend_type: str | None) -> bool:
    return normalize_backend_type(raw_backend_type) in {"diffsynth_piano", "diffsynth_guitar"}
