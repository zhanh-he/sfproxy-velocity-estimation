from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict


@dataclass(frozen=True)
class SfizzRenderCaps:
    binary: str
    supports_polyphony: bool
    supports_quality: bool
    supports_log: bool
    supports_track: bool
    supports_oversampling: bool


def find_sfizz_render_binary() -> str:
    """Return the sfizz offline renderer binary name/path.

    Depending on distro/version, it may be `sfizz_render` or `sfizz-render`.
    """
    # 1) Explicit override by environment variable.
    env_bin = os.environ.get("SFIZZ_RENDER_BIN")
    if env_bin:
        p = Path(env_bin).expanduser()
        if p.exists() and os.access(str(p), os.X_OK):
            return str(p)

    # 2) Look up from PATH.
    for name in ("sfizz_render", "sfizz-render"):
        p = shutil.which(name)
        if p:
            return p

    # 3) Fall back to common in-repo local build locations.
    # file: <repo>/data_analysis/rendering/sfizz.py -> repo root is parents[2]
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (
        repo_root / "sfizz/build/library/bin/sfizz_render",
        repo_root / "sfizz/build/library/bin/sfizz-render",
    ):
        if candidate.exists() and os.access(str(candidate), os.X_OK):
            return str(candidate)

    raise FileNotFoundError(
        "sfizz_render not found. Tried SFIZZ_RENDER_BIN, PATH "
        "(`sfizz_render`/`sfizz-render`), and local build path "
        f"{repo_root / 'sfizz/build/library/bin/sfizz_render'}."
    )


def _detect_caps(binary: str) -> SfizzRenderCaps:
    """Detect supported flags by parsing `--help` output."""
    try:
        proc = subprocess.run(
            [binary, "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        txt = proc.stdout or ""
    except Exception:
        txt = ""

    def has(token: str) -> bool:
        return token in txt

    return SfizzRenderCaps(
        binary=binary,
        supports_polyphony=has("--polyphony") or has("-p,"),
        supports_quality=has("--quality") or has("-q,"),
        supports_log=has("--log"),
        supports_track=has("--track") or has("-t,"),
        supports_oversampling=has("--oversampling"),
    )


def render_midi_with_sfz_sfizz(
    *,
    midi_path: str | Path,
    sfz_path: str | Path,
    wav_path: str | Path,
    sample_rate: int = 44100,
    block_size: int = 1024,
    polyphony: int = 256,
    quality: int = 3,
    track: Optional[int] = None,
    oversampling: Optional[int] = None,
    use_eot: bool = False,
    verbose: bool = False,
    extra_args: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Render a MIDI file with an SFZ instrument via `sfizz_render`.

    This is the correct path for your Salamander .sfz.

    Notes:
      - We set cwd to the SFZ directory so relative sample paths resolve.
      - We only pass flags which are supported by the installed binary.

    Returns:
      A dict with command, stdout summary (if any), and detected capabilities.
    """
    midi_path = Path(midi_path)
    sfz_path = Path(sfz_path)
    wav_path = Path(wav_path)

    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI not found: {midi_path}")
    if not sfz_path.exists():
        raise FileNotFoundError(f"SFZ not found: {sfz_path}")

    wav_path.parent.mkdir(parents=True, exist_ok=True)

    binary = find_sfizz_render_binary()
    caps = _detect_caps(binary)

    cmd = [
        binary,
        "--sfz",
        str(sfz_path),
        "--midi",
        str(midi_path),
        "--wav",
        str(wav_path),
        "--samplerate",
        str(int(sample_rate)),
        "--blocksize",
        str(int(block_size)),
    ]

    if caps.supports_polyphony:
        cmd += ["--polyphony", str(int(polyphony))]
    if caps.supports_quality:
        cmd += ["--quality", str(int(quality))]
    if caps.supports_track and track is not None:
        cmd += ["--track", str(int(track))]
    if caps.supports_oversampling and oversampling is not None:
        cmd += ["--oversampling", str(int(oversampling))]

    if use_eot:
        cmd += ["--use-eot"]
    if verbose:
        cmd += ["--verbose"]
    if extra_args:
        cmd += list(extra_args)

    # Important: SFZ often references samples with relative paths.
    cwd = str(sfz_path.parent)

    proc = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
    )

    return {
        "binary": binary,
        "caps": caps.__dict__,
        "cmd": cmd,
        "cwd": cwd,
        "stdout": (proc.stdout[-2000:] if proc.stdout else ""),
        "wav_path": str(wav_path),
    }
