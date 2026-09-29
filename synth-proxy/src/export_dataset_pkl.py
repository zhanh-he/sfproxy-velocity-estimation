# pylint: disable=W1203
"""Export rendered synthetic data to .pkl tensors.

This script generates synthetic note-event segments, renders them with a black-box
instrument renderer, and extracts note-wise dynamics features.

Outputs are stored as pickled torch tensors.

The export can be interrupted (SIGINT/SIGTERM/SIGTSTP). A resume_state.pkl will be written
in the hydra output directory and the export can be resumed by rerunning the same command.

Run example:
  python src/export_dataset_pkl.py --config-name data_piano dataset_size=20000

"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

# Allow running this file directly with `python src/export_dataset_pkl.py`
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import common.hydra  # noqa: F401

import signal

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.datasets_online import InstrumentSpec, RenderedInstrumentDataset
from data.note_samplers import make_sampler
from features.dynamics import DynamicsFeatureConfig, extract_note_features_padded

log = logging.getLogger(__name__)

RESUME_STATE_FILE = "resume_state.pkl"

is_interrupted = False


def graceful_shutdown(signum, frame) -> None:  # pylint: disable=unused-argument
    global is_interrupted  # pylint: disable=global-statement
    log.warning(f"Received signal {signum}. Export will stop after current iteration...")
    is_interrupted = True


signal.signal(signal.SIGTSTP, graceful_shutdown)
signal.signal(signal.SIGUSR1, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)


def _reset_output_dir(dir_path: Path) -> None:
    if not dir_path.exists():
        return
    for child in dir_path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _apply_velocity_stats_flag(sampler_cfg, use_velocity_stats: bool):
    if not isinstance(sampler_cfg, dict):
        return sampler_cfg
    sampler_type = str(sampler_cfg.get("type", "") or "").strip().lower()
    if sampler_type == "realism":
        sampler_cfg["use_velocity_stats"] = bool(use_velocity_stats)
        return sampler_cfg
    if sampler_type == "mixed":
        components = sampler_cfg.get("components", {})
        if isinstance(components, dict):
            for key, child in components.items():
                if isinstance(child, dict):
                    components[key] = _apply_velocity_stats_flag(dict(child), use_velocity_stats)
        sampler_cfg["components"] = components
    return sampler_cfg


def _resolve_sampler_dict(cfg: DictConfig) -> dict:
    preset = str(cfg.sampler_preset)
    if preset in cfg.sampler_options:
        sampler_dict = OmegaConf.to_container(cfg.sampler_options[preset], resolve=True)
        return _apply_velocity_stats_flag(dict(sampler_dict), bool(cfg.sampler_velo_present))

    if preset not in cfg.sampler_weights_v2:
        raise KeyError(f"Unknown sampler_preset: {preset}")

    components = OmegaConf.to_container(cfg.sampler_components_v2, resolve=True)
    weights = OmegaConf.to_container(cfg.sampler_weights_v2[preset], resolve=True)
    sampler_dict = {"type": "mixed", "components": {}}
    for name, component_cfg in dict(components).items():
        child_cfg = dict(component_cfg)
        child_cfg["weight"] = float(dict(weights).get(str(name), 0.0) or 0.0)
        sampler_dict["components"][str(name)] = child_cfg

    return _apply_velocity_stats_flag(dict(sampler_dict), bool(cfg.sampler_velo_present))


@hydra.main(version_base=None, config_path="../configs", config_name="data")
def export_dataset_pkl(cfg: DictConfig) -> None:
    global is_interrupted  # pylint: disable=global-statement

    output_dir = Path.cwd()
    if bool(cfg.get("reset_output_dir", False)):
        _reset_output_dir(output_dir)

    start_index = int(cfg.start_index)
    end_index = int(cfg.end_index)
    offset_index = 0

    # Resolve configs to python dicts
    instrument_dict = OmegaConf.to_container(cfg.instrument, resolve=True)
    sampler_dict = _resolve_sampler_dict(cfg)
    features_dict = OmegaConf.to_container(cfg.feature, resolve=True)

    instrument = InstrumentSpec(**instrument_dict)
    sampler = make_sampler(sampler_dict)
    feat_cfg = DynamicsFeatureConfig(**features_dict)

    sr = int(instrument_dict.get("sr", cfg.get("sr", 32000)))
    seg_len_s = float(instrument_dict.get("seg_len_s", cfg.get("seg_len_s", 2.0)))
    nmax = int(instrument_dict.get("nmax", cfg.get("nmax", 64)))

    if abs(float(sampler_dict.get("seg_len_s", seg_len_s)) - seg_len_s) > 1e-6:
        log.warning("sampler.seg_len_s differs from instrument.seg_len_s. Using instrument.seg_len_s")

    # Resume state
    resume_path = output_dir / RESUME_STATE_FILE

    if resume_path.exists():
        saved = torch.load(str(resume_path), map_location="cpu")
        resume_index = int(saved["resume_index"])
        inputs_pitch = saved["inputs_pitch"]
        inputs_cont = saved["inputs_cont"]
        inputs_mask = saved["inputs_mask"]
        targets_note = saved["targets_note"]

        offset_index = resume_index - start_index
        start_index = resume_index

        log.info(f"Found {RESUME_STATE_FILE}. Resuming from sample {start_index}...")

    else:
        out_audio_dir = None
        if int(cfg.export_audio) != 0:
            out_audio_dir = output_dir / "audio"
            out_audio_dir.mkdir(parents=True, exist_ok=True)

        exported_size = end_index - start_index
        configs_dict = {
            "instrument": instrument.name,
            "instrument_type": instrument.render_backend,
            "instrument_path": instrument.instrument_path,
            "bank": instrument.bank,
            "program": instrument.program,
            "polyphony": instrument.polyphony,
            "gain_db": instrument.gain_db,
            # Length of the stored tensors in this folder
            "dataset_size": int(exported_size),
            # Virtual dataset size used for deterministic sampling
            "virtual_dataset_size": int(cfg.dataset_size),
            "start_index": int(cfg.start_index),
            "end_index": int(cfg.end_index),
            "seed_offset": int(cfg.seed_offset),
            "sr": sr,
            "seg_len_s": seg_len_s,
            "nmax": nmax,
            "sampler_preset": cfg.get("sampler_preset"),
            "sampler": sampler_dict,
            "features": features_dict,
            "d_note": 2 + (1 if feat_cfg.include_f0_ratio else 0),
            "d_seg": None,
        }

        log.info(
            "Export %s split=%s preset=%s seg=%.3fs size=%d",
            instrument.name,
            str(cfg.split),
            str(cfg.get("sampler_preset")),
            seg_len_s,
            int(exported_size),
        )

        with open(output_dir / "configs.pkl", "wb") as f:
            torch.save(configs_dict, f)

        # Allocate output tensors
        num = exported_size
        inputs_pitch = torch.zeros((num, nmax), dtype=torch.long)
        inputs_cont = torch.zeros((num, nmax, 3), dtype=torch.float32)
        inputs_mask = torch.zeros((num, nmax), dtype=torch.bool)
        targets_note = torch.zeros((num, nmax, configs_dict["d_note"]), dtype=torch.float32)

    # Dataset
    dataset = RenderedInstrumentDataset(
        instrument=instrument,
        sampler=sampler,
        dataset_size=int(cfg.dataset_size),
        seed_offset=int(cfg.seed_offset),
        sr=sr,
        seg_len_s=seg_len_s,
        nmax=nmax,
        rms_range=(float(cfg.rms_min), float(cfg.rms_max)),
        peak_abs_max=float(cfg.peak_abs_max),
        max_tries=int(cfg.max_tries),
    )

    # Subset
    if start_index != 0 or end_index != int(cfg.dataset_size):
        dataset = Subset(dataset, range(start_index, end_index))
        log.info(f"Exporting samples {start_index} to {end_index - 1}")
    else:
        log.info(f"Exporting full dataset: {cfg.dataset_size} samples")

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        shuffle=False,
        drop_last=False,
    )

    total_batches = max(1, (end_index - start_index + int(cfg.batch_size) - 1) // int(cfg.batch_size))
    pbar = tqdm(loader, total=total_batches, dynamic_ncols=True)

    out_audio_dir = output_dir / "audio" if int(cfg.export_audio) != 0 else None

    for i, (pitch_b, cont_b, mask_b, audio_b, idx_b) in enumerate(pbar):
        slice_start = i * int(cfg.batch_size) + offset_index
        slice_end = slice_start + pitch_b.shape[0]

        # Normalize inputs
        cont_norm = cont_b.clone()
        cont_norm[..., 0] = cont_norm[..., 0] / float(seg_len_s)
        cont_norm[..., 1] = cont_norm[..., 1] / float(seg_len_s)
        cont_norm[..., 0:2] = cont_norm[..., 0:2].clamp(0.0, 1.0)

        # Extract targets per sample
        feats_list = []
        for j in range(pitch_b.shape[0]):
            audio_1d = audio_b[j, 0].to(torch.float32)
            feats, _ = extract_note_features_padded(
                audio=audio_1d,
                pitch=pitch_b[j],
                cont=cont_b[j],
                mask=mask_b[j],
                sr=sr,
                seg_len_s=seg_len_s,
                cfg=feat_cfg,
            )
            feats_list.append(feats)
        targets_b = torch.stack(feats_list, dim=0).to(torch.float32)

        inputs_pitch[slice_start:slice_end] = pitch_b.to(torch.long)
        inputs_cont[slice_start:slice_end] = cont_norm.to(torch.float32)
        inputs_mask[slice_start:slice_end] = mask_b.to(torch.bool)
        targets_note[slice_start:slice_end] = targets_b.cpu()

        # Optional audio export
        if out_audio_dir is not None:
            export_n = int(cfg.export_audio)
            if export_n == -1:
                export_n = 10**9
            for j in range(pitch_b.shape[0]):
                global_idx = start_index + slice_start + j
                if global_idx < export_n:
                    try:
                        from scipy.io import wavfile

                        wavfile.write(
                            str(out_audio_dir / f"{global_idx}.wav"),
                            sr,
                            audio_b[j, 0].cpu().numpy(),
                        )
                    except Exception:
                        pass

        if is_interrupted:
            new_resume_index = start_index + slice_end
            log.warning(
                f"Interrupt requested. Saving {RESUME_STATE_FILE} at resume_index={new_resume_index}"
            )
            with open(resume_path, "wb") as f:
                torch.save(
                    {
                        "resume_index": new_resume_index,
                        "inputs_pitch": inputs_pitch,
                        "inputs_cont": inputs_cont,
                        "inputs_mask": inputs_mask,
                        "targets_note": targets_note,
                    },
                    f,
                )
            sys.exit(0)

    # Clean resume file if finished
    if resume_path.exists():
        resume_path.unlink()

    # Save tensors
    # Always save canonical filenames so the folder can be directly used for training.
    torch.save(inputs_pitch, Path.cwd() / "inputs_pitch.pkl")
    torch.save(inputs_cont, Path.cwd() / "inputs_cont.pkl")
    torch.save(inputs_mask, Path.cwd() / "inputs_mask.pkl")
    torch.save(targets_note, Path.cwd() / "targets_note.pkl")

    log.info("Export completed successfully")


if __name__ == "__main__":
    export_dataset_pkl()  # pylint: disable=no-value-for-parameter
