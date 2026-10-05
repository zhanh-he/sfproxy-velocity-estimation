import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import h5py
import csv
import librosa
import logging
from tqdm import tqdm
from hydra import compose, initialize

from utilities import (
    TargetProcessor,
    create_folder,
    create_logging,
    float32_to_int16,
    get_filename,
    int16_to_float32,
    pad_truncate_sequence,
    read_metadata,
    read_midi,
    traverse_folder,
)


def _sr_tag(cfg):
    return f"sr{int(cfg.feature.sample_rate)}"


def _load_audio_mono(path, sample_rate: int) -> np.ndarray:
    """Load mono audio with librosa for consistent cross-machine behavior."""
    path = str(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        audio, _ = librosa.core.load(path, sr=sample_rate, mono=True)
    return np.asarray(audio, dtype=np.float32)


def _decode_hdf5_str(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.bytes_):
        return value.astype(str)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode()
    return value


def _build_aligned_note_events(cfg, note_events, start_time, note_shift):
    aligned = []
    seg_len = float(cfg.feature.segment_seconds)
    begin_note = int(cfg.feature.begin_note)
    end_note = begin_note + int(cfg.feature.classes_num) - 1
    min_dur = 1.0 / float(cfg.feature.frames_per_second)

    for event in note_events:
        midi_note = int(event["midi_note"]) + int(note_shift)
        if midi_note < begin_note or midi_note > end_note:
            continue

        onset_rel = float(event["onset_time"]) - float(start_time)
        offset_rel = float(event["offset_time"]) - float(start_time)
        if onset_rel < 0.0 or onset_rel >= seg_len:
            continue

        offset_rel = min(seg_len, max(onset_rel + min_dur, offset_rel))
        aligned.append({
            "midi_note": midi_note,
            "onset_time": onset_rel,
            "offset_time": offset_rel,
            "velocity": int(event["velocity"]),
        })

    aligned.sort(key=lambda item: (item["onset_time"], item["midi_note"]))
    return aligned


def _load_standard_midi_segment(cfg, random_state, hdf5_path, start_time, segment_samples, target_processor):
    data_dict = {}

    note_shift = random_state.randint(
        low=-cfg.feature.max_note_shift,
        high=cfg.feature.max_note_shift + 1,
    )

    with h5py.File(hdf5_path, "r") as hf:
        start_sample = int(start_time * cfg.feature.sample_rate)
        end_sample = start_sample + segment_samples

        if end_sample >= hf["waveform"].shape[0]:
            start_sample -= segment_samples
            end_sample -= segment_samples

        waveform = int16_to_float32(hf["waveform"][start_sample:end_sample])

        if cfg.feature.augmentor:
            waveform = cfg.feature.augmentor.augment(waveform)

        if note_shift != 0:
            waveform = librosa.effects.pitch_shift(
                waveform,
                cfg.feature.sample_rate,
                note_shift,
                bins_per_octave=12,
            )

        data_dict["waveform"] = waveform

        midi_events = [e.decode() for e in hf["midi_event"][:]]
        midi_events_time = hf["midi_event_time"][:]
        target_dict, note_events, _ = target_processor.process(
            start_time,
            midi_events_time,
            midi_events,
            extend_pedal=True,
            note_shift=note_shift,
        )

    data_dict.update(target_dict)
    data_dict["exframe_roll"] = target_dict["frame_roll"] * (1 - target_dict["onset_roll"])
    data_dict["aligned_note_events"] = _build_aligned_note_events(
        cfg=cfg,
        note_events=note_events,
        start_time=start_time,
        note_shift=note_shift,
    )
    data_dict["has_velocity_target"] = np.array(
        float(getattr(cfg.dataset, "has_velocity_target", True)),
        dtype=np.float32,
    )
    return data_dict


def _normalize_dataset_split(raw_split):
    split = str(raw_split).strip().lower()
    mapping = {
        "train": "train",
        "valid": "validation",
        "validate": "validation",
        "validation": "validation",
        "val": "validation",
        "test": "test",
    }
    if split not in mapping:
        raise ValueError(f"Unsupported dataset split '{raw_split}'.")
    return mapping[split]


def _normalize_gaps_split(raw_split):
    split = str(raw_split or "").strip().lower()
    if split in {"", "valid", "validate", "validation", "val"}:
        # The published GAPS split merge leaves unmatched rows blank.
        # We treat those rows as validation so train / validation / test remain usable.
        return "validation"
    if split in {"train", "test"}:
        return split
    raise ValueError(f"Unsupported GAPS split '{raw_split}'.")


def _normalize_aligned_dataset_name(flag):
    return str(flag).strip().lower()


def _resolve_hdf5_dir(workspace, sample_rate, dataset_name):
    sr_tag = f"sr{int(sample_rate)}"
    hdf5_root = os.path.join(workspace, "hdf5s")
    return os.path.join(hdf5_root, f"{dataset_name}_{sr_tag}")


def _read_francoisleduc_metadata(metadata_path):
    with open(metadata_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required_columns = {
        "split",
        "midi_filename",
        "audio_filename",
        "guitar_type",
        "slice_id",
        "artist",
        "name",
    }
    if not rows:
        raise ValueError(f"No rows found in metadata file: {metadata_path}")
    missing_columns = required_columns.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(
            f"FrancoisLeduc metadata is missing columns: {sorted(missing_columns)}"
        )
    return rows


def _read_gaps_metadata(metadata_path):
    with open(metadata_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required_columns = {
        "id",
        "split",
        "midi_path",
        "audio_path",
    }
    if not rows:
        raise ValueError(f"No rows found in metadata file: {metadata_path}")
    missing_columns = required_columns.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(f"GAPS metadata is missing columns: {sorted(missing_columns)}")
    return rows

def _fit_audio_for_hdf5(audio, source_path):
    # ffmpeg decode can yield a read-only view from np.frombuffer; make it writable
    # before any in-place normalization for HDF5 export.
    audio = np.array(audio, dtype=np.float32, copy=True)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        logging.warning(
            f"Rescaling audio with peak {peak:.5f} before HDF5 write: {source_path}"
        )
        audio *= 0.999 / peak
    return audio

class Maestro_Dataset(object):
    def __init__(self, cfg):
        """
        This class takes the meta of an audio segment as input and returns
        the waveform and targets of the audio segment. This class is used by 
        DataLoader.

        Args:
          cfg: OmegaConf configuration object.
        """
        self.cfg = cfg
        self.random_state = np.random.RandomState(cfg.exp.random_seed)
        self.hdf5s_dir = _resolve_hdf5_dir(cfg.exp.workspace, cfg.feature.sample_rate, "maestro")
        self.segment_samples = int(cfg.feature.sample_rate * cfg.feature.segment_seconds)
        # Used for processing MIDI events to target | GroundTruth
        self.target_processor = TargetProcessor(cfg.feature.segment_seconds, cfg)
    def __getitem__(self, meta):
        """
        Prepare input and target of a segment for training.
        Args:
          meta: dict, e.g. {
            'year': '2004', 
            'hdf5_name': 'MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_10_Track10_wav.h5, 
            'start_time': 65.0}
        Returns:
          data_dict: {
            'waveform': (samples_num,)
            'onset_roll': (frames_num, classes_num), 
            'offset_roll': (frames_num, classes_num), 
            'reg_onset_roll': (frames_num, classes_num), 
            'reg_offset_roll': (frames_num, classes_num), 
            'frame_roll': (frames_num, classes_num),
            ‘frame_exonset_roll':(frames_num, classes_num),
            'velocity_roll': (frames_num, classes_num), 
            'mask_roll':  (frames_num, classes_num), 
            'pedal_onset_roll': (frames_num,), 
            'pedal_offset_roll': (frames_num,), 
            'reg_pedal_onset_roll': (frames_num,), 
            'reg_pedal_offset_roll': (frames_num,), 
            'pedal_frame_roll': (frames_num,)}
        """
        [year, hdf5_name, start_time] = meta
        hdf5_path = os.path.join(self.hdf5s_dir, year, hdf5_name)
        return _load_standard_midi_segment(
            cfg=self.cfg,
            random_state=self.random_state,
            hdf5_path=hdf5_path,
            start_time=start_time,
            segment_samples=self.segment_samples,
            target_processor=self.target_processor,
        )


class MAPS_Dataset(object):
    def __init__(self, cfg):
        """
        Dataset class for the MAPS dataset.
        """
        self.cfg = cfg
        self.random_state = np.random.RandomState(cfg.exp.random_seed)
        self.hdf5s_dir = _resolve_hdf5_dir(cfg.exp.workspace, cfg.feature.sample_rate, "maps")
        self.segment_samples = int(cfg.feature.sample_rate * cfg.feature.segment_seconds)
        # Used for processing MIDI events to target | GroundTruth
        self.target_processor = TargetProcessor(cfg.feature.segment_seconds, cfg)

    def __getitem__(self, meta):
        """
        Prepare input and target for a segment.
        Args:
          meta: dict, e.g., {'hdf5_name': 'Bach_BWV849-01_001_20090916-SMD.h5', 
                             'start_time': 65.0}
        Returns:
          data_dict: dictionary containing waveform and target data.
        """
        [hdf5_name, start_time] = meta
        hdf5_path = os.path.join(self.hdf5s_dir, hdf5_name)
        return _load_standard_midi_segment(
            cfg=self.cfg,
            random_state=self.random_state,
            hdf5_path=hdf5_path,
            start_time=start_time,
            segment_samples=self.segment_samples,
            target_processor=self.target_processor,
        )


class SMD_Dataset(object):
    def __init__(self, cfg):
        """
        Dataset class for the SMD dataset.
        """
        self.cfg = cfg
        self.random_state = np.random.RandomState(cfg.exp.random_seed)
        self.hdf5s_dir = _resolve_hdf5_dir(cfg.exp.workspace, cfg.feature.sample_rate, "smd")
        self.segment_samples = int(cfg.feature.sample_rate * cfg.feature.segment_seconds)
        # Used for processing MIDI events to target | GroundTruth
        self.target_processor = TargetProcessor(cfg.feature.segment_seconds, cfg)

    def __getitem__(self, meta):
        """
        Prepare input and target for a segment.
        Args:
          meta: dict, e.g., {'hdf5_name': 'Bach_BWV849-01_001_20090916-SMD.h5', 
                             'start_time': 65.0}
        Returns:
          data_dict: dictionary containing waveform and target data.
        """
        [hdf5_name, start_time] = meta
        hdf5_path = os.path.join(self.hdf5s_dir, hdf5_name)
        return _load_standard_midi_segment(
            cfg=self.cfg,
            random_state=self.random_state,
            hdf5_path=hdf5_path,
            start_time=start_time,
            segment_samples=self.segment_samples,
            target_processor=self.target_processor,
        )


class _AlignedMidiHdf5Dataset(object):
    def __init__(self, cfg, dataset_name):
        self.cfg = cfg
        self.dataset_name = str(dataset_name)
        self.random_state = np.random.RandomState(cfg.exp.random_seed)
        self.hdf5s_dir = _resolve_hdf5_dir(
            cfg.exp.workspace,
            cfg.feature.sample_rate,
            self.dataset_name,
        )
        self.segment_samples = int(cfg.feature.sample_rate * cfg.feature.segment_seconds)
        self.target_processor = TargetProcessor(cfg.feature.segment_seconds, cfg)

    def __getitem__(self, meta):
        [hdf5_name, start_time] = meta
        hdf5_path = os.path.join(self.hdf5s_dir, hdf5_name)
        return _load_standard_midi_segment(
            cfg=self.cfg,
            random_state=self.random_state,
            hdf5_path=hdf5_path,
            start_time=start_time,
            segment_samples=self.segment_samples,
            target_processor=self.target_processor,
        )


class FrancoisLeduc_Dataset(_AlignedMidiHdf5Dataset):
    def __init__(self, cfg):
        """Dataset class for aligned audio+MIDI FrancoisLeduc HDF5 files."""
        super().__init__(cfg, dataset_name="francoisleduc")


class GAPS_Dataset(_AlignedMidiHdf5Dataset):
    def __init__(self, cfg):
        """Dataset class for aligned audio+MIDI GAPS HDF5 files."""
        super().__init__(cfg, dataset_name="gaps")


def pack_maestro_dataset_to_hdf5(cfg):
    dataset_dir = cfg.dataset.maestro_dir
    csv_path = os.path.join(dataset_dir, "maestro-v3.0.0.csv")
    dataset_name = f"maestro_{_sr_tag(cfg)}"
    waveform_hdf5s_dir = _resolve_hdf5_dir(cfg.exp.workspace, cfg.feature.sample_rate, "maestro")
    logs_dir = os.path.join(cfg.exp.workspace, "logs", f"{get_filename(__file__)}_{dataset_name}")
    create_logging(logs_dir, filemode="w")
    logging.info(f"Packing MAESTRO dataset: {dataset_dir}")

    meta_dict = read_metadata(csv_path)
    audios_num = len(meta_dict["canonical_composer"])
    logging.info(f"Total audios number: {audios_num}")

    feature_time = time.time()
    for n in tqdm(range(audios_num), desc="MAESTRO", unit="track"):

        midi_path = os.path.join(dataset_dir, meta_dict["midi_filename"][n])
        midi_dict = read_midi(midi_path, "maestro")

        audio_path = os.path.join(dataset_dir, meta_dict["audio_filename"][n])
        audio = _load_audio_mono(audio_path, sample_rate=cfg.feature.sample_rate)
        audio = _fit_audio_for_hdf5(audio, audio_path)

        packed_hdf5_path = os.path.join(
            waveform_hdf5s_dir,
            f"{os.path.splitext(meta_dict['audio_filename'][n])[0]}.h5",
        )
        create_folder(os.path.dirname(packed_hdf5_path))

        with h5py.File(packed_hdf5_path, "w") as hf:
            hf.attrs.create(
                "canonical_composer",
                data=meta_dict["canonical_composer"][n].encode(),
                dtype="S100",
            )
            hf.attrs.create(
                "canonical_title",
                data=meta_dict["canonical_title"][n].encode(),
                dtype="S100",
            )
            hf.attrs.create("split", data=meta_dict["split"][n].encode(), dtype="S20")
            hf.attrs.create("year", data=meta_dict["year"][n].encode(), dtype="S10")
            hf.attrs.create(
                "midi_filename", data=meta_dict["midi_filename"][n].encode(), dtype="S100"
            )
            hf.attrs.create(
                "audio_filename",
                data=meta_dict["audio_filename"][n].encode(),
                dtype="S100",
            )
            hf.attrs.create("duration", data=meta_dict["duration"][n], dtype=np.float32)

            hf.create_dataset(
                name="midi_event",
                data=[e.encode() for e in midi_dict["midi_event"]],
                dtype="S100",
            )
            hf.create_dataset(
                name="midi_event_time", data=midi_dict["midi_event_time"], dtype=np.float32
            )
            hf.create_dataset(
                name="waveform", data=float32_to_int16(audio), dtype=np.int16
            )

    logging.info(f"Write HDF5 to {waveform_hdf5s_dir}")
    logging.info(f"Time: {time.time() - feature_time:.3f} s")


def pack_maps_dataset_to_hdf5(cfg):
    dataset_dir = cfg.dataset.maps_dir
    pianos = ["ENSTDkCl", "ENSTDkAm"]
    dataset_name = f"maps_{_sr_tag(cfg)}"
    waveform_hdf5s_dir = _resolve_hdf5_dir(cfg.exp.workspace, cfg.feature.sample_rate, "maps")
    logs_dir = os.path.join(cfg.exp.workspace, "logs", f"{get_filename(__file__)}_{dataset_name}")
    create_logging(logs_dir, filemode="w")
    logging.info(f"Packing MAPS dataset: {dataset_dir}")

    feature_time = time.time()
    count = 0

    for piano in pianos:
        sub_dir = os.path.join(dataset_dir, piano, "MUS")
        audio_names = [
            os.path.splitext(name)[0]
            for name in os.listdir(sub_dir)
            if os.path.splitext(name)[-1] == ".mid"
        ]

        for audio_name in tqdm(audio_names, desc=f"MAPS {piano}", unit="track"):
            audio_path = f"{os.path.join(sub_dir, audio_name)}.wav"
            midi_path = f"{os.path.join(sub_dir, audio_name)}.mid"

            audio = _load_audio_mono(audio_path, sample_rate=cfg.feature.sample_rate)
            audio = _fit_audio_for_hdf5(audio, audio_path)
            midi_dict = read_midi(midi_path, "maps")
            duration = librosa.get_duration(y=audio, sr=cfg.feature.sample_rate)

            packed_hdf5_path = os.path.join(waveform_hdf5s_dir, f"{audio_name}.h5")
            create_folder(os.path.dirname(packed_hdf5_path))

            with h5py.File(packed_hdf5_path, "w") as hf:
                hf.attrs.create("split", data="test".encode(), dtype="S20")
                hf.attrs.create("duration", data=np.float32(duration))
                hf.attrs.create("midi_filename", data=f"{audio_name}.mid".encode(), dtype="S100")
                hf.attrs.create("audio_filename", data=f"{audio_name}.wav".encode(), dtype="S100")
                hf.create_dataset(
                    name="midi_event",
                    data=[e.encode() for e in midi_dict["midi_event"]],
                    dtype="S100",
                )
                hf.create_dataset(
                    name="midi_event_time",
                    data=midi_dict["midi_event_time"],
                    dtype=np.float32,
                )
                hf.create_dataset(
                    name="waveform", data=float32_to_int16(audio), dtype=np.int16
                )
            count += 1

    logging.info(f"Write HDF5 to {waveform_hdf5s_dir}")
    logging.info(f"Total files processed: {count}")
    logging.info(f"Time: {time.time() - feature_time:.3f} s")


def pack_smd_dataset_to_hdf5(cfg):
    dataset_dir = cfg.dataset.smd_dir
    dataset_name = f"smd_{_sr_tag(cfg)}"
    waveform_hdf5s_dir = _resolve_hdf5_dir(cfg.exp.workspace, cfg.feature.sample_rate, "smd")
    logs_dir = os.path.join(cfg.exp.workspace, "logs", f"{get_filename(__file__)}_{dataset_name}")
    create_logging(logs_dir, filemode="w")
    logging.info(f"Packing SMD dataset: {dataset_dir}")

    feature_time = time.time()
    count = 0

    audio_midi_pairs = [
        (os.path.splitext(name)[0], os.path.splitext(name)[-1].lower())
        for name in os.listdir(dataset_dir)
        if os.path.splitext(name)[-1].lower() in [".mid", ".mp3"]
    ]
    audio_midi_pairs = {name: ext for name, ext in audio_midi_pairs}

    excluded = {} #{"Beethoven_WoO080_001_20081107-SMD"}
    for audio_name, ext in tqdm(audio_midi_pairs.items(), desc="SMD", unit="track"):
        if audio_name in excluded:
            logging.info(f"Skipping excluded SMD track: {audio_name}")
            continue
        audio_path = os.path.join(dataset_dir, f"{audio_name}.mp3")
        midi_path = os.path.join(dataset_dir, f"{audio_name}.mid")

        audio = _load_audio_mono(audio_path, sample_rate=cfg.feature.sample_rate)
        audio = _fit_audio_for_hdf5(audio, audio_path)
        midi_dict = read_midi(midi_path, "smd")
        duration = librosa.get_duration(y=audio, sr=cfg.feature.sample_rate)

        packed_hdf5_path = os.path.join(waveform_hdf5s_dir, f"{audio_name}.h5")
        create_folder(os.path.dirname(packed_hdf5_path))

        with h5py.File(packed_hdf5_path, "w") as hf:
            hf.attrs.create("split", data="test".encode(), dtype="S20")
            hf.attrs.create("duration", data=np.float32(duration))
            hf.attrs.create("midi_filename", data=f"{audio_name}.mid".encode(), dtype="S100")
            hf.attrs.create("audio_filename", data=f"{audio_name}.mp3".encode(), dtype="S100")
            hf.create_dataset(
                name="midi_event",
                data=[e.encode() for e in midi_dict["midi_event"]],
                dtype="S100",
            )
            hf.create_dataset(
                name="midi_event_time",
                data=midi_dict["midi_event_time"],
                dtype=np.float32,
            )
            hf.create_dataset(
                name="waveform", data=float32_to_int16(audio), dtype=np.int16
            )
        count += 1

    logging.info(f"Write HDF5 to {waveform_hdf5s_dir}")
    logging.info(f"Total files processed: {count}")
    logging.info(f"Time taken: {time.time() - feature_time:.3f} s")


def pack_francoisleduc_dataset_to_hdf5(cfg):
    dataset_dir_value = getattr(cfg.dataset, "francoisleduc_dir")
    dataset_dir = Path(dataset_dir_value).expanduser().resolve()
    metadata_path = dataset_dir / "metadata.csv"
    rows = _read_francoisleduc_metadata(metadata_path)

    dataset_name = f"francoisleduc_{_sr_tag(cfg)}"
    waveform_hdf5s_dir = _resolve_hdf5_dir(cfg.exp.workspace, cfg.feature.sample_rate, "francoisleduc")
    logs_dir = os.path.join(cfg.exp.workspace, "logs", f"{get_filename(__file__)}_{dataset_name}")
    create_logging(logs_dir, filemode="w")
    logging.info(f"Packing FrancoisLeduc dataset: {dataset_dir}")
    logging.info(f"Using metadata: {metadata_path}")

    feature_time = time.time()
    count = 0

    for row in tqdm(rows, desc="FrancoisLeduc", unit="track"):
        split = _normalize_dataset_split(row["split"])
        audio_path = dataset_dir / row["audio_filename"]
        midi_path = dataset_dir / row["midi_filename"]
        stem = Path(row["audio_filename"]).stem

        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing FrancoisLeduc audio file: {audio_path}")
        if not midi_path.is_file():
            raise FileNotFoundError(f"Missing FrancoisLeduc MIDI file: {midi_path}")

        audio = _load_audio_mono(str(audio_path), sample_rate=cfg.feature.sample_rate)
        audio = _fit_audio_for_hdf5(audio, audio_path)
        midi_dict = read_midi(str(midi_path), "francoisleduc")
        duration = float(librosa.get_duration(y=audio, sr=cfg.feature.sample_rate))

        packed_hdf5_path = os.path.join(waveform_hdf5s_dir, f"{stem}.h5")
        create_folder(os.path.dirname(packed_hdf5_path))

        with h5py.File(packed_hdf5_path, "w") as hf:
            hf.attrs.create("split", data=split.encode(), dtype="S20")
            hf.attrs.create("duration", data=np.float32(duration))
            hf.attrs.create("audio_filename", data=row["audio_filename"].encode(), dtype="S200")
            hf.attrs.create("midi_filename", data=row["midi_filename"].encode(), dtype="S200")
            hf.attrs.create("guitar_type", data=row["guitar_type"].encode(), dtype="S64")
            hf.attrs.create("slice_id", data=row["slice_id"].encode(), dtype="S64")
            hf.attrs.create("artist", data=row["artist"].encode(), dtype="S200")
            hf.attrs.create("name", data=row["name"].encode(), dtype="S300")
            hf.create_dataset(
                name="midi_event",
                data=[e.encode() for e in midi_dict["midi_event"]],
                dtype="S100",
            )
            hf.create_dataset(
                name="midi_event_time",
                data=midi_dict["midi_event_time"],
                dtype=np.float32,
            )
            hf.create_dataset("waveform", data=float32_to_int16(audio), dtype=np.int16)

        count += 1

    logging.info(f"Write HDF5 to {waveform_hdf5s_dir}")
    logging.info(f"Total FrancoisLeduc files processed: {count}")
    logging.info(f"Time taken: {time.time() - feature_time:.3f} s")


def pack_gaps_dataset_to_hdf5(cfg):
    dataset_dir_value = getattr(cfg.dataset, "gaps_dir")
    dataset_dir = Path(dataset_dir_value).expanduser().resolve()
    metadata_path = dataset_dir / "gaps_metadata_with_splits.csv"
    rows = _read_gaps_metadata(metadata_path)

    dataset_name = f"gaps_{_sr_tag(cfg)}"
    waveform_hdf5s_dir = _resolve_hdf5_dir(cfg.exp.workspace, cfg.feature.sample_rate, "gaps")
    logs_dir = os.path.join(cfg.exp.workspace, "logs", f"{get_filename(__file__)}_{dataset_name}")
    create_logging(logs_dir, filemode="w")
    logging.info(f"Packing GAPS dataset: {dataset_dir}")
    logging.info(f"Using metadata: {metadata_path}")

    feature_time = time.time()
    count = 0
    skipped_count = 0
    skipped_rows = []

    for row in tqdm(rows, desc="GAPS", unit="track"):
        split = _normalize_gaps_split(row.get("split"))
        audio_path = dataset_dir / row["audio_path"]
        midi_path = dataset_dir / row["midi_path"]
        stem = str(row.get("id") or Path(row["audio_path"]).stem)

        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing GAPS audio file: {audio_path}")
        if not midi_path.is_file():
            raise FileNotFoundError(f"Missing GAPS MIDI file: {midi_path}")

        audio = _load_audio_mono(str(audio_path), sample_rate=cfg.feature.sample_rate)
        audio = _fit_audio_for_hdf5(audio, audio_path)
        try:
            midi_dict = read_midi(str(midi_path), "gaps")
        except AssertionError as exc:
            logging.warning(f"Skipping GAPS file {stem}: {exc} ({midi_path})")
            skipped_count += 1
            skipped_rows.append({
                "id": stem,
                "split": split,
                "audio_path": str(row["audio_path"]),
                "midi_path": str(row["midi_path"]),
                "reason": str(exc),
            })
            continue
        duration = float(librosa.get_duration(y=audio, sr=cfg.feature.sample_rate))

        packed_hdf5_path = os.path.join(waveform_hdf5s_dir, f"{stem}.h5")
        create_folder(os.path.dirname(packed_hdf5_path))

        with h5py.File(packed_hdf5_path, "w") as hf:
            hf.attrs.create("split", data=split.encode(), dtype="S20")
            hf.attrs.create("duration", data=np.float32(duration))
            hf.attrs.create("id", data=str(stem).encode(), dtype="S64")
            hf.attrs.create("audio_filename", data=row["audio_path"].encode(), dtype="S200")
            hf.attrs.create("midi_filename", data=row["midi_path"].encode(), dtype="S200")
            if row.get("title"):
                hf.attrs.create("title", data=row["title"].encode(), dtype="S300")
            if row.get("composer_name_normalized"):
                hf.attrs.create(
                    "composer_name_normalized",
                    data=row["composer_name_normalized"].encode(),
                    dtype="S200",
                )
            if row.get("performer_name"):
                hf.attrs.create("performer_name", data=row["performer_name"].encode(), dtype="S200")
            hf.create_dataset(
                name="midi_event",
                data=[e.encode() for e in midi_dict["midi_event"]],
                dtype="S100",
            )
            hf.create_dataset(
                name="midi_event_time",
                data=midi_dict["midi_event_time"],
                dtype=np.float32,
            )
            hf.create_dataset("waveform", data=float32_to_int16(audio), dtype=np.int16)

        count += 1

    if skipped_rows:
        skipped_csv_path = os.path.join(waveform_hdf5s_dir, "_skipped_gaps.csv")
        with open(skipped_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "split", "audio_path", "midi_path", "reason"],
            )
            writer.writeheader()
            writer.writerows(skipped_rows)
        logging.info(f"Wrote skipped GAPS entries to {skipped_csv_path}")

    logging.info(f"Write HDF5 to {waveform_hdf5s_dir}")
    logging.info(f"Total GAPS files processed: {count}")
    logging.info(f"Total GAPS files skipped: {skipped_count}")
    logging.info(f"Time taken: {time.time() - feature_time:.3f} s")

class Augmentor(object):
    def __init__(self, cfg):
        """Data augmentor."""
        self.cfg = cfg
        self.sample_rate = cfg.feature.sample_rate
        self.random_state = np.random.RandomState(cfg.exp.random_seed)

    def augment(self, x):
        clip_samples = len(x)
        aug_x = np.asarray(x, dtype=np.float32).copy()

        # Random time-stretch
        if self.random_state.rand() < 0.5:
            rate = float(self.random_state.uniform(0.9, 1.1))
            aug_x = librosa.effects.time_stretch(aug_x, rate)

        # Random pitch shift (±0.5 semitone)
        if self.random_state.rand() < 0.5:
            steps = float(self.random_state.uniform(-0.5, 0.5))
            aug_x = librosa.effects.pitch_shift(
                aug_x, self.sample_rate, n_steps=steps, bins_per_octave=12
            )

        aug_x = self._random_eq(aug_x)
        aug_x = self._simple_reverb(aug_x)

        noise_scale = float(self.random_state.uniform(0.001, 0.01))
        aug_x = aug_x + self.random_state.normal(0.0, noise_scale, size=len(aug_x))
        gain = float(self.random_state.uniform(0.8, 1.2))
        aug_x *= gain
        aug_x = np.clip(aug_x, -1.0, 1.0)
        aug_x = pad_truncate_sequence(aug_x, clip_samples)
        return aug_x

    def _random_eq(self, waveform: np.ndarray) -> np.ndarray:
        if self.random_state.rand() < 0.3:
            return waveform
        fft = np.fft.rfft(waveform)
        freqs = np.fft.rfftfreq(len(waveform), 1.0 / self.sample_rate)
        response = np.ones_like(freqs)
        num_bands = self.random_state.randint(1, 4)
        for _ in range(num_bands):
            center = self.random_state.uniform(80.0, self.sample_rate / 2.0)
            width = self.random_state.uniform(100.0, 2000.0)
            gain = self.random_state.uniform(0.5, 1.5)
            response *= 1 + (gain - 1) * np.exp(-0.5 * ((freqs - center) / width) ** 2)
        fft *= response
        shaped = np.fft.irfft(fft, n=len(waveform))
        return shaped.real

    def _simple_reverb(self, waveform: np.ndarray) -> np.ndarray:
        if self.random_state.rand() < 0.3:
            return waveform
        delay = self.random_state.randint(
            int(0.01 * self.sample_rate), int(0.05 * self.sample_rate)
        )
        decay = float(self.random_state.uniform(0.1, 0.5))
        kernel = np.zeros(delay + 1, dtype=np.float32)
        kernel[0] = 1.0
        kernel[-1] = decay
        reverbed = np.convolve(waveform, kernel, mode="full")
        return reverbed[: len(waveform)]

class Sampler(object):
    def __init__(self, cfg, split, is_eval=None):
        """
        Sampler is used to sample segments for training or evaluation.
        Args:
          cfg: OmegaConf configuration containing dataset and experiment details.
          split: 'train' | 'validation' | 'test'.
          random_seed: int, random seed for reproducibility.
        """
        assert split in ['train', 'validation', 'test']
        self.is_eval = is_eval
        dataset_name = _normalize_aligned_dataset_name(is_eval if split == "test" else cfg.dataset.train_set)
        self.hdf5s_dir = _resolve_hdf5_dir(
            cfg.exp.workspace,
            cfg.feature.sample_rate,
            dataset_name,
        )
        self.segment_seconds = cfg.feature.segment_seconds
        self.hop_seconds = cfg.feature.hop_seconds
        self.sample_rate = cfg.feature.sample_rate
        self.random_state = np.random.RandomState(cfg.exp.random_seed)
        self.batch_size = cfg.exp.batch_size
        self.dataset_type = dataset_name
        # self.dataset_type = cfg.dataset.test_set if split == "test" else cfg.dataset.train_set
        self.mini_data = cfg.exp.mini_data
        
        
        (hdf5_names, hdf5_paths) = traverse_folder(self.hdf5s_dir)
        self.segment_list = []

        n = 0
        for hdf5_path in hdf5_paths:
            base_name = os.path.basename(hdf5_path)
            if (not hdf5_path.lower().endswith(('.h5', '.hdf5'))
                or base_name.startswith('._')):
                continue  # Skip non-HDF5 files and AppleDouble copies
            with h5py.File(hdf5_path, 'r') as hf:
                file_split = _decode_hdf5_str(hf.attrs['split'])
                if file_split == split:
                    audio_name = hdf5_path.split('/')[-1]
                    start_time = 0

                    # Maestro-specific handling
                    if self.dataset_type == "maestro":
                        year = _decode_hdf5_str(hf.attrs['year'])
                        file_id = [year, audio_name]
                    elif self.dataset_type == "smd":
                        file_id = [audio_name]
                    elif self.dataset_type == "maps":
                        file_id = [audio_name]
                    elif self.dataset_type in {"francoisleduc", "gaps"}:
                        file_id = [audio_name]
                    else:
                        raise KeyError(f"Unsupported dataset type in sampler: {self.dataset_type}")

                    duration = float(hf.attrs['duration'])
                    while start_time + self.segment_seconds < duration:
                        self.segment_list.append(file_id + [start_time])
                        start_time += self.hop_seconds
                    n += 1

                    if self.mini_data and n == 10:
                        break

        """self.segment_list looks like:
        [['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 1.0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 2.0]
         ...]"""
        # Log segment count
        log_prefix = "eval " if self.is_eval else ""
        logging.info(f"{log_prefix}{split} segments: {len(self.segment_list)}")

        self.pointer = 0
        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)

    def __iter__(self):
        while True:
            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[self.pointer]
                self.pointer += 1

                if self.pointer >= len(self.segment_indexes):
                    self.pointer = 0
                    self.random_state.shuffle(self.segment_indexes)

                batch_segment_list.append(self.segment_list[index])
                i += 1

            yield batch_segment_list

    def __len__(self):
        return int(np.ceil(len(self.segment_list) / self.batch_size))
        
    def state_dict(self):
        state = {
            'pointer': self.pointer, 
            'segment_indexes': self.segment_indexes}
        return state
            
    def load_state_dict(self, state):
        self.pointer = state['pointer']
        self.segment_indexes = state['segment_indexes']


class EvalSampler(Sampler):
    def __init__(self, cfg, split, is_eval):
        """
        Sampler for Evaluation.

        Args:
          cfg: OmegaConf configuration containing dataset and experiment details.
          split: 'train' | 'validation' | 'test'.
          random_seed: int, random seed for reproducibility.
        """
        super().__init__(cfg, split, is_eval)
        default_iters = 20
        if cfg.exp.batch_size >= 30:
            default_iters = 10
        self.max_evaluate_iteration = default_iters  # Limit validation iterations

    def __iter__(self):
        pointer = 0
        iteration = 0

        while iteration < self.max_evaluate_iteration:
            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[pointer]
                pointer += 1
                batch_segment_list.append(self.segment_list[index])
                i += 1

            iteration += 1
            yield batch_segment_list


def collate_fn(list_data_dict):
    """Collate input and target of segments to a mini-batch.

    Args:
      list_data_dict: e.g. [
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        ...]

    Returns:
      np_data_dict: e.g. {
        'waveform': (batch_size, segment_samples)
        'frame_roll': (batch_size, segment_frames, classes_num), 
        ...}
    """
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        values = [data_dict[key] for data_dict in list_data_dict]
        if key == "aligned_note_events":
            np_data_dict[key] = values
        else:
            np_data_dict[key] = np.array(values)

    return np_data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset packing utilities")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Select a mode of operation")
    subparsers.add_parser("pack_maestro_dataset_to_hdf5", help="Pack Maestro dataset to HDF5")
    subparsers.add_parser("pack_maps_dataset_to_hdf5", help="Pack MAPS dataset to HDF5")
    subparsers.add_parser("pack_smd_dataset_to_hdf5", help="Pack SMD dataset to HDF5")
    subparsers.add_parser("pack_francoisleduc_dataset_to_hdf5", help="Pack FrancoisLeduc dataset to HDF5")
    subparsers.add_parser("pack_gaps_dataset_to_hdf5", help="Pack GAPS dataset to HDF5")

    args, hydra_overrides = parser.parse_known_args()

    initialize(config_path="./config", job_name="features", version_base=None)
    cfg = compose(config_name="config", overrides=hydra_overrides)

    mode_to_function = {
        "pack_maestro_dataset_to_hdf5": pack_maestro_dataset_to_hdf5,
        "pack_maps_dataset_to_hdf5": pack_maps_dataset_to_hdf5,
        "pack_smd_dataset_to_hdf5": pack_smd_dataset_to_hdf5,
        "pack_francoisleduc_dataset_to_hdf5": pack_francoisleduc_dataset_to_hdf5,
        "pack_gaps_dataset_to_hdf5": pack_gaps_dataset_to_hdf5,
    }

    if args.mode in mode_to_function:
        mode_to_function[args.mode](cfg)
    else:
        raise ValueError(f"Invalid mode '{args.mode}'. Use --help for available modes.")
