import os
import logging
import librosa
import numpy as np
import csv
import collections
from pathlib import Path
from mido import MidiFile

from piano_vad import (note_detection_with_onset_offset_regress, 
    pedal_detection_with_onset_offset_regress, onsets_frames_note_detection, onsets_frames_pedal_detection)


def get_model_name(cfg):
    """Construct model name based on model name and optional extra inputs."""
    extras = '+'.join(filter(None, [cfg.model.input2, cfg.model.input3]))
    return f"{cfg.model.type}" + (f"+{extras}" if extras else "")


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
        
def resolve_hdf5_dir(workspace, dataset_name, sample_rate=None):
    """Resolve the actual hdf5 directory by checking sr-specific suffixes."""
    base = os.path.join(workspace, "hdf5s")
    direct = os.path.join(base, dataset_name)
    if os.path.isdir(direct):
        return direct

    candidates = []
    if sample_rate is not None:
        sr_dir = os.path.join(base, f"{dataset_name}_sr{int(sample_rate)}")
        candidates.append(sr_dir)
    # look for any directory starting with dataset_name
    if os.path.isdir(base):
        for entry in os.listdir(base):
            if entry.startswith(dataset_name + "_sr"):
                candidates.append(os.path.join(base, entry))

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        f"Cannot resolve hdf5 directory for dataset '{dataset_name}' under {base}"
    )


def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def traverse_folder(folder):
    paths = []
    names = []
    
    for root, dirs, files in os.walk(folder):
        for name in files:
            filepath = os.path.join(root, name)
            names.append(name)
            paths.append(filepath)
            
    return names, paths


DEFAULT_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
DEFAULT_MIDI_EXTS = (".mid", ".midi")


def _match_with_suffix(base_path: Path, candidate_exts):
    base = base_path.with_suffix("")
    for ext in candidate_exts:
        candidate = base.with_suffix(ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate any of {candidate_exts} matching base '{base.name}' near {base.parent}"
    )


def resolve_audio_midi_pair(path, audio_exts=DEFAULT_AUDIO_EXTS, midi_exts=DEFAULT_MIDI_EXTS):
    """Return (audio_path, midi_path) for a file assumed to be part of a pair."""
    candidate = Path(path)
    suffix = candidate.suffix.lower()
    if suffix in audio_exts:
        audio_path = candidate
        midi_path = _match_with_suffix(candidate, midi_exts)
    elif suffix in midi_exts:
        midi_path = candidate
        audio_path = _match_with_suffix(candidate, audio_exts)
    else:
        raise ValueError(f"Unsupported file extension for '{candidate}'. Provide audio {audio_exts} or MIDI {midi_exts}.")
    return audio_path, midi_path


def collect_audio_midi_pairs(folder, audio_exts=DEFAULT_AUDIO_EXTS, midi_exts=DEFAULT_MIDI_EXTS):
    """Return sorted list of (audio_path, midi_path) pairs under folder."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a directory.")

    pairs = []
    for item in sorted(folder.iterdir()):
        if not item.is_file() or item.suffix.lower() not in audio_exts:
            continue
        try:
            midi_path = _match_with_suffix(item, midi_exts)
        except FileNotFoundError:
            continue
        pairs.append((item, midi_path))

    if not pairs:
        raise FileNotFoundError(f"No audio/MIDI pairs found in {folder}")
    return pairs


def load_mono_audio(audio_path, sample_rate):
    """Load mono audio with librosa."""
    path = Path(audio_path)
    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return audio.astype(np.float32)

    
def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging


def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.
    return (x * 32767.).astype(np.int16)


def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)
    

def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]


def read_metadata(csv_path):
    """Read metadata of MAESTRO dataset from csv file.

    Args:
      csv_path: str

    Returns:
      meta_dict, dict, e.g. {
        'canonical_composer': ['Alban Berg', ...], 
        'canonical_title': ['Sonata Op. 1', ...], 
        'split': ['train', ...], 
        'year': ['2018', ...]
        'midi_filename': ['2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi', ...], 
        'audio_filename': ['2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav', ...],
        'duration': [698.66116031, ...]}
    """

    with open(csv_path, 'r') as fr:
        reader = csv.reader(fr, delimiter=',')
        lines = list(reader)

    meta_dict = {'canonical_composer': [], 'canonical_title': [], 'split': [], 
        'year': [], 'midi_filename': [], 'audio_filename': [], 'duration': []}

    for n in range(1, len(lines)):
        meta_dict['canonical_composer'].append(lines[n][0])
        meta_dict['canonical_title'].append(lines[n][1])
        meta_dict['split'].append(lines[n][2])
        meta_dict['year'].append(lines[n][3])
        meta_dict['midi_filename'].append(lines[n][4])
        meta_dict['audio_filename'].append(lines[n][5])
        meta_dict['duration'].append(float(lines[n][6]))

    for key in meta_dict.keys():
        meta_dict[key] = np.array(meta_dict[key])
    
    return meta_dict


def read_midi(midi_path, dataset='maestro'):
    """
    Parse a MIDI file and return events with timestamps.

    Args:
        midi_path (str): Path to the MIDI file.
        dataset (str): One of 'maestro', 'hpt', 'francoisleduc', 'gaps', 'smd', or 'maps'.
            Determines where tempo and events are stored.

            - 'maestro', 'hpt', 'francoisleduc', or 'gaps': 2 tracks.
              • Track 0 holds all meta messages (set_tempo, time_signature, end_of_track).
              • Track 1 holds piano events.

            - 'smd': 2 tracks.
              • Track 0 holds meta messages, but tempo is the second index (track_name, set_tempo, time_signature, end_of_track).
              • Track 1 holds piano events.

            - 'maps': 1 track.
              • That track holds both meta messages and piano events (tempo is the first message).

    Returns:
        dict: {
            'midi_event': np.ndarray of message strings,
            'midi_event_time': np.ndarray of timestamps in seconds
        }
    """
    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat

    ds = dataset.lower()
    if ds in ('maestro', 'hpt', 'francoisleduc', 'gaps'):
        # Expect 2 tracks: track 0 for meta (tempo at index 0), track 1 for piano events
        assert len(midi_file.tracks) == 2, f"{dataset} format requires 2 tracks, found {len(midi_file.tracks)}"
        microseconds_per_beat = midi_file.tracks[0][0].tempo
        play_tracks = [midi_file.tracks[1]]

    elif ds == 'smd':
        # Expect 2 tracks: track 0 for meta (tempo at index 1), track 1 for piano events
        assert len(midi_file.tracks) == 2, f"SMD format requires 2 tracks, found {len(midi_file.tracks)}"
        microseconds_per_beat = midi_file.tracks[0][1].tempo
        play_tracks = [midi_file.tracks[1]]

    elif ds == 'maps':
        # Expect 1 track: contains both meta and piano events (tempo at index 0)
        assert len(midi_file.tracks) == 1, f"MAPS format requires 1 track, found {len(midi_file.tracks)}"
        microseconds_per_beat = midi_file.tracks[0][0].tempo
        play_tracks = [midi_file.tracks[0]]

    else:
        raise ValueError(f"Dataset not supported: {dataset}")

    # Convert ticks to seconds
    beats_per_second = 1e6 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second

    message_list = []
    time_in_second = []
    ticks_accum = 0

    for msg in play_tracks[0]:
        message_list.append(str(msg))
        ticks_accum += msg.time
        time_in_second.append(ticks_accum / ticks_per_second)

    return {
        'midi_event': np.array(message_list),
        'midi_event_time': np.array(time_in_second)
    }


class TargetProcessor(object):
    def __init__(self, segment_seconds, cfg):
        """Class for processing MIDI events to target.

        Args:
          segment_seconds: float
          frames_per_second: int
          begin_note: int, A0 MIDI note of a piano
          classes_num: int
        """
        self.segment_seconds = segment_seconds
        self.frames_per_second = cfg.feature.frames_per_second
        self.begin_note = cfg.feature.begin_note
        self.classes_num = cfg.feature.classes_num
        self.max_piano_note = self.classes_num - 1

    def process(self, start_time, midi_events_time, midi_events, 
        extend_pedal=True, note_shift=0):
        """Process MIDI events of an audio segment to target for training, 
        includes: 
        1. Parse MIDI events
        2. Prepare note targets
        3. Prepare pedal targets

        Args:
          start_time: float, start time of a segment
          midi_events_time: list of float, times of MIDI events of a recording, 
            e.g. [0, 3.3, 5.1, ...]
          midi_events: list of str, MIDI events of a recording, e.g.
            ['note_on channel=0 note=75 velocity=37 time=14',
             'control_change channel=0 control=64 value=54 time=20',
             ...]
          extend_pedal, bool, True: Notes will be set to ON until pedal is 
            released. False: Ignore pedal events.

        Returns:
          target_dict: {
            'onset_roll': (frames_num, classes_num), 
            'offset_roll': (frames_num, classes_num), 
            'reg_onset_roll': (frames_num, classes_num), 
            'reg_offset_roll': (frames_num, classes_num), 
            'frame_roll': (frames_num, classes_num), 
            'velocity_roll': (frames_num, classes_num), 
            'mask_roll':  (frames_num, classes_num), 
            'pedal_onset_roll': (frames_num,), 
            'pedal_offset_roll': (frames_num,), 
            'reg_pedal_onset_roll': (frames_num,), 
            'reg_pedal_offset_roll': (frames_num,), 
            'pedal_frame_roll': (frames_num,)}

          note_events: list of dict, e.g. [
            {'midi_note': 51, 'onset_time': 696.64, 'offset_time': 697.00, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 697.00, 'offset_time': 697.19, 'velocity': 50}
            ...]

          pedal_events: list of dict, e.g. [
            {'onset_time': 149.37, 'offset_time': 150.35}, 
            {'onset_time': 150.54, 'offset_time': 152.06}, 
            ...]
        """

        # ------ 1. Parse MIDI events ------
        # Search the begin index of a segment
        for bgn_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time:
                break
        """E.g., start_time: 709.0, bgn_idx: 18003, event_time: 709.0146"""

        # Search the end index of a segment
        for fin_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time + self.segment_seconds:
                break
        """E.g., start_time: 709.0, bgn_idx: 18196, event_time: 719.0115"""

        note_events = []
        """E.g. [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]"""

        pedal_events = []
        """E.g. [
            {'onset_time': 696.46875, 'offset_time': 696.62604}, 
            {'onset_time': 696.8063, 'offset_time': 698.50836}, 
            ...]"""

        buffer_dict = {}    # Used to store onset of notes to be paired with offsets
        pedal_dict = {}     # Used to store onset of pedal to be paired with offset of pedal

        # Backtrack bgn_idx to earlier indexes: ex_bgn_idx, which is used for 
        # searching cross segment pedal and note events. E.g.: bgn_idx: 1149, 
        # ex_bgn_idx: 981
        _delta = int((fin_idx - bgn_idx) * 1.)  
        ex_bgn_idx = max(bgn_idx - _delta, 0)
        
        for i in range(ex_bgn_idx, fin_idx):
            # Parse MIDI messiage
            attribute_list = midi_events[i].split(' ')

            # Note
            if attribute_list[0] in ['note_on', 'note_off']:
                """E.g. attribute_list: ['note_on', 'channel=0', 'note=41', 'velocity=0', 'time=10']"""

                midi_note = int(attribute_list[2].split('=')[1])
                velocity = int(attribute_list[3].split('=')[1])

                # Onset
                if attribute_list[0] == 'note_on' and velocity > 0:
                    buffer_dict[midi_note] = {
                        'onset_time': midi_events_time[i], 
                        'velocity': velocity}

                # Offset
                else:
                    if midi_note in buffer_dict.keys():
                        note_events.append({
                            'midi_note': midi_note, 
                            'onset_time': buffer_dict[midi_note]['onset_time'], 
                            'offset_time': midi_events_time[i], 
                            'velocity': buffer_dict[midi_note]['velocity']})
                        del buffer_dict[midi_note]

            # Pedal
            elif attribute_list[0] == 'control_change' and attribute_list[2] == 'control=64':
                """control=64 corresponds to pedal MIDI event. E.g. 
                attribute_list: ['control_change', 'channel=0', 'control=64', 'value=45', 'time=43']"""

                ped_value = int(attribute_list[3].split('=')[1])
                if ped_value >= 64:
                    if 'onset_time' not in pedal_dict:
                        pedal_dict['onset_time'] = midi_events_time[i]
                else:
                    if 'onset_time' in pedal_dict:
                        pedal_events.append({
                            'onset_time': pedal_dict['onset_time'], 
                            'offset_time': midi_events_time[i]})
                        pedal_dict = {}

        # Add unpaired onsets to events
        for midi_note in buffer_dict.keys():
            note_events.append({
                'midi_note': midi_note, 
                'onset_time': buffer_dict[midi_note]['onset_time'], 
                'offset_time': start_time + self.segment_seconds, 
                'velocity': buffer_dict[midi_note]['velocity']})

        # Add unpaired pedal onsets to data
        if 'onset_time' in pedal_dict.keys():
            pedal_events.append({
                'onset_time': pedal_dict['onset_time'], 
                'offset_time': start_time + self.segment_seconds})

        # Set notes to ON until pedal is released
        if extend_pedal:
            note_events = apply_pedal_extension(note_events, pedal_events)
        
        # Prepare targets
        frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1
        onset_roll = np.zeros((frames_num, self.classes_num))
        offset_roll = np.zeros((frames_num, self.classes_num))
        reg_onset_roll = np.ones((frames_num, self.classes_num))
        reg_offset_roll = np.ones((frames_num, self.classes_num))
        frame_roll = np.zeros((frames_num, self.classes_num))
        velocity_roll = np.zeros((frames_num, self.classes_num))
        mask_roll = np.ones((frames_num, self.classes_num))
        """mask_roll is used for masking out cross segment notes"""

        pedal_onset_roll = np.zeros(frames_num)
        pedal_offset_roll = np.zeros(frames_num)
        reg_pedal_onset_roll = np.ones(frames_num)
        reg_pedal_offset_roll = np.ones(frames_num)
        pedal_frame_roll = np.zeros(frames_num)

        # ------ 2. Get note targets ------
        # Process note events to target
        for note_event in note_events:
            """note_event: e.g., {'midi_note': 60, 'onset_time': 722.0719, 'offset_time': 722.47815, 'velocity': 103}"""

            piano_note = np.clip(note_event['midi_note'] - self.begin_note + note_shift, 0, self.max_piano_note) 
            """There are 88 keys on a piano"""

            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((note_event['onset_time'] - start_time) * self.frames_per_second))
                fin_frame = int(round((note_event['offset_time'] - start_time) * self.frames_per_second))

                if fin_frame >= 0:
                    frame_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = 1

                    offset_roll[fin_frame, piano_note] = 1
                    velocity_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = note_event['velocity']

                    # Vector from the center of a frame to ground truth offset
                    reg_offset_roll[fin_frame, piano_note] = \
                        (note_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)

                    if bgn_frame >= 0:
                        onset_roll[bgn_frame, piano_note] = 1

                        # Vector from the center of a frame to ground truth onset
                        reg_onset_roll[bgn_frame, piano_note] = \
                            (note_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)
                
                    # Mask out segment notes
                    else:
                        mask_roll[: fin_frame + 1, piano_note] = 0

        for k in range(self.classes_num):
            """Get regression targets"""
            reg_onset_roll[:, k] = regression_curve(reg_onset_roll[:, k], self.frames_per_second)
            reg_offset_roll[:, k] = regression_curve(reg_offset_roll[:, k], self.frames_per_second)

        # Process unpaired onsets to target
        for midi_note in buffer_dict.keys():
            piano_note = np.clip(midi_note - self.begin_note + note_shift, 0, self.max_piano_note)
            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((buffer_dict[midi_note]['onset_time'] - start_time) * self.frames_per_second))
                mask_roll[bgn_frame :, piano_note] = 0     

        # ------ 3. Get pedal targets ------
        # Process pedal events to target
        for pedal_event in pedal_events:
            bgn_frame = int(round((pedal_event['onset_time'] - start_time) * self.frames_per_second))
            fin_frame = int(round((pedal_event['offset_time'] - start_time) * self.frames_per_second))

            if fin_frame >= 0:
                pedal_frame_roll[max(bgn_frame, 0) : fin_frame + 1] = 1

                pedal_offset_roll[fin_frame] = 1
                reg_pedal_offset_roll[fin_frame] = \
                    (pedal_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)

                if bgn_frame >= 0:
                    pedal_onset_roll[bgn_frame] = 1
                    reg_pedal_onset_roll[bgn_frame] = \
                        (pedal_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)

        # Get regresssion padal targets
        reg_pedal_onset_roll = regression_curve(reg_pedal_onset_roll, self.frames_per_second)
        reg_pedal_offset_roll = regression_curve(reg_pedal_offset_roll, self.frames_per_second)

        target_dict = {
            'onset_roll': onset_roll, 'offset_roll': offset_roll,
            'reg_onset_roll': reg_onset_roll, 'reg_offset_roll': reg_offset_roll,
            'frame_roll': frame_roll, 'velocity_roll': velocity_roll, 
            'mask_roll': mask_roll, 'reg_pedal_onset_roll': reg_pedal_onset_roll, 
            'pedal_onset_roll': pedal_onset_roll, 'pedal_offset_roll': pedal_offset_roll, 
            'reg_pedal_offset_roll': reg_pedal_offset_roll, 'pedal_frame_roll': pedal_frame_roll
            }

        return target_dict, note_events, pedal_events


def prepare_aux_rolls(cfg, midi_events_time, midi_events, duration_sec):
    processor = TargetProcessor(segment_seconds=duration_sec, cfg=cfg)
    target_dict, note_events, pedal_events = processor.process(
        start_time=0.0,
        midi_events_time=midi_events_time,
        midi_events=midi_events,
        extend_pedal=True,
    )
    if "exframe_roll" not in target_dict:
        target_dict["exframe_roll"] = target_dict["frame_roll"] * (1 - target_dict["onset_roll"])
    return target_dict, note_events, pedal_events


def original_score_events(cfg, midi_events_time, midi_events, duration_sec):
    processor = TargetProcessor(segment_seconds=duration_sec, cfg=cfg)
    _, note_events, pedal_events = processor.process(
        start_time=0.0,
        midi_events_time=midi_events_time,
        midi_events=midi_events,
        extend_pedal=False,
    )
    return note_events, pedal_events


def select_condition_roll(target_dict, condition_name):
    if condition_name is None:
        return None
    name = str(condition_name).strip()
    if not name or name.lower() in {"none", "null"}:
        return None
    key = f"{name}_roll"
    if key not in target_dict:
        raise KeyError(f"Condition '{name}' requested, but '{key}' not found in target_dict.")
    return target_dict[key]


def pick_velocity_from_roll(note_events, velocity_roll, cfg, strategy):
    """Assign one MIDI velocity to each note event from a frame-wise velocity roll.

    Note: the supervision roll may carry the same velocity value across the full
    note duration for training stability. However, the physical meaning of MIDI
    velocity is onset strength. Therefore `onset_only` is the default inference
    strategy for HPT / score-HPT style models, while `max_frame` is kept as an
    optional alternative for architectures that empirically prefer it.
    """
    fps = cfg.feature.frames_per_second
    begin_note = cfg.feature.begin_note
    velocity_scale = cfg.feature.velocity_scale
    num_frames, num_keys = velocity_roll.shape

    for event in note_events:
        pitch_idx = event["midi_note"] - begin_note
        if pitch_idx < 0 or pitch_idx >= num_keys:
            continue

        onset_frame = int(round(event["onset_time"] * fps))
        offset_frame = int(round(event["offset_time"] * fps))
        onset_frame = np.clip(onset_frame, 0, max(0, num_frames - 1))
        offset_frame = max(onset_frame + 1, offset_frame)
        offset_frame = min(offset_frame, num_frames)

        note_curve = velocity_roll[onset_frame:offset_frame, pitch_idx]
        if note_curve.size == 0:
            picked = 0.0
        elif strategy == "max_frame":
            picked = float(np.max(note_curve))
        elif strategy == "onset_only":
            picked = float(note_curve[0])
        else:
            raise ValueError(f"Unknown velocity pick strategy: {strategy}")

        scaled_velocity = np.clip(picked * velocity_scale, 0, velocity_scale - 1)
        event["velocity"] = int(round(scaled_velocity))


def check_duration_alignment(audio_len, midi_events_time):
    midi_len = float(midi_events_time[-1]) if midi_events_time.size else 0.0
    diff = abs(audio_len - midi_len)
    if diff > 0.05:
        print(
            f"[warn] Audio duration ({audio_len:.2f}s) and MIDI duration ({midi_len:.2f}s) "
            f"differ by {diff:.2f}s. Proceeding with audio duration as reference.",
        )


def iteration_label_from_path(path, fallback=None):
    path = Path(path)
    stem = path.stem
    if stem.endswith("_iterations"):
        stem = stem[: -len("_iterations")]
    if stem:
        return stem
    if fallback:
        return str(fallback)
    return "custom"


def apply_pedal_extension(note_events, pedal_events):
    """Update the offset of all notes until pedal is released."""
    note_events = collections.deque(note_events)
    pedal_events = collections.deque(pedal_events)
    ex_note_events = []

    idx = 0
    while pedal_events:
        pedal_event = pedal_events.popleft()
        buffer_dict = {}
        while note_events:
            note_event = note_events.popleft()
            if pedal_event["onset_time"] < note_event["offset_time"] < pedal_event["offset_time"]:
                midi_note = note_event["midi_note"]
                if midi_note in buffer_dict:
                    _idx = buffer_dict[midi_note]
                    del buffer_dict[midi_note]
                    ex_note_events[_idx]["offset_time"] = note_event["onset_time"]
                note_event["offset_time"] = pedal_event["offset_time"]
                buffer_dict[midi_note] = idx

            ex_note_events.append(note_event)
            idx += 1
            if note_event["offset_time"] > pedal_event["offset_time"]:
                break

    while note_events:
        ex_note_events.append(note_events.popleft())

    return ex_note_events


def regression_curve(input_vec, frames_per_second):
    """Regression target used for onset/offset localization."""
    step = 1.0 / frames_per_second
    output = np.ones_like(input_vec)

    locts = np.where(input_vec < 0.5)[0]
    if locts.size:
        for t in range(0, locts[0]):
            output[t] = step * (t - locts[0]) - input_vec[locts[0]]

        for i in range(0, len(locts) - 1):
            mid = (locts[i] + locts[i + 1]) // 2
            for t in range(locts[i], mid):
                output[t] = step * (t - locts[i]) - input_vec[locts[i]]
            for t in range(mid, locts[i + 1]):
                output[t] = step * (t - locts[i + 1]) - input_vec[locts[i]]

        for t in range(locts[-1], len(input_vec)):
            output[t] = step * (t - locts[-1]) - input_vec[locts[-1]]

    output = np.clip(np.abs(output), 0.0, 0.05) * 20
    return 1.0 - output


def note_events_to_velocity_roll(
    note_events,
    frames_num,
    num_keys,
    frames_per_second,
    begin_note,
    velocity_scale,
):
    """Render note events with integer velocities back to a frame/key grid."""
    roll = np.zeros((frames_num, num_keys), dtype=np.float32)
    scale = float(velocity_scale)

    for event in note_events:
        midi_note = event.get("midi_note")
        if midi_note is None:
            continue
        pitch_idx = int(midi_note) - int(begin_note)
        if pitch_idx < 0 or pitch_idx >= num_keys:
            continue

        onset = int(round(float(event.get("onset_time", 0.0)) * frames_per_second))
        offset = int(round(float(event.get("offset_time", 0.0)) * frames_per_second))
        onset = np.clip(onset, 0, max(0, frames_num - 1))
        offset = max(onset + 1, offset)
        offset = min(offset, frames_num)

        value = float(event.get("velocity", 0)) / scale
        if value <= 0:
            continue
        roll[onset:offset, pitch_idx] = value
    return roll


def write_events_to_midi(start_time, note_events, pedal_events, midi_path):
    """Write out note events to MIDI file.

    Args:
      start_time: float
      note_events: list of dict, e.g. [
        {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
        {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
        ...]
      midi_path: str
    """
    from mido import Message, MidiFile, MidiTrack, MetaMessage
    
    # This configuration is the same as MIDIs in MAESTRO dataset
    ticks_per_beat = 384
    beats_per_second = 2
    ticks_per_second = ticks_per_beat * beats_per_second
    microseconds_per_beat = int(1e6 // beats_per_second)

    midi_file = MidiFile()
    midi_file.ticks_per_beat = ticks_per_beat

    # Track 0
    track0 = MidiTrack()
    track0.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
    track0.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    track0.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track0)

    # Track 1
    track1 = MidiTrack()
    
    # Message rolls of MIDI
    message_roll = []

    for note_event in note_events:
        # Onset
        message_roll.append({
            'time': note_event['onset_time'], 
            'midi_note': note_event['midi_note'], 
            'velocity': note_event['velocity']})

        # Offset
        message_roll.append({
            'time': note_event['offset_time'], 
            'midi_note': note_event['midi_note'], 
            'velocity': 0})

    if pedal_events:
        for pedal_event in pedal_events:
            message_roll.append({'time': pedal_event['onset_time'], 'control_change': 64, 'value': 127})
            message_roll.append({'time': pedal_event['offset_time'], 'control_change': 64, 'value': 0})

    # Sort MIDI messages by time
    message_roll.sort(key=lambda note_event: note_event['time'])

    previous_ticks = 0
    for message in message_roll:
        this_ticks = int((message['time'] - start_time) * ticks_per_second)
        if this_ticks >= 0:
            diff_ticks = this_ticks - previous_ticks
            previous_ticks = this_ticks
            if 'midi_note' in message.keys():
                track1.append(Message('note_on', note=message['midi_note'], velocity=message['velocity'], time=diff_ticks))
            elif 'control_change' in message.keys():
                track1.append(Message('control_change', channel=0, control=message['control_change'], value=message['value'], time=diff_ticks))
    track1.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track1)

    midi_file.save(midi_path)


class RegressionPostProcessor(object):
    def __init__(self, cfg):
        """Postprocess the output probabilities of a transription model to MIDI 
        events.

        Args:
          frames_per_second: int
          classes_num: int
          onset_threshold: float
          offset_threshold: float
          frame_threshold: float
          pedal_offset_threshold: float
        """
        self.frames_per_second = cfg.feature.frames_per_second
        self.classes_num = cfg.feature.classes_num
        self.begin_note = cfg.feature.begin_note
        self.velocity_scale = cfg.feature.velocity_scale

        self.frame_threshold = cfg.post.frame_threshold                 # 0.1
        self.onset_threshold = cfg.post.onset_threshold                 # 0.3
        self.offset_threshold = cfg.post.offset_threshold               # 0.3
        self.pedal_offset_threshold = cfg.post.pedal_offset_threshold   # 0.2

    def output_dict_to_midi_events(self, output_dict):
        """Main function. Post process model outputs to MIDI events.

        Args:
          output_dict: {
            'reg_onset_output': (segment_frames, classes_num), 
            'reg_offset_output': (segment_frames, classes_num), 
            'frame_output': (segment_frames, classes_num), 
            'velocity_output': (segment_frames, classes_num), 
            'reg_pedal_onset_output': (segment_frames, 1), 
            'reg_pedal_offset_output': (segment_frames, 1), 
            'pedal_frame_output': (segment_frames, 1)}

        Outputs:
          est_note_events: list of dict, e.g. [
            {'onset_time': 39.74, 'offset_time': 39.87, 'midi_note': 27, 'velocity': 83}, 
            {'onset_time': 11.98, 'offset_time': 12.11, 'midi_note': 33, 'velocity': 88}]

          est_pedal_events: list of dict, e.g. [
            {'onset_time': 0.17, 'offset_time': 0.96}, 
            {'osnet_time': 1.17, 'offset_time': 2.65}]
        """

        # Post process piano note outputs to piano note and pedal events information
        (est_on_off_note_vels, est_pedal_on_offs) = \
            self.output_dict_to_note_pedal_arrays(output_dict)
        """est_on_off_note_vels: (events_num, 4), the four columns are: [onset_time, offset_time, piano_note, velocity], 
        est_pedal_on_offs: (pedal_events_num, 2), the two columns are: [onset_time, offset_time]"""

        # Reformat notes to MIDI events
        est_note_events = self.detected_notes_to_events(est_on_off_note_vels)

        if est_pedal_on_offs is None:
            est_pedal_events = None
        else:
            est_pedal_events = self.detected_pedals_to_events(est_pedal_on_offs)

        return est_note_events, est_pedal_events

    def output_dict_to_note_pedal_arrays(self, output_dict):
        """Postprocess the output probabilities of a transription model to MIDI 
        events.

        Args:
          output_dict: dict, {
            'reg_onset_output': (frames_num, classes_num), 
            'reg_offset_output': (frames_num, classes_num), 
            'frame_output': (frames_num, classes_num), 
            'velocity_output': (frames_num, classes_num), 
            ...}

        Returns:
          est_on_off_note_vels: (events_num, 4), the 4 columns are onset_time, 
            offset_time, piano_note and velocity. E.g. [
             [39.74, 39.87, 27, 0.65], 
             [11.98, 12.11, 33, 0.69], 
             ...]

          est_pedal_on_offs: (pedal_events_num, 2), the 2 columns are onset_time 
            and offset_time. E.g. [
             [0.17, 0.96], 
             [1.17, 2.65], 
             ...]
        """

        # ------ 1. Process regression outputs to binarized outputs ------
        # For example, onset or offset of [0., 0., 0.15, 0.30, 0.40, 0.35, 0.20, 0.05, 0., 0.]
        # will be processed to [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]

        # Calculate binarized onset output from regression output
        (onset_output, onset_shift_output) = \
            self.get_binarized_output_from_regression(
                reg_output=output_dict['reg_onset_output'], 
                threshold=self.onset_threshold, neighbour=2)

        output_dict['onset_output'] = onset_output  # Values are 0 or 1
        output_dict['onset_shift_output'] = onset_shift_output  

        # Calculate binarized offset output from regression output
        (offset_output, offset_shift_output) = \
            self.get_binarized_output_from_regression(
                reg_output=output_dict['reg_offset_output'], 
                threshold=self.offset_threshold, neighbour=4)

        output_dict['offset_output'] = offset_output  # Values are 0 or 1
        output_dict['offset_shift_output'] = offset_shift_output

        if 'reg_pedal_onset_output' in output_dict.keys():
            """Pedal onsets are not used in inference. Instead, frame-wise pedal
            predictions are used to detect onsets. We empirically found this is 
            more accurate to detect pedal onsets."""
            pass

        if 'reg_pedal_offset_output' in output_dict.keys():
            # Calculate binarized pedal offset output from regression output
            (pedal_offset_output, pedal_offset_shift_output) = \
                self.get_binarized_output_from_regression(
                    reg_output=output_dict['reg_pedal_offset_output'], 
                    threshold=self.pedal_offset_threshold, neighbour=4)

            output_dict['pedal_offset_output'] = pedal_offset_output  # Values are 0 or 1
            output_dict['pedal_offset_shift_output'] = pedal_offset_shift_output

        # ------ 2. Process matrices results to event results ------
        # Detect piano notes from output_dict
        est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict)

        if 'reg_pedal_onset_output' in output_dict.keys():
            # Detect piano pedals from output_dict
            est_pedal_on_offs = self.output_dict_to_detected_pedals(output_dict)
 
        else:
            est_pedal_on_offs = None    

        return est_on_off_note_vels, est_pedal_on_offs

    def get_binarized_output_from_regression(self, reg_output, threshold, neighbour):
        """Calculate binarized output and shifts of onsets or offsets from the
        regression results.

        Args:
          reg_output: (frames_num, classes_num)
          threshold: float
          neighbour: int

        Returns:
          binary_output: (frames_num, classes_num)
          shift_output: (frames_num, classes_num)
        """
        binary_output = np.zeros_like(reg_output)
        shift_output = np.zeros_like(reg_output)
        (frames_num, classes_num) = reg_output.shape
        
        for k in range(classes_num):
            x = reg_output[:, k]
            for n in range(neighbour, frames_num - neighbour):
                if x[n] > threshold and self.is_monotonic_neighbour(x, n, neighbour):
                    binary_output[n, k] = 1

                    """See Section III-D in [1] for deduction.
                    [1] Q. Kong, et al., High-resolution Piano Transcription 
                    with Pedals by Regressing Onsets and Offsets Times, 2020."""
                    if x[n - 1] > x[n + 1]:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
                    else:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                    shift_output[n, k] = shift

        return binary_output, shift_output

    def is_monotonic_neighbour(self, x, n, neighbour):
        """Detect if values are monotonic in both side of x[n].

        Args:
          x: (frames_num,)
          n: int
          neighbour: int

        Returns:
          monotonic: bool
        """
        monotonic = True
        for i in range(neighbour):
            if x[n - i] < x[n - i - 1]:
                monotonic = False
            if x[n + i] < x[n + i + 1]:
                monotonic = False

        return monotonic

    def output_dict_to_detected_notes(self, output_dict):
        """Postprocess output_dict to piano notes.

        Args:
          output_dict: dict, e.g. {
            'onset_output': (frames_num, classes_num),
            'onset_shift_output': (frames_num, classes_num),
            'offset_output': (frames_num, classes_num),
            'offset_shift_output': (frames_num, classes_num),
            'frame_output': (frames_num, classes_num),
            'onset_output': (frames_num, classes_num),
            ...}

        Returns:
          est_on_off_note_vels: (notes, 4), the four columns are onsets, offsets, 
          MIDI notes and velocities. E.g.,
            [[39.7375, 39.7500, 27., 0.6638],
             [11.9824, 12.5000, 33., 0.6892],
             ...]
        """
        est_tuples = []
        est_midi_notes = []
        classes_num = output_dict['frame_output'].shape[-1]
 
        for piano_note in range(classes_num):
            """Detect piano notes"""
            est_tuples_per_note = note_detection_with_onset_offset_regress(
                frame_output=output_dict['frame_output'][:, piano_note], 
                onset_output=output_dict['onset_output'][:, piano_note], 
                onset_shift_output=output_dict['onset_shift_output'][:, piano_note], 
                offset_output=output_dict['offset_output'][:, piano_note], 
                offset_shift_output=output_dict['offset_shift_output'][:, piano_note], 
                velocity_output=output_dict['velocity_output'][:, piano_note], 
                frame_threshold=self.frame_threshold)
            
            est_tuples += est_tuples_per_note
            est_midi_notes += [piano_note + self.begin_note] * len(est_tuples_per_note)

        est_tuples = np.array(est_tuples)   # (notes, 5)
        """(notes, 5), the five columns are onset, offset, onset_shift, 
        offset_shift and normalized_velocity"""

        est_midi_notes = np.array(est_midi_notes) # (notes,)

        onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
        offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
        velocities = est_tuples[:, 4]
        
        est_on_off_note_vels = np.stack((onset_times, offset_times, est_midi_notes, velocities), axis=-1)
        """(notes, 3), the three columns are onset_times, offset_times and velocity."""

        est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)

        return est_on_off_note_vels

    def output_dict_to_detected_pedals(self, output_dict):
        """Postprocess output_dict to piano pedals.

        Args:
          output_dict: dict, e.g. {
            'pedal_frame_output': (frames_num,),
            'pedal_offset_output': (frames_num,),
            'pedal_offset_shift_output': (frames_num,),
            ...}

        Returns:
          est_on_off: (notes, 2), the two columns are pedal onsets and pedal
            offsets. E.g.,
              [[0.1800, 0.9669],
               [1.1400, 2.6458],
               ...]
        """
        frames_num = output_dict['pedal_frame_output'].shape[0]
        
        est_tuples = pedal_detection_with_onset_offset_regress(
            frame_output=output_dict['pedal_frame_output'][:, 0], 
            offset_output=output_dict['pedal_offset_output'][:, 0], 
            offset_shift_output=output_dict['pedal_offset_shift_output'][:, 0], 
            frame_threshold=0.5)

        est_tuples = np.array(est_tuples)
        """(notes, 2), the two columns are pedal onsets and pedal offsets"""
        
        if len(est_tuples) == 0:
            return np.array([])

        else:
            onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
            offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
            est_on_off = np.stack((onset_times, offset_times), axis=-1)
            est_on_off = est_on_off.astype(np.float32)
            return est_on_off

    def detected_notes_to_events(self, est_on_off_note_vels):
        """Reformat detected notes to midi events.

        Args:
          est_on_off_vels: (notes, 3), the three columns are onset_times, 
            offset_times and velocity. E.g.
            [[32.8376, 35.7700, 0.7932],
             [37.3712, 39.9300, 0.8058],
             ...]
        
        Returns:
          midi_events, list, e.g.,
            [{'onset_time': 39.7376, 'offset_time': 39.75, 'midi_note': 27, 'velocity': 84},
             {'onset_time': 11.9824, 'offset_time': 12.50, 'midi_note': 33, 'velocity': 88},
             ...]
        """
        midi_events = []
        for i in range(est_on_off_note_vels.shape[0]):
            midi_events.append({
                'onset_time': est_on_off_note_vels[i][0], 
                'offset_time': est_on_off_note_vels[i][1], 
                'midi_note': int(est_on_off_note_vels[i][2]), 
                'velocity': int(est_on_off_note_vels[i][3] * self.velocity_scale)})

        return midi_events

    def detected_pedals_to_events(self, pedal_on_offs):
        """Reformat detected pedal onset and offsets to events.

        Args:
          pedal_on_offs: (notes, 2), the two columns are pedal onsets and pedal
          offsets. E.g., 
            [[0.1800, 0.9669],
             [1.1400, 2.6458],
             ...]

        Returns:
          pedal_events: list of dict, e.g.,
            [{'onset_time': 0.1800, 'offset_time': 0.9669}, 
             {'onset_time': 1.1400, 'offset_time': 2.6458},
             ...]
        """
        pedal_events = []
        for i in range(len(pedal_on_offs)):
            pedal_events.append({
                'onset_time': pedal_on_offs[i, 0], 
                'offset_time': pedal_on_offs[i, 1]})
        
        return pedal_events


class OnsetsFramesPostProcessor(object):
    def __init__(self, cfg):
        """Postprocess the Googl's onsets and frames system output. Only used
        for comparison.

        Args:
          frames_per_second: int
          classes_num: int
        """
        self.frames_per_second = cfg.feature.frames_per_second
        self.classes_num = cfg.feature.classes_num
        self.begin_note = cfg.feature.begin_note
        self.velocity_scale = cfg.feature.velocity_scale
        
        self.frame_threshold = cfg.post.frame_threshold                 # 0.5
        self.onset_threshold = cfg.post.onset_threshold                 # 0.1
        self.offset_threshold = cfg.post.offset_threshold               # 0.3

    def output_dict_to_midi_events(self, output_dict):
        """Main function. Post process model outputs to MIDI events.

        Args:
          output_dict: {
            'reg_onset_output': (segment_frames, classes_num), 
            'reg_offset_output': (segment_frames, classes_num), 
            'frame_output': (segment_frames, classes_num), 
            'velocity_output': (segment_frames, classes_num), 
            'reg_pedal_onset_output': (segment_frames, 1), 
            'reg_pedal_offset_output': (segment_frames, 1), 
            'pedal_frame_output': (segment_frames, 1)}

        Outputs:
          est_note_events: list of dict, e.g. [
            {'onset_time': 39.74, 'offset_time': 39.87, 'midi_note': 27, 'velocity': 83}, 
            {'onset_time': 11.98, 'offset_time': 12.11, 'midi_note': 33, 'velocity': 88}]

          est_pedal_events: list of dict, e.g. [
            {'onset_time': 0.17, 'offset_time': 0.96}, 
            {'osnet_time': 1.17, 'offset_time': 2.65}]
        """

        # Post process piano note outputs to piano note and pedal events information
        (est_on_off_note_vels, est_pedal_on_offs) = \
            self.output_dict_to_note_pedal_arrays(output_dict)
        """est_on_off_note_vels: (events_num, 4), the four columns are: [onset_time, offset_time, piano_note, velocity], 
        est_pedal_on_offs: (pedal_events_num, 2), the two columns are: [onset_time, offset_time]"""
        
        # Reformat notes to MIDI events
        est_note_events = self.detected_notes_to_events(est_on_off_note_vels)

        if est_pedal_on_offs is None:
            est_pedal_events = None
        else:
            est_pedal_events = self.detected_pedals_to_events(est_pedal_on_offs)

        return est_note_events, est_pedal_events

    def output_dict_to_note_pedal_arrays(self, output_dict):
        """Postprocess the output probabilities of a transription model to MIDI 
        events.

        Args:
          output_dict: dict, {
            'reg_onset_output': (frames_num, classes_num), 
            'reg_offset_output': (frames_num, classes_num), 
            'frame_output': (frames_num, classes_num), 
            'velocity_output': (frames_num, classes_num), 
            ...}

        Returns:
          est_on_off_note_vels: (events_num, 4), the 4 columns are onset_time, 
            offset_time, piano_note and velocity. E.g. [
             [39.74, 39.87, 27, 0.65], 
             [11.98, 12.11, 33, 0.69], 
             ...]

          est_pedal_on_offs: (pedal_events_num, 2), the 2 columns are onset_time 
            and offset_time. E.g. [
             [0.17, 0.96], 
             [1.17, 2.65], 
             ...]
        """

        # Sharp onsets and offsets
        output_dict = self.sharp_output_dict(
            output_dict, onset_threshold=self.onset_threshold, 
            offset_threshold=self.offset_threshold)

        # Post process output_dict to piano notes
        est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict)

        if 'reg_pedal_onset_output' in output_dict.keys():
            # Detect piano pedals from output_dict
            est_pedal_on_offs = self.output_dict_to_detected_pedals(output_dict)
 
        else:
            est_pedal_on_offs = None    

        return est_on_off_note_vels, est_pedal_on_offs

    def sharp_output_dict(self, output_dict, onset_threshold, offset_threshold):
        """Sharp onsets and offsets. E.g. when threshold=0.3, for a note, 
        [0, 0.1, 0.4, 0.7, 0, 0] will be sharped to [0, 0, 0, 1, 0, 0]
        [0., 0., 1., 0., 0., 0.]

        Args:
          output_dict: {
            'reg_onset_output': (frames_num, classes_num), 
            'reg_offset_output': (frames_num, classes_num), 
            ...}
          onset_threshold: float
          offset_threshold: float

        Returns:
          output_dict: {
            'onset_output': (frames_num, classes_num), 
            'offset_output': (frames_num, classes_num)}
        """
        if 'reg_onset_output' in output_dict.keys():
            output_dict['onset_output'] = self.sharp_output(
                output_dict['reg_onset_output'], 
                threshold=onset_threshold)

        if 'reg_offset_output' in output_dict.keys():
            output_dict['offset_output'] = self.sharp_output(
                output_dict['reg_offset_output'], 
                threshold=offset_threshold)

        return output_dict

    def sharp_output(self, input, threshold=0.3):
        """Used for sharping onset or offset. E.g. when threshold=0.3, for a note, 
        [0, 0.1, 0.4, 0.7, 0, 0] will be sharped to [0, 0, 0, 1, 0, 0]

        Args:
          input: (frames_num, classes_num)

        Returns:
          output: (frames_num, classes_num)
        """
        (frames_num, classes_num) = input.shape
        output = np.zeros_like(input)

        for piano_note in range(classes_num):
            for i in range(1, frames_num - 1):
                # Check if the current frame is a peak
                if input[i, piano_note] > threshold and input[i, piano_note] > input[i - 1, piano_note] and input[i, piano_note] > input[i + 1, piano_note]:
                    output[i, piano_note] = 1

        return output

    def output_dict_to_detected_notes(self, output_dict):
        """Postprocess output_dict to piano notes.

        Args:
          output_dict: dict, e.g. {
            'onset_output': (frames_num, classes_num),
            'onset_shift_output': (frames_num, classes_num),
            'offset_output': (frames_num, classes_num),
            'offset_shift_output': (frames_num, classes_num),
            'frame_output': (frames_num, classes_num),
            'onset_output': (frames_num, classes_num),
            ...}

        Returns:
          est_on_off_note_vels: (notes, 4), the four columns are onsets, offsets, 
          MIDI notes and velocities. E.g.,
            [[39.7375, 39.7500, 27., 0.6638],
             [11.9824, 12.5000, 33., 0.6892],
             ...]
        """

        est_tuples = []
        est_midi_notes = []

        for piano_note in range(self.classes_num):
            
            est_tuples_per_note = onsets_frames_note_detection(
                frame_output=output_dict['frame_output'][:, piano_note], 
                onset_output=output_dict['onset_output'][:, piano_note], 
                offset_output=output_dict['offset_output'][:, piano_note], 
                velocity_output=output_dict['velocity_output'][:, piano_note], 
                threshold=self.frame_threshold)

            est_tuples += est_tuples_per_note
            est_midi_notes += [piano_note + self.begin_note] * len(est_tuples_per_note)

        est_tuples = np.array(est_tuples)   # (notes, 3)
        """(notes, 5), the five columns are onset, offset, onset_shift, 
        offset_shift and normalized_velocity"""

        est_midi_notes = np.array(est_midi_notes) # (notes,)
        
        if len(est_midi_notes) == 0:
            return []
        else:
            onset_times = est_tuples[:, 0] / self.frames_per_second
            offset_times = est_tuples[:, 1] / self.frames_per_second
            velocities = est_tuples[:, 2]
        
            est_on_off_note_vels = np.stack((onset_times, offset_times, est_midi_notes, velocities), axis=-1)
            """(notes, 3), the three columns are onset_times, offset_times and velocity."""

            est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)

            return est_on_off_note_vels

    def output_dict_to_detected_pedals(self, output_dict):
        """Postprocess output_dict to piano pedals.

        Args:
          output_dict: dict, e.g. {
            'pedal_frame_output': (frames_num,),
            'pedal_offset_output': (frames_num,),
            'pedal_offset_shift_output': (frames_num,),
            ...}

        Returns:
          est_on_off: (notes, 2), the two columns are pedal onsets and pedal
            offsets. E.g.,
              [[0.1800, 0.9669],
               [1.1400, 2.6458],
               ...]
        """

        frames_num = output_dict['pedal_frame_output'].shape[0]
        
        est_tuples = onsets_frames_pedal_detection(
            frame_output=output_dict['pedal_frame_output'][:, 0], 
            offset_output=output_dict['reg_pedal_offset_output'][:, 0], 
            frame_threshold=0.5)

        est_tuples = np.array(est_tuples)
        """(notes, 2), the two columns are pedal onsets and pedal offsets"""
        
        if len(est_tuples) == 0:
            return np.array([])

        else:
            onset_times = est_tuples[:, 0] / self.frames_per_second
            offset_times = est_tuples[:, 1] / self.frames_per_second
            est_on_off = np.stack((onset_times, offset_times), axis=-1)
            est_on_off = est_on_off.astype(np.float32)
            return est_on_off

    def detected_notes_to_events(self, est_on_off_note_vels):
        """Reformat detected notes to midi events.

        Args:
          est_on_off_vels: (notes, 3), the three columns are onset_times, 
            offset_times and velocity. E.g.
            [[32.8376, 35.7700, 0.7932],
             [37.3712, 39.9300, 0.8058],
             ...]
        
        Returns:
          midi_events, list, e.g.,
            [{'onset_time': 39.7376, 'offset_time': 39.75, 'midi_note': 27, 'velocity': 84},
             {'onset_time': 11.9824, 'offset_time': 12.50, 'midi_note': 33, 'velocity': 88},
             ...]
        """
        midi_events = []
        for i in range(len(est_on_off_note_vels)):
            midi_events.append({
                'onset_time': est_on_off_note_vels[i][0], 
                'offset_time': est_on_off_note_vels[i][1], 
                'midi_note': int(est_on_off_note_vels[i][2]), 
                'velocity': int(est_on_off_note_vels[i][3] * self.velocity_scale)})

        return midi_events

    def detected_pedals_to_events(self, pedal_on_offs):
        """Reformat detected pedal onset and offsets to events.

        Args:
          pedal_on_offs: (notes, 2), the two columns are pedal onsets and pedal
          offsets. E.g., 
            [[0.1800, 0.9669],
             [1.1400, 2.6458],
             ...]

        Returns:
          pedal_events: list of dict, e.g.,
            [{'onset_time': 0.1800, 'offset_time': 0.9669}, 
             {'onset_time': 1.1400, 'offset_time': 2.6458},
             ...]
        """
        pedal_events = []
        for i in range(len(pedal_on_offs)):
            pedal_events.append({
                'onset_time': pedal_on_offs[i, 0], 
                'offset_time': pedal_on_offs[i, 1]})
        
        return pedal_events
