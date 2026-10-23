import os
import librosa 
import note_seq
import numpy as np
from pandas import read_csv
from tqdm import tqdm

import torch

from ddsp_piano.utils.midi_encoders import MIDIRoll2Conditioning

seq_lib = note_seq.sequences_lib

def load_midi_as_note_sequence(mid_path):
    # Read MIDI file
    note_sequence = note_seq.midi_io.midi_file_to_note_sequence(mid_path)
    # Extend offset with sustain pedal
    note_sequence = note_seq.apply_sustain_control_changes(note_sequence)
    return note_sequence


def load_data(
    audio_path,
    midi_path,
    segment_duration=3.,
    max_polyphony=None,
    overlap=0.5,
    sample_rate=16000,
    frame_rate=250):
    """
    Load aligned audio and MIDI data (as conditioning sequence), then split
    into segments.
        Args:
            - audio_path (path): path to audio file.
            - midi_path (path): path to midi file.
            - segment_duration (float): length of segment chunks (in s).
            - max_polyphony (int): number of monophonic channels for the conditio-
            ning vector (return the piano rolls if None).
            - overlap (float): overlapping ratio between two consecutive segments.
            - sample_rate (int): number of audio samples per second.
            - frame_rate (int): number of conditioning vectors per second.
        Returns:
            - segment_audio (list [n_samples,]): list of audio segments.
            - segment_rolls (list [n_frames, max_polyphony, 2]): list of segments
            conditioning vectors.
            - segment_pedals (list [n_frames, 4]): list of segments pedals condi-
            tioning.
            - polyphony (list [n_frames, 1]): list of polyphony information in the
            original piano roll.
    """
    n_samples = int(segment_duration * sample_rate)
    n_frames = int(segment_duration * frame_rate)
    audio_hop_size = int(n_samples * (1 - overlap))
    midi_hop_size = int(n_frames * (1 - overlap))

    # Read audio file
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # Read MIDI file
    note_sequence = load_midi_as_note_sequence(
        midi_path
    )
    # Convert to pianoroll
    roll = seq_lib.sequence_to_pianoroll(note_sequence,
                                        frames_per_second=frame_rate,
                                        min_pitch=21,
                                        max_pitch=108)
    # Retrieve activity and onset velocities
    midi_roll = np.stack((roll.active, roll.onset_velocities), axis=-1)

    # Pedals are CC64, 66 and 67
    pedals = roll.control_changes[:, 64: 68] / 128.0
    if max_polyphony is not None:
        polyphony_manager = MIDIRoll2Conditioning(max_polyphony)
        midi_roll, polyphony = polyphony_manager(midi_roll)
    
    # Split into segments
    audio_t = 0
    midi_t = 0
    segment_audio = []
    segment_rolls = []
    segment_pedals = []
    segment_polyphony = []
    while midi_t + n_frames < np.shape(midi_roll)[0]:
        segment_audio.append(audio[audio_t: audio_t + n_samples])
        segment_rolls.append(midi_roll[midi_t: midi_t + n_frames])
        segment_pedals.append(pedals[midi_t: midi_t + n_frames])    
        if max_polyphony:
            segment_polyphony.append(polyphony[midi_t: midi_t + n_frames])

        audio_t += audio_hop_size
        midi_t += midi_hop_size

    n_segments = len(segment_rolls)
    if max_polyphony is None:
        return np.array(segment_audio), np.array(segment_rolls), np.array(segment_pedals), n_segments
    else:
        return np.array(segment_audio), np.array(segment_rolls), np.array(segment_pedals), np.array(segment_polyphony), n_segments

CACHE_FILES = ['audio.npy', 'conditioning.npy', 'pedal.npy', 'polyphony.npy', 'piano_model.npy']


def build_cache_path(cache_root, split, audio_rel_path):
    rel_audio_path = os.path.splitext(audio_rel_path)[0]
    return os.path.join(cache_root, split, rel_audio_path)


def check_files_exist(path):
    for f in CACHE_FILES:
        if not os.path.exists(os.path.join(path, f)):
            return False
    return True

def save_files(path, audio, conditioning, pedal, polyphony, piano_model):
    np.save(os.path.join(path, 'audio.npy'), audio)
    np.save(os.path.join(path, 'conditioning.npy'), conditioning)
    np.save(os.path.join(path, 'pedal.npy'), pedal)
    np.save(os.path.join(path, 'polyphony.npy'), polyphony)
    np.save(os.path.join(path, 'piano_model.npy'), piano_model)

def read_data_from_cache(path):
    return (
        np.load(os.path.join(path, 'audio.npy'), mmap_mode='r'),
        np.load(os.path.join(path, 'conditioning.npy'), mmap_mode='r'),
        np.load(os.path.join(path, 'pedal.npy'), mmap_mode='r'),
        np.load(os.path.join(path, 'polyphony.npy'), mmap_mode='r'),
        np.load(os.path.join(path, 'piano_model.npy'))
    )

class MaestroDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        dataframe,
        piano_models,
        max_polyphony=16,
        split='train',
        cache_data_path=None,
        device='cpu'):
        """
        dataframe: metadata read from csv file
        piano_models: different years of piano ex. 2011
        split: train, valid, test
        """
        if cache_data_path is None:
            raise ValueError('cache_data_path is required. Run the preprocessing step first.')

        self.split = split
        self.dataframe = dataframe
        self.cache_data_path = cache_data_path
        self.piano_models = torch.tensor(piano_models, requires_grad=False)
        self.max_polyphony = max_polyphony
        self.segment_index = []
        missing_caches = []

        for _, row in tqdm(self.dataframe.items(), desc='Indexing %s split' % self.split):
            audio_rel_path = row['audio_filename']
            cache_dir = build_cache_path(self.cache_data_path, self.split, audio_rel_path)
            if not check_files_exist(cache_dir):
                missing_caches.append(cache_dir)
                continue

            piano_model_from_cache = np.load(os.path.join(cache_dir, 'piano_model.npy'))
            polyphony = np.load(os.path.join(cache_dir, 'polyphony.npy'), mmap_mode='r')
            num_segments = polyphony.shape[0]

            if self.max_polyphony is None:
                valid_indices = np.arange(num_segments)
            else:
                current_maximum_polyphony = np.max(polyphony, axis=1)
                valid_indices = np.where(current_maximum_polyphony <= self.max_polyphony)[0]
            del polyphony

            for segment_idx in valid_indices:
                self.segment_index.append(
                    dict(
                        cache_dir=cache_dir,
                        segment_idx=int(segment_idx),
                        piano_model=piano_model_from_cache
                    )
                )

        if missing_caches:
            preview = '\n'.join(sorted(set(missing_caches))[:5])
            raise RuntimeError(
                'Missing cached data for split {split}. Please run preprocess.py before training. Example paths:\n{paths}'.format(
                    split=self.split,
                    paths=preview
                )
            )

    def __len__(self):
        return len(self.segment_index)
    
    def __getitem__(self, index):
        data = self.segment_index[index]
        cache_dir = data['cache_dir']
        segment_idx = data['segment_idx']

        audio = self._load_segment(cache_dir, 'audio', segment_idx)
        conditioning = self._load_segment(cache_dir, 'conditioning', segment_idx)
        pedal = self._load_segment(cache_dir, 'pedal', segment_idx)
        piano_model = data['piano_model']

        # Encode piano model as one-hot
        piano_model_tensor = torch.from_numpy(piano_model).int()
        piano_model_one_hot = torch.where(torch.eq(self.piano_models, piano_model_tensor))[0]
        
        return (
            torch.from_numpy(audio).float(),
            torch.from_numpy(conditioning).float(),
            torch.from_numpy(pedal).float(),
            piano_model_one_hot[0]
        )

    def _load_segment(self, cache_dir, filename, segment_idx):
        path = os.path.join(cache_dir, f'{filename}.npy')
        array = np.load(path, mmap_mode='r')
        data = np.array(array[segment_idx])
        del array
        return data


def preprocess_split(dataset_dir,
                     cache_root,
                     split='train',
                     max_polyphony=16,
                     segment_duration=3.,
                     overlap=0.5,
                     sample_rate=16000,
                     frame_rate=250):
    if max_polyphony is None:
        raise ValueError('max_polyphony must be provided when preprocessing caches.')

    csv_path = os.path.join(dataset_dir, 'maestro-v3.0.0.csv')
    df = read_csv(csv_path)
    df = df[df.split == split]

    if df.empty:
        return

    records = df.to_dict('records')
    for row in tqdm(records, total=len(records), desc=f'Preprocessing split {split}'):
        audio_rel_path = row['audio_filename']
        midi_rel_path = row['midi_filename']
        audio_path = os.path.join(dataset_dir, audio_rel_path)
        midi_path = os.path.join(dataset_dir, midi_rel_path)
        cache_prefix = build_cache_path(cache_root, split, audio_rel_path)
        os.makedirs(cache_prefix, exist_ok=True)

        if check_files_exist(cache_prefix):
            continue

        audio, conditioning, pedal, polyphony, _ = load_data(
            audio_path,
            midi_path,
            segment_duration=segment_duration,
            max_polyphony=max_polyphony,
            overlap=overlap,
            sample_rate=sample_rate,
            frame_rate=frame_rate
        )

        piano_model = np.array(row['year'])
        save_files(cache_prefix, audio, conditioning, pedal, polyphony, piano_model)
        #torch.from_numpy(audio).float(), torch.from_numpy(conditioning).float(), torch.from_numpy(pedal).float(), torch.from_numpy(polyphony).int(), piano_model_one_hot[0]
