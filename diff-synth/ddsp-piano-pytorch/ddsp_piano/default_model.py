from ddsp_piano.modules import sub_modules
from ddsp_piano.modules.piano_model import PianoModel

from ddsp_piano.modules.inharm_synth import MultiInharmonic
from ddsp_piano.ddsp_pytorch.noise import Noise
from ddsp_piano.ddsp_pytorch.reverb import Reverb


def build_polyphonic_ddsp_module(
    sample_rate=22050,
    duration=10,
    inference=False):
    """ Polyphonic bank of additive + filtered noise synthesizers.
    Args:
        - sample_rate (int): number of samples per second.
        - duration (float): length of generated sample (in seconds).
        - inference (bool): synthesis for inference (slower but can handle
        longer sequences).
    Returns:
        - ddsp_module 
    """
    n_samples = int(duration * sample_rate)

    # Init Harmonic + Noise Synthesizers
    harmonic_synthesizer = MultiInharmonic(n_samples, sample_rate, inference=inference)
    noise_synthesizer = Noise()
    reverb_effects = Reverb()

    return harmonic_synthesizer, noise_synthesizer, reverb_effects

def get_model(
    inference=False,
    duration=10,
    n_synths=16,
    n_substrings=2,
    n_piano_models=10,
    piano_embedding_dim=16,
    n_noise_filter_banks=64,
    frame_rate=100,
    sample_rate=22050,
    reverb_duration=1.5):
    # Self-contained sub-modules
    z_encoder = sub_modules.OneHotZEncoder(n_instruments=n_piano_models, z_dim=piano_embedding_dim, n_frames=int(duration * frame_rate))
    note_release = sub_modules.NoteRelease(frame_rate=frame_rate)
    parallelizer = sub_modules.Parallelizer(n_synths=n_synths)
    inharm_model = sub_modules.InharmonicityNetwork()
    detuner = sub_modules.Detuner(n_substrings=n_substrings)
    reverb_model = sub_modules.MultiInstrumentReverb(
        n_instruments=n_piano_models,
        reverb_length=int(reverb_duration * sample_rate)
    )

    # Neural modules
    context_network = sub_modules.ContextNetwork()

    monophonic_network = sub_modules.MonophonicNetwork()

    harmonic_synthesizer, noise_synthesizer, reverb_module = build_polyphonic_ddsp_module(
        sample_rate=sample_rate,
        duration=duration,
        inference=inference
    )

    # Full piano model definition
    model = PianoModel(
        n_synths=n_synths,
        z_encoder=z_encoder,
        note_release=note_release,
        context_network=context_network,
        parallelizer=parallelizer,
        monophonic_network=monophonic_network,
        inharm_model=inharm_model,
        detuner=detuner,
        reverb_model=reverb_model,
        harmonic_synthesizer=harmonic_synthesizer,
        noise_synthesizer=noise_synthesizer,
        reverb_module=reverb_module
    )

    return model
