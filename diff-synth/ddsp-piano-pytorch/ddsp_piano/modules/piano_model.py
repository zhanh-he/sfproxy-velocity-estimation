import torch
import torch.nn as nn 

import soundfile as sf

class PianoModel(nn.Module):
    """DDSP model for piano synthesis from MIDI conditioning.
    Args:
        - z_encoder: one-hot piano model embeddings.
        - note_release: extend active pitch conditioning.
        - context_network: context vector computation model from
        global inputs.
        - parallelizer: layer managing polyphony and batch axis
        merge and unmerge.
        - monophonic_network: monophonic string model as
        neural network.
        - inharm_model: inharmonicity model over tessitura.
        - detuner: tuning model for pitch to absolute f0
        frequency.
        - reverb_model: recording environment impulse responses.
    """
    def __init__(
        self,
        n_synths=16,
        z_encoder=None,
        note_release=None,
        context_network=None,
        parallelizer=None,
        monophonic_network=None,
        inharm_model=None,
        detuner=None,
        reverb_model=None,
        harmonic_synthesizer=None,
        noise_synthesizer=None,
        reverb_module=None):
        super(PianoModel, self).__init__()
        self.n_synths = n_synths
        self.z_encoder = z_encoder # num params ok 
        self.note_release = note_release # num params ok 
        self.context_network = context_network # num params ok
        self.parallelizer = parallelizer # num params ok
        self.monophonic_network = monophonic_network # num params ok
        self.inharm_model = inharm_model # num params ok
        self.detuner = detuner # num params ok
        self.harmonic_synthesizer = harmonic_synthesizer # num params ok
        self.noise_synthesizer = noise_synthesizer # num params ok
        self.reverb_module = reverb_module  # num params ok
        self.reverb_model = reverb_model # num params ok
        
    def alternate_training(self, first_phase=True):
        """Toggle trainability of submodules for the 1st or 2nd training phase.
        Args:
            - first_phase (bool): whether using the 1st phase training strategy
        """
        # Modules involved with partial frequency computing are frozen during
        # the first training phase strategy.
        for module in [self.inharm_model, self.detuner]:
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = not first_phase
        
        self.z_encoder.alternate_training(first_phase)

        # Modules not involved in freq computing have inversed trainability
        for module in [self.note_release,
                       self.context_network,
                       self.monophonic_network,
                       self.reverb_model]:
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = first_phase
        
        # Compute multiple note signals only when learning detuner weights
        self.detuner.use_detune = not first_phase

    def synthesize_harmonic_part(self, harmonic_synthesizer, amplitudes, harmonic_distribution, inharm_coef, f0_hz):
        params = harmonic_synthesizer.get_controls(amplitudes, harmonic_distribution, inharm_coef, f0_hz)
        harmonic_signal = harmonic_synthesizer(params["amplitudes"],
                                               params["harmonic_distribution"],
                                               params["harmonic_shifts"],
                                               params["f0_hz"])
        return harmonic_signal

    def forward(
        self,
        conditioning,
        pedal, 
        piano_model):

        # compute global feature 
        if self.z_encoder is not None:
            z, global_inharm, global_detuning = self.z_encoder(piano_model)
        if self.context_network is not None:
            context = self.context_network(conditioning, pedal, z)
        if self.reverb_model is not None:
            reverb_ir = self.reverb_model(piano_model.unsqueeze(-1))

        # parallel 
        if self.parallelizer is not None:
            conditioning, context, global_inharm, global_detuning = self.parallelizer(conditioning, context, global_inharm, global_detuning)
        
        # compute monophonic feature 
        if self.note_release is not None:
            extended_pitch = self.note_release(conditioning)

        if self.inharm_model is not None:
            inharm_coef = self.inharm_model(extended_pitch, global_inharm)
        
        if self.detuner is not None:
            f0_hz = self.detuner(extended_pitch, global_detuning)
        #print('f0_hz shape: ', f0_hz.shape)
        if self.monophonic_network is not None:
            amplitudes, harmonic_distribution, magnitudes = self.monophonic_network(conditioning, extended_pitch, context)


        # unparallel 
        # Disentangle polyphony and batch axis
        if self.parallelizer is not None:
            features = self.parallelizer(
                f0_hz=f0_hz, 
                inharm_coef=inharm_coef, 
                amplitudes=amplitudes, 
                harmonic_distribution=harmonic_distribution, 
                magnitudes=magnitudes, 
                parallelize=False)

        
        ########### DDSP
        #print('before f0_hz shape: ', f0_hz.shape)
        amplitudes, harmonic_distribution, inharm_coef, f0_hz, magnitudes = features[f"amplitudes_0"], features[f"harmonic_distribution_0"], features[f"inharm_coef_0"], features[f"f0_hz_0"], features[f"magnitudes_0"]
        #print('after f0_hz shape: ', f0_hz.shape)
        harmonic_part = self.synthesize_harmonic_part(self.harmonic_synthesizer, amplitudes, harmonic_distribution, inharm_coef, f0_hz) 
        noise_part = self.noise_synthesizer(harmonic_part, magnitudes)
        signal = harmonic_part + noise_part
        for i in range(1, self.n_synths):
            amplitudes, harmonic_distribution, inharm_coef, f0_hz, magnitudes = features[f"amplitudes_{i}"], features[f"harmonic_distribution_{i}"], features[f"inharm_coef_{i}"], features[f"f0_hz_{i}"], features[f"magnitudes_{i}"]
            sub_harmonic = self.synthesize_harmonic_part(self.harmonic_synthesizer, amplitudes, harmonic_distribution, inharm_coef, f0_hz)
            sub_noise = self.noise_synthesizer(sub_harmonic, magnitudes)
            signal += (sub_harmonic + sub_noise)
        
        non_ir_signal = signal.detach()
        signal = self.reverb_module(signal, reverb_ir)
        return signal, reverb_ir, non_ir_signal

            
