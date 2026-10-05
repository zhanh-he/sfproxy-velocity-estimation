import torch 
import torch.nn as nn 
from torch.nn.utils import weight_norm

import numpy as np 

from ddsp_piano.ddsp_pytorch.core import resample, midi_to_hz
from ddsp_piano.modules.inharm_synth import MultiInharmonic
from ddsp_piano.ddsp_pytorch.noise import Noise
from ddsp_piano.ddsp_pytorch.reverb import Reverb

# currently ok
class ContextNetwork(nn.Module):
    """Sequential model for computing the context vector from the global inputs
    """
    def __init__(self):
        super(ContextNetwork, self).__init__()

        self.linear = nn.Linear(52, 32)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.gru = nn.GRU(32, 64, 1, batch_first=True)
        self.layer_norm = nn.LayerNorm(64)
        #self.dense_out = weight_norm(nn.Linear(64, 32)) 
        self.dense_out = nn.Linear(64, 32) # same number of parameters with original ddsp-piano

    def collapse_last_axis(self, x, axis=-2):
        # Merge last axis of the tensor 'x'
        shape = x.shape
        new_shape = (shape[0], shape[1], shape[2] * shape[3])
        return torch.reshape(x, new_shape)

    def forward(self, conditioning, pedal, z):
        """
        Args:
            - conditioning (b, n_frames, 16, 2)
            - pedal (b, n_frames, 4)
            - z (b, n_frames, 16)
        """
        x = torch.cat([self.collapse_last_axis(conditioning), pedal, z], axis=-1) # (b, 750, 52)
        x = self.linear(x) 
        x = self.leaky_relu(x) 
        x, _ = self.gru(x) # h0 init 0 or not???????
        x = self.layer_norm(x)
        x = self.dense_out(x)
        return x #x[:,:,:32] # in the original ddsp-piano, it splits 32 dimension to be the output 

# parallelize part: ok, unparallelize: not yet
class Parallelizer(nn.Module):
    """Module for merging and unmerge the batch and polyphony axis of features.
    Args:
        - n_synths (int): size of polypohny axis.
    """
    def __init__(self, n_synths=16):
        super(Parallelizer, self).__init__()
        self.n_synths = n_synths
    
    def put_polyphony_axis_at_first(self, x):
        """Reshape feature before calling parallelize"""
        if len(x.shape) == 3:
            # Create the polyphony axis and share value over all mono channels
            x = x.repeat(self.n_synths, 1, 1, 1)
        elif len(x.shape) == 4:
            # Put polyphony axis as the first dimension
            x = x.permute(2, 0, 1, 3)
        return x
    
    def parallelize_feature(self, x):
        # Merge the polyphony and batch axis (which are the first two axis)
        shape = x.shape
        new_shape = (shape[0]*shape[1], shape[2], shape[3])
        return torch.reshape(x, new_shape)

    def parallelize_series_operation(self, x):
        return self.parallelize_feature(self.put_polyphony_axis_at_first(x))

    def parallelize(self, conditioning, context, global_inharm, global_detuning):
        return self.parallelize_series_operation(conditioning), self.parallelize_series_operation(context), self.parallelize_series_operation(global_inharm), self.parallelize_series_operation(global_detuning)

    def unparallelize_feature(self, x):
        shape = x.shape
        new_shape = ( self.n_synths, shape[0]//self.n_synths, shape[1], shape[2])
        return torch.reshape(x, new_shape)

    def disentangle(self, x, features, name=None):
        x = self.unparallelize_feature(x)
        #print(f'{name} : ', x.shape)
        for i in range(self.n_synths):
            features[name+f'_{i}'] = x[i]
        return features 

    def unparallelize(self, f0_hz, inharm_coef, amplitudes, harmonic_distribution, magnitudes):
        """Disentangle batch and polyphony axis and distribute features as
        monophonic controls.
        """
        features = dict()
        features = self.disentangle(f0_hz, features, 'f0_hz')
        features = self.disentangle(inharm_coef, features, 'inharm_coef')
        features = self.disentangle(amplitudes, features, 'amplitudes')
        features = self.disentangle(harmonic_distribution, features, 'harmonic_distribution')
        features = self.disentangle(magnitudes, features, 'magnitudes')
        return features

    def forward(self, 
                conditioning=None, 
                context=None, 
                global_inharm=None, 
                global_detuning=None, 
                f0_hz=None, 
                inharm_coef=None, 
                amplitudes=None, 
                harmonic_distribution=None, 
                magnitudes=None,
                parallelize=True):
        if parallelize:
            return self.parallelize(conditioning, context, global_inharm, global_detuning)
        else:# wait for unparallel stage
            return self.unparallelize(f0_hz, inharm_coef, amplitudes, harmonic_distribution, magnitudes)

# according to training strategy, some weights are trainable or not 
class InharmonicityNetwork(nn.Module):
    """ Compute inharmonicity coefficient corresponding to MIDI notes. """
    """ Initialize the MIDI note to inharmonicity coefficient network.
        Initial values are taken from results in F.Rigaud et al. "A Parametric
        Model of Piano tuning", Proc. of DAFx-2011.
        """
    def __init__(self):
        super(InharmonicityNetwork, self).__init__()
        self.midi_norm = 128.

        # Init weights
        treble_slope = 9.26e-2
        treble_intercept = - 13.64

        bass_slope = - 8.47e-2
        bass_intercept = - 5.82

        self.model_specific_weight = torch.nn.Parameter(torch.tensor(0.), requires_grad=True).float()
        
        init_slope_weight = torch.tensor([treble_slope * self.midi_norm, bass_slope * self.midi_norm])
        self.slopes = torch.nn.Parameter(init_slope_weight, requires_grad=False).float()
        
        init_offset_weight = torch.tensor([treble_intercept / (self.midi_norm * treble_slope), bass_intercept / (self.midi_norm * bass_slope)])
        self.offsets = torch.nn.Parameter(init_offset_weight, requires_grad=False).float()
        
        self.slopes_modifier = torch.nn.Parameter(torch.zeros(2), requires_grad=True).float()
        self.offsets_modifier = torch.nn.Parameter(torch.zeros(2), requires_grad=True).float()
        
        self.freeze()
        
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, extended_pitch, global_inharm=None):
        """ Compute inharmonicity coefficient corresponding to input pitch note
        Args:
            - extended_pitch (batch, n_frames, 1): input MIDI note conditioning
            signal.
            - global_inharm (batch, 1, 1): fine-tuning from a specific piano
            model.
        Returns:
            - inharm_coef (batch, n_frames, 1): inharmonicity coefficient.
        """
        reduced_notes = extended_pitch / self.midi_norm
        slopes = self.slopes + self.slopes_modifier
        offsets = self.offsets + self.offsets_modifier

        bridges_asymptotes = slopes * (reduced_notes + offsets)
        # Fine-tuning according to piano model
        if global_inharm is not None:
            # Scaling
            global_inharm *= 10.
            # Only the bass bridge is model specific
            global_inharm = torch.cat(
                [torch.zeros_like(global_inharm), global_inharm],
                axis=-1
            )
            bridges_asymptotes += self.model_specific_weight * global_inharm

        # Compute inharmonicity factor (batch, n_frames, 1)
        # beta = exp(treble_asymp) + exp(bass_asymp)
        inharm_coef = torch.sum(torch.exp(bridges_asymptotes),
                                    axis=-1,
                                    keepdims=True)
        return inharm_coef

# currently ok
class Detuner(nn.Module):
    """ Compute a detuning factor for each input MIDI note.
    Args:
        - n_substrings (int): number of piano strings per note.
        - use_detune (bool): use the predicted detuning for converting MIDI
        pitch to Hz.
    """
    def __init__(self, n_substrings=2, use_detune=True):
        super(Detuner, self).__init__()
        self.n_substrings = n_substrings
        self.use_detune = use_detune

        self.tanh = torch.nn.Tanh()
        self.layer = torch.nn.Linear(1, self.n_substrings)
        self.layer.requires_grad = False

    def forward(self, extended_pitch, global_detuning=None):
        """ Forward pass
        Args:
            - extended_pitch (batch, ..., 1): input active notes.
            - global_detuning (batch, ..., 1): global detuning from
            piano type.
        Returns:
            - detuned_factors (batch, ..., n_substrings): detuning factor
            for each substring.
         """
        if self.use_detune:
            detuning = self.tanh(self.layer(extended_pitch / 128.))
            
            if global_detuning is not None:
                #global_detuning = self.tanh(global_detuning)
                detuning = detuning + self.tanh(global_detuning) #global_detuning
            extended_pitch = detuning + extended_pitch
            #extended_pitch += detuning

        return midi_to_hz(extended_pitch)

# currently ok
class MonophonicNetwork(nn.Module):
    """Sequential model for computing monophonic synthesizer controls from the
    parallelized monophonic inputs.
    """
    def __init__(self):
        super(MonophonicNetwork, self).__init__()

        self.linear_1 = nn.Linear(35, 128) #
        self.leaky_relu_1 = nn.LeakyReLU(0.1)
        self.gru = nn.GRU(128, 192, 1, batch_first=True)
        self.linear_2 = nn.Linear(192, 192) #
        self.leaky_relu_2 = nn.LeakyReLU(0.1)
        self.layer_norm = nn.LayerNorm(192)
        #self.dense_out = weight_norm(nn.Linear(192, 161))
        self.dense_out = nn.Linear(192, 161) # same number of parameters with original ddsp-piano
        self.midi_norm = 128.

    def forward(self, conditioning, extended_pitch, context):
        """Forward parallelized monophonic inputs through the model.
        Args:
            - conditioning (batch * n_synths, n_frames, 2): parallelized active
            and onset conditioning.
            - extended_pitch (batch * n_synths, n_frames, 1): parallelized
            active prolonged pitch conditioning.
            - context (batch * n_synths, n_frames, context_dim): context signal
        """
        x = torch.cat([extended_pitch / self.midi_norm,
                       torch.div(conditioning, torch.tensor([self.midi_norm, 1.]).cuda()),
                       context],
                       axis=-1) # (b*16, 750, 35)
        
        x = self.linear_1(x)
        x = self.leaky_relu_1(x)
        x, _ = self.gru(x)
        x = self.linear_2(x)
        x = self.leaky_relu_2(x)
        x = self.layer_norm(x)
        x = self.dense_out(x)
        """
        in the original ddsp-piano, 
        it splits 
        - 1 to amplitude
        - 96 to harmonic_distribution   
        - 64 to maginutes
        """
        
        amplitudes, harmonic_distribution, maginutes = torch.split(x, [1, 96, 64], dim=-1)
        return amplitudes, harmonic_distribution, maginutes

# currently ok
class OneHotZEncoder(nn.Module):
    """ Transforms one-hot encoded instrument model into a Z embedding and
    model-specific detuning and inharmonicity coefficient.
    Args:
        - n_instruments (int): number of instrument to be supported.
        - z_dim (int): dimension of z embedding.
        - n_frames (int): pool embedding value over this number of time frames.
    """
    def __init__(
        self,
        n_instruments=10,
        z_dim=16,
        n_frames=None):
        super(OneHotZEncoder, self).__init__()
        self.n_instruments = n_instruments
        self.z_dim = z_dim
        self.n_frames = n_frames

        self.embedding = nn.Embedding(
            self.n_instruments,
            self.z_dim
        )

        self.inharm_embedding = nn.Embedding(
            self.n_instruments,
            1
        )

        self.detune_embedding = nn.Embedding(
            self.n_instruments,
            1
        )
    
    def alternate_training(self, first_phase=True):
        """Toggle trainability of models according to the training phase.
        (Modules involved with partial frequency computing are frozen during
        the first training phase).
        Args:
            - first_phase (bool): whether training with the 1st phase strategy
            or not.
        """
        self.embedding.requires_grad = first_phase
        self.inharm_embedding.requires_grad = not first_phase
        self.detune_embedding.requires_grad = not first_phase
    

    def forward(self, piano_model):
        # Compute Z embedding from instrument id 
        z = self.embedding(piano_model)
        global_inharm = self.inharm_embedding(piano_model)
        global_detuning = self.detune_embedding(piano_model)

        # Add time axis 
        if len(z.shape) == 2:
            z = z.unsqueeze(1)
            global_inharm = global_inharm.unsqueeze(1)
            global_detuning = global_detuning.unsqueeze(1)
        
        if self.n_frames is not None:
            # Expand time dim
            z = resample(z, self.n_frames)
            global_inharm = resample(global_inharm, self.n_frames)
            global_detuning = resample(global_detuning, self.n_frames)
        
        """
            z:  (b, 750, 16)
            global_inharm:  (b, 750, 1)
            global_detuning:  (b, 750, 1)
        """
        return z, global_inharm, global_detuning

# currently ok?
class F0ProcessorCell_RNN(nn.Module):
    def __init__(self, frame_rate=250):
        super(F0ProcessorCell_RNN, self).__init__()
        self.frame_rate = frame_rate
        self.state_size = 2 

        self.release_duration = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)

    def forward(self, midi_note, previous_state):
        """ Extend note
        Args:
            - midi_note (batch, 1): active MIDI vector frame.
            - previous_state (batch, 2): which note was played for how long.
        """
        previous_note = previous_state[0][..., 0:1]
        decayed_steps = previous_state[0][..., 1:2]

        note_activity = torch.gt(midi_note, 0)
        decay_end = torch.gt(decayed_steps, self.release_duration * self.frame_rate)

        note_activity = note_activity.type(torch.float32)
        decay_end = 1 - decay_end.type(torch.float32)
        
        midi_note = note_activity * midi_note + (1 - note_activity) * decay_end * previous_note
        decayed_steps = (1 - note_activity) * decay_end * (decayed_steps + 1)

        updated_state = torch.cat([midi_note, decayed_steps], axis=-1)
        return midi_note, [updated_state]
# RNN?
class NoteRelease(nn.Module):
    """NoteRelease layer for extending the active pitch conditioning.
    Based on the custom RNN F0ProcessorCell"""
    def __init__(self, frame_rate=250):
        super(NoteRelease, self).__init__()
        self.layer = F0ProcessorCell_RNN()
    
    def forward(self, conditioning):
        """
        Args:
            - conditioning (batch, n_frames, 2) # (6*16,750, 2)
        """
        active_pitch = conditioning[..., 0:1] # (batch, n_frames, 1) -> (6*16,750, 2)
        previous_state = [torch.randn(active_pitch.shape[0], active_pitch.shape[1], 2).cuda()]
        frames = conditioning.shape[1]
        for f in range(frames):
            extended_pitch, updated_state = self.layer(active_pitch, previous_state) # (batch, n_frames, 1) -> (6*16, 750, 1)
            previous_state = updated_state
        return extended_pitch 

# currently ok
class MultiInstrumentReverb(nn.Module):
    """Reverb with learnable impulse response compatible with a multi-
    environment setting.
    Args:
        - inference (bool): training or inference setting.
        - n_instruments (int): number of instrument reverbs to model.
        - reverb_length (int): number of samples for each impulse response.
    """
    def __init__(self, n_instruments=10, reverb_length=24000, inference=False):
        super(MultiInstrumentReverb, self).__init__()
        self.reverb_length = reverb_length
        self.n_instruments = n_instruments
        self.inference = inference

        self.reverb_dict = nn.Embedding(self.n_instruments, self.reverb_length)
        self.reverb_dict.weight.data.normal_(mean=0.0, std=1e-6)

    def exponential_decay_mask(self, ir, decay_exponent=4., decay_start=16000):
        """ Apply exponential decay mask on impulse responde as in MIDI-ddsp
        Args:
            - ir (batch, n_samples): raw impulse response.
        Returns:
            - ir (batch, n_samples): decayed impulse response.
        """
        time = torch.linspace(0.0, 1.0, self.reverb_length - decay_start).cuda()
        mask = torch.exp(- decay_exponent * time)
        mask = torch.cat([torch.ones(decay_start), mask], 0)
        return ir * mask.unsqueeze(0)

    def forward(self, piano_model):
        """Get reverb IR from instrument id"""
        ir = self.reverb_dict(piano_model)
        
        if len(ir.shape) == 3:
            ir = ir[:, 0]
        # Apply decay mask
        if self.inference:
            ir = self.exponential_decay_mask(ir)
        return ir


if __name__ == "__main__":
    # input 
    piano_model_one_hot = torch.tensor([0,2,3,2])
    conditioning = torch.randn(4, 750, 16, 2)
    pedal = torch.randn(4, 750, 4)

    # global
    z_encoder_network = OneHotZEncoder(n_frames=3*250)
    z, global_inharm, global_detuning = z_encoder_network(piano_model_one_hot)

    context_network = ContextNetwork()
    context = context_network(conditioning, pedal, z) # (b, 750, 64)
    
    reverb_network = MultiInstrumentReverb(inference=False)
    reverb_ir = reverb_network(piano_model_one_hot.unsqueeze(-1))

    # parallel 
    parallel_network = Parallelizer()
    conditioning, context, global_inharm, global_detuning = parallel_network(conditioning, context, global_inharm, global_detuning)
    
    # note release 
    note_release_network = NoteRelease()
    extended_pitch = note_release_network(conditioning)

    # inharm
    inharm_network = InharmonicityNetwork()
    inharm_coef = inharm_network(extended_pitch, global_inharm)
    
    # detuner 
    detuner_network = Detuner()
    f0_hz = detuner_network(extended_pitch, global_detuning)
    
    # MonophonicNetwork
    mono_network = MonophonicNetwork()
    amplitudes, harmonic_distribution, magnitudes = mono_network(conditioning, extended_pitch, context)
    
    # unparallel 
    features = parallel_network(
        f0_hz=f0_hz, 
        inharm_coef=inharm_coef, 
        amplitudes=amplitudes, 
        harmonic_distribution=harmonic_distribution, 
        magnitudes=magnitudes, 
        parallelize=False)
    #print(features)
    print('reverb_ir shape: ', reverb_ir.shape) #(b, reverb_duration)
    print('harmonic_distribution shape: ', features["harmonic_distribution_0"].shape)
    print('f0_hz shape: ', features["f0_hz_0"].shape)
    print('inharm_coef shape: ', features["inharm_coef_0"].shape)
    print('magnitudes shape: ', features["magnitudes_0"].shape)

    ########### DDSP
    ddsp_module_additive = MultiInharmonic(n_samples=48000)
    ddsp_module_noise = Noise()
    ddsp_module_reverb = Reverb()

    param = ddsp_module_additive.get_controls(
        features["amplitudes_0"],
        features["harmonic_distribution_0"],
        features["inharm_coef_0"],
        features["f0_hz_0"]
    )
    harmonic = ddsp_module_additive(
        param["amplitudes"],
        param["harmonic_distribution"],
        param["harmonic_shifts"],
        param["f0_hz"]
    )
    print('harmonic shape: ', harmonic.shape)
    noise = ddsp_module_noise( harmonic, features["magnitudes_0"])
    print('noise shape: ', noise.shape)
    signal = harmonic + noise
    print('signal shape: ', signal.shape)
    signal_with_ir = ddsp_module_reverb(signal, reverb_ir)
    print('signal_with_ir shape: ', signal_with_ir.shape)