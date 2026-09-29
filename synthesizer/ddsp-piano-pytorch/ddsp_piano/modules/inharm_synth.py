import torch 
import torch.nn as nn
from ddsp_piano.ddsp_pytorch.core import scale_function, remove_above_nyquist, frequency_filter
from ddsp_piano.ddsp_pytorch.harmonic_oscillator import HarmonicOscillator

def get_inharmonic_freq(f0_hz, inharm_coef, n_harmonics):
    """ Create inharmonic multiples of the fundamental frequency and provide
    deviations from pure harmonic frequencies.
    Args:
        - f0_hz (batch, :, 1): fundamental frequencies.
        - inharm_coef (batch, :, 1): inharmonicity coefficients.
        - n_harmonics (int): number of harmonics.
    Returns:
        - inharmonic_freq (batch, :, n_harmonics): oscillators
        frequencies in Hz.
        - harmonic_shifts (batch, :, n_harmonics): deviation from pure integer
        factor harmonicity.
    """
    f0_hz = f0_hz.to(torch.float32)

    # Integer ratios
    int_multiplier = torch.linspace(1.0, float(n_harmonics), int(n_harmonics)).cuda()
    int_multiplier = int_multiplier.unsqueeze(0).unsqueeze(0)

    # Inharmonicity factor
    inharm_factor = torch.pow(int_multiplier, 2).cuda()
    inharm_factor = inharm_factor * inharm_coef + 1.
    inharm_factor = torch.sqrt(inharm_factor)

    # Modal frequencies
    inharmonic_freq = f0_hz * int_multiplier * inharm_factor
    # Shifts
    harmonic_shifts = inharm_factor - 1.

    return inharmonic_freq, harmonic_shifts

class InHarmonic(nn.Module):
    """Synthesize audio with a bank of inharmonic sinusoidal oscillators.
    Args:
        - n_samples (int): number of audio samples to generate.
        - sample_rate (int): sample per second.
        - min_frequency (int): minimum supported frequency (in Hz).
        - use_amplitude (bool): use global amplitude or enable free harmonic
        amplitudes.
        - normalize_below_nyquist (bool): set amplitude of frequencies abow
        Nyquist to 0.
        - inference (bool): use angular cumsum (for inference only).
    """
    def __init__(self,
                n_samples=64000,
                sample_rate=16000,
                min_frequency=20,
                use_amplitude=True,
                normalize_below_nyquist=True,
                inference=False):
        super(InHarmonic, self).__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.min_frequency = min_frequency
        self.use_amplitude = use_amplitude
        self.normalize_below_nyquist = normalize_below_nyquist
        self.inference = inference

        self.harmonic_synthesizer = HarmonicOscillator(sample_rate, n_samples)
    def get_controls(self, 
                amplitudes, 
                harmonic_distribution, 
                inharm_coef, 
                f0_hz):
        """ Convert network output tensors into dict of synth controls.
        Args:
            - amplitudes (batch, time, 1): global amplitude control.
            - harmonic_distribution (batch, time, n_harmonics): per harmonic
            normalized amplitudes.
            - inharm_coef (batch, time, 1): inharmonicity coefficient.
            - f0_hz (batch, time, 1): fundamental frequency in hz.
        Returns:
            - controls (Dict): dict of synthesizer controls
        """

        ## Scale inputs
        amplitudes = scale_function(amplitudes)
        harmonic_distribution = scale_function(harmonic_distribution)

        ## Compute the inharmonic frequencies and harmonic shifts
        n_harmonics = int(harmonic_distribution.shape[-1])
        inharmonic_freq, harmonic_shifts = get_inharmonic_freq(
            f0_hz, inharm_coef, n_harmonics
        )

        # Bandlimit the harmonic distribution
        if self.normalize_below_nyquist:
            harmonic_distribution = remove_above_nyquist(harmonic_distribution, inharmonic_freq, self.sample_rate)
            # Set amplitude to zero if below hearable
            aa = (f0_hz > self.min_frequency).float() + 1e-4
            amplitudes = amplitudes * aa
        
        # Normalize
        #print('harmonic_distribution ', harmonic_distribution.shape) (b, 750, 96)
        if self.use_amplitude:
            harmonic_distribution = harmonic_distribution / torch.sum(harmonic_distribution, axis=-1, keepdims=True)
        else:
            amplitudes = tf.ones_like(amplitudes)
        
        return {'amplitudes': amplitudes,
                'harmonic_distribution': harmonic_distribution,
                'harmonic_shifts': harmonic_shifts,
                'f0_hz': f0_hz}

    def forward(self,
                amplitudes,
                harmonic_distribution,
                harmonic_shifts,
                f0_hz):
        """ Synthesize audio with inharmonic synthesizer from controls.
        Args:
            - amplitudes (batch, time, 1): global amplitude.
            - harmonic_distribution (batch, time, n_harmonics): harmonics
            relative amplitudes (sums to 1).
            - harmonic_shifts (batch, time, n_harmonics): harmonic shifts
            from perfect harmonic frequencies.
            - f0_hz (batch, time, 1): fundamental frequency, in Hz.
        """
        signal = self.harmonic_synthesizer(
            f0_hz,
            amplitudes,
            harmonic_shifts,
            harmonic_distribution)
        
        return signal

class MultiInharmonic(nn.Module):
    """Inharmonic synthesizer with multiple F0 controls."""
    def __init__(self,
                n_samples=64000,
                sample_rate=16000,
                min_frequency=20,
                use_amplitude=True,
                normalize_below_nyquist=True,
                inference=False):
        super(MultiInharmonic, self).__init__()
        self.single_inharmonic_module = InHarmonic(n_samples, sample_rate, min_frequency, use_amplitude, normalize_below_nyquist, inference)

    def get_controls(self,
                    amplitudes, #(b, n_frames, 1)
                    harmonic_distribution, #(b, n_frames, n_harmonics)
                    inharm_coef, #(b, n_frames, 1)
                    f0_hz): #(b, n_frames, 2)
        # Get partial amplitudes and inharmonicity displacement
        controls = self.single_inharmonic_module.get_controls(
            amplitudes,
            harmonic_distribution,
            inharm_coef,
            f0_hz[..., 0:1]
        )

        # Put back multi-f0 signal
        controls["f0_hz"] = f0_hz
        # Divide global amplitude by the number of substrings
        controls['amplitudes'] = controls['amplitudes'] / torch.tensor(f0_hz.shape[-1]).to(torch.float32)
        return controls
    
    def forward(self,
                amplitudes,
                harmonic_distribution,
                harmonic_shifts,
                f0_hz):
        n_substrings = f0_hz.shape[-1]
        #print('n_substrings: ', n_substrings)
        # Audio from first substring
        audio = self.single_inharmonic_module(
            amplitudes,
            harmonic_distribution,
            harmonic_shifts,
            f0_hz[...,0:1]
        )
        
        # Add other substrings signals
        for substring in range(1, n_substrings):
            audio += self.single_inharmonic_module(
                amplitudes,
                harmonic_distribution,
                harmonic_shifts,
                f0_hz[..., substring: substring + 1]
            )
        #harmonic = audio
        #harmonic = frequency_filter(audio, harmonic_distribution * (1.0 + harmonic_shifts)) #unsure

        return audio
        