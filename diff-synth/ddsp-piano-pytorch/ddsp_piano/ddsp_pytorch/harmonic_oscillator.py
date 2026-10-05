import torch
import torch.nn as nn 
from torch.nn import functional as F
import numpy as np

from ddsp_piano.ddsp_pytorch.core import remove_above_nyquist, resample, upsample


class HarmonicOscillator(nn.Module):
    """synthesize audio with a bank of harmonic oscillators"""
    def __init__(self, 
                fs, 
                n_samples,
                oscillator=torch.sin):
        super(HarmonicOscillator, self).__init__()
        self.fs = fs
        self.n_samples = n_samples
        
        self.oscillator = oscillator

    def get_harmonic_frequencies(self, f0, n_harmonics):
        """Create integer multiples of the fundamental frequency.
        Args:
            frequencies: Fundamental frequencies (Hz). Shape [batch_size, :, 1].
            n_harmonics: Number of harmonics.
        Returns:
            harmonic_frequencies: Oscillator frequencies (Hz).
            Shape [batch_size, :, n_harmonics].
        """
        f0 = f0.to(torch.float32)
        f_ratios = torch.linspace(1.0, float(n_harmonics), int(n_harmonics)).cuda()
        f_ratios = f_ratios.unsqueeze(0).unsqueeze(0)
        harmonic_frequencies = f0 * f_ratios
        return harmonic_frequencies
    
    def oscillator_bank(self,
                        frequency_envelope,
                        amplitude_envelope,
                        sample_rate,
                        sum_sinusoids=True):
        """Generates audio from sample-wise frequencies for a bank of oscillators.
            Args:
                frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
                [batch_size, n_samples, n_sinusoids].
                amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
                n_samples, n_sinusoids].
                sample_rate: Sample rate in samples per a second.
                sum_sinusoids: Add up audio from all the sinusoids.
            Returns:
                wav: Sample-wise audio. Shape [batch_size, n_samples, n_sinusoids] if
                sum_sinusoids=False, else shape is [batch_size, n_samples].
        """
        frequency_envelopes = frequency_envelope.to(torch.float32)
        amplitude_envelopes = amplitude_envelope.to(torch.float32)

        amplitude_envelopes = remove_above_nyquist(amplitude_envelopes, frequency_envelopes, sample_rate)
        
        omegas = frequency_envelopes * (2.0 * np.pi)
        omegas = omegas / float(sample_rate)

        phase = torch.cumsum(omegas, axis=1)
        audio = amplitude_envelopes * self.oscillator(phase)
        if sum_sinusoids:
            audio = audio.sum(-1, keepdim=True)
        return audio
    
    def forward(self, 
                f0, 
                amplitudes, 
                harmonic_shifts=None,
                harmonic_distribution=None,
                n_samples=None):
        '''
            f0: B x T x 1 (Hz), Frame-wise fundamental frequency in Hz.
            amplitudes: B x T x 1, Frame-wise oscillator peak amplitude.
            harmonic_shifts: B x T x n_harmonic, Harmonic frequency variation(Hz), zero-centered.
            Total frequency of a harmonic is equal to harmonic_frequencies *= (1.0 + harmonic_shifts) ### why?
            harmonic_distribution: B x T x n_harmonic, Harmonic amplitude variations, ranged from zero to one.
            Total amplitude of a harmonic is equal to harmonic_amplitudes = amplitudes * harmonic_distribution
          ---
            signal: B x T
        '''

        f0 = f0.to(torch.float32)
        amplitudes = amplitudes.to(torch.float32)
        
        n_harmonics = harmonic_distribution.shape[-1]

        # Create harmonic frequencies [batch_size, n_frames, n_harmonics].
        harmonic_frequencies = self.get_harmonic_frequencies(f0, n_harmonics)
        if harmonic_shifts is not None:
            harmonic_frequencies = harmonic_frequencies * (1.0 + harmonic_shifts)
        
        # Create harmonic amplitudes [batch_size, n_frames, n_harmonics].
        if harmonic_distribution is not None:
            harmonic_amplitudes = amplitudes * harmonic_distribution
        else:
            harmonic_amplitudes = amplitudes
        
        # Create sample-wise envelopes.
        frequency_envelopes = upsample(harmonic_frequencies, 64)  # cycles/sec
        amplitude_envelopes = upsample(harmonic_amplitudes, 64) # maybe window?

        audio = self.oscillator_bank(frequency_envelopes, amplitude_envelopes, self.fs).squeeze(-1)

        return audio
        

