import torch
import torch.nn as nn

from ddsp_piano.ddsp_pytorch.core import frequency_filter, scale_function


class Noise(nn.Module):
    """
    DDSP Noise module. 
    """
    def __init__(self):
        super(Noise, self).__init__()
    def forward(self, harmonic, noise_param):
        noise_param = scale_function(noise_param)
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1 # [-1, 1]
        noise = frequency_filter(noise, noise_param)
        return noise