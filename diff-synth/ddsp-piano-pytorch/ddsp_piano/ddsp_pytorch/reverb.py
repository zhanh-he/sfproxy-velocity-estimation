import torch
import torch.nn as nn

import numpy as np

from ddsp_piano.ddsp_pytorch.core import fft_convolve

class Reverb(nn.Module):
    """ Convolutional (FIR) reverb """
    def __init__(self):
        """Takes neural network outputs directly as the impulse response.
        """
        super(Reverb, self).__init__()
    def mask_dry_ir(self, ir):
        """Set first impulse response to zero to mask the dry signal."""
        # Make IR 2-D [batch, ir_size].
        if len(ir.shape) == 1:
            ir = ir.unsqueeze(0)  # Add a batch dimension
        if len(ir.shape) == 3:
            ir = ir[:, :, 0]  # Remove unnessary channel dimension.
        # Mask the dry signal.
        dry_mask = torch.zeros(int(ir.shape[0]), 1).to(torch.float32).cuda()
        
        return torch.cat([dry_mask, ir[:, 1:]], axis=1)
    def forward(self, audio, ir):
        """Apply impulse response.
        Args:
            audio: Dry audio, 2-D Tensor of shape [batch, n_samples].
            ir: 3-D Tensor of shape [batch, ir_size, 1] or 2D Tensor of shape
            [batch, ir_size].
        Returns:
            tensor of shape [batch, n_samples]
        """
        audio = audio.to(torch.float32)
        ir = ir.to(torch.float32)

        ir = self.mask_dry_ir(ir)
        wet = fft_convolve(audio, ir, padding='same', delay_compensation=0)
        return wet+audio