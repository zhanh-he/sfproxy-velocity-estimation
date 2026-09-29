import torch
import torch.nn as nn 
from torch.nn import functional as F
import torchaudio
import numpy as np

class HybridLoss(nn.Module):
    def __init__(self, n_ffts, inharm, phase, weight=0.01, loss_type='L1', l1_weight_of_inharm=0.1):
        super().__init__()
        self.inharm = inharm
        self.phase = phase 

        self.mssLoss = MSSLoss(n_ffts)
        self.reverb_l1_loss = ReverbRegularizer(weight, loss_type)
        self.l1_weight_of_inharm = l1_weight_of_inharm
    
    def forward(self, y_pred, y_true, reverb_ir):
        loss_mss = self.mssLoss(y_pred, y_true)
        loss_reverb_l1 = self.reverb_l1_loss(reverb_ir)
        if self.phase:
            return loss_mss + loss_reverb_l1, loss_mss, loss_reverb_l1
        else:
            l1_penalty = self.l1_weight_of_inharm * sum([self.inharm.slopes_modifier.abs().sum()]) + self.l1_weight_of_inharm * sum([self.inharm.offsets_modifier.abs().sum()]) 
            return loss_mss + loss_reverb_l1 + l1_penalty, loss_mss, loss_reverb_l1


###### from ddsp-singing-vocoder
class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss. 
    """

    def __init__(self, n_fft=111, alpha=1.0, overlap=0.75, eps=1e-7, name='SSSLoss'):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))  # 25% of the length
        self.spec = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)
        self.name = name
    def forward(self, x_true, x_pred):
        min_len = np.min([x_true.shape[1], x_pred.shape[1]])
    
        x_true = x_true[:, -min_len:]
        x_pred = x_pred[:, -min_len:]

        S_true = self.spec(x_true)
        S_pred = self.spec(x_pred)
        linear_term = F.l1_loss(S_pred, S_true)
        log_term = F.l1_loss((S_true + self.eps).log2(), (S_pred + self.eps).log2())

        loss = linear_term + self.alpha * log_term
        return {'loss':loss}


class MSSLoss(nn.Module):
    """
    Multi-scale Spectral Loss.
    Usage ::
    mssloss = MSSLoss([2048, 1024, 512, 256], alpha=1.0, overlap=0.75)
    mssloss(y_pred, y_gt)
    input(y_pred, y_gt) : two of torch.tensor w/ shape(batch, 1d-wave)
    output(loss) : torch.tensor(scalar)
    48k: n_ffts=[2048, 1024, 512, 256]
    24k: n_ffts=[1024, 512, 256, 128]
    """

    def __init__(self, n_ffts, alpha=1.0, ratio = 1.0, overlap=0.75, eps=1e-7, use_reverb=True, name='MultiScaleLoss'):
        super().__init__()
        self.losses = nn.ModuleList([SSSLoss(n_fft, alpha, overlap, eps) for n_fft in n_ffts])
        self.ratio = ratio
        self.name = name
    def forward(self, x_pred, x_true, return_spectrogram=True):
        x_pred = x_pred[..., :x_true.shape[-1]]
        if return_spectrogram:
            losses = []
            spec_true = []
            spec_pred = []
            for loss in self.losses:
                loss_dict = loss(x_true, x_pred)
                losses += [loss_dict['loss']]
        
        return self.ratio*sum(losses).sum()

##### reverb_loss
class ReverbRegularizer(nn.Module):
    """Regularization loss on the reverb impulse response.
    Params:
        - weight (float): loss weight.
        - loss_type {'L1', 'L2'}: compute L1 or L2 regularization.
    """
    def __init__(self, weight=0.01, loss_type='L1'):
        super(ReverbRegularizer, self).__init__()
        self.weight = weight
        self.loss_type = loss_type
    def forward(self, reverb_ir):
        if self.loss_type == 'L1':
            loss = torch.sum(torch.abs(reverb_ir))
        elif self.loss_type == 'L2':
            loss = torch.sum(torch.square(reverb_ir))
        loss /= reverb_ir.shape[0] # Divide by batch size
        return self.weight * loss
