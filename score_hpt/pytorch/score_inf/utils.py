from __future__ import annotations
import torch
import torch.nn.functional as F

def safe_logit(p: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)

def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x, mask shape broadcastable; mask should be 0/1
    num = (x * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den

def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean(torch.abs(pred - target), mask)

def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    return masked_mean(F.huber_loss(pred, target, reduction="none", delta=delta), mask)

def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # target in [0,1]
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return masked_mean(loss, mask)