from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class NoteTokenizerConfig:
    d_model: int = 256
    pitch_vocab: int = 128

    # If True, add a dedicated onset positional projection in addition to cont basis.
    use_onset_pos: bool = True


class NoteTokenizer(nn.Module):
    """Convert padded note sequences into transformer tokens."""

    def __init__(self, cfg: NoteTokenizerConfig):
        super().__init__()
        # Hydra may pass a DictConfig / dict. Convert to dataclass for attribute access.
        if not isinstance(cfg, NoteTokenizerConfig):
            cfg = NoteTokenizerConfig(**dict(cfg))
        self.cfg = cfg

        self.pitch_emb = nn.Embedding(int(cfg.pitch_vocab), int(cfg.d_model))
        # Continuous basis vectors for onset, duration, velocity
        self.cont_basis = nn.Parameter(torch.randn(3, int(cfg.d_model)) * 0.02)

        self.use_onset_pos = bool(cfg.use_onset_pos)
        if self.use_onset_pos:
            self.onset_proj = nn.Sequential(
                nn.Linear(1, int(cfg.d_model)),
                nn.GELU(),
                nn.Linear(int(cfg.d_model), int(cfg.d_model)),
            )
        else:
            self.onset_proj = None

        self.norm = nn.LayerNorm(int(cfg.d_model))

    def forward(self, pitch: torch.Tensor, cont: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Tokenize.

        Args:
            pitch: (B, N) int64
            cont: (B, N, 3) float32 [onset_norm, dur_norm, vel_01]
            mask: (B, N) bool, True means valid token

        Returns:
            tokens: (B, N, d_model)
        """

        if pitch.ndim != 2:
            raise ValueError("pitch must be (B, N)")
        if cont.ndim != 3 or cont.shape[-1] != 3:
            raise ValueError("cont must be (B, N, 3)")

        # Embeddings
        p = self.pitch_emb(pitch.clamp(0, self.cfg.pitch_vocab - 1))
        # (B, N, 3) @ (3, d) -> (B, N, d)
        c = torch.einsum("bnc,cd->bnd", cont, self.cont_basis)
        x = p + c

        if self.use_onset_pos and self.onset_proj is not None:
            onset = cont[..., 0:1]
            x = x + self.onset_proj(onset)

        x = self.norm(x)

        # Optional: zero-out padded tokens to reduce noise
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)

        return x
