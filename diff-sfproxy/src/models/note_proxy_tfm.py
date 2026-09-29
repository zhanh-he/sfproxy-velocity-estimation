from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import nn

from models.note_tokenizer import NoteTokenizer, NoteTokenizerConfig


@dataclass
class NoteProxyTransformerConfig:
    # Tokenizer
    tokenizer: NoteTokenizerConfig = field(default_factory=NoteTokenizerConfig)

    # Transformer
    num_layers: int = 6
    nhead: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # Heads
    d_note: int = 2
    d_seg: int = 0  # 0 means disabled


class NoteProxyTransformer(nn.Module):
    """Transformer proxy: note events -> note-wise dynamics features."""

    def __init__(self, cfg: NoteProxyTransformerConfig):
        super().__init__()
        # Hydra may pass a DictConfig / dict. Convert to dataclass for attribute access.
        if not isinstance(cfg, NoteProxyTransformerConfig):
            cfg_dict = dict(cfg)
            tok = cfg_dict.get("tokenizer", {})
            if not isinstance(tok, NoteTokenizerConfig):
                tok = NoteTokenizerConfig(**dict(tok))
            cfg_dict["tokenizer"] = tok
            cfg = NoteProxyTransformerConfig(**cfg_dict)
        self.cfg = cfg

        self.tokenizer = NoteTokenizer(cfg.tokenizer)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(cfg.tokenizer.d_model),
            nhead=int(cfg.nhead),
            dim_feedforward=int(cfg.dim_feedforward),
            dropout=float(cfg.dropout),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(cfg.num_layers))

        d_model = int(cfg.tokenizer.d_model)
        self.note_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, int(cfg.d_note)),
        )

        self.d_seg = int(cfg.d_seg)
        if self.d_seg > 0:
            self.seg_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, self.d_seg),
            )
        else:
            self.seg_head = None

    def forward(
        self,
        pitch: torch.Tensor,
        cont: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward.

        Args:
            pitch: (B, N)
            cont: (B, N, 3)
            mask: (B, N) bool

        Returns:
            note_feats: (B, N, d_note)
            seg_feats: (B, d_seg) or None
        """

        tokens = self.tokenizer(pitch, cont, mask)

        key_padding_mask = None
        if mask is not None:
            # transformer expects True for padding positions
            key_padding_mask = ~mask

        h = self.encoder(tokens, src_key_padding_mask=key_padding_mask)

        note_out = self.note_head(h)

        seg_out = None
        if self.seg_head is not None:
            if mask is None:
                pooled = h.mean(dim=1)
            else:
                w = mask.to(h.dtype).unsqueeze(-1)
                denom = torch.clamp(w.sum(dim=1), min=1.0)
                pooled = (h * w).sum(dim=1) / denom
            seg_out = self.seg_head(pooled)

        return note_out, seg_out
