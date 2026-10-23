from __future__ import annotations
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_score_inf
from .io_types import AcousticIO, CondIO
from .utils import safe_logit

def _run_length_from_dur_roll(dur_roll: torch.Tensor) -> torch.Tensor:
    T, P = dur_roll.shape
    dur = dur_roll.to(torch.int32)
    out = torch.zeros((T+1, P), device=dur.device, dtype=torch.int32)
    for t in range(T-1, -1, -1):
        out[t] = (out[t+1] + 1) * dur[t]
    return out[:-1]

class _ConformerLikeLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, conv_kernel: int = 7):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, padding=conv_kernel//2, groups=d_model)
        self.pwconv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.drop1(attn_out))

        y = x.transpose(1, 2)
        y = self.dwconv(y)
        y = self.pwconv(y)
        y = y.transpose(1, 2)
        x = self.norm2(x + self.drop2(y))

        y = self.ffn(x)
        x = self.norm3(x + self.drop3(y))
        return x

@register_score_inf("note_editor")
class NoteEventEditor(nn.Module):
    """
    Tokenize note onsets from cond.onset (GT) and predict delta velocity per note.
    Uses only:
      - vel / vel_logits from acoustic
      - dur/sus/exframe/frame from cond (optional)
    """
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        arch: str = "conformer",        # transformer / conformer
        alpha: float = 0.2,
        max_frames: int = 4096,
        use_cond_feats: List[str] = ["frame", "exframe"],  # optional extras for token features
    ):
        super().__init__()
        assert arch in ("transformer", "conformer")
        self.arch = arch
        self.alpha = alpha
        self.use_cond_feats = use_cond_feats

        self.pitch_emb = nn.Embedding(88, d_model)
        self.time_emb = nn.Embedding(max_frames, d_model)

        # continuous features: v0 + dur_norm + sus + exframe + frame (optional)
        # We'll build a fixed vector length depending on use_cond_feats.
        self.feat_dim = 1 + len(use_cond_feats)  # v0 + selected cond feats
        self.feat_proj = nn.Linear(self.feat_dim, d_model)

        if arch == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
                dropout=dropout, batch_first=True, activation="gelu"
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        else:
            self.encoder = nn.ModuleList([
                _ConformerLikeLayer(d_model, n_heads, dropout=dropout)
                for _ in range(n_layers)
            ])

        self.head = nn.Linear(d_model, 1)

    def forward(self, acoustic: AcousticIO, cond: CondIO):
        vel0 = acoustic.vel if acoustic.vel is not None else torch.sigmoid(acoustic.vel_logits)
        B, T, P = vel0.shape
        device = vel0.device

        if cond.onset is None:
            raise ValueError("note_editor requires onset conditioning, but cond.onset is None.")

        onset = cond.onset
        if onset.dim() != 3:
            raise ValueError(f"Expected cond.onset to be 3D [B,T,P], but got shape={tuple(onset.shape)}")
        if onset.size(0) != B:
            raise ValueError(
                f"Batch mismatch between vel0 and cond.onset: vel0.B={B}, cond.onset.B={onset.size(0)}"
            )

        # Keep onset tensor aligned with acoustic pitch/time sizes to avoid OOB indexing.
        if onset.size(1) != T:
            if onset.size(1) > T:
                onset = onset[:, :T]
            else:
                onset = F.pad(onset, (0, 0, 0, T - onset.size(1)))
        if onset.size(2) != P:
            if onset.size(2) > P:
                onset = onset[:, :, :P]
            else:
                onset = F.pad(onset, (0, P - onset.size(2), 0, 0))

        onset_mask = (onset > 0.5)

        token_embs = []
        token_coords = []
        lengths = []

        # precompute duration run-length if dur available
        dur_run = None

        for b in range(B):
            coords = onset_mask[b].nonzero(as_tuple=False)  # (N,2) [t,p]
            n = coords.shape[0]
            lengths.append(n)
            token_coords.append(coords)

            if n == 0:
                token_embs.append(torch.zeros((0, self.pitch_emb.embedding_dim), device=device))
                continue

            t_idx = coords[:, 0]
            pitch_max = min(P - 1, self.pitch_emb.num_embeddings - 1)
            p_idx = coords[:, 1].clamp_max(pitch_max)

            v0 = vel0[b, t_idx, p_idx].float()

            cont_feats = [v0]

            # cond features
            for k in self.use_cond_feats:
                v = getattr(cond, k)

                if k == "dur":
                    # turn dur roll into a normalized duration length
                    runlen = _run_length_from_dur_roll(v[b])  # (T,P) int
                    dur_frames = runlen[t_idx, p_idx].float()
                    cont_feats.append(torch.clamp(dur_frames / 200.0, 0.0, 1.0))
                else:
                    cont_feats.append(v[b, t_idx, p_idx].float())

            cont = torch.stack(cont_feats, dim=-1)  # (N,feat_dim)
            cont_emb = self.feat_proj(cont)

            pitch_emb = self.pitch_emb(p_idx)
            time_emb = self.time_emb(t_idx.clamp_max(self.time_emb.num_embeddings - 1))

            token_embs.append(pitch_emb + time_emb + cont_emb)

        Nmax = max(lengths) if lengths else 0
        if Nmax == 0:
            vel_corr = vel0
            return {"vel_corr": vel_corr, "delta": None, "debug": {"note_count": torch.tensor(lengths, device=device)}}

        D = self.pitch_emb.embedding_dim
        x = torch.zeros((B, Nmax, D), device=device)
        key_padding_mask = torch.ones((B, Nmax), device=device, dtype=torch.bool)
        for b in range(B):
            n = lengths[b]
            if n > 0:
                x[b, :n] = token_embs[b]
                key_padding_mask[b, :n] = False

        # Some CUDA/PyTorch combinations are unstable when a sample is fully padded.
        # Run attention only on rows with at least one token.
        valid_rows = [b for b, n in enumerate(lengths) if n > 0]
        if len(valid_rows) == B:
            if self.arch == "transformer":
                h = self.encoder(x, src_key_padding_mask=key_padding_mask)
            else:
                h = x
                for layer in self.encoder:
                    h = layer(h, key_padding_mask=key_padding_mask)
        else:
            h = torch.zeros_like(x)
            v_idx = torch.tensor(valid_rows, device=device, dtype=torch.long)
            xv = x[v_idx]
            kv = key_padding_mask[v_idx]
            if self.arch == "transformer":
                hv = self.encoder(xv, src_key_padding_mask=kv)
            else:
                hv = xv
                for layer in self.encoder:
                    hv = layer(hv, key_padding_mask=kv)
            h[v_idx] = hv

        delta_tok = self.head(h).squeeze(-1)  # (B,Nmax)

        vel_corr = vel0.clone()
        for b in range(B):
            n = lengths[b]
            if n == 0:
                continue
            coords = token_coords[b]
            dv = self.alpha * torch.tanh(delta_tok[b, :n])
            t_idx = coords[:, 0]
            p_idx = coords[:, 1]
            vel_corr[b, t_idx, p_idx] = torch.clamp(vel0[b, t_idx, p_idx] + dv, 0.0, 1.0)

        return {"vel_corr": vel_corr, "delta": delta_tok, "debug": {"note_count": torch.tensor(lengths, device=device)}}
