import torch

from models.note_proxy_tfm import NoteProxyTransformer, NoteProxyTransformerConfig


def test_note_proxy_forward_shapes():
    cfg = NoteProxyTransformerConfig(
        num_layers=2,
        nhead=4,
        dim_feedforward=128,
        dropout=0.0,
        d_note=2,
        d_seg=0,
    )
    model = NoteProxyTransformer(cfg)

    b, n = 2, 8
    pitch = torch.randint(0, 128, (b, n), dtype=torch.long)
    cont = torch.rand(b, n, 3)
    mask = torch.rand(b, n) > 0.3

    note_out, seg_out = model(pitch=pitch, cont=cont, mask=mask)

    assert note_out.shape == (b, n, 2)
    assert seg_out is None
