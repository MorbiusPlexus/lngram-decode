"""M1.T2: the draft path reads no backbone activations; row gather equals the full lookup."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ffnmem.config import load_config
from ffnmem.memory.draft import lookup_rows_at
from ffnmem.model import Transformer

CFG = Path(__file__).resolve().parents[1] / "configs"
SD_A = ["model.memory.router_input=hidden", "model.memory.token_router={enabled: true, loss_weight: 0.1}",
        "model.ssm=null"]
VARIANTS = {"sdc": [], "sdb": ["model.memory.router_input=embed2", "model.ssm=null"], "sda": SD_A}


def sd_model(variant: str = "sdc") -> Transformer:
    torch.manual_seed(0)
    model = Transformer(load_config(CFG / "tiny_cpu_sd.yaml", VARIANTS[variant])["model"])
    torch.manual_seed(1)
    torch.nn.init.normal_(model.blocks[1].memory.module.readout_o_proj.weight, std=0.02)
    return model


def all_positions(b: int, t: int):
    return torch.arange(b).repeat_interleave(t), torch.arange(t).repeat(b)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_backbone_randomization_leaves_draft_logits_bitwise_unchanged(variant):
    model = sd_model(variant).eval()
    x = torch.randint(0, 256, (2, 32))
    bi, ti = all_positions(2, 32)
    with torch.no_grad():
        before = model.draft_logits(x, bi, ti)
        g = torch.Generator().manual_seed(5)
        for block in model.blocks:
            for name, p in block.named_parameters():
                if not name.startswith("memory."):  # attention, FFN, block norms
                    p.copy_(torch.randn(p.shape, generator=g))
        model.norm.weight.copy_(torch.randn(model.norm.weight.shape, generator=g))
        after = model.draft_logits(x, bi, ti)
    assert before.shape == (64, 256)
    assert torch.equal(before, after)


def test_training_forward_uses_the_same_draft_path():
    model = sd_model().eval()
    x = torch.randint(0, 256, (2, 33))
    bi, ti = all_positions(2, 32)
    with torch.no_grad():
        logits, _ = model(x[:, :-1], x[:, 1:], draft_sel=(bi, ti))
        q = model.draft_logits(x[:, :-1], bi, ti)
    from ffnmem.losses import draft_kl_chunk

    kl, agree, _ = draft_kl_chunk(q, logits[bi, ti])
    torch.testing.assert_close(model.last_aux["draft_kl"], kl / 64)
    assert float(model.last_aux["draft_top1_agree"]) == float(agree) / 64


def test_rows_at_positions_equal_full_sequence_lookup():
    model = sd_model().train()
    m = model.blocks[1].memory.module
    x = torch.randint(0, 256, (3, 32))
    emb = model.tok_emb(x)
    q = model.draft_route_logits(emb.detach(), model.ssm(emb.detach()))
    codes = m.quantize_route_logits(q)
    with torch.no_grad():
        full = m._lookup_route_chunk(codes, None, 0, m.num_routes, None, None)
        bi, ti = all_positions(3, 32)
        rows = lookup_rows_at(m, codes, None, bi, ti)
    torch.testing.assert_close(rows, full.reshape(-1, *full.shape[2:]), rtol=0, atol=0)
    assert torch.count_nonzero(rows[0]) == 0  # t = 0: no complete n-gram


def test_sink_mass_and_shapes():
    model = sd_model().eval()
    m = model.blocks[1].memory.module
    x = torch.randint(0, 256, (1, 32))
    bi, ti = all_positions(1, 32)
    with torch.no_grad():
        h = model.draft.hidden(m, model.tok_emb(x)[0], lookup_rows_at(
            m, m.quantize_route_logits(model.draft_route_logits(model.tok_emb(x), model.ssm(model.tok_emb(x)))),
            None, bi, ti), model.ssm(model.tok_emb(x))[0])
    assert h.shape == (32, model.cfg["d_model"]) and torch.isfinite(h).all()


def test_no_nan_under_fp16_autocast():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = sd_model().to(device).train()
    x = torch.randint(0, 256, (2, 33), device=device)
    bi, ti = all_positions(2, 32)
    with torch.autocast(device_type=device, dtype=torch.float16):
        logits, loss = model(x[:, :-1], x[:, 1:], draft_sel=(bi.to(device), ti.to(device)))
        q = model.draft_logits(x[:, :-1], bi.to(device), ti.to(device))
    assert q.dtype == torch.float16 and q.shape == (64, 256)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(q).all()
    for n, p in model.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n
