"""M1.T1: token-only router inputs."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ffnmem.config import load_config
from ffnmem.model import Transformer

CFG = Path(__file__).resolve().parents[1] / "configs"


def model_with(router_input: str, **extra) -> Transformer:
    over = [f"model.memory.router_input={router_input}"]
    if router_input == "embed2_ssm":
        over.append("model.ssm={state_dim: 8, chunk: 8, feeds: [router]}")
    over += [f"{k}={v}" for k, v in extra.items()]
    torch.manual_seed(0)
    return Transformer(load_config(CFG / "tiny_cpu.yaml", over)["model"])


def codes_of(model: Transformer, x: torch.Tensor) -> torch.Tensor:
    mem = model.blocks[1].memory
    emb = model.tok_emb(x)
    s = model.ssm(emb) if model.ssm is not None else None
    return mem.module.quantize_route_logits(mem.route_logits(mem.router_features(emb, s)))


def test_embed2_codes_depend_only_on_current_and_previous_token():
    model = model_with("embed2").eval()
    torch.manual_seed(1)
    x = torch.randint(0, 256, (2, 32))
    y = x.clone()
    y[:, :20] = torch.randint(0, 256, (2, 20))  # permute/replace everything before t-1 = 20
    with torch.no_grad():
        a, b = codes_of(model, x), codes_of(model, y)
    assert torch.equal(a[:, 21:], b[:, 21:])
    assert not torch.equal(a[:, :21], b[:, :21])
    # and through the full model's memory layer: identical memory rows from t = 21 on
    with torch.no_grad():
        perm = x.clone()
        perm[:, :20] = x[:, :20].flip(1)
        assert torch.equal(codes_of(model, x)[:, 21:], codes_of(model, perm)[:, 21:])


def test_embed2_start_vector_at_t0():
    model = model_with("embed2").eval()
    x = torch.randint(0, 256, (1, 8))
    feats = model.blocks[1].memory.router_features(model.tok_emb(x))
    d = model.cfg["d_model"]
    rf = model.blocks[1].memory.router_features
    torch.testing.assert_close(feats[0, 0, :d], rf.prev_norm(rf.start))


def test_embed2_ssm_codes_ignore_backbone():
    model = model_with("embed2_ssm").eval()
    x = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        before = codes_of(model, x)
        for block in model.blocks:
            for p in list(block.attn.parameters()) + list(block.ffn.parameters()):
                p.normal_()
        assert torch.equal(before, codes_of(model, x))


@pytest.mark.parametrize("router_input", ["embed2", "embed2_ssm"])
@pytest.mark.parametrize("act_ckpt", [False, True])
def test_surrogate_reaches_router_and_embeddings(router_input, act_ckpt):
    model = model_with(router_input).train()
    model.act_ckpt = act_ckpt
    torch.nn.init.normal_(model.blocks[1].memory.module.readout_o_proj.weight, std=0.02)
    x = torch.randint(0, 256, (2, 33))
    _, loss = model(x[:, :-1], x[:, 1:])
    loss.backward()
    mem = model.blocks[1].memory
    assert float(mem.router.weight.grad.abs().sum()) > 0
    assert float(mem.router_features.start.grad.abs().sum()) > 0
    # Embedding gradient through the router only: compare against a run with the router input zero-weighted.
    g_full = model.tok_emb.weight.grad.clone()
    model.zero_grad()
    with torch.no_grad():
        mem.router.weight.mul_(0)
    _, loss = model(x[:, :-1], x[:, 1:])
    loss.backward()
    assert not torch.equal(g_full, model.tok_emb.weight.grad)
    if router_input == "embed2_ssm":
        model.zero_grad()
        with torch.no_grad():
            torch.nn.init.normal_(mem.router.weight, std=0.02)
        _, loss = model(x[:, :-1], x[:, 1:])
        loss.backward()
        assert float(model.ssm.w_in.weight.grad.abs().sum()) > 0


def test_hidden_router_is_default_and_has_no_features():
    model = model_with("hidden")
    mem = model.blocks[1].memory
    assert mem.router.weight.shape[1] == model.cfg["d_model"]
    assert not hasattr(mem, "router_features")
