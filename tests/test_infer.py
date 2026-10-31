"""Cached decoding equals the full forward (fp32), for Wave 1 and every SD variant."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ffnmem import checkpoint as ckpt
from ffnmem.config import load_config
from ffnmem.infer import DecodeState, decode_step, greedy_generate, load_model
from ffnmem.model import Transformer

CFG = Path(__file__).resolve().parents[1] / "configs"
VARIANTS = {
    "wave1": ("tiny_cpu.yaml", []),
    "sdb": ("tiny_cpu_sd.yaml", ["model.memory.router_input=embed2", "model.ssm=null"]),
    "sdc": ("tiny_cpu_sd.yaml", []),
    "sda": ("tiny_cpu_sd.yaml", ["model.memory.router_input=hidden", "model.ssm=null",
                                 "model.memory.token_router={enabled: true}"]),
}


def make(variant: str) -> Transformer:
    name, over = VARIANTS[variant]
    torch.manual_seed(0)
    model = Transformer(load_config(CFG / name, over)["model"]).eval()
    torch.manual_seed(1)
    for b in model.blocks:
        if b.memory is not None:
            torch.nn.init.normal_(b.memory.module.readout_o_proj.weight, std=0.05)
            torch.nn.init.normal_(b.memory.module.short_conv.conv.weight, std=0.2)
    return model


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_prefill_and_incremental_match_full_forward(variant):
    model = make(variant)
    x = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        full = model(x)
    st = DecodeState.empty(model, 2, 32)
    torch.testing.assert_close(decode_step(model, st, x), full, rtol=1e-4, atol=1e-4)
    st = DecodeState.empty(model, 2, 32)
    outs = [decode_step(model, st, x[:, :5])]
    for t in range(5, 32, 3):  # mixed block sizes
        outs.append(decode_step(model, st, x[:, t:t + 3]))
    torch.testing.assert_close(torch.cat(outs, 1), full, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("variant", ["wave1", "sdc"])
def test_truncate_then_recompute(variant):
    model = make(variant)
    x = torch.randint(0, 256, (1, 24))
    y = x.clone()
    y[:, 14:] = torch.randint(0, 256, (1, 10))
    st = DecodeState.empty(model, 1, 24)
    decode_step(model, st, x[:, :12])
    decode_step(model, st, x[:, 12:20])     # a "rejected" block
    st.truncate(14)                         # keep 2 of it (x and y agree before 14)
    tail = decode_step(model, st, y[:, 14:])
    with torch.no_grad():
        full = model(y)
    torch.testing.assert_close(tail, full[:, 14:], rtol=1e-4, atol=1e-4)


def test_greedy_generate_matches_naive_loop():
    model = make("sdb")
    prompts = torch.randint(0, 256, (2, 8))
    gen, state = greedy_generate(model, prompts, 10)
    seq = prompts
    with torch.no_grad():
        for _ in range(10):
            seq = torch.cat((seq, model(seq)[:, -1].argmax(-1, keepdim=True)), 1)
    assert torch.equal(gen, seq[:, 8:])
    assert state.length == 17


def test_load_model_roundtrip(tmp_path):
    cfg = load_config(CFG / "tiny_cpu_sd.yaml")
    model = make("sdc")
    path = tmp_path / "checkpoints" / "step_0000005.pt"
    ckpt.save_atomic({"model": model.state_dict(), "config": cfg, "step": 5}, path)
    loaded, lcfg = load_model(tmp_path)
    x = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        assert torch.equal(loaded(x), model(x))
    assert lcfg["_step"] == 5
