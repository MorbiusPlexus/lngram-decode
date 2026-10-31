from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from ffnmem.attention import attend
from ffnmem.config import load_config
from ffnmem.model import Transformer, count_params
from ffnmem.train import build_optimizer

CFG = Path(__file__).resolve().parents[1] / "configs"


def tiny(**changes):
    cfg = load_config(CFG / "tiny_cpu.yaml")
    cfg["model"].update(changes)
    return cfg


def nontrivial_memory(model):
    """Output projection is zero at init; perturb it so the branch contributes."""
    for m in model.memory_modules():
        torch.nn.init.normal_(m.module.readout_o_proj.weight, std=0.02)


def test_per_layer_widths_and_deletion():
    cfg = load_config(CFG / "w1_mem_cut_near.yaml")["model"]
    with torch.device("meta"):
        model = Transformer(cfg)
    for block in model.blocks:
        expected = 1536 if block.layer_id in (2, 3, 4, 5, 12, 13, 14, 15) else 2816
        assert block.ffn.w1.weight.shape == (expected, 1024)
        assert block.ffn.w2.weight.shape == (1024, expected)
        assert block.attn.wq.weight.shape == (1536, 1024)
        assert block.attn.wk.weight.shape == (256, 1024)
        assert block.attn.wo.weight.shape == (1024, 1536)
    deleted = copy.deepcopy(cfg)
    deleted["layers"] = [i for i in range(24) if i not in (3, 4, 5, 13, 14)]
    with torch.device("meta"):
        model = Transformer(deleted)
    assert len(model.blocks) == 19
    assert [b.layer_id for b in model.blocks] == deleted["layers"]
    assert [pos for pos, b in enumerate(model.blocks) if b.memory is not None] == [1, 11]


def test_memory_only_where_configured():
    cfg = load_config(CFG / "w1_mem_full.yaml")["model"]
    with torch.device("meta"):
        model = Transformer(cfg)
    assert [i for i, b in enumerate(model.blocks) if b.memory is not None] == [1, 11]
    with torch.device("meta"):
        dense = Transformer(load_config(CFG / "w1_dense_full.yaml")["model"])
    assert not dense.memory_modules()


@pytest.mark.parametrize("with_memory", [True, False])
def test_causality(with_memory):
    torch.manual_seed(0)
    cfg = tiny() if with_memory else tiny(memory=None, memory_layers=[])
    model = Transformer(cfg["model"]).eval()
    nontrivial_memory(model)
    x = torch.randint(0, 256, (2, 32))
    t = 17
    x2 = x.clone()
    x2[:, t] = (x2[:, t] + 1) % 256
    with torch.no_grad():
        a, b = model(x), model(x2)
    assert torch.equal(a[:, :t], b[:, :t])
    assert not torch.allclose(a[:, t:], b[:, t:])


def test_memory_block_equals_plain_block_at_init():
    torch.manual_seed(0)
    model = Transformer(tiny()["model"]).train()
    block = model.blocks[1]
    assert block.memory is not None
    h = torch.randn(2, 32, 64)
    out_mem = block(h, model.rope_cos, model.rope_sin)
    mem = block.memory
    block.memory = None
    out_plain = block(h, model.rope_cos, model.rope_sin)
    block.memory = mem
    assert torch.equal(out_mem, out_plain)


@pytest.mark.parametrize("act_ckpt", [False, True])
def test_memory_gradient_flow(act_ckpt):
    torch.manual_seed(0)
    cfg = tiny()
    model = Transformer(cfg["model"]).train()
    model.act_ckpt = act_ckpt
    opt = build_optimizer(model, dict(cfg["train"], lr=3e-3), torch.device("cpu"), zero1=False)
    for _ in range(5):
        x = torch.randint(0, 256, (4, 33))
        _, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    mem = model.blocks[1].memory
    checks = {
        "tables": [p.grad for p in mem.module.tables.parameters()],
        "router": [mem.router.weight.grad],
        "readout_q": [mem.module.readout_q_proj.weight.grad],
        "readout_kv": [mem.module.readout_kv_proj.weight.grad],
        "readout_o": [mem.module.readout_o_proj.weight.grad],
        "conv": [mem.module.short_conv.conv.weight.grad],
    }
    for name, grads in checks.items():
        assert all(g is not None and float(g.abs().sum()) > 0 for g in grads), name


def test_activation_checkpointing_is_exact():
    torch.manual_seed(0)
    model = Transformer(tiny()["model"]).train()
    nontrivial_memory(model)
    x = torch.randint(0, 256, (2, 33))
    grads = []
    for ckpt_on in (False, True):
        model.zero_grad()
        model.act_ckpt = ckpt_on
        _, loss = model(x[:, :-1], x[:, 1:])
        loss.backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    router = [n for n in grads[0] if "router" in n]
    assert router and all(float(grads[1][n].abs().sum()) > 0 for n in router)
    for n in grads[0]:
        torch.testing.assert_close(grads[0][n], grads[1][n], rtol=1e-5, atol=1e-7)


def test_param_groups():
    cfg = tiny()
    model = Transformer(cfg["model"])
    opt = build_optimizer(model, cfg["train"], torch.device("cpu"), zero1=False)
    by = {g["group"]: g for g in opt.param_groups}
    tables = {id(p) for p in model.blocks[1].memory.module.tables.parameters()}
    assert {id(p) for p in by["tables"]["params"]} == tables
    assert by["tables"]["lr_mult"] == 4.0 and by["tables"]["weight_decay"] == 0.0
    assert by["router"]["weight_decay"] == 0.0 and len(by["router"]["params"]) == 1
    assert by["decay"]["weight_decay"] == 0.1
    assert all(p.ndim == 1 for p in by["no_decay"]["params"])
    n_opt = sum(len(g["params"]) for g in opt.param_groups)
    assert n_opt == len(list(model.parameters()))


@pytest.mark.parametrize("n_heads,n_kv", [(4, 4), (4, 2), (6, 2)])
def test_sdpa_matches_naive_fp32(n_heads, n_kv):
    torch.manual_seed(0)
    q = torch.randn(2, n_heads, 48, 16)
    k = torch.randn(2, n_kv, 48, 16)
    v = torch.randn(2, n_kv, 48, 16)
    torch.testing.assert_close(attend(q, k, v, "sdpa"), attend(q, k, v, "naive"), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_sdpa_matches_naive_fp16_gpu():
    torch.manual_seed(0)
    q = torch.randn(2, 24, 512, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 4, 512, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 4, 512, 64, device="cuda", dtype=torch.float16)
    a = attend(q, k, v, "sdpa").float()
    b = attend(q, k, v, "naive").float()
    assert (a - b).norm() / b.norm() < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_fp16_memory_forward_backward_gpu():
    torch.manual_seed(0)
    cfg = load_config(CFG / "local_gpu_sanity.yaml",
                      ["model.n_layers=2", "model.memory_layers=[1]", "model.seq_len=256"])
    model = Transformer(cfg["model"]).cuda().train()
    model.act_ckpt = True
    nontrivial_memory(model)
    x = torch.randint(0, 32768, (1, 257), device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        _, loss = model(x[:, :-1], x[:, 1:])
    loss.backward()
    assert torch.isfinite(loss)
    for n, p in model.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n


def _has_flash() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _has_flash(), reason="flash-attn not installed (optional; cluster only)")
def test_flash_matches_sdpa_fp16():
    torch.manual_seed(0)
    shape_q, shape_kv = (2, 24, 1024, 64), (2, 4, 1024, 64)
    q = torch.randn(shape_q, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(shape_kv, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(shape_kv, device="cuda", dtype=torch.float16, requires_grad=True)
    outs, grads = [], []
    for backend in ("sdpa", "flash"):
        q.grad = k.grad = v.grad = None
        o = attend(q, k, v, backend)
        o.float().pow(2).mean().backward()
        outs.append(o.detach().float())
        grads.append([t.grad.float().clone() for t in (q, k, v)])
    assert (outs[0] - outs[1]).norm() / outs[0].norm() < 1e-2
    for a, b in zip(*grads):
        assert (a - b).norm() / a.norm() < 1e-2


def rank1_tiny():
    cfg = tiny()
    cfg["model"]["memory_layers"] = [1, 2]
    cfg["model"]["memory_overrides"] = {1: {"rank1": {"dim": 8}}, 2: {"mem_dim": 8}}
    return cfg


def test_memory_overrides_and_rank1_shapes():
    model = Transformer(rank1_tiny()["model"])
    m1, m2 = model.blocks[1].memory, model.blocks[2].memory
    assert m1.rank1_dim == 8 and m2.rank1_dim == 0
    assert m1.module.memory_dim == 16 and m2.module.memory_dim == 8
    assert m1.rank1_tables["ngram_3"].weight.shape == (4 * 16**3, 16)
    # rank-1 tables are counted as memory tables and trained like tables
    counts = count_params(model)
    rank1 = sum(p.numel() for p in m1.rank1_tables.parameters())
    v2 = sum(p.numel() for m in (m1, m2) for p in m.module.tables.parameters())
    assert counts["memory_tables"] == v2 + rank1
    opt = build_optimizer(model, rank1_tiny()["train"], torch.device("cpu"), zero1=False)
    tables = {id(p) for g in opt.param_groups if g["group"] == "tables" for p in g["params"]}
    assert {id(p) for p in m1.rank1_tables.parameters()} <= tables


def test_rank1_zero_at_init_and_causal():
    torch.manual_seed(0)
    model = Transformer(rank1_tiny()["model"]).eval()
    mem = model.blocks[1].memory
    h = torch.randn(2, 32, 64)
    hn = model.blocks[1].attn_norm(h)
    codes = mem.module.quantize_route_logits(mem.route_logits(hn))
    assert torch.count_nonzero(mem._rank1(hn, codes)) == 0  # U_out zero-init
    torch.nn.init.normal_(mem.rank1_out.weight, std=0.02)
    nontrivial_memory(model)
    x = torch.randint(0, 256, (2, 32))
    x2 = x.clone()
    x2[:, 20] = (x2[:, 20] + 1) % 256
    with torch.no_grad():
        a, b = model(x), model(x2)
    assert torch.equal(a[:, :20], b[:, :20]) and not torch.allclose(a[:, 20:], b[:, 20:])


def test_rank1_gradient_flow():
    torch.manual_seed(0)
    cfg = rank1_tiny()
    model = Transformer(cfg["model"]).train()
    opt = build_optimizer(model, dict(cfg["train"], lr=3e-3), torch.device("cpu"), zero1=False)
    for _ in range(5):
        x = torch.randint(0, 256, (4, 33))
        _, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    mem = model.blocks[1].memory
    for name, p in [("rank1_out", mem.rank1_out.weight), ("rank1_in", mem.rank1_in.weight)] + [
        (f"rank1_table_{k}", t.weight) for k, t in mem.rank1_tables.items()
    ]:
        assert p.grad is not None and float(p.grad.abs().sum()) > 0, name


def test_rank1_config_is_param_matched():
    from ffnmem.params import check_match, model_counts

    ref = model_counts(load_config(CFG / "w1_mem_full.yaml")["model"])["non_embedding"]
    counts = model_counts(load_config(CFG / "w2_mem_rank1.yaml")["model"])
    check_match(counts, ref)
    assert counts["non_embedding"] == 369_767_808
