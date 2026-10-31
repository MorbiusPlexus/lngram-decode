"""M5.T1/T2: bundle export -> load roundtrip; the reference CPU drafter equals the model's draft head."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from drafting.cpu_drafter import RefDrafter
from drafting.export_drafter import export, load_bundle, read_header
from drafting.posthoc_model import PosthocDrafter
from drafting.sketches import Sketch
from ffnmem.config import load_config
from ffnmem.model import Transformer

CFG = Path(__file__).resolve().parents[1] / "configs"
VARIANTS = {
    "wave1": ("tiny_cpu.yaml", []),
    "sdb": ("tiny_cpu_sd.yaml", ["model.memory.router_input=embed2", "model.ssm=null"]),
    "sdc": ("tiny_cpu_sd.yaml", []),
    "sda": ("tiny_cpu_sd.yaml", ["model.memory.router_input=hidden", "model.ssm=null",
                                 "model.memory.token_router={enabled: true}"]),
}


def make_model(variant: str, seq_len: int = 96, seed: int = 0) -> Transformer:
    name, over = VARIANTS[variant]
    torch.manual_seed(seed)
    model = Transformer(load_config(CFG / name, over + [f"model.seq_len={seq_len}"])["model"]).eval()
    torch.manual_seed(seed + 1)
    for b in model.blocks:
        if b.memory is not None:
            torch.nn.init.normal_(b.memory.module.readout_o_proj.weight, std=0.05)
            torch.nn.init.normal_(b.memory.module.short_conv.conv.weight, std=0.2)
    if model.draft is not None:  # sharper draft readout so argmaxes are not near-ties
        torch.nn.init.normal_(model.draft.w_o.weight, std=0.2)
    return model


def round_tables_fp16(model) -> None:
    with torch.no_grad():
        for b in model.blocks:
            if b.memory is not None:
                for t in b.memory.module.tables.values():
                    t.weight.copy_(t.weight.half().float())


def random_sketch(model, k: int = 4, seed: int = 0, empty_frac: float = 0.3) -> Sketch:
    layer = next(i for i, b in enumerate(model.blocks) if b.memory is not None)
    m = model.blocks[layer].memory.module
    sk = Sketch.empty(m.num_routes, m.bits_per_route, m.ngrams, layer, k, model.cfg["vocab"])
    rng = np.random.default_rng(seed)
    for n in m.ngrams:
        sk.ids[n][:] = rng.integers(0, model.cfg["vocab"], sk.ids[n].shape)
        c = rng.integers(1, 50, sk.counts[n].shape)
        c[rng.random(c.shape) < empty_frac] = 0
        sk.counts[n][:] = c
    return sk


def make_posthoc(ssm_dim: int = 0) -> tuple[Transformer, PosthocDrafter]:
    target = make_model("wave1")
    torch.manual_seed(7)
    d = PosthocDrafter(target, q_heads=2, head_dim=16, ssm_dim=ssm_dim, ssm_chunk=8).eval()
    torch.nn.init.normal_(d.draft.w_o.weight, std=0.2)
    return target, d


def bundle_for(tmp_path, variant: str, **kw):
    model = make_model(variant)
    round_tables_fp16(model)
    path = export(tmp_path / f"{variant}.bin", model, **kw)
    return model, load_bundle(path)


# ---------------------------------------------------------------------- tests
def test_roundtrip_arrays_and_header(tmp_path):
    model = make_model("sdc")
    sk = random_sketch(model)
    path = export(tmp_path / "b.bin", model, sketch=sk, shortlist_cap=16, source={"note": "x"})
    head = read_header(path)
    b = load_bundle(path)
    c = b.config
    assert c["kind"] == "sd_c" and c["router_kind"] == "embed2_ssm" and c["ssm_dim"] == 8 and c["query_ssm"]
    assert c["routes"] == 4 and c["ngrams"] == [2, 3] and c["groups"] == 1 and c["shortlist_cap"] == 16
    assert b.source == {"note": "x"}
    for name, meta in head["arrays"].items():
        assert meta["offset"] % 64 == 0
        assert list(b.arrays[name].shape) == meta["shape"]
    m = model.blocks[1].memory.module
    np.testing.assert_array_equal(b.arrays["embedding"], model.tok_emb.weight.detach().numpy())
    assert b.arrays["table_3"].dtype == np.float16
    np.testing.assert_array_equal(b.arrays["table_3"], m.tables["ngram_3"].weight.detach().half().numpy())
    np.testing.assert_array_equal(b.arrays["kv_proj"], m.readout_kv_proj.weight.detach().numpy())
    np.testing.assert_array_equal(b.arrays["ssm_decay"], model.ssm.decay().detach().numpy())
    np.testing.assert_array_equal(b.arrays["sketch_ids_2"], sk.ids[2])
    np.testing.assert_array_equal(b.arrays["sketch_counts_3"], sk.counts[3])
    assert len(b.arrays["fallback"]) == 16
    b2 = load_bundle(path, mmap=False)
    for k in b.arrays:
        np.testing.assert_array_equal(b.arrays[k], b2.arrays[k])
    t = b.tensors()
    assert t["sketch_ids_2"].dtype == torch.int32 and t["table_2"].dtype == torch.float16


@pytest.mark.parametrize("variant", ["sdb", "sdc", "sda"])
def test_reference_equals_draft_head_full_vocab(tmp_path, variant):
    model, b = bundle_for(tmp_path, variant)
    assert b.config["kind"] == {"sdb": "sd_b", "sdc": "sd_c", "sda": "sd_a"}[variant]
    x = torch.randint(0, 256, (1, 60))
    with torch.no_grad():
        ref = model.draft_logits(x, torch.zeros(60, dtype=torch.long), torch.arange(60))
    _, ids, lg = RefDrafter(b).trace(x[0].tolist())
    assert (ids == np.arange(256)).all()
    torch.testing.assert_close(torch.from_numpy(lg), ref, rtol=1e-5, atol=1e-5)
    assert (lg.argmax(1) == ref.argmax(1).numpy()).all()


def test_fp32_tables_match_unrounded_model(tmp_path):
    model = make_model("sdb")
    b = load_bundle(export(tmp_path / "f32.bin", model, table_dtype="fp32"))
    x = torch.randint(0, 256, (1, 40))
    with torch.no_grad():
        ref = model.draft_logits(x, torch.zeros(40, dtype=torch.long), torch.arange(40))
    _, _, lg = RefDrafter(b).trace(x[0].tolist())
    torch.testing.assert_close(torch.from_numpy(lg), ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("ssm_dim", [0, 8])
def test_reference_equals_posthoc_drafter(tmp_path, ssm_dim):
    target, d = make_posthoc(ssm_dim)
    round_tables_fp16(target)
    b = load_bundle(export(tmp_path / "ph.bin", target, posthoc=d))
    assert b.config["kind"] == "posthoc" and b.config["ssm_dim"] == ssm_dim
    x = torch.randint(0, 256, (1, 50))
    with torch.no_grad():
        ref = d(x, torch.zeros(50, dtype=torch.long), torch.arange(50))
        codes = d.codes(x)[0].numpy()
    c, _, lg = RefDrafter(b).trace(x[0].tolist())
    np.testing.assert_array_equal(c, codes)
    torch.testing.assert_close(torch.from_numpy(lg), ref, rtol=1e-5, atol=1e-5)
    # target codes override (offline-harness live rule): same as the module with codes=...
    tc = torch.randint(0, 16, (1, 50, 4), dtype=torch.int32)
    with torch.no_grad():
        ref2 = d(x, torch.zeros(50, dtype=torch.long), torch.arange(50), codes=tc)
    _, _, lg2 = RefDrafter(b).trace(x[0].tolist(), codes=tc[0].numpy())
    torch.testing.assert_close(torch.from_numpy(lg2), ref2, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("variant", ["sdb", "sdc"])
def test_shortlist_restriction_equals_draft_head(tmp_path, variant):
    model = make_model(variant)
    round_tables_fp16(model)
    sk = random_sketch(model, seed=3)
    b = load_bundle(export(tmp_path / "s.bin", model, sketch=sk, shortlist_cap=16))
    x = torch.randint(0, 256, (1, 60))
    with torch.no_grad():
        ref = model.draft_logits(x, torch.zeros(60, dtype=torch.long), torch.arange(60)).numpy()
    codes, ids, lg = RefDrafter(b).trace(x[0].tolist())
    assert ids.shape == (60, 16)
    contained = 0
    for t in range(60):
        sl = ids[t][ids[t] >= 0]
        assert len(sl) == len(set(sl.tolist()))
        np.testing.assert_allclose(lg[t, :len(sl)], ref[t, sl], rtol=1e-5, atol=1e-5)
        if ref[t].argmax() in set(sl.tolist()):
            contained += 1
            assert sl[lg[t, :len(sl)].argmax()] == ref[t].argmax()
    assert contained > 0
    # shortlist order: sketch candidates by (summed count desc, id asc), then fallback tokens
    d = RefDrafter(b)
    t = 30
    rows = {n: sk.rows_at(codes, n)[t] for n in sk.ngrams}
    score: dict = {}
    for n, r in rows.items():
        for i, c in zip(sk.ids[n][r].ravel(), sk.counts[n][r].ravel()):
            if c > 0:
                score[int(i)] = score.get(int(i), 0) + int(c)
    expect = sorted(score, key=lambda v: (-score[v], v))[:16]
    fb = [int(f) for f in b.arrays["fallback"] if int(f) not in expect]
    expect = expect + fb[:16 - len(expect)]
    assert ids[t].tolist() == expect
    assert d.cap == 16
