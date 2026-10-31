"""M2.T1 dumps: shapes, sharding and consistency with a direct forward (tiny CPU model)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from analysis import dump_codes as dc
from ffnmem import checkpoint as ckpt
from ffnmem.config import load_config
from ffnmem.data.loader import read_meta
from ffnmem.infer import load_model
from ffnmem.model import Transformer

CFG = Path(__file__).resolve().parents[1] / "configs"
VARIANTS = {
    "wave1": ("tiny_cpu.yaml", []),
    "two_mem": ("tiny_cpu.yaml", ["model.memory_layers=[1,2]"]),
    "sdb": ("tiny_cpu_sd.yaml", ["model.memory.router_input=embed2", "model.ssm=null"]),
}


def tiny_checkpoint(out: Path, variant: str = "wave1") -> Path:
    """A tiny model with non-trivial memory readout (o_proj, conv) saved as a checkpoint run dir."""
    name, over = VARIANTS[variant]
    cfg = load_config(CFG / name, over)
    torch.manual_seed(0)
    model = Transformer(cfg["model"]).eval()
    torch.manual_seed(1)
    for b in model.blocks:
        if b.memory is not None:
            torch.nn.init.normal_(b.memory.module.readout_o_proj.weight, std=0.05)
            torch.nn.init.normal_(b.memory.module.short_conv.conv.weight, std=0.2)
    ckpt.save_atomic({"model": model.state_dict(), "config": cfg, "step": 7}, ckpt.ckpt_path(out, 7))
    return out


def test_dump_matches_direct_forward(tmp_path, tiny_data):
    run = tiny_checkpoint(tmp_path / "run", "two_mem")
    out = tmp_path / "dumps"
    meta = dc.main(["--ckpt", str(run), "--data", str(tiny_data), "--run", "t", "--out_root", str(out),
                    "--max_tokens", "300", "--batch", "3", "--shard_positions", "100", "--topk", "8",
                    "--device", "cpu"])
    assert meta["n_windows"] == 10 and meta["n_positions"] == 320  # ceil(300 / 32) windows
    assert [s["positions"] for s in meta["shards"]] == [192, 128] and meta["memory_layers"] == [1, 2]
    _, d = dc.load_dump(out / "t")
    n = 320
    assert d["codes_L1"].shape == (n, 4) and d["codes_L1"].dtype == np.uint8
    assert d["topk_ids"].shape == (n, 8) and d["sink_L2"].shape == (n,)

    # tokens / documents against val.bin
    meta_d = read_meta(tiny_data)
    val = np.fromfile(tiny_data / "val.bin", dtype="<u2").astype(np.int64)
    assert np.array_equal(d["tokens"], val[:n]) and np.array_equal(d["next"], val[1:n + 1])
    doc = d["doc"].astype(np.int64)
    eos = val[:n] == meta_d["eos_id"]
    assert doc[0] == 0 and np.array_equal(np.diff(doc), eos[:-1].astype(np.int64))
    assert np.array_equal(d["pos"], np.tile(np.arange(32), 10))

    # direct forward of the same windows
    model, _ = load_model(run)
    x = torch.from_numpy(val[:n].reshape(10, 32))
    mems = {p: model.blocks[p].memory for p in (1, 2)}
    for m in mems.values():
        m.capture_route_logits = True
        m.collect_diagnostics = True
    norms = {}
    hooks = [m.register_forward_hook(lambda mod, a, o, p=p: norms.__setitem__(p, o.norm(dim=-1)))
             for p, m in mems.items()]
    with torch.no_grad():
        logits = model(x)
    for h in hooks:
        h.remove()
    lp = logits.float().log_softmax(-1).reshape(n, -1)
    assert np.array_equal(d["top1"], lp.argmax(-1).numpy())
    np.testing.assert_allclose(d["topk_logp"].astype(np.float32), lp.topk(8).values.numpy(), atol=2e-3)
    np.testing.assert_allclose(d["logp_next"], lp.gather(1, torch.from_numpy(val[1:n + 1])[:, None])[:, 0].numpy(),
                               atol=1e-5)
    for p, m in mems.items():
        codes = m.module.quantize_route_logits(m.last_route_logits).reshape(n, -1).numpy()
        assert np.array_equal(d[f"codes_L{p}"], codes)
        np.testing.assert_allclose(d[f"memnorm_L{p}"], norms[p].reshape(-1).numpy(), rtol=1e-5, atol=1e-6)
        # mean dumped Sink mass == the wrapper's own diagnostic (same scores, averaged the same way)
        np.testing.assert_allclose(d[f"sink_L{p}"].mean(), m.diagnostics()["sink_mass"], rtol=1e-5)
        assert ((d[f"sink_L{p}"] > 0) & (d[f"sink_L{p}"] < 1)).all()


def test_capture_restores_model(tmp_path):
    run = tiny_checkpoint(tmp_path / "run", "sdb")
    model, _ = load_model(run)
    x = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        ref = model(x)
        with dc.Capture(model) as cap:
            got = model(x)
        assert cap.codes[1].shape == (2, 32, 4) and cap.sink[1].shape == (2, 32)
        after = model(x)
    assert torch.equal(ref, got) and torch.equal(ref, after)
    m = model.blocks[1].memory
    assert not m.capture_route_logits and "_real_route_probabilities" not in m.module.__dict__
