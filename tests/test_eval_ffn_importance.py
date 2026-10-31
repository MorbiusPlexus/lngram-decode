"""FFN zeroing: exact, local to one block, and fully undone afterwards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from eval.ffn_importance import ffn_importance, main, window_losses, zero_ffn
from ffnmem import checkpoint as ckpt
from ffnmem.config import load_config
from ffnmem.data.loader import ValWindows
from ffnmem.model import Transformer

TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny_cpu.yaml"


def make_model() -> Transformer:
    torch.manual_seed(0)
    return Transformer(load_config(TINY)["model"]).eval()


class Zero(torch.nn.Module):
    def forward(self, h):
        return torch.zeros_like(h)


def capture_ffn_outputs(model, x):
    outs, hooks = {}, []
    for pos, b in enumerate(model.blocks):
        hooks.append(b.ffn.register_forward_hook(lambda m, i, o, pos=pos: outs.__setitem__(pos, o.clone())))
    with torch.no_grad():
        logits = model(x)
    for h in hooks:
        h.remove()
    return logits, outs


def test_zero_ffn_is_local_and_restored():
    model = make_model()
    x = torch.randint(0, 256, (2, 32))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    base, base_outs = capture_ffn_outputs(model, x)
    k = 2
    with zero_ffn(model, k):
        zeroed, outs = capture_ffn_outputs(model, x)
    # FFNs before the zeroed block see identical inputs; the zeroed one contributes nothing
    for pos in range(k):
        torch.testing.assert_close(outs[pos], base_outs[pos])
    assert torch.count_nonzero(outs[k]) == 0
    assert not torch.allclose(zeroed, base)
    # reference: the same forward with block k's FFN removed by hand
    ffn = model.blocks[k].ffn
    model.blocks[k].ffn = Zero()
    with torch.no_grad():
        ref = model(x)
    model.blocks[k].ffn = ffn
    torch.testing.assert_close(zeroed, ref)
    # restored: weights untouched, no hooks left, output back to baseline
    for name, v in model.state_dict().items():
        assert torch.equal(v, before[name])
    assert all(len(b.ffn._forward_hooks) == 0 for b in model.blocks)
    with torch.no_grad():
        assert torch.equal(model(x), base)


def test_ffn_importance_and_cli(tiny_data, tmp_path):
    model = make_model()
    val = ValWindows(tiny_data, 32, 1024)
    base = window_losses(model, val, batch_size=4, precision="fp32")
    res = ffn_importance(model, val, batch_size=4, precision="fp32", layers=[0, 3], n_boot=200, log=lambda s: None)
    assert res["n_windows"] == len(base) == val.num_windows
    assert res["base_loss"] == pytest.approx(base.mean())
    assert [r["pos"] for r in res["layers"]] == [0, 3]
    for r in res["layers"]:
        assert r["ci_lo"] <= r["delta"] <= r["ci_hi"]
        assert r["delta"] == pytest.approx(r["loss"] - res["base_loss"], abs=1e-9)
    np.testing.assert_array_equal(window_losses(model, val, 4, "fp32"), base)  # nothing left zeroed

    cfg = load_config(TINY)
    cfg["train"]["data_dir"] = str(tiny_data)
    ckpt.save_atomic({"model": model.state_dict(), "config": cfg, "step": 7},
                     tmp_path / "run" / "checkpoints" / "step_0000007.pt")
    main(["--checkpoint", str(tmp_path / "run"), "--run", "tiny", "--device", "cpu", "--precision", "fp32",
          "--val_tokens", "1024", "--n_boot", "100", "--out_dir", str(tmp_path / "results")])
    out = json.loads((tmp_path / "results" / "ffn_importance_tiny.json").read_text())
    assert out["step"] == 7 and len(out["layers"]) == 4
    assert (tmp_path / "results" / "ffn_importance_tiny.png").stat().st_size > 0
