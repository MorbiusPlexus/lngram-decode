"""M2.T3 substitution: own codes reproduce the baseline exactly; other modes change the loss."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from analysis import substitution as sb
from analysis.code_predictability import CodeMLP
from ffnmem.infer import load_model

from .test_analysis_dump import tiny_checkpoint


class OracleMLP(nn.Module):
    """Stands in for a perfect T2 MLP on a token-router model: one-hot of the model's own codes."""

    def __init__(self, model):
        super().__init__()
        self.model = [model]

    def window_logits(self, emb_pad, idx):
        m = self.model[0]
        mem = m.blocks[1].memory
        codes = mem.module.quantize_route_logits(mem.route_logits(mem.router_features(m.tok_emb(idx))))
        return F.one_hot(codes.long(), 1 << mem.bits).float()


def test_modes_on_tiny_model(tmp_path, tiny_data):
    run = tiny_checkpoint(tmp_path / "run", "wave1")
    mlp = CodeMLP(64, routes=4, bits=4, hidden=32, layer=1)
    mlp.save(tmp_path / "mlp_L1.pt")
    res = sb.main(["--ckpt", str(run), "--data", str(tiny_data), "--run", "t", "--mlp", str(tmp_path / "mlp_L1.pt"),
                   "--max_tokens", "640", "--batch", "4", "--n_boot", "100", "--device", "cpu",
                   "--out_dir", str(tmp_path / "res")])
    m = res["modes"]
    assert set(m) == set(sb.MODES) and res["n_windows"] == 20
    assert m["own"]["max_abs_window_diff"] == 0.0 and m["own"]["loss"]["value"] == m["baseline"]["loss"]["value"]
    for mode in ("mlp", "random", "zero"):
        assert m[mode]["max_abs_window_diff"] > 1e-4, mode
        d = m[mode]["delta_vs_baseline"]
        assert d["lo"] <= d["value"] <= d["hi"]
    assert 0.0 <= res["mlp_code_accuracy"]["1"] <= 1.0
    assert json.loads((tmp_path / "res" / "substitution_t.json").read_text())["run"] == "t"

    # the model is untouched afterwards
    model, _ = load_model(run)
    assert "forward" not in model.blocks[1].memory.__dict__


def test_oracle_mlp_codes_equal_baseline(tmp_path, tiny_data):
    run = tiny_checkpoint(tmp_path / "run", "sdb")
    model, _ = load_model(run)
    res = sb.run_substitution(model, tiny_data, modes=("baseline", "mlp", "zero"), layers=[1],
                              mlps={1: OracleMLP(model)}, max_tokens=320, batch=5, n_boot=50,
                              device=torch.device("cpu"))
    assert res["mlp_code_accuracy"]["1"] == 1.0
    assert res["modes"]["mlp"]["max_abs_window_diff"] == 0.0
    assert res["modes"]["zero"]["max_abs_window_diff"] > 0.0
    assert "forward" not in model.blocks[1].memory.__dict__
    assert np.isfinite(res["modes"]["zero"]["loss"]["value"])
