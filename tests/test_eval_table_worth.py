"""table_worth: final eval records, budget flags, recomputed paired CIs."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from eval.table_worth import last_eval, main
from ffnmem import checkpoint as ckpt
from ffnmem.config import load_config
from ffnmem.model import Transformer

TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny_cpu.yaml"


def fake_run(d: Path, evals: list[tuple[int, float, int]], torn: bool = False) -> Path:
    (d / "logs").mkdir(parents=True)
    with open(d / "logs" / "train.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"event": "start", "step": 0}) + "\n")
        for step, loss, tokens in evals:
            f.write(json.dumps({"step": step, "loss": loss + 0.1}) + "\n")
            f.write(json.dumps({"event": "eval", "step": step, "val_loss": loss, "val_tokens": 20_000_000,
                                "tokens": tokens}) + "\n")
        if torn:
            f.write('{"step": 99')
    return d


def test_logs_mode_flags_budgets(tmp_path):
    runs = {
        "D270": fake_run(tmp_path / "d270", [(1000, 3.2, 524_288_000), (3815, 2.90, 2_000_158_720)], torn=True),
        "Run2": fake_run(tmp_path / "run2", [(3815, 2.95, 2_000_158_720)]),
        "SB": fake_run(tmp_path / "sb", [(1907, 3.00, 999_817_216)]),
    }
    assert last_eval(runs["D270"])["val_loss"] == 2.90
    out = tmp_path / "tw.json"
    main(["--runs"] + [f"{k}={v}" for k, v in runs.items()] + ["--out", str(out)])
    res = json.loads(out.read_text())
    comps = {(c["a"], c["b"]): c for c in res["comparisons"]}
    assert set(comps) == {("D270", "Run2"), ("D270", "SB")}
    c = comps[("D270", "Run2")]
    assert c["comparable"] and abs(c["diff"] + 0.05) < 1e-12
    assert not comps[("D270", "SB")]["comparable"]
    assert "not comparable" in comps[("D270", "SB")]["notes"][0]


def test_recompute_paired_ci(tiny_data, tmp_path):
    cfg = load_config(TINY)
    cfg["train"]["data_dir"] = str(tiny_data)
    paths = {}
    for name, seed in [("D270", 0), ("Run2", 1), ("Run2b", 1)]:
        torch.manual_seed(seed)
        m = Transformer(cfg["model"])
        p = tmp_path / name / "checkpoints" / "step_0000020.pt"
        ckpt.save_atomic({"model": m.state_dict(), "config": cfg, "step": 20}, p)
        paths[name] = p
    out = tmp_path / "tw.json"
    main(["--recompute", "--device", "cpu", "--precision", "fp32", "--val_tokens", "1024", "--n_boot", "200",
          "--pairs", "D270:Run2", "Run2:Run2b", "--out", str(out)]
         + ["--runs"] + [f"{k}={v}" for k, v in paths.items()])
    res = json.loads(out.read_text())
    assert res["mode"] == "recompute"
    a, b = res["comparisons"]
    assert a["ci_lo"] <= a["paired_diff"] <= a["ci_hi"] and a["comparable"] and a["n_windows"] == 31
    assert abs(a["paired_diff"] - a["diff"]) < 1e-9
    assert b["paired_diff"] == 0 and not b["excludes_zero"]
