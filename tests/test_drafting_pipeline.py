"""End to end on tiny data: post-hoc distillation (+ resume), offline acceptance, gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from drafting import baselines, build_sketches, gates, offline_acceptance, posthoc_distill
from drafting.posthoc_model import PosthocDrafter
from ffnmem import checkpoint as ckpt
from ffnmem.config import load_config
from ffnmem.infer import load_model
from tests.test_infer import CFG, VARIANTS, make


def save_target(variant: str, root: Path) -> Path:
    name, over = VARIANTS[variant]
    run = root / f"target_{variant}"
    ckpt.save_atomic({"model": make(variant).state_dict(), "config": load_config(CFG / name, over), "step": 1},
                     ckpt.ckpt_path(run, 1))
    return run


def distill_args(target, data, out, steps, extra=()):
    return ["--target", str(target), "--data_dir", str(data), "--out_dir", str(out), "--q_heads", "2",
            "--head_dim", "16", "--tokens", str(8 * 32 * 4), "--global_batch", "8", "--micro_batch", "4",
            "--max_steps", str(steps), "--eval_every", "2", "--ckpt_every", "2", "--val_tokens", "512",
            "--eval_batches", "2", "--log_every", "1", "--kl_chunk", "16", "--device", "cpu",
            "--precision", "fp32", *extra]


def test_distill_runs_and_resumes_exactly(tiny_data, tmp_path):
    target = save_target("wave1", tmp_path)
    straight = posthoc_distill.main(distill_args(target, tiny_data, tmp_path / "a", 4))
    posthoc_distill.main(distill_args(target, tiny_data, tmp_path / "b", 2))
    resumed = posthoc_distill.main(distill_args(target, tiny_data, tmp_path / "b", 4, ["--resume", "auto"]))
    assert straight["step"] == resumed["step"] == 4
    assert np.isfinite(straight["loss"]) and 0 <= straight["val_top1_agree"] <= 1
    model, _ = load_model(target)
    da = PosthocDrafter.load(tmp_path / "a" / "drafter.pt", model)
    db = PosthocDrafter.load(tmp_path / "b" / "drafter.pt", model)
    init = PosthocDrafter(model, q_heads=2, head_dim=16)
    for (k, va), vb in zip(da.state_dict().items(), db.state_dict().values()):
        assert torch.equal(va, vb), k
    assert not torch.equal(da.router.weight, init.router.weight)
    lines = (tmp_path / "a" / "logs" / "train.jsonl").read_text().splitlines()
    assert any("val_top1_agree" in json.loads(x) for x in lines)


def test_distill_with_ssm_and_surrogate(tiny_data, tmp_path):
    target = save_target("wave1", tmp_path)
    out = posthoc_distill.main(distill_args(target, tiny_data, tmp_path / "s", 2, ["--ssm_dim", "8", "--surrogate"]))
    assert np.isfinite(out["loss"])
    d = PosthocDrafter.load(tmp_path / "s" / "drafter.pt", load_model(target)[0])
    assert d.ssm is not None and d.config["ssm_dim"] == 8


def acceptance_args(target, data, root, drafters, extra=()):
    return ["--target", str(target), "--data_dir", str(data), "--name", "tiny", "--drafters", *drafters,
            "--n_prompts", "3", "--n_new", "8", "--open_len", "16", "--long_len", "24", "--long_min_doc", "120",
            "--gammas", "2", "4", "--n_boot", "200", "--cache_dir", str(root / "cache"),
            "--out", str(root / "acceptance_tiny.json"), "--device", "cpu", *extra]


def test_offline_acceptance_and_gates_end_to_end(tiny_data, tmp_path):
    target = save_target("wave1", tmp_path)
    model, _ = load_model(target)
    sk = tmp_path / "sk.npz"
    build_sketches.main(["--target", str(target), "--data_dir", str(tiny_data), "--out", str(sk),
                         "--tokens", "4096", "--batch", "8", "--val_tokens", "0", "--device", "cpu"])
    ng = tmp_path / "ng.npz"
    baselines.main(["--data_dir", str(tiny_data), "--out", str(ng), "--tokens", "4096", "--seq_len", "32",
                    "--buckets3", "4096", "--buckets2", "65536", "--device", "cpu"])
    for ssm, name in ((0, "ph.pt"), (8, "ph_ssm.pt")):
        torch.manual_seed(ssm)
        PosthocDrafter(model, q_heads=2, head_dim=16, ssm_dim=ssm, ssm_chunk=8).save(tmp_path / name)
    timings = tmp_path / "timings.json"
    timings.write_text(json.dumps({"c_draft": {"pld": 0.01, "posthoc": 0.05}}))
    res = offline_acceptance.main(acceptance_args(target, tiny_data, tmp_path, [
        "pld", f"ngram=ngram@{ng}", f"sketch=sketch@{sk}@{tmp_path / 'ph.pt'}+online",
        f"posthoc=posthoc@{tmp_path / 'ph.pt'}", f"posthoc_ssm=posthoc@{tmp_path / 'ph_ssm.pt'}", "oracle"],
        ["--timings", str(timings), "--c_draft", "oracle=0.5"]))
    tt = res["text_types"]
    assert "skipped" in tt["code"] and tt["open"]["n_prompts"] == 3 and tt["long"]["prompt_len"] == 24
    orc = res["drafters"]["oracle"]["results"]
    for t in ("open", "long"):
        for g in (2, 4):
            r = orc[t][str(g)]
            assert r["mean_accepted"] == g and r["ci95"] == [g, g] and len(r["per_prompt"]) == 3
            assert r["speedup"] == pytest.approx((g + 1) / (1 + g * 0.5))
    assert res["drafters"]["pld"]["results"]["open"]["2"]["c_draft"] == 0.01
    assert res["drafters"]["ngram"]["results"]["open"]["2"]["speedup"] is None
    assert res["drafters"]["sketch"]["online"] and res["drafters"]["posthoc"]["kind"] == "posthoc"
    # records are cached: a second call adds a drafter without regenerating
    cache = tmp_path / "cache" / "records_open.npz"
    mtime = cache.stat().st_mtime_ns
    res2 = offline_acceptance.main(acceptance_args(target, tiny_data, tmp_path, ["pld2=pld"]))
    assert cache.stat().st_mtime_ns == mtime and {"pld", "pld2", "oracle"} <= set(res2["drafters"])
    for t in ("open", "long"):
        assert (res2["drafters"]["pld2"]["results"][t]["4"]["per_prompt"]
                == res["drafters"]["pld"]["results"][t]["4"]["per_prompt"])
    out = gates.main(["--acceptance", str(tmp_path / "acceptance_*.json"), "--ssm_file",
                      str(tmp_path / "acceptance_tiny.json"), "--out", str(tmp_path / "gates.json"),
                      "--n_boot", "200"])
    assert out["ssm_gate"]["status"] in ("pass", "fail") and out["ssm_gate"]["text_type"] == "long"
    assert out["live_gate"]["status"] in ("pass", "fail")
    assert {c["drafter"] for c in out["live_gate"]["comparisons"]} == {"posthoc", "posthoc_ssm"}
    assert (tmp_path / "gates.json").exists()


def test_offline_acceptance_sd_head(tiny_data, tmp_path):
    target = save_target("sdb", tmp_path)
    res = offline_acceptance.main(acceptance_args(target, tiny_data, tmp_path, ["sdhead", "sdhead_online=sdhead+online"],
                                                  ["--types", "open"]))
    for label in ("sdhead", "sdhead_online"):
        r = res["drafters"][label]["results"]["open"]["4"]
        assert 0 <= r["mean_accepted"] <= 4 and r["ci95"][0] <= r["mean_accepted"] <= r["ci95"][1]


def test_gates_logic():
    a = np.full(50, 2.2)
    b = np.full(50, 2.0)
    a[::2] += 0.05
    def res(**d):
        return {"target": "x", "drafters": {k: {"kind": kind, "results": {"long": {"4": {"per_prompt": list(v)}},
                                                                      "open": {"4": {"per_prompt": list(v)}}}}
                                            for k, (kind, v) in d.items()}}
    r = res(posthoc_ssm=("posthoc", a), posthoc=("posthoc", b), ngram=("ngram", b + 0.1))
    s = gates.ssm_gate(r, "posthoc_ssm", "posthoc")
    assert s["status"] == "pass" and s["relative_gain"] > 0.1 and s["ci95"][0] > 0
    assert gates.ssm_gate(res(posthoc_ssm=("posthoc", b * 1.02), posthoc=("posthoc", b)), "posthoc_ssm",
                          "posthoc")["status"] == "fail"  # +2% < 5%
    live = gates.live_gate([r])
    assert live["status"] == "pass" and live["best"]["drafter"] == "posthoc_ssm"
    assert gates.live_gate([res(posthoc=("posthoc", b), ngram=("ngram", b + 0.1))])["status"] == "fail"
