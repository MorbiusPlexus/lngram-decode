"""Self-drafting training end to end (tiny CPU): resume, data/position order, logging, retention."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ffnmem import checkpoint as ckpt
from ffnmem.train import main as train_main

ROOT = Path(__file__).resolve().parents[1]
TINY_SD = str(ROOT / "configs" / "tiny_cpu_sd.yaml")


def run(tiny_data, out, *extra):
    return train_main(["--config", TINY_SD, "--device", "cpu", "--data_dir", str(tiny_data), "--out_dir", str(out), *extra])


def losses(result):
    return {h["step"]: h["loss"] for h in result["history"]}


def records(out):
    return [json.loads(l) for l in (Path(out) / "logs" / "train.jsonl").read_text().splitlines()]


def test_sd_resume_matches_uninterrupted(tiny_data, tmp_path):
    full = run(tiny_data, tmp_path / "full")
    run(tiny_data, tmp_path / "part", "--max_steps", "10")
    resumed = run(tiny_data, tmp_path / "part", "--resume", "auto")
    a, b = losses(full), losses(resumed)
    for s in range(11, 21):
        assert b[s] == pytest.approx(a[s], rel=1e-6, abs=1e-7), s
    ka = [r for r in records(tmp_path / "full") if r.get("step") == 20 and "loss/draft_kl" in r]
    kb = [r for r in records(tmp_path / "part") if r.get("step") == 20 and "loss/draft_kl" in r]
    assert ka and kb and ka[0]["loss/draft_kl"] == pytest.approx(kb[0]["loss/draft_kl"], rel=1e-6)


def test_sd_independent_of_micro_batch(tiny_data, tmp_path):
    a = run(tiny_data, tmp_path / "mb2", "--max_steps", "5")
    b = run(tiny_data, tmp_path / "mb4", "--max_steps", "5", "--set", "train.micro_batch=4")
    la, lb = losses(a), losses(b)
    for s in range(1, 6):
        assert lb[s] == pytest.approx(la[s], rel=1e-5), s
    ra = {r["step"]: r for r in records(tmp_path / "mb2") if "loss/draft_kl" in r}
    rb = {r["step"]: r for r in records(tmp_path / "mb4") if "loss/draft_kl" in r}
    assert ra[5]["loss/draft_kl"] == pytest.approx(rb[5]["loss/draft_kl"], rel=1e-5)


def test_sd_logs_and_params_json(tiny_data, tmp_path):
    out = tmp_path / "log"
    run(tiny_data, out, "--max_steps", "10")
    recs = records(out)
    train_recs = [r for r in recs if r.get("event") is None and "loss" in r]
    for key in ("loss/lm", "loss/draft_kl", "draft/top1_agree", "draft/accept_proxy"):
        assert key in train_recs[0], key
    assert train_recs[0]["loss"] == train_recs[0]["loss/lm"]
    ev = [r for r in recs if r.get("event") == "eval"][0]
    assert "val_draft_kl" in ev and "val_draft_top1_agree" in ev
    p = json.loads((out / "params.json").read_text())
    assert p["breakdown"]["draft"] > 0 and p["breakdown"]["ssm"] > 0 and p["routes"] == [4]


def test_save_steps_and_optimizer_stripping(tiny_data, tmp_path):
    out = tmp_path / "keep"
    run(tiny_data, out, "--set", "train.ckpt_every=4", "--set", "train.keep_last=1", "--set", "train.keep_every=0",
        "--set", "train.save_steps=[6]", "--set", "train.strip_old_optimizer=true")
    found = ckpt.list_checkpoints(out)
    assert [s for s, _ in found] == [6, 20]
    assert ckpt.load(found[0][1])["optimizer"] is None
    assert ckpt.load(found[-1][1])["optimizer"] is not None


@pytest.mark.slow
def test_sd_ddp_two_process_matches_single(tiny_data, tmp_path):
    from ffnmem.distlaunch import run as dist_run

    single = losses(run(tiny_data, tmp_path / "one", "--max_steps", "4"))
    proc = dist_run(2, ["-m", "ffnmem.train", "--config", TINY_SD, "--device", "cpu", "--data_dir", str(tiny_data),
                        "--out_dir", str(tmp_path / "two"), "--max_steps", "4"], timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    ddp = {r["step"]: r["loss"] for r in records(tmp_path / "two") if "loss" in r and r.get("event") is None}
    for s in range(1, 5):
        assert ddp[s] == pytest.approx(single[s], rel=1e-5), s
