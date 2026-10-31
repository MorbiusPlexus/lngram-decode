from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from ffnmem import checkpoint as ckpt
from ffnmem.config import load_config
from ffnmem.model import Transformer
from ffnmem.train import lr_at, main as train_main, schedule_for

ROOT = Path(__file__).resolve().parents[1]
TINY = str(ROOT / "configs" / "tiny_cpu.yaml")


def run(tiny_data, out, *extra):
    return train_main(["--config", TINY, "--device", "cpu", "--data_dir", str(tiny_data), "--out_dir", str(out), *extra])


def losses(result):
    return {h["step"]: h["loss"] for h in result["history"]}


def test_schedule_shape():
    s = schedule_for({"global_batch": 256, "seq_len": 2048, "tokens_total": 2_000_000_000, "warmup_min_steps": 100,
                      "warmup_frac": 0.02, "lr": 6e-4, "min_lr_ratio": 0.1})
    assert s["total_steps"] == 3815 and s["warmup_steps"] == 100
    assert lr_at(0, s) == pytest.approx(6e-6)
    assert lr_at(99, s) == pytest.approx(6e-4)
    assert lr_at(3814, s) == pytest.approx(6e-5, rel=1e-3)
    assert all(lr_at(i, s) >= lr_at(i + 1, s) for i in range(100, 3814))


def test_resume_matches_uninterrupted(tiny_data, tmp_path):
    full = run(tiny_data, tmp_path / "full")
    assert full["step"] == 20
    part = run(tiny_data, tmp_path / "part", "--max_steps", "10")
    assert part["step"] == 10
    assert ckpt.latest_checkpoint(tmp_path / "part").name == "step_0000010.pt"
    resumed = run(tiny_data, tmp_path / "part", "--resume", "auto")
    a, b = losses(full), losses(resumed)
    assert sorted(b) == list(range(11, 21))
    for s in range(11, 21):
        assert b[s] == pytest.approx(a[s], rel=1e-6, abs=1e-7), s
    # checkpoint content is complete
    state = ckpt.load(ckpt.latest_checkpoint(tmp_path / "part"))
    for key in ("model", "optimizer", "scaler", "schedule", "rng", "step", "data_offset", "config", "git"):
        assert key in state
    assert state["step"] == 20


def test_resume_auto_without_checkpoint_starts_fresh(tiny_data, tmp_path):
    r = run(tiny_data, tmp_path / "fresh", "--resume", "auto", "--max_steps", "2")
    assert r["step"] == 2


def test_retention(tiny_data, tmp_path):
    out = tmp_path / "ret"
    run(tiny_data, out, "--set", "train.ckpt_every=2", "--set", "train.keep_last=3", "--set", "train.keep_every=10")
    steps = [s for s, _ in ckpt.list_checkpoints(out)]
    assert steps == [10, 16, 18, 20]
    assert not list((out / "checkpoints").glob("*.tmp"))


def test_init_from_new_schedule(tiny_data, tmp_path):
    run(tiny_data, tmp_path / "src", "--max_steps", "10")
    src = ckpt.latest_checkpoint(tmp_path / "src")
    r = run(tiny_data, tmp_path / "ext", "--init_from", str(src), "--new_schedule", "--max_steps", "3")
    assert r["step"] == 3
    state = ckpt.load(ckpt.latest_checkpoint(tmp_path / "ext"))
    assert state["data_offset"] == 10  # continues after the source run's data
    with pytest.raises(SystemExit):
        run(tiny_data, tmp_path / "ext2", "--init_from", str(src))  # --new_schedule required


def test_logs_written(tiny_data, tmp_path):
    run(tiny_data, tmp_path / "log", "--max_steps", "10")
    recs = [json.loads(l) for l in (tmp_path / "log" / "logs" / "train.jsonl").read_text().splitlines()]
    evals = [r for r in recs if r.get("event") == "eval"]
    assert evals and "val_loss" in evals[0] and "mem1/sink_mass" in evals[0]
    assert "mem1/eff_codes_mean" in evals[0] and "table_grad_norm" in evals[0]
    assert recs[0]["event"] == "start" and recs[-1]["event"] == "end"


@pytest.mark.parametrize("with_memory", [True, False])
def test_overfit_single_sequence(with_memory):
    torch.manual_seed(0)
    over = ["model.seq_len=512"] + ([] if with_memory else ["model.memory=null"])
    cfg = load_config(TINY, over)
    model = Transformer(cfg["model"]).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.0)
    seq = torch.randint(0, 256, (1, 513), generator=torch.Generator().manual_seed(1))
    loss_v = None
    for step in range(300):
        _, loss = model(seq[:, :-1], seq[:, 1:])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_v = float(loss.detach())
        if loss_v < 0.1:
            break
    assert loss_v < 0.1, f"loss {loss_v:.3f} after {step + 1} steps"


def _torchrun(tiny_data, out, nproc, *extra):
    from ffnmem.distlaunch import run as dist_run

    proc = dist_run(nproc, ["-m", "ffnmem.train", "--config", TINY, "--device", "cpu", "--data_dir", str(tiny_data),
                            "--out_dir", str(out), *extra], timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    recs = [json.loads(l) for l in (Path(out) / "logs" / "train.jsonl").read_text().splitlines()]
    return {r["step"]: r["loss"] for r in recs if "loss" in r and r.get("event") is None}


@pytest.mark.slow
def test_ddp_two_process_matches_single(tiny_data, tmp_path):
    single = losses(run(tiny_data, tmp_path / "one", "--max_steps", "6"))
    ddp = _torchrun(tiny_data, tmp_path / "two", 2, "--max_steps", "6")
    for s in range(1, 7):
        assert ddp[s] == pytest.approx(single[s], rel=1e-5), s


@pytest.mark.slow
def test_ddp_zero1_resume(tiny_data, tmp_path):
    full = _torchrun(tiny_data, tmp_path / "zfull", 2, "--zero1", "--max_steps", "12")
    _torchrun(tiny_data, tmp_path / "zpart", 2, "--zero1", "--max_steps", "10")
    resumed = _torchrun(tiny_data, tmp_path / "zpart", 2, "--zero1", "--resume", "auto", "--max_steps", "12")
    for s in (11, 12):
        assert resumed[s] == pytest.approx(full[s], rel=1e-5), s


@pytest.mark.slow
def test_ddp_zero1_matches_single(tiny_data, tmp_path):
    single = losses(run(tiny_data, tmp_path / "one_z", "--max_steps", "6"))
    zero = _torchrun(tiny_data, tmp_path / "two_z", 2, "--zero1", "--max_steps", "6")
    for s in range(1, 7):
        assert zero[s] == pytest.approx(single[s], rel=1e-5), s


def test_zero1_param_groups_propagate(tmp_path):
    import torch.distributed as dist

    from ffnmem.train import Dist, build_optimizer

    os.environ["MASTER_PORT"] = str(__import__("ffnmem.distlaunch", fromlist=["free_port"]).free_port())
    d = Dist("cpu", force_pg=True)
    try:
        cfg = load_config(TINY)
        model = Transformer(cfg["model"])
        opt = build_optimizer(model, cfg["train"], torch.device("cpu"), zero1=True)
        x = torch.randint(0, 256, (2, 33))
        model(x[:, :-1], x[:, 1:])[1].backward()
        opt.step()
        local = {g["group"]: g for g in opt.optim.param_groups}
        base = float(cfg["train"]["lr"])
        assert local["tables"]["lr_mult"] == 4.0 and local["tables"]["lr"] == pytest.approx(4 * base)
        assert local["tables"]["weight_decay"] == 0.0 and local["router"]["weight_decay"] == 0.0
        assert local["decay"]["weight_decay"] == 0.1
        tables = {id(p) for p in model.blocks[1].memory.module.tables.parameters()}
        assert {id(p) for p in local["tables"]["params"]} == tables
    finally:
        d.close()
        for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
            os.environ.pop(k, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("zero1", [False, True])
def test_resume_gpu_fp16_fused(tiny_data, tmp_path, zero1):
    """Fused CUDA AdamW + GradScaler resume (ZeRO-1 load leaves optimizer state on CPU)."""
    if zero1 and os.name == "nt":
        pytest.skip("gloo with CUDA tensors crashes on Windows; ZeRO-1 GPU resume is verified on the cluster (NCCL)")
    extra = ["--set", "train.precision=fp16"] + (["--zero1"] if zero1 else [])
    os.environ["MASTER_PORT"] = str(__import__("ffnmem.distlaunch", fromlist=["free_port"]).free_port())

    def gpu_run(out, *more):
        try:
            return train_main(["--config", TINY, "--device", "cuda", "--data_dir", str(tiny_data),
                               "--out_dir", str(out), *extra, *more])
        finally:
            for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
                os.environ.pop(k, None)
            os.environ["MASTER_PORT"] = str(__import__("ffnmem.distlaunch", fromlist=["free_port"]).free_port())

    full = losses(gpu_run(tmp_path / "gfull", "--max_steps", "14"))
    gpu_run(tmp_path / "gpart", "--max_steps", "10")
    resumed = losses(gpu_run(tmp_path / "gpart", "--resume", "auto", "--max_steps", "14"))
    for s in range(11, 15):
        assert resumed[s] == pytest.approx(full[s], rel=1e-3), s
