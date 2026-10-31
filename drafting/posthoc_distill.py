"""M3.T3: distil a post-hoc drafter (``drafting/posthoc_model.py``) from a frozen target.

    python -m drafting.posthoc_distill --target runs/w3_sb_1b --data_dir $DATA --out_dir runs/ph_sb \
        --ssm_dim 0 --tokens 200e6 --wandb
    python -m drafting.posthoc_distill --target runs/w3_sb_1b --data_dir $DATA --out_dir runs/ph_sb_ssm \
        --ssm_dim 256 --tokens 200e6 --wandb --resume auto

Per optimizer step (64 x 2048 tokens by default, micro-batches + gradient accumulation):
the frozen target runs online (no grad, fp16 autocast on CUDA) and gives next-token
logits p and its real router logits at the memory layer; the drafter is trained on

    KL(sg(p) || q)  on a random ``--kl_frac`` of positions (chunked, fp32 softmax)
    + 0.1 * BCE(drafter route logits, target router bits)  at every position.

Positions are seeded by (data_seed, step, window id), so they do not depend on the
micro-batch.  The router learns from the BCE only; ``--surrogate`` also passes the KL to
the router through v2's surrogate (the frozen memory module is put in train mode for it).
AdamW (0.9, 0.95), lr 1e-3, warmup then cosine to ``min_lr_ratio``, weight decay 0.1 on
matrices, clip 1.0, GradScaler under fp16.  Periodic val: draft top-1 agreement with the
target, KL, router bit / code agreement.  Checkpoints in ``<out_dir>/checkpoints``
(``--resume auto``), the drafter itself in ``<out_dir>/drafter.pt`` (``PosthocDrafter.load``),
logs in ``<out_dir>/logs/train.jsonl`` (+ W&B project morbius-wave3 with ``--wandb``).
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from drafting.posthoc_model import PosthocDrafter
from ffnmem import checkpoint as ckpt
from ffnmem.data.loader import TrainWindows, ValWindows
from ffnmem.losses import draft_kl_chunk, position_seed, select_positions
from ffnmem.memory.draft import lookup_rows_at
from ffnmem.train import Dist, Logger, lr_at

EVAL_SEED_STEP = 2**32 - 2


def amp_ctx(device: torch.device, amp: bool):
    return torch.autocast(device.type, dtype=torch.float16) if amp else contextlib.nullcontext()


@torch.no_grad()
def teacher(target, layer: int, x: torch.Tensor, device: torch.device, amp: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Target logits [B, T, V] and router logits [B, T, R*bits] from one forward."""
    mem = target.blocks[layer].memory
    mem.capture_route_logits = True
    try:
        with amp_ctx(device, amp):
            logits = target(x)
        return logits, mem.last_route_logits
    finally:
        mem.capture_route_logits = False


def student_loss(drafter: PosthocDrafter, x: torch.Tensor, b_idx: torch.Tensor, t_idx: torch.Tensor,
                 p_sel: torch.Tensor, route_t: torch.Tensor, chunk: int, surrogate: bool,
                 use_ckpt: bool = True) -> tuple[torch.Tensor, dict]:
    """KL (mean over selected positions) and router BCE; ``p_sel`` [N, V] teacher logits at (b_idx, t_idx)."""
    m = drafter.memory_module
    emb = drafter.embed(x)
    s = drafter.states(emb)
    q = drafter.route_logits(emb, s)
    codes = m.quantize_route_logits(q)
    bce = F.binary_cross_entropy_with_logits(q, (route_t > 0).float())
    unembed = drafter.target.tok_emb.weight.detach()

    def one(emb, s, q, b, t, p):
        rows = lookup_rows_at(m, codes, q if surrogate else None, b, t)
        logits = drafter.draft(m, emb[b, t], rows, unembed, s[b, t] if s is not None else None)
        return draft_kl_chunk(logits, p)

    n = b_idx.numel()
    kl = agree = accept = 0.0
    for c0 in range(0, n, chunk):
        args = (emb, s, q, b_idx[c0:c0 + chunk], t_idx[c0:c0 + chunk], p_sel[c0:c0 + chunk])
        parts = checkpoint(one, *args, use_reentrant=False) if use_ckpt and torch.is_grad_enabled() else one(*args)
        kl, agree, accept = kl + parts[0], agree + parts[1], accept + parts[2]
    kl = kl / max(n, 1)
    with torch.no_grad():
        bit_acc = ((q > 0) == (route_t > 0)).float().mean()
        code_acc = (codes == m.quantize_route_logits(route_t)).float().mean()
    return kl, {"kl": kl.detach(), "bce": bce.detach(), "top1_agree": agree / max(n, 1),
                "accept_proxy": accept / max(n, 1), "bit_acc": bit_acc, "code_acc": code_acc,
                "bce_t": bce}


@torch.no_grad()
def evaluate(drafter: PosthocDrafter, val: ValWindows, batch: int, max_batches: int, frac: float, chunk: int,
             data_seed: int, device: torch.device, amp: bool) -> dict:
    sums: dict = {}
    nb = 0
    for i, (x, _) in enumerate(val.batches(batch, max_batches)):
        x = x.to(device)
        logits, route_t = teacher(drafter.target, drafter.layer, x, device, amp)
        seeds = [position_seed(data_seed, EVAL_SEED_STEP, i * batch + j) for j in range(x.shape[0])]
        bi, ti = select_positions(seeds, x.shape[1], frac, device)
        with amp_ctx(device, amp):
            _, parts = student_loss(drafter, x, bi, ti, logits[bi, ti], route_t, chunk, False, False)
        for k, v in parts.items():
            if k != "bce_t":
                sums[k] = sums.get(k, 0.0) + float(v)
        nb += 1
    return {f"val_{k}": v / max(nb, 1) for k, v in sums.items()}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, help="checkpoint file or run dir (frozen)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--layer", type=int, help="memory block position (default: first memory layer)")
    p.add_argument("--ssm_dim", type=int, default=0, help="0 (no SSM) | 256")
    p.add_argument("--q_heads", type=int, default=4)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--tokens", type=float, default=200e6)
    p.add_argument("--global_batch", type=int, default=64)
    p.add_argument("--micro_batch", type=int, default=4)
    p.add_argument("--seq_len", type=int, help="default: the target's seq_len")
    p.add_argument("--data_seed", type=int, default=1234, help="window order (index file)")
    p.add_argument("--data_start_step", type=int, default=0, help="start at this step of the window order")
    p.add_argument("--seed", type=int, default=0, help="drafter init")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_frac", type=float, default=0.02)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--kl_frac", type=float, default=0.25, help="fraction of positions for the KL")
    p.add_argument("--kl_chunk", type=int, default=512)
    p.add_argument("--bce_weight", type=float, default=0.1)
    p.add_argument("--surrogate", action="store_true", help="KL gradient to the router via v2's surrogate")
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--val_tokens", type=float, default=2e6)
    p.add_argument("--eval_batches", type=int, default=8)
    p.add_argument("--ckpt_every", type=int, default=200)
    p.add_argument("--keep_last", type=int, default=2)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--resume", help="'auto' or a checkpoint path")
    p.add_argument("--max_steps", type=int, help="stop after this many steps (schedule unchanged)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--precision", default="auto", choices=["auto", "fp16", "fp32"])
    return p.parse_args(argv)


def main(argv=None) -> dict:
    from ffnmem.infer import load_model

    a = parse_args(argv)
    d = Dist(a.device)
    dev = d.device
    amp = a.precision == "fp16" or (a.precision == "auto" and dev.type == "cuda")
    out_dir = Path(a.out_dir)
    target, tcfg = load_model(a.target, device=dev)
    target.requires_grad_(False)
    seq_len = a.seq_len or int(target.cfg["seq_len"])
    torch.manual_seed(a.seed)
    drafter = PosthocDrafter(target, a.layer, a.q_heads, a.head_dim, a.ssm_dim).to(dev)
    if a.surrogate:
        drafter.memory_module.train()  # surrogate is active only in train mode (dropout is 0)
    tw = TrainWindows(a.data_dir, seq_len, a.data_seed, a.global_batch, a.micro_batch,
                      model_vocab=int(target.cfg["vocab"]))
    total = math.ceil(a.tokens / (a.global_batch * seq_len))
    sched = {"total_steps": total, "warmup_steps": max(1, round(a.warmup_frac * total)), "peak_lr": a.lr,
             "min_lr_ratio": a.min_lr_ratio}
    decay = [p for p in drafter.parameters() if p.ndim >= 2]
    no_decay = [p for p in drafter.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95),
                            **({"fused": True} if dev.type == "cuda" else {}))
    scaler = torch.amp.GradScaler(dev.type, enabled=amp and dev.type == "cuda")
    start = 0
    resume = ckpt.latest_checkpoint(out_dir) if a.resume == "auto" else (Path(a.resume) if a.resume else None)
    if resume is not None:
        st = ckpt.load(resume)
        drafter.load_state_dict(st["drafter"])
        opt.load_state_dict(st["optimizer"])
        scaler.load_state_dict(st["scaler"])
        ckpt.set_rng_state(st["rng"])
        start = int(st["step"])
        print(f"[distill] resumed from {resume} at step {start}", flush=True)
    val = ValWindows(a.data_dir, seq_len, int(a.val_tokens)) if a.val_tokens > 0 else None
    run_name = out_dir.name
    log = Logger(out_dir, True, a.wandb, run_name,
                 {"wandb": {"project": "morbius-wave3"}, "args": vars(a), "target": tcfg.get("_checkpoint"),
                  "drafter": drafter.config})
    accum = a.global_batch // a.micro_batch

    def save(step: int) -> None:
        ckpt.save_atomic({"drafter": drafter.state_dict(), "config": drafter.config, "optimizer": opt.state_dict(),
                          "scaler": scaler.state_dict(), "step": step, "rng": ckpt.rng_state(), "args": vars(a),
                          "target": tcfg.get("_checkpoint")}, ckpt.ckpt_path(out_dir, step))
        drafter.save(out_dir / "drafter.pt")
        ckpt.prune(out_dir, a.keep_last, 0)

    end = total if a.max_steps is None else min(total, a.max_steps)
    last: dict = {}
    t0, tok0 = time.time(), 0
    for step in range(start, end):
        lr = lr_at(step, sched)
        for g in opt.param_groups:
            g["lr"] = lr
        data_step = a.data_start_step + step
        ids = tw.rank_window_ids(data_step)
        sums: dict = {}
        for i, (x, _) in enumerate(tw.micro_batches(data_step)):
            x = x.to(dev, non_blocking=True)
            logits, route_t = teacher(target, drafter.layer, x, dev, amp)
            seeds = [position_seed(a.data_seed, data_step, int(w))
                     for w in ids[i * a.micro_batch:(i + 1) * a.micro_batch]]
            bi, ti = select_positions(seeds, x.shape[1], a.kl_frac, dev)
            p_sel = logits[bi, ti]
            del logits
            with amp_ctx(dev, amp):
                kl, parts = student_loss(drafter, x, bi, ti, p_sel, route_t, a.kl_chunk, a.surrogate)
            loss = kl + a.bce_weight * parts.pop("bce_t")
            scaler.scale(loss / accum).backward()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v) / accum
            tok0 += x.numel()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(drafter.parameters(), a.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        done = step + 1
        last = {"step": done, "lr": lr, "grad_norm": float(gnorm), "loss": sums["kl"] + a.bce_weight * sums["bce"],
                **sums, "scale": float(scaler.get_scale()) if scaler.is_enabled() else 1.0}
        if done % a.log_every == 0 or done == end:
            last["tok_per_s"] = tok0 / max(time.time() - t0, 1e-9)
            t0, tok0 = time.time(), 0
            log.log(last)
            print(f"[distill] step {done}/{total} kl {sums['kl']:.4f} bce {sums['bce']:.4f} "
                  f"agree {sums['top1_agree']:.3f} bits {sums['bit_acc']:.3f}", flush=True)
        if val is not None and (done % a.eval_every == 0 or done == end):
            ev = evaluate(drafter, val, a.micro_batch, a.eval_batches, a.kl_frac, a.kl_chunk, a.data_seed, dev, amp)
            log.log({"step": done, **ev})
            last.update(ev)
            print(f"[distill] val top1 agree {ev['val_top1_agree']:.4f} kl {ev['val_kl']:.4f}", flush=True)
        if done % a.ckpt_every == 0 or done == end:
            save(done)
    log.close()
    return last


if __name__ == "__main__":
    main()
