"""FFN importance: val-loss increase when one layer's FFN is zeroed (M6.T2).

    python -m eval.ffn_importance --checkpoint runs/w1_mem_full --run Run1 --device cuda
    python -m eval.ffn_importance --checkpoint runs/w1_dense_cut_near/checkpoints/step_0003815.pt --run Run4 \\
        --data_dir /data/climbmix --val_tokens 5000000 --batch_size 4
    # -> results/ffn_importance_<run>.json + .png

Each block's FFN output is replaced by zeros through a forward hook (removed
afterwards, weights untouched), one block at a time.  Loss is the mean token
cross-entropy per val window (the first ``val_tokens`` tokens of ``val.bin``,
non-overlapping ``seq_len`` windows, as in training eval); Δloss per layer is the
mean paired per-window difference with a percentile bootstrap CI over windows.
fp16 autocast on CUDA.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ffnmem.data.loader import ValWindows  # noqa: E402
from ffnmem.infer import load_model  # noqa: E402

from eval.bootstrap_lmeval import bootstrap_ci  # noqa: E402


def amp_context(device: torch.device, precision: str):
    if precision == "fp16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.no_grad()
def window_losses(model: torch.nn.Module, val: ValWindows, batch_size: int, precision: str = "fp16",
                  max_batches: int | None = None) -> np.ndarray:
    """Mean next-token cross-entropy of each val window, in ``val.ids`` order."""
    dev = next(model.parameters()).device
    out = []
    for x, y in val.batches(batch_size, max_batches):
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with amp_context(dev, precision):
            logits = model(x)
        for r in range(x.shape[0]):  # one row at a time keeps the fp32 copy small
            out.append(float(F.cross_entropy(logits[r].float(), y[r])))
    return np.asarray(out, dtype=np.float64)


@contextlib.contextmanager
def zero_ffn(model: torch.nn.Module, pos: int):
    """Block ``pos``'s FFN returns zeros inside the context."""
    handle = model.blocks[pos].ffn.register_forward_hook(lambda m, inp, out: torch.zeros_like(out))
    try:
        yield
    finally:
        handle.remove()


def ffn_importance(model: torch.nn.Module, val: ValWindows, batch_size: int, precision: str = "fp16",
                   layers: list[int] | None = None, n_boot: int = 10_000, seed: int = 0,
                   max_batches: int | None = None, log=print) -> dict:
    base = window_losses(model, val, batch_size, precision, max_batches)
    log(f"[ffn] base loss {base.mean():.4f} over {len(base)} windows")
    rows = []
    for pos in (range(len(model.blocks)) if layers is None else layers):
        block = model.blocks[pos]
        d_ff = int(block.ffn.w1.out_features)
        if d_ff == 0:
            rows.append({"pos": pos, "layer_id": int(block.layer_id), "d_ff": 0, "skipped": True})
            continue
        t0 = time.time()
        with zero_ffn(model, pos):
            loss = window_losses(model, val, batch_size, precision, max_batches)
        delta, lo, hi = bootstrap_ci(loss - base, n_boot, seed)
        rows.append({"pos": pos, "layer_id": int(block.layer_id), "d_ff": d_ff,
                     "has_memory": block.memory is not None, "loss": float(loss.mean()),
                     "delta": delta, "ci_lo": lo, "ci_hi": hi, "seconds": time.time() - t0})
        log(f"[ffn] pos {pos} (layer {block.layer_id}, d_ff {d_ff}): loss {loss.mean():.4f} "
            f"delta {delta:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    return {"base_loss": float(base.mean()), "n_windows": int(len(base)), "layers": rows}


def plot(result: dict, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in result["layers"] if not r.get("skipped")]
    x = [r["pos"] for r in rows]
    y = np.array([r["delta"] for r in rows])
    err = np.array([[r["delta"] - r["ci_lo"] for r in rows], [r["ci_hi"] - r["delta"] for r in rows]])
    colors = ["#c4562b" if r.get("has_memory") else "#3a6ea5" for r in rows]
    fig, ax = plt.subplots(figsize=(max(5.0, 0.35 * len(rows) + 2), 3.2))
    ax.bar(x, y, color=colors, yerr=err, capsize=2, linewidth=0)
    ax.set_xlabel("block (kept-layer position); orange = memory layer")
    ax.set_ylabel("Δ val loss (nats)")
    ax.set_title(f"FFN importance: {result.get('run', '')} (base {result['base_loss']:.3f})")
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="checkpoint file or run dir (latest)")
    ap.add_argument("--run", default=None, help="label for the output names (default: config name)")
    ap.add_argument("--data_dir", default=None, help="default: the checkpoint config's train.data_dir")
    ap.add_argument("--val_tokens", type=int, default=5_000_000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="block positions (default: all)")
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=str(ROOT / "results"))
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args(argv)

    model, cfg = load_model(args.checkpoint, device=args.device)
    run = args.run or cfg.get("name") or Path(args.checkpoint).name
    data_dir = args.data_dir or cfg["train"]["data_dir"]
    val = ValWindows(data_dir, int(model.cfg["seq_len"]), args.val_tokens)
    res = ffn_importance(model, val, args.batch_size, args.precision, args.layers, args.n_boot, args.seed)
    res = {"run": run, "checkpoint": cfg["_checkpoint"], "step": cfg["_step"], "config": cfg.get("name"),
           "data_dir": str(data_dir), "val_tokens": args.val_tokens, "seq_len": int(model.cfg["seq_len"]),
           "precision": args.precision, "n_boot": args.n_boot, **res}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ffn_importance_{run}.json"
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    if not args.no_plot:
        plot(res, path.with_suffix(".png"))
        print(f"wrote {path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
