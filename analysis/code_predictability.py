"""M2.T2 (RQ2): how predictable are the router codes from the tokens?

    python -m analysis.code_predictability --dump dumps/sb --run sb
    python -m analysis.code_predictability --dump dumps/sb --run sb --ckpt runs/w3_sb_1b --mlp_steps 4000

Input: a dump from ``analysis/dump_codes.py``.  Documents are split into fit / eval
halves (``--split halves``: first half of the documents in val order fits, the
second half evaluates; ``interleave``: even / odd document ids).  Per memory layer
and route, on the eval half (all entropies and cross-entropies in bits):

- H(C) of the route's code and exp-equivalent ``2^H`` (effective codes);
- held-out CE, top-1 and top-3 accuracy of predicting c[t] from
  ``uni``: x[t] (count table, add-alpha smoothing with the route's code unigram as
  the prior, i.e. backoff), ``bi``: (x[t-1], x[t]) (counts, backing off to the
  ``uni`` model), ``mlp``: a small MLP over the checkpoint's (frozen, RMS-normalised)
  embeddings of x[t-2..t] (one MLP per layer, all routes jointly);
- fraction explained ``1 - CE / H(C)`` (per route, mean over routes, and the
  ratio of sums ``1 - sum CE / sum H``);
- next step: c[t+1] from (c[t-1], c[t]) counts -- per-route top-k recall and
  P(any route misses) for k in {1, 2, 4, 8}.

Context before a window start is "start" (the model never sees across windows).
CIs: percentile bootstrap over eval documents (``--n_boot`` resamples, one
multinomial weight matrix applied to per-document sums, so every statistic --
ratios and entropies included -- is recomputed per resample).

Outputs: ``results/rq2_<run>.json``, ``results/figures/rq2_<run>_L<p>.png`` and the
MLPs ``results/rq2_<run>_mlp_L<p>.pt`` (used by ``analysis/substitution.py``).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from analysis.dump_codes import load_dump, read_dump_meta

TOPK = (1, 2, 4, 8)
PREDICTORS = ("uni", "bi", "mlp")


# ------------------------------------------------------------------ bootstrap
def bootstrap_weights(n_units: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """[n_boot + 1, n_units] resample counts; row 0 is the point estimate (all ones)."""
    rng = np.random.default_rng(seed)
    w = rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=n_boot).astype(np.float64)
    return np.vstack([np.ones((1, n_units)), w])


def unit_sums(values: np.ndarray, unit: np.ndarray, n_units: int) -> np.ndarray:
    """Per-unit sums of values [n] or [n, M] -> [n_units] or [n_units, M]."""
    if values.ndim == 1:
        return np.bincount(unit, weights=values, minlength=n_units)
    return np.stack([np.bincount(unit, weights=values[:, j], minlength=n_units) for j in range(values.shape[1])], 1)


def summarize(rows: np.ndarray, level: float = 0.95) -> dict:
    """rows [n_boot + 1]: point estimate (row 0) and percentile CI over the resamples."""
    rows = np.asarray(rows, dtype=np.float64)
    a = (1 - level) / 2
    lo, hi = np.nanpercentile(rows[1:], [100 * a, 100 * (1 - a)]) if len(rows) > 1 else (rows[0], rows[0])
    return {"value": float(rows[0]), "lo": float(lo), "hi": float(hi)}


def entropy_bits(counts: np.ndarray) -> np.ndarray:
    """Entropy (bits) of count rows [..., K]."""
    p = counts / np.clip(counts.sum(-1, keepdims=True), 1e-300, None)
    return -(p * np.log2(np.clip(p, 1e-300, None))).sum(-1)


# ------------------------------------------------------------------ MLP predictor
class CodeMLP(nn.Module):
    """Codes of every route of one memory layer from embeddings of x[t-2], x[t-1], x[t]."""

    def __init__(self, d_model: int, routes: int, bits: int = 4, hidden: int = 512, layer: int = -1):
        super().__init__()
        self.config = dict(d_model=d_model, routes=routes, bits=bits, hidden=hidden, layer=layer)
        self.routes, self.k = routes, 1 << bits
        self.net = nn.Sequential(nn.Linear(3 * d_model, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, routes * self.k))

    @staticmethod
    def padded_embeddings(tok_emb: Tensor) -> Tensor:
        """RMS-normalised embedding rows [V, d] + a zero row V for "no token" (before the window start)."""
        e = tok_emb.detach().float()
        e = e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + 1e-6)
        return torch.cat((e, e.new_zeros(1, e.shape[1])), 0)

    def forward(self, emb_pad: Tensor, x2: Tensor, x1: Tensor, x: Tensor) -> Tensor:
        """Logits [..., R, K]; x2/x1/x index ``emb_pad`` (V = start)."""
        h = torch.cat((emb_pad[x2], emb_pad[x1], emb_pad[x]), -1)
        out = self.net(h)
        return out.view(*out.shape[:-1], self.routes, self.k)

    def window_logits(self, emb_pad: Tensor, idx: Tensor) -> Tensor:
        """Logits [B, T, R, K] for whole windows idx [B, T] (no context before t = 0)."""
        start = emb_pad.shape[0] - 1
        pad = torch.full_like(idx[:, :1], start)
        x1 = torch.cat((pad, idx[:, :-1]), 1)
        x2 = torch.cat((pad, pad, idx[:, :-2]), 1)[:, : idx.shape[1]]
        return self(emb_pad, x2, x1, idx)

    def save(self, path: str | Path, **extra) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.state_dict(), "config": self.config, **extra}, path)

    @classmethod
    def load(cls, path: str | Path, map_location="cpu") -> "CodeMLP":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(**blob["config"])
        m.load_state_dict(blob["state"])
        return m.eval()


def load_embeddings(ckpt_path: str | Path) -> Tensor:
    from ffnmem import checkpoint as ckpt

    p = Path(ckpt_path)
    if p.is_dir():
        found = ckpt.latest_checkpoint(p)
        if found is None:
            raise FileNotFoundError(f"no checkpoint in {p}")
        p = found
    return ckpt.load(p)["model"]["tok_emb.weight"].float()


def train_code_mlp(emb_pad: Tensor, x: np.ndarray, x1: np.ndarray, x2: np.ndarray, codes: np.ndarray,
                   fit_idx: np.ndarray, *, bits: int, layer: int, hidden: int = 512, steps: int = 4000,
                   batch: int = 4096, lr: float = 2e-3, seed: int = 0, device: torch.device | None = None,
                   log_every: int = 500) -> CodeMLP:
    device = device or emb_pad.device
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mlp = CodeMLP(emb_pad.shape[1], codes.shape[1], bits, hidden, layer).to(device)
    emb_pad = emb_pad.to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    mlp.train()
    for step in range(steps):
        sel = fit_idx[rng.integers(0, len(fit_idx), size=batch)]
        t = lambda a: torch.from_numpy(np.asarray(a[sel], dtype=np.int64)).to(device)  # noqa: E731
        logits = mlp(emb_pad, t(x2), t(x1), t(x))
        loss = F.cross_entropy(logits.reshape(-1, mlp.k).float(), t(codes).reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"[mlp L{layer}] step {step} CE {float(loss.detach()) / math.log(2):.3f} bits/route", flush=True)
    return mlp.eval()


@torch.no_grad()
def mlp_unit_sums(mlp: CodeMLP, emb_pad: Tensor, x: np.ndarray, x1: np.ndarray, x2: np.ndarray, codes: np.ndarray,
                  ev_idx: np.ndarray, unit: np.ndarray, n_units: int, chunk: int = 65536) -> np.ndarray:
    """Per-unit sums [n_units, R, 3] of (CE bits, top-1 hit, top-3 hit) on the eval positions."""
    dev = next(mlp.parameters()).device
    emb_pad = emb_pad.to(dev)
    out = torch.zeros(n_units, mlp.routes, 3, dtype=torch.float64, device=dev)
    for s in range(0, len(ev_idx), chunk):
        sel = ev_idx[s: s + chunk]
        t = lambda a: torch.from_numpy(np.asarray(a[sel], dtype=np.int64)).to(dev)  # noqa: E731
        lp = mlp(emb_pad, t(x2), t(x1), t(x)).float().log_softmax(-1)  # [n, R, K]
        c = t(codes)  # [n, R]
        ce = -lp.gather(-1, c[..., None])[..., 0] / math.log(2)
        top = lp.topk(3, dim=-1).indices
        t1 = (top[..., 0] == c).double()
        t3 = (top == c[..., None]).any(-1).double()
        u = torch.from_numpy(unit[s: s + chunk]).to(dev)
        out.index_add_(0, u, torch.stack((ce.double(), t1, t3), -1))
    return out.cpu().numpy()


# ------------------------------------------------------------------ count predictors
def smoothed_table(counts: np.ndarray, prior: np.ndarray, beta: float) -> np.ndarray:
    """(n(ctx, c) + beta * prior(c)) / (n(ctx) + beta); prior broadcastable to counts."""
    return (counts + beta * prior) / (counts.sum(-1, keepdims=True) + beta)


def topk_hits(p: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(top-1 hit, top-3 hit) of true codes c [n] under rows p [n, K] (stable tie order)."""
    order = np.argsort(-p, axis=1, kind="stable")[:, :3]
    return order[:, 0] == c, (order == c[:, None]).any(1)


def split_fit(doc: np.ndarray, mode: str) -> np.ndarray:
    if mode == "halves":
        u = np.unique(doc)
        return doc < u[len(u) // 2]
    if mode == "interleave":
        return doc % 2 == 0
    raise ValueError(f"split must be halves | interleave, got {mode!r}")


# ------------------------------------------------------------------ main analysis
def analyze(dump_dir: str | Path, *, layers: list[int] | None = None, alpha: float = 0.5, n_boot: int = 1000,
            seed: int = 0, split: str = "halves", emb: Tensor | None = None, mlp_kwargs: dict | None = None,
            mlp_dir: str | Path | None = None, run: str = "run", max_positions: int | None = None,
            route_chunk: int = 16, eval_chunk: int = 1 << 21, device: torch.device | None = None) -> dict:
    t0 = time.time()
    meta = read_dump_meta(dump_dir)
    layers = [int(p) for p in (layers or meta["memory_layers"])]
    _, d = load_dump(dump_dir, ["tokens", "pos", "doc"] + [f"codes_L{p}" for p in layers], max_positions)
    V = int(meta["vocab"])
    bits = int(meta["bits"])
    K = 1 << bits
    x = d["tokens"].astype(np.int64)
    pos = d["pos"].astype(np.int64)
    doc = d["doc"].astype(np.int64)
    n = len(x)
    x1 = np.full(n, V, np.int64)
    x1[1:] = x[:-1]
    x1[pos < 1] = V
    x2 = np.full(n, V, np.int64)
    x2[2:] = x[:-2]
    x2[pos < 2] = V

    fit = split_fit(doc, split)
    fit_idx, ev_idx = np.flatnonzero(fit), np.flatnonzero(~fit)
    ev_docs, ev_unit = np.unique(doc[ev_idx], return_inverse=True)
    D = len(ev_docs)
    W = bootstrap_weights(D, n_boot, seed)
    n_b = W @ unit_sums(np.ones(len(ev_idx)), ev_unit, D)

    # next-step pairs: target j (pos >= 1), context (c[j-2] or start, c[j-1])
    jn = np.flatnonzero(pos >= 1)
    jn_f, jn_e = jn[fit[jn]], jn[~fit[jn]]
    nx_unit = np.searchsorted(ev_docs, doc[jn_e])
    nn_b = W @ unit_sums(np.ones(len(jn_e)), nx_unit, D)

    # bigram keys
    key = x1 * (V + 1) + x
    uk, inv_f = np.unique(key[fit_idx], return_inverse=True)
    U = len(uk)
    se = np.searchsorted(uk, key[ev_idx])
    se_c = np.minimum(se, U - 1)
    seen = uk[se_c] == key[ev_idx]
    xf, xe = x[fit_idx], x[ev_idx]
    beta = alpha * K

    out_layers = {}
    for p in layers:
        codes = d[f"codes_L{p}"]
        R = codes.shape[1]
        mlp_sums = None
        if emb is not None:
            mk = dict(mlp_kwargs or {})
            mlp = train_code_mlp(CodeMLP.padded_embeddings(emb), x, x1, x2, codes, fit_idx, bits=bits, layer=p,
                                 device=device, seed=seed, **mk)
            if mlp_dir is not None:
                mlp.save(Path(mlp_dir) / f"rq2_{run}_mlp_L{p}.pt", dump=str(dump_dir))
            mlp_sums = mlp_unit_sums(mlp, CodeMLP.padded_embeddings(emb), x, x1, x2, codes, ev_idx, ev_unit, D)
        preds = [q for q in PREDICTORS if q != "mlp" or mlp_sums is not None]
        stat_names = ["H", "eff_codes"] + [f"{s}_{q}" for q in preds for s in ("ce", "frac", "top1", "top3")] + \
                     [f"next_recall@{k}" for k in TOPK]
        per_route = {s: [] for s in stat_names}
        per_route_ci = {s: [] for s in stat_names}
        sums = {s: np.zeros(n_boot + 1) for s in stat_names}
        tot_ce = {q: np.zeros(n_boot + 1) for q in preds}
        tot_h = np.zeros(n_boot + 1)
        all_hit = {k: np.ones(len(jn_e), bool) for k in TOPK}

        for r0 in range(0, R, route_chunk):
            cols = np.ascontiguousarray(codes[:, r0: r0 + route_chunk].T).astype(np.int64)  # [rc, N]
            for i, col in enumerate(cols):
                r = r0 + i
                cf, ce = col[fit_idx], col[ev_idx]
                S = [unit_sums(np.ones(len(ce)), ev_unit * K + ce, D * K).reshape(D, K)]  # code counts per doc
                # uni: x[t]
                nxc = np.bincount(xf * K + cf, minlength=(V + 1) * K).reshape(V + 1, K).astype(np.float64)
                nc = nxc.sum(0)
                pc = (nc + alpha) / (nc.sum() + K * alpha)
                px = smoothed_table(nxc, pc[None], beta)
                lpx = np.log2(px)
                ox = np.argsort(-px, axis=1, kind="stable")[:, :3]
                t1 = ox[xe, 0] == ce
                t3 = (ox[xe] == ce[:, None]).any(1)
                S.append(unit_sums(np.stack((-lpx[xe, ce], t1, t3), 1).astype(np.float64), ev_unit, D))
                # bi: (x[t-1], x[t]) -> backoff to uni
                nkc = np.bincount(inv_f * K + cf, minlength=U * K).reshape(U, K)
                parts = []
                for s in range(0, len(ev_idx), eval_chunk):
                    sl = slice(s, s + eval_chunk)
                    pe = px[xe[sl]]
                    m = seen[sl]
                    pe[m] = smoothed_table(nkc[se_c[sl][m]].astype(np.float64), pe[m], beta)
                    a1, a3 = topk_hits(pe, ce[sl])
                    parts.append(np.stack((-np.log2(pe[np.arange(len(pe)), ce[sl]]), a1, a3), 1))
                S.append(unit_sums(np.concatenate(parts).astype(np.float64), ev_unit, D))
                if mlp_sums is not None:
                    S.append(mlp_sums[:, r, :])
                # next step: c[t+1] from (c[t-1], c[t])
                prev = np.where(pos[jn] >= 2, col[np.maximum(jn - 2, 0)], K)
                ctx = prev * K + col[jn - 1]
                is_f = fit[jn]
                nctx = np.bincount(ctx[is_f] * K + col[jn_f], minlength=(K + 1) * K * K).reshape((K + 1) * K, K)
                order = np.argsort(-(nctx + beta * pc[None]), axis=1, kind="stable")
                rank = np.empty_like(order)
                np.put_along_axis(rank, order, np.arange(K)[None].repeat(len(order), 0), axis=1)
                rk = rank[ctx[~is_f], col[jn_e]]
                hits = np.stack([rk < k for k in TOPK], 1)
                for j, k in enumerate(TOPK):
                    all_hit[k] &= hits[:, j]
                S_next = unit_sums(hits.astype(np.float64), nx_unit, D)

                A = W @ np.concatenate(S, 1)
                An = W @ S_next
                H = entropy_bits(A[:, :K])
                st = {"H": H, "eff_codes": np.exp2(H)}
                tot_h += H
                for qi, q in enumerate(preds):
                    b = K + 3 * qi
                    ce_b = A[:, b] / n_b
                    st[f"ce_{q}"] = ce_b
                    st[f"frac_{q}"] = 1.0 - ce_b / np.clip(H, 1e-12, None)
                    st[f"top1_{q}"] = A[:, b + 1] / n_b
                    st[f"top3_{q}"] = A[:, b + 2] / n_b
                    tot_ce[q] += ce_b
                for j, k in enumerate(TOPK):
                    st[f"next_recall@{k}"] = An[:, j] / nn_b
                for s in stat_names:
                    sm = summarize(st[s])
                    per_route[s].append(sm["value"])
                    per_route_ci[s].append([sm["lo"], sm["hi"]])
                    sums[s] += st[s]
            print(f"[rq2 L{p}] routes {min(r0 + route_chunk, R)}/{R} ({time.time() - t0:.0f}s)", flush=True)

        summary = {f"mean_{s}": summarize(sums[s] / R) for s in stat_names}
        for q in preds:
            summary[f"frac_total_{q}"] = summarize(1.0 - tot_ce[q] / np.clip(tot_h, 1e-12, None))
        miss = np.stack([~all_hit[k] for k in TOPK], 1).astype(np.float64)
        Am = W @ unit_sums(miss, nx_unit, D)
        for j, k in enumerate(TOPK):
            summary[f"p_any_miss@{k}"] = summarize(Am[:, j] / nn_b)
        out_layers[str(p)] = {"routes": R, "summary": summary, "per_route": per_route, "per_route_ci": per_route_ci}

    return {
        "run": run, "dump": str(dump_dir), "checkpoint": meta.get("checkpoint"), "step": meta.get("step"),
        "config": {"alpha": alpha, "n_boot": n_boot, "seed": seed, "split": split, "bits": bits,
                   "mlp": None if emb is None else dict(mlp_kwargs or {}), "units": "bits",
                   "ci": "95% percentile bootstrap over eval documents"},
        "n_positions": int(n), "n_fit": int(len(fit_idx)), "n_eval": int(len(ev_idx)),
        "n_next_eval": int(len(jn_e)), "docs_eval": int(D), "docs_fit": int(len(np.unique(doc[fit_idx]))),
        "layers": out_layers, "seconds": round(time.time() - t0, 1),
    }


# ------------------------------------------------------------------ plotting
def plot_layer(res: dict, layer: str, path: str | Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[rq2] matplotlib not installed; no figure", flush=True)
        return False
    L = res["layers"][layer]
    pr, sm = L["per_route"], L["summary"]
    preds = [q for q in PREDICTORS if f"frac_{q}" in pr]
    order = np.argsort(pr[f"frac_{preds[-1]}"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for q in preds:
        ax[0].plot(np.asarray(pr[f"frac_{q}"])[order], label=f"{q} (mean {sm[f'mean_frac_{q}']['value']:.2f})")
    ax[0].set(xlabel="route (sorted)", ylabel="fraction explained 1 - CE/H", title=f"{res['run']} layer {layer}")
    ax[0].axhline(0, color="grey", lw=0.5)
    ax[0].legend()
    ks = np.array(TOPK)
    rec = [sm[f"mean_next_recall@{k}"] for k in TOPK]
    miss = [sm[f"p_any_miss@{k}"] for k in TOPK]
    for vals, lab in ((rec, "per-route recall"), (miss, "P(any route misses)")):
        v = np.array([x["value"] for x in vals])
        err = np.array([[x["value"] - x["lo"], x["hi"] - x["value"]] for x in vals]).T
        ax[1].errorbar(ks, v, yerr=err, marker="o", capsize=3, label=lab)
    ax[1].set(xscale="log", xticks=ks, xticklabels=[str(k) for k in ks], ylim=(-0.02, 1.02),
              xlabel="k", title="next-step c[t+1] from (c[t-1], c[t])")
    ax[1].minorticks_off()
    ax[1].legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="dumps/<run> directory")
    ap.add_argument("--run", default=None, help="name for outputs (default: the dump's run)")
    ap.add_argument("--ckpt", default=None, help="checkpoint for the MLP's embeddings (default: the dump's)")
    ap.add_argument("--no_mlp", action="store_true")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--split", choices=["halves", "interleave"], default="halves")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_positions", type=int, default=None)
    ap.add_argument("--mlp_hidden", type=int, default=512)
    ap.add_argument("--mlp_steps", type=int, default=4000)
    ap.add_argument("--mlp_batch", type=int, default=4096)
    ap.add_argument("--mlp_lr", type=float, default=2e-3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--no_plot", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    meta = read_dump_meta(args.dump)
    run = args.run or meta["run"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    emb = None
    if not args.no_mlp:
        src = args.ckpt or meta.get("checkpoint")
        if not src:
            raise ValueError("MLP predictor needs --ckpt (or pass --no_mlp)")
        emb = load_embeddings(src)
    out_dir = Path(args.out_dir)
    res = analyze(args.dump, layers=args.layers, alpha=args.alpha, n_boot=args.n_boot, seed=args.seed,
                  split=args.split, emb=emb, run=run, max_positions=args.max_positions, device=device,
                  mlp_kwargs=dict(hidden=args.mlp_hidden, steps=args.mlp_steps, batch=args.mlp_batch, lr=args.mlp_lr),
                  mlp_dir=out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"rq2_{run}.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    if not args.no_plot:
        for p in res["layers"]:
            plot_layer(res, p, out_dir / "figures" / f"rq2_{run}_L{p}.png")
    for p, L in res["layers"].items():
        s = L["summary"]
        line = " ".join(f"frac_{q}={s[f'mean_frac_{q}']['value']:.3f}" for q in PREDICTORS if f"mean_frac_{q}" in s)
        print(f"[rq2] L{p}: H={s['mean_H']['value']:.3f} bits {line} "
              f"P(any miss)@8={s['p_any_miss@8']['value']:.3f}", flush=True)
    print(f"[rq2] wrote {path}", flush=True)
    return res


if __name__ == "__main__":
    main()
