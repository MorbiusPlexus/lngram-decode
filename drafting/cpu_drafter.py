"""CPU drafter over an exported bundle (M5.T2): PyTorch reference + wrapper for the C++ extension.

    bundle = load_bundle("bundles/sdb.bin")
    d = RefDrafter(bundle)                 # or CppDrafter(bundle, threads=2) / make_drafter(bundle, "auto")
    d.commit(prompt_tokens)                # verified tokens (optionally with the target's codes)
    out = d.draft(4)                       # out.tokens, out.topk_ids [4, 3], out.topk_logits
    out = d.draft(5, hyp=out.tokens)       # draft after hypothetical tokens (nothing is committed)
    d.truncate(n)                          # roll back committed tokens

    python -m drafting.cpu_drafter --bench bundles/sdb.bin --impl cpp --threads 1 2 4
    python -m drafting.cpu_drafter --check bundles/sdb.bin --positions 10000     # C++ vs reference

Per token x[t] (position t; the draft predicts x[t+1]), all in fp32:

1. e = E[x[t]]; SSM step ``s[t] = a * s[t-1] + (1 - a) * W_in RMSNorm(e)`` (if the drafter has an SSM).
2. Router: ``concat(RMSNorm(E[x[t-1]] or start), RMSNorm(e)[, s[t]]) -> R*bits logits``;
   bit i of route r is logit ``r*bits + i > 0``; code = sum bit_i << i.  Codes of committed
   positions are the drafter's own unless ``commit(tokens, codes=...)`` supplies the target's.
3. Rows of the n-grams ending at t (v2 layout: ``r * 16^n + sum_k c[t-n+1+k] << 4k``; zero
   row when t < n-1), fp16 -> fp32, concatenated over orders -> [R, Dt].
4. Shortlist (deterministic): over orders (ascending), routes and sketch slots with count > 0,
   sum the counts per token; sort by (count desc, token id asc); keep ``cap`` (64); pad with the
   bundle's ``fallback`` tokens (global frequency order) not already present; if it is still
   empty (no valid rows and no fallback), or the bundle has no sketch / cap <= 0: full vocab.
5. Draft readout = ``DraftReadout.hidden``: memory RMSNorm -> KV projection; query
   ``W_q RMSNorm(e) (+ W_sq s[t])``; scores * head_dim^-0.5 over the R routes with v2's
   zero-value Sink (fixed logit) in the softmax denominator; ``W_o`` -> RMSNorm.
6. Logits = E[shortlist] . h; top-k by (logit desc, shortlist index asc).

Stateful: ``commit`` appends verified tokens (their codes and SSM states are kept per
position, so ``truncate`` is exact rollback); ``draft`` runs on a scratch cursor and never
mutates the committed state.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .export_drafter import Bundle, load_bundle


@dataclass
class DraftOut:
    tokens: list            # gamma drafted tokens (top-1 chain)
    topk_ids: np.ndarray    # int64 [gamma, k] (-1 = fewer candidates)
    topk_logits: np.ndarray  # float32 [gamma, k]
    codes: np.ndarray       # uint8 [len(hyp) + advanced drafts, R]: codes of hyp tokens, then drafted tokens

    @property
    def runner_up(self) -> int:
        """Runner-up at the first drafted position (repair branch), -1 if none."""
        return int(self.topk_ids[0, 1]) if self.topk_ids.shape[1] > 1 else -1


def _rms(x: Tensor, w: Tensor, eps: float) -> Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


@dataclass(frozen=True)
class _Cursor:
    t: int                  # position of the last processed token (-1: nothing yet)
    tok: int
    s: Tensor | None        # SSM state at t
    hist: tuple             # codes (np.uint8 [R]) of the last max(ngrams) positions, oldest first


class RefDrafter:
    """PyTorch reference of the per-token CPU drafter (slow; the C++ extension is the fast path)."""

    name = "ref"

    def __init__(self, bundle: Bundle | str | Path, topk: int = 3, shortlist_cap: int | None = None):
        if not isinstance(bundle, Bundle):
            bundle = load_bundle(bundle)
        c = self.cfg = dict(bundle.config)
        t = bundle.tensors()
        self.topk = int(topk)
        self.ngrams = [int(n) for n in c["ngrams"]]
        self.maxn = max(self.ngrams)
        self.R, self.bits, self.V = int(c["routes"]), int(c["bits"]), int(c["vocab"])
        self.kv, self.hd, self.G = int(c["kv_heads"]), int(c["head_dim"]), int(c["groups"])
        self.E = t["embedding"].float()
        self.tables = {n: t[f"table_{n}"] for n in self.ngrams}
        self.mem_norm, self.kv_proj = t["mem_norm"].float(), t["kv_proj"].float()
        self.router = t["router"].float()
        self.prev_norm, self.cur_norm, self.start = t["router_prev_norm"], t["router_cur_norm"], t["router_start"]
        self.has_ssm = "ssm_w_in" in t
        if self.has_ssm:
            self.ssm_norm, self.ssm_w_in, self.ssm_a = t["ssm_norm"], t["ssm_w_in"], t["ssm_decay"]
        self.router_ssm = c["router_kind"] == "embed2_ssm"
        self.q_norm, self.w_q, self.w_o, self.out_norm = t["q_norm"], t["w_q"], t["w_o"], t["out_norm"]
        self.w_sq = t.get("w_sq")
        self.sk_ids = {n: bundle.arrays[f"sketch_ids_{n}"] for n in self.ngrams if f"sketch_ids_{n}" in bundle.arrays}
        self.sk_cnt = {n: bundle.arrays[f"sketch_counts_{n}"] for n in self.ngrams if n in self.sk_ids}
        self.fallback = [int(x) for x in bundle.arrays.get("fallback", np.zeros(0, np.int32))]
        self.cap = int(c["shortlist_cap"] if shortlist_cap is None else shortlist_cap)
        self.shifts = torch.arange(self.bits, dtype=torch.int32)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._tokens: list = []
        self._codes: list = []
        self._ssm: list = []

    @property
    def length(self) -> int:
        return len(self._tokens)

    @property
    def tokens(self) -> list:
        return list(self._tokens)

    def committed_codes(self) -> np.ndarray:
        return np.stack(self._codes) if self._codes else np.zeros((0, self.R), np.uint8)

    def truncate(self, length: int) -> None:
        if not 0 <= length <= self.length:
            raise ValueError(f"cannot truncate to {length} (length {self.length})")
        del self._tokens[length:], self._codes[length:], self._ssm[length:]

    def commit(self, tokens, codes=None) -> None:
        """Append verified tokens; ``codes`` [n, R] (optional) replace the drafter's own codes."""
        cur = self._cursor()
        for i, tok in enumerate(tokens):
            cur = self._advance(cur, int(tok), None if codes is None else np.asarray(codes[i], np.uint8))
            self._tokens.append(int(tok))
            self._codes.append(cur.hist[-1])
            self._ssm.append(cur.s)

    def _cursor(self) -> _Cursor:
        n = self.length
        if n == 0:
            return _Cursor(-1, -1, None, ())
        return _Cursor(n - 1, self._tokens[-1], self._ssm[-1], tuple(self._codes[max(0, n - self.maxn):]))

    # ------------------------------------------------------------------ per token
    @torch.no_grad()
    def _advance(self, cur: _Cursor, tok: int, code: np.ndarray | None = None) -> _Cursor:
        e = self.E[tok]
        s = None
        if self.has_ssm:
            u = self.ssm_w_in @ _rms(e, self.ssm_norm, self.cfg["ssm_eps"])
            prev = cur.s if cur.s is not None else torch.zeros_like(u)
            s = self.ssm_a * prev + (1 - self.ssm_a) * u
        if code is None:
            ep = self.E[cur.tok] if cur.t >= 0 else self.start
            eps = self.cfg["router_eps"]
            parts = [_rms(ep, self.prev_norm, eps), _rms(e, self.cur_norm, eps)]
            if self.router_ssm:
                parts.append(s)
            logits = self.router @ torch.cat(parts)
            bits = (logits > 0).view(self.R, self.bits).to(torch.int32)
            code = (bits << self.shifts).sum(-1).to(torch.uint8).numpy()
        return _Cursor(cur.t + 1, tok, s, (cur.hist + (code,))[-self.maxn:])

    def _rows(self, cur: _Cursor) -> dict:
        out = {}
        r = np.arange(self.R, dtype=np.int64)
        for n in self.ngrams:
            if cur.t < n - 1:
                out[n] = None
                continue
            addr = np.zeros(self.R, np.int64)
            for k in range(n):
                addr |= cur.hist[len(cur.hist) - n + k].astype(np.int64) << (k * self.bits)
            out[n] = r * (1 << self.bits) ** n + addr
        return out

    def _shortlist(self, rows: dict) -> Tensor | None:
        if self.cap <= 0 or not self.sk_ids:
            return None
        ids, cnt = [], []
        for n in self.ngrams:
            if rows[n] is None:
                continue
            ids.append(np.asarray(self.sk_ids[n][rows[n]], np.int64).ravel())
            cnt.append(np.asarray(self.sk_cnt[n][rows[n]], np.int64).ravel())
        out: list = []
        if ids:
            i, c = np.concatenate(ids), np.concatenate(cnt)
            i, c = i[c > 0], c[c > 0]
            if len(i):
                uniq, inv = np.unique(i, return_inverse=True)
                tot = np.zeros(len(uniq), np.int64)
                np.add.at(tot, inv, c)
                order = np.lexsort((uniq, -tot))[: self.cap]
                out = [int(x) for x in uniq[order]]
        if len(out) < self.cap and self.fallback:
            have = set(out)
            for f in self.fallback:
                if len(out) >= self.cap:
                    break
                if f not in have:
                    out.append(f)
                    have.add(f)
        return torch.tensor(out, dtype=torch.long) if out else None

    @torch.no_grad()
    def hidden(self, cur: _Cursor, rows: dict | None = None) -> Tensor:
        rows = self._rows(cur) if rows is None else rows
        c = self.cfg
        mem_dim = int(c["mem_dim"])
        parts = []
        for n in self.ngrams:
            if rows[n] is None:
                parts.append(torch.zeros(self.R, mem_dim))
            else:
                parts.append(self.tables[n][torch.from_numpy(rows[n])].float())
        tok = torch.cat(parts, -1)
        kv = _rms(tok, self.mem_norm, c["mem_eps"]) @ self.kv_proj.T  # [R, 2KH]
        kh = self.kv * self.hd
        key = kv[:, :kh].reshape(self.R, self.kv, self.hd).transpose(0, 1)    # [kv, R, hd]
        val = kv[:, kh:].reshape(self.R, self.kv, self.hd).transpose(0, 1)
        e = self.E[cur.tok]
        q = self.w_q @ _rms(e, self.q_norm, c["draft_eps"])
        if self.w_sq is not None:
            q = q + self.w_sq @ cur.s
        q = q.view(self.kv, self.G, self.hd)
        scores = (q @ key.transpose(-1, -2)) * c["softmax_scale"]               # [kv, G, R]
        sink = torch.tensor(c["sink_logit"], dtype=scores.dtype)
        mx = torch.maximum(scores.amax(-1, keepdim=True), sink)
        ex = torch.exp(scores - mx)
        probs = ex / (ex.sum(-1, keepdim=True) + torch.exp(sink - mx))
        out = (probs @ val).reshape(-1)
        return _rms(self.w_o @ out, self.out_norm, c["draft_eps"])

    @torch.no_grad()
    def _readout(self, cur: _Cursor, full: bool = False) -> tuple[Tensor, Tensor]:
        rows = self._rows(cur)
        h = self.hidden(cur, rows)
        sl = None if full else self._shortlist(rows)
        if sl is None:
            return torch.arange(self.V), self.E @ h
        return sl, self.E[sl] @ h

    # ------------------------------------------------------------------ API
    def draft(self, gamma: int, hyp=(), topk: int | None = None, last_codes: bool = False) -> DraftOut:
        """``gamma`` greedy drafts after the committed tokens + ``hyp`` (nothing is committed).
        ``last_codes`` also advances the last drafted token (its codes, e.g. to stage rows)."""
        k = int(topk or self.topk)
        cur = self._cursor()
        if cur.t < 0 and not len(hyp):
            raise ValueError("draft needs at least one committed or hypothetical token")
        codes = []
        for tok in hyp:
            cur = self._advance(cur, int(tok))
            codes.append(cur.hist[-1])
        ids = np.full((gamma, k), -1, np.int64)
        lg = np.full((gamma, k), -np.inf, np.float32)
        toks = []
        for i in range(gamma):
            sl, logits = self._readout(cur)
            order = torch.sort(logits, descending=True, stable=True).indices[:k]
            ids[i, :len(order)] = sl[order].numpy()
            lg[i, :len(order)] = logits[order].numpy()
            toks.append(int(ids[i, 0]))
            if i + 1 < gamma or last_codes:
                cur = self._advance(cur, toks[-1])
                codes.append(cur.hist[-1])
        return DraftOut(toks, ids, lg, np.stack(codes) if codes else np.zeros((0, self.R), np.uint8))

    def trace(self, tokens, codes=None, full: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Teacher-forced pass from an empty state: per position (codes [T, R], shortlist ids
        [T, S] (-1 pad), logits [T, S] (-inf pad)); S = vocab if ``full`` or no sketch, else cap."""
        tokens = [int(x) for x in tokens]
        full_vocab = full or self.cap <= 0 or not self.sk_ids
        s_dim = self.V if full_vocab else self.cap
        n = len(tokens)
        out_c = np.zeros((n, self.R), np.uint8)
        out_i = np.full((n, s_dim), -1, np.int64)
        out_l = np.full((n, s_dim), -np.inf, np.float32)
        cur = _Cursor(-1, -1, None, ())
        for t, tok in enumerate(tokens):
            cur = self._advance(cur, tok, None if codes is None else np.asarray(codes[t], np.uint8))
            out_c[t] = cur.hist[-1]
            sl, lg = self._readout(cur, full)
            m = min(len(sl), s_dim)  # full-vocab fallback of an empty shortlist is cut to S
            out_i[t, :m], out_l[t, :m] = sl[:m].numpy(), lg[:m].numpy()
        return out_c, out_i, out_l


# ---------------------------------------------------------------------- C++ extension
_EXT = None
_EXT_ERR: str | None = None


def load_extension(verbose: bool = False):
    """The compiled extension (installed by drafting/cpu/setup.py, else JIT-built), or None."""
    global _EXT, _EXT_ERR
    if _EXT is not None or _EXT_ERR is not None:
        return _EXT
    try:
        import morbius_cpu_drafter as ext  # type: ignore
    except ImportError:
        try:
            from torch.utils.cpp_extension import load

            src = Path(__file__).resolve().parent / "cpu" / "drafter.cpp"
            if os.name == "nt":  # the C++ standard flag is left to cpp_extension (matches torch's headers)
                flags = ["/O2"]
            else:
                flags = ["-O3"]
                if os.environ.get("MORBIUS_DRAFTER_NATIVE", "1") == "1":
                    flags.append("-march=native")
            ext = load("morbius_cpu_drafter", [str(src)], extra_cflags=flags, verbose=verbose)
        except Exception as exc:  # no compiler / ninja / headers: callers skip or fall back
            _EXT_ERR = f"{type(exc).__name__}: {exc}"
            return None
    _EXT = ext
    return ext


def extension_error() -> str | None:
    return _EXT_ERR


class CppDrafter:
    """Same API as ``RefDrafter``; every call runs in C++ with the GIL released."""

    name = "cpp"

    def __init__(self, bundle: Bundle | str | Path, topk: int = 3, shortlist_cap: int | None = None,
                 threads: int = 1):
        ext = load_extension()
        if ext is None:
            raise RuntimeError(f"C++ drafter extension unavailable: {_EXT_ERR}")
        if not isinstance(bundle, Bundle):
            bundle = load_bundle(bundle)
        c = self.cfg = dict(bundle.config)
        self.topk = int(topk)
        self.R = int(c["routes"])
        tensors = bundle.tensors()
        tensors = {k: (v if k.startswith("table_") else v.contiguous()) for k, v in tensors.items()}
        cap = int(c["shortlist_cap"] if shortlist_cap is None else shortlist_cap)
        num = {k: float(c[k]) for k in ("d_model", "vocab", "routes", "bits", "mem_dim", "kv_heads", "head_dim",
                                        "q_heads", "ssm_dim", "sketch_k", "norm_eps", "ssm_eps", "router_eps",
                                        "draft_eps", "mem_eps", "softmax_scale", "sink_logit")}
        num["shortlist_cap"] = float(cap)
        num["router_ssm"] = float(c["router_kind"] == "embed2_ssm")
        num["query_ssm"] = float(bool(c["query_ssm"]))
        self._core = ext.Drafter(tensors, num, [int(n) for n in c["ngrams"]], int(threads))
        self._tokens: list = []

    @property
    def length(self) -> int:
        return int(self._core.length())

    @property
    def tokens(self) -> list:
        return list(self._tokens)

    def committed_codes(self) -> np.ndarray:
        return self._core.committed_codes().numpy()

    def reset(self) -> None:
        self._core.reset()
        self._tokens = []

    def truncate(self, length: int) -> None:
        if not 0 <= length <= self.length:
            raise ValueError(f"cannot truncate to {length} (length {self.length})")
        self._core.truncate(int(length))
        del self._tokens[length:]

    def commit(self, tokens, codes=None) -> None:
        toks = [int(x) for x in tokens]
        ct = None if codes is None else torch.as_tensor(np.asarray(codes), dtype=torch.int32).contiguous()
        self._core.commit(toks, ct)
        self._tokens.extend(toks)

    def draft(self, gamma: int, hyp=(), topk: int | None = None, last_codes: bool = False) -> DraftOut:
        toks, ids, lg, codes = self._core.draft(int(gamma), [int(x) for x in hyp], int(topk or self.topk),
                                                bool(last_codes))
        return DraftOut([int(x) for x in toks.tolist()], ids.numpy(), lg.numpy(), codes.numpy().astype(np.uint8))

    def trace(self, tokens, codes=None, full: bool = False):
        ct = None if codes is None else torch.as_tensor(np.asarray(codes), dtype=torch.int32).contiguous()
        c, i, lg = self._core.trace([int(x) for x in tokens], ct, bool(full))
        return c.numpy().astype(np.uint8), i.numpy(), lg.numpy()


def make_drafter(bundle: Bundle | str | Path, impl: str = "auto", **kw):
    """``impl``: ref | cpp | auto (C++ if it builds, else the reference)."""
    if impl == "ref" or (impl == "auto" and load_extension() is None):
        kw.pop("threads", None)
        return RefDrafter(bundle, **kw)
    return CppDrafter(bundle, **kw)


# ---------------------------------------------------------------------- CLI
def _tokens_for(bundle: Bundle, path: str | None, n: int, seed: int) -> list:
    if path:
        arr = np.load(path) if path.endswith(".npy") else np.fromfile(path, dtype="<u2")
        return [int(x) for x in np.asarray(arr).ravel()[:n]]
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(0, int(bundle.config["vocab"]), n)]


def bench(bundle: Bundle, impl: str, threads: int, gamma: int, iters: int, prompt: list, warmup: int = 20) -> dict:
    d = make_drafter(bundle, impl, threads=threads) if impl == "cpp" else RefDrafter(bundle)
    d.commit(prompt)
    per_tok = []
    for i in range(warmup + iters):
        t0 = time.perf_counter()
        out = d.draft(gamma)
        dt = time.perf_counter() - t0
        if i >= warmup:
            per_tok.append(dt / gamma * 1e6)
        d.commit(out.tokens[:1])
    a = np.asarray(per_tok)
    return {"impl": d.name, "threads": threads if d.name == "cpp" else 1, "gamma": gamma, "iters": iters,
            "us_per_token_median": float(np.median(a)), "us_per_token_p99": float(np.percentile(a, 99)),
            "us_per_token_mean": float(a.mean())}


def check(bundle: Bundle, tokens: list, threads: int = 1) -> dict:
    """C++ vs reference on teacher-forced positions (same codes, hence the same shortlist)."""
    ref = RefDrafter(bundle)
    cpp = CppDrafter(bundle, threads=threads)
    c_ref, i_ref, l_ref = ref.trace(tokens)
    c_cpp, _, _ = cpp.trace(tokens)
    _, i_cpp, l_cpp = cpp.trace(tokens, codes=c_ref)
    fin = np.isfinite(l_ref)
    return {"positions": len(tokens), "code_mismatch_frac": float((c_ref != c_cpp).mean()),
            "shortlist_equal": bool(np.array_equal(i_ref, i_cpp)),
            "max_abs_logit_diff": float(np.abs(l_ref[fin] - l_cpp[fin]).max()) if fin.any() else 0.0,
            "argmax_agree": float((l_ref.argmax(1) == l_cpp.argmax(1)).mean())}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bench", metavar="BUNDLE")
    g.add_argument("--check", metavar="BUNDLE")
    ap.add_argument("--impl", choices=("cpp", "ref"), default="cpp")
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--prompt_len", type=int, default=256)
    ap.add_argument("--positions", type=int, default=10000)
    ap.add_argument("--tokens", default=None, help=".npy or raw uint16 token file (real text); default random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    path = args.bench or args.check
    bundle = load_bundle(path)
    if args.check:
        res = {"bundle": path, **check(bundle, _tokens_for(bundle, args.tokens, args.positions, args.seed),
                                       args.threads[0])}
        tag = "check"
    else:
        prompt = _tokens_for(bundle, args.tokens, args.prompt_len, args.seed)
        res = {"bundle": path, "config": bundle.config,
               "runs": [bench(bundle, args.impl, t, args.gamma, args.iters, prompt) for t in args.threads]}
        tag = "bench"
    out = Path(args.out or f"results/cpu_drafter_{tag}_{Path(path).stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
