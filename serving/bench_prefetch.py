"""Per-token decode latency of host-RAM serving (M4.T2): B0 / B1 / B4, eager and CUDA-graph.

    python -m serving.bench_prefetch --ckpt runs/w3_sdb_1b --server gpu0 --run sdb --dtype fp16
    python -m serving.bench_prefetch --config configs/tiny_cpu.yaml --server laptop --run tiny \
        --device cpu --batches 1,2 --warmup 4 --tokens 16 --repeats 2

Conditions (``--modes``):
  b0   table on the GPU, eager ``decode_step``
  b0g  table on the GPU, ``StaticDecoder`` (fixed-shape step) replayed as CUDA graphs
  b1   table in pinned host RAM, eager ``decode_step`` + ``HostGather`` (sync exact gather)
  b4   table in pinned host RAM, eager ``decode_step`` + ``CandidatePrefetcher``
  b4g  table in pinned host RAM, ``StaticDecoder`` graphs + candidate prefetch thread

Per condition and batch (1, 2, 4, 8, 16, 32): ``--repeats`` runs of ``--warmup`` +
``--tokens`` greedy steps after a ``--prompt_len`` prefill; each step is timed on the
host with a device synchronize after it (per-token latency); reported median / p99
over all repeats, tokens/s, compute-stream wait on the copy events (CUDA events),
PCIe bytes per token, prefetch hits/misses, ``torch.cuda.max_memory_allocated`` and
the process's ``nvidia-smi`` resident memory.  B0 conditions run first; then the
tables are offloaded (meta tensors) for B1 / B4, so the memory columns show the saving.

What the CUDA graphs capture (``StaticDecoder``): one decode step with fixed shapes
(KV cache over ``max_len`` with a length mask, position / length in a device tensor,
greedy argmax written back into the static token buffer), split into segments at
each memory layer's codes.  Captured: all model kernels, the codes-window D2H copy
into the pinned window buffer, and (B4) the GPU selection of rows from the staged
candidates.  NOT captured (host side, between segment replays): joining the staging
thread, ``wait_event`` on the candidate copy event, recording the codes event and
submitting the next staging job; the thread's ``index_select`` and its side-stream
H2D copy.  B4 needs two graph chains (staging slot parity).  B1 cannot be captured
(host gather in the middle of the step).  The static step needs L >= max(n-gram
history, conv left padding) and therefore runs after an eager prefill.
JSON: results/prefetch_<server>_<run>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ffnmem.infer import DecodeState, decode_step, rope_at, ssm_continue
from serving.exactness import DTYPES, build_model, load_prompts
from serving.prefetch import CandidatePrefetcher, HostGather, HostTables, ServeStats, offload_tables, restore_tables


# ------------------------------------------------------------------ static step
class StaticDecoder:
    """Fixed-shape single-token decode step on a ``DecodeState`` (see module docstring).

    ``mode`` "b0" (device table) or "b4" (rows from ``prefetcher``'s staged candidates).
    ``start(L, token)`` sets the position (L tokens cached) and the input token [B];
    ``step()`` runs one position and returns logits [B, 1, V]; the greedy next token is
    already in the token buffer, so ``step()`` can be called repeatedly."""

    def __init__(self, model, state: DecodeState, mode: str = "b0", prefetcher: CandidatePrefetcher | None = None):
        if mode not in ("b0", "b4"):
            raise ValueError(mode)
        if mode == "b4" and prefetcher is None:
            raise ValueError("b4 needs a CandidatePrefetcher")
        self.model, self.state, self.mode, self.pf = model, state, mode, prefetcher
        dev = model.tok_emb.weight.device
        self.dev, self.cuda = dev, dev.type == "cuda"
        b = state.batch
        self.mem = [p for p, blk in enumerate(model.blocks) if blk.memory is not None]
        self.hist = {p: model.blocks[p].memory.module.route_history_length for p in self.mem}
        self.lpad = {p: model.blocks[p].memory.module.short_conv.left_padding for p in self.mem}
        self.tok = torch.zeros(b, 1, dtype=torch.long, device=dev)
        self.Lbuf = torch.zeros(1, dtype=torch.long, device=dev)
        self.ar = torch.arange(state.max_len, device=dev)
        self.key = (id(state), b)
        self.L = 0
        self.buf: dict = {}
        self.graphs: dict | None = None
        self.keep: dict = {}
        if self.pf is not None:
            self.pf.stager.ensure(b)

    @property
    def min_length(self) -> int:
        return max([1] + [max(self.hist[p], self.lpad[p]) for p in self.mem])

    def start(self, L: int, token: torch.Tensor) -> None:
        if L < self.min_length:
            raise ValueError(f"static step needs L >= {self.min_length}, got {L}")
        self.L = L
        self.state.length = L
        self.Lbuf.fill_(L)
        self.tok.copy_(token.view(-1, 1))

    # -------------------------------------------------------------- pieces
    def _route_codes(self, pos: int, h, emb, s_new):
        mem = self.model.blocks[pos].memory
        st, Lb = self.state, self.Lbuf
        if mem.router_input == "hidden":
            q = mem.route_logits(self.model.blocks[pos].attn_norm(h))
        else:
            prev = self.model.tok_emb(st.tokens.index_select(1, Lb - 1)).to(emb.dtype)
            s_cat = None if s_new is None else torch.cat((st.ssm.index_select(1, Lb - 1), s_new), 1)
            q = mem.route_logits(mem.router_features(torch.cat((prev, emb), 1), s_cat)[:, 1:])
        codes = mem.module.quantize_route_logits(q)  # [B, 1, R]
        st.codes[pos].index_copy_(1, Lb, codes.to(st.codes[pos].dtype))
        return codes

    def _window(self, pos: int):
        H = self.hist[pos]
        return self.state.codes[pos].index_select(1, self.Lbuf - H + self.ar[:H + 1])  # [B, H+1, R]

    def _readout(self, pos: int, h, codes, par: int):
        m = self.model.blocks[pos].memory.module
        st, Lb, b = self.state, self.Lbuf, h.shape[0]
        if self.mode == "b0":
            tokens = m._lookup_route_chunk(self._window(pos), None, 0, m.num_routes, None, None, target_length=1)
        else:
            tokens = self.pf.stager.select(pos, par, codes[:, 0])[:, None].to(m.readout_kv_proj.weight.dtype)
        key, value = m._project_memory(tokens)
        head = m._manual_gqa(m._project_query(h), key, value)
        mixed = m.readout_o_proj(head.reshape(b, 1, m.readout_inner_dim))
        conv = m.short_conv
        st.conv_in[pos].index_copy_(1, Lb, conv.norm(mixed).to(st.conv_in[pos].dtype))
        lp = self.lpad[pos]
        window = st.conv_in[pos].index_select(1, Lb - lp + self.ar[:lp + 1])
        conv_out = conv.act(conv.conv(window.transpose(1, 2).to(mixed.dtype))).transpose(1, 2)
        return mixed + conv_out

    def _attend(self, pos: int, x):
        a = self.model.blocks[pos].attn
        st, Lb = self.state, self.Lbuf
        b = x.shape[0]
        positions = Lb.view(1, 1).expand(b, 1)
        cos, sin = self.model.rope_cos, self.model.rope_sin
        q = rope_at(a.wq(x).view(b, 1, a.n_heads, a.head_dim), positions, cos, sin).transpose(1, 2)
        k = rope_at(a.wk(x).view(b, 1, a.n_kv_heads, a.head_dim), positions, cos, sin).transpose(1, 2)
        v = a.wv(x).view(b, 1, a.n_kv_heads, a.head_dim).transpose(1, 2)
        st.k[pos].index_copy_(2, Lb, k.to(st.k[pos].dtype))
        st.v[pos].index_copy_(2, Lb, v.to(st.v[pos].dtype))
        kk, vv = st.k[pos], st.v[pos]
        g = a.n_heads // a.n_kv_heads
        if g > 1:
            kk, vv = kk.repeat_interleave(g, 1), vv.repeat_interleave(g, 1)
        mask = (self.ar <= Lb).view(1, 1, 1, -1)
        out = F.scaled_dot_product_attention(q, kk.to(q.dtype), vv.to(q.dtype), attn_mask=mask)
        return a.wo(out.transpose(1, 2).reshape(b, 1, a.n_heads * a.head_dim))

    def _blocks(self, h, lo: int, hi: int):
        for pos in range(lo, hi):
            blk = self.model.blocks[pos]
            h = h + self._attend(pos, blk.attn_norm(h))
            h = h + blk.ffn(blk.ffn_norm(h))
        return h

    # -------------------------------------------------------------- segments
    def n_segments(self) -> int:
        return len(self.mem) + 1

    def segment(self, i: int, par: int) -> None:
        """Segment i: (i > 0) readout + rest of memory block mem[i-1], plain blocks up to the
        next memory layer, (i < k) its codes + codes-window D2H; the last one ends the step."""
        model, st, buf, Lb = self.model, self.state, self.buf, self.Lbuf
        k = len(self.mem)
        if i == 0:
            emb = model.tok_emb(self.tok)
            s_new = None
            if model.ssm is not None:
                s_new = ssm_continue(model, emb.detach(), st.ssm.index_select(1, Lb - 1)[:, 0])
            buf.update(emb=emb, s_new=s_new)
            h, lo = emb, 0
        else:
            pos = self.mem[i - 1]
            h = buf["h"]
            blk = model.blocks[pos]
            h = h + self._readout(pos, h, buf["codes"], par)
            h = h + self._attend(pos, blk.attn_norm(h))
            h = h + blk.ffn(blk.ffn_norm(h))
            lo = pos + 1
        if i < k:
            pos = self.mem[i]
            h = self._blocks(h, lo, pos)
            codes = self._route_codes(pos, h, buf["emb"], buf["s_new"])
            if self.mode == "b4":
                H = self.hist[pos]
                self.pf.stager.window(pos, par + 1, H + 1).copy_(self._window(pos), non_blocking=True)
            buf.update(h=h, codes=codes)
        else:
            h = self._blocks(h, lo, len(model.blocks))
            if buf["s_new"] is not None:
                st.ssm.index_copy_(1, Lb, buf["s_new"])
            st.tokens.index_copy_(1, Lb, self.tok)
            logits = model.output(model.norm(h))
            self.tok.copy_(logits[:, -1].argmax(-1, keepdim=True))
            Lb.add_(1)
            buf["logits"] = logits

    # -------------------------------------------------------------- driver
    def step(self) -> torch.Tensor:
        L, k = self.L, len(self.mem)
        par = L % 2 if self.mode == "b4" else 0
        stg = self.pf.stager if self.pf is not None else None
        for i in range(k + 1):
            if stg is not None and i >= 1:
                pos = self.mem[i - 1]
                if not stg.ready(pos, L, self.key):
                    raise RuntimeError(f"no staged candidates for layer {pos} position {L}")
                stg.wait(pos, L)
            if stg is not None and i < k:
                stg.acquire(self.mem[i], L + 1)
            if self.graphs is not None:
                self.graphs[par][i].replay()
            else:
                self.segment(i, par)
            if stg is not None and i < k:
                ev = None
                if self.cuda:
                    ev = torch.cuda.Event()
                    ev.record()
                    self.pf.stats.add(d2h_bytes=self._window_bytes(self.mem[i]))
                stg.submit(self.mem[i], L + 1, self.hist[self.mem[i]] + 1, self.key, ev)
        if self.pf is not None:
            self.pf.stats.add(tokens=self.state.batch, hits=k)
        self.L = L + 1
        self.state.length = self.L
        return (self.keep[par]["logits"] if self.graphs is not None else self.buf["logits"])

    def _window_bytes(self, pos: int) -> int:
        return self.state.batch * (self.hist[pos] + 1) * self.model.blocks[pos].memory.routes

    def capture(self, warmup: int = 2) -> None:
        """Capture every segment as a CUDA graph (both parities for B4).  State writes made
        by the warmup runs are at position L and are overwritten by the next real step."""
        if not self.cuda:
            raise RuntimeError("CUDA graphs need a CUDA device")
        if self.pf is not None:
            self.pf.stager.drain()
        L0, tok0 = self.L, self.tok.clone()
        pars = (0, 1) if self.mode == "b4" else (0,)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                for par in pars:
                    self.Lbuf.fill_(L0)
                    self.tok.copy_(tok0)
                    for i in range(self.n_segments()):
                        self.segment(i, par)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.Lbuf.fill_(L0)
        self.tok.copy_(tok0)
        graphs = {}
        for par in pars:
            chain, pool = [], None
            for i in range(self.n_segments()):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool, capture_error_mode="thread_local"):
                    self.segment(i, par)
                pool = g.pool()
                chain.append(g)
            graphs[par] = chain
            self.keep[par] = dict(self.buf)  # keep this chain's inter-segment tensors alive
        if self.mode == "b0":
            graphs[1] = graphs[0]
            self.keep[1] = self.keep[0]
        self.graphs = graphs
        torch.cuda.synchronize()
        self.Lbuf.fill_(L0)
        self.tok.copy_(tok0)


# ------------------------------------------------------------------ measurement
def sync(dev) -> None:
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize(dev)


def nvidia_smi_mib() -> float | None:
    """This process's resident GPU memory from nvidia-smi (MiB)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0] == str(os.getpid()):
            return float(parts[1])
    return None


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


@torch.no_grad()
def bench_condition(model, mode: str, prompts: torch.Tensor, warmup: int, tokens: int, repeats: int,
                    host: HostTables | None, graph: bool = True, threaded: bool = True) -> dict:
    dev = model.tok_emb.weight.device
    b, p = prompts.shape
    prompts = prompts.to(dev)
    provider, stats = None, ServeStats()
    if mode == "b1":
        provider = HostGather(host, dev, stats)
    elif mode in ("b4", "b4g"):
        provider = CandidatePrefetcher(host, dev, threaded=threaded, stats=stats)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    state = DecodeState.empty(model, b, p + warmup + tokens)
    decode_step(model, state, prompts[:, :-1], row_provider=provider)
    static = None
    info: dict = {"mode": mode, "batch": b}
    if mode in ("b0g", "b4g"):
        static = StaticDecoder(model, state, "b0" if mode == "b0g" else "b4", provider)
        if p - 1 < static.min_length:
            raise ValueError(f"prompt_len must be > {static.min_length} for {mode}")
    lat: list[float] = []
    waits: list[float] = []
    agg = {"h2d_bytes": 0, "d2h_bytes": 0, "tokens": 0, "hits": 0, "misses": 0, "stale": 0, "host_wait_ms": 0.0}
    for rep in range(repeats):
        state.truncate(p - 1)
        logits = decode_step(model, state, prompts[:, -1:], row_provider=provider)
        nxt = logits[:, -1].argmax(-1)
        if static is not None:
            static.start(p, nxt)
            if graph and static.graphs is None and dev.type == "cuda":
                try:
                    static.capture()
                    info["graph"] = True
                except Exception as e:  # noqa: BLE001 (report and fall back to the eager static step)
                    info["graph"], info["graph_error"] = False, repr(e)[:500]
                    static.graphs = None
                    static.start(p, nxt)
            info.setdefault("graph", static.graphs is not None)
        sync(dev)
        for i in range(warmup + tokens):
            if i == warmup:
                if isinstance(provider, CandidatePrefetcher):
                    provider.stager.drain()  # count exactly the jobs issued by timed steps
                stats.reset()
            t0 = time.perf_counter()
            if static is not None:
                static.step()
            else:
                logits = decode_step(model, state, nxt[:, None], row_provider=provider)
                nxt = logits[:, -1].argmax(-1)
            sync(dev)
            if i >= warmup:
                lat.append(time.perf_counter() - t0)
        s = stats.summary()
        waits += stats.gpu_waits_ms()
        for k in ("h2d_bytes", "d2h_bytes", "tokens", "hits", "misses", "stale"):
            agg[k] += s[k]
        agg["host_wait_ms"] += s["host_wait_ms_total"]
    if provider is not None:
        provider.close()
    ms = [1e3 * x for x in lat]
    n_tok = b * len(lat)
    info.update({"latency_ms_median": statistics.median(ms), "latency_ms_p99": pct(ms, 0.99),
                 "latency_ms_mean": statistics.fmean(ms), "tokens_per_s": n_tok / sum(lat),
                 "steps": len(lat), "repeats": repeats})
    if provider is not None:
        tok = max(1, agg["tokens"])
        info.update({"pcie_bytes_per_token": (agg["h2d_bytes"] + agg["d2h_bytes"]) / tok,
                     "h2d_bytes_per_token": agg["h2d_bytes"] / tok, "hits": agg["hits"], "misses": agg["misses"],
                     "stale": agg["stale"], "host_wait_ms_per_step": agg["host_wait_ms"] / max(1, len(lat)),
                     "gpu_wait_ms_per_step": sum(waits) / max(1, len(lat)),
                     "gpu_wait_ms_median": statistics.median(waits) if waits else None,
                     "gpu_wait_ms_p99": pct(waits, 0.99) if waits else None})
    if dev.type == "cuda":
        info["max_memory_allocated_mib"] = torch.cuda.max_memory_allocated(dev) / 2 ** 20
        info["nvidia_smi_mib"] = nvidia_smi_mib()
    return info


def run_bench(model, prompts_all: torch.Tensor, batches, modes, warmup: int, tokens: int, repeats: int,
              graph: bool = True, threaded: bool = True, log=print) -> list[dict]:
    host = HostTables.from_model(model) if any(m.startswith(("b1", "b4")) for m in modes) else None
    rows = []
    phases = [[m for m in modes if m.startswith("b0")], [m for m in modes if not m.startswith("b0")]]
    for phase, ms in enumerate(phases):
        if not ms:
            continue
        if phase == 1:
            offload_tables(model)
        for b in batches:
            for mode in ms:
                r = bench_condition(model, mode, prompts_all[:b], warmup, tokens, repeats, host, graph, threaded)
                log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
                rows.append(r)
    if host is not None and phases[1]:
        restore_tables(model, host)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt")
    src.add_argument("--config")
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--server", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--modes", default="b0,b0g,b1,b4,b4g")
    ap.add_argument("--prompt_len", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_graph", action="store_true")
    ap.add_argument("--no_thread", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    batches = [int(x) for x in args.batches.split(",")]
    model = build_model(args, DTYPES[args.dtype], args.prompt_len + args.warmup + args.tokens)
    prompts = load_prompts(max(batches), args.prompt_len, model.cfg["vocab"], args.data_dir, args.seed)
    rows = run_bench(model, prompts, batches, args.modes.split(","), args.warmup, args.tokens, args.repeats,
                     graph=not args.no_graph, threaded=not args.no_thread)
    out = {"server": args.server, "run": args.run, "ckpt": args.ckpt, "config": args.config, "dtype": args.dtype,
           "device": args.device, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
           "torch": torch.__version__, "cuda": torch.version.cuda, "prompt_len": args.prompt_len,
           "warmup": args.warmup, "tokens": args.tokens, "repeats": args.repeats,
           "table_bytes": HostTables.from_model(model, pin=False).nbytes, "rows": rows}
    path = Path(args.out_dir) / f"prefetch_{args.server}_{args.run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
