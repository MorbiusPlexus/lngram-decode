"""Serving the Lngram memory table from host RAM (M4.T2): conditions B0 / B1 / B4.

    model, cfg = load_model(ckpt, device="cuda", dtype=torch.float16)
    host = HostTables.from_model(model)            # serving layout, page-locked host RAM
    offload_tables(model)                           # drop the GPU copy (meta tensors)
    b1 = HostGather(host, "cuda")                   # B1: synchronous exact gather
    b4 = CandidatePrefetcher(host, "cuda")          # B4: 16-candidate prefetch for step t+1
    logits = decode_step(model, state, tok, row_provider=b4)
    print(b4.stats.summary())

B0 is plain ``decode_step`` with the table on the GPU (no provider).

B1 (``HostGather``): at the memory layer the step's codes (+ n-gram history) are
copied to the host, the exact rows are gathered with one ``index_select`` from the
page-locked table into a pinned buffer and copied to the GPU; the step waits.

B4 (``CandidatePrefetcher``): as soon as step t's codes exist (inside the provider
call of step t, after step t's rows were selected) the codes window is copied to a
pinned buffer (async D2H + event) and a background thread gathers, per route and
order, the contiguous 16-row candidate block of step t+1 (serving layout,
``serving/layout.py``) into a pinned staging buffer and issues one ``non_blocking``
H2D copy on a side stream, recording an event.  Staging is double-buffered by
position parity.  At step t+1 the provider joins the thread (host wait), makes the
compute stream wait on the copy event (timed with CUDA events) and selects the
rows by c[t+1] on the GPU (``select_candidates``).  Prefill / multi-token blocks
and misses use the exact synchronous gather.  Positions where an n-gram does not
fit (t < n-1) read a zero row, as in the v2 lookup, so B1 and B4 are bitwise equal
to B0.

Speculative decoding (``BlockStager``, used by ``drafting/live_spec.py``): the
drafted tokens' codes are known on the CPU before verification, so the rows of
the whole verify block are staged exactly; the verifier's bonus token (whose code
is unknown) gets 16-candidate blocks for every acceptance length::

    stager = BlockStager(host, "cuda")
    blk = stager.stage(start=L, codes={pos: uint8 [B, n, R]}, history={pos: uint8 [B, min(L, H), R]})
    ...draft the next block while blk is staged in the background...
    logits = decode_step(model, state, block_tokens, row_provider=stager.provider(blk))
    if not blk.codes_match(): ...       # drafter codes != verifier codes: redo with stager.fallback
    # next verify block starts with the bonus token at position L + a + 1 (a drafts accepted):
    nxt = stager.stage(start=L + a + 2, codes=..., history=...)
    decode_step(model, state, [bonus] + drafts, row_provider=stager.provider(nxt, head=(blk, a)))

On CPU (tests) pinned memory, streams and events degrade to plain tensors and
synchronous copies; the row selection logic is the same.  Transfer mode (ii),
zero-copy reads of mapped host memory, is not implemented (``ZeroCopyGather``).
"""

from __future__ import annotations

import threading
import time
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from serving.layout import candidate_base, permute_table, select_candidates, serving_address


# ------------------------------------------------------------------ host memory
def cuda_device(device) -> bool:
    return torch.device(device).type == "cuda" and torch.cuda.is_available()


def norm_device(device) -> torch.device:
    """torch.device with an explicit index for CUDA (thread initializers need one)."""
    d = torch.device(device)
    if d.type == "cuda" and d.index is None and torch.cuda.is_available():
        d = torch.device("cuda", torch.cuda.current_device())
    return d


def _unregister(ptr: int) -> None:
    try:
        torch.cuda.cudart().cudaHostUnregister(ptr)
    except Exception:  # interpreter shutdown
        pass


def pin_(t: Tensor) -> Tensor:
    """Page-lock ``t`` in place with cudaHostRegister (no power-of-two rounding as in the
    caching host allocator, and the pages stay on the NUMA node that first touched them).
    No-op without CUDA."""
    if not torch.cuda.is_available() or t.device.type != "cpu" or t.is_pinned():
        return t
    rt = torch.cuda.cudart()
    err = rt.cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
    if int(err) != 0:
        raise RuntimeError(f"cudaHostRegister failed ({err}) for {t.numel() * t.element_size()} bytes")
    weakref.finalize(t, _unregister, t.data_ptr())
    return t


def host_empty(shape, dtype, pinned: bool = True) -> Tensor:
    """Small host buffer (staging); pinned when CUDA is available."""
    return torch.empty(shape, dtype=dtype, pin_memory=bool(pinned and torch.cuda.is_available()))


def _ramp(n: int) -> Tensor:
    return torch.arange(n, dtype=torch.long)


# ------------------------------------------------------------------ tables
class LayerTable:
    """All n-gram orders of one memory layer in ONE host tensor, serving layout.

    Rows ``offsets[o] + r * 16^o + addr`` (``addr`` with the current code least
    significant, ``serving/layout.py``); the last row is zeros and is read where an
    n-gram does not fit.  One ``index_select`` then fills [B, n, R, len(ngrams), D],
    which is the route-token layout [B, n, R, len(ngrams) * D] of the v2 lookup."""

    def __init__(self, weight: Tensor, routes: int, ngrams, bits: int):
        self.weight = weight
        self.routes, self.ngrams, self.bits = int(routes), tuple(int(o) for o in ngrams), int(bits)
        self.alphabet = 1 << self.bits
        self.dim = int(weight.shape[1])
        self.offsets, off = {}, 0
        for o in self.ngrams:
            self.offsets[o] = off
            off += self.routes * self.alphabet ** o
        if weight.shape[0] != off + 1:
            raise ValueError(f"weight has {weight.shape[0]} rows, layout needs {off + 1}")
        self.zero_row = off
        self.history = max(self.ngrams) - 1

    @staticmethod
    def num_rows(routes: int, ngrams, bits: int) -> int:
        return sum(int(routes) * (1 << bits) ** int(o) for o in ngrams) + 1

    @classmethod
    def from_module(cls, m, pin: bool = True) -> "LayerTable":
        """From a ``RouteNgramMemoryV2`` (v2 layout tables, any device)."""
        r, bits, dim = m.num_routes, m.bits_per_route, m.memory_dim
        w0 = m.tables[f"ngram_{m.ngrams[0]}"].weight
        weight = torch.empty(cls.num_rows(r, m.ngrams, bits), dim, dtype=w0.dtype)
        off = 0
        for o in m.ngrams:
            w = m.tables[f"ngram_{o}"].weight.detach().to("cpu")
            weight[off:off + w.shape[0]] = permute_table(w, r, o, bits)
            off += w.shape[0]
        weight[off].zero_()
        if pin:
            pin_(weight)
        return cls(weight, r, m.ngrams, bits)

    @classmethod
    def random(cls, routes: int, ngrams, bits: int, dim: int, dtype=torch.float16, seed: int = 0,
               pin: bool = True, block_rows: int = 1 << 16) -> "LayerTable":
        """Random-weight table (scaled-table benchmark).  Filled by tiling one random block,
        so a 10 GB table is written at memcpy speed (values do not matter for latency)."""
        rows = cls.num_rows(routes, ngrams, bits)
        weight = torch.empty(rows, dim, dtype=dtype)
        g = torch.Generator().manual_seed(seed)
        block = torch.randn(min(block_rows, rows), dim, generator=g).to(dtype)
        for s in range(0, rows - 1, block.shape[0]):
            e = min(rows - 1, s + block.shape[0])
            weight[s:e] = block[: e - s]
        weight[-1].zero_()
        if pin:
            pin_(weight)
        return cls(weight, routes, ngrams, bits)

    @property
    def nbytes(self) -> int:
        return self.weight.numel() * self.weight.element_size()

    @property
    def row_bytes(self) -> int:
        return self.dim * self.weight.element_size()

    def table(self, order: int) -> Tensor:
        """Serving-layout table of one order (a view)."""
        off = self.offsets[order]
        return self.weight[off: off + self.routes * self.alphabet ** order]

    def _route_base(self, order: int) -> Tensor:
        return _ramp(self.routes) * self.alphabet ** order + self.offsets[order]

    def exact_index(self, ctx: Tensor, n: int) -> Tensor:
        """Row indices [B, n, R, len(ngrams)] for the last ``n`` positions of ``ctx``.

        ``ctx`` [B, h + n, R] codes; ``h`` history codes precede the n new positions and
        h = min(L, history) (L = absolute position of the first new one), so an n-gram
        that runs off the start of ``ctx`` starts before position 0 -> zero row."""
        b, t, r = ctx.shape
        h = t - n
        c = ctx.to(torch.long)
        out = torch.full((b, n, r, len(self.ngrams)), self.zero_row, dtype=torch.long)
        for k, o in enumerate(self.ngrams):
            j0 = max(0, o - 1 - h)
            if j0 >= n:
                continue
            addr = serving_address(c[:, h + j0 - (o - 1):], o, self.bits)  # windows ending at h+j0 .. t-1
            out[:, j0:, :, k] = addr + self._route_base(o).view(1, 1, r)
        return out

    def candidate_index(self, hist: Tensor, position: int) -> Tensor:
        """Row indices [B, R, len(ngrams), 16] of the candidate blocks for ``position``.

        ``hist`` [B, h, R]: codes ending at position-1, h = min(position, history)."""
        b, h, r = hist.shape
        if h != min(position, self.history):
            raise ValueError(f"history of {h} codes for position {position} (need {min(position, self.history)})")
        out = torch.full((b, r, len(self.ngrams), self.alphabet), self.zero_row, dtype=torch.long)
        for k, o in enumerate(self.ngrams):
            if h < o - 1:
                continue  # position < o - 1: the n-gram does not fit
            base = candidate_base(hist[:, h - (o - 1):].to(torch.long), o, self.bits) + self.offsets[o]
            out[:, :, k] = base[..., None] + _ramp(self.alphabet)
        return out

    def gather(self, idx: Tensor, out: Tensor | None = None) -> Tensor:
        """``index_select`` of rows ``idx`` (any shape) -> [*idx.shape, D] (into ``out`` if given)."""
        flat = idx.reshape(-1)
        if out is None:
            return self.weight.index_select(0, flat).view(*idx.shape, self.dim)
        torch.index_select(self.weight, 0, flat, out=out.view(-1, self.dim))
        return out.view(*idx.shape, self.dim)


class HostTables(dict):
    """{memory block pos: LayerTable} for a model."""

    @classmethod
    def from_model(cls, model, pin: bool = True) -> "HostTables":
        out = cls()
        for pos, block in enumerate(model.blocks):
            if block.memory is not None:
                out[pos] = LayerTable.from_module(block.memory.module, pin=pin)
        if not out:
            raise ValueError("model has no memory layers")
        return out

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for t in self.values())


def offload_tables(model) -> int:
    """Replace every memory table on the model by a meta tensor (frees the GPU copy).

    After this only ``row_provider`` decoding works (B0 lookups fail loudly).  Call it
    after any ``model.to(...)``.  Returns the bytes freed."""
    freed = 0
    for block in model.blocks:
        if block.memory is None:
            continue
        for emb in block.memory.module.tables.values():
            w = emb.weight
            if w.device.type == "meta":
                continue
            freed += w.numel() * w.element_size()
            emb.weight = nn.Parameter(torch.empty(w.shape, dtype=w.dtype, device="meta"), requires_grad=False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return freed


def restore_tables(model, host: HostTables) -> None:
    """Undo ``offload_tables`` (the serving permutation is an involution)."""
    dev = model.tok_emb.weight.device
    for pos, lt in host.items():
        m = model.blocks[pos].memory.module
        for o in lt.ngrams:
            w = permute_table(lt.table(o), lt.routes, o, lt.bits).to(dev)
            m.tables[f"ngram_{o}"].weight = nn.Parameter(w, requires_grad=False)


# ------------------------------------------------------------------ accounting
@dataclass
class ServeStats:
    tokens: int = 0            # token positions served (counted once per step, not per layer)
    h2d_bytes: int = 0         # table rows moved host -> GPU
    d2h_bytes: int = 0         # codes moved GPU -> host
    hits: int = 0              # single-token steps served from staged candidates / staged blocks
    misses: int = 0            # single-token steps (or staged blocks) that fell back to the exact gather
    exact_gathers: int = 0     # synchronous exact gathers (B1, prefill, multi-token blocks, misses)
    stale: int = 0             # staged history later found inconsistent with the cache (must stay 0)
    host_wait_s: float = 0.0   # main thread waiting for the staging thread
    gather_s: float = 0.0      # host index_select time (both paths)
    _waits: list = field(default_factory=list)       # (start, end) CUDA event pairs on the compute stream
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def reset(self) -> None:
        with self._lock:
            for k in ("tokens", "h2d_bytes", "d2h_bytes", "hits", "misses", "exact_gathers", "stale"):
                setattr(self, k, 0)
            self.host_wait_s = self.gather_s = 0.0
            self._waits.clear()

    def gpu_waits_ms(self) -> list[float]:
        if not self._waits:
            return []
        self._waits[-1][1].synchronize()
        return [a.elapsed_time(b) for a, b in self._waits]

    def summary(self) -> dict:
        waits = sorted(self.gpu_waits_ms())
        tok = max(1, self.tokens)
        pick = lambda q: waits[min(len(waits) - 1, int(q * len(waits)))] if waits else None  # noqa: E731
        return {"tokens": self.tokens, "h2d_bytes": self.h2d_bytes, "d2h_bytes": self.d2h_bytes,
                "pcie_bytes_per_token": (self.h2d_bytes + self.d2h_bytes) / tok,
                "h2d_bytes_per_token": self.h2d_bytes / tok, "hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / max(1, self.hits + self.misses), "exact_gathers": self.exact_gathers,
                "stale": self.stale, "host_wait_ms_total": 1e3 * self.host_wait_s,
                "gather_ms_total": 1e3 * self.gather_s, "gpu_wait_ms_total": sum(waits) if waits else 0.0,
                "gpu_wait_ms_median": pick(0.5), "gpu_wait_ms_p99": pick(0.99), "gpu_waits": len(waits)}


# ------------------------------------------------------------------ B1
class HostGather:
    """B1 row provider: exact rows gathered synchronously from the host table."""

    def __init__(self, host: HostTables, device="cpu", stats: ServeStats | None = None):
        self.host, self.device = host, norm_device(device)
        self.cuda = cuda_device(self.device)
        self.stats = stats or ServeStats()
        self.first = min(host)
        self._host_buf: dict = {}
        self._dev_buf: dict = {}

    def _buffers(self, pos: int, rows: int, dim: int, dtype) -> tuple[Tensor, Tensor]:
        hb = self._host_buf.get(pos)
        if hb is None or hb.shape[0] < rows:
            hb = self._host_buf[pos] = host_empty((rows, dim), dtype)
            if self.cuda:
                self._dev_buf[pos] = torch.empty(rows, dim, dtype=dtype, device=self.device)
        db = self._dev_buf[pos] if self.cuda else None
        return hb[:rows], (db[:rows] if db is not None else None)

    def context(self, pos: int, state, codes_new: Tensor) -> Tensor:
        """History (min(L, H) codes) + new codes on the GPU: [B, h + n, R]."""
        L = state.length
        h = min(L, self.host[pos].history)
        return torch.cat((state.codes[pos][:, L - h:L], codes_new.to(state.codes[pos].dtype)), 1)

    def gather_rows(self, pos: int, ctx: Tensor, n: int) -> Tensor:
        """Exact route tokens [B, n, R, len*D] on ``self.device`` for codes ``ctx`` (any device)."""
        lt = self.host[pos]
        ctx_h = ctx.to("cpu")  # synchronous D2H
        self.stats.add(d2h_bytes=ctx.numel() * ctx.element_size() if ctx.device.type != "cpu" else 0)
        idx = lt.exact_index(ctx_h, n)
        b, _, r, k = idx.shape
        hb, db = self._buffers(pos, idx.numel(), lt.dim, lt.weight.dtype)
        t0 = time.perf_counter()
        lt.gather(idx, hb)
        self.stats.add(gather_s=time.perf_counter() - t0, exact_gathers=1, h2d_bytes=hb.numel() * hb.element_size())
        if db is None:
            return hb.clone().view(b, n, r, k * lt.dim)
        db.copy_(hb, non_blocking=True)  # buffers are reused next step: the next D2H on this stream orders it
        return db.view(b, n, r, k * lt.dim)

    def __call__(self, pos: int, state, codes_new: Tensor) -> Tensor:
        if pos == self.first:
            self.stats.add(tokens=codes_new.shape[0] * codes_new.shape[1])
        return self.gather_rows(pos, self.context(pos, state, codes_new), codes_new.shape[1])

    def close(self) -> None:
        pass


# ------------------------------------------------------------------ B4 staging
class _Slot:
    __slots__ = ("window", "host_cand", "dev_cand", "copy_event", "future", "position", "key", "consumed")

    def __init__(self):
        self.future: Future | None = None
        self.position, self.key, self.consumed = -1, None, False


class CandidateStager:
    """Double-buffered 16-candidate staging for a set of LayerTables (one slot per
    position parity per layer), filled by one background thread.

    Per layer and slot: ``window`` pinned uint8 [B*(H+1)*R] (codes window, written by
    the compute stream), ``host_cand`` pinned [B*R*len*16, D], ``dev_cand`` GPU copy,
    ``copy_event``.  Shapes are fixed after ``ensure(batch)`` (CUDA-graph friendly)."""

    def __init__(self, tables: dict, device="cpu", *, threaded: bool = True, stats: ServeStats | None = None):
        self.tables, self.device = tables, norm_device(device)
        self.cuda = cuda_device(self.device)
        self.stats = stats or ServeStats()
        self.batch = 0
        self.slots: dict[int, list[_Slot]] = {}
        self._last: dict[int, tuple[int, Tensor, _Slot]] = {}
        self.side = torch.cuda.Stream(self.device) if self.cuda else None
        init = (lambda: torch.cuda.set_device(self.device)) if self.cuda else None
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="prefetch", initializer=init) if threaded else None

    def ensure(self, batch: int) -> None:
        if batch == self.batch:
            return
        self.drain()
        self.batch = batch
        for pos, lt in self.tables.items():
            rows = batch * lt.routes * len(lt.ngrams) * lt.alphabet
            slots = []
            for _ in range(2):
                s = _Slot()
                s.window = host_empty(batch * (lt.history + 1) * lt.routes, torch.uint8)
                s.host_cand = host_empty((rows, lt.dim), lt.weight.dtype)
                s.dev_cand = torch.empty(rows, lt.dim, dtype=lt.weight.dtype, device=self.device) \
                    if self.cuda else s.host_cand
                s.copy_event = torch.cuda.Event() if self.cuda else None
                slots.append(s)
            self.slots[pos] = slots
        self._last.clear()

    def slot(self, pos: int, position: int) -> _Slot:
        return self.slots[pos][position % 2]

    def window(self, pos: int, position: int, h: int) -> Tensor:
        """Pinned codes window [B, h, R] for the job of ``position`` (codes ending at position-1)."""
        lt = self.tables[pos]
        return self.slot(pos, position).window[: self.batch * h * lt.routes].view(self.batch, h, lt.routes)

    def cand_view(self, pos: int, position: int) -> Tensor:
        lt = self.tables[pos]
        return self.slot(pos, position).dev_cand.view(self.batch, lt.routes, len(lt.ngrams), lt.alphabet, lt.dim)

    def submit(self, pos: int, position: int, h: int, key=None, codes_event=None) -> None:
        """Stage candidates for ``position`` from the codes already in ``window(pos, position, h)``."""
        s = self.slot(pos, position)
        s.position, s.key, s.consumed = position, key, False
        if self.pool is None:
            s.future = Future()
            s.future.set_result(self._job(pos, s, position, h, codes_event))
        else:
            s.future = self.pool.submit(self._job, pos, s, position, h, codes_event)

    def acquire(self, pos: int, position: int) -> _Slot:
        """Make the slot of ``position`` free for writing (its previous job finished)."""
        s = self.slot(pos, position)
        if s.future is not None:
            s.future.result()
        return s

    def _job(self, pos: int, s: _Slot, position: int, h: int, codes_event) -> None:
        if codes_event is not None:
            codes_event.synchronize()
        lt = self.tables[pos]
        win = s.window[: self.batch * h * lt.routes].view(self.batch, h, lt.routes).to(torch.long)
        self._check_stale(pos, position, win)
        hist = win[:, h - min(h, lt.history):]
        idx = lt.candidate_index(hist, position)
        t0 = time.perf_counter()
        lt.gather(idx, s.host_cand)
        if self.cuda:
            with torch.cuda.stream(self.side):
                s.dev_cand.copy_(s.host_cand, non_blocking=True)
                s.copy_event.record(self.side)
        self.stats.add(gather_s=time.perf_counter() - t0, h2d_bytes=s.host_cand.numel() * s.host_cand.element_size())
        self._last[pos] = (position, hist.clone(), s)

    def _check_stale(self, pos: int, position: int, win: Tensor) -> None:
        last = self._last.get(pos)
        if last is None or last[0] + 1 != position or not last[2].consumed:
            return
        p0, hist, _ = last
        a0, w0 = p0 - hist.shape[1], position - win.shape[1]
        lo = max(a0, w0)
        if p0 > lo and not torch.equal(hist[:, lo - a0:p0 - a0], win[:, lo - w0:p0 - w0]):
            self.stats.add(stale=1)

    def ready(self, pos: int, position: int, key=None) -> bool:
        slots = self.slots.get(pos)
        if not slots:
            return False
        s = slots[position % 2]
        return s.future is not None and s.position == position and s.key == key

    def wait(self, pos: int, position: int) -> Tensor:
        """Join the job, make the current stream wait on its copy event (timed); returns the
        candidate view [B, R, len, 16, D] on the device."""
        s = self.slot(pos, position)
        t0 = time.perf_counter()
        s.future.result()
        self.stats.add(host_wait_s=time.perf_counter() - t0)
        if self.cuda:
            cur = torch.cuda.current_stream(self.device)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(cur)
            cur.wait_event(s.copy_event)
            e1.record(cur)
            self.stats._waits.append((e0, e1))
        s.consumed = True
        return self.cand_view(pos, position)

    def select(self, pos: int, position: int, codes: Tensor) -> Tensor:
        """Rows [B, R, len*D] for the actual codes [B, R] from the staged candidates (GPU)."""
        lt = self.tables[pos]
        cv = self.cand_view(pos, position)
        return select_candidates({o: cv[:, :, k] for k, o in enumerate(lt.ngrams)}, codes, lt.ngrams)

    def drain(self) -> None:
        for slots in self.slots.values():
            for s in slots:
                if s.future is not None:
                    s.future.result()

    def invalidate(self) -> None:
        self.drain()
        for slots in self.slots.values():
            for s in slots:
                s.position, s.key = -1, None
        self._last.clear()

    def close(self) -> None:
        self.drain()
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None


class CandidatePrefetcher(HostGather):
    """B4 row provider (transfer mode i).  See the module docstring.

    Single-token steps whose candidates were staged by the previous call for this
    layer (same state object, batch and position) are hits; anything else (prefill,
    multi-token blocks, a new state, rollback) uses the exact synchronous gather.
    Every call then issues the prefetch for the position after its last one."""

    def __init__(self, host: HostTables, device="cpu", *, threaded: bool = True, stats: ServeStats | None = None):
        super().__init__(host, device, stats)
        self.stager = CandidateStager(host, device, threaded=threaded, stats=self.stats)

    def __call__(self, pos: int, state, codes_new: Tensor) -> Tensor:
        L, (b, n, _) = state.length, codes_new.shape
        if pos == self.first:
            self.stats.add(tokens=b * n)
        self.stager.ensure(b)
        key = (id(state), b)
        if n == 1 and self.stager.ready(pos, L, key):
            self.stager.wait(pos, L)
            rows = self.stager.select(pos, L, codes_new[:, 0])[:, None]
            self.stats.add(hits=1)
        else:
            if n == 1:
                self.stats.add(misses=1)
            rows = self.gather_rows(pos, self.context(pos, state, codes_new), n)
        self.issue(pos, state, codes_new, key)
        return rows

    def issue(self, pos: int, state, codes_new: Tensor, key=None) -> None:
        """Prefetch candidates for position L + n (codes window: last min(L+n, H+1) codes)."""
        lt = self.host[pos]
        L, n = state.length, codes_new.shape[1]
        p = L + n
        h = min(p, lt.history + 1)
        self.stager.acquire(pos, p)
        hist = state.codes[pos][:, max(0, L - h):L]
        win = torch.cat((hist, codes_new.to(hist.dtype)), 1)[:, -h:]
        dst = self.stager.window(pos, p, h)
        ev = None
        if self.cuda:
            dst.copy_(win, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.device))
            self.stats.add(d2h_bytes=win.numel())
        else:
            dst.copy_(win)
        self.stager.submit(pos, p, h, key, ev)

    def invalidate(self) -> None:
        self.stager.invalidate()

    def close(self) -> None:
        self.stager.close()


class ZeroCopyGather:
    """Transfer mode (ii): a kernel reading rows straight from mapped (cudaHostRegisterMapped)
    host memory.  Not implemented: PyTorch cannot wrap a mapped host pointer as a device
    tensor without a custom extension (cudaHostGetDevicePointer + a gather kernel)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("zero-copy transfer mode (ii) is a stretch goal and is not implemented; "
                                  "use CandidatePrefetcher (mode i)")


# ------------------------------------------------------------------ speculative decoding
class StagedBlock:
    """Rows staged for one verify block (positions start .. start+n-1) of every memory layer.

    ``rows(pos)`` [B, n, R, len*D]; ``bonus_rows(pos, j, codes)`` selects the rows of the
    verifier's token at position start+j+1 (j drafts accepted after the block's first
    token, so the history is the block's codes through start+j) for its actual codes."""

    def __init__(self, stager: "BlockStager", start: int, n: int, batch: int, bonus: str | None):
        self.stager, self.start, self.n, self.batch, self.bonus = stager, start, n, batch, bonus
        self.future: Future | None = None
        self.event = None
        self.layout: dict[int, tuple[int, int, int]] = {}   # pos -> (rows_off, cand_off, n_bonus)
        self.codes_dev: dict[int, Tensor] = {}
        self.mismatch: Tensor | None = None
        self.buf: Tensor | None = None
        self._waited = False

    def wait(self) -> None:
        if self._waited:
            return
        t0 = time.perf_counter()
        self.future.result()
        st = self.stager
        st.stats.add(host_wait_s=time.perf_counter() - t0)
        if st.cuda:
            cur = torch.cuda.current_stream(st.device)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(cur)
            cur.wait_event(self.event)
            e1.record(cur)
            st.stats._waits.append((e0, e1))
        self._waited = True

    def rows(self, pos: int) -> Tensor:
        self.wait()
        lt = self.stager.host[pos]
        off, _, _ = self.layout[pos]
        k = len(lt.ngrams)
        size = self.batch * self.n * lt.routes * k
        return self.buf[off: off + size].view(self.batch, self.n, lt.routes, k * lt.dim)

    def bonus_rows(self, pos: int, j: int, codes: Tensor) -> Tensor | None:
        """Rows [B, R, len*D] at position start+j+1 for its codes [B, R] (None if not staged)."""
        self.wait()
        lt = self.stager.host[pos]
        _, coff, nb = self.layout[pos]
        jj = j - (self.n - nb)  # bonus "last" stages only j = n-1
        if not 0 <= jj < nb:
            return None
        k = len(lt.ngrams)
        size = self.batch * nb * lt.routes * k * lt.alphabet
        cv = self.buf[coff: coff + size].view(self.batch, nb, lt.routes, k, lt.alphabet, lt.dim)[:, jj]
        return select_candidates({o: cv[:, :, i] for i, o in enumerate(lt.ngrams)}, codes, lt.ngrams)

    def codes_match(self) -> bool:
        """True if the verifier's codes equalled the staged (drafter) codes on every
        position served so far (syncs).  False -> the verify output is not exact; redo
        the block with ``stager.fallback``."""
        return self.mismatch is None or not bool(self.mismatch.item())


class BlockStager:
    """Exact row staging for speculative verification (+ candidates for the bonus token).

    ``stage`` runs on a background thread (index_select into a pinned slot, one H2D
    copy on a side stream, event); two slots are cycled, so a block stays valid until
    two more blocks have been staged.  ``provider`` returns a ``row_provider`` for
    ``decode_step`` that serves a staged block, optionally preceded by the bonus
    position from the previous block's candidates; any mismatch of positions falls
    back to the exact synchronous gather (``self.fallback``, counted as a miss)."""

    def __init__(self, host: HostTables, device="cpu", *, threaded: bool = True, bonus: str | None = "all",
                 stats: ServeStats | None = None):
        if bonus not in ("all", "last", None):
            raise ValueError("bonus must be 'all', 'last' or None")
        self.host, self.device, self.bonus = host, norm_device(device), bonus
        self.cuda = cuda_device(self.device)
        self.stats = stats or ServeStats()
        self.fallback = HostGather(host, device, self.stats)
        self.side = torch.cuda.Stream(self.device) if self.cuda else None
        init = (lambda: torch.cuda.set_device(self.device)) if self.cuda else None
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="blockstage", initializer=init) if threaded else None
        self._slots: list[list] = [[None, None, None], [None, None, None]]  # host, dev, last block
        self._next = 0

    def _slot_buffers(self, rows: int, dim: int, dtype) -> tuple[Tensor, Tensor, int]:
        k = self._next
        self._next ^= 1
        hb, db, prev = self._slots[k]
        if prev is not None and prev.future is not None:
            prev.future.result()
            if self.cuda and prev.event is not None:
                prev.event.synchronize()  # H2D from this host slot finished
        if hb is None or hb.shape[0] < rows or hb.shape[1] != dim or hb.dtype != dtype:
            hb = host_empty((rows, dim), dtype)
            db = torch.empty(rows, dim, dtype=dtype, device=self.device) if self.cuda else hb
            self._slots[k][:2] = [hb, db]
        return hb, db, k

    def stage(self, start: int, codes: dict, history: dict, *, bonus: str | None | bool = True) -> StagedBlock:
        """Stage rows for positions start .. start+n-1.

        codes: {pos: uint8 [B, n, R]} drafted tokens' codes (CPU); history:
        {pos: [B, min(start, H), R]} the committed codes before ``start``.  ``bonus``
        (default: the stager's setting) adds candidate blocks for position start+j+1,
        j = 0..n-1 ("all") or j = n-1 only ("last")."""
        bonus = self.bonus if bonus is True else (None if bonus is False else bonus)
        any_pos = next(iter(codes))
        b, n, _ = codes[any_pos].shape
        dims = {lt.dim for lt in self.host.values()}
        dtypes = {lt.weight.dtype for lt in self.host.values()}
        if len(dims) != 1 or len(dtypes) != 1:
            raise ValueError("all memory layers must share mem_dim and dtype")
        blk = StagedBlock(self, start, n, b, bonus)
        nb = {"all": n, "last": 1, None: 0}[bonus]
        total = 0
        for pos, lt in self.host.items():
            k = len(lt.ngrams)
            rows_n = b * n * lt.routes * k
            blk.layout[pos] = (total, total + rows_n, nb)
            total += rows_n + b * nb * lt.routes * k * lt.alphabet
        hb, db, k = self._slot_buffers(total, dims.pop(), dtypes.pop())
        self._slots[k][2] = blk
        blk.buf = db
        all_codes = []
        for pos, lt in self.host.items():
            hc = history.get(pos)
            h = min(start, lt.history)
            hc = codes[pos].new_zeros(b, 0, lt.routes) if hc is None and h == 0 else hc
            if hc is None or hc.shape[1] < h:
                raise ValueError(f"layer {pos}: need {h} history codes before position {start}")
            hc = hc[:, hc.shape[1] - h:]
            all_codes.append((pos, lt, torch.cat((hc.to(torch.uint8), codes[pos].to(torch.uint8)), 1), h))
            if self.cuda:
                blk.codes_dev[pos] = codes[pos].to(self.device, torch.uint8, non_blocking=False)
            else:
                blk.codes_dev[pos] = codes[pos].to(torch.uint8).clone()
        blk.mismatch = torch.zeros((), dtype=torch.bool, device=self.device)
        free = None
        if self.cuda:  # the compute stream's earlier reads of this slot (block staged two calls ago)
            free = torch.cuda.Event()
            free.record(torch.cuda.current_stream(self.device))

        def job():
            t0 = time.perf_counter()
            for pos, lt, ctx, h in all_codes:
                roff, coff, nbk = blk.layout[pos]
                idx = lt.exact_index(ctx, n)
                lt.gather(idx, hb[roff: roff + idx.numel()])
                if nbk:
                    c = ctx.to(torch.long)
                    cidx = torch.stack([lt.candidate_index(c[:, :h + j + 1][:, -min(start + j + 1, lt.history):],
                                                           start + j + 1)
                                        for j in range(n - nbk, n)], 1)  # [B, nb, R, len, 16]
                    lt.gather(cidx, hb[coff: coff + cidx.numel()])
            if self.cuda:
                with torch.cuda.stream(self.side):
                    self.side.wait_event(free)
                    db[:total].copy_(hb[:total], non_blocking=True)
                    blk.event = torch.cuda.Event()
                    blk.event.record(self.side)
            self.stats.add(gather_s=time.perf_counter() - t0, h2d_bytes=total * hb.shape[1] * hb.element_size())

        if self.pool is None:
            blk.future = Future()
            job()
            blk.future.set_result(None)
        else:
            blk.future = self.pool.submit(job)
        return blk

    def provider(self, blk: StagedBlock, head: tuple[StagedBlock, int] | None = None):
        """``row_provider`` serving ``blk`` (positions blk.start ..), or, with ``head=(prev, j)``,
        the bonus position prev.start+j+1 == blk.start-1 first, then ``blk``."""
        first = min(self.host)

        def fn(pos: int, state, codes_new: Tensor) -> Tensor:
            L, (b, n, _) = state.length, codes_new.shape
            if pos == first:
                self.stats.add(tokens=b * n)
            off = 0 if head is None else 1
            ok = L + off == blk.start and n == blk.n + off and b == blk.batch
            if ok and head is not None:
                prev, j = head
                ok = prev.start + j + 1 == L and prev.batch == b
            parts = []
            if ok and head is not None:
                br = head[0].bonus_rows(pos, head[1], codes_new[:, 0])
                ok = br is not None
                parts.append(br)
            if not ok:
                self.stats.add(misses=1)
                return self.fallback.gather_rows(pos, self.fallback.context(pos, state, codes_new), n)
            self.stats.add(hits=1)
            blk.mismatch |= (codes_new[:, off:].to(torch.uint8) != blk.codes_dev[pos]).any()
            parts = [p[:, None] for p in parts] + [blk.rows(pos)]
            return parts[0] if len(parts) == 1 else torch.cat(parts, 1)

        return fn

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None
