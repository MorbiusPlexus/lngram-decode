"""Scaled host tables (M4.T3): B1 vs B4 gather + transfer latency per step, in isolation.

    python -m serving.scaled_tables --codes dumps/w3_sdb_1b/codes_layer1.npy --server gpu0 \
        --sizes_gb 0.1,1,10 --max_frac_ram 0.5 --include_max --batches 1,8,32 --gap_ms 0,5
    python -m serving.scaled_tables --synthetic --server laptop --sizes_gb 0.01 --device cpu --steps 32

Random-weight tables (fp16, serving layout, page-locked) of each target size are
grown (a) ``routes``: orders (2, 3), more routes; (b) ``order4``: orders (2, 3, 4),
routes chosen to reach the same size (the 4-gram makes each route 16x larger).
Sizes above ``--max_frac_ram`` of host RAM are skipped; ``--include_max`` adds one
table at that limit.  The code stream is real: ``--codes`` is a uint8 [T, R0]
array of one memory layer (``analysis/dump_codes.py``); route r reads column
r mod R0 (replicated streams), batch row b starts at offset b*T/B, and the 4-gram
uses the real code of the extra predecessor c[t-3] from the same stream.

No model: each step starts from the step's codes on the GPU.
  B1  D2H of the codes window -> exact index_select into pinned -> H2D -> sync
      (all on the critical path).
  B4  wait on the copy event of step t's candidates -> select by c[t] on the GPU;
      then D2H of the codes window, event, and the staging thread gathers + copies the
      16-row blocks for step t+1 while the GPU "computes" for ``gap_ms`` (a spin kernel
      standing in for the rest of the decode step).  Exposed latency = step - gap.
JSON: results/scaled_<server>.json.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from serving.prefetch import CandidateStager, LayerTable, ServeStats, cuda_device

GB = 1e9


def total_ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        pass
    if sys.platform == "win32":
        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MemStatus()
        st.dwLength = ctypes.sizeof(MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullTotalPhys)
    return None


GROWTH = {"routes": (2, 3), "order4": (2, 3, 4)}


def route_bytes(ngrams, dim: int, elem: int, bits: int = 4) -> int:
    return sum((1 << bits) ** o for o in ngrams) * dim * elem


def routes_for_bytes(target: float, ngrams, dim: int, elem: int, bits: int = 4) -> int:
    return max(1, int(round(target / route_bytes(ngrams, dim, elem, bits))))


def plan_sizes(sizes_gb, max_frac_ram: float, include_max: bool, ram: int | None) -> list[float]:
    """Target sizes in bytes, capped at max_frac_ram * RAM."""
    cap = None if ram is None else max_frac_ram * ram
    out = [s * GB for s in sizes_gb if cap is None or s * GB <= cap]
    if include_max and cap is not None and cap not in out:
        out.append(cap)
    return out


def route_codes(stream: np.ndarray, routes: int, batch: int, length: int) -> torch.Tensor:
    """[batch, length, routes] uint8 from a real [T, R0] stream: route r <- column r % R0,
    batch row b starts at b * T // batch (wrapping)."""
    t_total, r0 = stream.shape
    cols = np.arange(routes) % r0
    rows = (np.arange(batch)[:, None] * (t_total // batch) + np.arange(length)[None]) % t_total
    return torch.from_numpy(np.ascontiguousarray(stream[rows][:, :, cols]).astype(np.uint8))


def load_stream(path: str | None, r0: int, length: int, seed: int) -> np.ndarray:
    if path:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"{path}: expected [T, R] codes, got {arr.shape}")
        return np.asarray(arr, dtype=np.uint8)
    return np.random.default_rng(seed).integers(0, 16, (length, r0), dtype=np.uint8)


def _spin(gap_ms: float, dev: torch.device, cycles_per_ms: float | None) -> None:
    if gap_ms <= 0:
        return
    if dev.type == "cuda":
        torch.cuda._sleep(int(gap_ms * cycles_per_ms))
    else:
        time.sleep(gap_ms / 1e3)


def calibrate_spin(dev: torch.device) -> float | None:
    if dev.type != "cuda":
        return None
    torch.cuda._sleep(1000)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    torch.cuda._sleep(10_000_000)
    e1.record()
    e1.synchronize()
    return 10_000_000 / e0.elapsed_time(e1)


def _q(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def _summ(ms: list[float]) -> dict:
    return {"median_ms": statistics.median(ms), "p99_ms": _q(ms, 0.99), "mean_ms": statistics.fmean(ms)}


def sync(dev) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


@torch.no_grad()
def bench_b1(lt: LayerTable, codes: torch.Tensor, dev: torch.device, warmup: int, steps: int) -> dict:
    """codes [B, S, R] on dev; steps t = H .. ; exact rows of step t."""
    H = lt.history
    b = codes.shape[0]
    idx_shape = (b, 1, lt.routes, len(lt.ngrams))
    host = torch.empty(int(np.prod(idx_shape)), lt.dim, dtype=lt.weight.dtype,
                       pin_memory=dev.type == "cuda")
    devbuf = torch.empty_like(host, device=dev) if dev.type == "cuda" else None
    lat, gat = [], []
    for i in range(warmup + steps):
        t = H + i
        sync(dev)
        t0 = time.perf_counter()
        ctx = codes[:, t - H:t + 1].to("cpu")
        idx = lt.exact_index(ctx, 1)
        g0 = time.perf_counter()
        lt.gather(idx, host)
        g1 = time.perf_counter()
        if devbuf is not None:
            devbuf.copy_(host, non_blocking=True)
        sync(dev)
        if i >= warmup:
            lat.append(1e3 * (time.perf_counter() - t0))
            gat.append(1e3 * (g1 - g0))
    return {**_summ(lat), "gather_ms_median": statistics.median(gat),
            "h2d_bytes_per_step": host.numel() * host.element_size(),
            "d2h_bytes_per_step": b * (H + 1) * lt.routes}


@torch.no_grad()
def bench_b4(lt: LayerTable, codes: torch.Tensor, dev: torch.device, warmup: int, steps: int, gap_ms: float,
             cycles_per_ms: float | None, threaded: bool = True) -> dict:
    H = lt.history
    b = codes.shape[0]
    stats = ServeStats()
    stg = CandidateStager({0: lt}, dev, threaded=threaded, stats=stats)
    stg.ensure(b)
    cuda = dev.type == "cuda"

    def issue(t: int) -> None:  # stage candidates for position t + 1 (codes window ends at t)
        stg.acquire(0, t + 1)
        dst = stg.window(0, t + 1, H + 1)
        ev = None
        if cuda:
            dst.copy_(codes[:, t - H:t + 1], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
        else:
            dst.copy_(codes[:, t - H:t + 1])
        stg.submit(0, t + 1, H + 1, None, ev)

    t_first = H + 1
    issue(t_first - 1)
    lat, exposed = [], []
    try:
        for i in range(warmup + steps):
            t = t_first + i
            if i == warmup:
                stg.drain()  # count exactly the jobs issued by timed steps
                stats.reset()
            sync(dev)
            t0 = time.perf_counter()
            stg.wait(0, t)
            rows = stg.select(0, t, codes[:, t])
            issue(t)
            _spin(gap_ms, dev, cycles_per_ms)
            rows.sum()  # consume
            sync(dev)
            if i >= warmup:
                dt = 1e3 * (time.perf_counter() - t0)
                lat.append(dt)
                exposed.append(max(0.0, dt - gap_ms))
        stg.drain()
    finally:
        stg.close()
    s = stats.summary()
    waits = stats.gpu_waits_ms()
    return {"gap_ms": gap_ms, "step": _summ(lat), "exposed": _summ(exposed),
            "gpu_wait_ms_median": statistics.median(waits) if waits else None,
            "gpu_wait_ms_p99": _q(waits, 0.99) if waits else None,
            "host_wait_ms_per_step": s["host_wait_ms_total"] / steps,
            "gather_ms_per_step": s["gather_ms_total"] / steps,
            "h2d_bytes_per_step": s["h2d_bytes"] / steps}


def run(stream: np.ndarray, targets: list[float], growths, batches, dim: int, dtype, dev: torch.device,
        warmup: int, steps: int, gaps, pin: bool = True, threaded: bool = True, log=print) -> list[dict]:
    cyc = calibrate_spin(dev)
    elem = torch.empty((), dtype=dtype).element_size()
    rows = []
    for growth in growths:
        ngrams = GROWTH[growth]
        for target in targets:
            routes = routes_for_bytes(target, ngrams, dim, elem)
            t0 = time.perf_counter()
            lt = LayerTable.random(routes, ngrams, 4, dim, dtype, pin=pin and dev.type == "cuda")
            build_s = time.perf_counter() - t0
            for b in batches:
                codes = route_codes(stream, routes, b, warmup + steps + lt.history + 2).to(dev)
                r = {"growth": growth, "target_gb": target / GB, "routes": routes, "ngrams": list(ngrams),
                     "table_bytes": lt.nbytes, "batch": b, "build_s": build_s,
                     "b1": bench_b1(lt, codes, dev, warmup, steps),
                     "b4": [bench_b4(lt, codes, dev, warmup, steps, g, cyc, threaded) for g in gaps]}
                log(json.dumps({"growth": growth, "gb": round(lt.nbytes / GB, 3), "batch": b,
                                "b1_ms": round(r["b1"]["median_ms"], 4),
                                "b4_exposed_ms": [round(x["exposed"]["median_ms"], 4) for x in r["b4"]]}))
                rows.append(r)
            del lt
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--codes", help="uint8 [T, R] codes of one memory layer (.npy)")
    src.add_argument("--synthetic", action="store_true", help="random codes (smoke test only)")
    ap.add_argument("--server", required=True)
    ap.add_argument("--sizes_gb", default="0.1,1,10")
    ap.add_argument("--max_frac_ram", type=float, default=0.5)
    ap.add_argument("--include_max", action="store_true")
    ap.add_argument("--growth", default="routes,order4")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--gap_ms", default="0,5")
    ap.add_argument("--mem_dim", type=int, default=128)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--warmup", type=int, default=32)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_thread", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    dev = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    stream = load_stream(args.codes, 16, 4096, args.seed)
    ram = total_ram_bytes()
    targets = plan_sizes([float(x) for x in args.sizes_gb.split(",")], args.max_frac_ram, args.include_max, ram)
    rows = run(stream, targets, args.growth.split(","), [int(x) for x in args.batches.split(",")], args.mem_dim,
               dtype, dev, args.warmup, args.steps, [float(x) for x in args.gap_ms.split(",")],
               pin=cuda_device(dev), threaded=not args.no_thread)
    out = {"server": args.server, "codes": args.codes, "stream_shape": list(stream.shape), "ram_bytes": ram,
           "max_frac_ram": args.max_frac_ram, "mem_dim": args.mem_dim, "dtype": args.dtype, "device": args.device,
           "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None, "torch": torch.__version__,
           "warmup": args.warmup, "steps": args.steps, "rows": rows}
    path = Path(args.out_dir) / f"scaled_{args.server}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
