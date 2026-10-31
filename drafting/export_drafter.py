"""Export a drafter to a flat binary bundle for the CPU drafter (M5.T1).

    python -m drafting.export_drafter --ckpt runs/w3_sdb_1b --sketch sketches/w3_sdb_1b.npz --out bundles/sdb.bin
    python -m drafting.export_drafter --ckpt runs/w3_sb_1b --posthoc runs/posthoc_sb/drafter.pt \
        --sketch sketches/w3_sb_1b.npz --out bundles/sb_posthoc.bin
    bundle = load_bundle("bundles/sdb.bin")            # header dict + numpy arrays (memory-mapped)

Sources: an SD checkpoint (SD-B ``embed2``, SD-C ``embed2_ssm``, SD-A ``token_router``),
or a target checkpoint + a ``PosthocDrafter`` checkpoint (its own router / SSM /
readout over the target's frozen table, memory norm and KV projection).

File layout (all little-endian)::

    offset 0   8 bytes   magic b"MRBDRFT1"
    offset 8   uint64    header length H
    offset 16  H bytes   header JSON (UTF-8)
    ...        arrays, each starting at a 64-byte aligned absolute file offset

Header keys: ``version``, ``config`` (see ``bundle_config``), ``arrays``
{name: {dtype, shape, offset, nbytes}}, ``source`` (provenance).

Arrays (fp32 unless noted; ``F`` = router feature dim, ``N`` = SSM state dim,
``Dt`` = len(ngrams) * mem_dim, ``KH`` = kv_heads * head_dim, ``I`` = q_heads * head_dim):

    embedding            [V, d]                tied token embedding (draft query input and unembedding)
    table_<n>            [R * 16^n, mem_dim]   fp16 (default) or fp32; v2 row layout (oldest code least significant)
    mem_norm             [Dt]                  v2 readout_memory_norm weight (RMSNorm)
    kv_proj              [2 * KH, Dt]          v2 readout_kv_proj weight (keys first, then values)
    router               [R * bits, F]         router weight; F = 2d (+ N for embed2_ssm)
    router_prev_norm     [d]   router_cur_norm [d]   router_start [d]   (TokenRouterFeatures)
    ssm_norm [d], ssm_w_in [N, d], ssm_decay [N] (= sigmoid(theta), fp32)      -- if the drafter has an SSM
    q_norm [d], w_q [I, d], w_sq [I, N] (if the query reads the SSM), w_o [d, I], out_norm [d]
    sketch_ids_<n>       [R * 16^n, K]         uint16, sketch_counts_<n> uint32 (drafting/sketches.py)
    fallback             [M]                   int32: tokens by total sketch count (shortlist padding)
"""

from __future__ import annotations

import argparse
import json
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

MAGIC = b"MRBDRFT1"
VERSION = 1
ALIGN = 64


@dataclass
class Bundle:
    config: dict
    arrays: dict          # name -> numpy array (memory-mapped when loaded from disk)
    source: dict

    def tensors(self, names=None) -> dict:
        """torch views (fp32 / int32 conversions for C++: uint16 ids and uint32 counts -> int32)."""
        out = {}
        for k, a in self.arrays.items():
            if names is not None and k not in names:
                continue
            if a.dtype == np.uint16 or a.dtype == np.uint32:
                a = np.minimum(a.astype(np.int64), np.iinfo(np.int32).max).astype(np.int32)
            with warnings.catch_warnings():  # read-only memory maps are shared, never written
                warnings.simplefilter("ignore", UserWarning)
                out[k] = torch.from_numpy(np.ascontiguousarray(a))
        return out


# ------------------------------------------------------------------ writing / reading
def write_bundle(path: str | Path, config: dict, arrays: dict, source: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta, offset = {}, 0
    arrs = {k: np.ascontiguousarray(v) for k, v in arrays.items()}
    for k, a in arrs.items():
        meta[k] = {"dtype": a.dtype.str, "shape": list(a.shape), "offset": 0, "nbytes": int(a.nbytes)}
    # Header size depends on the offsets; iterate until stable.
    while True:
        header = json.dumps({"version": VERSION, "config": config, "arrays": meta, "source": source or {}},
                            sort_keys=True).encode("utf-8")
        offset = _align(16 + len(header) + 256)  # slack so that offsets with more digits still fit
        new = {}
        for k, a in arrs.items():
            new[k] = dict(meta[k], offset=offset)
            offset = _align(offset + a.nbytes)
        if new == meta:
            break
        meta = new
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for k, a in arrs.items():
            f.write(b"\0" * (meta[k]["offset"] - f.tell()))
            a.astype(a.dtype.newbyteorder("<"), copy=False).tofile(f)
    return path


def _align(x: int) -> int:
    return (x + ALIGN - 1) // ALIGN * ALIGN


def read_header(path: str | Path) -> dict:
    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError(f"{path}: not a drafter bundle")
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n).decode("utf-8"))


def load_bundle(path: str | Path, mmap: bool = True) -> Bundle:
    head = read_header(path)
    if head["version"] != VERSION:
        raise ValueError(f"bundle version {head['version']} != {VERSION}")
    arrays = {}
    for k, m in head["arrays"].items():
        dt = np.dtype(m["dtype"])
        shape = tuple(m["shape"])
        if mmap and m["nbytes"]:
            arrays[k] = np.memmap(path, dtype=dt, mode="r", offset=m["offset"], shape=shape)
        else:
            with open(path, "rb") as f:
                f.seek(m["offset"])
                arrays[k] = np.frombuffer(f.read(m["nbytes"]), dtype=dt).reshape(shape).copy()
    return Bundle(head["config"], arrays, head.get("source", {}))


# ------------------------------------------------------------------ collecting parameters
def _np(t: torch.Tensor, dtype=np.float32) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(dtype)


def bundle_from_model(model, posthoc=None, sketch=None, table_dtype: str = "fp16",
                      shortlist_cap: int = 64) -> tuple[dict, dict]:
    """(config, arrays) for an SD model (``model.draft``) or a target + ``PosthocDrafter``."""
    from ffnmem.memory.lngram_v2 import RMSNorm as V2RMSNorm

    if posthoc is not None:
        layer = posthoc.layer
        feats, router_w, ssm, draft = posthoc.features, posthoc.router.weight, posthoc.ssm, posthoc.draft
        router_kind = feats.kind
        kind = "posthoc"
    else:
        if model.draft is None:
            raise ValueError("model has no draft head; pass a PosthocDrafter")
        layer = model.draft_pos
        mem = model.blocks[layer].memory
        draft = model.draft
        if mem.token_router is not None:
            feats, router_w, router_kind, kind = mem.token_router.features, mem.token_router.proj.weight, \
                "token_router", "sd_a"
        else:
            feats, router_w, router_kind = mem.router_features, mem.router.weight, mem.router_input
            kind = "sd_c" if router_kind == "embed2_ssm" else "sd_b"
        uses_ssm = router_kind == "embed2_ssm" or draft.w_sq is not None
        ssm = model.ssm if uses_ssm else None
    mem = model.blocks[layer].memory
    m = mem.module
    if not isinstance(m.readout_memory_norm, V2RMSNorm):
        raise ValueError("the CPU drafter implements the RMSNorm memory norm only")
    if ssm is None and (router_kind == "embed2_ssm" or draft.w_sq is not None):
        raise ValueError("drafter reads an SSM state but has no SSM")
    tdt = {"fp16": np.float16, "fp32": np.float32}[table_dtype]
    arrays = {"embedding": _np(model.tok_emb.weight)}
    for n in m.ngrams:
        arrays[f"table_{n}"] = _np(m.tables[f"ngram_{n}"].weight, tdt)
    arrays["mem_norm"] = _np(m.readout_memory_norm.weight)
    arrays["kv_proj"] = _np(m.readout_kv_proj.weight)
    arrays["router"] = _np(router_w)
    arrays["router_prev_norm"] = _np(feats.prev_norm.weight)
    arrays["router_cur_norm"] = _np(feats.cur_norm.weight)
    arrays["router_start"] = _np(feats.start)
    if ssm is not None:
        arrays["ssm_norm"] = _np(ssm.norm.weight)
        arrays["ssm_w_in"] = _np(ssm.w_in.weight)
        arrays["ssm_decay"] = _np(ssm.decay())
    arrays["q_norm"] = _np(draft.q_norm.weight)
    arrays["w_q"] = _np(draft.w_q.weight)
    if draft.w_sq is not None:
        arrays["w_sq"] = _np(draft.w_sq.weight)
    arrays["w_o"] = _np(draft.w_o.weight)
    arrays["out_norm"] = _np(draft.out_norm.weight)
    k = 0
    if sketch is not None:
        if sketch.routes != m.num_routes or sketch.bits != m.bits_per_route or tuple(sketch.ngrams) != tuple(m.ngrams):
            raise ValueError("sketch shape does not match the memory layer")
        k = sketch.k
        for n in sketch.ngrams:
            arrays[f"sketch_ids_{n}"] = np.asarray(sketch.ids[n], np.uint16)
            arrays[f"sketch_counts_{n}"] = np.asarray(sketch.counts[n], np.uint32)
        arrays["fallback"] = fallback_tokens(sketch, max(shortlist_cap, 1))
    eps = float(model.cfg["norm_eps"])
    config = bundle_config(
        kind=kind, router_kind=router_kind, d_model=int(model.cfg["d_model"]), vocab=int(model.cfg["vocab"]),
        routes=m.num_routes, bits=m.bits_per_route, ngrams=[int(n) for n in m.ngrams], mem_dim=m.memory_dim,
        kv_heads=m.readout_num_kv_heads, head_dim=draft.head_dim, q_heads=draft.q_heads,
        ssm_dim=0 if ssm is None else ssm.state_dim, query_ssm=draft.w_sq is not None,
        norm_eps=eps, ssm_eps=float(ssm.norm.eps) if ssm is not None else eps,
        router_eps=float(feats.prev_norm.eps), draft_eps=float(draft.q_norm.eps),
        mem_eps=float(m.readout_memory_norm.eps), softmax_scale=float(m.softmax_scale),
        sink_logit=float(m.readout_sink_logit), sketch_k=k, shortlist_cap=int(shortlist_cap) if k else 0,
        table_dtype=table_dtype, layer=int(layer))
    return config, arrays


def bundle_config(**kw) -> dict:
    kw["groups"] = kw["q_heads"] // kw["kv_heads"]
    kw["route_token_dim"] = len(kw["ngrams"]) * kw["mem_dim"]
    return kw


def fallback_tokens(sketch, n: int) -> np.ndarray:
    """Tokens by total count over the lowest order's sketch rows (desc; ties by token id)."""
    order = min(sketch.ngrams)
    ids = np.asarray(sketch.ids[order], np.int64).ravel()
    cnt = np.asarray(sketch.counts[order], np.float64).ravel()
    tot = np.bincount(ids, weights=cnt, minlength=sketch.vocab)
    nz = np.flatnonzero(tot > 0)
    nz = nz[np.lexsort((nz, -tot[nz]))]
    return nz[:n].astype(np.int32)


def export(out: str | Path, model, posthoc=None, sketch=None, table_dtype: str = "fp16",
           shortlist_cap: int = 64, source: dict | None = None) -> Path:
    config, arrays = bundle_from_model(model, posthoc, sketch, table_dtype, shortlist_cap)
    return write_bundle(out, config, arrays, source)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="SD checkpoint, or the target for --posthoc")
    ap.add_argument("--posthoc", default=None, help="PosthocDrafter checkpoint")
    ap.add_argument("--sketch", default=None, help="sketches/<run>.npz (shortlist); none -> full-vocab scoring")
    ap.add_argument("--out", required=True)
    ap.add_argument("--table_dtype", choices=("fp16", "fp32"), default="fp16")
    ap.add_argument("--shortlist_cap", type=int, default=64)
    args = ap.parse_args(argv)
    from ffnmem.infer import load_model

    from .sketches import Sketch

    model, cfg = load_model(args.ckpt)
    posthoc = None
    if args.posthoc:
        from .posthoc_model import PosthocDrafter

        posthoc = PosthocDrafter.load(args.posthoc, model).eval()
    sketch = Sketch.load(args.sketch) if args.sketch else None
    src = {"ckpt": cfg["_checkpoint"], "step": cfg["_step"], "posthoc": args.posthoc, "sketch": args.sketch}
    path = export(args.out, model, posthoc, sketch, args.table_dtype, args.shortlist_cap, src)
    head = read_header(path)
    print(json.dumps({"out": str(path), "bytes": path.stat().st_size, "config": head["config"]}, indent=2))


if __name__ == "__main__":
    main()
