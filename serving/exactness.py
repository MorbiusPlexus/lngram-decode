"""Exactness of host-RAM serving (M4.T2): B0 vs B1 vs B4 on greedy decoding.

    python -m serving.exactness --ckpt runs/w3_sdb_1b --run sdb --data_dir $DATA_DIR \
        --prompts 500 --tokens 256 --batch 16 --dtypes fp32,fp16
    python -m serving.exactness --config configs/tiny_cpu_sd.yaml --run tiny --device cpu --prompts 8 --tokens 32

The three conditions run in lockstep on the same model (B0 reads the device table;
B1 / B4 get every memory row from the host table via ``row_provider``, so the device
copy is never read).  Each condition feeds back its own greedy tokens.  Reported per
dtype and condition: fraction of sequences with identical tokens, max |logit diff|
vs B0 over positions whose inputs are still identical, and the prefetch counters
(hits / misses / stale).  Pass = identical tokens everywhere and max diff == 0.
JSON: results/exactness_<run>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ffnmem.infer import DecodeState, decode_step, load_model
from serving.prefetch import CandidatePrefetcher, HostGather, HostTables

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def load_prompts(n: int, plen: int, vocab: int, data_dir: str | None = None, seed: int = 0) -> torch.Tensor:
    """n prompts of plen tokens: consecutive windows of the val file, else seeded random ids."""
    if data_dir:
        from ffnmem.data.loader import ShardSet

        shards = ShardSet(data_dir)
        toks = shards.memmap(shards.meta["val_file"])
        if len(toks) < n * plen:
            raise ValueError(f"val file has {len(toks)} tokens < {n} x {plen}")
        arr = np.asarray(toks[: n * plen], dtype=np.int64).reshape(n, plen)
        return torch.from_numpy(np.minimum(arr, vocab - 1))
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n, plen), generator=g)


def build_model(args, dtype: torch.dtype, max_len: int):
    if args.ckpt:
        model, _ = load_model(args.ckpt, device=args.device, dtype=dtype, seq_len=max_len)
        return model
    from ffnmem.config import load_config
    from ffnmem.model import Transformer

    torch.manual_seed(args.seed)
    mcfg = dict(load_config(args.config, args.set or [])["model"])
    mcfg["seq_len"] = max(int(mcfg["seq_len"]), max_len)
    model = Transformer(mcfg)
    for b in model.blocks:  # random checkpoints: make the rows and the readout matter
        if b.memory is not None:
            for t in b.memory.module.tables.values():
                torch.nn.init.normal_(t.weight)
            torch.nn.init.normal_(b.memory.module.readout_o_proj.weight, std=0.05)
    return model.to(args.device, dtype).eval()


@torch.no_grad()
def lockstep(model, providers: dict, prompts: torch.Tensor, n_new: int) -> dict:
    """Greedy decode prompts under B0 and every provider; per-condition comparison vs B0."""
    b, p = prompts.shape
    names = ["b0", *providers]
    states = {k: DecodeState.empty(model, b, p + n_new) for k in names}
    logits = {k: decode_step(model, states[k], prompts, row_provider=providers.get(k)) for k in names}
    same = {k: torch.ones(b, dtype=torch.bool, device=prompts.device) for k in providers}  # inputs identical so far
    diff = {k: 0.0 for k in providers}
    first_div = {k: torch.full((b,), -1, dtype=torch.long) for k in providers}
    for step in range(n_new):
        nxt = {k: logits[k][:, -1].argmax(-1) for k in names}
        for k in providers:
            d = (logits[k].float() - logits["b0"].float()).abs().amax(dim=(1, 2))
            if same[k].any():
                diff[k] = max(diff[k], float(d[same[k]].max()))
            newly = same[k] & (nxt[k] != nxt["b0"])
            first_div[k][newly.cpu()] = step
            same[k] &= ~newly
        if step == n_new - 1:
            break
        for k in names:
            logits[k] = decode_step(model, states[k], nxt[k][:, None], row_provider=providers.get(k))
    return {k: {"identical": same[k].cpu(), "max_abs_logit_diff": diff[k], "first_divergence": first_div[k]}
            for k in providers}


def run_exactness(model, prompts: torch.Tensor, n_new: int, batch: int, threaded: bool = True,
                  modes=("b1", "b4")) -> dict:
    host = HostTables.from_model(model)
    dev = model.tok_emb.weight.device
    providers = {}
    if "b1" in modes:
        providers["b1"] = HostGather(host, dev)
    if "b4" in modes:
        providers["b4"] = CandidatePrefetcher(host, dev, threaded=threaded)
    acc = {k: {"identical": [], "max_abs_logit_diff": 0.0, "first_divergence": []} for k in providers}
    t0 = time.perf_counter()
    try:
        for i in range(0, len(prompts), batch):
            out = lockstep(model, providers, prompts[i:i + batch].to(dev), n_new)
            for k, r in out.items():
                acc[k]["identical"].append(r["identical"])
                acc[k]["max_abs_logit_diff"] = max(acc[k]["max_abs_logit_diff"], r["max_abs_logit_diff"])
                acc[k]["first_divergence"].append(r["first_divergence"])
    finally:
        for p in providers.values():
            p.close()
    res = {}
    for k, a in acc.items():
        ident = torch.cat(a["identical"])
        fd = torch.cat(a["first_divergence"])
        res[k] = {"sequences": len(ident), "tokens_identical_frac": float(ident.float().mean()),
                  "max_abs_logit_diff": a["max_abs_logit_diff"],
                  "first_divergence_steps": sorted(int(x) for x in fd[fd >= 0].tolist())[:50],
                  "exact": bool(ident.all()) and a["max_abs_logit_diff"] == 0.0,
                  "serve_stats": providers[k].stats.summary()}
    res["seconds"] = time.perf_counter() - t0
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="checkpoint file or run dir")
    src.add_argument("--config", help="random-weight model from a config (smoke test)")
    ap.add_argument("--set", nargs="*", help="config overrides (with --config)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--data_dir", default=None, help="prompts from the val file (else random ids)")
    ap.add_argument("--prompts", type=int, default=500)
    ap.add_argument("--prompt_len", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dtypes", default="fp32,fp16")
    ap.add_argument("--modes", default="b1,b4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_thread", action="store_true", help="stage candidates inline (debug)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    out = {"run": args.run, "ckpt": args.ckpt, "config": args.config, "device": args.device,
           "prompts": args.prompts, "prompt_len": args.prompt_len, "tokens": args.tokens, "batch": args.batch,
           "data_dir": args.data_dir, "torch": torch.__version__, "results": {}}
    for name in args.dtypes.split(","):
        model = build_model(args, DTYPES[name], args.prompt_len + args.tokens)
        prompts = load_prompts(args.prompts, args.prompt_len, model.cfg["vocab"], args.data_dir, args.seed)
        res = run_exactness(model, prompts, args.tokens, args.batch, threaded=not args.no_thread,
                            modes=tuple(args.modes.split(",")))
        out["results"][name] = res
        for k, r in res.items():
            if k != "seconds":
                print(f"{name} {k}: identical {r['tokens_identical_frac']:.4f}  max|dlogit| "
                      f"{r['max_abs_logit_diff']:.3g}  exact={r['exact']}  hits {r['serve_stats']['hits']}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out["pass"] = all(r["exact"] for res in out["results"].values() for k, r in res.items() if k != "seconds")
    path = Path(args.out_dir) / f"exactness_{args.run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"pass={out['pass']} -> {path}")


if __name__ == "__main__":
    main()
