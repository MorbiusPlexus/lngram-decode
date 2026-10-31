"""Wave 3 parameter table (M1.T5).

    python scripts/print_params.py                       # all Wave 3 configs + references
    python scripts/print_params.py configs/w3_sdb_1b.yaml

Prints the breakdown (attention / FFN / tables / readout / router / draft / SSM) and
checks: SD-* within 0.1% of SB; D270 within 0.5% of Run 2's active 269.5M.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ffnmem.config import load_config  # noqa: E402
from ffnmem.model import Transformer, count_params  # noqa: E402
from ffnmem.params import breakdown  # noqa: E402

DEFAULT = ["w1_mem_cut_near", "w2_mem_super1", "w3_sb_1b", "w3_sdb_1b", "w3_sdc_1b", "w3_sda_1b", "w3_dense270"]


def stats(path: Path) -> dict:
    cfg = load_config(path)
    with torch.device("meta"):
        model = Transformer(cfg["model"])
    mem = cfg["model"].get("memory") or {}
    return {"name": cfg["name"], "routes": mem.get("routes", "-"), "bd": breakdown(model),
            "counts": count_params(model), "cfg": cfg}


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or [ROOT / "configs" / f"{n}.yaml" for n in DEFAULT]
    rows = [stats(p) for p in paths]
    keys = list(rows[0]["bd"].keys())
    w = max(len(r["name"]) for r in rows) + 2
    print("".ljust(w) + "R".rjust(6) + "".join(k[:14].rjust(16) for k in keys) + "active/token".rjust(16))
    for r in rows:
        print(r["name"].ljust(w) + str(r["routes"]).rjust(6)
              + "".join(f"{r['bd'][k]:,}".rjust(16) for k in keys)
              + f"{r['counts']['active_non_embedding_per_token']:,}".rjust(16))
    failures = []
    for r in rows:
        target = r["cfg"].get("params", {}).get("match_target")
        if target:
            rel = (r["bd"]["non_embedding"] - target) / target
            ok = abs(rel) < 0.001
            print(f"{r['name']}: {rel:+.4%} vs {r['cfg']['params']['match_config']} -> {'ok' if ok else 'FAIL'}")
            failures += [] if ok else [r["name"]]
        if r["name"] == "w3_dense270":
            run2 = 269_543_936  # w1_mem_cut_near active non-embedding per token (count_params)
            rel = (r["bd"]["non_embedding"] - run2) / run2
            ok = abs(rel) < 0.005
            print(f"{r['name']}: {rel:+.3%} vs Run 2 active {run2:,} -> {'ok' if ok else 'FAIL'}")
            failures += [] if ok else [r["name"]]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
