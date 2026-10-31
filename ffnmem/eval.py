"""Validation loss of a saved checkpoint on the fixed held-out slice.

    python -m ffnmem.eval --ckpt runs/w1_mem_full/checkpoints/step_0001000.pt [--data_dir ...]
    torchrun --nproc_per_node=4 -m ffnmem.eval --ckpt ...          # split over GPUs

Uses the config stored in the checkpoint (``--set`` overrides allowed, e.g.
``train.eval_max_batches=50``).  Prints one JSON line.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import checkpoint as ckpt
from .config import apply_overrides
from .data.loader import ValWindows
from .train import Dist, build_model, evaluate


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args(argv)

    state = ckpt.load(args.ckpt)
    cfg = state["config"]
    overrides = list(args.set) + ([f"train.data_dir={args.data_dir}"] if args.data_dir else [])
    cfg = apply_overrides(cfg, overrides)
    tcfg = cfg["train"]
    d = Dist(args.device)
    model = build_model(cfg, d.device)
    model.load_state_dict(state["model"])
    del state
    val = ValWindows(tcfg["data_dir"], cfg["model"]["seq_len"], int(tcfg["val_tokens"]), d.rank, d.world_size)
    result = evaluate(model, val, d, tcfg)
    result["ckpt"] = str(args.ckpt)
    if d.is_main:
        print(json.dumps(result))
    d.close()
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
