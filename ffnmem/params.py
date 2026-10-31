"""Parameter accounting and matching.

    python -m ffnmem.params configs/w1_*.yaml            # side-by-side table + match check
    python -m ffnmem.params configs/w1_*.yaml --solve    # also print solver suggestions

Counts come from instantiating the model on the ``meta`` device (exact, no memory).
Matching is on the non-embedding total (the tied embedding is identical in every
run); tolerance 0.5% of the reference run (the config with ``params.reference: true``,
else the first file given).
"""

from __future__ import annotations

import argparse
import copy
import sys

import torch

from .config import load_config
from .model import COMPONENTS, Transformer, component_of, count_params

TOLERANCE = 0.005


def model_counts(model_cfg: dict) -> dict:
    with torch.device("meta"):
        model = Transformer(model_cfg)
    return count_params(model)


def breakdown(model) -> dict:
    """Finer split for Wave 3 tables: router (incl. token-router features), readout
    without the router, draft, SSM. Embedding counted once (tied)."""
    out = {k: 0 for k in ("embedding", "attention", "ffn", "norms", "memory_tables", "memory_readout",
                          "router", "draft", "ssm")}
    for name, p in model.named_parameters():
        if ".memory.router." in name or ".memory.router_features." in name or ".memory.token_router." in name:
            key = "router"
        else:
            key = component_of(name)
        out[key] += p.numel()
    out["non_embedding"] = sum(v for k, v in out.items() if k != "embedding")
    return out


def _with_dff_shift(model_cfg: dict, delta: int) -> dict:
    m = copy.deepcopy(model_cfg)
    m["d_ff"] = [v + delta for v in m["d_ff"]]
    return m


def _with_routes(model_cfg: dict, routes: int) -> dict:
    m = copy.deepcopy(model_cfg)
    m["memory"] = dict(m["memory"], routes=routes)
    return m


def solve_dense_dff(model_cfg: dict, target: int, multiple: int = 32) -> tuple[int, dict]:
    """Shift every layer's d_ff by the same amount so the non-embedding total is
    closest to ``target``, keeping the *base* width a multiple of ``multiple``.

    Returns (shift, counts).  Cuts in the config are preserved (they are relative).
    """
    base = model_counts(model_cfg)["non_embedding"]
    slope = 3 * model_cfg["d_model"] * len(model_cfg["layers"])  # params per unit of d_ff, all layers
    base_width = max(model_cfg["d_ff"])  # uncut layers carry the base width
    raw_shift = (target - base) / slope
    best = None
    for cand_width in (
        (int((base_width + raw_shift) // multiple)) * multiple,
        (int((base_width + raw_shift) // multiple) + 1) * multiple,
    ):
        shift = cand_width - base_width
        counts = model_counts(_with_dff_shift(model_cfg, shift))
        err = abs(counts["non_embedding"] - target)
        if best is None or err < best[0]:
            best = (err, shift, counts)
    _, shift, counts = best
    _check_residual(counts["non_embedding"], target, "solve_dense_dff")
    return shift, counts


def solve_routes(model_cfg: dict, target: int) -> tuple[int, dict]:
    """Integer route count R whose non-embedding total is closest to ``target``."""
    if not model_cfg.get("memory"):
        raise ValueError("solve_routes needs a memory config")
    r0 = int(model_cfg["memory"]["routes"])
    c0 = model_counts(model_cfg)["non_embedding"]
    c1 = model_counts(_with_routes(model_cfg, r0 + 1))["non_embedding"]
    slope = c1 - c0
    guess = r0 + round((target - c0) / slope)
    best = None
    for r in (guess - 1, guess, guess + 1):
        if r <= 0:
            continue
        counts = model_counts(_with_routes(model_cfg, r))
        err = abs(counts["non_embedding"] - target)
        if best is None or err < best[0]:
            best = (err, r, counts)
    _, routes, counts = best
    _check_residual(counts["non_embedding"], target, "solve_routes")
    return routes, counts


def _check_residual(value: int, target: int, what: str) -> None:
    rel = abs(value - target) / target
    if rel > TOLERANCE:
        raise ValueError(f"{what}: residual {rel:.3%} exceeds {TOLERANCE:.1%} (value {value:,}, target {target:,})")


def check_match(counts: dict, reference: int, tolerance: float = TOLERANCE) -> float:
    rel = (counts["non_embedding"] - reference) / reference
    if abs(rel) > tolerance:
        raise AssertionError(
            f"non-embedding {counts['non_embedding']:,} differs from reference {reference:,} by {rel:.3%}"
        )
    return rel


def describe(cfg: dict) -> dict:
    m = cfg["model"]
    counts = model_counts(m)
    d_ff = [m["d_ff"][lid] for lid in m["layers"]]
    return {
        "name": cfg["name"],
        "counts": counts,
        "layers": len(m["layers"]),
        "d_ff": f"{max(d_ff)}" + (f" / {min(d_ff)} in {m.get('ffn_cut_layers')}" if min(d_ff) != max(d_ff) else ""),
        "attn": f"{m['n_heads']}h/{m['n_kv_heads']}kv",
        "memory": (
            f"R={m['memory']['routes']} dm={m['memory']['mem_dim']} @ {m['memory_layers']}"
            + "".join(f"; @{k}: {v}" for k, v in (m.get("memory_overrides") or {}).items())
            if m.get("memory") else "none"
        ),
    }


def format_table(rows: list[dict], reference: int) -> str:
    def mfmt(n: int) -> str:
        return f"{n / 1e6:.3f}M"

    header = ["component"] + [r["name"] for r in rows]
    lines = [header]
    lines.append(["layers"] + [str(r["layers"]) for r in rows])
    lines.append(["d_ff"] + [r["d_ff"] for r in rows])
    lines.append(["attention"] + [r["attn"] for r in rows])
    lines.append(["memory"] + [r["memory"] for r in rows])
    for comp in COMPONENTS:
        lines.append([f"params.{comp}"] + [mfmt(r["counts"][comp]) for r in rows])
    for key in ("non_embedding", "total", "active_non_embedding_per_token"):
        lines.append([key] + [mfmt(r["counts"][key]) for r in rows])
    lines.append(["non_embedding exact"] + [f"{r['counts']['non_embedding']:,}" for r in rows])
    lines.append(
        ["vs reference"] + [f"{(r['counts']['non_embedding'] - reference) / reference:+.3%}" for r in rows]
    )
    widths = [max(len(line[i]) for line in lines) for i in range(len(header))]
    out = []
    for i, line in enumerate(lines):
        out.append("| " + " | ".join(cell.ljust(w) for cell, w in zip(line, widths)) + " |")
        if i == 0:
            out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("configs", nargs="+")
    p.add_argument("--solve", action="store_true", help="print d_ff / R that would match the reference")
    p.add_argument("--no-assert", action="store_true")
    args = p.parse_args(argv)

    cfgs = [load_config(c) for c in args.configs]
    ref_cfg = next((c for c in cfgs if c.get("params", {}).get("reference")), cfgs[0])
    rows = [describe(c) for c in cfgs]
    reference = next(r for r in rows if r["name"] == ref_cfg["name"])["counts"]["non_embedding"]
    print(f"reference: {ref_cfg['name']} ({reference:,} non-embedding params); tolerance {TOLERANCE:.1%}\n")
    print(format_table(rows, reference))

    if args.solve:
        print()
        for c in cfgs:
            if c is ref_cfg:
                continue
            m = c["model"]
            if m.get("memory"):
                routes, counts = solve_routes(m, reference)
                print(f"{c['name']}: routes {m['memory']['routes']} -> {routes} ({counts['non_embedding']:,})")
            else:
                shift, counts = solve_dense_dff(m, reference)
                print(f"{c['name']}: base d_ff {max(m['d_ff'])} -> {max(m['d_ff']) + shift} ({counts['non_embedding']:,})")

    failures = []
    for c, r in zip(cfgs, rows):
        expected = c.get("params", {}).get("expected_non_embedding")
        if expected is not None and int(expected) != r["counts"]["non_embedding"]:
            failures.append(f"{c['name']}: expected_non_embedding {int(expected):,} != {r['counts']['non_embedding']:,}")
        if c.get("params", {}).get("match_reference", True):
            try:
                check_match(r["counts"], reference)
            except AssertionError as e:
                failures.append(f"{c['name']}: {e}")
    if failures and not args.no_assert:
        print("\nMATCH FAILURES:\n  " + "\n  ".join(failures))
        return 1
    print("\nall matched configs within tolerance" if not failures else "\n(failures ignored: --no-assert)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
