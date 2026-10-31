"""YAML config loading.

A config file may name a parent with ``extends: <file>`` (resolved relative to
the child).  Mappings are merged recursively; everything else is replaced.
``--set a.b=value`` overrides are applied last (values parsed as YAML).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_raw(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    parent = raw.pop("extends", None)
    if parent is not None:
        raw = _merge(_load_raw((path.parent / parent).resolve()), raw)
    return raw


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def resolve_model(model: dict) -> dict:
    """Expand shorthand into explicit per-layer fields.

    - ``d_ff`` may be an int (same for every original layer) or a list.
    - ``ffn_cut: {layers: [...], delta: N}`` subtracts N from those layers.
    - ``layers`` defaults to all original ids ``0..n_layers-1``.
    """
    m = copy.deepcopy(model)
    n_layers = int(m["n_layers"])
    d_ff = m["d_ff"]
    if isinstance(d_ff, int):
        d_ff = [d_ff] * n_layers
    d_ff = [int(v) for v in d_ff]
    if len(d_ff) != n_layers:
        raise ValueError(f"d_ff has {len(d_ff)} entries, expected n_layers={n_layers}")
    cut = m.pop("ffn_cut", None)
    if cut:
        for layer in cut["layers"]:
            d_ff[int(layer)] -= int(cut["delta"])
        m["ffn_cut_layers"] = [int(v) for v in cut["layers"]]
        m["ffn_cut_delta"] = int(cut["delta"])
    if min(d_ff) <= 0:
        raise ValueError(f"non-positive d_ff after cut: {d_ff}")
    m["d_ff"] = d_ff
    if m.get("layers") is None:
        m["layers"] = list(range(n_layers))
    m["layers"] = [int(v) for v in m["layers"]]
    m["memory_layers"] = [int(v) for v in (m.get("memory_layers") or [])]
    if not m.get("memory"):
        m["memory"] = None
        m["memory_layers"] = []
    for idx in m["memory_layers"]:
        if not 0 <= idx < len(m["layers"]):
            raise ValueError(f"memory layer {idx} outside kept-layer range 0..{len(m['layers']) - 1}")
    overrides = m.get("memory_overrides") or {}
    m["memory_overrides"] = {int(k): dict(v) for k, v in overrides.items()} if m["memory"] else {}
    for pos in m["memory_overrides"]:
        if pos not in m["memory_layers"]:
            raise ValueError(f"memory_overrides for position {pos}, which is not a memory layer {m['memory_layers']}")
    m.setdefault("head_dim", 64)
    m.setdefault("norm_eps", 1e-5)
    m.setdefault("rope_theta", 10000.0)
    m.setdefault("init_std", 0.02)
    # Wave 3 (absent -> None, Wave 1 behaviour)
    m["draft"] = dict(m["draft"]) if m.get("draft") and m["draft"].get("enabled", True) else None
    m["ssm"] = dict(m["ssm"]) if m.get("ssm") and m["ssm"].get("enabled", True) else None
    return m


def _resolve_auto_routes(cfg: dict, path: Path) -> None:
    """``memory.routes: auto`` -> the integer R whose non-embedding total is closest to the
    config named by ``params.match_config`` (path relative to this config)."""
    mem = cfg["model"].get("memory")
    if not mem or mem.get("routes") != "auto":
        return
    ref_name = cfg.get("params", {}).get("match_config")
    if not ref_name:
        raise ValueError("memory.routes: auto needs params.match_config")
    from .params import model_counts, solve_routes  # params imports the model; resolve lazily

    ref = load_config((path.parent / ref_name).resolve())
    target = model_counts(ref["model"])["non_embedding"]
    start = dict(cfg["model"], memory=dict(mem, routes=int(ref["model"]["memory"]["routes"])))
    routes, _ = solve_routes(start, target)
    mem["routes"] = routes
    cfg["params"]["match_target"] = target


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path).resolve()
    cfg = apply_overrides(_load_raw(path), overrides)
    cfg.setdefault("name", path.stem)
    cfg["model"] = resolve_model(cfg["model"])
    _resolve_auto_routes(cfg, path)
    cfg.setdefault("train", {})
    cfg["_source"] = str(path)
    return cfg


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
