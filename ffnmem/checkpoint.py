"""Checkpoint files: atomic save, discovery, retention.

A checkpoint holds everything needed to continue a run exactly: model,
optimizer (consolidated when ZeRO-1 is used), LR schedule, GradScaler, per-rank
Python/NumPy/Torch/CUDA RNG states, optimizer step, data position, config and
git hash.  Layout: ``<out_dir>/checkpoints/step_0000200.pt``.
"""

from __future__ import annotations

import os
import random
import re
import subprocess
from pathlib import Path

import numpy as np
import torch

_NAME = re.compile(r"^step_(\d{7,})\.pt$")


def ckpt_dir(out_dir: str | Path) -> Path:
    return Path(out_dir) / "checkpoints"


def ckpt_path(out_dir: str | Path, step: int) -> Path:
    return ckpt_dir(out_dir) / f"step_{step:07d}.pt"


def list_checkpoints(out_dir: str | Path) -> list[tuple[int, Path]]:
    d = ckpt_dir(out_dir)
    if not d.is_dir():
        return []
    found = []
    for p in d.iterdir():
        m = _NAME.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


def latest_checkpoint(out_dir: str | Path) -> Path | None:
    found = list_checkpoints(out_dir)
    return found[-1][1] if found else None


def save_atomic(state: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic on POSIX and Windows (same filesystem)


def load(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def prune(out_dir: str | Path, keep_last: int, keep_every: int, keep_steps=()) -> list[Path]:
    """Keep the newest ``keep_last`` checkpoints, every ``keep_every``-th step and ``keep_steps``."""
    found = list_checkpoints(out_dir)
    keep = {step for step, _ in found[-keep_last:]} if keep_last > 0 else set()
    keep |= set(keep_steps)
    removed = []
    for step, path in found:
        if step in keep or (keep_every and step % keep_every == 0):
            continue
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed


def strip_optimizer_except_latest(out_dir: str | Path) -> list[Path]:
    """Rewrite every checkpoint but the newest without optimizer state (disk); they stay
    loadable for analysis (``--init_from``), only the newest can ``--resume``."""
    found = list_checkpoints(out_dir)
    stripped = []
    # Older ones were stripped when they stopped being the newest; only re-read the previous newest.
    for _, path in found[-2:-1]:
        state = load(path)
        if state.get("optimizer") is None:
            continue
        state["optimizer"] = None
        state["optimizer_stripped"] = True
        save_atomic(state, path)
        stripped.append(path)
    return stripped


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
        rev = out.stdout.strip() if out.returncode == 0 else "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            rev += "-dirty"
        return rev
    except (OSError, subprocess.SubprocessError):
        return "unknown"
