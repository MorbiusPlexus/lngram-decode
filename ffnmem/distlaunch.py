"""Start N local ranks of ``python -m <module> ...``.

On Linux this is ``torchrun --standalone``.  On Windows, torchrun's agent
creates a libuv TCPStore regardless of ``USE_LIBUV=0`` (torch 2.11), so the
ranks are spawned directly with the ``env://`` variables instead.  Used by the
tests and by ``fitcheck``; cluster jobs call torchrun themselves.

    python -m ffnmem.distlaunch --nproc 2 -- -m ffnmem.train --config ... --device cpu
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(nproc: int, args: list[str], timeout: float | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run ``python <args>`` on ``nproc`` ranks; returns rank 0's output (stderr of all ranks joined)."""
    base_env = dict(os.environ if env is None else env)
    base_env["PYTHONPATH"] = str(ROOT) + os.pathsep + base_env.get("PYTHONPATH", "")
    if os.name != "nt":
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", *args]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=base_env, cwd=ROOT)
    base_env["USE_LIBUV"] = "0"
    port = str(free_port())
    procs = []
    for rank in range(nproc):
        e = dict(base_env, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(nproc),
                 LOCAL_WORLD_SIZE=str(nproc), MASTER_ADDR="127.0.0.1", MASTER_PORT=port)
        procs.append(subprocess.Popen([sys.executable, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True, env=e, cwd=ROOT))
    outs = []
    try:
        for p in procs:
            outs.append(p.communicate(timeout=timeout))
    except subprocess.TimeoutExpired:
        for p in procs:
            p.kill()
        raise
    rc = next((p.returncode for p in procs if p.returncode != 0), 0)
    return subprocess.CompletedProcess(procs[0].args, rc, outs[0][0], "\n".join(o[1] for o in outs))


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) < 3 or argv[0] != "--nproc" or "--" not in argv:
        sys.exit("usage: python -m ffnmem.distlaunch --nproc N -- <python args>")
    nproc = int(argv[1])
    args = argv[argv.index("--") + 1 :]
    proc = run(nproc, args)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
