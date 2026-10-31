"""Record the timing environment of a server (M4.T4) into results/env_<server>.txt.

    python -m serving.env_capture --server gpu0
    python -m serving.env_capture --server gpu1 --out_dir results

Captures: date, host, kernel, Python / PyTorch / CUDA / cuDNN versions, driver,
GPU names, PCIe link generation and width, ``nvidia-smi topo -m``, the NUMA node of
every GPU (sysfs), ``numactl --hardware``, ``free -g``, ``lscpu`` / CPU model,
CPU frequency governor, and the other processes on the GPUs (the timing window
must be exclusive, see serving/TIMING.md).  Missing tools are recorded as such.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path


def sh(cmd: list[str], timeout: int = 30) -> str:
    if shutil.which(cmd[0]) is None:
        return f"<{cmd[0]} not found>"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + (("\n[stderr] " + r.stderr) if r.stderr.strip() else "")).rstrip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"<{' '.join(cmd)} failed: {e}>"


def read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return "<unavailable>"


def gpu_numa_nodes() -> str:
    out = sh(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"])
    if out.startswith("<"):
        return out
    lines = []
    for line in out.splitlines():
        idx, bus = [x.strip() for x in line.split(",")]
        sysfs = bus.lower()[-12:]  # 00000000:3B:00.0 -> 0000:3b:00.0
        node = read(f"/sys/bus/pci/devices/{sysfs}/numa_node")
        cpus = read(f"/sys/bus/pci/devices/{sysfs}/local_cpulist")
        lines.append(f"GPU {idx}  bus {bus}  numa_node {node}  local_cpus {cpus}")
    return "\n".join(lines)


def torch_info() -> str:
    try:
        import torch
    except ImportError:
        return "<torch not importable>"
    lines = [f"torch {torch.__version__}", f"torch.version.cuda {torch.version.cuda}",
             f"cudnn {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}",
             f"cuda available {torch.cuda.is_available()}"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            lines.append(f"cuda:{i} {p.name} cc {p.major}.{p.minor} mem {p.total_memory / 2 ** 30:.1f} GiB "
                         f"SMs {p.multi_processor_count}")
    return "\n".join(lines)


def capture(server: str) -> str:
    sections = [
        ("date", dt.datetime.now().astimezone().isoformat()),
        ("host", f"{socket.gethostname()}  {platform.platform()}"),
        ("uname", sh(["uname", "-a"])),
        ("python", sys.version.replace("\n", " ")),
        ("pytorch", torch_info()),
        ("nvcc", sh(["nvcc", "--version"])),
        ("nvidia-smi", sh(["nvidia-smi"])),
        ("gpus", sh(["nvidia-smi", "--query-gpu=index,name,driver_version,pci.bus_id,pcie.link.gen.max,"
                     "pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,memory.total,"
                     "clocks.max.sm,persistence_mode", "--format=csv"])),
        ("nvidia-smi topo -m", sh(["nvidia-smi", "topo", "-m"])),
        ("gpu numa nodes", gpu_numa_nodes()),
        ("gpu compute processes (must be empty during timing)",
         sh(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"])),
        ("free -g", sh(["free", "-g"])),
        ("numactl --hardware", sh(["numactl", "--hardware"])),
        ("lscpu", sh(["lscpu"])),
        ("cpu governor", read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")),
        ("transparent hugepages", read("/sys/kernel/mm/transparent_hugepage/enabled")),
        ("ulimit -l (locked memory)", sh(["bash", "-c", "ulimit -l"])),
        ("logged-in users", sh(["who"])),
        ("load average", read("/proc/loadavg")),
    ]
    return f"# env_{server}\n\n" + "\n\n".join(f"== {k} ==\n{v}" for k, v in sections) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    path = Path(args.out_dir) / f"env_{args.server}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(capture(args.server))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
