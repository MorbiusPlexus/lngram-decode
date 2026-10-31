# Timing protocol (M4.T4)

Applies to every latency number in M4 (`serving/bench_prefetch.py`,
`serving/scaled_tables.py`) and M5 (live speculative loop).

## 1. Exclusive window

1. Reserve the machine for the timing window (GPUs, start, expected length).
   Nobody starts jobs on that server during it, on any GPU. Training runs are
   paused or finished first; a timing run never shares the machine with training.
2. Before starting, check that the machine is idle and record that:
   ```bash
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv   # must list nothing
   uptime; who
   ```
3. Record the environment once per window (it also lists the GPU processes and load):
   ```bash
   python -m serving.env_capture --server gpu0     # -> results/env_gpu0.txt
   ```
4. Persistence mode on, if we have the rights (otherwise note it in the results):
   `sudo nvidia-smi -pm 1`. Do not lock clocks unless every measured machine does the same.
5. Run each benchmark on one GPU (`CUDA_VISIBLE_DEVICES=<gpu>`), and only one benchmark
   at a time. Other GPUs on the server stay idle; host RAM bandwidth and the PCIe root
   complex are shared.
6. If anything else ran during the window (check `nvidia-smi` and `uptime` again at the
   end), discard the numbers and repeat.

## 2. NUMA placement

The staging thread, the main thread and all page-locked buffers (the host table and the
staging buffers) must sit on the NUMA node of the GPU's PCIe root. Otherwise every H2D
copy crosses the inter-socket link.

Find the node of each GPU (also in `results/env_<server>.txt`):
```bash
nvidia-smi topo -m                                   # "NUMA Affinity" / "CPU Affinity" columns
cat /sys/bus/pci/devices/0000:3b:00.0/numa_node      # bus id from nvidia-smi --query-gpu=pci.bus_id
numactl --hardware
```

Run with CPU and memory bound to that node (example: GPU 0 on node 0):
```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8          # intra-op threads for index_select; <= cores of the node
numactl --cpunodebind=0 --membind=0 python -m serving.bench_prefetch --ckpt runs/w3_sdb_1b \
    --server gpu0 --run sdb --dtype fp16
numactl --cpunodebind=0 --membind=0 python -m serving.scaled_tables \
    --codes dumps/w3_sdb_1b/codes_layer1.npy --server gpu0 --sizes_gb 0.1,1,10 --max_frac_ram 0.5 --include_max
```

Why this is enough:
- The staging thread is created by the process and inherits `--cpunodebind`.
- The host table is allocated and filled by the main thread under `--membind`, so its pages
  are first touched on that node, then page-locked in place with `cudaHostRegister`
  (`serving.prefetch.pin_`). This also avoids the caching host allocator's power-of-two
  rounding for large tables. Small staging buffers come from `pin_memory=True`, which also
  allocates under the membind policy.
- With `--membind`, a table that does not fit in the node's free memory fails to allocate
  instead of silently spilling to the other node. Size `--max_frac_ram` against the
  node's memory (`numactl --hardware`), not the machine total, or the largest tables will
  not fit. Report the node size next to the results.
- Locked-memory limit: `ulimit -l` must be `unlimited` (recorded by env_capture), or
  `cudaHostRegister` fails for large tables.

For a cross-node control (optional, for the paper's discussion): repeat one size with
`--cpunodebind=0 --membind=1` and report it separately.

## 3. What to run, in order

1. `python -m serving.env_capture --server <S>`
2. `python -m serving.exactness --ckpt <run> --run <name> --data_dir $DATA_DIR`
   (500 prompts x 256 tokens, fp32 and fp16). It does not need an exclusive window, but
   timing numbers without `pass: true` are not reported.
3. `python -m serving.bench_prefetch --ckpt <run> --server <S> --run <name>` (batches
   1-32, 64 warmup + 512 tokens, 5 repeats, all five conditions).
4. `python -m serving.scaled_tables --codes <dump> --server <S> ...` on each machine.

## 4. Measurement rules (as implemented)

- Per-token latency is host wall time per decode step with a device synchronize after
  every step. Report median and p99 over the 5 x 512 timed steps, after 64 warmup
  steps per repeat. Tokens/s = batch x steps / total timed time.
- The compute-stream wait is measured with CUDA events recorded right before and after
  `wait_event(copy_event)` on the compute stream. The host-side wait on the staging thread
  is reported separately (`host_wait_ms_per_step`).
- PCIe bytes per token = (rows H2D + codes D2H) / generated tokens. B4 moves 16x the
  rows of B1 per single-token step. Prefill gathers are exact and are excluded from the
  timed window.
- Memory: `torch.cuda.max_memory_allocated` (peak since the condition started) and the
  process's resident memory from `nvidia-smi` (includes the CUDA context). B0 runs first
  with the table on the GPU; then the tables are offloaded for B1/B4.
- CUDA graphs: `b0g`/`b4g` replay the fixed-shape decode step (`StaticDecoder`). The
  staging thread and the event waits stay outside the graphs (see the bench_prefetch
  docstring). If capture fails, the JSON has `graph: false` and `graph_error`, and the
  numbers are for the eager fixed-shape step.
- `stale` must be 0 and B4 `misses` 0 in the timed window. Otherwise the run is invalid.
