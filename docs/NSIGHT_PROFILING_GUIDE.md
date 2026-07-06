# Profiling EPIC's GPU kernels with NVIDIA Nsight — Student Guide

A step-by-step, reproducible recipe for profiling the **original EPIC** GPU-OLTP
engine (`epic_driver`) with:

- **Nsight Systems** (`nsys`) — the whole-program timeline: CUDA API calls,
  kernel launches, H2D/D2H copies, and (optionally) GPU hardware-counter
  time-series.
- **Nsight Compute** (`ncu`) — one kernel at a time: achieved occupancy, DRAM
  throughput, L2 hit rate, warp-stall reasons, memory- vs compute-bound.

> **Scope.** This guide assumes you have a checkout of the **original EPIC**
> repository (the OSDI '24 "Massively Parallel Multi-Versioned Transaction
> Processing" artifact, Shujian Qian's `epic-eval`) and that commands are run
> from the **root of that checkout**. Everything here uses only EPIC's own CLI,
> build system, and kernels. The profiling *method* (capture → export → slice
> one epoch → deep-dive) is general and works on any CUDA program; only the
> example `epic_driver` flags are EPIC-specific.

The core idea, which is what makes EPIC easy to profile: EPIC processes
transactions in **epochs**, and every epoch runs the *same* sequence of GPU
kernels once. So a run of `--num_epochs 300` launches each kernel ~300 times.
That repetition lets you **run many epochs, then pull the numbers for one clean
steady-state epoch out of the exported database.**

---

## 0. TL;DR

```bash
# (from the root of your EPIC checkout, after building -- see Part 1)

# 1) Capture a multi-epoch timeline trace
nsys profile --trace=cuda --force-overwrite=true --output=myrun \
  ./build/epic_driver -b ycsbf -d epic -w 1 -a 0.5 -r false \
    -c 8 -e 200 -s 100000 -f true -m false -n 1000000 -x gpu

# 2) Export the trace to a queryable SQLite "database"
nsys export --type sqlite --force-overwrite true --output myrun.sqlite myrun.nsys-rep

# 3) See the kernels and their per-epoch share
nsys stats --report cuda_gpu_kern_sum myrun.nsys-rep | head -30

# 4) Pull ONE steady-state epoch out of the DB (script in Part 3.4)
python3 one_epoch.py myrun.sqlite 150
```

Then use `ncu` (Part 4) to deep-dive the one or two kernels that dominate.

---

## 1. Prerequisites

### 1.1 Tools

`nsys` and `ncu` ship with the CUDA Toolkit. Confirm they're on your PATH:

```bash
nsys --version        # Nsight Systems
ncu  --version        # Nsight Compute
```
If missing, they live under `/usr/local/cuda/bin/` (CLI) and
`/opt/nvidia/nsight-systems*/`, `/opt/nvidia/nsight-compute*/` (with GUIs
`nsys-ui` / `ncu-ui`). Any reasonably recent version (2022+) is fine.

Make sure the GPU is idle before profiling — a shared GPU makes timings
meaningless:
```bash
nvidia-smi
```

### 1.2 Build EPIC

EPIC builds with CMake. Dependencies: a CUDA toolkit (`nvcc`), a C++17 host
compiler, OpenMP, and **jemalloc** (`libjemalloc-dev` on Debian/Ubuntu).

```bash
# from the repo root
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target epic_driver
# -> produces ./build/epic_driver
```

**Set the CUDA architecture for *your* GPU.** EPIC's `CMakeLists.txt` sets:
```cmake
set_property(TARGET epic PROPERTY CUDA_ARCHITECTURES 80;86;89)
```
`80` = A100, `86` = A6000 / RTX 3090 (the GPU in the EPIC paper), `89` = Ada
(RTX 4090 / RTX 5000 Ada). Add/replace with your GPU's compute capability
(`nvidia-smi --query-gpu=compute_cap --format=csv`) so the build doesn't JIT or
fail at launch.

> Build it **Release** for realistic numbers. A Debug build (`-G`) disables
> optimizations and device-side inlining and will badly distort kernel timings.

### 1.3 The `epic_driver` flags you need

EPIC uses single-letter `getopt` flags (defined in `main.cpp`). The full set:

| Short | Long | Meaning | For profiling |
|-------|------|---------|---------------|
| `-b` | `--benchmark` | `ycsba` `ycsbb` `ycsbc` `ycsbf`, `tpccn` `tpccfull` … | pick one workload |
| `-d` | `--database` | `epic` (or `gacco`) | use `epic` |
| `-w` | `--num_warehouses` | TPC-C warehouse count | TPC-C size/contention knob (ignored by YCSB) |
| `-a` | `--skew_factor` | YCSB Zipfian θ, `0`–`0.99` | contention knob; high θ = hot records |
| `-r` | `--fullread` | `true`/`false` | read whole record regardless of field split |
| `-c` | `--cpu_exec_num_threads` | CPU worker threads | only matters for `-x cpu` |
| `-e` | `--num_epochs` | number of epochs | use a few hundred so steady state exists |
| `-s` | `--num_txns` | transactions per epoch | epoch size (paper uses 100000) |
| `-f` | `--split_fields` | `true`/`false` | split a record into per-field sub-records |
| `-m` | `--commutative_ops` | `true`/`false` | commutative (atomic) TPC-C NewOrder path |
| `-n` | `--num_records` | total records (YCSB) | **GPU-fit knob** (see below) |
| `-x` | `--exec_device` | `cpu` \| `gpu` | **`gpu` to profile GPU kernels** |

There is **no execution-mode or writeback flag** — EPIC's GPU vs CPU choice is
just `-x gpu` vs `-x cpu`. (Indexing and MVCC initialization always run on the
GPU; `-x` only chooses where the *execution* phase runs.)

> **GPU-only execution requires the database to fit in VRAM.** EPIC's YCSB record
> is ~1 KB (ten 100 B fields), stored in two versions, so `-n 1000000` (≈ 2 GB)
> fits any modern GPU. Scale `-n` (YCSB) or `-w` (TPC-C) up only as far as your
> VRAM allows; if a `-x gpu` run fails to allocate, lower it.

---

## 2. What you're profiling: EPIC's per-epoch kernel pipeline

Per epoch, EPIC runs these GPU phases in order (driver loop in EPIC's
`benchmarks/ycsb.cpp` / `tpcc.cpp`). Knowing the phase→kernel mapping is what
lets you read a profile:

| Phase | Representative kernels (verified in EPIC source) | What it does |
|-------|--------------------------------------------------|--------------|
| **Index transfer** | (H2D `cudaMemcpy`, not a kernel) | ship txn params to GPU |
| **Indexing** | `prepareYcsbIndexKernel`, `indexYcsbKernel` (+ internal `cuco` hash-map kernels); TPC-C: `prepareTpccIndexKernel`, `indexTpccTxnKernel`, B-tree kernels | key → record-ID |
| **Submission** | `prepareSubmitYcsbTxn`, `submitYcsbTxn` | flatten read/write ops |
| **MVCC initialization** | CUB library kernels (`DeviceRadixSort`, `DeviceScan`/`DeviceSelectSweepKernel`) + `calcOperationType` + `scatterRWLocation` | pre-compute every op's version location |
| **Execution** | `gpuExecKernel`, or `gpuPiecewiseExecKernel` when `-f true` | run all txns, **one warp per transaction** |

Two phases dominate and are the interesting profiling targets:

- **MVCC initialization** — the paper's central claim is that this phase is
  *expensive but massively parallel*, which is why it belongs on the GPU. It is
  built on CUB sort + prefix-sum, so it is typically **memory-bandwidth-bound**;
  in a trace the `cub::...DeviceRadixSort` / `DeviceScan` kernels usually own the
  largest slice.
- **Execution** (`gpuExecKernel` / `gpuPiecewiseExecKernel`) — warp-cooperative
  (32 threads cooperate on one transaction), so expect **coalesced** loads and
  memory-bound behavior, *not* compute-bound. Under high skew (`-a 0.99`) the
  execution kernel slows because RAW dependencies on hot records serialize warps
  (reduced effective occupancy) — a great thing to *see* in `ncu`.

**Do not hard-code kernel names from this table.** They're C++ templates and the
exact mangled/demangled string depends on your build and flags. Always read the
real names off *your* run first:
```bash
nsys stats --report cuda_gpu_kern_sum myrun.nsys-rep | head -30
```

---

## 3. Nsight Systems (`nsys`) — the timeline

### 3.1 Capture a trace

```bash
nsys profile \
  --trace=cuda \
  --force-overwrite=true \
  --output=myrun \
  ./build/epic_driver -b ycsbf -d epic -w 1 -a 0.5 -r false \
    -c 8 -e 200 -s 100000 -f true -m false -n 1000000 -x gpu
```

Flags:
- `--trace=cuda` — capture CUDA runtime API + GPU kernels + memcpy. Add `osrt`
  for OS/thread scheduling and `nvtx` if the code has NVTX ranges:
  `--trace=cuda,osrt,nvtx`.
- `--force-overwrite=true` — overwrite an existing report of the same name.
- `--output=myrun` — writes `myrun.nsys-rep`.
- `--stats=true` — *optional*: also print summary tables when the run finishes.

**Pin to the GPU's NUMA node (optional but stabilizes numbers)** on multi-socket
machines: find the GPU's node with `nvidia-smi topo -m`, then wrap the binary in
`numactl --cpunodebind=<N> --membind=<N> ./build/epic_driver ...`.

A few-hundred-epoch YCSB trace with `--trace=cuda` is only a few MB. You don't
need thousands of epochs — just enough to reach steady state.

### 3.2 The quick look (GUI or `nsys stats`)

GUI:
```bash
nsys-ui myrun.nsys-rep &
```
The timeline shows the CPU thread's CUDA API row, the GPU kernel row, and
**separate H2D and D2H copy-engine rows** — handy for spotting transfer/compute
overlap (or the lack of it).

No GUI? Built-in summary reports:
```bash
nsys stats --report cuda_gpu_kern_sum     myrun.nsys-rep   # kernel time, by name
nsys stats --report cuda_gpu_mem_time_sum myrun.nsys-rep   # H2D/D2H time
nsys stats --report cuda_gpu_mem_size_sum myrun.nsys-rep   # H2D/D2H bytes
nsys stats --report cuda_api_sum          myrun.nsys-rep   # CPU-side API time
```

### 3.3 Export to SQLite (the "database")

For exact, scriptable per-epoch numbers, export to SQLite:

```bash
nsys export --type sqlite --force-overwrite true --output myrun.sqlite myrun.nsys-rep
```

The tables you'll use:

| Table | Columns of interest |
|-------|---------------------|
| `CUPTI_ACTIVITY_KIND_KERNEL` | `start`, `end` (ns), `demangledName` → `StringIds.id` |
| `CUPTI_ACTIVITY_KIND_MEMCPY` | `start`, `end`, `bytes`, `copyKind` (1=H2D, 2=D2H, 8=D2D) |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | `start`, `end`, `globalTid`, `nameId` → `StringIds.id` (CPU-side API calls) |
| `StringIds` | `id` → `value` (kernel / API name strings) |
| `GPU_METRICS` | hardware-counter time series (only if captured with `--gpu-metrics-device`, Part 3.5) |

### 3.4 Pull ONE epoch out of the database

The technique: the **execution kernel runs exactly once per epoch**, so its Nth
launch marks epoch N. Take the window between two consecutive execution-kernel
launches and you've isolated one full epoch's pipeline (index → submit → init →
exec), free of warmup.

Save as `one_epoch.py`:

```python
#!/usr/bin/env python3
# one_epoch.py  <trace.sqlite>  <epoch_index>  [exec_kernel_substring]
# Slice one steady-state epoch out of an nsys SQLite export and print
# per-kernel GPU time + H2D/D2H bytes for that single epoch.
import sqlite3, sys

db   = sys.argv[1]
E    = int(sys.argv[2])                       # a steady-state epoch, e.g. 150
anch = sys.argv[3] if len(sys.argv) > 3 else "ExecKernel"   # epoch boundary marker

c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

# 1) Epoch boundaries = each launch of the execution kernel, in time order.
bounds = [r[0] for r in c.execute(
    "SELECT k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k "
    "JOIN StringIds n ON k.demangledName = n.id "
    "WHERE n.value LIKE ? ORDER BY k.start", (f"%{anch}%",))]
if not bounds:
    sys.exit("no kernel matched %r -- run: nsys stats --report cuda_gpu_kern_sum" % anch)
if E >= len(bounds):
    sys.exit(f"only {len(bounds)} epochs in trace; pick E < {len(bounds)}")
t0, t1 = bounds[E-1], bounds[E]               # window = [prev exec, this exec]
print(f"epoch {E}: window {(t1-t0)/1e3:.1f} us wall\n  kernels (this one epoch):")

# 2) Per-kernel GPU time inside that one epoch.
for name, cnt, us in c.execute(
    "SELECT n.value, COUNT(*), SUM(k.end-k.start)/1e3 "
    "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds n ON k.demangledName=n.id "
    "WHERE k.start>=? AND k.start<? GROUP BY n.value ORDER BY 3 DESC", (t0, t1)):
    print(f"    {us:9.2f} us  x{cnt:<3d} {name[:70]}")

# 3) Copy volume inside that one epoch (1=H2D, 2=D2H, 8=D2D).
print("  memcpy (this one epoch):")
for kind, cnt, mb, us in c.execute(
    "SELECT copyKind, COUNT(*), SUM(bytes)/1e6, SUM(end-start)/1e3 "
    "FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND start<? "
    "GROUP BY copyKind ORDER BY 4 DESC", (t0, t1)):
    label = {1: "H2D", 2: "D2H", 8: "D2D"}.get(kind, f"kind{kind}")
    print(f"    {us:9.2f} us  {mb:8.2f} MB  x{cnt:<3d} {label}")
```

Run it:
```bash
python3 one_epoch.py myrun.sqlite 150
```

**Pick a steady-state epoch.** The first ~10 s / first tens of epochs are warmup
(allocation, first-touch, cold caches, CUDA module load). With `-e 200`, epoch
~150 is safely in steady state. Sanity check: run it for two adjacent epochs
(e.g. 149 and 151); they should match within a few percent. If one is an
outlier, you hit a spike — use a neighbour.

**Variant — average over all steady-state epochs.** Instead of one window, filter
by a timestamp cutoff and divide by the number of epochs after it:
```sql
-- pick a cutoff_ns past warmup (e.g. the start of epoch 100 from the script above)
SELECT n.value, SUM(k.end-k.start)/1e3 AS us_total
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds n ON k.demangledName=n.id
WHERE k.start >= <cutoff_ns>
GROUP BY n.value ORDER BY us_total DESC;
-- then divide each us_total by (num_epochs_after_cutoff) for per-epoch us
```
One clean epoch and the steady-state average should agree closely.

### 3.5 GPU hardware metrics (DRAM / PCIe / SM utilization)

To capture **hardware-counter time series** (DRAM read/write %, copy-engine
activity, SM active %, PCIe RX/TX), add `--gpu-metrics-device`:

```bash
sudo nsys profile --trace=cuda --gpu-metrics-device=0 \
  --force-overwrite=true --output=myrun_gpumetrics \
  ./build/epic_driver -b ycsbf -d epic -w 1 -a 0.5 -r false \
    -c 8 -e 100 -s 100000 -f true -m false -n 1000000 -x gpu
```
- **Needs elevated privileges** (`sudo`) to read GPU performance counters.
- The export is **large** (millions of `GPU_METRICS` rows). Keep these runs short
  (`-e 100`).
- Easiest to read in `nsys-ui` as named %-utilization rows. Use this to answer
  "is initialization actually DRAM-bandwidth-bound?" and "does any phase saturate
  PCIe / the copy engine?".

---

## 4. Nsight Compute (`ncu`) — single-kernel deep dive

Use `ncu` once `nsys` has told you *which* kernel dominates and you need *why*:
occupancy, achieved DRAM throughput, L2 hit rate, warp-stall breakdown,
memory- vs compute-bound (Speed-of-Light). `ncu` runs the program but **replays
each profiled kernel** to read several counter sets, so it is slow and selective.

### 4.1 A practical recipe

Confirm the real kernel name from your `nsys` run, then profile a handful of
steady-state launches with a light metric set first. **Counter access is
privileged — you will almost certainly need to prefix `ncu` with `sudo`** (or
apply the driver-parameter fix in Part 4.2); without it `ncu` stops immediately
with `ERR_NVGPUCTRPERM`.

```bash
# real execution-kernel name on your build:
nsys stats --report cuda_gpu_kern_sum myrun.nsys-rep | head

# profile ~5 steady-state launches of the execution kernel:
ncu --kernel-name-base demangled \
    --kernel-name "regex:gpu.*ExecKernel" \
    --launch-skip 150 --launch-count 5 \
    --set basic \
    --export exec_kernel --force-overwrite \
    ./build/epic_driver -b ycsbf -d epic -w 1 -a 0.5 -r false \
      -c 8 -e 200 -s 100000 -f true -m false -n 1000000 -x gpu
```

- `--launch-skip 150` skips warmup launches; `--launch-count 5` profiles five.
- `--set basic` is fast; escalate to `--set full`, or pick sections directly:
  `--section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy`.
- `--kernel-name "regex:..."` handles templated / `<unnamed>`-namespace names.
- `--export exec_kernel` writes `exec_kernel.ncu-rep` → open with
  `ncu-ui exec_kernel.ncu-rep`. Drop `--export` to print to the terminal instead.

**Suggested targets** (the two phases that matter, Part 2):
1. The execution kernel (`gpu.*ExecKernel`).
2. The MVCC-initialization sort: `--kernel-name "regex:.*DeviceRadixSort.*"`
   (and the prefix-sum `.*DeviceScan.*`). These are usually the bandwidth-heavy
   part of the epoch; check `Memory Throughput` / `DRAM Throughput` in the
   Speed-of-Light section.

**A clean, paper-aligned experiment:** profile `gpu.*ExecKernel` at low skew
(`-a 0.01`) and high skew (`-a 0.99`), export both, and diff them in `ncu-ui`
(Baseline feature). You should see the high-skew run lose effective occupancy /
gain warp-stall cycles from hot-record dependency serialization — exactly the
effect the EPIC paper describes for execution under contention.

### 4.2 `ncu` gotchas specific to EPIC

1. **Permissions.** If you get `ERR_NVGPUCTRPERM`, either run under `sudo`, or set
   the driver module parameter once and reboot:
   ```bash
   # /etc/modprobe.d/nvidia.conf
   options nvidia "NVreg_RestrictProfilingToAdminUsers=0"
   ```
2. **Kernel replay vs. EPIC's spin-waits.** EPIC's execution kernel synchronizes
   read-after-write dependencies by **spinning on per-version epoch IDs** (with
   `__threadfence()`). `ncu`'s default `--replay-mode kernel` save/restores device
   memory and replays the *same* launch, which can hang or skew numbers for a
   kernel that busy-waits on global state. If that happens, use
   `--replay-mode application` (re-runs the whole app per pass — slower but no
   in-kernel state surgery) or `--launch-count 1`.
3. **It's slow.** Each profiled launch is replayed many times; `--set full` on a
   big kernel can take minutes. Keep `--launch-skip` (skip warmup) and a small
   `--launch-count`.
4. **Exclusive GPU.** `ncu` serializes the GPU — make sure nothing else is using
   it (`nvidia-smi`).

---

## 5. A complete worked exercise

Reproduce EPIC's per-epoch run-time breakdown (the paper's "Run Time Breakdown")
and see where the GPU spends an epoch:

```bash
# 1) capture
nsys profile --trace=cuda --force-overwrite=true --output=ycsb_mid \
  ./build/epic_driver -b ycsbf -d epic -w 1 -a 0.5 -r false \
    -c 8 -e 200 -s 100000 -f true -m false -n 1000000 -x gpu

# 2) export + name the kernels
nsys export --type sqlite --force-overwrite true --output ycsb_mid.sqlite ycsb_mid.nsys-rep
nsys stats --report cuda_gpu_kern_sum ycsb_mid.nsys-rep | head -30

# 3) one steady-state epoch, broken down by kernel
python3 one_epoch.py ycsb_mid.sqlite 150
```
Group the kernels from step 3 into the phases in Part 2 (indexing / submission /
initialization / execution) and sum each — that's your per-epoch breakdown.
Then:

- Repeat at `-a 0.01` and `-a 0.99` and watch the execution slice grow with skew.
- `ncu` the dominant kernel (usually the CUB sort or the exec kernel) to learn
  whether it's bandwidth- or latency-bound.
- Optionally re-run with `-x cpu` (CPU execution): in the trace the GPU now stops
  after initialization and a **D2H transfer** of the execution plan dominates —
  the data-transfer cost EPIC's paper highlights for the out-of-VRAM regime.

---

## 6. Checklist / common pitfalls

- [ ] **Release build**, with `CUDA_ARCHITECTURES` set to your GPU. (Debug `-G`
      destroys kernel timings.)
- [ ] **GPU idle** (`nvidia-smi`) and, on multi-socket boxes, NUMA-pinned.
- [ ] **`-x gpu`** to profile GPU execution; the database must fit in VRAM (scale
      `-n` / `-w` down if allocation fails).
- [ ] **Exclude warmup** — slice a steady-state epoch (e.g. 150) or a
      post-cutoff average, never epoch 0.
- [ ] **Read real kernel names** from `nsys stats --report cuda_gpu_kern_sum`
      before writing SQL or `--kernel-name` filters; names are templated.
- [ ] **`nsys` measures from the CPU side.** A long `cudaStreamSynchronize` means
      *either* a slow kernel *or* CPU-side delay — use `ncu`'s GPU-side
      `Duration` to tell them apart.
- [ ] **`--gpu-metrics-device` needs `sudo`** and makes large DBs; keep `-e` small.
- [ ] **`ncu` replays kernels** — for EPIC's spin-waiting exec kernel, fall back
      to `--replay-mode application` if you get a hang or garbage counters.

---

## Quick reference: the commands

```bash
# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target epic_driver

# Systems: capture -> export -> summarize -> one epoch
nsys profile --trace=cuda --force-overwrite=true --output=R  ./build/epic_driver <flags>
nsys export --type sqlite --force-overwrite true --output R.sqlite R.nsys-rep
nsys stats  --report cuda_gpu_kern_sum R.nsys-rep
python3 one_epoch.py R.sqlite 150

# Systems: GPU hardware counters (needs sudo)
sudo nsys profile --trace=cuda --gpu-metrics-device=0 --output=Rm ./build/epic_driver <flags>

# Compute: deep-dive one kernel
ncu --kernel-name-base demangled --kernel-name "regex:gpu.*ExecKernel" \
    --launch-skip 150 --launch-count 5 --set basic \
    --export K --force-overwrite  ./build/epic_driver <flags>
ncu-ui K.ncu-rep
```
where `<flags>` is e.g.
`-b ycsbf -d epic -w 1 -a 0.5 -r false -c 8 -e 200 -s 100000 -f true -m false -n 1000000 -x gpu`.
