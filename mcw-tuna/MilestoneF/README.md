# Milestone F — Benchmarking & Tuning (IME `vmadot` vs RVV baseline)

Quantifies the performance gain of the SpaceMiT IME (`vmadot`) microkernel
implemented in Milestones C/D over the plain RVV vectorised codegen path
across six realistic `s8s8s32` matmul shapes.

Requires Milestones [C](../MilestoneC/README.md) and [D](../MilestoneD/README.md)
(IME ukernel registered in runtime; compiler emits IME tile for `+xsmtvdot`).

Covers issue [#24576](https://github.com/iree-org/iree/issues/24576) tasks **F1–F4**.

---

## Overview

| Concern | Decision |
|---|---|
| Tile (M0,N0,K0) | **(12, 16, 8)** — confirmed by Milestone E E1 pass condition |
| IME features | `+m,+a,+f,+d,+c,+v,+zvl256b,+xsmtvdot` |
| RVV baseline | `+m,+a,+f,+d,+c,+v,+zvl256b` (no `xsmtvdot`) |
| Core affinity | **Cluster-0 (cores 0–3)** for both variants — IME traps on Cluster-1 |
| Measurement | `iree-benchmark-module --benchmark_format=json --benchmark_min_time=3.0` |
| Metric | GOPS = 2·M·N·K / time\_s / 1e9 ; speedup = IME\_GOPS / RVV\_GOPS |

---

## Benchmark shapes

| Label | M | K | N | Notes |
|---|---|---|---|---|
| `small_aligned` | 96 | 64 | 128 | Cache-resident; 8×M0, 8×K0, 8×N0 |
| `medium_aligned` | 384 | 256 | 512 | Spills L2; stresses memory BW |
| `large_aligned` | 768 | 512 | 1024 | Large square; memory-BW-bound |
| `llm_decode` | 12 | 4096 | 4096 | Narrow-M / single-token decode |
| `llm_prefill` | 384 | 4096 | 4096 | Batched prefill; highest abs. GOPS |
| `non_aligned` | 100 | 100 | 100 | Non-tile-aligned; exercises padding |

---

## Files

| File | Purpose |
|---|---|
| `bench_small_aligned.mlir` | MLIR for small cache-resident shape |
| `bench_medium_aligned.mlir` | MLIR for medium shape |
| `bench_large_aligned.mlir` | MLIR for large shape |
| `bench_llm_decode.mlir` | MLIR for narrow-M decode shape |
| `bench_llm_prefill.mlir` | MLIR for batched prefill shape |
| `bench_non_aligned.mlir` | MLIR for non-aligned shape |
| `compile_benchmarks.sh` | Cross-compile all shapes × 2 variants → `/tmp/bench_f/*.vmfb` |
| `deploy_board.sh` | Copy VMFBs + binary to board over SSH |
| `run_benchmarks.sh` | Run benchmarks on board (Cluster-0), save JSON |
| `compare_results.py` | Parse JSONs, print GOPS + speedup table |

---

## Step-by-step

All commands run from the **repo root** unless noted.

---

### Step 0 — Prerequisites

```sh
# Host: iree-compile built with RISC-V backend
test -x build-host/install/bin/iree-compile

# RISC-V cross build: iree-benchmark-module
test -x build-riscv/tools/iree-benchmark-module

# IME ukernel linked into the cross build
grep IREE_UK_BUILD_RISCV_64_XSMTVDOT \
  build-riscv/runtime/src/iree/builtins/ukernel/arch/riscv_64/config_riscv_64.h
# expect: #define IREE_UK_BUILD_RISCV_64_XSMTVDOT

# Board SSH reachable
ssh bananapi "uname -m"   # expect: riscv64
```

---

### Step 1 — Compile all benchmark VMFBs (host)

```sh
bash mcw-tuna/MilestoneF/compile_benchmarks.sh
```

Produces **12 files** in `/tmp/bench_f/`:
- `{shape}_ime.vmfb` — compiled with `+xsmtvdot` (vmadot microkernel)
- `{shape}_rvv.vmfb` — compiled without `+xsmtvdot` (RVV codegen baseline)

**Sanity-check** that the IME tile is selected (not the RVV fallback):

```sh
./build-host/install/bin/iree-compile \
  mcw-tuna/MilestoneF/bench_medium_aligned.mlir \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=riscv64 \
  --iree-llvmcpu-target-abi=lp64d \
  --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+c,+v,+zvl256b,+xsmtvdot" \
  --iree-opt-data-tiling \
  -o /dev/null \
  --mlir-print-ir-after-all 2>&1 | grep 'ukernel.generic "iree_uk_mmt4d"'
```

**Pass:** output contains `%c12_i32, %c16_i32, %c8_i32`.
**Fail** (RVV fallback selected): `%c7_i32, %c32_i32, %c1_i32`.

---

### Step 2 — Deploy to board

```sh
ssh bananapi "mkdir -p ~/bench_f"
bash mcw-tuna/MilestoneF/deploy_board.sh bananapi
```

Copies to `bananapi:~/bench_f/`:
- All 12 VMFBs
- `iree-benchmark-module` (RISC-V binary)
- `run_benchmarks.sh`

---

### Step 3 — Run benchmarks on board (Cluster-0)

```sh
# SSH into board then:
bash ~/bench_f/run_benchmarks.sh
```

Each shape runs twice — once for each variant. Both runs use:
- `taskset -c 0-3` — pins to Cluster-0 (IME traps on Cluster-1, cores 4–7)
- `--task_topology_cpu_ids=0,1,2,3` — IREE task system respects same constraint
- `--benchmark_min_time=3.0` — at least 3 s per variant for stable numbers

The IME run additionally sets `IREE_CPU_FORCE_RISCV_64_XSMTVDOT=1` to bypass
the runtime's CPU-feature detection (which relies on `/proc/cpuinfo`; the K1
kernel does not advertise IME there).

Results land in `~/bench_f/results/{shape}_{ime,rvv}.json`.

---

### Step 4 — Pull results and compare (host)

```sh
scp -r bananapi:~/bench_f/results/ /tmp/bench_f_results/

python3 mcw-tuna/MilestoneF/compare_results.py /tmp/bench_f_results/
```

Optional CSV output:

```sh
python3 mcw-tuna/MilestoneF/compare_results.py /tmp/bench_f_results/ --csv \
  > /tmp/bench_f_results/comparison.csv
```

---

## Expected output

```
shape                     M      K      N      IME ms     RVV ms    IME GOPS   RVV GOPS   speedup
-------------------------------------------------------------------------------------------------
small_aligned            96     64    128        x.xx       x.xx       x.xxx      x.xxx     x.xxX
medium_aligned          384    256    512        x.xx       x.xx       x.xxx      x.xxx     x.xxX
large_aligned           768    512   1024        x.xx       x.xx       x.xxx      x.xxx     x.xxX
llm_decode               12   4096   4096        x.xx       x.xx       x.xxx      x.xxx     x.xxX
llm_prefill             384   4096   4096        x.xx       x.xx       x.xxx      x.xxx     x.xxX
non_aligned             100    100    100        x.xx       x.xx       x.xxx      x.xxx     x.xxX

Geometric mean speedup (IME / RVV): x.xxX  (over 6 shapes)
```

**GOPS** = `2·M·N·K / time_s / 1e9` (one multiply + one accumulate = 2 ops per MAC).
**Speedup** = `IME_GOPS / RVV_GOPS`.

PR #23734 reported ~4× for RVV i8 over scalar codegen; the IME path is
expected to improve on that. The `llm_prefill` shape (large, tile-aligned,
compute-bound) should show the best absolute GOPS. The `non_aligned` and
`llm_decode` shapes expose partial-tile overhead.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `%c7_i32, %c32_i32, %c1_i32` in sanity-check IR | `+xsmtvdot` not in features or Milestone D not applied; rebuild `iree-compile` |
| No `iree_uk_mmt4d` in IR | Add `--iree-llvmcpu-enable-ukernels=mmt4d` (only needed if D3 auto-enable not wired) |
| `Illegal instruction` on board | Run missed `taskset -c 0-3`; job landed on Cluster-1 |
| IME and RVV numbers identical | `IREE_CPU_FORCE_RISCV_64_XSMTVDOT=1` not set for IME run, or the IME vmfb is actually missing `+xsmtvdot` |
| JSON parse error in `compare_results.py` | Benchmark crashed mid-run; re-run `run_benchmarks.sh` for the affected shape |
| Speedup < 1× | Likely a tile-shape mismatch — re-check invariant §1.3 (issue text): `(type, M0, K0)` from compiler must match registered ukernel tile |

---

## Success checklist (F1–F4)

- [ ] **F1** — `compile_benchmarks.sh` completes; sanity-check IR shows IME tile
- [ ] **F2** — `run_benchmarks.sh` completes on board; 12 JSON files produced
- [ ] **F3** — `compare_results.py` prints a complete table; speedup > 1× for all aligned shapes
- [ ] **F4** — `llm_prefill` speedup substantially exceeds `non_aligned` (confirms tile-alignment matters)
- [ ] **F5** — Cluster-0 affinity (`taskset -c 0-3` + `--task_topology_cpu_ids=0,1,2,3`) used for every run (F4 workaround documented in issue §Optional 2)
