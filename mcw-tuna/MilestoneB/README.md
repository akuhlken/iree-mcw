# Milestone B — Standalone IME Microkernels

This directory contains the Milestone B deliverables for issue #24576:

1. **Task 1** — `mmt4d_ime_s8s8s32.c` — self-contained `mmt4d`-shaped IME
   microkernels issuing `vmadot` via inline assembly. Five tile shapes are
   provided, each hand-tuned for its register layout.
2. **Task 2** — `mmt4d_ime_validate.c` — validation harness that checks
   bit-exact agreement between every IME kernel and the scalar reference, for
   both accumulate modes and multiple `K1` depths.
3. **Benchmark** — `mmt4d_ime_benchmark.c` — throughput harness that reports
   wall-clock `us/call` and `GOP/s` for every shape using `CLOCK_MONOTONIC`.

---

## The IME atom and the supported shapes

Every kernel is built from the same hardware atom: one `vmadot` call on the A60
core (VLEN=256) computes a `4 x 4 x 8` (M x N x K) int8 -> int32 MAC tile. A
macro-tile is an `MT x NT` grid of these atoms, with a fixed `K0 = 8` (one atom
deep). The `K1` reduction loop handles arbitrary K depth.

| Kernel                                    | M0 x N0 x K0 | grid (MT x NT) | atoms | acc pairs | vmadot/step |
| ----------------------------------------- | ------------ | -------------- | ----- | --------- | ----------- |
| `iree_uk_mmt4d_tile_s8s8s32_4x8x8_ime`    | 4 x 8 x 8    | 1 x 2          | 2     | 2         | 2           |
| `iree_uk_mmt4d_tile_s8s8s32_8x4x8_ime`    | 8 x 4 x 8    | 2 x 1          | 2     | 2         | 2           |
| `iree_uk_mmt4d_tile_s8s8s32_8x8x8_ime`    | 8 x 8 x 8    | 2 x 2          | 4     | 4         | 4           |
| `iree_uk_mmt4d_tile_s8s8s32_8x16x8_ime`   | 8 x 16 x 8   | 2 x 4          | 8     | 8         | 8           |
| `iree_uk_mmt4d_tile_s8s8s32_16x8x8_ime`   | 16 x 8 x 8   | 4 x 2          | 8     | 8         | 8           |

### Hardware constraints (A60 core, VLEN=256)

| Parameter         | Value                                | Source                                            |
| ----------------- | ------------------------------------ | ------------------------------------------------- |
| Atomic IME tile   | `4 × 4 × 8` (M × N × K)              | docs §2.5 "Tile Shape Overview", A60 row          |
| Accumulator pair  | `{vd, vd+1}`, `vd` must be even      | docs §2.3                                         |
| `MUL_C`           | 2 (dest tile occupies 2 vector regs) | docs §2.1.1, formula `VLEN/(SEW×λ²)=256/(32×4)=2` |
| `LMUL`            | must be `m1`                         | docs §5.1.4 illegal conditions                    |
| Total vector regs | 32 (`v0`–`v31`)                      | RISC-V V spec                                     |

---

## Per-kernel register allocation

Each kernel keeps **all** of its accumulators resident in vector registers
across the entire `K1` loop, and loads each distinct A row-atom / B col-atom
exactly once per `K1` step (maximum operand reuse). Accumulator pairs always
start at `v16` and step by 2 (even-aligned); operands use the low registers.

| Kernel   | A atoms            | B atoms                | accumulator pairs                          | regs used |
| -------- | ------------------ | ---------------------- | ------------------------------------------ | --------- |
| 4x8x8    | `v0`               | `v1, v2`               | `v16, v18`                                 | 7 / 32    |
| 8x4x8    | `v0, v2`           | `v1`                   | `v16, v18`                                 | 7 / 32    |
| 8x8x8    | `v0, v2`           | `v1, v3`               | `v16, v18, v20, v22`                       | 12 / 32   |
| 8x16x8   | `v0, v2`           | `v4, v6, v8, v10`      | `v16, v18, v20, v22, v24, v26, v28, v30`   | 22 / 32   |
| 16x8x8   | `v0, v2, v4, v6`   | `v8, v10`              | `v16, v18, v20, v22, v24, v26, v28, v30`   | 22 / 32   |

The two 8-atom shapes are the tightest and still leave 10 vector registers
free, so full accumulator residency is preserved for every shape.

### Accumulator <-> output layout

A `vmadot` accumulator pair holds one `4 x 4` atom as 16 contiguous row-major
int32 lanes. The kernels move these atoms to/from the row-major `out` tile
(row stride `N0`) with two shared C helpers (`ime_gather_acc` /
`ime_scatter_acc`) that run **outside** the `K1` loop, so they do not affect
inner-loop performance:

```
scratch[(mt*NT + nt)*16 + r*4 + c]  <->  out[(mt*4 + r)*N0 + (nt*4 + c)]
```

---

## Files

```
mmt4d_ime_s8s8s32.h     — public interface (5 tile funcs + reference + descriptor)
mmt4d_ime_s8s8s32.c     — IME kernels (vmadot asm) + scalar reference
mmt4d_ime_validate.c    — validation harness (table-driven over all shapes)
mmt4d_ime_benchmark.c   — benchmark harness (wall-clock timing)
Makefile                — build & run helpers
README.md               — this file
```

---

## Building

### On-board (BPI-F3, SpaceMiT GCC or Clang pre-installed)

```sh
# Validator (default target):
make CC=riscv64-linux-gnu-gcc

# Benchmark:
make benchmark CC=riscv64-linux-gnu-gcc

# Or with clang (named mnemonics, LLVM >= 20):
make CC=clang MARCH=rv64gcv_xsmtvdot
```

### Cross-compile from x86 host

```sh
# Using upstream riscv64 GCC toolchain:
make CC=riscv64-unknown-linux-gnu-gcc

# Push and run:
scp mmt4d_ime_validate mmt4d_ime_benchmark root@bpi-f3:~
ssh root@bpi-f3 taskset -c 0-3 ./mmt4d_ime_validate
```

### Using IREE's bundled clang (recommended — knows smt.vmadot)

```sh
# NOTE: `--target=riscv64-...` is REQUIRED. Without it, clang stays in host
# (x86_64) mode and treats `-march=rv64gcv_xsmtvdot` as an x86 -mcpu value:
#   error: unknown target CPU 'rv64gcv_xsmtvdot'
# GCC_TOOLCHAIN only points clang at headers/libs; it does NOT set the arch.
CLANG=/path/to/iree-build/llvm-project/bin/clang
make CC=$CLANG \
     CROSS_FLAGS="--target=riscv64-unknown-linux-gnu --gcc-toolchain=/usr"
```

---

## Running

```sh
# IMPORTANT: pin to Cluster-0 (cores 0-3) — only these cores have the IME.
# Running on Cluster-1 (cores 4-7) will raise an illegal instruction fault.

# Validation:
taskset -c 0-3 ./mmt4d_ime_validate     # or: make run

# Benchmark:
taskset -c 0-3 ./mmt4d_ime_benchmark    # or: make run-bench
```

### Expected validator output (all passing)

The suite runs 14 cases per shape across all 5 shapes (70 total). Each line is
prefixed with the kernel name, e.g.:

```
=== s8s8s32_4x8x8_ime (4x8x8) ===
[PASS] s8s8s32_4x8x8_ime overwrite_random K1=1
...
=== s8s8s32_16x8x8_ime (16x8x8) ===
[PASS] s8s8s32_16x8x8_ime stress_accumulate K1=8

70 / 70 tests passed.
```

### Benchmark output (format)

```
kernel                 shape    depth        us/call         GOP/s
s8s8s32_4x8x8_ime      4x8      K1=1024  ...
s8s8s32_8x4x8_ime      8x4      K1=1024  ...
s8s8s32_8x8x8_ime      8x8      K1=1024  ...
s8s8s32_8x16x8_ime     8x16     K1=1024  ...
s8s8s32_16x8x8_ime     16x8     K1=1024  ...
```

### Understanding K1

`K1` is the **reduction depth along K** — the number of K-panels summed inside
one tile-function call. Together with the fixed tile dimensions it defines how
much work each call performs:

```
C[M0 x N0] += sum over k1=0..K1-1 of  A[k1][M0 x K0]  *  B[k1][N0 x K0]
```

| Symbol | Meaning | In these kernels |
| ------ | ------- | ---------------- |
| **M0** | Output rows per tile | 4, 8, or 16 (depends on shape) |
| **N0** | Output cols per tile | 4, 8, or 16 |
| **K0** | K depth per panel | Always **8** (one IME atom deep) |
| **K1** | Number of K-panels to sum | **Variable** — passed at call time |

Inside the kernel, the hot path is a loop over `k1`. Each iteration loads one A
panel (`M0 x K0` int8s) and one B panel (`N0 x K0` int8s), issues the shape's
`vmadot` grid, and accumulates into register-resident tiles. One-time setup
(zero/load accumulators) and teardown (store/scatter to `out`) run **outside**
this loop.

Work per `k1` step: `M0 * N0 * K0` MACs.  
Work per kernel call: `M0 * N0 * K0 * K1` MACs.

Example for `8x8x8` with `K1 = 1024`: `8 * 8 * 8 = 512` MACs per step, so
`512 * 1024 = 524,288` MACs (~0.5M) per call.

#### K1 in a full matmul

In a real `C[M,N] = A[M,K] * B[K,N]`, the global K dimension is tiled into
chunks of size `K0`. For a given output tile:

```
K1_total = K / K0
```

Example: `K = 8192` and `K0 = 8` gives `K1 = 1024`. IREE invokes the ukernel
once per output tile; each invocation walks all `K1` panels for that tile.
`K1` is not a hardware constant — it is how much K-reduction work is packed
into a single call.

#### Why the benchmark uses `K1 = 1024`

The benchmark fixes `BENCH_K1 = 1024` for all shapes so that:

1. **Steady-state dominates** — accumulator init/load and final store/scatter
   are amortized over a long inner loop, so `GOP/s` reflects load + `vmadot`
   efficiency rather than per-call overhead.
2. **Stable timing** — each call does enough work that wall-clock measurement
   noise is small.
3. **Realistic K** — `K1 = 1024` corresponds to `K = 8192`, a common benchmark
   reduction size.

The **validator** uses small `K1` values (1, 2, 4, 8) to prove correctness at
different reduction depths. The **benchmark** uses one large `K1` to measure
peak sustained throughput.

If you halve `K1`, each call does half the MACs and roughly half the time —
`us/call` drops but `GOP/s` should stay about the same in steady state. Very
small `K1` (e.g. 1) makes setup/teardown dominate and `GOP/s` looks artificially
low; that measures overhead, not inner-loop efficiency.

#### Interpreting `us/call` and `GOP/s`

The benchmark reports:

- **`us/call`** — wall-clock latency per tile-function invocation (diagnostic).
- **`GOP/s`** — `total_MACs / total_time`; billions of int8 MACs per second.

They are related by `GOP/s = MACs_per_call / (us_per_call * 1e-6) / 1e9`. For a
single shape they carry the same information (inverses). **Across shapes,
compare `GOP/s` only** — larger tiles do more MACs per call, so higher
`us/call` is expected and does not mean worse efficiency. **`GOP/s` is the
figure of merit** for choosing a default tile shape in Milestones C/D: it
answers which kernel uses the IME hardware best. Drive it toward peak by
maximizing operand reuse (more `vmadot` calls per loaded A/B atom) within the
32-register budget.

For CPU-cycle counts (`rdcycle` is disabled for user mode on the SpaceMiT X60 /
BPI-F3), measure externally, e.g.
`perf stat -e cycles taskset -c 0-3 ./mmt4d_ime_benchmark`.


---

## Notes for Milestones C/D

The `8x8x8` shape remains the canonical default. The
`enumerateMatmulTileRiscv64` branch (D1) and the `tiles.inl` registration (C3)
select among these shapes by `(M0, N0, K0)`; each row points at the matching
`iree_uk_mmt4d_tile_s8s8s32_<M0>x<N0>x<K0>_ime` function.
