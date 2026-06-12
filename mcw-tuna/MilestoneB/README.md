# Milestone B — Standalone IME Microkernel

This directory contains the two deliverables for Milestone B of issue #24576:

1. **Task 1** — `mmt4d_ime_s8s8s32.c` — self-contained `mmt4d`-shaped IME
  microkernel issuing `vmadot` via inline assembly.
2. **Task 2** — `mmt4d_ime_validate.c` — validation harness that checks
  bit-exact agreement between the IME kernel and the scalar reference for
   the chosen `(M0, N0, K0)` tile, both accumulate modes, and multiple `K1`
   depths.

---

## Tile-shape decision: `(M0, N0, K0) = (8, 8, 8)`

### Hardware constraints (A60 core, VLEN=256)


| Parameter         | Value                                | Source                                            |
| ----------------- | ------------------------------------ | ------------------------------------------------- |
| Atomic IME tile   | `4 × 4 × 8` (M × N × K)              | docs §2.5 "Tile Shape Overview", A60 row          |
| Accumulator pair  | `{vd, vd+1}`, `vd` must be even      | docs §2.3                                         |
| `MUL_C`           | 2 (dest tile occupies 2 vector regs) | docs §2.1.1, formula `VLEN/(SEW×λ²)=256/(32×4)=2` |
| `LMUL`            | must be `m1`                         | docs §5.1.4 illegal conditions                    |
| Total vector regs | 32 (`v0`–`v31`)                      | RISC-V V spec                                     |


### Register budget for `(8, 8, 8)`

One `vmadot` call covers a `4×4×8` MAC.  To tile `8×8`, we need a `2×2` grid
of atomic tiles:

```
              N[0..3]    N[4..7]
M[0..3]:   acc00={v16,v17}  acc01={v18,v19}
M[4..7]:   acc10={v20,v21}  acc11={v22,v23}
```

Register usage:


| Purpose      | Registers | Count        |
| ------------ | --------- | ------------ |
| Accumulators | `v16–v23` | 8            |
| A sub-tiles  | `v0, v2`  | 2            |
| B sub-tiles  | `v1, v3`  | 2            |
| **Total**    |           | **12 of 32** |


This leaves 20 registers free — plenty for the loop overhead and future
pipelining.  An `(8, 16, 8)` tile (issue §2 candidate) would add 4 more acc
pairs for 16 registers, still feasible, but we start at `(8, 8, 8)` for
simplicity and expand after measuring.

### Why K0 = 8 (not 16 or 32)?

- The atomic K depth is 8 (spec: `K_eff = λ × W × LMUL = 2 × 4 × 1 = 8`).
- A `K0 = 8` tile means one pass of the inner loop issues exactly one `vmadot`
per `(4,4)` accumulator sub-tile: clean and easy to reason about.
- The outer K reduction loop (`K1`) handles arbitrary K depths; there is no
hardware pressure to increase K0 beyond 8 for correctness.
- A larger `K0` (e.g. 16) would require two `vmadot` calls per accumulator
sub-tile per K1 step, which is valid but adds complexity before we have
benchmark data.

### `(M0, N0, K0) = (8, 8, 8)` summary

```
Accumulator: 8 registers  ← fits in 32 total
Operands:    4 registers
Overhead:    20 registers free
K1 loop:     arbitrary (≥1)
int8 MACs per iteration:  4 × 4 × 8  × (2×2 tiles) = 512
```

This is the canonical shape for Milestones C and D.  The `enumerateMatmulTileRiscv64`
branch (D1) must return `TileMxNxK{8, 8, 8}` and the `tiles.inl` row (C3)
must register `(S8S8S32, M0=8, K0=8)` pointing at
`iree_uk_mmt4d_tile_s8s8s32_8x8x8_ime`.

---

## Files

```
mmt4d_ime_s8s8s32.h     — public interface (tile func + scalar reference)
mmt4d_ime_s8s8s32.c     — IME kernel (vmadot asm) + scalar reference
mmt4d_ime_validate.c    — validation harness (14 test cases)
Makefile                — build & run helpers
README.md               — this file
```

---

## Building

### On-board (BPI-F3, SpaceMiT GCC or Clang pre-installed)

```sh
# If the vendor toolchain knows xsmtvdot:
make CC=riscv64-linux-gnu-gcc

# Or with clang (named mnemonics, LLVM >= 20):
make CC=clang MARCH=rv64gcv_xsmtvdot
```

### Cross-compile from x86 host

```sh
# Using upstream riscv64 GCC toolchain:
make CC=riscv64-unknown-linux-gnu-gcc \
     GCC_TOOLCHAIN=/path/to/sysroot

# Push and run:
scp mmt4d_ime_validate root@bpi-f3:~
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
taskset -c 0-3 ./mmt4d_ime_validate
```

Expected output (all passing):

```
[PASS] overwrite_random K1=1
[PASS] overwrite_random K1=2
[PASS] overwrite_random K1=4
[PASS] overwrite_random K1=8
[PASS] accumulate_random K1=2
[PASS] zero_A K1=1
[PASS] zero_B K1=1
[PASS] ones K1=1 (expect 8)
[PASS] max_pos A=127 B=127 K1=1
[PASS] extreme neg A=-128 B=127 K1=1
[PASS] accumulate_large_init K1=1
[PASS] checkerboard K1=1
[PASS] stress_overwrite K1=8
[PASS] stress_accumulate K1=8

14 / 14 tests passed.
```

---



