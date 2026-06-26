# Milestone C — IME ukernel in the IREE runtime (12×16×8)

This milestone integrates the standalone SpaceMiT IME (`vmadot`) microkernel from
[Milestone B](../MilestoneB/README.md) into IREE's runtime ukernel machinery.
When the target advertises `+xsmtvdot`, the runtime tile selector dispatches the
IME tile for data-tiled `s8s8s32` `mmt4d`.

Scope is a single macro-tile shape: **12×16×8** (`M0 × N0 × K0`). It is built
from the hardware IME atom (one `smt.vmadot` = a 4×4×8 int8→int32 MAC on the
X60) in a 3×4 grid of atoms (12 `vmadot` per K panel), with all 12 accumulator
pairs held in vector registers across the full K reduction.

Tasks **C1–C6** of issue
[#24576](https://github.com/iree-org/iree/issues/24576).

---

## Dispatch path

```
iree_uk_mmt4d_p
  └─ iree_uk_mmt4d_select_tile_func_arch   (arch dispatch)
       └─ #include "mmt4d_riscv_64_tiles.inl"   (table of (type, M0, K0, suffix))
            └─ matches (s8s8s32, M0=12, K0=8, _xsmtvdot)
               AND iree_uk_cpu_riscv_64_xsmtvdot(cpu_data) == true
                 └─ iree_uk_mmt4d_tile_s8s8s32_12xXXx8_riscv_64_xsmtvdot
                      └─ smt.vmadot inline asm
```

**Invariant** (issue §1.3): the `(type, M0, K0)` triple in the tile table, the
generated tile-function symbol name, and the compiler-chosen tile must all agree.
A mismatch causes data-tiling to fall back to plain codegen with no error.

| Piece                                       | Value                                                  |
| ------------------------------------------- | ------------------------------------------------------ |
| Tile table row (`mmt4d_riscv_64_tiles.inl`) | `s8, s8, s32, 12, 8, _xsmtvdot`                        |
| Generated symbol                            | `iree_uk_mmt4d_tile_s8s8s32_12xXXx8_riscv_64_xsmtvdot` |
| CPU gate predicate                          | `iree_uk_cpu_riscv_64_xsmtvdot`                        |
| Build feature macro                         | `IREE_UK_BUILD_RISCV_64_XSMTVDOT`                      |
| Feature string / `-march`                   | `xsmtvdot` / `-march=rv64gc_xsmtvdot`                  |

`N0=16` is not part of the dispatch key (the table keys on `M0`/`K0` only). The
kernel asserts `params->N0 == 16` and `params->K0 == 8` to enforce its contract.

---

## Implementation

All paths below are relative to the repository root.

### C1 — Feature gate (`+xsmtvdot`)

| File | Role |
| ---- | ---- |
| `runtime/src/iree/schemas/cpu_feature_bits.inl` | Defines `RISCV_64 / XSMTVDOT` (bit 3). Shared by compiler and runtime; no Milestone C edits. |
| `runtime/src/iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h` | `iree_uk_cpu_riscv_64_xsmtvdot()` reads `IREE_CPU_DATA0_RISCV_64_XSMTVDOT`. `IREE_UK_BUILD_RISCV_64_XSMTVDOT` is defined in the `IREE_DEVICE_STANDALONE` block (bitcode builds always enable it, same as `V`/`ZVFH`/`ZVFHMIN`). |
| `runtime/src/iree/builtins/ukernel/arch/riscv_64/config_riscv_64.h.in` | `#cmakedefine IREE_UK_BUILD_RISCV_64_XSMTVDOT` gates the feature on toolchain `-march` support. |

`mmt4d_riscv_64_entry_point.c` already referenced `iree_uk_cpu_riscv_64_xsmtvdot`;
C1 supplies the predicate and build macro it depends on.

### C2 — Tile function

`runtime/src/iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_xsmtvdot.c`

Port of the Milestone B kernel `iree_uk_mmt4d_tile_s8s8s32_12x16x8_ime`, adapted
to IREE's `iree_uk_mmt4d_tile_func_t` contract:

- Operands: `(out_tile, lhs_panel, rhs_panel, params)`.
- K reduction: `K1 = params->K` (number of `K0`-deep panels), same pattern as
  `mmt4d_riscv_64_v_i8.c`.
- Accumulate mode: `params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE`.
- Gather/scatter uses `iree_uk_int32_t acc_scratch[]`.
- Freestanding: `iree_uk_*` types only; no libc.
- Registered via
  `IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(..., iree_uk_mmt4d_tile_s8s8s32_12xXXx8_riscv_64_xsmtvdot, 12)`.

### C3 — Tile table and entry point

| File | Change |
| ---- | ------ |
| `mmt4d_riscv_64_tiles.inl` | IME row: `(s8, s8, s32, 12, 8, _xsmtvdot)` |
| `mmt4d_riscv_64_entry_point.c` | Unchanged; `_xsmtvdot` suffix machinery was already present |

### C4 — Build

`runtime/src/iree/builtins/ukernel/arch/riscv_64/CMakeLists.txt`

- `IREE_UK_COPTS_RISCV_64_XSMTVDOT = -march=rv64gc_xsmtvdot`
- `check_cxx_compiler_flag(... IREE_UK_BUILD_RISCV_64_XSMTVDOT)` — system-toolchain
  library is built only when the toolchain accepts the IME `-march`
- `ukernel_bitcode_arch_riscv_64_xsmtvdot.bc` in `iree_link_bitcode` (data-tiling
  uses the bitcode path)

`BUILD.bazel` mirrors the bitcode link list.

**`-march` note:** in IREE's bundled LLVM, `FeatureVendorXSMTVDot` depends on
`Zve32f`, so `rv64gc_xsmtvdot` provides the `vsetvli`/`vle8`/`vle32`/`vse32`/`vmv`
ops the kernel needs alongside `smt.vmadot`. If a toolchain rejects vector ops
under that string, use `rv64gcv_xsmtvdot` (full `V`), as in the Milestone B
standalone build.

### C5 — Host feature detection

Compiler-driven builds bake the `cpu_data` bit at compile time. The ukernel unit
test and benchmark construct `cpu_data` from runtime host detection
(`iree_cpu_initialize` → `riscv_hwprobe`) and skip cases whose required feature
is absent.

`riscv_hwprobe` reports `V`/`ZVFH`/`ZVFHMIN` only — there is no OS probe for IME.
On hardware with IME, tests would otherwise always `SKIP` the `xsmtvdot` case.

`runtime/src/iree/base/internal/cpu.c` sets `IREE_CPU_DATA0_RISCV_64_XSMTVDOT`
when `IREE_CPU_FORCE_RISCV_64_XSMTVDOT` is set. This opt-in override (default
off) lets a process pinned to IME-capable Cluster-0 cores (`taskset -c 0-3`)
advertise the feature. Unset env var → unchanged behavior on heterogeneous
systems. Auto-detect via `marchid`/topology is a follow-up (issue §3.3).

### C6 — Tests and benchmark

| File | Registration |
| ---- | ------------ |
| `tools/mmt4d_test.c` | `iree_uk_test_mmt4d(..., 12, 16, 8, "xsmtvdot")` |
| `tools/mmt4d_benchmark.c` | `iree_uk_benchmark_register_mmt4d(..., 12, 16, 8, "xsmtvdot")` |

The harness sets `ALLOW_GENERIC_FALLBACK`. A pass on non-IME hosts (QEMU, etc.)
only validates plumbing — the scalar fallback can satisfy the test. To exercise
`vmadot`, run on IME hardware (Banana Pi BPI-F3 / Milk-V Jupiter, Cluster-0
cores 0–3) with `iree_uk_cpu_riscv_64_xsmtvdot()` returning true.

---

## Build and run (cross-compile → Banana Pi)

### Prerequisites

- LLVM/Clang RISC-V cross toolchain with `smt.vmadot` support (LLVM ≥ 20).
  Set `RISCV_TOOLCHAIN_ROOT`; `build_tools/cmake/linux_riscv64.cmake` uses
  `${prefix}clang` from there (same toolchain as Milestone B).
- Banana Pi BPI-F3 or Milk-V Jupiter over SSH.

### 1. Host tools (once)

```sh
cmake -G Ninja -B build-host \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX=build-host/install \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DIREE_ENABLE_THIN_ARCHIVES=ON \
    -DCMAKE_C_COMPILER=/usr/bin/clang-14 \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++-14 \
    -DIREE_ENABLE_LLD=ON
cmake --build build-host --target install
```

### 2. Cross-compile for RISC-V

```sh
export RISCV_TOOLCHAIN_ROOT=/path/to/llvm-riscv-toolchain
export IREE_HOST_BIN_DIR="$(pwd)/build-host/install/bin"
./build_tools/cmake/build_riscv.sh build-riscv
```

Builds via `linux_riscv64.cmake`, including `riscv_64_xsmtvdot`
(`-march=rv64gc_xsmtvdot`) when `IREE_UK_BUILD_RISCV_64_XSMTVDOT` is set.

Test binaries (executable names, not CMake target names):

```
build-riscv/runtime/src/iree/builtins/ukernel/tools/mmt4d_test
build-riscv/runtime/src/iree/builtins/ukernel/tools/mmt4d_benchmark
```

### 3. Run on Cluster-0

IME instructions exist only on Cluster-0 (cores 0–3). Cluster-1 (cores 4–7)
raises illegal instruction; always use `taskset -c 0-3`.

Set `IREE_CPU_FORCE_RISCV_64_XSMTVDOT=1` — without it, host detection never
reports `xsmtvdot` and IME cases are `SKIPPED`.

```sh
scp build-riscv/runtime/src/iree/builtins/ukernel/tools/mmt4d_test \
    build-riscv/runtime/src/iree/builtins/ukernel/tools/mmt4d_benchmark \
    bananapi:~

ssh bananapi 'IREE_CPU_FORCE_RISCV_64_XSMTVDOT=1 taskset -c 0-3 ./mmt4d_test'
ssh bananapi 'IREE_CPU_FORCE_RISCV_64_XSMTVDOT=1 taskset -c 0-3 ./mmt4d_benchmark'
```

### 4. Expected output

**Correctness:** all cases pass, including
`types:s8s8s32 tile:12x16x8 cpu_features:xsmtvdot`. With the force env var,
the harness logs `🚀 CPU supports required features` (not `🦕 ... SKIPPED`).
Output is compared to an on-device scalar reference in accumulate and overwrite
modes — a pass means bit-exact agreement with `vmadot`.

**Throughput:** `mmt4d_benchmark` reports `s8s8s32 ... 12x16x8 ... xsmtvdot`.
Compare `items/s` (MACs/s) to the `s8s8s32 ... 7x32x1 ... v` RVV row.

### 5. Verify `vmadot` in the binary (optional)

```sh
${RISCV_TOOLCHAIN_ROOT}/bin/llvm-nm build-riscv/runtime/src/iree/builtins/ukernel/tools/mmt4d_test \
  | grep iree_uk_mmt4d_tile_s8s8s32_12xXXx8_riscv_64_xsmtvdot

${RISCV_TOOLCHAIN_ROOT}/bin/llvm-objdump -d build-riscv/runtime/src/iree/builtins/ukernel/tools/mmt4d_test \
  | grep -A2 vmadot | head
```

On device, `🚀 CPU supports required features` plus a pass on Cluster-0 with
`IREE_CPU_FORCE_RISCV_64_XSMTVDOT=1` confirms the IME tile was selected.

---

## File index

Paths under `runtime/src/iree/builtins/ukernel/` unless noted.

| File | Description |
| ---- | ----------- |
| `arch/riscv_64/common_riscv_64.h` | `xsmtvdot` predicate; standalone build define |
| `arch/riscv_64/config_riscv_64.h.in` | `#cmakedefine` for build flag |
| `arch/riscv_64/mmt4d_riscv_64_xsmtvdot.c` | 12×16×8 IME tile function |
| `arch/riscv_64/mmt4d_riscv_64_tiles.inl` | `s8s8s32, 12, 8, _xsmtvdot` registration |
| `arch/riscv_64/CMakeLists.txt` | COPTS, build-flag check, bitcode link |
| `arch/riscv_64/BUILD.bazel` | bitcode link list |
| `base/internal/cpu.c` | `IREE_CPU_FORCE_RISCV_64_XSMTVDOT` override |
| `tools/mmt4d_test.c` | 12×16×8 `xsmtvdot` test case |
| `tools/mmt4d_benchmark.c` | 12×16×8 `xsmtvdot` benchmark case |
