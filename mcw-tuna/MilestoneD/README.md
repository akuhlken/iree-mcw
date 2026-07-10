# Milestone D — Compiler IME Tile Enablement (12×16×8)

This milestone wires the IREE compiler to emit the SpaceMiT IME (`vmadot`) tile
for `s8s8s32` `mmt4d` when the target has `+xsmtvdot`. The runtime ukernel from
[Milestone C](../MilestoneC/README.md) is registered for **12×16×8**
(`M0=12, N0=16, K0=8`). The compiler must choose the same shape or data-tiling
falls back to generic codegen with no error.

Tasks **D1–D5** of issue
[#24576](https://github.com/iree-org/iree/issues/24576).

---

## Compiler ↔ runtime contract

| Piece | Value |
| ----- | ----- |
| Compiler tile (`enumerateMatmulTileRiscv64`) | `{12, 16, 8}` only |
| Runtime tile row (`mmt4d_riscv_64_tiles.inl`) | `s8, s8, s32, 12, 8, _xsmtvdot` |
| Feature gate | `+v` and `+xsmtvdot` in `cpu_features` |
| Default ukernels (`+xsmtvdot` only) | `mmt4d` (no explicit `--iree-llvmcpu-enable-ukernels=mmt4d`) |

`N0=16` is fixed by the IME hardware atom grid; it is not derived from VLEN.

---

## Implementation

Paths are relative to the repository root.

### D1 — Tile enumeration

**File:** `compiler/src/iree/compiler/Codegen/ExternalInterfaces/CPUEncodingExternalModels.cpp`

In `enumerateMatmulTileRiscv64`, the `i8×i8→i32` branch gated on `+xsmtvdot`
returns:

```cpp
TileMxNxK{12, 16, 8},  // IME 3×4 vmadot atom grid
```


Without `+xsmtvdot`, the existing RVV widening path (`M0=7, K0=1, N0=vlen/8`)
is unchanged.

### D2 — Feature helper

**Files:** `compiler/src/iree/compiler/Codegen/LLVMCPU/Utils.h`,
`compiler/src/iree/compiler/Codegen/LLVMCPU/Utils.cpp`

`hasXsmtvdotFeature(DictionaryAttr)` — mirrors `hasVFeature` / `hasI8mmFeature`.
`CPUEncodingExternalModels.cpp` uses `hasFeature(config, "+xsmtvdot")` directly
to avoid an `ExternalInterfaces` → `LLVMCPU` build dependency.

### D3 — Default ukernels

**File:** `compiler/src/iree/compiler/Codegen/Utils/Utils.cpp`

`getDefaultEnabledUkernels` returns `"mmt4d"` for `riscv64` targets with
`+xsmtvdot`. Plain `+v` RISC-V targets still default to `"none"`.

### D4 — Runtime tile table

Already done in Milestone C:
`runtime/src/iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_tiles.inl`.

### D5 — Lit tests

**File:** `compiler/src/iree/compiler/Codegen/Common/test/materialize_encoding_riscv.mlir`

| Test function | Checks |
| ------------- | ------ |
| `matmul_lowering_i8i8i32_riscv64_xsmtvdot` | `linalg.mmt4d` with `12×8` / `16×8` / `12×16` tiles (`ukernels = "all"`) |
| `matmul_lowering_i8i8i32_riscv64_xsmtvdot_default_ukernels` | Same tiles with no `ukernels` key (D3 auto-enable) |

---

## File index

| File | Role |
| ---- | ---- |
| `compiler/.../CPUEncodingExternalModels.cpp` | D1: `+xsmtvdot` tile enumeration |
| `compiler/.../LLVMCPU/Utils.h`, `Utils.cpp` | D2: `hasXsmtvdotFeature` |
| `compiler/.../Codegen/Utils/Utils.cpp` | D3: default `mmt4d` for `+xsmtvdot` |
| `compiler/.../test/materialize_encoding_riscv.mlir` | D5: lit tests |

---

## Validation

### Prerequisites

Host build with tests enabled:

```sh
cmake -G Ninja -B build-host \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX=build-host/install \
  -DIREE_BUILD_TESTS=ON
cmake --build build-host --target install
```

Use `build-host/tools/iree-opt` and `lit.py` from the build tree — not a
system-installed `iree-opt` on `$PATH`.

### 1. Lit test (host)

```sh
python3 third_party/llvm-project/llvm/utils/lit/lit.py \
  --path build-host/llvm-project/bin \
  --path build-host/tools \
  compiler/src/iree/compiler/Codegen/Common/test/materialize_encoding_riscv.mlir \
  -v
```

**Pass:** `Passed: 1 (100.00%)`. One lit test covers all `// -----` splits in the
file, including both `riscv64_xsmtvdot` cases.

### 2. Encoding materialization (host)

Confirm `linalg.mmt4d` uses the IME inner tiles:

```sh
./build-host/tools/iree-opt \
  --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" \
  --split-input-file \
  compiler/src/iree/compiler/Codegen/Common/test/materialize_encoding_riscv.mlir \
  2>&1 | grep -E "12x8|16x8|12x16|mmt4d"
```

**Pass:** both `@matmul_lowering_i8i8i32_riscv64_xsmtvdot` and
`@matmul_lowering_i8i8i32_riscv64_xsmtvdot_default_ukernels` show
`tensor<?x?x12x8xi8>`, `tensor<?x?x16x8xi8>`, `tensor<?x?x12x16xi32>`, and
`linalg.mmt4d`.

### 3. Ukernel lowering (host)

Confirm the ukernel op carries M0=12, N0=16, K0=8:

```sh
./build-host/tools/iree-opt \
  --split-input-file \
  --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding,iree-codegen-cpu-lower-to-ukernels{skip-intermediate-roundings=true}))" \
  compiler/src/iree/compiler/Codegen/Common/test/materialize_encoding_riscv.mlir \
  2>&1 | grep -A5 "ukernel.generic"
```

**Pass:** two `iree_codegen.ukernel.generic "iree_uk_mmt4d"` blocks with
`%c12_i32, %c16_i32, %c8_i32` for the xsmtvdot splits.

### 4. Runtime symbol check (cross-compile)

See [Milestone C README §2](../MilestoneC/README.md) for the RISC-V cross-build
setup.

```sh
./build_tools/cmake/build_riscv.sh build-riscv

${RISCV_TOOLCHAIN_ROOT}/bin/llvm-nm \
  build-riscv/runtime/src/iree/builtins/ukernel/tools/mmt4d_benchmark \
  | grep xsmtvdot
```

**Pass:** `iree_uk_mmt4d_tile_s8s8s32_12xXXx8_riscv_64_xsmtvdot` is present.

### 5. Hardware ukernel test (BPI-F3 / Jupiter)

IME runs on Cluster-0 only (cores 0–3). Pin with `taskset -c 0-3`.

```sh
scp build-riscv/runtime/src/iree/builtins/ukernel/tools/mmt4d_test \
    build-riscv/runtime/src/iree/builtins/ukernel/tools/mmt4d_benchmark \
    bananapi:~

ssh bananapi 'IREE_CPU_FORCE_RISCV_64_XSMTVDOT=1 taskset -c 0-3 ./mmt4d_test'
```

**Pass:** `types:s8s8s32 tile:12x16x8 cpu_features:xsmtvdot` → `PASS`.

Optional benchmark filter:

```sh
ssh bananapi 'IREE_CPU_FORCE_RISCV_64_XSMTVDOT=1 taskset -c 0-3 \
  ./mmt4d_benchmark --benchmark_filter="12x16x8.*xsmtvdot"'
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `llvm-lit` or `iree-opt: No such file` | Wrong binary path | Use `build-host/llvm-project/bin/llvm-lit` via `lit.py` (step 1) and `build-host/tools/iree-opt` |
| Lit fails with `iteration_sizes` parse error | System `iree-opt` on `$PATH` | Pass `--path build-host/tools` to `lit.py` |
| Lit reports `1 of 1` not `3 of 3` | Expected — one file = one test | All splits run inside that single test |
| Wrong tile shape (`8×8` instead of `12×16`) in step 2 | Stale `iree-opt` | `cmake --build build-host --target iree-opt` |
| No `mmt4d` for `_default_ukernels` split | D3 not compiled in | Rebuild `iree-opt` |
| Ukernel op shows `%c7_i32, %c32_i32, %c1_i32` | RVV fallback, not IME | Check `cpu_features` includes `+xsmtvdot` |
| `llvm-nm` shows no `xsmtvdot` symbol | Toolchain lacks `+xsmtvdot` | Milestone C README §2 Option A/B |
| Board test `SKIP` | Feature bit not set | `IREE_CPU_FORCE_RISCV_64_XSMTVDOT=1` |
| Illegal instruction on board | Cluster-1 core | `taskset -c 0-3` |
| `linalg.matmul` survives in step 2 | Missing encoding resolver | Target attr needs `iree_cpu.cpu_encoding_resolver<>` |
