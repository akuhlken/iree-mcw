# Milestone E — End-to-End Correctness (i8×i8→i32 IME)

Validates the full compiler → runtime → hardware path for data-tiled `s8s8s32`
matmul on SpaceMiT K1 (BPI-F3 / Milk-V Jupiter). Requires Milestones
[C](../MilestoneC/README.md) and [D](../MilestoneD/README.md). Covers issue
[#24576](https://github.com/iree-org/iree/issues/24576) tasks **E1–E3**.


| Task   | Where                        | Pass condition                                                                                                |
| ------ | ---------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **E1** | Host (`iree-compile`)        | `iree_codegen.ukernel.generic "iree_uk_mmt4d"` with tile **M0=12, N0=16, K0=8** — not RVV fallback (`7×32×1`) |
| **E2** | Board (Cluster-0, cores 0–3) | `iree-run-module` output **bit-exact** vs reference                                                           |
| **E3** | CI / host build system       | `riscv_64:ime:+v,+zvl256b,+xsmtvdot` variant present in `e2e_matmul_cpu_dt_i8_i32` and `e2e_matmul_cpu_dt_uk_i8_i32`; stock-QEMU run exercises plumbing in fallback mode |


The issue text says `iree_codegen.ukernel.mmt4d`; the lowered op is
`iree_codegen.ukernel.generic "iree_uk_mmt4d"`.

**Test files:** `matmul_i8_i32_aligned.mlir` (E1 + E2), `matmul_i8_i32_narrow_m.mlir`,
`matmul_accumulate_i8_i32.mlir` (E2 edge cases), `generate_inputs.py`.

All commands run from the **repo root**.

---



### Step 1 — Prerequisites

```sh
# Host compiler + RISC-V runtime (see Milestone C §2)
test -x build-host/install/bin/iree-compile
test -x build-riscv/tools/iree-run-module

# IME ukernel linked in cross-build
grep IREE_UK_BUILD_RISCV_64_XSMTVDOT \
  build-riscv/runtime/src/iree/builtins/ukernel/arch/riscv_64/config_riscv_64.h
# expect: #define IREE_UK_BUILD_RISCV_64_XSMTVDOT
```

Board SSH access (e.g. `bananapi`). IME runs only on **cores 0–3**.

Shared compile flags (K1 X60, VLEN=256):

```
--iree-hal-target-device=local
--iree-hal-local-target-device-backends=llvm-cpu
--iree-llvmcpu-target-triple=riscv64
--iree-llvmcpu-target-abi=lp64d
--iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+c,+v,+zvl256b,+xsmtvdot"
--iree-opt-data-tiling
```

`--iree-llvmcpu-enable-ukernels=mmt4d` is omitted — D3 auto-enables when `+xsmtvdot` is present.

---



### Step 2 — E1: Confirm IME ukernel in dispatch IR (host)

```sh
./build-host/install/bin/iree-compile \
  mcw-tuna/MilestoneE/matmul_i8_i32_aligned.mlir \
  -o /tmp/matmul_i8_i32_aligned.vmfb \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=riscv64 \
  --iree-llvmcpu-target-abi=lp64d \
  --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+c,+v,+zvl256b,+xsmtvdot" \
  --iree-opt-data-tiling \
  --mlir-print-ir-after-all 2>/tmp/ime_e1_ir.log

grep -E 'ukernel.generic "iree_uk_mmt4d"|%c12_i32.*%c16_i32.*%c8_i32' \
  /tmp/ime_e1_ir.log
```

**Pass:** `iree_uk_mmt4d` with `%c12_i32, %c16_i32, %c8_i32`. **Fail:** RVV fallback `%c7_i32, %c32_i32, %c1_i32`.

---



### Step 3 — E2: Compile remaining modules (host)

```sh
for f in matmul_i8_i32_narrow_m matmul_accumulate_i8_i32; do
  ./build-host/install/bin/iree-compile \
    mcw-tuna/MilestoneE/${f}.mlir \
    -o /tmp/${f}.vmfb \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=riscv64 \
    --iree-llvmcpu-target-abi=lp64d \
    --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+c,+v,+zvl256b,+xsmtvdot" \
    --iree-opt-data-tiling
done
```

`matmul_i8_i32_aligned.vmfb` is already built in Step 2.

---



### Step 4 — Generate inputs and reference (host)

Requires `numpy` (`pip install numpy`). Writes `.bin` files and prints the
`iree-run-module` command for each case.

```sh
python3 mcw-tuna/MilestoneE/generate_inputs.py aligned    /tmp/ime_aligned
python3 mcw-tuna/MilestoneE/generate_inputs.py narrow_m   /tmp/ime_narrow_m
python3 mcw-tuna/MilestoneE/generate_inputs.py accumulate /tmp/ime_accumulate
```

---



### Step 5 — Deploy to board

```sh
scp /tmp/matmul_i8_i32_aligned.vmfb \
    /tmp/matmul_i8_i32_narrow_m.vmfb \
    /tmp/matmul_accumulate_i8_i32.vmfb \
    build-riscv/tools/iree-run-module \
    bananapi:~

scp -r /tmp/ime_aligned /tmp/ime_narrow_m /tmp/ime_accumulate bananapi:~
```

---



### Step 6 — Run on Cluster-0 (board)

IME traps on Cluster-1 (cores 4–7). For each case, `cd` into its input dir and
run the command printed by `generate_inputs.py`:

```sh
IREE_CPU_FORCE_RISCV_64_XSMTVDOT=1 taskset -c 0-3 \
  ~/iree-run-module \
    --device=local-task \
    --task_topology_cpu_ids=0,1,2,3 \
    --module=~/<case>.vmfb \
    --function=<func> \
    --input="MxKxi8=@lhs.bin" \
    --input="KxNxi8=@rhs.bin" \
    --expected_output="MxNxi32=@expected.bin"
```

`accumulate` adds `--input="MxNxi32=@acc.bin"` before `--expected_output`.

**Pass:** exit 0, no mismatch. Run all three: `aligned`, `narrow_m`, `accumulate`.

---



## Troubleshooting


| Symptom                            | Fix                                                       |
| ---------------------------------- | --------------------------------------------------------- |
| `%c7_i32, %c32_i32, %c1_i32` in IR | Add `+xsmtvdot` to `cpu_features`; rebuild `iree-compile` |
| No `iree_uk_mmt4d` in IR           | Add `--iree-llvmcpu-enable-ukernels=mmt4d`                |
| Illegal instruction on board       | `taskset -c 0-3` + `--task_topology_cpu_ids=0,1,2,3`      |
| `iree-run-module` mismatch         | Re-run `generate_inputs.py`; re-copy vmfb and bins        |


---



## Success checklist

- [x] Step 1: `IREE_UK_BUILD_RISCV_64_XSMTVDOT` defined; builds exist
- [x] Step 2: IR log has `iree_uk_mmt4d` with `%c12/%c16/%c8`
- [x] Step 3: three `.vmfb` files in `/tmp`
- [x] Step 4: three input dirs with `.bin` files
- [x] Step 5: vmfb, bins, and `iree-run-module` on board
- [x] Step 6: all three `iree-run-module` runs exit 0
- [x] Step 7 (E3): `riscv_64:ime` variant added to `e2e_matmul_cpu_dt_i8_i32` and `e2e_matmul_cpu_dt_uk_i8_i32`

---



### Step 7 — E3: Register the IME variant in the e2e test suite

**Files changed:**
- `tests/e2e/matmul/CMakeLists.txt` — added `"riscv_64:ime:+v,+zvl256b,+xsmtvdot"` to `TARGET_CPU_FEATURES_VARIANTS` in both `e2e_matmul_cpu_dt_i8_i32` and `e2e_matmul_cpu_dt_uk_i8_i32`.
- `tests/e2e/matmul/BUILD.bazel` — added `"riscv_64:ime:+v,+zvl256b,+xsmtvdot"` to the `i8/i32` branch of the `target_cpu_features_variants` comprehension covering both `dt` and `dt_uk` rules.

**What each variant does:**

| Test rule | `noriscv` label | Behaviour on stock RISC-V QEMU CI |
| --- | --- | --- |
| `e2e_matmul_cpu_dt_i8_i32_llvm-cpu_local-task_ime` | yes (skipped on RISC-V CI) | Plumbing test for cross-compile + real-hardware runs |
| `e2e_matmul_cpu_dt_uk_i8_i32_llvm-cpu_local-task_ime` | no | Runs on RISC-V CI; ukernel falls back to generic on stock QEMU (IME not emulated); exercises the full compile→run path |

**To verify locally (host build, cross-compiling for RISC-V):**

```sh
# Configure a RISC-V cross build with xsmtvdot enabled and IREE_ARCH=riscv_64.
# The cmake test system will generate test names ending in _ime for the new variant.
ctest -R "e2e_matmul_cpu_dt_uk_i8_i32.*_ime" -V
```

On real BPI-F3 hardware the `_ime` variant actually exercises `vmadot` (requires
`taskset -c 0-3`; see Step 6 / issue §Optional 2).