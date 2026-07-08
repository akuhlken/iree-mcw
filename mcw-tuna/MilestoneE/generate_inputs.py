#!/usr/bin/env python3
"""Write binary inputs/expected for one Milestone E test case.

Always writes .bin files; run command uses shape=@file syntax so there are
no long inline strings to copy-paste.

Requires numpy: pip install numpy

Usage (from repo root):
  python3 mcw-tuna/MilestoneE/generate_inputs.py aligned   /tmp/ime_aligned
  python3 mcw-tuna/MilestoneE/generate_inputs.py narrow_m  /tmp/ime_narrow_m
  python3 mcw-tuna/MilestoneE/generate_inputs.py accumulate /tmp/ime_accumulate
"""
import argparse
import os
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")

CASES = {
    "aligned": {
        "mlir": "mcw-tuna/MilestoneE/matmul_i8_i32_aligned.mlir",
        "vmfb": "matmul_i8_i32_aligned.vmfb",
        "function": "matmul",
        "m": 24, "k": 16, "n": 32,
        "accumulate": False,
    },
    "narrow_m": {
        "mlir": "mcw-tuna/MilestoneE/matmul_i8_i32_narrow_m.mlir",
        "vmfb": "matmul_i8_i32_narrow_m.vmfb",
        "function": "matmul",
        "m": 9, "k": 9, "n": 9,
        "accumulate": False,
    },
    "accumulate": {
        "mlir": "mcw-tuna/MilestoneE/matmul_accumulate_i8_i32.mlir",
        "vmfb": "matmul_accumulate_i8_i32.vmfb",
        "function": "matmul_accumulate",
        "m": 12, "k": 8, "n": 16,
        "accumulate": True,
    },
}

SEED = 24576


def shape(arr):
    return "x".join(str(d) for d in arr.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("outdir", help="directory to write .bin files into")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    s = CASES[args.case]
    m, k, n = s["m"], s["k"], s["n"]
    rng = np.random.default_rng(args.seed)

    lhs = rng.integers(-128, 127, (m, k), dtype=np.int8)
    rhs = rng.integers(-128, 127, (k, n), dtype=np.int8)
    mm  = lhs.astype(np.int32) @ rhs.astype(np.int32)

    if s["accumulate"]:
        acc = rng.integers(-1000, 1000, (m, n), dtype=np.int32)
        ref = mm + acc
    else:
        acc = None
        ref = mm

    os.makedirs(args.outdir, exist_ok=True)
    d = args.outdir

    lhs.tofile(f"{d}/lhs.bin")
    rhs.tofile(f"{d}/rhs.bin")
    if acc is not None:
        acc.tofile(f"{d}/acc.bin")
    ref.tofile(f"{d}/expected.bin")

    # Build --input / --expected_output flags using @file syntax
    input_flags = [
        f'--input="{shape(lhs)}xi8=@lhs.bin"',
        f'--input="{shape(rhs)}xi8=@rhs.bin"',
    ]
    if acc is not None:
        input_flags.append(f'--input="{shape(acc)}xi32=@acc.bin"')
    expected_flag = f'--expected_output="{shape(ref)}xi32=@expected.bin"'

    inputs_joined = " \\\n    ".join(input_flags)

    print(f"# case={args.case}  seed={args.seed}")
    print(f"# wrote: {d}/lhs.bin  rhs.bin" + ("  acc.bin" if acc is not None else "") + "  expected.bin")
    print()
    print("# ── compile (host) ──────────────────────────────────────────────")
    print(
        f"./build-host/install/bin/iree-compile \\\n"
        f"  {s['mlir']} \\\n"
        f"  -o /tmp/{s['vmfb']} \\\n"
        f"  --iree-hal-target-device=local \\\n"
        f"  --iree-hal-local-target-device-backends=llvm-cpu \\\n"
        f"  --iree-llvmcpu-target-triple=riscv64 \\\n"
        f"  --iree-llvmcpu-target-abi=lp64d \\\n"
        f'  --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+c,+v,+zvl256b,+xsmtvdot" \\\n'
        f"  --iree-opt-data-tiling"
    )
    print()
    print("# ── run (board, Cluster-0, cd into outdir first) ────────────────")
    print(
        f"IREE_CPU_FORCE_RISCV_64_XSMTVDOT=1 taskset -c 0-3 \\\n"
        f"  ~/iree-run-module \\\n"
        f"    --device=local-task \\\n"
        f"    --task_topology_cpu_ids=0,1,2,3 \\\n"
        f"    --module={s['vmfb']} \\\n"
        f"    --function={s['function']} \\\n"
        f"    {inputs_joined} \\\n"
        f"    {expected_flag}"
    )


if __name__ == "__main__":
    main()
