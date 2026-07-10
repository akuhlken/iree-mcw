#!/usr/bin/env bash
# compile_benchmarks.sh — Milestone F
#
# Compiles every benchmark shape in two variants:
#   ime: +xsmtvdot enabled  → vmadot microkernel selected
#   rvv: no +xsmtvdot       → plain RVV vectorised codegen (fallback baseline)
#
# Output: /tmp/bench_f/{shape}_{ime,rvv}.vmfb  (18 files total)
#
# Prerequisites:
#   build-host/install/bin/iree-compile must exist (host build with RISC-V target).
#
# Run from the repo root:
#   bash mcw-tuna/MilestoneF/compile_benchmarks.sh

set -euo pipefail

IREE_COMPILE=${IREE_COMPILE:-./build-host/install/bin/iree-compile}
OUT=${BENCH_OUT:-/tmp/bench_f}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x "$IREE_COMPILE" ]]; then
  echo "ERROR: iree-compile not found at $IREE_COMPILE" >&2
  echo "  Set IREE_COMPILE= or build the host tree first." >&2
  exit 1
fi

mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# Common flags (K1 X60 target, VLEN=256, data-tiling on).
# ---------------------------------------------------------------------------
COMMON_FLAGS=(
  --iree-hal-target-device=local
  --iree-hal-local-target-device-backends=llvm-cpu
  --iree-llvmcpu-target-triple=riscv64
  --iree-llvmcpu-target-abi=lp64d
  --iree-opt-data-tiling
)

# IME variant: vmadot microkernel path.
IME_FEATURES="+m,+a,+f,+d,+c,+v,+zvl256b,+xsmtvdot"

# RVV baseline: same ISA but without xsmtvdot → falls back to RVV codegen.
RVV_FEATURES="+m,+a,+f,+d,+c,+v,+zvl256b"

# ---------------------------------------------------------------------------
SHAPES=(
  small_aligned
  medium_aligned
  large_aligned
  llm_decode
  llm_prefill
  non_aligned
  # Accumulate (C += A×B) — exercises ime_gather_acc; no M0 truncations.
  small_accumulate
  medium_accumulate
  llm_decode_accumulate
)

echo "iree-compile: $IREE_COMPILE"
echo "Output dir  : $OUT"
echo ""

for shape in "${SHAPES[@]}"; do
  mlir="$SCRIPT_DIR/bench_${shape}.mlir"
  if [[ ! -f "$mlir" ]]; then
    echo "SKIP $shape — $mlir not found" >&2
    continue
  fi

  for variant in ime rvv; do
    out_vmfb="$OUT/${shape}_${variant}.vmfb"
    features="$IME_FEATURES"
    [[ "$variant" == "rvv" ]] && features="$RVV_FEATURES"

    printf "  %-22s %-3s -> %s\n" "$shape" "$variant" "$out_vmfb"
    "$IREE_COMPILE" "$mlir" \
      "${COMMON_FLAGS[@]}" \
      --iree-llvmcpu-target-cpu-features="$features" \
      -o "$out_vmfb"
  done
done

echo ""
echo "Done. VMFBs written to $OUT"
echo ""
echo "Sanity-check — verify IME tile selected (expect %c12_i32, %c16_i32, %c8_i32):"
echo "  $IREE_COMPILE $SCRIPT_DIR/bench_medium_aligned.mlir \\"
echo "    ${COMMON_FLAGS[*]} \\"
echo "    --iree-llvmcpu-target-cpu-features=\"$IME_FEATURES\" \\"
echo "    -o /dev/null --mlir-print-ir-after-all 2>&1 \\"
echo "    | grep 'ukernel.generic \"iree_uk_mmt4d\"'"
