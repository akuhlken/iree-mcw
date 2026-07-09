#!/usr/bin/env bash
# run_benchmarks.sh — Milestone F  (run on the BPI-F3 / Milk-V Jupiter board)
#
# Benchmarks every shape in both the IME (vmadot) and plain RVV variants.
# Results land in ~/bench_f/results/{shape}_{ime,rvv}.json.
#
# IMPORTANT:
#   - All runs are pinned to Cluster-0 (cores 0–3) via taskset + IREE flag.
#     IME instructions only exist on Cluster-0; running on Cluster-1 traps.
#   - IREE_CPU_FORCE_RISCV_64_XSMTVDOT=1 forces the runtime tile-selector to
#     pick the vmadot kernel even when /proc/cpuinfo carries no IME flag.
#
# Usage (after deploy_board.sh):
#   bash ~/bench_f/run_benchmarks.sh
#
# Optional overrides:
#   BENCH_MIN_TIME=5.0   bench_f/run_benchmarks.sh   (longer stabilisation)
#   BENCH_REPS=3         bench_f/run_benchmarks.sh   (repeat each variant N times)

set -euo pipefail

BIN=~/bench_f/iree-benchmark-module
OUT=~/bench_f/results
BENCH_MIN_TIME=${BENCH_MIN_TIME:-3.0}
BENCH_REPS=${BENCH_REPS:-1}

mkdir -p "$OUT"

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: $BIN not found or not executable." >&2
  echo "  Run deploy_board.sh from the host first." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Shape table: name -> "M K N"
# ---------------------------------------------------------------------------
SHAPE_NAMES=(
  small_aligned
  medium_aligned
  large_aligned
  llm_decode
  llm_prefill
  non_aligned
  small_aligned_8
  medium_aligned_8
  large_aligned_8
  non_aligned_9
  small_aligned_4
  medium_aligned_4
  large_aligned_4
  non_aligned_5
)

declare -A SHAPE_DIMS=(
  [small_aligned]="96 64 128"
  [medium_aligned]="384 256 512"
  [large_aligned]="768 512 1024"
  [llm_decode]="12 4096 4096"
  [llm_prefill]="384 4096 4096"
  [non_aligned]="100 100 100"
  [small_aligned_8]="8 64 128"
  [medium_aligned_8]="8 256 512"
  [large_aligned_8]="8 512 1024"
  [non_aligned_9]="9 100 100"
  [small_aligned_4]="4 64 128"
  [medium_aligned_4]="4 256 512"
  [large_aligned_4]="4 512 1024"
  [non_aligned_5]="5 100 100"
)

# ---------------------------------------------------------------------------
echo "=== Milestone F benchmark run ==="
echo "Binary  : $BIN"
echo "Results : $OUT"
echo "Min time: ${BENCH_MIN_TIME}s per variant"
echo ""

run_variant() {
  local shape=$1
  local variant=$2
  local M=$3 K=$4 N=$5

  local vmfb=~/bench_f/${shape}_${variant}.vmfb
  local out_json="$OUT/${shape}_${variant}.json"

  if [[ ! -f "$vmfb" ]]; then
    echo "  SKIP: $vmfb not found" >&2
    return
  fi

  local env_extra=()
  if [[ "$variant" == "ime" ]]; then
    env_extra=(env IREE_CPU_FORCE_RISCV_64_XSMTVDOT=1)
  fi

  printf "  %-22s %-3s  M=%-4d K=%-5d N=%-5d ... " \
    "$shape" "$variant" "$M" "$K" "$N"

  taskset -c 0-3 \
    "${env_extra[@]}" \
    "$BIN" \
      --device=local-task \
      --task_topology_cpu_ids=0,1,2,3 \
      --module="$vmfb" \
      --function=matmul \
      --input="${M}x${K}xi8" \
      --input="${K}x${N}xi8" \
      --benchmark_min_time="$BENCH_MIN_TIME" \
      --benchmark_format=json \
      --benchmark_out="$out_json" \
      2>/dev/null

  echo "done -> $out_json"
}

for shape in "${SHAPE_NAMES[@]}"; do
  dims="${SHAPE_DIMS[$shape]}"
  read -r M K N <<< "$dims"
  echo "$shape  (M=$M K=$K N=$N)"
  for rep in $(seq 1 "$BENCH_REPS"); do
    [[ "$BENCH_REPS" -gt 1 ]] && echo "  rep $rep/$BENCH_REPS"
    run_variant "$shape" ime "$M" "$K" "$N"
    run_variant "$shape" rvv "$M" "$K" "$N"
  done
  echo ""
done

echo "All benchmarks complete."
echo "Pull results to host:"
echo "  scp -r \$(hostname):~/bench_f/results/ /tmp/bench_f_results/"
echo "Then compare:"
echo "  python3 mcw-tuna/MilestoneF/compare_results.py /tmp/bench_f_results/"
