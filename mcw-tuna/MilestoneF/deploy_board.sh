#!/usr/bin/env bash
# deploy_board.sh — Milestone F
#
# Copies all benchmark VMFBs, the benchmark binary, and the run script to the
# BPI-F3 / Milk-V Jupiter board over SSH.
#
# Usage (from the repo root):
#   bash mcw-tuna/MilestoneF/deploy_board.sh <ssh-host>
#
# Example:
#   bash mcw-tuna/MilestoneF/deploy_board.sh bananapi
#   bash mcw-tuna/MilestoneF/deploy_board.sh user@10.42.4.2
#
# Prerequisites:
#   - compile_benchmarks.sh has been run  → /tmp/bench_f/*.vmfb
#   - build-riscv/tools/iree-benchmark-module exists (RISC-V cross build)

set -euo pipefail

BOARD=${1:?"usage: deploy_board.sh <ssh-host>"}
BENCH_OUT=${BENCH_OUT:-/tmp/bench_f}
IREE_BM=${IREE_BENCHMARK_MODULE:-build-riscv/tools/iree-benchmark-module}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Verify local artifacts exist before connecting.
if [[ ! -d "$BENCH_OUT" ]] || [[ -z "$(ls "$BENCH_OUT"/*.vmfb 2>/dev/null)" ]]; then
  echo "ERROR: no .vmfb files found in $BENCH_OUT" >&2
  echo "  Run compile_benchmarks.sh first." >&2
  exit 1
fi

if [[ ! -x "$IREE_BM" ]]; then
  echo "ERROR: iree-benchmark-module not found at $IREE_BM" >&2
  echo "  Set IREE_BENCHMARK_MODULE= or build the RISC-V cross tree." >&2
  exit 1
fi

echo "Board           : $BOARD"
echo "VMFBs from      : $BENCH_OUT"
echo "Benchmark binary: $IREE_BM"
echo ""

# Create remote directory.
ssh "$BOARD" "mkdir -p ~/bench_f/results"

# Copy VMFBs.
echo "Copying VMFBs ..."
scp "$BENCH_OUT"/*.vmfb "$BOARD":~/bench_f/

# Copy benchmark binary.
echo "Copying iree-benchmark-module ..."
scp "$IREE_BM" "$BOARD":~/bench_f/iree-benchmark-module
ssh "$BOARD" "chmod +x ~/bench_f/iree-benchmark-module"

# Copy the board-side run script.
echo "Copying run_benchmarks.sh ..."
scp "$SCRIPT_DIR/run_benchmarks.sh" "$BOARD":~/bench_f/run_benchmarks.sh
ssh "$BOARD" "chmod +x ~/bench_f/run_benchmarks.sh"

echo ""
echo "Deploy complete. On the board run:"
echo "  bash ~/bench_f/run_benchmarks.sh"
echo ""
echo "Then pull results back with:"
echo "  scp -r $BOARD:~/bench_f/results/ /tmp/bench_f_results/"
