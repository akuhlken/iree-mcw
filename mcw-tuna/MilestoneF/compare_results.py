#!/usr/bin/env python3
"""compare_results.py — Milestone F

Parse iree-benchmark-module JSON output for every (shape, variant) pair and
print a side-by-side IME vs RVV comparison table with GOPS and speedup.

Usage (run on host after pulling ~/bench_f/results/ from the board):
  python3 mcw-tuna/MilestoneF/compare_results.py /tmp/bench_f_results/

The script also prints a machine-readable CSV to stdout when --csv is given.

Output columns:
  shape        — benchmark label
  M / K / N    — matrix dimensions
  IME ms       — median wall-clock time for the IME (vmadot) run (milliseconds)
  RVV ms       — median wall-clock time for the RVV fallback run (milliseconds)
  IME GOPS     — effective throughput of IME path (GOPS = 2·M·N·K / time_s / 1e9)
  RVV GOPS     — effective throughput of RVV path
  speedup      — IME_GOPS / RVV_GOPS
"""

import argparse
import json
import pathlib
import sys

# ---------------------------------------------------------------------------
# Shape registry (must match bench_*.mlir and run_benchmarks.sh).
# ---------------------------------------------------------------------------
SHAPES = [
    ("small_aligned",           96,   64,   128),
    ("medium_aligned",         384,  256,  512),
    ("large_aligned",          768,  512,  1024),
    ("llm_decode",              12,  4096, 4096),
    ("llm_prefill",            384,  4096, 4096),
    ("non_aligned",            100,   100,  100),
    # Accumulate (C += A×B) — ime_gather_acc path; no M0 truncations.
    ("small_accumulate",        96,   64,   128),
    ("medium_accumulate",      384,  256,  512),
    ("llm_decode_accumulate",   12,  4096, 4096),
]

# ---------------------------------------------------------------------------

def load_median_ns(path: pathlib.Path) -> float:
    """Return the median real_time in nanoseconds from a Google Benchmark JSON.

    iree-benchmark-module emits the standard Google Benchmark JSON schema:
      {
        "benchmarks": [
          {"run_type": "iteration", "real_time": ..., "time_unit": "ns", ...},
          {"run_type": "aggregate", "aggregate_name": "mean",   ...},
          {"run_type": "aggregate", "aggregate_name": "median", ...},
          {"run_type": "aggregate", "aggregate_name": "stddev", ...}
        ]
      }
    We prefer the "median" aggregate; fall back to the first iteration entry.
    """
    with open(path) as f:
        data = json.load(f)

    benches = data.get("benchmarks", [])
    if not benches:
        raise ValueError(f"No benchmark entries in {path}")

    # Prefer the median aggregate.
    for b in benches:
        if b.get("run_type") == "aggregate" and b.get("aggregate_name") == "median":
            return _to_ns(b["real_time"], b.get("time_unit", "ns"))

    # Fallback: first iteration entry.
    b = benches[0]
    return _to_ns(b["real_time"], b.get("time_unit", "ns"))


def _to_ns(value: float, unit: str) -> float:
    """Convert a real_time value in the given unit to nanoseconds."""
    factors = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}
    factor = factors.get(unit)
    if factor is None:
        raise ValueError(f"Unknown time_unit '{unit}'")
    return value * factor


def gops(M: int, K: int, N: int, time_ns: float) -> float:
    """Compute effective GOPS.

    Each M×K × K×N matmul performs 2·M·N·K arithmetic operations
    (one multiply + one accumulate per element of the K reduction).
    """
    return 2.0 * M * N * K / (time_ns * 1e-9) / 1e9


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=".",
        help="Directory containing {shape}_{ime,rvv}.json files (default: .)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also print a CSV table after the human-readable table.",
    )
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)

    # ------------------------------------------------------------------
    # Collect results.
    # ------------------------------------------------------------------
    rows = []
    missing = []

    for (shape, M, K, N) in SHAPES:
        ime_path = results_dir / f"{shape}_ime.json"
        rvv_path = results_dir / f"{shape}_rvv.json"

        if not ime_path.exists() or not rvv_path.exists():
            missing.append(shape)
            rows.append((shape, M, K, N, None, None, None, None, None))
            continue

        try:
            ime_ns = load_median_ns(ime_path)
            rvv_ns = load_median_ns(rvv_path)
        except (KeyError, ValueError) as exc:
            print(f"WARNING: could not parse results for {shape}: {exc}", file=sys.stderr)
            rows.append((shape, M, K, N, None, None, None, None, None))
            continue

        ime_g   = gops(M, K, N, ime_ns)
        rvv_g   = gops(M, K, N, rvv_ns)
        speedup = ime_g / rvv_g if rvv_g > 0 else float("inf")
        rows.append((shape, M, K, N, ime_ns, rvv_ns, ime_g, rvv_g, speedup))

    # ------------------------------------------------------------------
    # Human-readable table.
    # ------------------------------------------------------------------
    COL = "{:<26} {:>6} {:>6} {:>6}  {:>10} {:>10}  {:>10} {:>10}  {:>8}"
    header = COL.format("shape", "M", "K", "N",
                        "IME ms", "RVV ms",
                        "IME GOPS", "RVV GOPS", "speedup")
    sep = "-" * len(header)

    print()
    print(header)
    print(sep)

    for (shape, M, K, N, ime_ns, rvv_ns, ime_g, rvv_g, speedup) in rows:
        if ime_ns is None:
            print(COL.format(shape, M, K, N, "N/A", "N/A", "N/A", "N/A", "N/A"))
            continue
        print(COL.format(
            shape, M, K, N,
            f"{ime_ns / 1e6:.2f}",
            f"{rvv_ns / 1e6:.2f}",
            f"{ime_g:.3f}",
            f"{rvv_g:.3f}",
            f"{speedup:.2f}x",
        ))

    print()

    # Geometric mean speedup over shapes with valid results.
    valid_speedups = [r[8] for r in rows if r[8] is not None]
    if valid_speedups:
        import math
        geo_mean = math.exp(sum(math.log(s) for s in valid_speedups) / len(valid_speedups))
        print(f"Geometric mean speedup (IME / RVV): {geo_mean:.2f}x  "
              f"(over {len(valid_speedups)} shapes)")
    print()

    if missing:
        print(f"Missing results for: {', '.join(missing)}")
        print(f"  Run run_benchmarks.sh on the board and scp results back.")
        print()

    # ------------------------------------------------------------------
    # Optional CSV output.
    # ------------------------------------------------------------------
    if args.csv:
        print("# CSV")
        print("shape,M,K,N,ime_ms,rvv_ms,ime_gops,rvv_gops,speedup")
        for (shape, M, K, N, ime_ns, rvv_ns, ime_g, rvv_g, speedup) in rows:
            if ime_ns is None:
                print(f"{shape},{M},{K},{N},,,,,")
            else:
                print(f"{shape},{M},{K},{N},"
                      f"{ime_ns/1e6:.4f},{rvv_ns/1e6:.4f},"
                      f"{ime_g:.4f},{rvv_g:.4f},{speedup:.4f}")


if __name__ == "__main__":
    main()
