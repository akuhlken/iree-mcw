// mmt4d_ime_benchmark.c
//
// Milestone B — IME microkernel benchmark harness.
//
// Measures sustained throughput of every registered IME kernel shape
// (4x8x8, 8x4x8, 8x8x8, 8x16x8, 16x8x8) using CLOCK_MONOTONIC for wall-clock
// timing.
//
// For each kernel we run a long K1 reduction (so the steady-state inner loop
// dominates over the one-time accumulator load/store and scatter/gather), with
// the L/R panels sized to stay resident in cache, repeated many times.
//
// Reported per kernel:
//   us/call   — average wall-clock time for one tile-function invocation
//   GOP/s     — billions of int8 MACs per second (1 MAC = 2 ops if you prefer
//               FLOP-style counting; here we report MACs directly)
//
// IMPORTANT: numbers are only meaningful pinned to Cluster-0 (cores with the
// IME unit). Run:
//   taskset -c 0-3 ./mmt4d_ime_benchmark
//
// Build:
//   make benchmark CC=<riscv toolchain>

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mmt4d_ime_s8s8s32.h"

// ---------------------------------------------------------------------------
// Kernel registry (same set as the validator).
// ---------------------------------------------------------------------------
static const ime_kernel_desc_t kKernels[] = {
    {"s8s8s32_4x8x8_ime",  iree_uk_mmt4d_tile_s8s8s32_4x8x8_ime,   4,  8, 8},
    {"s8s8s32_8x4x8_ime",  iree_uk_mmt4d_tile_s8s8s32_8x4x8_ime,   8,  4, 8},
    {"s8s8s32_8x8x8_ime",  iree_uk_mmt4d_tile_s8s8s32_8x8x8_ime,   8,  8, 8},
    {"s8s8s32_8x16x8_ime", iree_uk_mmt4d_tile_s8s8s32_8x16x8_ime,  8, 16, 8},
    {"s8s8s32_16x8x8_ime", iree_uk_mmt4d_tile_s8s8s32_16x8x8_ime, 16,  8, 8},
};
#define NUM_KERNELS (int)(sizeof(kKernels) / sizeof(kKernels[0]))

// Benchmark configuration.
// #define BENCH_K1     1024   // reduction depth per call (cache-resident panels)
#define BENCH_ITERS  20000  // tile-function calls per measured run
#define WARMUP_ITERS 1000

// ---------------------------------------------------------------------------
static inline double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_random_s8(int8_t *buf, size_t n, unsigned *seed)
{
    for (size_t i = 0; i < n; i++) {
        int v = (int)(rand_r(seed) % 128) - 64;
        buf[i] = (int8_t)v;
    }
}

// ---------------------------------------------------------------------------
static void bench_kernel(const ime_kernel_desc_t *k, const int K1)
{
    const int M0 = k->M0;
    const int N0 = k->N0;
    const int K0 = k->K0;
    // const int K1 = BENCH_K1;

    int8_t  *lhs = malloc((size_t)K1 * M0 * K0);
    int8_t  *rhs = malloc((size_t)K1 * N0 * K0);
    int32_t *out = malloc((size_t)M0 * N0 * sizeof(int32_t));
    if (!lhs || !rhs || !out) {
        fprintf(stderr, "alloc failed for %s\n", k->name);
        free(lhs); free(rhs); free(out);
        return;
    }

    unsigned seed = 0xC0FFEE ^ (unsigned)M0 ^ ((unsigned)N0 << 8);
    fill_random_s8(lhs, (size_t)K1 * M0 * K0, &seed);
    fill_random_s8(rhs, (size_t)K1 * N0 * K0, &seed);
    memset(out, 0, (size_t)M0 * N0 * sizeof(int32_t));

    // Warm up caches / branch predictors. Accumulate mode keeps `out` live so
    // the compiler cannot elide the work.
    for (int i = 0; i < WARMUP_ITERS; i++)
        k->fn(lhs, rhs, out, M0, K0, /*flags=*/1, K1);

    double t0 = now_seconds();
    for (int i = 0; i < BENCH_ITERS; i++)
        k->fn(lhs, rhs, out, M0, K0, /*flags=*/1, K1);
    double t1 = now_seconds();

    double total_time = t1 - t0;
    double calls      = (double)BENCH_ITERS;

    // int8 MACs performed: M0*N0*K0 per K1 step, K1 steps per call.
    double macs_per_call = (double)M0 * N0 * K0 * K1;
    double total_macs    = macs_per_call * calls;

    double us_per_call = total_time / calls * 1e6;
    double gops        = total_macs / total_time / 1e9;

    // Consume `out` so it is not optimized away.
    volatile int32_t sink = out[0];
    (void)sink;

    char shape[16];
    snprintf(shape, sizeof shape, "%dx%d", M0, N0);

    printf("%-22s %-8s K1=%-5d  %10.2f us/call  %9.3f GOP/s\n",
           k->name, shape, K1, us_per_call, gops);

    free(lhs);
    free(rhs);
    free(out);
}

// ---------------------------------------------------------------------------
int main(void)
{
    printf("IME mmt4d kernel benchmark (iters=%d)\n", BENCH_ITERS);
    printf("Pin to Cluster-0 for valid numbers: taskset -c 0-3 ./mmt4d_ime_benchmark\n\n");
    printf("%-22s %-8s %-9s %12s  %12s\n",
           "kernel", "shape", "depth", "us/call", "GOP/s");

    // int K1s[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int K1s[] = {1024};
    for (int x = 0; x < (int)(sizeof K1s / sizeof K1s[0]); x++) {
        for (int i = 0; i < NUM_KERNELS; i++)
            bench_kernel(&kKernels[i], K1s[x]);
    }
    return 0;
}
