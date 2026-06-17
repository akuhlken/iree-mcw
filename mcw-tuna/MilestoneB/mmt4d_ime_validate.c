// mmt4d_ime_validate.c
//
// Milestone B — Task 2: Validate IME kernels against scalar reference
//
// Runs the full test suite against every registered IME kernel shape
// (4x8x8, 8x4x8, 8x8x8, 8x16x8, 16x8x8):
//   1. Overwrite mode (flags=0): K1=1,2,4,8 with random int8 inputs
//   2. Accumulate mode (flags=1): verify C += A*B semantics
//   3. Edge cases: all-zero A, all-zero B, extreme values (±127/-128)
//
// Run on BPI-F3 pinned to Cluster-0:
//   taskset -c 0-3 ./mmt4d_ime_validate
//
// Expected output on success:
//   [PASS] ...  (one line per test case, prefixed with the kernel name)
//   All N tests passed.
//
// Any mismatch prints the failing element coordinates and the got/want values,
// then exits with code 1 so it is easy to detect in scripts.

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mmt4d_ime_s8s8s32.h"

// ---------------------------------------------------------------------------
// Kernel registry — every shape validated by the same suite.
// ---------------------------------------------------------------------------
static const ime_kernel_desc_t kKernels[] = {
    {"s8s8s32_4x8x8_ime",  iree_uk_mmt4d_tile_s8s8s32_4x8x8_ime,   4,  8, 8},
    {"s8s8s32_8x4x8_ime",  iree_uk_mmt4d_tile_s8s8s32_8x4x8_ime,   8,  4, 8},
    {"s8s8s32_8x8x8_ime",  iree_uk_mmt4d_tile_s8s8s32_8x8x8_ime,   8,  8, 8},
    {"s8s8s32_8x16x8_ime", iree_uk_mmt4d_tile_s8s8s32_8x16x8_ime,  8, 16, 8},
    {"s8s8s32_16x8x8_ime", iree_uk_mmt4d_tile_s8s8s32_16x8x8_ime, 16,  8, 8},
};
#define NUM_KERNELS (int)(sizeof(kKernels) / sizeof(kKernels[0]))

// Buffer sizing for the largest shape (M0<=16, N0<=16, K0=8, K1<=8).
#define MAX_M0  16
#define MAX_N0  16
#define MAX_K0  8
#define K1_MAX  8

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void fill_random_s8(int8_t *buf, size_t n, unsigned *seed)
{
    for (size_t i = 0; i < n; i++) {
        // rand_r produces values in [0, RAND_MAX]; map to [-64, 63] to avoid
        // overflow concerns in spot checks, while still exercising sign paths.
        int v = (int)(rand_r(seed) % 128) - 64;
        buf[i] = (int8_t)v;
    }
}

static void fill_constant_s8(int8_t *buf, size_t n, int8_t val)
{
    for (size_t i = 0; i < n; i++) buf[i] = val;
}

// Compare IME output vs reference; print differences and return 0 on mismatch.
static int compare_s32(
    const int32_t *got,
    const int32_t *want,
    int M0, int N0,
    const char *label)
{
    int ok = 1;
    for (int m = 0; m < M0; m++) {
        for (int n = 0; n < N0; n++) {
            int32_t g = got [m * N0 + n];
            int32_t w = want[m * N0 + n];
            if (g != w) {
                fprintf(stderr,
                    "[FAIL] %s: C[%d][%d] got=%d want=%d\n",
                    label, m, n, g, w);
                ok = 0;
            }
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Single test case runner
// ---------------------------------------------------------------------------

static int run_test(
    const ime_kernel_desc_t *k,
    const char    *case_label,
    const int8_t  *lhs,    // [K1][M0][K0]
    const int8_t  *rhs,    // [K1][N0][K0]
    const int32_t *c_init, // [M0][N0] initial accumulator (NULL = zero)
    int32_t        K1,
    int32_t        flags)
{
    const int M0 = k->M0;
    const int N0 = k->N0;
    const int K0 = k->K0;

    int32_t got [MAX_M0 * MAX_N0];
    int32_t want[MAX_M0 * MAX_N0];

    // Initialise both output buffers identically.
    if (c_init) {
        memcpy(got,  c_init, sizeof(int32_t) * M0 * N0);
        memcpy(want, c_init, sizeof(int32_t) * M0 * N0);
    } else {
        memset(got,  0, sizeof(int32_t) * M0 * N0);
        memset(want, 0, sizeof(int32_t) * M0 * N0);
    }

    // Run IME kernel (vmadot path).
    k->fn(lhs, rhs, got, M0, K0, flags, K1);

    // Run scalar reference.
    mmt4d_reference_s8s8s32(
        lhs, rhs, want, M0, N0, K0, K1, /*accumulate=*/(flags & 1));

    char label[96];
    snprintf(label, sizeof label, "%s %s", k->name, case_label);

    int ok = compare_s32(got, want, M0, N0, label);
    if (ok) printf("[PASS] %s\n", label);
    return ok;
}

// ---------------------------------------------------------------------------
// Run the full suite for one kernel shape.
// ---------------------------------------------------------------------------

static void run_suite(const ime_kernel_desc_t *k, int *total, int *passed)
{
    const int M0 = k->M0;
    const int N0 = k->N0;
    const int K0 = k->K0;

    int8_t  lhs_buf[K1_MAX * MAX_M0 * MAX_K0];
    int8_t  rhs_buf[K1_MAX * MAX_N0 * MAX_K0];
    int32_t c_init [MAX_M0 * MAX_N0];

    // 1. Overwrite mode (flags=0), increasing K1, random data
    {
        unsigned seed = 42;
        int K1s[] = {1, 2, 4, 8};
        for (int i = 0; i < (int)(sizeof K1s / sizeof K1s[0]); i++) {
            int K1 = K1s[i];
            fill_random_s8(lhs_buf, (size_t)(K1 * M0 * K0), &seed);
            fill_random_s8(rhs_buf, (size_t)(K1 * N0 * K0), &seed);

            char label[64];
            snprintf(label, sizeof label, "overwrite_random K1=%d", K1);
            *passed += run_test(k, label, lhs_buf, rhs_buf, NULL, K1, 0);
            (*total)++;
        }
    }

    // 2. Accumulate mode (flags=1), K1=2, random data + random initial C
    {
        unsigned seed = 1337;
        fill_random_s8(lhs_buf, (size_t)(2 * M0 * K0), &seed);
        fill_random_s8(rhs_buf, (size_t)(2 * N0 * K0), &seed);
        for (int i = 0; i < M0 * N0; i++) {
            c_init[i] = (int32_t)((int)(rand_r(&seed) % 512) - 256);
        }
        *passed += run_test(k, "accumulate_random K1=2",
            lhs_buf, rhs_buf, c_init, 2, /*flags=*/1);
        (*total)++;
    }

    // 3. Edge: all-zero A  →  C should remain zero
    {
        fill_constant_s8(lhs_buf, M0 * K0, 0);
        fill_constant_s8(rhs_buf, N0 * K0, 7);
        *passed += run_test(k, "zero_A K1=1", lhs_buf, rhs_buf, NULL, 1, 0);
        (*total)++;
    }

    // 4. Edge: all-zero B  →  C should remain zero
    {
        fill_constant_s8(lhs_buf, M0 * K0, 3);
        fill_constant_s8(rhs_buf, N0 * K0, 0);
        *passed += run_test(k, "zero_B K1=1", lhs_buf, rhs_buf, NULL, 1, 0);
        (*total)++;
    }

    // 5. Edge: A=1, B=1  →  each C[m][n] = K0 = 8
    {
        fill_constant_s8(lhs_buf, M0 * K0, 1);
        fill_constant_s8(rhs_buf, N0 * K0, 1);
        *passed += run_test(k, "ones K1=1 (expect 8)",
            lhs_buf, rhs_buf, NULL, 1, 0);
        (*total)++;
    }

    // 6. Edge: extreme values A=127, B=127  →  C[m][n] = 127*127*K0 = 129032
    {
        fill_constant_s8(lhs_buf, M0 * K0, 127);
        fill_constant_s8(rhs_buf, N0 * K0, 127);
        *passed += run_test(k, "max_pos A=127 B=127 K1=1",
            lhs_buf, rhs_buf, NULL, 1, 0);
        (*total)++;
    }

    // 7. Edge: extreme values A=-128, B=127 (signed × signed worst case)
    {
        fill_constant_s8(lhs_buf, M0 * K0, -128);
        fill_constant_s8(rhs_buf, N0 * K0,  127);
        *passed += run_test(k, "extreme neg A=-128 B=127 K1=1",
            lhs_buf, rhs_buf, NULL, 1, 0);
        (*total)++;
    }

    // 8. Accumulate mode, K1=1, ones + large non-zero C_init
    {
        fill_constant_s8(lhs_buf, M0 * K0, 1);
        fill_constant_s8(rhs_buf, N0 * K0, 1);
        for (int i = 0; i < M0 * N0; i++) c_init[i] = 1000000;
        *passed += run_test(k, "accumulate_large_init K1=1",
            lhs_buf, rhs_buf, c_init, 1, /*flags=*/1);
        (*total)++;
    }

    // 9. Checkerboard pattern: A and B alternate [1,-1,1,-1,...]
    {
        for (int i = 0; i < M0 * K0; i++) lhs_buf[i] = (i % 2) ? -1 : 1;
        for (int i = 0; i < N0 * K0; i++) rhs_buf[i] = (i % 2) ? -1 : 1;
        *passed += run_test(k, "checkerboard K1=1",
            lhs_buf, rhs_buf, NULL, 1, 0);
        (*total)++;
    }

    // 10. Stress: K1=8, random data, both accumulate modes
    {
        unsigned seed = 0xdeadbeef;
        fill_random_s8(lhs_buf, (size_t)(K1_MAX * M0 * K0), &seed);
        fill_random_s8(rhs_buf, (size_t)(K1_MAX * N0 * K0), &seed);
        *passed += run_test(k, "stress_overwrite K1=8",
            lhs_buf, rhs_buf, NULL, K1_MAX, 0);
        (*total)++;

        for (int i = 0; i < M0 * N0; i++) c_init[i] = 42;
        *passed += run_test(k, "stress_accumulate K1=8",
            lhs_buf, rhs_buf, c_init, K1_MAX, 1);
        (*total)++;
    }
}

// ---------------------------------------------------------------------------
int main(void)
{
    int total = 0, passed = 0;

    for (int i = 0; i < NUM_KERNELS; i++) {
        printf("=== %s (%dx%dx%d) ===\n",
               kKernels[i].name, kKernels[i].M0, kKernels[i].N0, kKernels[i].K0);
        run_suite(&kKernels[i], &total, &passed);
    }

    printf("\n%d / %d tests passed.\n", passed, total);
    return (passed == total) ? 0 : 1;
}
