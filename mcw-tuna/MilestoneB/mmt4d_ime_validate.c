// mmt4d_ime_validate.c
//
// Milestone B — Task 2: Validate IME kernel against scalar reference
//
// Tests:
//   1. Overwrite mode (flags=0): K1=1,2,4 with random int8 inputs
//   2. Accumulate mode (flags=1): verify C += A*B semantics
//   3. Edge cases: all-zero A, all-zero B, extreme values (±127)
//
// Run on BPI-F3 pinned to Cluster-0:
//   taskset -c 0-3 ./mmt4d_ime_validate
//
// Expected output on success:
//   [PASS] ...  (one line per test case)
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
// Helpers
// ---------------------------------------------------------------------------

static void fill_random_s8(int8_t *buf, size_t n, unsigned *seed)
{
    // for (size_t i = 0; i < n; i++) {
    //     // rand_r produces values in [0, RAND_MAX]; map to [-64, 63] to avoid
    //     // overflow concerns in spot checks, while still exercising sign paths.
    //     int v = (int)(rand_r(seed) % 128) - 64;
    //     buf[i] = (int8_t)v;
    // }

    for (size_t i = 0; i < n; i++) {
        buf[i] = (int8_t)(i);
        printf("%d, ",buf[i]);
    }
    // printf("\n");
}

// static void fill_constant_s8(int8_t *buf, size_t n, int8_t val)
// {
//     for (size_t i = 0; i < n; i++) buf[i] = val;
// }

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
    const char    *label,
    const int8_t  *lhs,    // [K1][M0][K0]
    const int8_t  *rhs,    // [K1][N0][K0]
    const int32_t *c_init, // [M0][N0] initial accumulator (NULL = zero)
    int32_t        M0,
    int32_t        N0,
    int32_t        K0,
    int32_t        K1,
    int32_t        flags)
{
    // const int M0 = IME_M0;
    // const int N0 = IME_N0;
    // const int K0 = IME_K0;

    int32_t got [M0 * N0];
    int32_t want[M0 * N0];

    // Initialise both output buffers identically.
    if (c_init) {
        memcpy(got,  c_init, sizeof(got));
        memcpy(want, c_init, sizeof(want));
    } else {
        memset(got,  0, sizeof(got));
        memset(want, 0, sizeof(want));
    }

    // Run IME kernel (vmadot path).
    // printf("run_test: M0 = %d, N0 = %d, K0 = %d, K1 = %d\n", M0, N0, K0, K1);
    iree_uk_mmt4d_tile_s8s8s32_8x8x8_ime(
        lhs, rhs, got, M0, N0, K0, flags, K1);

    // Run scalar reference.
    mmt4d_reference_s8s8s32(
        lhs, rhs, want, M0, N0, K0, K1, /*accumulate=*/(flags & 1));

    int ok = compare_s32(got, want, M0, N0, label);
    // if (ok) printf("[PASS] %s\n", label);
    return ok;
}

static double get_time_diff(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) + 
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

int main(void)
{
    int M0 = IME_M0;
    int N0 = IME_N0;
    const int K0 = IME_K0;
    printf("m0 = %d, n0 = %d, k0 = %d \n", M0, N0, K0);

    int total = 0, passed = 0;
    struct timespec start, end;

    // Allocate panel buffers large enough for K1_max=8.
    const int K1_MAX = 8;
    int8_t  lhs_buf[K1_MAX * M0 * K0];
    int8_t  rhs_buf[K1_MAX * N0 * K0];
    // int32_t c_init [M0 * N0];

    // -----------------------------------------------------------------------
    // 1. Overwrite mode (flags=0), increasing K1, random data
    // -----------------------------------------------------------------------
    // {
    //     unsigned seed = 42;
    //     int K1s[] = {1, 2, 4, 8};
    //     for (int i = 0; i < (int)(sizeof K1s / sizeof K1s[0]); i++) {
    //         int K1 = K1s[i];
    //         fill_random_s8(lhs_buf, (size_t)(K1 * M0 * K0), &seed);
    //         fill_random_s8(rhs_buf, (size_t)(K1 * N0 * K0), &seed);

    //         char label[64];
    //         snprintf(label, sizeof label,
    //             "overwrite_random K1=%d", K1);
    //         clock_gettime(CLOCK_MONOTONIC, &start);
    //         passed += run_test(label, lhs_buf, rhs_buf, NULL, K1, 0);
    //         clock_gettime(CLOCK_MONOTONIC, &end);
    //         total++;
    //         printf("Kernel runtime: %f seconds\n\n", get_time_diff(start, end));

    //     }
    // }

    {
        unsigned seed = 42;
        int K1s[] = {1, 2, 4, 8};
        for (int i = 0; i < (int)(sizeof K1s / sizeof K1s[0]); i++) {
            int K1 = K1s[i];
            fill_random_s8(lhs_buf, (size_t)(K1 * M0 * K0), &seed);
            printf("\n\n");
            fill_random_s8(rhs_buf, (size_t)(K1 * N0 * K0), &seed);
            printf("\n");

            char label[64];
            snprintf(label, sizeof label,
                "overwrite_random K1=%d", K1);
            clock_gettime(CLOCK_MONOTONIC, &start);
            passed += run_test(label, lhs_buf, rhs_buf, NULL, M0, N0, K0, K1, 0);
            clock_gettime(CLOCK_MONOTONIC, &end);
            total++;
            printf("Kernel runtime: %f seconds\n", get_time_diff(start, end));

        }
    }

    M0 = 4;
    N0 = 4;
    printf("m0 = %d, n0 = %d, k0 = %d \n", M0, N0, K0);

    // Allocate panel buffers large enough for K1_max=8.
    int8_t lhs_buf4[K1_MAX * M0 * K0];
    int8_t rhs_buf4[K1_MAX * N0 * K0];

    {
        unsigned seed = 42;
        int K1s[] = {1, 2, 4, 8};
        for (int i = 0; i < (int)(sizeof K1s / sizeof K1s[0]); i++) {
            int K1 = K1s[i];
            fill_random_s8(lhs_buf4, (size_t)(K1 * M0 * K0), &seed);
            printf("\n\n");
            fill_random_s8(rhs_buf4, (size_t)(K1 * N0 * K0), &seed);
            printf("\n");

            char label[64];
            snprintf(label, sizeof label,
                "overwrite_random K1=%d", K1);
            clock_gettime(CLOCK_MONOTONIC, &start);
            passed += run_test(label, lhs_buf4, rhs_buf4, NULL, M0, N0, K0, K1, 0);
            clock_gettime(CLOCK_MONOTONIC, &end);
            total++;
            printf("Kernel runtime: %f seconds\n\n", get_time_diff(start, end));

        }
    }
   
    // Summary
    // -----------------------------------------------------------------------
    printf("\n%d / %d tests passed.\n", passed, total);
    
    return 1;
}
