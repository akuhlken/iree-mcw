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
    const char    *label,
    const int8_t  *lhs,    // [K1][M0][K0]
    const int8_t  *rhs,    // [K1][N0][K0]
    const int32_t *c_init, // [M0][N0] initial accumulator (NULL = zero)
    int32_t        K1,
    int32_t        flags)
{
    const int M0 = IME_M0;
    const int N0 = IME_N0;
    const int K0 = IME_K0;

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
    iree_uk_mmt4d_tile_s8s8s32_8x8x8_ime(
        lhs, rhs, got, M0, K0, flags, K1);

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
    const int M0 = IME_M0;
    const int N0 = IME_N0;
    const int K0 = IME_K0;
    printf("m0 = %d, n0 = %d, k0 = %d \n", M0, N0, K0);

    int total = 0, passed = 0;
    struct timespec start, end;

    // Allocate panel buffers large enough for K1_max=8.
    const int K1_MAX = 8;
    int8_t  lhs_buf[K1_MAX * M0 * K0];
    int8_t  rhs_buf[K1_MAX * N0 * K0];
    int32_t c_init [M0 * N0];

    // -----------------------------------------------------------------------
    // 1. Overwrite mode (flags=0), increasing K1, random data
    // -----------------------------------------------------------------------
    {
        unsigned seed = 42;
        int K1s[] = {1, 2, 4, 8};
        for (int i = 0; i < (int)(sizeof K1s / sizeof K1s[0]); i++) {
            int K1 = K1s[i];
            fill_random_s8(lhs_buf, (size_t)(K1 * M0 * K0), &seed);
            fill_random_s8(rhs_buf, (size_t)(K1 * N0 * K0), &seed);

            char label[64];
            snprintf(label, sizeof label,
                "overwrite_random K1=%d", K1);
            clock_gettime(CLOCK_MONOTONIC, &start);
            passed += run_test(label, lhs_buf, rhs_buf, NULL, K1, 0);
            clock_gettime(CLOCK_MONOTONIC, &end);
            total++;
            printf("Kernel runtime: %f seconds\n\n", get_time_diff(start, end));

        }
    }

    // -----------------------------------------------------------------------
    // 2. Accumulate mode (flags=1), K1=2, random data + random initial C
    // -----------------------------------------------------------------------
    {
        unsigned seed = 1337;
        fill_random_s8(lhs_buf, (size_t)(2 * M0 * K0), &seed);
        fill_random_s8(rhs_buf, (size_t)(2 * N0 * K0), &seed);
        // Random initial accumulator values (stay small to avoid i32 overflow)
        for (int i = 0; i < M0 * N0; i++) {
            c_init[i] = (int32_t)((int)(rand_r(&seed) % 512) - 256);
        }

        clock_gettime(CLOCK_MONOTONIC, &start);
        passed += run_test("accumulate_random K1=2",
            lhs_buf, rhs_buf, c_init, 2, /*flags=*/1);
        clock_gettime(CLOCK_MONOTONIC, &end);
        total++;
        printf("Kernel runtime: %f seconds\n\n", get_time_diff(start, end));

    }

    // -----------------------------------------------------------------------
    // 3. Edge: all-zero A  →  C should remain zero
    // -----------------------------------------------------------------------
    {
        fill_constant_s8(lhs_buf, M0 * K0, 0);
        fill_constant_s8(rhs_buf, N0 * K0, 7);
        clock_gettime(CLOCK_MONOTONIC, &start);
        passed += run_test("zero_A K1=1",
            lhs_buf, rhs_buf, NULL, 1, 0);
        clock_gettime(CLOCK_MONOTONIC, &end);
        total++;
        printf("Kernel runtime: %f seconds\n\n", get_time_diff(start, end));

    }

    // -----------------------------------------------------------------------
    // 4. Edge: all-zero B  →  C should remain zero
    // -----------------------------------------------------------------------
    {
        fill_constant_s8(lhs_buf, M0 * K0, 3);
        fill_constant_s8(rhs_buf, N0 * K0, 0);
        clock_gettime(CLOCK_MONOTONIC, &start);
        passed += run_test("zero_B K1=1",
            lhs_buf, rhs_buf, NULL, 1, 0);
        clock_gettime(CLOCK_MONOTONIC, &end);
        total++;
        printf("Kernel runtime: %f seconds\n\n", get_time_diff(start, end));

    }

    // -----------------------------------------------------------------------
    // 5. Edge: A=1, B=1  →  each C[m][n] = K0 = 8
    // -----------------------------------------------------------------------
    {
        fill_constant_s8(lhs_buf, M0 * K0, 1);
        fill_constant_s8(rhs_buf, N0 * K0, 1);
        clock_gettime(CLOCK_MONOTONIC, &start);
        passed += run_test("ones K1=1 (expect 8)",
            lhs_buf, rhs_buf, NULL, 1, 0);
        clock_gettime(CLOCK_MONOTONIC, &end);
        total++;
        printf("Kernel runtime: %f seconds\n\n", get_time_diff(start, end));
    }

    // -----------------------------------------------------------------------
    // 6. Edge: extreme values A=127, B=127  →  C[m][n] = 127*127*K0 = 129032
    //    Verifies no silent saturation or wrong widening.
    // -----------------------------------------------------------------------
    {
        fill_constant_s8(lhs_buf, M0 * K0, 127);
        fill_constant_s8(rhs_buf, N0 * K0, 127);
        clock_gettime(CLOCK_MONOTONIC, &start);
        passed += run_test("max_pos A=127 B=127 K1=1",
            lhs_buf, rhs_buf, NULL, 1, 0);
        clock_gettime(CLOCK_MONOTONIC, &end);
        total++;
        printf("Kernel runtime: %f seconds\n\n", get_time_diff(start, end));
    }

    // -----------------------------------------------------------------------
    // 7. Edge: extreme values A=-128, B=127 (signed × signed worst case)
    // -----------------------------------------------------------------------
    {
        fill_constant_s8(lhs_buf, M0 * K0, -128);
        fill_constant_s8(rhs_buf, N0 * K0,  127);
        clock_gettime(CLOCK_MONOTONIC, &start);
        passed += run_test("extreme neg A=-128 B=127 K1=1",
            lhs_buf, rhs_buf, NULL, 1, 0);
        clock_gettime(CLOCK_MONOTONIC, &end);
        total++;
        printf("Kernel runtime: %f seconds\n\n", get_time_diff(start, end));
    }

    // -----------------------------------------------------------------------
    // 8. Accumulate mode, K1=1, extreme values + non-zero C_init
    //    Checks that accumulation uses the full int32 range.
    // -----------------------------------------------------------------------
    {
        fill_constant_s8(lhs_buf, M0 * K0, 1);
        fill_constant_s8(rhs_buf, N0 * K0, 1);
        for (int i = 0; i < M0 * N0; i++) c_init[i] = 1000000;
        clock_gettime(CLOCK_MONOTONIC, &start);
        passed += run_test("accumulate_large_init K1=1",
            lhs_buf, rhs_buf, c_init, 1, /*flags=*/1);
        clock_gettime(CLOCK_MONOTONIC, &end);
        total++;
        printf("Kernel runtime: %f seconds\n\n", get_time_diff(start, end));
    }

    // -----------------------------------------------------------------------
    // 9. Checkerboard pattern: A alternates [1, -1, 1, -1, ...]
    //                          B alternates [1, -1, 1, -1, ...]
    //    Dot product of length K0=8 of +-1 sequences = 8 (all same sign sums)
    // -----------------------------------------------------------------------
    {
        for (int i = 0; i < M0 * K0; i++) lhs_buf[i] = (i % 2) ? -1 : 1;
        for (int i = 0; i < N0 * K0; i++) rhs_buf[i] = (i % 2) ? -1 : 1;
        clock_gettime(CLOCK_MONOTONIC, &start);
        passed += run_test("checkerboard K1=1",
            lhs_buf, rhs_buf, NULL, 1, 0);
        clock_gettime(CLOCK_MONOTONIC, &end);
        total++;
        printf("Kernel runtime: %f seconds\n\n", get_time_diff(start, end));
    }

    // -----------------------------------------------------------------------
    // 10. Stress: K1=8, random data, both accumulate modes
    // -----------------------------------------------------------------------
    {
        unsigned seed = 0xdeadbeef;
        fill_random_s8(lhs_buf, (size_t)(K1_MAX * M0 * K0), &seed);
        fill_random_s8(rhs_buf, (size_t)(K1_MAX * N0 * K0), &seed);
        clock_gettime(CLOCK_MONOTONIC, &start);
        passed += run_test("stress_overwrite K1=8",
            lhs_buf, rhs_buf, NULL, K1_MAX, 0);
        clock_gettime(CLOCK_MONOTONIC, &end);
        total++;
        printf("Kernel runtime: %f seconds\n\n", get_time_diff(start, end));

        for (int i = 0; i < M0 * N0; i++) c_init[i] = 42;
        clock_gettime(CLOCK_MONOTONIC, &start);
        passed += run_test("stress_accumulate K1=8",
            lhs_buf, rhs_buf, c_init, K1_MAX, 1);
        clock_gettime(CLOCK_MONOTONIC, &end);
        total++;
        printf("Kernel runtime: %f seconds\n\n", get_time_diff(start, end));
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    printf("\n%d / %d tests passed.\n", passed, total);
   

    // testing all possible combos for M0, N0, K0
    printf("flag, M0, N0, K0, K1, time\n");

    {    
        unsigned seed = 42;
        int flags[] = {0,1};
        int K1s[] = {8, 16, 32};
        int M0s[] = {4,8,16};
        int N0s[] = {4,8,16};
        // K0 should be the SEW, the size of the type.

        // loop thru flag options
        for (int f = 0; f < (int)(sizeof flags / sizeof flags[0]); f++) {
            // m0 loop
            for (int m = 0; m < (int)(sizeof M0s / sizeof M0s[0]); m++) {
                // n0 loop
                for (int n = 0; n < (int)(sizeof N0s / sizeof N0s[0]); n++) {
                    // k1 loop
                    for (int i = 0; i < (int)(sizeof K1s / sizeof K1s[0]); i++) {
                        int K1 = K1s[i];
                        int M0 = M0s[m];
                        int N0 = N0s[n];
                        int K0 = 8;
                        
                        // printf("trying: %d, %d, %d, %d, %d\n", flags[f], M0, N0, K0, K1);
                        
                        fill_random_s8(lhs_buf, (size_t)(K1 * M0 * K0), &seed);
                        fill_random_s8(rhs_buf, (size_t)(K1 * N0 * K0), &seed);

                        clock_gettime(CLOCK_MONOTONIC, &start);
                        passed += run_test(("overwrite_random K1"), lhs_buf, rhs_buf, NULL, K1, flags[f]);
                        clock_gettime(CLOCK_MONOTONIC, &end);
                        printf("%d, %d, %d, %d, %d, %f\n", flags[f], M0, N0, K0, K1, get_time_diff(start, end));
                    }
                    // printf("end n loop\n");
                }
                // printf("end m loop\n");
            }
            // printf("end f loop\n");
        } 
        // printf("end test\n");  
    }
    printf("done\n");
    return 1;
}
