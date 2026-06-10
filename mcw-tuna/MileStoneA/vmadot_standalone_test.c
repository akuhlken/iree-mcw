#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Tile dimensions for SpaceMiT X60 vmadot (VLEN=256).
// SEW=8  -> 4x4x8 int8  -> int32 (32 B operand tiles)
// SEW=16 -> 4x4x4 int16 -> int32 (32 B operand tiles)
#define M0_S8 4
#define N0_S8 4
#define K0_S8 8

#define M0_S16 4
#define N0_S16 4
#define K0_S16 4

typedef enum {
    TEST_INT8 = 1,
    TEST_INT16 = 2,
    TEST_BOTH = TEST_INT8 | TEST_INT16,
} test_mask_t;

static void scalar_mmt4d_tile_s8s8s32(const int8_t* A, const int8_t* B, int32_t* C,
                                      int accumulate) {
    for (int m = 0; m < M0_S8; ++m) {
        for (int n = 0; n < N0_S8; ++n) {
            int32_t acc = accumulate ? C[m * N0_S8 + n] : 0;
            for (int k = 0; k < K0_S8; ++k) {
                acc += (int32_t)A[m * K0_S8 + k] * (int32_t)B[n * K0_S8 + k];
            }
            C[m * N0_S8 + n] = acc;
        }
    }
}

static void scalar_mmt4d_tile_s16s16s32(const int16_t* A, const int16_t* B, int32_t* C,
                                        int accumulate) {
    for (int m = 0; m < M0_S16; ++m) {
        for (int n = 0; n < N0_S16; ++n) {
            int32_t acc = accumulate ? C[m * N0_S16 + n] : 0;
            for (int k = 0; k < K0_S16; ++k) {
                acc += (int32_t)A[m * K0_S16 + k] * (int32_t)B[n * K0_S16 + k];
            }
            C[m * N0_S16 + n] = acc;
        }
    }
}

// Hardware implementation using SpaceMiT IME (vmadot) inline assembly.
//
// vmadot operand/layout contract on X60 (VLEN=256):
//   SEW=8  (4x4x8 int8):
//     A (LHS) : (M0,K0) row-major -> 4x8 int8 = 32 B (v2)
//     B (RHS) : (N0,K0) row-major (transposed) -> 32 B (v4)
//   SEW=16 (4x4x4 int16):
//     A (LHS) : (M0,K0) row-major -> 4x4 int16 = 32 B (v2)
//     B (RHS) : (N0,K0) row-major (transposed) -> 32 B (v4)
//   C (ACC) : (M0,N0) int32 in even-aligned pair {v6,v7} = 64 B
//
// The MAC is issued via `smt.vmadot v6, v2, v4` (-march=...xsmtvdot). SEW at
// issue time selects int8 vs int16. Raw fallback:
//   smt.vmadot v6, v2, v4  ==  0xe241332b  ==  .insn r 0x2b, 3, 0x71, v6, v2, v4
static void hardware_vmadot_tile_s8s8s32(const int8_t* A, const int8_t* B, int32_t* C,
                                         int accumulate) {
    asm volatile(
        "li t0, 32\n\t"
        "vsetvli zero, t0, e8, m1, ta, ma\n\t"
        "vle8.v v2, (%0)\n\t"
        "vle8.v v4, (%1)\n\t"

        "li t0, 16\n\t"
        "vsetvli zero, t0, e32, m2, ta, ma\n\t"
        "bnez %3, 1f\n\t"
        "vmv.v.i v6, 0\n\t"
        "j 2f\n\t"
        "1:\n\t"
        "vle32.v v6, (%2)\n\t"
        "2:\n\t"

        "li t0, 32\n\t"
        "vsetvli zero, t0, e8, m1, ta, ma\n\t"
        "smt.vmadot v6, v2, v4\n\t"

        "li t0, 16\n\t"
        "vsetvli zero, t0, e32, m2, ta, ma\n\t"
        "vse32.v v6, (%2)\n\t"
        :
        : "r"(A), "r"(B), "r"(C), "r"(accumulate)
        : "t0", "v2", "v4", "v6", "v7", "memory");
}

static void hardware_vmadot_tile_s16s16s32(const int16_t* A, const int16_t* B, int32_t* C,
                                           int accumulate) {
    asm volatile(
        "li t0, 16\n\t"
        "vsetvli zero, t0, e16, m1, ta, ma\n\t"
        "vle16.v v2, (%0)\n\t"
        "vle16.v v4, (%1)\n\t"

        "li t0, 16\n\t"
        "vsetvli zero, t0, e32, m2, ta, ma\n\t"
        "bnez %3, 1f\n\t"
        "vmv.v.i v6, 0\n\t"
        "j 2f\n\t"
        "1:\n\t"
        "vle32.v v6, (%2)\n\t"
        "2:\n\t"

        "li t0, 16\n\t"
        "vsetvli zero, t0, e16, m1, ta, ma\n\t"
        "smt.vmadot v6, v2, v4\n\t"

        "li t0, 16\n\t"
        "vsetvli zero, t0, e32, m2, ta, ma\n\t"
        "vse32.v v6, (%2)\n\t"
        :
        : "r"(A), "r"(B), "r"(C), "r"(accumulate)
        : "t0", "v2", "v4", "v6", "v7", "memory");
}

static int check_tile(const char* dtype, const char* label, int m0, int n0,
                      const int32_t* ref, const int32_t* hw) {
    size_t tile_bytes = (size_t)m0 * (size_t)n0 * sizeof(int32_t);
    if (memcmp(ref, hw, tile_bytes) == 0) {
        printf("PASS [%s/%s]: vmadot is bit-exact with scalar reference.\n", dtype, label);
        return 0;
    }
    printf("FAIL [%s/%s]: mismatch between hardware and scalar reference.\n", dtype, label);
    for (int m = 0; m < m0; ++m) {
        for (int n = 0; n < n0; ++n) {
            int idx = m * n0 + n;
            const char* mark = (ref[idx] != hw[idx]) ? "  <-- diff" : "";
            printf("  C[%d][%d] ref=%-12d hw=%-12d%s\n", m, n, ref[idx], hw[idx], mark);
        }
    }
    return 1;
}

static int run_int8_tests(void) {
    int8_t A[M0_S8 * K0_S8];
    int8_t B[N0_S8 * K0_S8];
    int32_t C_ref[M0_S8 * N0_S8];
    int32_t C_hw[M0_S8 * N0_S8];
    int failures = 0;

    printf("=== int8 vmadot (4x4x8 -> 4x4 i32) ===\n");

    for (int i = 0; i < M0_S8 * K0_S8; ++i) A[i] = (i % 2 == 0) ? 127 : -128;
    for (int i = 0; i < N0_S8 * K0_S8; ++i) B[i] = (i % 3 == 0) ? 2 : -3;

    memset(C_ref, 0, sizeof(C_ref));
    memset(C_hw, 0, sizeof(C_hw));
    scalar_mmt4d_tile_s8s8s32(A, B, C_ref, 0);
    hardware_vmadot_tile_s8s8s32(A, B, C_hw, 0);
    failures += check_tile("int8", "no-accumulate", M0_S8, N0_S8, C_ref, C_hw);

    for (int i = 0; i < M0_S8 * N0_S8; ++i) C_ref[i] = C_hw[i] = (i - 8) * 1000;
    scalar_mmt4d_tile_s8s8s32(A, B, C_ref, 1);
    hardware_vmadot_tile_s8s8s32(A, B, C_hw, 1);
    failures += check_tile("int8", "accumulate", M0_S8, N0_S8, C_ref, C_hw);

    return failures;
}

static int run_int16_tests(void) {
    int16_t A[M0_S16 * K0_S16];
    int16_t B[N0_S16 * K0_S16];
    int32_t C_ref[M0_S16 * N0_S16];
    int32_t C_hw[M0_S16 * N0_S16];
    int failures = 0;

    printf("=== int16 vmadot (4x4x4 -> 4x4 i32) ===\n");

    for (int i = 0; i < M0_S16 * K0_S16; ++i) A[i] = (i % 2 == 0) ? 32767 : -32768;
    for (int i = 0; i < N0_S16 * K0_S16; ++i) B[i] = (i % 3 == 0) ? 2 : -3;

    memset(C_ref, 0, sizeof(C_ref));
    memset(C_hw, 0, sizeof(C_hw));
    scalar_mmt4d_tile_s16s16s32(A, B, C_ref, 0);
    hardware_vmadot_tile_s16s16s32(A, B, C_hw, 0);
    failures += check_tile("int16", "no-accumulate", M0_S16, N0_S16, C_ref, C_hw);

    for (int i = 0; i < M0_S16 * N0_S16; ++i) C_ref[i] = C_hw[i] = (i - 8) * 100000;
    scalar_mmt4d_tile_s16s16s32(A, B, C_ref, 1);
    hardware_vmadot_tile_s16s16s32(A, B, C_hw, 1);
    failures += check_tile("int16", "accumulate", M0_S16, N0_S16, C_ref, C_hw);

    return failures;
}

static test_mask_t parse_test_mask(int argc, char** argv) {
    if (argc < 2) {
        return TEST_BOTH;
    }

    if (strcmp(argv[1], "int8") == 0 || strcmp(argv[1], "s8") == 0) {
        return TEST_INT8;
    }
    if (strcmp(argv[1], "int16") == 0 || strcmp(argv[1], "s16") == 0) {
        return TEST_INT16;
    }
    if (strcmp(argv[1], "both") == 0 || strcmp(argv[1], "all") == 0) {
        return TEST_BOTH;
    }

    fprintf(stderr,
            "Usage: %s [int8|int16|both]\n"
            "  int8 / s8    - run int8 vmadot tests only\n"
            "  int16 / s16  - run int16 vmadot tests only\n"
            "  both / all   - run both (default)\n",
            argv[0]);
    exit(2);
}

int main(int argc, char** argv) {
    test_mask_t mask = parse_test_mask(argc, argv);
    int failures = 0;
    int modes_run = 0;

    if (mask & TEST_INT8) {
        failures += run_int8_tests();
        modes_run++;
        if (mask & TEST_INT16) {
            printf("\n");
        }
    }
    if (mask & TEST_INT16) {
        failures += run_int16_tests();
        modes_run++;
    }

    if (failures == 0) {
        printf("\nSUCCESS: %d dtype mode(s) bit-exact in both accumulate modes.\n", modes_run);
        return 0;
    }
    printf("\nFAILURE: %d mode(s) mismatched. If operands look transposed, swap A/B (v2/v4).\n",
           failures);
    return 1;
}
