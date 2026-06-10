#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Tile dimensions for SpaceMiT X60 vmadot (int8 -> int32)
#define M0 4
#define N0 4
#define K0 8

// Scalar reference implementation for a single tile
void scalar_mmt4d_tile_s8s8s32(const int8_t* A, const int8_t* B, int32_t* C, int accumulate) {
    for (int m = 0; m < M0; ++m) {
        for (int n = 0; n < N0; ++n) {
            int32_t acc = accumulate ? C[m * N0 + n] : 0;
            for (int k = 0; k < K0; ++k) {
                // A is row-major (M0 x K0)
                // B is provided transposed (N0 x K0) as per X60 specification
                acc += (int32_t)A[m * K0 + k] * (int32_t)B[n * K0 + k];
            }
            C[m * N0 + n] = acc;
        }
    }
}

// Hardware implementation using SpaceMiT IME (vmadot) inline assembly.
//
// vmadot operand/layout contract on X60 (VLEN=256, SEW=8 selects the 4x4x8 unit):
//   A (LHS) : (M0,K0) row-major, K-contiguous -> 4x8 int8 = 32 B = one vreg (v2)
//   B (RHS) : (N0,K0) row-major (already transposed), K-contiguous -> 32 B (v4)
//   C (ACC) : (M0,N0) int32 in the even-aligned 2-register group {v6,v7} = 64 B
//
// The MAC is issued via the named XSMTVDot mnemonic `smt.vmadot v6, v2, v4`, which
// requires assembling with -march=...xsmtvdot. It is bit-identical to the raw
// CUSTOM_1 form (verified via assemble->objdump round-trip):
//   smt.vmadot v6, v2, v4  ==  0xe241332b  ==  .insn r 0x2b, 3, 0x71, v6, v2, v4
// If the on-device system toolchain predates XSMTVDot, fall back to that raw .insn.
void hardware_vmadot_tile_s8s8s32(const int8_t* A, const int8_t* B, int32_t* C, int accumulate) {
    asm volatile (
        // Select the 4x4x8 int8 MAC unit and load the two 32-byte operand tiles.
        "li t0, 32\n\t"
        "vsetvli zero, t0, e8, m1, ta, ma\n\t"
        "vle8.v v2, (%0)\n\t"                       // A (LHS)  -> v2
        "vle8.v v4, (%1)\n\t"                       // B (RHS)  -> v4

        // Prepare the int32 accumulator pair {v6,v7} as 16 i32 elements.
        "li t0, 16\n\t"
        "vsetvli zero, t0, e32, m2, ta, ma\n\t"
        "bnez %3, 1f\n\t"
        "vmv.v.i v6, 0\n\t"                         // zero {v6,v7}
        "j 2f\n\t"
        "1:\n\t"
        "vle32.v v6, (%2)\n\t"                      // accumulate: load existing C into {v6,v7}
        "2:\n\t"

        // Re-select the 4x4x8 MAC unit and issue vmadot.
        "li t0, 32\n\t"
        "vsetvli zero, t0, e8, m1, ta, ma\n\t"
        "smt.vmadot v6, v2, v4\n\t"

        // View the accumulator as 16 i32 and store the 4x4 tile (64 B).
        "li t0, 16\n\t"
        "vsetvli zero, t0, e32, m2, ta, ma\n\t"
        "vse32.v v6, (%2)\n\t"
        :
        : "r"(A), "r"(B), "r"(C), "r"(accumulate)
        : "t0", "v2", "v4", "v6", "v7", "memory"
    );
}

static int check_tile(const char* label, const int32_t* ref, const int32_t* hw) {
    if (memcmp(ref, hw, sizeof(int32_t) * M0 * N0) == 0) {
        printf("PASS [%s]: int8 vmadot is bit-exact with scalar reference.\n", label);
        return 0;
    }
    printf("FAIL [%s]: mismatch between hardware and scalar reference.\n", label);
    for (int m = 0; m < M0; ++m) {
        for (int n = 0; n < N0; ++n) {
            int idx = m * N0 + n;
            const char* mark = (ref[idx] != hw[idx]) ? "  <-- diff" : "";
            printf("  C[%d][%d] ref=%-12d hw=%-12d%s\n", m, n, ref[idx], hw[idx], mark);
        }
    }
    return 1;
}

int main() {
    int8_t A[M0 * K0];
    int8_t B[N0 * K0];
    int32_t C_ref[M0 * N0];
    int32_t C_hw[M0 * N0];
    int failures = 0;

    // Initialize with varied values to exercise sign-extension boundaries.
    for (int i = 0; i < M0 * K0; ++i) A[i] = (i % 2 == 0) ? 127 : -128;
    for (int i = 0; i < N0 * K0; ++i) B[i] = (i % 3 == 0) ? 2 : -3;

    // Test 1: non-accumulating (C := A * B^T).
    memset(C_ref, 0, sizeof(C_ref));
    memset(C_hw, 0, sizeof(C_hw));
    scalar_mmt4d_tile_s8s8s32(A, B, C_ref, 0);
    hardware_vmadot_tile_s8s8s32(A, B, C_hw, 0);
    failures += check_tile("no-accumulate", C_ref, C_hw);

    // Test 2: accumulating (C += A * B^T) onto a non-zero starting tile.
    for (int i = 0; i < M0 * N0; ++i) C_ref[i] = C_hw[i] = (i - 8) * 1000;
    scalar_mmt4d_tile_s8s8s32(A, B, C_ref, 1);
    hardware_vmadot_tile_s8s8s32(A, B, C_hw, 1);
    failures += check_tile("accumulate", C_ref, C_hw);

    if (failures == 0) {
        printf("\nSUCCESS: int8 vmadot bit-exact in both modes.\n");
        return 0;
    }
    printf("\nFAILURE: %d mode(s) mismatched. If operands look transposed, swap A/B (v2/v4).\n", failures);
    return 1;
}