#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void ime_microkernel_8x8x8(const int8_t *lhs, const int8_t *rhs, int32_t *dst, int32_t K1) {
    asm volatile (
        // 1. Configure Engine for 256-bit VLEN compute (VL=32, SEW=8, LMUL=1)
        "li    t0, 32\n\t"
        "vsetvli zero, t0, e8, m1, ta, ma\n\t"

        // Clear primary anchor registers (Note: depending on the hardware, 
        // you may also need to clear v1, v5, v13, v21 if they don't auto-zero)
        "vxor.vv v0, v0, v0\n\t"
        "vxor.vv v4, v4, v4\n\t"
        "vxor.vv v12, v12, v12\n\t"
        "vxor.vv v20, v20, v20\n\t"

        "beqz    %3, 2f\n\t"

        "1:\n\t"
        // Load LHS matrix panels
        "vle8.v  v16, (%1)\n\t"
        "addi    %1, %1, 32\n\t"
        "vle8.v  v17, (%1)\n\t"
        "addi    %1, %1, 32\n\t"

        // Load RHS matrix panels
        "vle8.v  v18, (%2)\n\t"
        "addi    %2, %2, 32\n\t"
        "vle8.v  v19, (%2)\n\t"
        "addi    %2, %2, 32\n\t"

        // Compute 4x4 sub-tiles using non-overlapping strides of 4
        "smt.vmadot v0,  v16, v18\n\t"   // Left-Top     (Rows 0,1 in v0; Rows 2,3 in v1)
        "smt.vmadot v4,  v16, v19\n\t"   // Right-Top    (Rows 0,1 in v4; Rows 2,3 in v5)
        "smt.vmadot v12, v17, v18\n\t"   // Left-Bottom  (Rows 4,5 in v12; Rows 6,7 in v13)
        "smt.vmadot v20, v17, v19\n\t"   // Right-Bottom (Rows 4,5 in v20; Rows 6,7 in v21)

        "addi    %3, %3, -1\n\t"
        "bnez    %3, 1b\n\t"

        "2:\n\t"
        // 2. Store Phase: 256-bit merge-and-store optimization
        "vsetivli zero, 8, e32, m1, ta, ma\n\t" // VL=8 (Full row of 8 int32s)

        // --- Rows 0 and 1 ---
        "vslidedown.vi v24, v0, 4\n\t"    // Extract Row 1 Left
        "vslidedown.vi v25, v4, 4\n\t"    // Extract Row 1 Right
        "vslideup.vi   v0, v4, 4\n\t"     // Merge Row 0 Right into v0
        "vslideup.vi   v24, v25, 4\n\t"   // Merge Row 1 Right into v24
        "vse32.v       v0, (%0)\n\t"      // Store complete Row 0 (32 bytes)
        "addi          %0, %0, 32\n\t"
        "vse32.v       v24, (%0)\n\t"     // Store complete Row 1
        "addi          %0, %0, 32\n\t"

        // --- Rows 2 and 3 ---
        "vslidedown.vi v24, v1, 4\n\t"    // Extract Row 3 Left
        "vslidedown.vi v25, v5, 4\n\t"    // Extract Row 3 Right
        "vslideup.vi   v1, v5, 4\n\t"     // Merge Row 2 Right into v1
        "vslideup.vi   v24, v25, 4\n\t"   // Merge Row 3 Right into v24
        "vse32.v       v1, (%0)\n\t"      // Store complete Row 2
        "addi          %0, %0, 32\n\t"
        "vse32.v       v24, (%0)\n\t"     // Store complete Row 3
        "addi          %0, %0, 32\n\t"

        // --- Rows 4 and 5 ---
        "vslidedown.vi v24, v12, 4\n\t"   // Extract Row 5 Left
        "vslidedown.vi v25, v20, 4\n\t"   // Extract Row 5 Right
        "vslideup.vi   v12, v20, 4\n\t"   // Merge Row 4 Right into v12
        "vslideup.vi   v24, v25, 4\n\t"   // Merge Row 5 Right into v24
        "vse32.v       v12, (%0)\n\t"     // Store complete Row 4
        "addi          %0, %0, 32\n\t"
        "vse32.v       v24, (%0)\n\t"     // Store complete Row 5
        "addi          %0, %0, 32\n\t"

        // --- Rows 6 and 7 ---
        "vslidedown.vi v24, v13, 4\n\t"   // Extract Row 7 Left
        "vslidedown.vi v25, v21, 4\n\t"   // Extract Row 7 Right
        "vslideup.vi   v13, v21, 4\n\t"   // Merge Row 6 Right into v13
        "vslideup.vi   v24, v25, 4\n\t"   // Merge Row 7 Right into v24
        "vse32.v       v13, (%0)\n\t"     // Store complete Row 6
        "addi          %0, %0, 32\n\t"
        "vse32.v       v24, (%0)\n\t"     // Store complete Row 7

        : "+r"(dst), "+r"(lhs), "+r"(rhs), "+r"(K1)
        :
        : "v0", "v1", "v4", "v5", "v12", "v13", "v16", "v17", "v18", "v19", "v20", "v21", 
          "v24", "v25", "t0", "cc", "memory"
    );
}

int main() {
    const int32_t K1 = 1;
    const int32_t M0 = 8, N0 = 8, K0 = 8;

    size_t lhs_bytes = K1 * M0 * K0 * sizeof(int8_t);
    size_t rhs_bytes = K1 * N0 * K0 * sizeof(int8_t);
    size_t dst_bytes = M0 * N0 * sizeof(int32_t);

    int8_t *lhs = (int8_t *)malloc(lhs_bytes);
    int8_t *rhs = (int8_t *)malloc(rhs_bytes);
    int32_t *dst = (int32_t *)malloc(dst_bytes);

    // Initialize LHS: Sequential matrix sequence [1..64]
    for (size_t i = 0; i < K1 * M0 * K0; i++) {
        lhs[i] = (int8_t)(i + 1);
    }

    // Initialize RHS: 8x8 Identity Matrix
    memset(rhs, 0, rhs_bytes);
    for (int r = 0; r < N0; r++) {
        rhs[r * N0 + r] = 1; 
    }

    memset(dst, 0, dst_bytes);

    ime_microkernel_8x8x8(lhs, rhs, dst, K1);

    printf("=== MMT4D LINEAR RESOLVED OUTPUT [M0=8, N0=8] ===\n");
    for (int m = 0; m < M0; m++) {
        printf("  Row %d: ", m);
        for (int n = 0; n < N0; n++) {
            printf("%6d ", dst[m * N0 + n]);
        }
        printf("\n");
    }

    free(lhs); free(rhs); free(dst);
    return 0;
}