#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// v1 is the faster option
void ime_microkernel_8x8x8_v1(const int8_t *lhs, const int8_t *rhs, int32_t *dst, int32_t K1) {asm volatile (
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
void ime_microkernel_8x8x8_v2(const int8_t *lhs, const int8_t *rhs, int32_t *dst, int32_t K1) {
    asm volatile (
        "li      t0, 32\n\t"
        "vsetvli zero, t0, e8, m1, ta, ma\n\t"

        "vxor.vv v0,  v0,  v0\n\t"
        "vxor.vv v1,  v1,  v1\n\t"
        "vxor.vv v4,  v4,  v4\n\t"
        "vxor.vv v5,  v5,  v5\n\t"
        "vxor.vv v12, v12, v12\n\t"
        "vxor.vv v13, v13, v13\n\t"
        "vxor.vv v20, v20, v20\n\t"
        "vxor.vv v21, v21, v21\n\t"

        "beqz    %3, 2f\n\t"
        "1:\n\t"
        "vle8.v  v16, (%1)\n\t"  "addi %1, %1, 32\n\t"
        "vle8.v  v17, (%1)\n\t"  "addi %1, %1, 32\n\t"
        "vle8.v  v18, (%2)\n\t"  "addi %2, %2, 32\n\t"
        "vle8.v  v19, (%2)\n\t"  "addi %2, %2, 32\n\t"

        "smt.vmadot v0,  v16, v18\n\t"
        "smt.vmadot v4,  v16, v19\n\t"
        "smt.vmadot v12, v17, v18\n\t"
        "smt.vmadot v20, v17, v19\n\t"

        "addi %3, %3, -1\n\t"
        "bnez %3, 1b\n\t"

        "2:\n\t"
        "vsetivli zero, 4, e32, m1, ta, ma\n\t"

        /*
         * 8 vslidedown + 16 vse32. No shuffles, no merges, no pointer math.
         * Each vslidedown shifts upper 4 elements into a temp for the odd row.
         * Offsets: row r starts at dst + r*32. Col half: +0 or +16.
         */

        /* rows 0-3, left (cols 0-3) */
        "vse32.v v0,  (%0)\n\t"                        /* row 0 */
        "vslidedown.vi v22, v0,  4\n\t"
        "addi    t0, %0,  32\n\t"  "vse32.v v22, (t0)\n\t"  /* row 1 */
        "addi    t0, %0,  64\n\t"  "vse32.v v1,  (t0)\n\t"  /* row 2 */
        "vslidedown.vi v22, v1,  4\n\t"
        "addi    t0, %0,  96\n\t"  "vse32.v v22, (t0)\n\t"  /* row 3 */

        /* rows 0-3, right (cols 4-7) */
        "addi    t0, %0,  16\n\t"  "vse32.v v4,  (t0)\n\t"  /* row 0 */
        "vslidedown.vi v22, v4,  4\n\t"
        "addi    t0, %0,  48\n\t"  "vse32.v v22, (t0)\n\t"  /* row 1 */
        "addi    t0, %0,  80\n\t"  "vse32.v v5,  (t0)\n\t"  /* row 2 */
        "vslidedown.vi v22, v5,  4\n\t"
        "addi    t0, %0, 112\n\t"  "vse32.v v22, (t0)\n\t"  /* row 3 */

        /* rows 4-7, left (cols 0-3) */
        "addi    t0, %0, 128\n\t"  "vse32.v v12, (t0)\n\t"  /* row 4 */
        "vslidedown.vi v22, v12, 4\n\t"
        "addi    t0, %0, 160\n\t"  "vse32.v v22, (t0)\n\t"  /* row 5 */
        "addi    t0, %0, 192\n\t"  "vse32.v v13, (t0)\n\t"  /* row 6 */
        "vslidedown.vi v22, v13, 4\n\t"
        "addi    t0, %0, 224\n\t"  "vse32.v v22, (t0)\n\t"  /* row 7 */

        /* rows 4-7, right (cols 4-7) */
        "addi    t0, %0, 144\n\t"  "vse32.v v20, (t0)\n\t"  /* row 4 */
        "vslidedown.vi v22, v20, 4\n\t"
        "addi    t0, %0, 176\n\t"  "vse32.v v22, (t0)\n\t"  /* row 5 */
        "addi    t0, %0, 208\n\t"  "vse32.v v21, (t0)\n\t"  /* row 6 */
        "vslidedown.vi v22, v21, 4\n\t"
        "addi    t0, %0, 240\n\t"  "vse32.v v22, (t0)\n\t"  /* row 7 */

        : "+r"(dst), "+r"(lhs), "+r"(rhs), "+r"(K1)
        :
        : "v0","v1","v4","v5","v12","v13","v16","v17",
          "v18","v19","v20","v21","v22","t0","cc","memory"
    );
}

static double get_time_diff(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) + 
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

int main() {
    const int32_t K1 = 1;
    const int32_t M0 = 8, N0 = 8, K0 = 8;
    const int iterations = 1000000;

    size_t lhs_bytes = K1 * M0 * K0 * sizeof(int8_t);
    size_t rhs_bytes = K1 * N0 * K0 * sizeof(int8_t);
    size_t dst_bytes = M0 * N0 * sizeof(int32_t);

    int8_t *lhs = (int8_t *)aligned_alloc(32, lhs_bytes);
    int8_t *rhs = (int8_t *)aligned_alloc(32, rhs_bytes);
    int32_t *dst = (int32_t *)aligned_alloc(32, dst_bytes);

    struct timespec start, end;

    // Benchmark Kernel V1
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        ime_microkernel_8x8x8_v1(lhs, rhs, dst, K1);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Kernel V1 Time: %f seconds\n", get_time_diff(start, end));

    // Benchmark Kernel V2
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        ime_microkernel_8x8x8_v2(lhs, rhs, dst, K1);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Kernel V2 Time: %f seconds\n", get_time_diff(start, end));

    free(lhs); free(rhs); free(dst);
    return 0;
}