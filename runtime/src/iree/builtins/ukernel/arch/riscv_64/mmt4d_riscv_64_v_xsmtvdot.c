// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <riscv_vector.h>

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_internal.h"

#define IME_M_ATOM  4
#define IME_N_ATOM  4
#define IME_K_ATOM  8
#define IME_K0      8

static void ime_gather_acc(const iree_uk_int32_t *out, vint32m4_t *scratch,
                           int MT, int NT, int N0)
{
    for (int mt = 0; mt < MT; mt++) {
        for (int nt = 0; nt < NT; nt++) {
            vint32m4_t *dst = scratch + (mt * NT + nt) * 16;
            const iree_uk_int32_t *src = out + (mt * IME_M_ATOM) * N0 + (nt * IME_N_ATOM);
            for (int r = 0; r < IME_M_ATOM; r++)
                for (int c = 0; c < IME_N_ATOM; c++)
                    dst[r * IME_N_ATOM + c] = src[r * N0 + c];
        }
    }
}

static void ime_scatter_acc(int32_t *out, const vint32m4_t *scratch,
                            int MT, int NT, int N0)
{
    for (int mt = 0; mt < MT; mt++) {
        for (int nt = 0; nt < NT; nt++) {
            const int32_t *src = scratch + (mt * NT + nt) * 16;
            int32_t *dst = out + (mt * IME_M_ATOM) * N0 + (nt * IME_N_ATOM);
            for (int r = 0; r < IME_M_ATOM; r++)
                for (int c = 0; c < IME_N_ATOM; c++)
                    dst[r * N0 + c] = src[r * IME_N_ATOM + c];
        }
    }
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s8s32_8x16x8_riscv_64_v(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {

    // confirm that M0 = 8, which is the only valid config for this ukernel.
    IREE_UK_ASSERT(M0 == 8);

    int N0 = params->N0;
    int K0 = params->K0;
    int K = params->K;
    int K1 = K / K0;

    const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
    const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
    iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;

    vint32m4_t acc0, acc1, acc2, acc3, acc4, acc5, acc6;

    // if m0 == 8 (always true):

    // MT, NT = number of tiles in each dimension
    // M0_T, N0_T = number of elements in each dimension (x0 * xT)
    enum { MT = 2, NT = 4, M0_T = 8, N0_T = 16 };
    vint32m4_t acc_scratch[MT * NT * 16];

    // populate the accumulators
    if (flags & 1) {

        // should replace this helper func call/build it in.
        // commented out attempt below failed.
        ime_gather_acc(out, acc_scratch, MT, NT, N0_T);
        // void* scratch = &acc_scratch;
        // for (int mt = 0; mt < MT; mt++) {
        //     for (int nt = 0; nt < NT; nt++) {
        //         vint32m4_t *dst = scratch + (mt * NT + nt) * 16;
        //         const iree_uk_int32_t *src = out + (mt * IME_M_ATOM) * N0_T + (nt * IME_N_ATOM);
        //         for (int r = 0; r < IME_M_ATOM; r++)
        //             for (int c = 0; c < IME_N_ATOM; c++)
        //                 dst[r * IME_N_ATOM + c] = src[r * N0_T + c];
        //     }
        // }

        __asm__ volatile(
            "vsetvli      t0, x0, e32, m2, ta, ma       \n\t"
            "vle32.v      v16, (%[s0])                   \n\t"
            "vle32.v      v18, (%[s1])                   \n\t"
            "vle32.v      v20, (%[s2])                   \n\t"
            "vle32.v      v22, (%[s3])                   \n\t"
            "vle32.v      v24, (%[s4])                   \n\t"
            "vle32.v      v26, (%[s5])                   \n\t"
            "vle32.v      v28, (%[s6])                   \n\t"
            "vle32.v      v30, (%[s7])                   \n\t"
            :
            : [s0] "r"(acc_scratch + 0),   [s1] "r"(acc_scratch + 16),
              [s2] "r"(acc_scratch + 32),  [s3] "r"(acc_scratch + 48),
              [s4] "r"(acc_scratch + 64),  [s5] "r"(acc_scratch + 80),
              [s6] "r"(acc_scratch + 96),  [s7] "r"(acc_scratch + 112)
            : "memory", "t0", "v16", "v17", "v18", "v19", "v20", "v21",
              "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
              "v30", "v31");
    } else {
        __asm__ volatile(
            "vsetvli      t0, x0, e32, m2, ta, ma       \n\t"
            "vmv.v.i      v16, 0                         \n\t"
            "vmv.v.i      v18, 0                         \n\t"
            "vmv.v.i      v20, 0                         \n\t"
            "vmv.v.i      v22, 0                         \n\t"
            "vmv.v.i      v24, 0                         \n\t"
            "vmv.v.i      v26, 0                         \n\t"
            "vmv.v.i      v28, 0                         \n\t"
            "vmv.v.i      v30, 0                         \n\t"
            :
            :
            : "t0", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
              "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    }

    for (int k1 = 0; k1 < K1; k1++) {
        const int8_t *A0 = lhs + k1 * (M0_T * IME_K0) + 0 * IME_M_ATOM * IME_K0;
        const int8_t *A1 = lhs + k1 * (M0_T * IME_K0) + 1 * IME_M_ATOM * IME_K0;
        const int8_t *B0 = rhs + k1 * (N0_T * IME_K0) + 0 * IME_N_ATOM * IME_K0;
        const int8_t *B1 = rhs + k1 * (N0_T * IME_K0) + 1 * IME_N_ATOM * IME_K0;
        const int8_t *B2 = rhs + k1 * (N0_T * IME_K0) + 2 * IME_N_ATOM * IME_K0;
        const int8_t *B3 = rhs + k1 * (N0_T * IME_K0) + 3 * IME_N_ATOM * IME_K0;

        __asm__ volatile(
            "vsetvli      t0, x0, e8, m1, ta, ma        \n\t"
            "vle8.v       v0, (%[A0])                    \n\t"
            "vle8.v       v2, (%[A1])                    \n\t"
            "vle8.v       v4, (%[B0])                    \n\t"
            "vle8.v       v6, (%[B1])                    \n\t"
            "vle8.v       v8, (%[B2])                    \n\t"
            "vle8.v       v10, (%[B3])                   \n\t"
            "smt.vmadot   v16, v0, v4                    \n\t"
            "smt.vmadot   v18, v0, v6                    \n\t"
            "smt.vmadot   v20, v0, v8                    \n\t"
            "smt.vmadot   v22, v0, v10                   \n\t"
            "smt.vmadot   v24, v2, v4                    \n\t"
            "smt.vmadot   v26, v2, v6                    \n\t"
            "smt.vmadot   v28, v2, v8                    \n\t"
            "smt.vmadot   v30, v2, v10                   \n\t"
            :
            : [A0] "r"(A0), [A1] "r"(A1),
              [B0] "r"(B0), [B1] "r"(B1), [B2] "r"(B2), [B3] "r"(B3)
            : "memory", "t0",
              "v0", "v2", "v4", "v6", "v8", "v10",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
              "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    }

    __asm__ volatile(
        "vsetvli      t0, x0, e32, m2, ta, ma       \n\t"
        "vse32.v      v16, (%[s0])                   \n\t"
        "vse32.v      v18, (%[s1])                   \n\t"
        "vse32.v      v20, (%[s2])                   \n\t"
        "vse32.v      v22, (%[s3])                   \n\t"
        "vse32.v      v24, (%[s4])                   \n\t"
        "vse32.v      v26, (%[s5])                   \n\t"
        "vse32.v      v28, (%[s6])                   \n\t"
        "vse32.v      v30, (%[s7])                   \n\t"
        :
        : [s0] "r"(acc_scratch + 0),   [s1] "r"(acc_scratch + 16),
          [s2] "r"(acc_scratch + 32),  [s3] "r"(acc_scratch + 48),
          [s4] "r"(acc_scratch + 64),  [s5] "r"(acc_scratch + 80),
          [s6] "r"(acc_scratch + 96),  [s7] "r"(acc_scratch + 112)
        : "memory", "t0");

    // should replace with a built-in version instead of a helper
    // func call if possible.
    ime_scatter_acc(out, acc_scratch, MT, NT, N0_T);
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_8x16x8_riscv_64_v,
    iree_uk_mmt4d_tile_s8s8s32_8x16x8_riscv_64_v, 8)
