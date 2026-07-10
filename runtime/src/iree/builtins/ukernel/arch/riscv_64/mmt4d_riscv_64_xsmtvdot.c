// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SpaceMiT IME (XSMTVDot) s8s8s32 mmt4d microkernel.
//
// One `smt.vmadot` on the SpaceMiT X60 (VLEN=256, SEW=8) computes a 4x4x8
// (M x N x K) int8->int32 MAC tile -- the IME "atom". This kernel tiles a
// 12x16x8 (M0 x N0 x K0) macro-tile as a 3x4 grid of atoms (12 vmadot per K
// panel), keeping all 12 accumulator pairs resident in vector registers across
// the whole K reduction. The macro-tile shape is fixed; the K reduction depth
// (K1 = K / K0 panels) is handled by the inner loop.
//
// mmt4d panel layout (matches what vmadot expects natively):
//   lhs : int8 [K1][M0][K0]   row-major, K0-contiguous
//   rhs : int8 [K1][N0][K0]   already transposed to (N0,K0), K0-contiguous
//   out : int32 [M0][N0]      row-major
//
// Gated on the +xsmtvdot CPU feature (see common_riscv_64.h) and built with
// -march=...xsmtvdot so the assembler accepts the `smt.vmadot` mnemonic.

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_internal.h"

#define IME_M_ATOM 4
#define IME_N_ATOM 4
#define IME_K_ATOM 8
#define IME_K0 8

inline static void ime_gather_acc(const int32_t *out, int32_t *scratch,
                           int MT, int NT, int N0)
{
    for (int mt = 0; mt < MT; mt++) {
        for (int nt = 0; nt < NT; nt++) {
            // for each tile combo, aka for each accumulator.
            int32_t *dst = scratch + (mt * NT + nt) * 16;  // the destination tile
            const int32_t *src = out + (mt * IME_M_ATOM) * N0 + (nt * IME_N_ATOM);
            for (int r = 0; r < IME_M_ATOM; r++)
                for (int c = 0; c < IME_N_ATOM; c++)
                    dst[r * IME_N_ATOM + c] = src[r * N0 + c];
        }
    }
}

inline static void ime_scatter_acc(int32_t *out, const int32_t *scratch,
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
iree_uk_mmt4d_tile_s8s8s32_12x16x8_riscv_64_xsmtvdot(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 == 12);
  IREE_UK_ASSERT(params->N0 == 16);
  IREE_UK_ASSERT(params->K0 == IME_K0);

  const iree_uk_int8_t* IREE_UK_RESTRICT lhs = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out = out_tile;

  // K1 is the number of K0-deep panels reduced by this call.
  const int K1 = (int)(params->K);

  // N0 is always 16 (confirmed via assert), so NT is always 4.
  enum { MT = 3, NT = 4, N0 = 16 };
  iree_uk_int32_t acc_scratch[MT * NT * 16];

  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      ime_gather_acc(out, acc_scratch, MT, NT, N0);
      __asm__ volatile(
          "vsetvli      t0, x0, e32, m2, ta, ma        \n\t"
          "vle32.v      v8, (%[s0])                    \n\t"
          "vle32.v      v10, (%[s1])                   \n\t"
          "vle32.v      v12, (%[s2])                   \n\t"
          "vle32.v      v14, (%[s3])                   \n\t"
          "vle32.v      v16, (%[s4])                   \n\t"
          "vle32.v      v18, (%[s5])                   \n\t"
          "vle32.v      v20, (%[s6])                   \n\t"
          "vle32.v      v22, (%[s7])                   \n\t"
          "vle32.v      v24, (%[s8])                   \n\t"
          "vle32.v      v26, (%[s9])                   \n\t"
          "vle32.v      v28, (%[s10])                  \n\t"
          "vle32.v      v30, (%[s11])                  \n\t"
          :
          : [s0] "r"(acc_scratch + 0), [s1] "r"(acc_scratch + 16),
            [s2] "r"(acc_scratch + 32), [s3] "r"(acc_scratch + 48),
            [s4] "r"(acc_scratch + 64), [s5] "r"(acc_scratch + 80),
            [s6] "r"(acc_scratch + 96), [s7] "r"(acc_scratch + 112),
            [s8] "r"(acc_scratch + 128), [s9] "r"(acc_scratch + 144),
            [s10] "r"(acc_scratch + 160), [s11] "r"(acc_scratch + 176)
          : "memory", "t0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
  } else {
      __asm__ volatile(
          "vsetvli      t0, x0, e32, m2, ta, ma       \n\t"
          "vmv.v.i      v8, 0                         \n\t"
          "vmv.v.i      v10, 0                         \n\t"
          "vmv.v.i      v12, 0                         \n\t"
          "vmv.v.i      v14, 0                         \n\t"
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
          : "t0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
          "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
          "v27", "v28", "v29", "v30", "v31");
  }

  for (int k1 = 0; k1 < K1; ++k1) {
      const iree_uk_int8_t* A0 =
          lhs + k1 * (M0 * IME_K0) + 0 * IME_M_ATOM * IME_K0;
      const iree_uk_int8_t* A1 =
          lhs + k1 * (M0 * IME_K0) + 1 * IME_M_ATOM * IME_K0;
      const iree_uk_int8_t* A2 =
          lhs + k1 * (M0 * IME_K0) + 2 * IME_M_ATOM * IME_K0;
      const iree_uk_int8_t* B0 =
          rhs + k1 * (N0 * IME_K0) + 0 * IME_N_ATOM * IME_K0;
      const iree_uk_int8_t* B1 =
          rhs + k1 * (N0 * IME_K0) + 1 * IME_N_ATOM * IME_K0;
      const iree_uk_int8_t* B2 =
          rhs + k1 * (N0 * IME_K0) + 2 * IME_N_ATOM * IME_K0;
      const iree_uk_int8_t* B3 =
          rhs + k1 * (N0 * IME_K0) + 3 * IME_N_ATOM * IME_K0;

      __asm__ volatile(
          "vsetvli      t0, x0, e8, m1, ta, ma        \n\t"
          "vle8.v       v0, (%[A0])                    \n\t"
          "vle8.v       v2, (%[A1])                    \n\t"
          "vle8.v       v4, (%[A2])                    \n\t"
          "vle8.v       v1, (%[B0])                    \n\t"
          "vle8.v       v3, (%[B1])                    \n\t"
          "vle8.v       v5, (%[B2])                    \n\t"
          "vle8.v       v6, (%[B3])                    \n\t"
          "smt.vmadot   v8, v0, v1                     \n\t"
          "smt.vmadot   v10, v0, v3                    \n\t"
          "smt.vmadot   v12, v0, v5                    \n\t"
          "smt.vmadot   v14, v0, v6                    \n\t"
          "smt.vmadot   v16, v2, v1                    \n\t"
          "smt.vmadot   v18, v2, v3                    \n\t"
          "smt.vmadot   v20, v2, v5                    \n\t"
          "smt.vmadot   v22, v2, v6                    \n\t"
          "smt.vmadot   v24, v4, v1                    \n\t"
          "smt.vmadot   v26, v4, v3                    \n\t"
          "smt.vmadot   v28, v4, v5                    \n\t"
          "smt.vmadot   v30, v4, v6                    \n\t"
          :
          : [A0] "r"(A0), [A1] "r"(A1), [A2] "r"(A2),
            [B0] "r"(B0), [B1] "r"(B1), [B2] "r"(B2), [B3] "r"(B3)
          : "memory", "t0", "v0", "v2", "v4", "v6", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
          "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
  }

  __asm__ volatile(
      "vsetvli      t0, x0, e32, m2, ta, ma        \n\t"
      "vse32.v      v8, (%[s0])                    \n\t"
      "vse32.v      v10, (%[s1])                   \n\t"
      "vse32.v      v12, (%[s2])                   \n\t"
      "vse32.v      v14, (%[s3])                   \n\t"
      "vse32.v      v16, (%[s4])                   \n\t"
      "vse32.v      v18, (%[s5])                   \n\t"
      "vse32.v      v20, (%[s6])                   \n\t"
      "vse32.v      v22, (%[s7])                   \n\t"
      "vse32.v      v24, (%[s8])                   \n\t"
      "vse32.v      v26, (%[s9])                   \n\t"
      "vse32.v      v28, (%[s10])                  \n\t"
      "vse32.v      v30, (%[s11])                  \n\t"
      :
      : [s0] "r"(acc_scratch + 0), [s1] "r"(acc_scratch + 16),
        [s2] "r"(acc_scratch + 32), [s3] "r"(acc_scratch + 48),
        [s4] "r"(acc_scratch + 64), [s5] "r"(acc_scratch + 80),
        [s6] "r"(acc_scratch + 96), [s7] "r"(acc_scratch + 112),
        [s8] "r"(acc_scratch + 128), [s9] "r"(acc_scratch + 144),
        [s10] "r"(acc_scratch + 160), [s11] "r"(acc_scratch + 176)
      : "memory", "t0");
  ime_scatter_acc(out, acc_scratch, MT, NT, N0);
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_12x16x8_riscv_64_xsmtvdot,
    iree_uk_mmt4d_tile_s8s8s32_12xXXx8_riscv_64_xsmtvdot, 12)
