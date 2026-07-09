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
// The whole vector body (accumulator init, K reduction, and store-out) for each
// M0 case lives in a single `__asm__ volatile` block. This is deliberate: the
// i32 accumulators must stay resident in the {vd,vd+1} register groups across
// the entire reduction, and a single asm statement is the only way to express
// that liveness to the compiler (separate asm statements have no way to chain
// live vector registers, and every register the block touches is declared in
// the clobber list). Keeping the K loop inside the asm also lets `vsetvli` be
// hoisted out of the loop (one SEW/LMUL switch per phase instead of per panel)
// and lets the assembler-level scheduler overlap the vle8 loads of the next
// panel with the vmadot of the current one.
//
// Gated on the +xsmtvdot CPU feature (see common_riscv_64.h) and built with
// -march=...xsmtvdot so the assembler accepts the `smt.vmadot` mnemonic.

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_internal.h"

#define IME_M_ATOM 4
#define IME_N_ATOM 4
#define IME_K_ATOM 8
#define IME_K0 8

// Gather the strided (M0,N0) int32 output tile into a contiguous per-atom
// scratch layout: atom (mt,nt) occupies scratch[(mt*NT+nt)*16] as a row-major
// 4x4 block, which is exactly the layout the {vd,vd+1} accumulator group holds.
inline static void ime_gather_acc(const iree_uk_int32_t* out,
                                  iree_uk_int32_t* scratch, int MT, int NT,
                                  int N0) {
  for (int mt = 0; mt < MT; mt++) {
    for (int nt = 0; nt < NT; nt++) {
      iree_uk_int32_t* dst = scratch + (mt * NT + nt) * 16;
      const iree_uk_int32_t* src =
          out + (mt * IME_M_ATOM) * N0 + (nt * IME_N_ATOM);
      for (int r = 0; r < IME_M_ATOM; r++)
        for (int c = 0; c < IME_N_ATOM; c++)
          dst[r * IME_N_ATOM + c] = src[r * N0 + c];
    }
  }
}

// Inverse of ime_gather_acc: scatter the contiguous per-atom scratch tiles back
// into the strided (M0,N0) output.
inline static void ime_scatter_acc(iree_uk_int32_t* out,
                                   const iree_uk_int32_t* scratch, int MT,
                                   int NT, int N0) {
  for (int mt = 0; mt < MT; mt++) {
    for (int nt = 0; nt < NT; nt++) {
      const iree_uk_int32_t* src = scratch + (mt * NT + nt) * 16;
      iree_uk_int32_t* dst = out + (mt * IME_M_ATOM) * N0 + (nt * IME_N_ATOM);
      for (int r = 0; r < IME_M_ATOM; r++)
        for (int c = 0; c < IME_N_ATOM; c++)
          dst[r * N0 + c] = src[r * IME_N_ATOM + c];
    }
  }
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s8s32_4x16x8_to_12x16x8_riscv_64_xsmtvdot(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 == 4 || M0 == 8 || M0 == 12);
  IREE_UK_ASSERT(params->N0 == 16);
  IREE_UK_ASSERT(params->K0 == IME_K0);

  const iree_uk_int8_t* IREE_UK_RESTRICT lhs = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out = out_tile;

  // K1 is the number of K0-deep panels reduced by this call.
  const int K1 = (int)(params->K);

  // N0 is always 16 (confirmed via assert), so NT is always 4.
  enum { NT = 4, N0 = 16 };
  const int accumulate =
      (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) != 0;

  if (M0 == 4) {
    enum { MT = 1 };
    iree_uk_int32_t acc_scratch[MT * NT * 16];
    if (accumulate) ime_gather_acc(out, acc_scratch, MT, NT, N0);

    // Accumulators: v6,v8,v10,v12 (each an {vd,vd+1} i32 group).
    // Operands: A0=v0; B0=v1,B1=v2,B2=v3,B3=v4. lhs panel stride = 4*8 = 32 B,
    // rhs panel stride = 16*8 = 128 B (advanced in-asm by the addi trail).
    __asm__ volatile(
        "    beqz        %[acc], .Lzero%=                 \n\t"
        "    vsetvli     t0, x0, e32, m2, ta, ma          \n\t"
        "    mv          t1, %[scr]                       \n\t"
        "    vle32.v     v6,  (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v8,  (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v10, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v12, (t1)                        \n\t"
        "    j           .Lloop_setup%=                   \n\t"
        ".Lzero%=:                                        \n\t"
        "    vsetvli     t0, x0, e32, m2, ta, ma          \n\t"
        "    vmv.v.i     v6,  0                           \n\t"
        "    vmv.v.i     v8,  0                           \n\t"
        "    vmv.v.i     v10, 0                           \n\t"
        "    vmv.v.i     v12, 0                           \n\t"
        ".Lloop_setup%=:                                  \n\t"
        "    mv          t1, %[lhs]                       \n\t"
        "    mv          t2, %[rhs]                       \n\t"
        "    mv          t3, %[K]                         \n\t"
        "    vsetvli     t0, x0, e8, m1, ta, ma           \n\t"
        "    beqz        t3, .Lstore%=                    \n\t"
        ".Lloop%=:                                        \n\t"
        "    vle8.v      v0, (t1)                         \n\t"
        "    addi        t1, t1, 32                       \n\t"
        "    vle8.v      v1, (t2)                         \n\t"
        "    addi        t2, t2, 32                       \n\t"
        "    vle8.v      v2, (t2)                         \n\t"
        "    addi        t2, t2, 32                       \n\t"
        "    vle8.v      v3, (t2)                         \n\t"
        "    addi        t2, t2, 32                       \n\t"
        "    vle8.v      v4, (t2)                         \n\t"
        "    addi        t2, t2, 32                       \n\t"
        "    smt.vmadot  v6,  v0, v1                      \n\t"
        "    smt.vmadot  v8,  v0, v2                      \n\t"
        "    smt.vmadot  v10, v0, v3                      \n\t"
        "    smt.vmadot  v12, v0, v4                      \n\t"
        "    addi        t3, t3, -1                       \n\t"
        "    bnez        t3, .Lloop%=                     \n\t"
        ".Lstore%=:                                       \n\t"
        "    vsetvli     t0, x0, e32, m2, ta, ma          \n\t"
        "    mv          t1, %[scr]                       \n\t"
        "    vse32.v     v6,  (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v8,  (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v10, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v12, (t1)                        \n\t"
        :
        : [lhs] "r"(lhs), [rhs] "r"(rhs), [scr] "r"(acc_scratch),
          [K] "r"(K1), [acc] "r"(accumulate)
        : "memory", "t0", "t1", "t2", "t3", "v0", "v1", "v2", "v3", "v4", "v6",
          "v7", "v8", "v9", "v10", "v11", "v12", "v13");
    ime_scatter_acc(out, acc_scratch, MT, NT, N0);
  } else if (M0 == 8) {
    enum { MT = 2 };
    iree_uk_int32_t acc_scratch[MT * NT * 16];
    if (accumulate) ime_gather_acc(out, acc_scratch, MT, NT, N0);

    // Accumulators: v16,v18,v20,v22,v24,v26,v28,v30.
    // Operands: A0=v0,A1=v2; B0=v4,B1=v6,B2=v8,B3=v10.
    __asm__ volatile(
        "    beqz        %[acc], .Lzero%=                 \n\t"
        "    vsetvli     t0, x0, e32, m2, ta, ma          \n\t"
        "    mv          t1, %[scr]                       \n\t"
        "    vle32.v     v16, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v18, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v20, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v22, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v24, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v26, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v28, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v30, (t1)                        \n\t"
        "    j           .Lloop_setup%=                   \n\t"
        ".Lzero%=:                                        \n\t"
        "    vsetvli     t0, x0, e32, m2, ta, ma          \n\t"
        "    vmv.v.i     v16, 0                           \n\t"
        "    vmv.v.i     v18, 0                           \n\t"
        "    vmv.v.i     v20, 0                           \n\t"
        "    vmv.v.i     v22, 0                           \n\t"
        "    vmv.v.i     v24, 0                           \n\t"
        "    vmv.v.i     v26, 0                           \n\t"
        "    vmv.v.i     v28, 0                           \n\t"
        "    vmv.v.i     v30, 0                           \n\t"
        ".Lloop_setup%=:                                  \n\t"
        "    mv          t1, %[lhs]                       \n\t"
        "    mv          t2, %[rhs]                       \n\t"
        "    mv          t3, %[K]                         \n\t"
        "    vsetvli     t0, x0, e8, m1, ta, ma           \n\t"
        "    beqz        t3, .Lstore%=                    \n\t"
        ".Lloop%=:                                        \n\t"
        "    vle8.v      v0, (t1)                         \n\t"
        "    addi        t1, t1, 32                       \n\t"
        "    vle8.v      v2, (t1)                         \n\t"
        "    addi        t1, t1, 32                       \n\t"
        "    vle8.v      v4, (t2)                         \n\t"
        "    addi        t2, t2, 32                       \n\t"
        "    vle8.v      v6, (t2)                         \n\t"
        "    addi        t2, t2, 32                       \n\t"
        "    vle8.v      v8, (t2)                         \n\t"
        "    addi        t2, t2, 32                       \n\t"
        "    vle8.v      v10, (t2)                        \n\t"
        "    addi        t2, t2, 32                       \n\t"
        "    smt.vmadot  v16, v0, v4                      \n\t"
        "    smt.vmadot  v18, v0, v6                      \n\t"
        "    smt.vmadot  v20, v0, v8                      \n\t"
        "    smt.vmadot  v22, v0, v10                     \n\t"
        "    smt.vmadot  v24, v2, v4                      \n\t"
        "    smt.vmadot  v26, v2, v6                      \n\t"
        "    smt.vmadot  v28, v2, v8                      \n\t"
        "    smt.vmadot  v30, v2, v10                     \n\t"
        "    addi        t3, t3, -1                       \n\t"
        "    bnez        t3, .Lloop%=                     \n\t"
        ".Lstore%=:                                       \n\t"
        "    vsetvli     t0, x0, e32, m2, ta, ma          \n\t"
        "    mv          t1, %[scr]                       \n\t"
        "    vse32.v     v16, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v18, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v20, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v22, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v24, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v26, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v28, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v30, (t1)                        \n\t"
        :
        : [lhs] "r"(lhs), [rhs] "r"(rhs), [scr] "r"(acc_scratch),
          [K] "r"(K1), [acc] "r"(accumulate)
        : "memory", "t0", "t1", "t2", "t3", "v0", "v2", "v4", "v6", "v8", "v10",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
    ime_scatter_acc(out, acc_scratch, MT, NT, N0);
  } else if (M0 == 12) {
    enum { MT = 3 };
    iree_uk_int32_t acc_scratch[MT * NT * 16];
    if (accumulate) ime_gather_acc(out, acc_scratch, MT, NT, N0);

    // Accumulators: v8,v10,...,v30 (12 groups spanning v8..v31 -- max register
    // occupancy, 24 acc regs + 7 operand regs = 31 <= 32).
    // Operands: A0=v0,A1=v2,A2=v4; B0=v1,B1=v3,B2=v5,B3=v6.
    __asm__ volatile(
        "    beqz        %[acc], .Lzero%=                 \n\t"
        "    vsetvli     t0, x0, e32, m2, ta, ma          \n\t"
        "    mv          t1, %[scr]                       \n\t"
        "    vle32.v     v8,  (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v10, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v12, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v14, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v16, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v18, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v20, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v22, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v24, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v26, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v28, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vle32.v     v30, (t1)                        \n\t"
        "    j           .Lloop_setup%=                   \n\t"
        ".Lzero%=:                                        \n\t"
        "    vsetvli     t0, x0, e32, m2, ta, ma          \n\t"
        "    vmv.v.i     v8,  0                           \n\t"
        "    vmv.v.i     v10, 0                           \n\t"
        "    vmv.v.i     v12, 0                           \n\t"
        "    vmv.v.i     v14, 0                           \n\t"
        "    vmv.v.i     v16, 0                           \n\t"
        "    vmv.v.i     v18, 0                           \n\t"
        "    vmv.v.i     v20, 0                           \n\t"
        "    vmv.v.i     v22, 0                           \n\t"
        "    vmv.v.i     v24, 0                           \n\t"
        "    vmv.v.i     v26, 0                           \n\t"
        "    vmv.v.i     v28, 0                           \n\t"
        "    vmv.v.i     v30, 0                           \n\t"
        ".Lloop_setup%=:                                  \n\t"
        "    mv          t1, %[lhs]                       \n\t"
        "    mv          t2, %[rhs]                       \n\t"
        "    mv          t3, %[K]                         \n\t"
        "    vsetvli     t0, x0, e8, m1, ta, ma           \n\t"
        "    beqz        t3, .Lstore%=                    \n\t"
        ".Lloop%=:                                        \n\t"
        "    vle8.v      v0, (t1)                         \n\t"
        "    addi        t1, t1, 32                       \n\t"
        "    vle8.v      v2, (t1)                         \n\t"
        "    addi        t1, t1, 32                       \n\t"
        "    vle8.v      v4, (t1)                         \n\t"
        "    addi        t1, t1, 32                       \n\t"
        "    vle8.v      v1, (t2)                         \n\t"
        "    addi        t2, t2, 32                       \n\t"
        "    vle8.v      v3, (t2)                         \n\t"
        "    addi        t2, t2, 32                       \n\t"
        "    vle8.v      v5, (t2)                         \n\t"
        "    addi        t2, t2, 32                       \n\t"
        "    vle8.v      v6, (t2)                         \n\t"
        "    addi        t2, t2, 32                       \n\t"
        "    smt.vmadot  v8,  v0, v1                      \n\t"
        "    smt.vmadot  v10, v0, v3                      \n\t"
        "    smt.vmadot  v12, v0, v5                      \n\t"
        "    smt.vmadot  v14, v0, v6                      \n\t"
        "    smt.vmadot  v16, v2, v1                      \n\t"
        "    smt.vmadot  v18, v2, v3                      \n\t"
        "    smt.vmadot  v20, v2, v5                      \n\t"
        "    smt.vmadot  v22, v2, v6                      \n\t"
        "    smt.vmadot  v24, v4, v1                      \n\t"
        "    smt.vmadot  v26, v4, v3                      \n\t"
        "    smt.vmadot  v28, v4, v5                      \n\t"
        "    smt.vmadot  v30, v4, v6                      \n\t"
        "    addi        t3, t3, -1                       \n\t"
        "    bnez        t3, .Lloop%=                     \n\t"
        ".Lstore%=:                                       \n\t"
        "    vsetvli     t0, x0, e32, m2, ta, ma          \n\t"
        "    mv          t1, %[scr]                       \n\t"
        "    vse32.v     v8,  (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v10, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v12, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v14, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v16, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v18, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v20, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v22, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v24, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v26, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v28, (t1)                        \n\t"
        "    addi        t1, t1, 64                       \n\t"
        "    vse32.v     v30, (t1)                        \n\t"
        :
        : [lhs] "r"(lhs), [rhs] "r"(rhs), [scr] "r"(acc_scratch),
          [K] "r"(K1), [acc] "r"(accumulate)
        : "memory", "t0", "t1", "t2", "t3", "v0", "v1", "v2", "v3", "v4", "v5",
          "v6", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
          "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
          "v27", "v28", "v29", "v30", "v31");
    ime_scatter_acc(out, acc_scratch, MT, NT, N0);
  }
}

// TODO: fix the names. N0 is a set value and that's not reflected here.
// the N0 values default to XX in
// /runtime/src/iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_entry_point.c
// so a change would have to be made there if desired.
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_4x16x8_to_12x16x8_riscv_64_xsmtvdot,
    iree_uk_mmt4d_tile_s8s8s32_4xXXx8_riscv_64_xsmtvdot, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_4x16x8_to_12x16x8_riscv_64_xsmtvdot,
    iree_uk_mmt4d_tile_s8s8s32_8xXXx8_riscv_64_xsmtvdot, 8)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_4x16x8_to_12x16x8_riscv_64_xsmtvdot,
    iree_uk_mmt4d_tile_s8s8s32_12xXXx8_riscv_64_xsmtvdot, 12)
