// mmt4d_ime_s8s8s32.c
//
// Milestone B — Task 1: Standalone IME microkernels in mmt4d panel layout
//
// Implements: C[M0 x N0] += A[K1, M0, K0] * B[K1, N0, K0]  (s8 x s8 -> s32)
// using SpaceMiT IME `vmadot` (Xsmti8i32mm) inline assembly.
//
// Hardware target:  SpaceMiT X60 / A60 core  (K1 SoC, Cluster 0, cores 0-3)
//   VLEN = 256, lambda = 2, W = 4
//   One vmadot call = 4x4x8 (M x N x K) int8 -> int32 MAC tile (the "atom")
//   Accumulator {vd, vd+1}: 2-vreg group (MUL_C = 2), vd must be even
//
// This file provides five mmt4d tile functions, each hand-tuned for its shape.
// Every shape is built from the same 4x4x8 IME atom; they differ only in how
// many atoms tile the M and N directions (the "grid"):
//
//   shape     grid (MT x NT)   atoms   acc pairs   vmadot/step
//   4x8x8       1 x 2            2        2            2
//   8x4x8       2 x 1            2        2            2
//   8x8x8       2 x 2            4        4            4
//   8x16x8      2 x 4            8        8            8
//   16x8x8      4 x 2            8        8            8
//
//   A panel: [K1][M0][K0]  int8 bytes   (A row-major, K0-contiguous)
//   B panel: [K1][N0][K0]  int8 bytes   (B already column-major within each K0
//            chunk — i.e. B[n*K0 + k] — the mmt4d RHS "transposed" layout that
//            vmadot expects.)
//   C tile:  [M0][N0]      int32 elements (row-major)
//
// Each kernel keeps ALL of its accumulators resident in vector registers across
// the entire K1 reduction loop, loading each distinct A row-atom and B col-atom
// exactly once per K1 step (maximum operand reuse). The largest shapes use 8
// accumulator pairs (v16,v18,...,v30) plus up to 8 operand registers, which
// fits comfortably in the 32-register file.
//
// See SpaceMiT docs-ai: §2.1.1, §2.5 (tile 4x8x4 on A60), §2.7.3 (layouts)
// and issue #24576 §1.2 for the mmt4d <-> vmadot layout correspondence.
//
// Compile for the BPI-F3 (pinned to cores 0-3):
//   riscv64-unknown-linux-gnu-gcc -O2 -march=rv64gcv_xsmtvdot \
//       -o mmt4d_ime_test mmt4d_ime_s8s8s32.c && \
//   taskset -c 0-3 ./mmt4d_ime_test
//
// Or with SpaceMiT bundled clang (named mnemonic, same result):
//   clang -O2 -target riscv64-linux-gnu -march=rv64gcv_xsmtvdot \
//       -o mmt4d_ime_test mmt4d_ime_s8s8s32.c
//
// The `smt.vmadot` mnemonic is in LLVM >= 20 (merged 2025-08-18, PR #151706).
// If your assembler predates that, swap to the .insn form shown in comments.

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "mmt4d_ime_s8s8s32.h"

// ---------------------------------------------------------------------------
// Atomic tile from hardware: one vmadot covers 4 x 4 x 8 on A60 (VLEN=256).
// (IME_M_ATOM / IME_N_ATOM / IME_K_ATOM / IME_K0 come from the header.)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// vmadot inline-asm macro
//
// Computes:  C_acc{vd,vd+1} += A_tile{vs1} * B_tile{vs2}
//
// Constraints:
//   * vd  must be even (register pressure: use v16/v18/.../v30 for accum)
//   * LMUL = m1 (set by prior vsetvli e8,m1)
//   * A tile is 4x8 int8 in vs1 (row-major, K-contiguous)
//   * B tile is 4x8 int8 in vs2 (column-major = B[n*K0+k], K-contiguous)
//
// Named mnemonic form (LLVM >= 20 / assembler supporting xsmtvdot):
//   "smt.vmadot %[vd], %[vs1], %[vs2]"
//
// Raw .insn fallback (works with any RISC-V assembler, encoding derived from
// SpaceMiT spec §8 and remlab analysis; func7=0x38 for vmadot s8xs8):
//   ".insn r 0x2b, 0x2, 0x38, %[vd], %[vs1], %[vs2]"
//   OPCODE=custom-1 (0x2b), func3=0x2 (signed x signed), func7=0x38
//
// We use the named form as the primary; the raw form is left in the comment
// so it can be dropped in if the toolchain does not yet know the mnemonic.
// ---------------------------------------------------------------------------
#define VMADOT(vd, vs1, vs2)                              \
    __asm__ volatile(                                     \
        "smt.vmadot " #vd ", " #vs1 ", " #vs2 "\n\t"     \
        : /* no pure C outputs — vd is a vector reg */    \
        : /* no pure C inputs  */                         \
        : "memory")

// ---------------------------------------------------------------------------
// Accumulator scatter/gather helpers (plain C, called once per kernel, outside
// the K1 loop — zero impact on the inner reduction loop).
//
// Each vmadot accumulator pair {vd,vd+1} holds one 4x4 atom as 16 contiguous
// row-major int32 lanes. The kernels load/store those 16-lane atoms from/to a
// contiguous scratch buffer with a single vle32.v/vse32.v (e32,m2). These
// helpers convert between that atom-packed scratch layout and the row-major
// `out` tile (row stride N0):
//
//   scratch[(mt*NT + nt) * 16 + r * 4 + c]  <->  out[(mt*4 + r)*N0 + (nt*4 + c)]
//
// MT = M0 / 4 atoms down, NT = N0 / 4 atoms across.
// ---------------------------------------------------------------------------
static void ime_gather_acc(const int32_t *out, int32_t *scratch,
                           int MT, int NT, int N0)
{
    for (int mt = 0; mt < MT; mt++) {
        for (int nt = 0; nt < NT; nt++) {
            int32_t *dst = scratch + (mt * NT + nt) * 16;
            const int32_t *src = out + (mt * IME_M_ATOM) * N0 + (nt * IME_N_ATOM);
            for (int r = 0; r < IME_M_ATOM; r++)
                for (int c = 0; c < IME_N_ATOM; c++)
                    dst[r * IME_N_ATOM + c] = src[r * N0 + c];
        }
    }
}

static void ime_scatter_acc(int32_t *out, const int32_t *scratch,
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

// ===========================================================================
// 4x8x8  — grid 1 x 2 (1 M-atom, 2 N-atoms)
//   A atoms: v0
//   B atoms: v1, v2
//   acc    : {v16,v17}=A0*B0 , {v18,v19}=A0*B1
// ===========================================================================
void iree_uk_mmt4d_tile_s8s8s32_4x8x8_ime(
    const int8_t *lhs, const int8_t *rhs, int32_t *out,
    int32_t M0, int32_t K0, int32_t flags, int32_t K1)
{
    (void)M0;
    (void)K0;
    enum { MT = 1, NT = 2, M0_T = 4, N0_T = 8 };
    int32_t acc_scratch[MT * NT * 16];

    if (flags & 1) {
        ime_gather_acc(out, acc_scratch, MT, NT, N0_T);
        __asm__ volatile(
            "vsetvli      t0, x0, e32, m2, ta, ma       \n\t"
            "vle32.v      v16, (%[s0])                   \n\t"
            "vle32.v      v18, (%[s1])                   \n\t"
            :
            : [s0] "r"(acc_scratch + 0), [s1] "r"(acc_scratch + 16)
            : "memory", "t0", "v16", "v17", "v18", "v19");
    } else {
        __asm__ volatile(
            "vsetvli      t0, x0, e32, m2, ta, ma       \n\t"
            "vmv.v.i      v16, 0                         \n\t"
            "vmv.v.i      v18, 0                         \n\t"
            :
            :
            : "t0", "v16", "v17", "v18", "v19");
    }

    for (int k1 = 0; k1 < K1; k1++) {
        const int8_t *A0 = lhs + k1 * (M0_T * IME_K0) + 0 * IME_M_ATOM * IME_K0;
        const int8_t *B0 = rhs + k1 * (N0_T * IME_K0) + 0 * IME_N_ATOM * IME_K0;
        const int8_t *B1 = rhs + k1 * (N0_T * IME_K0) + 1 * IME_N_ATOM * IME_K0;
        // printf("k1 = %d, A0 = %d, B0 = %d, B1 = %d\n", k1, *A0, *B0, *B1);

        __asm__ volatile(
            "vsetvli      t0, x0, e8, m1, ta, ma        \n\t"
            "vle8.v       v0, (%[A0])                    \n\t"
            "vle8.v       v1, (%[B0])                    \n\t"
            "vle8.v       v2, (%[B1])                    \n\t"
            "smt.vmadot   v16, v0, v1                    \n\t"
            "smt.vmadot   v18, v0, v2                    \n\t"
            :
            : [A0] "r"(A0), [B0] "r"(B0), [B1] "r"(B1)
            : "memory", "t0", "v0", "v1", "v2",
              "v16", "v17", "v18", "v19");
    }

    __asm__ volatile(
        "vsetvli      t0, x0, e32, m2, ta, ma       \n\t"
        "vse32.v      v16, (%[s0])                   \n\t"
        "vse32.v      v18, (%[s1])                   \n\t"
        :
        : [s0] "r"(acc_scratch + 0), [s1] "r"(acc_scratch + 16)
        : "memory", "t0");
    ime_scatter_acc(out, acc_scratch, MT, NT, N0_T);
}

// ===========================================================================
// 8x4x8  — grid 2 x 1 (2 M-atoms, 1 N-atom)
//   A atoms: v0, v2
//   B atoms: v1
//   acc    : {v16,v17}=A0*B0 , {v18,v19}=A1*B0
// ===========================================================================
void iree_uk_mmt4d_tile_s8s8s32_8x4x8_ime(
    const int8_t *lhs, const int8_t *rhs, int32_t *out,
    int32_t M0, int32_t K0, int32_t flags, int32_t K1)
{
    (void)M0;
    (void)K0;
    enum { MT = 2, NT = 1, M0_T = 8, N0_T = 4 };
    int32_t acc_scratch[MT * NT * 16];

    if (flags & 1) {
        ime_gather_acc(out, acc_scratch, MT, NT, N0_T);
        __asm__ volatile(
            "vsetvli      t0, x0, e32, m2, ta, ma       \n\t"
            "vle32.v      v16, (%[s0])                   \n\t"
            "vle32.v      v18, (%[s1])                   \n\t"
            :
            : [s0] "r"(acc_scratch + 0), [s1] "r"(acc_scratch + 16)
            : "memory", "t0", "v16", "v17", "v18", "v19");
    } else {
        __asm__ volatile(
            "vsetvli      t0, x0, e32, m2, ta, ma       \n\t"
            "vmv.v.i      v16, 0                         \n\t"
            "vmv.v.i      v18, 0                         \n\t"
            :
            :
            : "t0", "v16", "v17", "v18", "v19");
    }

    for (int k1 = 0; k1 < K1; k1++) {
        const int8_t *A0 = lhs + k1 * (M0_T * IME_K0) + 0 * IME_M_ATOM * IME_K0;
        const int8_t *A1 = lhs + k1 * (M0_T * IME_K0) + 1 * IME_M_ATOM * IME_K0;
        const int8_t *B0 = rhs + k1 * (N0_T * IME_K0) + 0 * IME_N_ATOM * IME_K0;

        __asm__ volatile(
            "vsetvli      t0, x0, e8, m1, ta, ma        \n\t"
            "vle8.v       v0, (%[A0])                    \n\t"
            "vle8.v       v2, (%[A1])                    \n\t"
            "vle8.v       v1, (%[B0])                    \n\t"
            "smt.vmadot   v16, v0, v1                    \n\t"
            "smt.vmadot   v18, v2, v1                    \n\t"
            :
            : [A0] "r"(A0), [A1] "r"(A1), [B0] "r"(B0)
            : "memory", "t0", "v0", "v1", "v2",
              "v16", "v17", "v18", "v19");
    }

    __asm__ volatile(
        "vsetvli      t0, x0, e32, m2, ta, ma       \n\t"
        "vse32.v      v16, (%[s0])                   \n\t"
        "vse32.v      v18, (%[s1])                   \n\t"
        :
        : [s0] "r"(acc_scratch + 0), [s1] "r"(acc_scratch + 16)
        : "memory", "t0");
    ime_scatter_acc(out, acc_scratch, MT, NT, N0_T);
}

// ===========================================================================
// 8x8x8  — grid 2 x 2 (2 M-atoms, 2 N-atoms)  [canonical shape]
//   A atoms: v0, v2
//   B atoms: v1, v3
//   acc    : {v16}=A0*B0 {v18}=A0*B1 {v20}=A1*B0 {v22}=A1*B1
// ===========================================================================
void iree_uk_mmt4d_tile_s8s8s32_8x8x8_ime(
    const int8_t *lhs, const int8_t *rhs, int32_t *out,
    int32_t M0, int32_t K0, int32_t flags, int32_t K1)
{
    (void)M0;
    (void)K0;
    enum { MT = 2, NT = 2, M0_T = 8, N0_T = 8 };
    int32_t acc_scratch[MT * NT * 16];

    if (flags & 1) {
        ime_gather_acc(out, acc_scratch, MT, NT, N0_T);
        __asm__ volatile(
            // Use rd!=x0 (t0) so vl is set to VLMAX; the rd=x0,rs1=x0
            // "keep-vl" form leaves vtype illegal on a fresh context and
            // traps the next vector op (SIGILL) on the SpaceMiT X60.
            "vsetvli      t0, x0, e32, m2, ta, ma       \n\t"
            "vle32.v      v16, (%[s0])                   \n\t"
            "vle32.v      v18, (%[s1])                   \n\t"
            "vle32.v      v20, (%[s2])                   \n\t"
            "vle32.v      v22, (%[s3])                   \n\t"
            :
            : [s0] "r"(acc_scratch + 0),  [s1] "r"(acc_scratch + 16),
              [s2] "r"(acc_scratch + 32), [s3] "r"(acc_scratch + 48)
            : "memory", "t0", "v16", "v17", "v18", "v19",
                         "v20", "v21", "v22", "v23");
    } else {
        __asm__ volatile(
            "vsetvli      t0, x0, e32, m2, ta, ma       \n\t"
            "vmv.v.i      v16, 0                         \n\t"
            "vmv.v.i      v18, 0                         \n\t"
            "vmv.v.i      v20, 0                         \n\t"
            "vmv.v.i      v22, 0                         \n\t"
            :
            :
            : "t0", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23");
    }

    for (int k1 = 0; k1 < K1; k1++) {
        const int8_t *A0 = lhs + k1 * (M0_T * IME_K0) + 0 * IME_M_ATOM * IME_K0;
        const int8_t *A1 = lhs + k1 * (M0_T * IME_K0) + 1 * IME_M_ATOM * IME_K0;
        const int8_t *B0 = rhs + k1 * (N0_T * IME_K0) + 0 * IME_N_ATOM * IME_K0;
        const int8_t *B1 = rhs + k1 * (N0_T * IME_K0) + 1 * IME_N_ATOM * IME_K0;

        __asm__ volatile(
            "vsetvli      t0, x0, e8, m1, ta, ma        \n\t"
            "vle8.v       v0, (%[A0])                    \n\t"
            "vle8.v       v2, (%[A1])                    \n\t"
            "vle8.v       v1, (%[B0])                    \n\t"
            "vle8.v       v3, (%[B1])                    \n\t"
            "smt.vmadot   v16, v0, v1                    \n\t"
            "smt.vmadot   v18, v0, v3                    \n\t"
            "smt.vmadot   v20, v2, v1                    \n\t"
            "smt.vmadot   v22, v2, v3                    \n\t"
            :
            : [A0] "r"(A0), [A1] "r"(A1),
              [B0] "r"(B0), [B1] "r"(B1)
            : "memory", "t0",
              "v0", "v1", "v2", "v3",
              "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23");
    }

    __asm__ volatile(
        "vsetvli      t0, x0, e32, m2, ta, ma       \n\t"
        "vse32.v      v16, (%[s0])                   \n\t"
        "vse32.v      v18, (%[s1])                   \n\t"
        "vse32.v      v20, (%[s2])                   \n\t"
        "vse32.v      v22, (%[s3])                   \n\t"
        :
        : [s0] "r"(acc_scratch + 0),  [s1] "r"(acc_scratch + 16),
          [s2] "r"(acc_scratch + 32), [s3] "r"(acc_scratch + 48)
        : "memory", "t0");
    ime_scatter_acc(out, acc_scratch, MT, NT, N0_T);
}

// ===========================================================================
// 8x16x8 — grid 2 x 4 (2 M-atoms, 4 N-atoms)
//   A atoms: v0, v2
//   B atoms: v4, v6, v8, v10
//   acc    : 8 pairs {v16,v18,v20,v22,v24,v26,v28,v30}
//     row mt=0: A0 * {B0,B1,B2,B3} -> v16 v18 v20 v22
//     row mt=1: A1 * {B0,B1,B2,B3} -> v24 v26 v28 v30
// ===========================================================================
void iree_uk_mmt4d_tile_s8s8s32_8x16x8_ime(
    const int8_t *lhs, const int8_t *rhs, int32_t *out,
    int32_t M0, int32_t K0, int32_t flags, int32_t K1)
{
    (void)M0;
    (void)K0;
    enum { MT = 2, NT = 4, M0_T = 8, N0_T = 16 };
    int32_t acc_scratch[MT * NT * 16];

    if (flags & 1) {
        ime_gather_acc(out, acc_scratch, MT, NT, N0_T);
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
        // printf("k1 = %d, A0 = %d, A1 = %d, B0 = %d, B1 = %d, B2 = %d, B3 = %d\n", k1, *A0, *A1, *B0, *B1, *B2, *B3);


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
    ime_scatter_acc(out, acc_scratch, MT, NT, N0_T);
}

// ===========================================================================
// 16x8x8 — grid 4 x 2 (4 M-atoms, 2 N-atoms)
//   A atoms: v0, v2, v4, v6
//   B atoms: v8, v10
//   acc    : 8 pairs {v16,v18,v20,v22,v24,v26,v28,v30}
//     mt=0: A0 * {B0,B1} -> v16 v18
//     mt=1: A1 * {B0,B1} -> v20 v22
//     mt=2: A2 * {B0,B1} -> v24 v26
//     mt=3: A3 * {B0,B1} -> v28 v30
// ===========================================================================
void iree_uk_mmt4d_tile_s8s8s32_16x8x8_ime(
    const int8_t *lhs, const int8_t *rhs, int32_t *out,
    int32_t M0, int32_t K0, int32_t flags, int32_t K1)
{
    (void)M0;
    (void)K0;
    enum { MT = 4, NT = 2, M0_T = 16, N0_T = 8 };
    int32_t acc_scratch[MT * NT * 16];

    if (flags & 1) {
        ime_gather_acc(out, acc_scratch, MT, NT, N0_T);
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
        const int8_t *A2 = lhs + k1 * (M0_T * IME_K0) + 2 * IME_M_ATOM * IME_K0;
        const int8_t *A3 = lhs + k1 * (M0_T * IME_K0) + 3 * IME_M_ATOM * IME_K0;
        const int8_t *B0 = rhs + k1 * (N0_T * IME_K0) + 0 * IME_N_ATOM * IME_K0;
        const int8_t *B1 = rhs + k1 * (N0_T * IME_K0) + 1 * IME_N_ATOM * IME_K0;

        __asm__ volatile(
            "vsetvli      t0, x0, e8, m1, ta, ma        \n\t"
            "vle8.v       v0, (%[A0])                    \n\t"
            "vle8.v       v2, (%[A1])                    \n\t"
            "vle8.v       v4, (%[A2])                    \n\t"
            "vle8.v       v6, (%[A3])                    \n\t"
            "vle8.v       v8, (%[B0])                    \n\t"
            "vle8.v       v10, (%[B1])                   \n\t"
            "smt.vmadot   v16, v0, v8                    \n\t"
            "smt.vmadot   v18, v0, v10                   \n\t"
            "smt.vmadot   v20, v2, v8                    \n\t"
            "smt.vmadot   v22, v2, v10                   \n\t"
            "smt.vmadot   v24, v4, v8                    \n\t"
            "smt.vmadot   v26, v4, v10                   \n\t"
            "smt.vmadot   v28, v6, v8                    \n\t"
            "smt.vmadot   v30, v6, v10                   \n\t"
            :
            : [A0] "r"(A0), [A1] "r"(A1), [A2] "r"(A2), [A3] "r"(A3),
              [B0] "r"(B0), [B1] "r"(B1)
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
    ime_scatter_acc(out, acc_scratch, MT, NT, N0_T);
}

// ---------------------------------------------------------------------------
// Scalar reference kernel
//
// Computes the same mmt4d contraction in plain C.  Used to validate vmadot
// bit-exactness (see Milestone B Task 2 / mmt4d_ime_validate.c).
//
// C[m][n] += sum_k1 sum_k  A[k1][m][k] * B[k1][n][k]
// ---------------------------------------------------------------------------
void mmt4d_reference_s8s8s32(
    const int8_t  *lhs,    // [K1][M0][K0]
    const int8_t  *rhs,    // [K1][N0][K0]
    int32_t       *out,    // [M0][N0]
    int32_t        M0,
    int32_t        N0,
    int32_t        K0,
    int32_t        K1,
    int32_t        accumulate)
{
    if (!accumulate) {
        for (int m = 0; m < M0; m++)
            for (int n = 0; n < N0; n++)
                out[m * N0 + n] = 0;
    }

    int num_out = 0;
    for (int k1 = 0; k1 < K1; k1++) {
        const int8_t *A_panel = lhs + k1 * M0 * K0;
        const int8_t *B_panel = rhs + k1 * N0 * K0;
        for (int m = 0; m < M0; m++) {
            for (int n = 0; n < N0; n++) {
                int32_t acc = 0;
                for (int k = 0; k < K0; k++) {
                    // A[k1][m][k] = A_panel[m*K0 + k]
                    // B[k1][n][k] = B_panel[n*K0 + k]  (B stored col-major
                    //                per mmt4d convention: n-stride, K-fast)
                    acc += (int32_t)A_panel[m * K0 + k]
                         * (int32_t)B_panel[n * K0 + k];
                }
                out[m * N0 + n] += acc;
                num_out++;
            }
        }
    }
    printf("vals in c: %d\n", num_out);
}