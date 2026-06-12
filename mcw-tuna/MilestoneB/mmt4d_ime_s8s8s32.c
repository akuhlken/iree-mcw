// mmt4d_ime_s8s8s32.c
//
// Milestone B — Task 1: Standalone IME microkernel in mmt4d panel layout
//
// Implements: C[M0 x N0] += A[K1, M0, K0] * B[K1, N0, K0]  (s8 x s8 -> s32)
// using SpaceMiT IME `vmadot` (Xsmti8i32mm) inline assembly.
//
// Hardware target:  SpaceMiT X60 / A60 core  (K1 SoC, Cluster 0, cores 0-3)
//   VLEN = 256, lambda = 2, W = 4
//   One vmadot call = 4x4x8 (M x N x K) int8 -> int32 MAC tile
//   Accumulator {vd, vd+1}: 2-vreg group (MUL_C = 2), vd must be even
//
// Fixed macro-tile: M0=8, N0=8, K0=8  (two 4x4x8 IME tiles wide in M and N)
//   A panel: [K1][M0][K0] = K1 * 8 * 8  int8 bytes
//   B panel: [K1][N0][K0] = K1 * 8 * 8  int8 bytes  (B already column-major
//            within each K0 chunk — i.e., B[n*K0 + k] — which is exactly
//            the mmt4d RHS "transposed" layout and what vmadot expects.)
//   C tile:  [M0][N0]    =      8 * 8  int32 elements
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

// ---------------------------------------------------------------------------
// Tile dimensions — must agree with the IREE compiler's enumerate branch
// and the tiles.inl registration (Milestones C/D).
// ---------------------------------------------------------------------------
#define IME_M0   8   // Output rows  per macro-tile   (2 x IME_M_ATOM)
#define IME_N0   8   // Output cols  per macro-tile   (2 x IME_N_ATOM)
#define IME_K0   8   // Shared depth per macro-tile   (1 x IME_K_ATOM)

// Atomic tile from hardware: one vmadot covers 4 x 4 x 8 on A60 (VLEN=256)
#define IME_M_ATOM  4
#define IME_N_ATOM  4
#define IME_K_ATOM  8

// ---------------------------------------------------------------------------
// vmadot inline-asm macro
//
// Computes:  C_acc{vd,vd+1} += A_tile{vs1} * B_tile{vs2}
//
// Constraints:
//   * vd  must be even (register pressure: use v16/v18/v20/v22 for accum)
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
// IME microkernel — mmt4d tile function
//
// Signature mirrors IREE's iree_uk_mmt4d_tile_func_t so it can be dropped
// into arch/riscv_64/mmt4d_riscv_64_ime.c (Milestone C2) without changes.
//
//   lhs   : int8_t[K1][M0][K0]  — A panels in row-major K0-contiguous order
//   rhs   : int8_t[K1][N0][K0]  — B panels in col-major K0-contiguous order
//             (i.e. B[k1][n][k] stored at rhs[k1*N0*K0 + n*K0 + k])
//   out   : int32_t[M0][N0]     — accumulator tile, row-major
//   M0    : must equal IME_M0 (8)
//   K0    : must equal IME_K0 (8)
//   flags : bit 0 = accumulate (1) or overwrite (0)
//   K1    : reduction panel count
// ---------------------------------------------------------------------------
void iree_uk_mmt4d_tile_s8s8s32_8x8x8_ime(
    const int8_t  *lhs,   // [K1][M0][K0]
    const int8_t  *rhs,   // [K1][N0][K0]
    int32_t       *out,   // [M0][N0]
    int32_t        M0,
    int32_t        K0,
    int32_t        flags,
    int32_t        K1)
{
    // Silence unused-parameter warnings for parameters that are fixed by the
    // tile size; IREE's dispatch mechanism guarantees M0==8 and K0==8.
    (void)M0;
    (void)K0;

    // -----------------------------------------------------------------------
    // Accumulator register layout (M0=8, N0=8, K0=8):
    //
    // We compute a 2x2 grid of 4x4 atomic tiles:
    //
    //            N[0..3]    N[4..7]
    //   M[0..3]:  acc00      acc01
    //   M[4..7]:  acc10      acc11
    //
    // Each acc{row,col} occupies an even-aligned pair {vd, vd+1}:
    //   acc00 -> {v16, v17}
    //   acc01 -> {v18, v19}
    //   acc10 -> {v20, v21}
    //   acc11 -> {v22, v23}
    //
    // Total accumulator registers: 8  (well within the 32-reg budget)
    // Operand registers: v0 (A_row0), v1 (B_col0), v2 (A_row1), v3 (B_col1)
    // -----------------------------------------------------------------------

    // --- initialise / zero accumulators -----------------------------------
    if (flags & 1) {
        // Accumulate mode: each acc pair {vd,vd+1} holds its 4x4 sub-tile as
        // 16 contiguous row-major lanes (the vmadot/standalone contract).
        // `out` is an 8-wide matrix, so gather each 4x4 sub-block into a
        // contiguous scratch before the contiguous vle32.v.
        int32_t acc_scratch[4 * 16];
        for (int r = 0; r < IME_M_ATOM; r++) {
            for (int c = 0; c < IME_N_ATOM; c++) {
                acc_scratch[ 0 + r * 4 + c] = out[(0 + r) * IME_N0 + (0 + c)];
                acc_scratch[16 + r * 4 + c] = out[(0 + r) * IME_N0 + (4 + c)];
                acc_scratch[32 + r * 4 + c] = out[(4 + r) * IME_N0 + (0 + c)];
                acc_scratch[48 + r * 4 + c] = out[(4 + r) * IME_N0 + (4 + c)];
            }
        }
        __asm__ volatile(
            // Use rd!=x0 (t0) so vl is set to VLMAX; the rd=x0,rs1=x0
            // "keep-vl" form leaves vtype illegal on a fresh context and
            // traps the next vector op (SIGILL) on the SpaceMiT X60.
            "vsetvli      t0, x0, e32, m2, ta, ma       \n\t"
            "vle32.v      v16, (%[s00])                  \n\t"
            "vle32.v      v18, (%[s01])                  \n\t"
            "vle32.v      v20, (%[s10])                  \n\t"
            "vle32.v      v22, (%[s11])                  \n\t"
            :
            : [s00] "r"(acc_scratch + 0),
              [s01] "r"(acc_scratch + 16),
              [s10] "r"(acc_scratch + 32),
              [s11] "r"(acc_scratch + 48)
            : "memory", "t0", "v16", "v17", "v18", "v19",
                         "v20", "v21", "v22", "v23");
    } else {
        // Overwrite mode: zero accumulators with vxor.
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

    // --- main K1 reduction loop -------------------------------------------
    for (int k1 = 0; k1 < K1; k1++) {
        // Pointers to the two M0-half A panels and two N0-half B panels:
        //   lhs layout: [K1][M0][K0] = [K1][8][8]
        //     A_row0 = lhs[k1 * 8*8 + 0*8]   -> rows 0..3
        //     A_row1 = lhs[k1 * 8*8 + 4*8]   -> rows 4..7
        //   rhs layout: [K1][N0][K0] = [K1][8][8]
        //     B_col0 = rhs[k1 * 8*8 + 0*8]   -> cols 0..3
        //     B_col1 = rhs[k1 * 8*8 + 4*8]   -> cols 4..7
        const int8_t *A_row0 = lhs + k1 * (IME_M0 * IME_K0) + 0 * IME_K_ATOM * IME_K0;
        const int8_t *A_row1 = lhs + k1 * (IME_M0 * IME_K0) + IME_M_ATOM * IME_K0;
        const int8_t *B_col0 = rhs + k1 * (IME_N0 * IME_K0) + 0 * IME_N_ATOM * IME_K0;
        const int8_t *B_col1 = rhs + k1 * (IME_N0 * IME_K0) + IME_N_ATOM * IME_K0;

        __asm__ volatile(
            // Set e8,m1: SEW=8, LMUL=1 — required by vmadot on A60.
            // rd!=x0 (t0) sets vl=VLMAX; the keep-vl form (x0,x0) traps.
            "vsetvli      t0, x0, e8, m1, ta, ma        \n\t"

            // Load A sub-tiles (32 bytes each = one 256-bit vector register)
            "vle8.v       v0, (%[A0])                    \n\t"  // A[0..3][0..7]
            "vle8.v       v2, (%[A1])                    \n\t"  // A[4..7][0..7]

            // Load B sub-tiles  (B is column-major in mmt4d: B[n*K0 + k])
            "vle8.v       v1, (%[B0])                    \n\t"  // B[0..3][0..7]
            "vle8.v       v3, (%[B1])                    \n\t"  // B[4..7][0..7]

            // 2x2 grid of vmadot calls
            // acc00 (rows 0..3 x cols 0..3):  {v16,v17} += v0 * v1
            "smt.vmadot   v16, v0, v1                    \n\t"
            // acc01 (rows 0..3 x cols 4..7):  {v18,v19} += v0 * v3
            "smt.vmadot   v18, v0, v3                    \n\t"
            // acc10 (rows 4..7 x cols 0..3):  {v20,v21} += v2 * v1
            "smt.vmadot   v20, v2, v1                    \n\t"
            // acc11 (rows 4..7 x cols 4..7):  {v22,v23} += v2 * v3
            "smt.vmadot   v22, v2, v3                    \n\t"
            :
            : [A0] "r"(A_row0), [A1] "r"(A_row1),
              [B0] "r"(B_col0), [B1] "r"(B_col1)
            : "memory", "t0",
              "v0", "v1", "v2", "v3",
              "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23");
    }

    // --- store accumulators back to C -------------------------------------
    // Each acc pair holds a 4x4 sub-tile as 16 contiguous row-major lanes.
    // Store to a contiguous scratch, then scatter into the 8-wide `out`.
    {
        int32_t acc_scratch[4 * 16];
        __asm__ volatile(
            "vsetvli      t0, x0, e32, m2, ta, ma       \n\t"
            "vse32.v      v16, (%[s00])                  \n\t"
            "vse32.v      v18, (%[s01])                  \n\t"
            "vse32.v      v20, (%[s10])                  \n\t"
            "vse32.v      v22, (%[s11])                  \n\t"
            :
            : [s00] "r"(acc_scratch + 0),
              [s01] "r"(acc_scratch + 16),
              [s10] "r"(acc_scratch + 32),
              [s11] "r"(acc_scratch + 48)
            : "memory", "t0");

        for (int r = 0; r < IME_M_ATOM; r++) {
            for (int c = 0; c < IME_N_ATOM; c++) {
                out[(0 + r) * IME_N0 + (0 + c)] = acc_scratch[ 0 + r * 4 + c];
                out[(0 + r) * IME_N0 + (4 + c)] = acc_scratch[16 + r * 4 + c];
                out[(4 + r) * IME_N0 + (0 + c)] = acc_scratch[32 + r * 4 + c];
                out[(4 + r) * IME_N0 + (4 + c)] = acc_scratch[48 + r * 4 + c];
            }
        }
    }
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
    int32_t       *out,    // [M0][N0], overwritten (no accumulate flag here)
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
            }
        }
    }
}
