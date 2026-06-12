// mmt4d_ime_s8s8s32.h
//
// Public interface for the standalone IME microkernel and scalar reference.
// Include this in the validation harness (mmt4d_ime_validate.c).

#ifndef MMT4D_IME_S8S8S32_H_
#define MMT4D_IME_S8S8S32_H_

#include <stdint.h>

// Fixed macro-tile geometry for the A60 IME path (VLEN=256, lambda=2, W=4).
// One vmadot call covers a 4x4x8 (M x N x K) int8->int32 tile.
// The chosen macro-tile is 8x8x8 (two atoms in M and N, one atom in K).
#define IME_M0   8
#define IME_N0   8
#define IME_K0   8

// ---------------------------------------------------------------------------
// IME microkernel — mmt4d tile function.
//
//   lhs   : int8_t  [K1][M0][K0]  — LHS panels
//   rhs   : int8_t  [K1][N0][K0]  — RHS panels (B column-major per mmt4d)
//   out   : int32_t [M0][N0]      — accumulator tile, row-major
//   M0    : must == IME_M0 (8)
//   K0    : must == IME_K0 (8)
//   flags : bit 0 = 1 → accumulate into out; 0 → overwrite out
//   K1    : reduction panel count (>= 1)
// ---------------------------------------------------------------------------
void iree_uk_mmt4d_tile_s8s8s32_8x8x8_ime(
    const int8_t  *lhs,
    const int8_t  *rhs,
    int32_t       *out,
    int32_t        M0,
    int32_t        K0,
    int32_t        flags,
    int32_t        K1);

// ---------------------------------------------------------------------------
// Scalar reference kernel — identical semantics, pure C.
//
//   accumulate : 0 = zero out first; 1 = add to existing values
// ---------------------------------------------------------------------------
void mmt4d_reference_s8s8s32(
    const int8_t  *lhs,
    const int8_t  *rhs,
    int32_t       *out,
    int32_t        M0,
    int32_t        N0,
    int32_t        K0,
    int32_t        K1,
    int32_t        accumulate);

#endif  // MMT4D_IME_S8S8S32_H_
