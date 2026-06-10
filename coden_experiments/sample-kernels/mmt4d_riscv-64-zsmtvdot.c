// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <riscv_vector.h>

// #include "/home/coden/iree/runtime/src/iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
// #include "iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_internal.h"

typedef struct mmt4d_params_t {
  const void* lhs_buffer;
  iree_uk_index_t lhs_offset;
  iree_uk_index_t lhs_stride0;
  const void* rhs_buffer;
  iree_uk_index_t rhs_offset;
  iree_uk_index_t rhs_stride0;
  void* out_buffer;
  iree_uk_index_t out_offset;
  iree_uk_index_t out_stride0;
  iree_uk_index_t M;
  iree_uk_index_t N;
  iree_uk_index_t K;
  uint32_t M0;
  uint32_t N0;
  uint32_t K0;
  uint32_t flags;
  const uint64_t* cpu_data;
} mmt4d_params_t;

// IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_7xXXx1_riscv_64_v(
    void* out_tile, const void* lhs_panel,
    const void* rhs_panel,
    const mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 7);
  const int8_t* lhs_ptr = lhs_panel;
  const int8_t* rhs_ptr = rhs_panel;
  int32_t* out_ptr = out_tile;

  vint32m4_t acc0, acc1, acc2, acc3, acc4, acc5, acc6;
  int N0 = params->N0;
  size_t vl = N0;

  if (M0 == 1) {
    if (params->flags) {
      acc0 = __riscv_vle32_v_i32m4(out_ptr, vl);
    } else {
      acc0 = __riscv_vmv_v_x_i32m4(0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vint8m1_t rhs = __riscv_vle8_v_i8m1(rhs_ptr, vl);
      vint16m2_t rhs_wide = __riscv_vsext_vf2_i16m2(rhs, vl);
      rhs_ptr += N0;
      int16_t lhs = (int16_t)*lhs_ptr++;
      acc0 = __riscv_vwmacc_vx_i32m4(acc0, lhs, rhs_wide, vl);
    }
    __riscv_vse32_v_i32m4(out_ptr, acc0, vl);
  } else if (M0 == 2) {
    if (params->flags) {
      acc0 = __riscv_vle32_v_i32m4(out_ptr, vl);
      acc1 = __riscv_vle32_v_i32m4(out_ptr + N0, vl);
    } else {
      acc0 = __riscv_vmv_v_x_i32m4(0, vl);
      acc1 = __riscv_vmv_v_x_i32m4(0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vint8m1_t rhs = __riscv_vle8_v_i8m1(rhs_ptr, vl);
      vint16m2_t rhs_wide = __riscv_vsext_vf2_i16m2(rhs, vl);
      rhs_ptr += N0;
      int16_t lhs0 = (int16_t)*lhs_ptr++;
      int16_t lhs1 = (int16_t)*lhs_ptr++;
      acc0 = __riscv_vwmacc_vx_i32m4(acc0, lhs0, rhs_wide, vl);
      acc1 = __riscv_vwmacc_vx_i32m4(acc1, lhs1, rhs_wide, vl);
    }
    __riscv_vse32_v_i32m4(out_ptr, acc0, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0, acc1, vl);
  } else if (M0 == 4) {
    if (params->flags) {
      acc0 = __riscv_vle32_v_i32m4(out_ptr, vl);
      acc1 = __riscv_vle32_v_i32m4(out_ptr + N0, vl);
      acc2 = __riscv_vle32_v_i32m4(out_ptr + N0 * 2, vl);
      acc3 = __riscv_vle32_v_i32m4(out_ptr + N0 * 3, vl);
    } else {
      acc0 = __riscv_vmv_v_x_i32m4(0, vl);
      acc1 = __riscv_vmv_v_x_i32m4(0, vl);
      acc2 = __riscv_vmv_v_x_i32m4(0, vl);
      acc3 = __riscv_vmv_v_x_i32m4(0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vint8m1_t rhs = __riscv_vle8_v_i8m1(rhs_ptr, vl);
      vint16m2_t rhs_wide = __riscv_vsext_vf2_i16m2(rhs, vl);
      rhs_ptr += N0;
      int16_t lhs0 = (int16_t)*lhs_ptr++;
      int16_t lhs1 = (int16_t)*lhs_ptr++;
      int16_t lhs2 = (int16_t)*lhs_ptr++;
      int16_t lhs3 = (int16_t)*lhs_ptr++;
      acc0 = __riscv_vwmacc_vx_i32m4(acc0, lhs0, rhs_wide, vl);
      acc1 = __riscv_vwmacc_vx_i32m4(acc1, lhs1, rhs_wide, vl);
      acc2 = __riscv_vwmacc_vx_i32m4(acc2, lhs2, rhs_wide, vl);
      acc3 = __riscv_vwmacc_vx_i32m4(acc3, lhs3, rhs_wide, vl);
    }
    __riscv_vse32_v_i32m4(out_ptr, acc0, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0, acc1, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 2, acc2, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 3, acc3, vl);
  } else if (M0 == 7) {
    if (params->flags) {
      acc0 = __riscv_vle32_v_i32m4(out_ptr, vl);
      acc1 = __riscv_vle32_v_i32m4(out_ptr + N0, vl);
      acc2 = __riscv_vle32_v_i32m4(out_ptr + N0 * 2, vl);
      acc3 = __riscv_vle32_v_i32m4(out_ptr + N0 * 3, vl);
      acc4 = __riscv_vle32_v_i32m4(out_ptr + N0 * 4, vl);
      acc5 = __riscv_vle32_v_i32m4(out_ptr + N0 * 5, vl);
      acc6 = __riscv_vle32_v_i32m4(out_ptr + N0 * 6, vl);
    } else {
      acc0 = __riscv_vmv_v_x_i32m4(0, vl);
      acc1 = __riscv_vmv_v_x_i32m4(0, vl);
      acc2 = __riscv_vmv_v_x_i32m4(0, vl);
      acc3 = __riscv_vmv_v_x_i32m4(0, vl);
      acc4 = __riscv_vmv_v_x_i32m4(0, vl);
      acc5 = __riscv_vmv_v_x_i32m4(0, vl);
      acc6 = __riscv_vmv_v_x_i32m4(0, vl);
    }
    for (int k = 0; k < params->K; ++k) {
      vint8m1_t rhs = __riscv_vle8_v_i8m1(rhs_ptr, vl);
      vint16m2_t rhs_wide = __riscv_vsext_vf2_i16m2(rhs, vl);
      rhs_ptr += N0;
      int16_t lhs0 = (int16_t)*lhs_ptr++;
      int16_t lhs1 = (int16_t)*lhs_ptr++;
      int16_t lhs2 = (int16_t)*lhs_ptr++;
      int16_t lhs3 = (int16_t)*lhs_ptr++;
      int16_t lhs4 = (int16_t)*lhs_ptr++;
      int16_t lhs5 = (int16_t)*lhs_ptr++;
      int16_t lhs6 = (int16_t)*lhs_ptr++;
      acc0 = __riscv_vwmacc_vx_i32m4(acc0, lhs0, rhs_wide, vl);
      acc1 = __riscv_vwmacc_vx_i32m4(acc1, lhs1, rhs_wide, vl);
      acc2 = __riscv_vwmacc_vx_i32m4(acc2, lhs2, rhs_wide, vl);
      acc3 = __riscv_vwmacc_vx_i32m4(acc3, lhs3, rhs_wide, vl);
      acc4 = __riscv_vwmacc_vx_i32m4(acc4, lhs4, rhs_wide, vl);
      acc5 = __riscv_vwmacc_vx_i32m4(acc5, lhs5, rhs_wide, vl);
      acc6 = __riscv_vwmacc_vx_i32m4(acc6, lhs6, rhs_wide, vl);
    }
    __riscv_vse32_v_i32m4(out_ptr, acc0, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0, acc1, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 2, acc2, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 3, acc3, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 4, acc4, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 5, acc5, vl);
    __riscv_vse32_v_i32m4(out_ptr + N0 * 6, acc6, vl);
  }
}

// IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
//     iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_7xXXx1_riscv_64_v,
//     iree_uk_mmt4d_tile_s8s8s32_1xXXx1_riscv_64_v, 1)
// IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
//     iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_7xXXx1_riscv_64_v,
//     iree_uk_mmt4d_tile_s8s8s32_2xXXx1_riscv_64_v, 2)
// IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
//     iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_7xXXx1_riscv_64_v,
//     iree_uk_mmt4d_tile_s8s8s32_4xXXx1_riscv_64_v, 4)
// IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
//     iree_uk_mmt4d_tile_s8s8s32_1xXXx1_to_7xXXx1_riscv_64_v,
//     iree_uk_mmt4d_tile_s8s8s32_7xXXx1_riscv_64_v, 7)
