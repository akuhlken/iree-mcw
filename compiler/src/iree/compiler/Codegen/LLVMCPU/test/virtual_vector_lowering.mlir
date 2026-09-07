// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-virtual-vector-lowering))" --split-input-file %s | FileCheck %s

// For an n-D row-major contiguous memref, the gather lowering should add the
// gather index directly into the innermost offset rather than emit a per-lane
// affine.linearize_index/affine.delinearize_index pair.

func.func @gather_contiguous_3d(%buffer: memref<50x40x40xi8>,
                                %idx: vector<8xindex>,
                                %mask: vector<8xi1>,
                                %pass: vector<8xi8>,
                                %i: index, %j: index, %k: index)
    -> vector<8xi8> {
  %r = vector.gather %buffer[%i, %j, %k] [%idx], %mask, %pass
     : memref<50x40x40xi8>, vector<8xindex>, vector<8xi1>, vector<8xi8>
       into vector<8xi8>
  return %r : vector<8xi8>
}

// CHECK-LABEL:   func.func @gather_contiguous_3d(
// CHECK-SAME:        %[[BUF:[^:]+]]: memref<50x40x40xi8>
// CHECK-SAME:        %[[I:[^:]+]]: index, %[[J:[^:]+]]: index, %[[K:[^:]+]]: index
// CHECK-NOT:   vector.gather
// CHECK-NOT:   affine.linearize_index
// CHECK-NOT:   affine.delinearize_index
// CHECK:       arith.addi %[[K]],
// CHECK:       vector.load %[[BUF]][%[[I]], %[[J]], %{{.+}}] : memref<50x40x40xi8>, vector<1xi8>

// -----

// For a non-contiguous (strided, non-row-major) memref, the gather lowering
// must keep the linearize/delinearize path because the per-dimension and
// linearized addresses no longer agree.

func.func @gather_non_contiguous_2d(
    %buffer: memref<10x20xf32, strided<[40, 1]>>,
    %idx: vector<4xindex>, %mask: vector<4xi1>, %pass: vector<4xf32>,
    %i: index, %j: index) -> vector<4xf32> {
  %r = vector.gather %buffer[%i, %j] [%idx], %mask, %pass
     : memref<10x20xf32, strided<[40, 1]>>, vector<4xindex>,
       vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %r : vector<4xf32>
}

// CHECK-LABEL:   func.func @gather_non_contiguous_2d(
// CHECK-NOT:   vector.gather
// CHECK:       affine.linearize_index
// CHECK:       affine.delinearize_index

// -----

// Dynamic-shape identity-layout memref also qualifies as row-major contiguous,
// so the gather lowering should still take the contiguous path even though
// `memref::isStaticShapeAndContiguousRowMajor` returns false here.

func.func @gather_dynamic_identity_2d(%buffer: memref<?x?xf32>,
                                      %idx: vector<4xindex>,
                                      %mask: vector<4xi1>,
                                      %pass: vector<4xf32>,
                                      %i: index, %j: index) -> vector<4xf32> {
  %r = vector.gather %buffer[%i, %j] [%idx], %mask, %pass
     : memref<?x?xf32>, vector<4xindex>, vector<4xi1>, vector<4xf32>
       into vector<4xf32>
  return %r : vector<4xf32>
}

// CHECK-LABEL:   func.func @gather_dynamic_identity_2d(
// CHECK-SAME:        %[[BUF:[^:]+]]: memref<?x?xf32>
// CHECK-SAME:        %[[I:[^:]+]]: index, %[[J:[^:]+]]: index
// CHECK-NOT:   vector.gather
// CHECK-NOT:   affine.linearize_index
// CHECK-NOT:   affine.delinearize_index
// CHECK:       arith.addi %[[J]],
// CHECK:       vector.load %[[BUF]][%[[I]], %{{.+}}] : memref<?x?xf32>, vector<1xf32>

// -----

// A static-shape memref with an explicit strided<[N, 1]> layout whose strides
// match the row-major contiguous sequence should also take the contiguous
// path via `isStaticShapeAndContiguousRowMajor`.

func.func @gather_strided_contiguous_2d(
    %buffer: memref<10x20xf32, strided<[20, 1]>>,
    %idx: vector<4xindex>, %mask: vector<4xi1>, %pass: vector<4xf32>,
    %i: index, %j: index) -> vector<4xf32> {
  %r = vector.gather %buffer[%i, %j] [%idx], %mask, %pass
     : memref<10x20xf32, strided<[20, 1]>>, vector<4xindex>,
       vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %r : vector<4xf32>
}

// CHECK-LABEL:   func.func @gather_strided_contiguous_2d(
// CHECK-SAME:        %[[BUF:[^:]+]]: memref<10x20xf32, strided<[20, 1]>>
// CHECK-SAME:        %[[I:[^:]+]]: index, %[[J:[^:]+]]: index
// CHECK-NOT:   vector.gather
// CHECK-NOT:   affine.linearize_index
// CHECK-NOT:   affine.delinearize_index
// CHECK:       arith.addi %[[J]],
// CHECK:       vector.load %[[BUF]][%[[I]], %{{.+}}] : memref<10x20xf32, strided<[20, 1]>>, vector<1xf32>

// -----

// Tensor base falls through to the upstream pattern, which lowers each lane
// to a `tensor.extract`. The IREE-local pattern does not match (memref-only)
// and must not interfere.

func.func @gather_tensor_base_2d(%buffer: tensor<10x20xf32>,
                                 %idx: vector<4xindex>,
                                 %mask: vector<4xi1>,
                                 %pass: vector<4xf32>,
                                 %i: index, %j: index) -> vector<4xf32> {
  %r = vector.gather %buffer[%i, %j] [%idx], %mask, %pass
     : tensor<10x20xf32>, vector<4xindex>, vector<4xi1>, vector<4xf32>
       into vector<4xf32>
  return %r : vector<4xf32>
}

// CHECK-LABEL:   func.func @gather_tensor_base_2d(
// CHECK-SAME:        %[[BUF:[^:]+]]: tensor<10x20xf32>
// CHECK-NOT:   vector.gather
// CHECK:       tensor.extract %[[BUF]]

// -----

// `iree_codegen.inner_tiled` lowering. The pass runs three pattern sets in
// sequence on the op (unroll non-unit iter dims to unit, drop the now-unit
// iter domain, lower the iter-free vector `inner_tiled` to
// `llvm.call_intrinsic` via the kind's `buildUnderlyingOperations`) and ends
// with `IREE::Util::eliminateHoistableConversions` to cancel the inverse
// conversion pairs the lowering patterns leave around the ACC. On a single-
// tile AVX-512 1x16x1 f32 matmul, the end result is one `llvm.fma.v16f32`
// call with no surviving `util.hoistable_conversion`s.

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @lower_avx512_1x16x1_f32(
    %lhs: vector<1x1x1x1xf32>, %rhs: vector<1x1x16x1xf32>,
    %acc: vector<1x1x1x16xf32>) -> vector<1x1x1x16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>,
                      #linalg.iterator_type<parallel>,
                      #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x1_F32_F32>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<1x1x1x1xf32>, vector<1x1x16x1xf32> into vector<1x1x1x16xf32>
  return %0 : vector<1x1x1x16xf32>
}

// CHECK-LABEL: func @lower_avx512_1x16x1_f32
//   CHECK-NOT:   iree_codegen.inner_tiled
//   CHECK-NOT:   util.hoistable_conversion
//       CHECK:   llvm.call_intrinsic "llvm.fma.v16f32"({{.*}}) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>

// -----

// NEON `fmla` natural orientation (1×4×1): the scalar LHS is broadcast to the
// 4-lane RHS/acc vector and the whole thing lowers to one fixed-width
// `llvm.fma.v4f32`; the `inner_tiled` op is gone.

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @lower_arm_neon_fmla_1x4x1_f32(
    %lhs: vector<1x1x1x1xf32>, %rhs: vector<1x1x4x1xf32>,
    %acc: vector<1x1x1x4xf32>) -> vector<1x1x1x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>,
                      #linalg.iterator_type<parallel>,
                      #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_ARM_NEON_FMLA_1x4x1_F32_F32>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<1x1x1x1xf32>, vector<1x1x4x1xf32> into vector<1x1x1x4xf32>
  return %0 : vector<1x1x1x4xf32>
}

// CHECK-LABEL: func @lower_arm_neon_fmla_1x4x1_f32
//   CHECK-NOT:   iree_codegen.inner_tiled
//   CHECK-NOT:   util.hoistable_conversion
//       CHECK:   vector.broadcast {{.*}} : f32 to vector<4xf32>
//       CHECK:   llvm.call_intrinsic "llvm.fma.v4f32"({{.*}}) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>

// -----

// NEON `fmla` swapped orientation (4×1×1): the scalar RHS is broadcast.

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @lower_arm_neon_fmla_4x1x1_f32(
    %lhs: vector<1x1x4x1xf32>, %rhs: vector<1x1x1x1xf32>,
    %acc: vector<1x1x4x1xf32>) -> vector<1x1x4x1xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>,
                      #linalg.iterator_type<parallel>,
                      #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_ARM_NEON_FMLA_4x1x1_F32_F32>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<1x1x4x1xf32>, vector<1x1x1x1xf32> into vector<1x1x4x1xf32>
  return %0 : vector<1x1x4x1xf32>
}

// CHECK-LABEL: func @lower_arm_neon_fmla_4x1x1_f32
//   CHECK-NOT:   iree_codegen.inner_tiled
//   CHECK-NOT:   util.hoistable_conversion
//       CHECK:   llvm.call_intrinsic "llvm.fma.v4f32"({{.*}}) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>

// -----

// SVE `fmla` natural orientation (1×4VL×1): the scalar LHS is broadcast to a
// *scalable* vector and the intrinsic is `llvm.aarch64.sve.fmla`. The scalable `[4]`
// type survives all the way through the lowering.

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @lower_arm_sve_fmla_1x4vlx1_f32(
    %lhs: vector<1x1x1x1xf32>, %rhs: vector<1x1x[4]x1xf32>,
    %acc: vector<1x1x[4]x1xf32>) -> vector<1x1x[4]x1xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>,
                      #linalg.iterator_type<parallel>,
                      #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_ARM_SVE_FMLA_1x4VLx1_F32_F32>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<1x1x1x1xf32>, vector<1x1x[4]x1xf32> into vector<1x1x[4]x1xf32>
  return %0 : vector<1x1x[4]x1xf32>
}

// CHECK-LABEL: func @lower_arm_sve_fmla_1x4vlx1_f32
//   CHECK-NOT:   iree_codegen.inner_tiled
//   CHECK-NOT:   util.hoistable_conversion
//       CHECK:   vector.broadcast {{.*}} : f32 to vector<[4]xf32>
//       CHECK:   llvm.call_intrinsic "llvm.aarch64.sve.fmla"({{.*}}) : (vector<[4]xi1>, vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf32>) -> vector<[4]xf32>

// -----

// SVE `fmla` swapped orientation (4VL×1×1): the scalar RHS is broadcast to a
// scalable vector.

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @lower_arm_sve_fmla_4vlx1x1_f32(
    %lhs: vector<1x1x[4]x1xf32>, %rhs: vector<1x1x1x1xf32>,
    %acc: vector<1x1x[4]x1xf32>) -> vector<1x1x[4]x1xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>,
                      #linalg.iterator_type<parallel>,
                      #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_ARM_SVE_FMLA_4VLx1x1_F32_F32>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<1x1x[4]x1xf32>, vector<1x1x1x1xf32> into vector<1x1x[4]x1xf32>
  return %0 : vector<1x1x[4]x1xf32>
}

// CHECK-LABEL: func @lower_arm_sve_fmla_4vlx1x1_f32
//   CHECK-NOT:   iree_codegen.inner_tiled
//   CHECK-NOT:   util.hoistable_conversion
//       CHECK:   llvm.call_intrinsic "llvm.aarch64.sve.fmla"({{.*}}) : (vector<[4]xi1>, vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf32>) -> vector<[4]xf32>

// -----

// SVE `fmla` with intrinsics_n = 2: the cross-intrinsic N dim is unrolled into
// two per-intrinsic `llvm.aarch64.sve.fmla` calls, and the scalable extraction /
// reassembly keeps the `[4]` scalable flag throughout.

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_arm_sve_fmla_intrinsics_n2(
    %lhs: vector<1x1xf32>, %rhs: vector<2x[4]x1xf32>,
    %acc: vector<2x[4]x1xf32>) -> vector<2x[4]x1xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_ARM_SVE_FMLA_1x4VLx1_F32_F32, intrinsics_n = 2>,
    semantics = #iree_cpu.mma_semantics<>
  } : vector<1x1xf32>, vector<2x[4]x1xf32> into vector<2x[4]x1xf32>
  return %0 : vector<2x[4]x1xf32>
}

// CHECK-LABEL: func @lower_arm_sve_fmla_intrinsics_n2
//   CHECK-NOT:   iree_codegen.inner_tiled
//       CHECK:   llvm.call_intrinsic "llvm.aarch64.sve.fmla"({{.*}}) : (vector<[4]xi1>, vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf32>) -> vector<[4]xf32>
//       CHECK:   llvm.call_intrinsic "llvm.aarch64.sve.fmla"({{.*}}) : (vector<[4]xi1>, vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf32>) -> vector<[4]xf32>
