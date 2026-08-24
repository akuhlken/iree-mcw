// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/Utils/MMAUtils.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler::IREE::Codegen {

bool incrementIndices(MutableArrayRef<int64_t> indices,
                      ArrayRef<int64_t> sizes) {
  for (int i = indices.size() - 1; i >= 0; --i) {
    if (++indices[i] == sizes[i]) {
      indices[i] = 0;
    } else {
      return true;
    }
  }
  return false;
}

// Flattens vector `value` to 1-D if its rank is greater than 1; otherwise
// returns it unchanged. Fixed-width only: it builds the flat type from the
// integer element count, which is undefined for scalable vectors. Callers with
// scalable vectors must use the scalable branch in
// `distributeMmaFragmentToIntrinsics` (which shape_casts away unit dims while
// preserving scalable flags) instead.
static Value flattenVector(OpBuilder &builder, Location loc, Value value) {
  auto vectorType = cast<VectorType>(value.getType());
  if (vectorType.getRank() <= 1) {
    return value;
  }
  auto flatVectorType = VectorType::get({vectorType.getNumElements()},
                                        vectorType.getElementType());
  return vector::ShapeCastOp::create(builder, loc, flatVectorType, value);
}

SmallVector<Value>
distributeMmaFragmentToIntrinsics(OpBuilder &builder, Location loc, Value value,
                                  const TileSwizzle &swizzle) {
  Type elemType = cast<VectorType>(value.getType()).getElementType();
  auto internalType = getTileVectorType(swizzle, elemType,
                                        [](TileSwizzle::Dim dim) {
                                          return dim.kind() ==
                                                 TileSwizzle::Dim::Kind::Internal;
                                        });
  auto internalShape = internalType.getShape();
  auto crossIntrinsicShape =
      sliceSwizzledShape(swizzle, [](TileSwizzle::Dim dim) {
        return dim.kind() == TileSwizzle::Dim::Kind::CrossIntrinsic;
      });
  bool hasScalable =
      llvm::any_of(internalType.getScalableDims(),
                   [](bool b) { return b; });
  int rank = internalShape.size();
  SmallVector<int64_t> indices(rank, 0);
  SmallVector<int64_t> strides(rank, 1);
  SmallVector<Value> distributedValues;
  do {
    if (hasScalable) {
      // Scalable-safe extraction. The distributed value's cross-intrinsic dims
      // are all leading (fixed), and the internal dims — including the scalable
      // one — follow (see the comment above `fullDistributedType`). Peel the
      // leading cross-intrinsic dims with a single `vector.extract` to obtain
      // the internal tile (e.g. `vector<[4]x1xf32>` for SVE), then `shape_cast`
      // away the non-scalable unit dims to yield the per-intrinsic scalable
      // fragment (`vector<[4]xf32>`). The fixed-width `ExtractStridedSliceOp`
      // path below cannot produce a scalable result, so this branch exists
      // specifically for SVE/RVV.
      int numCrossIntrinsic = 0;
      for (int64_t size : crossIntrinsicShape) {
        numCrossIntrinsic += (size > 1);
      }
      SmallVector<int64_t> pos;
      for (int d = 0; d < numCrossIntrinsic; ++d) {
        pos.push_back(indices[d]);
      }
      Value piece = vector::ExtractOp::create(builder, loc, value, pos);
      auto pieceType = cast<VectorType>(piece.getType());
      SmallVector<int64_t> fragShape;
      SmallVector<bool> fragScalable;
      ArrayRef<int64_t> pieceShape = pieceType.getShape();
      ArrayRef<bool> pieceScalable = pieceType.getScalableDims();
      for (unsigned i = 0; i < pieceShape.size(); ++i) {
        if (pieceScalable[i] || pieceShape[i] != 1) {
          fragShape.push_back(pieceShape[i]);
          fragScalable.push_back(pieceScalable[i]);
        }
      }
      auto fragType = VectorType::get(fragShape, elemType, fragScalable);
      if (pieceType != fragType) {
        piece = vector::ShapeCastOp::create(builder, loc, fragType, piece);
      }
      distributedValues.push_back(piece);
    } else {
      Value extract = vector::ExtractStridedSliceOp::create(
          builder, loc, value, indices, internalShape, strides);
      distributedValues.push_back(flattenVector(builder, loc, extract));
    }
  } while (incrementIndices(indices, crossIntrinsicShape));
  return distributedValues;
}

// Returns the swizzle's distributed N-D vector type, preserving the scalable
// flags carried by `Internal` dims with a symbolic multiplier (e.g. SVE's
// `ArmSveVLIn128bitUnits`, RVV's `RiscvVlenIn64bitUnits`). A plain
// `VectorType::get(shape, ...)` over `fullDistributedShape` would silently
// drop those flags, converting a scalable `vector<1x[4]xf32>` into a
// fixed-width `vector<1x4xf32>`.
static VectorType fullDistributedType(const TileSwizzle &swizzle,
                                      Type elemType) {
  return getTileVectorType(swizzle, elemType, [](TileSwizzle::Dim d) {
    return d.kind() != TileSwizzle::Dim::Kind::CrossThread;
  });
}

// Reshapes `value` to the swizzle's distributed N-D vector type if it is not
// already in that form. "Distributed" here means every dim of the swizzle's
// expand groups, with CrossThread dim sizes collapsed to 1 (those factors live
// in the lane id, not the per-lane vector). CPU `getDistributedTileTypes`
// produces a 2-D (outer × inner) collapsed form while GPU produces the
// distributed N-D form; this reshape lets the shared lowering body operate
// uniformly. Both sides apply the swizzle's permutation (CPU's permutation is
// always identity, see `IREECPUAttrs.cpp:getSwizzle`), so a plain `shape_cast`
// suffices — no transpose is needed. Scalable dims are preserved via
// `fullDistributedType` so the resulting vector keeps its scalable flags.
static Value reshapeToSwizzleDistributed(OpBuilder &builder, Location loc,
                                         Value value,
                                         const TileSwizzle &swizzle) {
  auto vecType = cast<VectorType>(value.getType());
  VectorType fullType = fullDistributedType(swizzle, vecType.getElementType());
  if (vecType == fullType) {
    return value;
  }
  return vector::ShapeCastOp::create(builder, loc, fullType, value);
}

LogicalResult buildDataTiledMMAUnderlyingOperations(
    OpBuilder &builder, Location loc, const TileSwizzle &lhsSwizzle,
    const TileSwizzle &rhsSwizzle, const TileSwizzle &accSwizzle,
    int64_t intrinsicsM, int64_t intrinsicsN, int64_t intrinsicsK,
    ValueRange inputs, ValueRange outputs,
    DataTiledMMAIntrinsicEmitter emitIntrinsic,
    SmallVectorImpl<Value> &results) {
  if (inputs.size() != 2 || outputs.size() != 1) {
    return failure();
  }

  // Reshape LHS/RHS to the swizzle's distributed form so the GPU-style
  // distribution code can index every cross-intrinsic dim independently.
  Value lhs = reshapeToSwizzleDistributed(builder, loc, inputs[0], lhsSwizzle);
  Value rhs = reshapeToSwizzleDistributed(builder, loc, inputs[1], rhsSwizzle);
  SmallVector<Value> intrinsicsLhs =
      distributeMmaFragmentToIntrinsics(builder, loc, lhs, lhsSwizzle);
  SmallVector<Value> intrinsicsRhs =
      distributeMmaFragmentToIntrinsics(builder, loc, rhs, rhsSwizzle);

  // The ACC reshape happens inside the HoistableConversionOp body so the
  // shape_cast pairs sit *outside* the conversion and the distribute /
  // reassemble bodies remain exact inverses (required for the elimination
  // pass to cancel them across reduction loop boundaries).
  auto distributeAccOp = IREE::Util::HoistableConversionOp::create(
      builder, loc, /*tag=*/kDataTiledAccDistribute,
      /*inverseTag=*/kDataTiledAccReassemble, ValueRange{outputs[0]},
      [&](OpBuilder &b, Location loc, ValueRange args) -> SmallVector<Value> {
        Value accDistributed =
            reshapeToSwizzleDistributed(b, loc, args[0], accSwizzle);
        return distributeMmaFragmentToIntrinsics(b, loc, accDistributed,
                                                 accSwizzle);
      });
  SmallVector<Value> intrinsicsAcc(distributeAccOp.getResults());

  // Loop over the unroll_{m,n,k} dimensions to emit per-intrinsic ops.
  for (int64_t mu = 0; mu < intrinsicsM; ++mu) {
    for (int64_t nu = 0; nu < intrinsicsN; ++nu) {
      for (int64_t ku = 0; ku < intrinsicsK; ++ku) {
        Value lhsPiece = intrinsicsLhs[mu * intrinsicsK + ku];
        Value rhsPiece = intrinsicsRhs[nu * intrinsicsK + ku];
        Value &acc = intrinsicsAcc[mu * intrinsicsN + nu];
        Value newAcc = emitIntrinsic(builder, loc, lhsPiece, rhsPiece, acc);
        if (!newAcc) {
          return failure();
        }
        acc = newAcc;
      }
    }
  }

  // Reassemble per-intrinsic ACC pieces into the swizzle's distributed form,
  // then shape_cast to the original output type as the final step inside the
  // HoistableConversionOp body (paired with the inverse cast above).
  SmallVector<int64_t> accCrossIntrinsicShape =
      sliceSwizzledShape(accSwizzle, [](TileSwizzle::Dim dim) {
        return dim.kind() == TileSwizzle::Dim::Kind::CrossIntrinsic;
      });
  SmallVector<int64_t> accInternalShape =
      sliceSwizzledShape(accSwizzle, [](TileSwizzle::Dim dim) {
        return dim.kind() == TileSwizzle::Dim::Kind::Internal;
      });
  Type origAccType = outputs[0].getType();
  Type accElemType = cast<VectorType>(origAccType).getElementType();
  auto fullAccType = fullDistributedType(accSwizzle, accElemType);
  bool accHasScalable =
      llvm::any_of(fullAccType.getScalableDims(),
                   [](bool b) { return b; });

  auto reassembleOp = IREE::Util::HoistableConversionOp::create(
      builder, loc, /*tag=*/kDataTiledAccReassemble,
      /*inverseTag=*/kDataTiledAccDistribute, intrinsicsAcc,
      [&](OpBuilder &b, Location loc, ValueRange args) -> SmallVector<Value> {
        int64_t dstRank = accCrossIntrinsicShape.size();
        SmallVector<int64_t> strides(dstRank, 1);
        SmallVector<int64_t> indices(dstRank, 0);
        Value acc =
            arith::ConstantOp::create(b, loc, b.getZeroAttr(fullAccType));
        if (accHasScalable) {
          // Scalable-safe reassembly. Each per-intrinsic fragment is the
          // scalable subvector (e.g. `vector<[4]xf32>` for SVE); shape_cast it
          // back to the internal tile type (`vector<[4]x1xf32>`) and
          // `vector.insert` it at the leading cross-intrinsic coordinates. The
          // `numCrossIntrinsic` leading fixed dims are the insertion
          // coordinates; the scalable dims stay in place at the tail.
          int64_t numCrossIntrinsic = 0;
          for (int64_t size : accCrossIntrinsicShape) {
            numCrossIntrinsic += (size > 1);
          }
          auto internalAccType = VectorType::get(
              fullAccType.getShape().drop_front(numCrossIntrinsic), accElemType,
              fullAccType.getScalableDims().drop_front(numCrossIntrinsic));
          for (Value intrAcc : args) {
            if (intrAcc.getType() != internalAccType) {
              intrAcc = vector::ShapeCastOp::create(b, loc, internalAccType,
                                                    intrAcc);
            }
            acc = vector::InsertOp::create(
                b, loc, intrAcc, acc,
                ArrayRef<int64_t>(indices).take_front(numCrossIntrinsic));
            incrementIndices(
                MutableArrayRef<int64_t>(indices).take_front(numCrossIntrinsic),
                ArrayRef<int64_t>(accCrossIntrinsicShape)
                    .take_front(numCrossIntrinsic));
          }
        } else {
          for (Value intrAcc : args) {
            Value expandedAcc = vector::ShapeCastOp::create(
                b, loc, VectorType::get(accInternalShape, accElemType),
                intrAcc);
            acc = vector::InsertStridedSliceOp::create(b, loc, expandedAcc, acc,
                                                       indices, strides);
            incrementIndices(indices, accCrossIntrinsicShape);
          }
        }
        if (acc.getType() != origAccType) {
          acc = vector::ShapeCastOp::create(b, loc, origAccType, acc);
        }
        return {acc};
      });
  results.push_back(reassembleOp.getResult(0));
  return success();
}

} // namespace mlir::iree_compiler::IREE::Codegen
