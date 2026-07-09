// F-bench: small_aligned — M=96, K=64, N=128.
// All dims are multiples of the IME tile (M0=12, N0=16, K0=8):
//   96 = 8×12, 64 = 8×8, 128 = 8×16.
// Fits in L1/L2; isolates raw compute throughput from memory-bandwidth effects.
func.func @matmul(%lhs: tensor<4x64xi8>,
                  %rhs: tensor<64x128xi8>) -> tensor<4x128xi32> {
  %c0   = arith.constant 0 : i32
  %init = tensor.empty() : tensor<4x128xi32>
  %acc  = linalg.fill ins(%c0 : i32) outs(%init : tensor<4x128xi32>)
              -> tensor<4x128xi32>
  %r    = linalg.matmul
              ins(%lhs, %rhs : tensor<4x64xi8>, tensor<64x128xi8>)
              outs(%acc : tensor<4x128xi32>) -> tensor<4x128xi32>
  return %r : tensor<4x128xi32>
}
