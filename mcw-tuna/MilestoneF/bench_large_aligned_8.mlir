// F-bench: large_aligned — M=768, K=512, N=1024.
// All dims are multiples of the IME tile (M0=12, N0=16, K0=8):
//   768 = 64×12, 512 = 64×8, 1024 = 64×16.
// Memory-bandwidth-bound regime; representative of large quantized GEMM layers.
func.func @matmul(%lhs: tensor<8x512xi8>,
                  %rhs: tensor<512x1024xi8>) -> tensor<8x1024xi32> {
  %c0   = arith.constant 0 : i32
  %init = tensor.empty() : tensor<8x1024xi32>
  %acc  = linalg.fill ins(%c0 : i32) outs(%init : tensor<8x1024xi32>)
              -> tensor<8x1024xi32>
  %r    = linalg.matmul
              ins(%lhs, %rhs : tensor<8x512xi8>, tensor<512x1024xi8>)
              outs(%acc : tensor<8x1024xi32>) -> tensor<8x1024xi32>
  return %r : tensor<8x1024xi32>
}
