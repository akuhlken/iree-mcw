// F-bench: non_aligned — M=100, K=100, N=100.
// Dimensions are not multiples of the IME tile (M0=12, N0=16, K0=8).
// Exercises the compiler's M-truncation tiles and K-padding path.
// Measures overhead of partial tiles vs the aligned cases.
func.func @matmul(%lhs: tensor<100x100xi8>,
                  %rhs: tensor<100x100xi8>) -> tensor<100x100xi32> {
  %c0   = arith.constant 0 : i32
  %init = tensor.empty() : tensor<100x100xi32>
  %acc  = linalg.fill ins(%c0 : i32) outs(%init : tensor<100x100xi32>)
              -> tensor<100x100xi32>
  %r    = linalg.matmul
              ins(%lhs, %rhs : tensor<100x100xi8>, tensor<100x100xi8>)
              outs(%acc : tensor<100x100xi32>) -> tensor<100x100xi32>
  return %r : tensor<100x100xi32>
}
