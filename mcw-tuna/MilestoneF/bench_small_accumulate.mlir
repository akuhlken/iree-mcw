// F-bench: small_accumulate — M=96, K=64, N=128.
// Same dims as small_aligned, but C is a live input (C += A x B).
// Forces the IME ukernel ACCUMULATE path: ime_gather_acc → vmadot → scatter.
func.func @matmul_accumulate(%lhs: tensor<96x64xi8>,
                             %rhs: tensor<64x128xi8>,
                             %acc: tensor<96x128xi32>) -> tensor<96x128xi32> {
  %r = linalg.matmul
           ins(%lhs, %rhs : tensor<96x64xi8>, tensor<64x128xi8>)
           outs(%acc : tensor<96x128xi32>) -> tensor<96x128xi32>
  return %r : tensor<96x128xi32>
}
