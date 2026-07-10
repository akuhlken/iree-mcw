// F-bench: llm_decode_accumulate — M=12, K=4096, N=4096.
// Same dims as llm_decode, but C is a live input (C += A x B).
// One M0 tile + deep K1: gather once, then long vmadot reduction, then scatter.
func.func @matmul_accumulate(%lhs: tensor<12x4096xi8>,
                             %rhs: tensor<4096x4096xi8>,
                             %acc: tensor<12x4096xi32>) -> tensor<12x4096xi32> {
  %r = linalg.matmul
           ins(%lhs, %rhs : tensor<12x4096xi8>, tensor<4096x4096xi8>)
           outs(%acc : tensor<12x4096xi32>) -> tensor<12x4096xi32>
  return %r : tensor<12x4096xi32>
}
