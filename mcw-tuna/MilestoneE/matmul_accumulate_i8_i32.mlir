// E2 test case: accumulate mode (C += A x B).
func.func @matmul_accumulate(
    %lhs: tensor<12x8xi8>,
    %rhs: tensor<8x16xi8>,
    %acc: tensor<12x16xi32>) -> tensor<12x16xi32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<12x8xi8>, tensor<8x16xi8>)
      outs(%acc : tensor<12x16xi32>) -> tensor<12x16xi32>
  return %result : tensor<12x16xi32>
}
