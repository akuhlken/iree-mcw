// E2 test case: M=9 exercises M-truncation tiles (8/4/2/1 x 16 x 8).
func.func @matmul(
    %lhs: tensor<9x9xi8>,
    %rhs: tensor<9x9xi8>) -> tensor<9x9xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<9x9xi32>
  %acc = linalg.fill ins(%c0 : i32) outs(%init : tensor<9x9xi32>)
      -> tensor<9x9xi32>
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<9x9xi8>, tensor<9x9xi8>)
      outs(%acc : tensor<9x9xi32>) -> tensor<9x9xi32>
  return %result : tensor<9x9xi32>
}
