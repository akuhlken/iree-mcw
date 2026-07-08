// E2 test case: M/N/K are multiples of the IME tile (12x16x8).
// M=24 (2x12), K=16 (2x8), N=32 (2x16).
func.func @matmul(
    %lhs: tensor<24x16xi8>,
    %rhs: tensor<16x32xi8>) -> tensor<24x32xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<24x32xi32>
  %acc = linalg.fill ins(%c0 : i32) outs(%init : tensor<24x32xi32>)
      -> tensor<24x32xi32>
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<24x16xi8>, tensor<16x32xi8>)
      outs(%acc : tensor<24x32xi32>) -> tensor<24x32xi32>
  return %result : tensor<24x32xi32>
}
