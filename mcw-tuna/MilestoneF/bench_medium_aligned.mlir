// F-bench: medium_aligned — M=384, K=256, N=512.
// All dims are multiples of the IME tile (M0=12, N0=16, K0=8):
//   384 = 32×12, 256 = 32×8, 512 = 32×16.
// Sized to spill out of L2; starts to stress memory bandwidth.
func.func @matmul(%lhs: tensor<384x256xi8>,
                  %rhs: tensor<256x512xi8>) -> tensor<384x512xi32> {
  %c0   = arith.constant 0 : i32
  %init = tensor.empty() : tensor<384x512xi32>
  %acc  = linalg.fill ins(%c0 : i32) outs(%init : tensor<384x512xi32>)
              -> tensor<384x512xi32>
  %r    = linalg.matmul
              ins(%lhs, %rhs : tensor<384x256xi8>, tensor<256x512xi8>)
              outs(%acc : tensor<384x512xi32>) -> tensor<384x512xi32>
  return %r : tensor<384x512xi32>
}
