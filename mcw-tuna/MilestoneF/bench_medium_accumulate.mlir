// F-bench: medium_accumulate — M=384, K=256, N=512.
// Same dims as medium_aligned, but C is a live input (C += A x B).
// Forces the IME ukernel ACCUMULATE path under L2-spill / BW pressure.
func.func @matmul_accumulate(%lhs: tensor<384x256xi8>,
                             %rhs: tensor<256x512xi8>,
                             %acc: tensor<384x512xi32>) -> tensor<384x512xi32> {
  %r = linalg.matmul
           ins(%lhs, %rhs : tensor<384x256xi8>, tensor<256x512xi8>)
           outs(%acc : tensor<384x512xi32>) -> tensor<384x512xi32>
  return %r : tensor<384x512xi32>
}
