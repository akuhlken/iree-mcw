// F-bench: llm_decode — M=12, K=4096, N=4096.
// Narrow-M shape typical of single-token LLM decode (one output row at a time).
// M=12 is exactly one IME M0 tile; exercises the minimum-M path with a very
// deep K reduction (K=4096 = 512×8 panels).  N=4096 = 256×16 panels.
func.func @matmul(%lhs: tensor<12x4096xi8>,
                  %rhs: tensor<4096x4096xi8>) -> tensor<12x4096xi32> {
  %c0   = arith.constant 0 : i32
  %init = tensor.empty() : tensor<12x4096xi32>
  %acc  = linalg.fill ins(%c0 : i32) outs(%init : tensor<12x4096xi32>)
              -> tensor<12x4096xi32>
  %r    = linalg.matmul
              ins(%lhs, %rhs : tensor<12x4096xi8>, tensor<4096x4096xi8>)
              outs(%acc : tensor<12x4096xi32>) -> tensor<12x4096xi32>
  return %r : tensor<12x4096xi32>
}
