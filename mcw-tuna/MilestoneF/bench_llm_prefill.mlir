// F-bench: llm_prefill — M=384, K=4096, N=4096.
// Batched LLM prefill / projection layer (e.g. 384-token context).
// M=384 = 32×12, K=4096 = 512×8, N=4096 = 256×16 — all tile-aligned.
// Compute-bound; the shape that should show the highest absolute GOPS gain.
func.func @matmul(%lhs: tensor<384x4096xi8>,
                  %rhs: tensor<4096x4096xi8>) -> tensor<384x4096xi32> {
  %c0   = arith.constant 0 : i32
  %init = tensor.empty() : tensor<384x4096xi32>
  %acc  = linalg.fill ins(%c0 : i32) outs(%init : tensor<384x4096xi32>)
              -> tensor<384x4096xi32>
  %r    = linalg.matmul
              ins(%lhs, %rhs : tensor<384x4096xi8>, tensor<4096x4096xi8>)
              outs(%acc : tensor<384x4096xi32>) -> tensor<384x4096xi32>
  return %r : tensor<384x4096xi32>
}
