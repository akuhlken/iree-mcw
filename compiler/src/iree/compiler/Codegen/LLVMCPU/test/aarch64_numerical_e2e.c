// Numerical end-to-end verification for the AArch64 `fmla` intrinsics used by
// the inner_tiled path: `llvm.fma.v4f32` (NEON) and `llvm.aarch64.sve.fmla`
// (SVE). Cross-compiled with clang to `aarch64-none-elf` and run under
// qemu-aarch64 (see run_aarch64_numerical_test.sh). Returns exit code 0 when
// every matmul result matches the expected value, non-zero otherwise.
//
// Note: freestanding (no libc), so results are reported via the exit syscall
// and arrays are initialized with explicit loops to avoid memcpy/memset.

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#include <arm_neon.h>

// Exit syscall (aarch64 linux: __NR_exit = 93, code in x0).
static void exit_with(int code) {
  __asm__ volatile("mov x0, %0\nmov x8, #93\nsvc #0"
                   :
                   : "r"((long)code)
                   : "x0", "x8", "memory");
  __builtin_unreachable();
}

// Absolute difference without libm. `fmla` is a fused multiply-add (single
// rounding) while a scalar reference would lower to `llvm.fmuladd` (which may
// round twice), so an exact `==` comparison against a scalar reference is too
// strict; the expected values below are exact integers and the tolerance only
// guards against last-ULP differences.
static float fabsf_manual(float x) {
  return x < 0.0f ? -x : x;
}

// A and B are 4x4 row-major matrices (f32).
// A = [[1,2,3,4],[5,6,7,8],[2,3,4,5],[6,7,8,9]]
// B = [[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]]
// C = A*B =
//   [[ 30, 40, 50, 60],
//    [ 70, 96,122,148],
//    [ 40, 54, 68, 82],
//    [ 80,110,140,170]]
static void init_matrices(float A[16], float B[16]) {
  int i;
  for (i = 0; i < 16; i++) { A[i] = 0.0f; B[i] = 0.0f; }
  A[0]=1;A[1]=2;A[2]=3;A[3]=4;
  A[4]=5;A[5]=6;A[6]=7;A[7]=8;
  A[8]=2;A[9]=3;A[10]=4;A[11]=5;
  A[12]=6;A[13]=7;A[14]=8;A[15]=9;
  B[0]=1;B[1]=2;B[2]=3;B[3]=4;
  B[4]=2;B[5]=3;B[6]=4;B[7]=5;
  B[8]=3;B[9]=4;B[10]=5;B[11]=6;
  B[12]=4;B[13]=5;B[14]=6;B[15]=7;
}

static int match_expected(float C[16]) {
  static const float expected[16] = {
      30, 40, 50, 60, 70, 96, 122, 148,
      40, 54, 68, 82, 80, 110, 140, 170};
  int i;
  for (i = 0; i < 16; i++)
    if (fabsf_manual(C[i] - expected[i]) > 1e-3f) return 1;
  return 0;
}

// 4x4 matmul C = A * B using NEON `vfmaq_f32` (fmla v.4s).
static int neon_matmul(void) {
  float A[16], B[16], C[16];
  int i, k;
  init_matrices(A, B);
  for (i = 0; i < 16; i++) C[i] = 0.0f;
  for (i = 0; i < 4; i++) {
    float32x4_t crow = vdupq_n_f32(0.0f);
    for (k = 0; k < 4; k++) {
      float32x4_t brow = vld1q_f32(&B[k * 4]);
      crow = vfmaq_n_f32(crow, brow, A[i * 4 + k]);
    }
    vst1q_f32(&C[i * 4], crow);
  }
  return match_expected(C);
}

#ifdef __ARM_FEATURE_SVE
// Same matmul using SVE `svmla` (fmla z.s).
static int sve_matmul(void) {
  float A[16], B[16], C[16];
  int i, k;
  init_matrices(A, B);
  for (i = 0; i < 16; i++) C[i] = 0.0f;
  svbool_t pg = svptrue_b32();
  for (i = 0; i < 4; i++) {
    svfloat32_t crow = svdup_f32(0.0f);
    for (k = 0; k < 4; k++) {
      svfloat32_t brow = svld1_f32(pg, &B[k * 4]);
      crow = svmla_n_f32_m(pg, crow, brow, A[i * 4 + k]);
    }
    svst1_f32(pg, &C[i * 4], crow);
  }
  return match_expected(C);
}
#endif

void _start(void) {
  int code = neon_matmul();
#ifdef __ARM_FEATURE_SVE
  code |= sve_matmul();
#endif
  exit_with(code);
}
