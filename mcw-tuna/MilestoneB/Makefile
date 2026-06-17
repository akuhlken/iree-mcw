# Makefile — Milestone B: standalone IME microkernels + validation + benchmark
#
# Usage (on BPI-F3, cross-compile, or QEMU):
#
#   # Native build on BPI-F3 (SpaceMiT-provided GCC / Clang):
#   make CC=riscv64-linux-gnu-gcc
#
#   # Cross-compile with upstream GCC toolchain:
#   make CC=riscv64-unknown-linux-gnu-gcc SYSROOT=/path/to/sysroot
#
#   # Cross-compile with IREE's bundled clang (recommended for named mnemonics):
#   make CC=/path/to/iree/third_party/llvm-project/build/bin/clang \
#        CROSS_FLAGS="-target riscv64-linux-gnu --gcc-toolchain=/opt/riscv"
#
# Targets:
#   make            -> build validator (mmt4d_ime_validate)
#   make benchmark  -> build benchmark (mmt4d_ime_benchmark)
#   make run        -> run validator pinned to Cluster-0
#   make run-bench  -> run benchmark pinned to Cluster-0
#
# Run pinned to Cluster-0 cores that have the IME unit:
#   taskset -c 0-3 ./mmt4d_ime_validate
#   taskset -c 0-3 ./mmt4d_ime_benchmark
#
# NOTES:
#   * -march must include 'v' (RVV 1.0) and 'xsmtvdot' (IME extension).
#   * If your assembler predates LLVM 20 and does not recognise 'smt.vmadot',
#     compile with -DIME_USE_DOT_INSN and the .insn encoding in the .c file
#     will be used instead.
#   * These harnesses link the kernels directly; they do NOT use IREE's build
#     system. The same kernel file will later be compiled into IREE's ukernel
#     bitcode by the IREE CMake/Bazel build.

CC          ?= riscv64-unknown-linux-gnu-gcc
MARCH       ?= rv64gcv_xsmtvdot
CFLAGS      ?= -O2 -march=$(MARCH) -Wall -Wextra
CROSS_FLAGS ?=

KERNEL_SRC   = mmt4d_ime_s8s8s32.c
HEADER       = mmt4d_ime_s8s8s32.h

VALIDATE_SRCS  = $(KERNEL_SRC) mmt4d_ime_validate.c
VALIDATE_BIN   = mmt4d_ime_validate

BENCH_SRCS     = $(KERNEL_SRC) mmt4d_ime_benchmark.c
BENCH_BIN      = mmt4d_ime_benchmark

.PHONY: all benchmark clean run run-bench

all: $(VALIDATE_BIN)

benchmark: $(BENCH_BIN)

$(VALIDATE_BIN): $(VALIDATE_SRCS) $(HEADER)
	$(CC) $(CFLAGS) $(CROSS_FLAGS) -o $@ $(VALIDATE_SRCS)

$(BENCH_BIN): $(BENCH_SRCS) $(HEADER)
	$(CC) $(CFLAGS) $(CROSS_FLAGS) -o $@ $(BENCH_SRCS)

run: $(VALIDATE_BIN)
	taskset -c 0-3 ./$(VALIDATE_BIN)

run-bench: $(BENCH_BIN)
	taskset -c 0-3 ./$(BENCH_BIN)

clean:
	rm -f $(VALIDATE_BIN) $(BENCH_BIN)
