# Makefile — Milestone B: standalone IME microkernel + validation
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
# Run validator (pin to Cluster-0 cores that have the IME unit):
#   taskset -c 0-3 ./mmt4d_ime_validate
#
# NOTES:
#   * -march must include 'v' (RVV 1.0) and 'xsmtvdot' (IME extension).
#   * If your assembler predates LLVM 20 and does not recognise 'smt.vmadot',
#     compile with -DIME_USE_DOT_INSN and the .insn encoding in the .c file
#     will be used instead.
#   * The validator links the kernel and reference into a single binary;
#     it does NOT use IREE's build system.  The same kernel file will later
#     be compiled into IREE's ukernel bitcode by the IREE CMake/Bazel build.

CC       ?= riscv64-unknown-linux-gnu-gcc
MARCH    ?= rv64gcv_xsmtvdot
CFLAGS   ?= -O2 -march=$(MARCH) -Wall -Wextra
CROSS_FLAGS ?= 


SRCS = mmt4d_ime_s8s8s32.c mmt4d_ime_validate.c
TARGET = mmt4d_ime_validate

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS) mmt4d_ime_s8s8s32.h
	$(CC) $(CFLAGS) $(CROSS_FLAGS) -o $@ $(SRCS)

run: $(TARGET)
	taskset -c 0-3 ./$(TARGET)

clean:
	rm -f $(TARGET)
