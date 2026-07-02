#!/bin/bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build a local RISC-V cross-toolchain wrapper for Milestone C (+xsmtvdot).
#
# Uses a host Clang that accepts -march=rv64gc_xsmtvdot (typically from
# build-host) with sysroot, libgcc, and binutils from an existing RISC-V
# toolchain install.
#
# Output (gitignored): toolchains/riscv-xsmtvdot/
#
# Required environment variable:
#   RISCV_BASE_TOOLCHAIN  Root of the stock RISC-V bundle (contains bin/,
#                         sysroot/, lib/gcc/, …). Same layout as IREE's
#                         linux_riscv64.cmake expects for RISCV_TOOLCHAIN_ROOT.
#
# Optional:
#   RISCV_HOST_CLANG      Path to host clang (must support +xsmtvdot).
#   IREE_XSMTVDOT_TOOLCHAIN_DIR  Output directory (default: toolchains/riscv-xsmtvdot).
#
# Example:
#   export RISCV_BASE_TOOLCHAIN=/path/to/riscv/clang/linux/RISCV
#   ./toolchains/setup-riscv-xsmtvdot.sh
#   export RISCV_TOOLCHAIN_ROOT="$(pwd)/toolchains/riscv-xsmtvdot"
#   ./build_tools/cmake/build_riscv.sh build-riscv

set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0")

Creates toolchains/riscv-xsmtvdot/ (wrapper) for cross-compiling with +xsmtvdot.

Required:
  RISCV_BASE_TOOLCHAIN   Stock RISC-V toolchain root (sysroot + binutils + libgcc)

Optional:
  RISCV_HOST_CLANG       Host clang binary (default: build-host/llvm-project/bin/clang)
  IREE_XSMTVDOT_TOOLCHAIN_DIR  Output path (default: <repo>/toolchains/riscv-xsmtvdot)

Example:
  export RISCV_BASE_TOOLCHAIN=/opt/riscv/linux/RISCV
  $(basename "$0")
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${IREE_XSMTVDOT_TOOLCHAIN_DIR:-${REPO_ROOT}/toolchains/riscv-xsmtvdot}"

if [[ -z "${RISCV_BASE_TOOLCHAIN:-}" ]]; then
  echo "error: RISCV_BASE_TOOLCHAIN is not set." >&2
  echo >&2
  usage >&2
  exit 1
fi

RISCV_BASE_TOOLCHAIN="$(realpath "${RISCV_BASE_TOOLCHAIN}")"

resolve_host_clang() {
  if [[ -n "${RISCV_HOST_CLANG:-}" ]]; then
    echo "$(realpath "${RISCV_HOST_CLANG}")"
    return
  fi
  local candidates=(
    "${REPO_ROOT}/build-host/llvm-project/bin/clang"
    "${REPO_ROOT}/build-host/install/bin/clang"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      echo "$(realpath "${candidate}")"
      return
    fi
  done
  echo "error: no host clang found. Build build-host first or set RISCV_HOST_CLANG." >&2
  exit 1
}

HOST_CLANG="$(resolve_host_clang)"
HOST_CLANG_DIR="$(dirname "${HOST_CLANG}")"
HOST_CLANGXX="${HOST_CLANG_DIR}/clang++"
if [[ ! -x "${HOST_CLANGXX}" ]]; then
  HOST_CLANGXX="${HOST_CLANG}"
fi

BASE_BIN="${RISCV_BASE_TOOLCHAIN}/bin"
BASE_SYSROOT="${RISCV_BASE_TOOLCHAIN}/sysroot"

for path in "${BASE_BIN}" "${BASE_SYSROOT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "error: RISCV_BASE_TOOLCHAIN is missing expected path: ${path}" >&2
    exit 1
  fi
done

for tool in llvm-ar llvm-ranlib llvm-strip llvm-nm llvm-objdump; do
  if [[ ! -x "${BASE_BIN}/${tool}" ]]; then
    echo "error: RISCV_BASE_TOOLCHAIN is missing ${BASE_BIN}/${tool}" >&2
    exit 1
  fi
done

echo "Checking +xsmtvdot support in ${HOST_CLANG} ..."
if ! "${HOST_CLANG}" \
  --target=riscv64-unknown-linux-gnu \
  --sysroot="${BASE_SYSROOT}" \
  --gcc-toolchain="${RISCV_BASE_TOOLCHAIN}" \
  -march=rv64gc_xsmtvdot -c -x c /dev/null -o /dev/null 2>/dev/null; then
  echo "error: ${HOST_CLANG} does not accept -march=rv64gc_xsmtvdot." >&2
  echo "Build build-host from this branch or set RISCV_HOST_CLANG to a new enough Clang." >&2
  exit 1
fi

echo "Writing wrapper toolchain to ${OUTPUT_DIR}"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/bin"

cat >"${OUTPUT_DIR}/bin/clang" <<EOF
#!/bin/bash
exec ${HOST_CLANG} \\
  --target=riscv64-unknown-linux-gnu \\
  --sysroot=${BASE_SYSROOT} \\
  --gcc-toolchain=${RISCV_BASE_TOOLCHAIN} \\
  "\$@"
EOF

cat >"${OUTPUT_DIR}/bin/clang++" <<EOF
#!/bin/bash
exec ${HOST_CLANGXX} \\
  --target=riscv64-unknown-linux-gnu \\
  --sysroot=${BASE_SYSROOT} \\
  --gcc-toolchain=${RISCV_BASE_TOOLCHAIN} \\
  "\$@"
EOF

chmod +x "${OUTPUT_DIR}/bin/clang" "${OUTPUT_DIR}/bin/clang++"

for tool in llvm-ar llvm-ranlib llvm-strip llvm-nm llvm-objdump; do
  ln -sfn "${BASE_BIN}/${tool}" "${OUTPUT_DIR}/bin/${tool}"
done

for dir in lib libexec include share riscv64-unknown-linux-gnu sysroot; do
  if [[ -e "${RISCV_BASE_TOOLCHAIN}/${dir}" ]]; then
    ln -sfn "${RISCV_BASE_TOOLCHAIN}/${dir}" "${OUTPUT_DIR}/${dir}"
  fi
done

echo "Done."
echo
echo "Next steps:"
echo "  export RISCV_TOOLCHAIN_ROOT=${OUTPUT_DIR}"
echo "  export IREE_HOST_BIN_DIR=${REPO_ROOT}/build-host/install/bin"
echo "  ./build_tools/cmake/build_riscv.sh build-riscv"
