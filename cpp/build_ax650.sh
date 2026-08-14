#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPP_DIR="${ROOT}/cpp"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-${CPP_DIR}/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu}"
BSP_MSP_DIR="${BSP_MSP_DIR:-${CPP_DIR}/toolchains/ax650n_bsp_sdk/msp/out}"
BUILD_DIR="${CPP_DIR}/build/ax650"

if [[ ! -x "${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-g++" ]]; then
  echo "ERROR: AArch64 toolchain not found: ${TOOLCHAIN_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${BSP_MSP_DIR}/include/ax_engine_api.h" ]]; then
  echo "ERROR: AX650 BSP msp/out not found: ${BSP_MSP_DIR}" >&2
  exit 2
fi

mkdir -p "${BUILD_DIR}" "${CPP_DIR}/bin"
cmake -S "${CPP_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER="${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-gcc" \
  -DCMAKE_CXX_COMPILER="${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-g++" \
  -DAXERA_TARGET=AX650 \
  -DBSP_MSP_DIR="${BSP_MSP_DIR}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"
mkdir -p "${CPP_DIR}/bin" "${CPP_DIR}/lib/ax650"
cp "${CPP_DIR}/bin/ax650/openwakeword_ax650" "${CPP_DIR}/bin/"
"${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-strip" --strip-unneeded \
  "${CPP_DIR}/bin/openwakeword_ax650"
echo "Built ${CPP_DIR}/bin/openwakeword_ax650 and ${CPP_DIR}/lib/ax650/"
