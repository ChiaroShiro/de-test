#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
flags=(-std=c++17 -O2 -arch=sm_100a --expt-relaxed-constexpr --extended-lambda)
nvcc "${flags[@]}" checks.cu -o build/checks
./build/checks

# Reuse NVIDIA's real 2-SM instruction and CPU reference. Shrink the tutorial's
# 4x4 cluster to 2x1 so exactly one pair computes a 256x256 output tile.
cutlass=${CUTLASS_DIR:-"$PWD/cutlass-v4.2.1"}
if [[ -z ${CUTLASS_DIR:-} && ! -d "$cutlass" ]]; then
    git clone --quiet --branch v4.2.1 --depth 1 https://github.com/NVIDIA/cutlass.git "$cutlass"
fi
tutorial="$cutlass/examples/cute/tutorial/blackwell"
src="$tutorial/04_mma_tma_2sm_sm100.cu"
old='auto cluster_shape = make_shape(Int<4>{}, Int<4>{}, Int<1>{});'
grep -Fq "$old" "$src" || { echo 'Expected CUTLASS v4.2.1 tutorial' >&2; exit 1; }
sed -e 's/auto cluster_shape = make_shape(Int<4>{}, Int<4>{}, Int<1>{});/auto cluster_shape = make_shape(Int<2>{}, Int<1>{}, Int<1>{});/' \
    -e '/using SMEMStorage = SharedStorage<TypeA, TypeB,/a\
  printf("Operand SMEM per CTA: A=%zu B, B=%zu B; full B tile=%zu B\\n", sizeof(SMEMStorage::A), sizeof(SMEMStorage::B), size_t(bN) * size_t(bK) * sizeof(TypeB));' \
    "$src" > build/mma_pair.cu
nvcc "${flags[@]}" -I"$cutlass/include" -I"$cutlass/tools/util/include" -I"$tutorial" \
    build/mma_pair.cu -o build/mma_pair -lcuda
./build/mma_pair 256 256 64 > build/mma.log || { cat build/mma.log; exit 1; }
grep -E 'Operand SMEM|Grid launched|Cluster launched|relative error|Execution is' build/mma.log || true
# The tutorial also exits zero for a numerical failure or a skipped build.
grep -Fxq 'Execution is successful.' build/mma.log || { cat build/mma.log; exit 1; }
echo '2-CTA MMA PASS  256x256x64 FP16 GEMM checked against CPU reference'

nvcc "${flags[@]}" -diag-suppress=177 -I"$cutlass/include" -I"$cutlass/tools/util/include" \
    gemm_compare.cu -o build/gemm_compare -lcuda
./build/gemm_compare
