#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/packed_stride.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ct = cutlass;
using namespace cute;
using Half = ct::half_t;
using Row = ct::layout::RowMajor;
using Col = ct::layout::ColumnMajor;
constexpr int M = 4096, N = 4096, K = 4096;
constexpr int batches = 9, repeats = 20;

void check(cudaError_t s) {
    if (s != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(s)); std::exit(1); }
}
void check(ct::Status s) {
    if (s != ct::Status::kSuccess) { std::fprintf(stderr, "CUTLASS: %s\n", cutlassGetStatusString(s)); std::exit(1); }
}

// Both variants use Blackwell Tensor Cores, 2x1 clusters, three pipeline stages
// and 128x128 output elements per CTA. Only the pair shares an MMA instruction.
template<bool pair>
struct Benchmark {
    using Tile = Shape<Int<pair ? 256 : 128>, _128, _64>;
    using Cluster = Shape<_2,_1,_1>;
    using Arch = ct::arch::Sm100;
    using Op = ct::arch::OpClassTensorOp;
    using EpiSchedule = std::conditional_t<pair, ct::epilogue::TmaWarpSpecialized2Sm,
                                                 ct::epilogue::TmaWarpSpecialized1Sm>;
    using MainSchedule = std::conditional_t<pair, ct::gemm::KernelTmaWarpSpecialized2SmSm100,
                                                  ct::gemm::KernelTmaWarpSpecialized1SmSm100>;
    using Epi = typename ct::epilogue::collective::CollectiveBuilder<
        Arch, Op, Tile, Cluster, ct::epilogue::collective::EpilogueTileAuto,
        float, float, float, Row, 4, float, Row, 4, EpiSchedule>::CollectiveOp;
    using Main = typename ct::gemm::collective::CollectiveBuilder<
        Arch, Op, Half, Row, 8, Half, Col, 8, float, Tile, Cluster,
        ct::gemm::collective::StageCount<3>, MainSchedule>::CollectiveOp;
    using Kernel = ct::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>;
    using Gemm = ct::gemm::device::GemmUniversalAdapter<Kernel>;
    Gemm gemm;
    ct::DeviceAllocation<uint8_t> workspace;

    Benchmark(Half* a, Half* b, float* d) {
        auto sa = ct::make_cute_packed_stride(typename Kernel::StrideA{}, {M,K,1});
        auto sb = ct::make_cute_packed_stride(typename Kernel::StrideB{}, {N,K,1});
        auto sd = ct::make_cute_packed_stride(typename Kernel::StrideD{}, {M,N,1});
        typename Gemm::Arguments args{ct::gemm::GemmUniversalMode::kGemm, {M,N,K,1},
                                      {a,sa,b,sb}, {{1.f,0.f},d,sd,d,sd}};
        check(gemm.can_implement(args));
        size_t bytes = Gemm::get_workspace_size(args);
        if (bytes) workspace.reset(bytes);
        check(gemm.initialize(args, workspace.get()));
    }
};

// Periodic row/column patterns permit an exact CPU reference for every output
// without a separate O(M*N*K) CPU GEMM. Small integers sum exactly in FP32.
__host__ __device__ int a_value(int row, int k) { return ((row % 17 + 1)*k + row % 17) % 13 - 6; }
__host__ __device__ int b_value(int col, int k) { return ((col % 19 + 1)*k + col % 19) % 11 - 5; }
__global__ void fill(Half* a, Half* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M*K) a[i] = Half(a_value(i/K, i%K));
    if (i < N*K) b[i] = Half(b_value(i/K, i%K));
}

int main() {
    ct::DeviceAllocation<Half> a(M*K), b(N*K);
    ct::DeviceAllocation<float> d1(M*N), d2(M*N);
    fill<<<(std::max(M,N)*K+255)/256,256>>>(a.get(), b.get());
    check(cudaGetLastError());
    // Use distinct sentinels so an unwritten output cannot pass in both modes.
    check(cudaMemset(d1.get(), 0, M*N*sizeof(float)));
    check(cudaMemset(d2.get(), 0x40, M*N*sizeof(float)));
    Benchmark<false> one(a.get(), b.get(), d1.get());
    Benchmark<true> two(a.get(), b.get(), d2.get());
    for (int i=0; i<3; ++i) { check(one.gemm.run()); check(two.gemm.run()); }
    check(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    float ms[2][batches];
    for (int batch=0; batch<batches; ++batch) {
        for (int slot=0; slot<2; ++slot) {
            int mode = (batch + slot) % 2;
            check(cudaEventRecord(start));
            for (int i=0; i<repeats; ++i) check(mode ? two.gemm.run() : one.gemm.run());
            check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
            float elapsed;
            check(cudaEventElapsedTime(&elapsed, start, stop));
            ms[mode][batch] = elapsed / repeats;
        }
    }
    int ref[17][19]{};
    for (int r=0; r<17; ++r) for (int c=0; c<19; ++c)
        for (int k=0; k<K; ++k) ref[r][c] += a_value(r,k) * b_value(c,k);
    std::vector<float> output(M*N);
    std::printf("GEMM %dx%dx%d FP16/FP32, same inputs; %d batches x %d GEMMs per mode\n",M,N,K,batches,repeats);
    for (int mode=0; mode<2; ++mode) {
        (mode ? d2 : d1).copy_to_host(output.data(), output.size());
        for (int i=0; i<M*N; ++i) {
            int expected = ref[(i/N)%17][(i%N)%19];
            if (output[i] != expected) {
                std::fprintf(stderr,"%d-CTA FAIL at %d,%d: %g expected %d\n",mode+1,i/N,i%N,output[i],expected);
                return 1;
            }
        }
        std::sort(ms[mode], ms[mode]+batches);
        float median = ms[mode][batches/2];
        std::printf("%d-CTA PASS  median %.4f ms/GEMM  %.1f TFLOP/s  batch range %.4f..%.4f ms/GEMM\n",
                    mode+1, median, 2.0*M*N*K/(median*1e9), ms[mode][0], ms[mode][batches-1]);
    }
    std::printf("1-CTA/2-CTA %.3fx (>1: 2-CTA faster)\n",ms[0][batches/2]/ms[1][batches/2]);
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
}
