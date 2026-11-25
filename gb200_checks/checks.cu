#include <cuda_runtime.h>
#include <cuda/ptx>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;
#define CUDA(call) do { auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
    std::exit(1); } } while (0)

// Four single-thread CTAs. Rank 0 and rank 3 belong to different MMA pairs.
__global__ void __cluster_dims__(4, 1, 1) dsm_check(int* result) {
    __shared__ int value;
    auto cluster = cg::this_cluster();
    int rank = cluster.block_rank();
    value = 100 + rank;
    cluster.sync();
    if (rank == 0) {
        volatile int* remote = cluster.map_shared_rank(&value, 3);
        result[0] = *remote;
        *remote = 777;
    }
    cluster.sync();
    if (rank == 3) result[1] = value;
}

constexpr int tasks = 32768;
constexpr int lanes = 32;
__host__ __device__ unsigned work(int task, int lane) {
    unsigned x = unsigned(task * lanes + lane);
    int rounds = (task % 8 == 0) ? 192 : 12;
    for (int i = 0; i < rounds; ++i) x = (x ^ (x >> 13)) * 0x85ebca6bu + 1;
    return x;
}

// One warp per CTA. Both schedulers start with blockIdx.x and ask for one
// additional task at a time. Only lane 0 touches either scheduler.
template<bool use_clc>
__global__ void schedule(int* head, unsigned* output, int* visits, int* owner) {
    [[maybe_unused]] __shared__ uint4 reply;
    [[maybe_unused]] __shared__ std::uint64_t barrier;
    [[maybe_unused]] int phase = 0;
    int lane = threadIdx.x, task = blockIdx.x;
    if constexpr (use_clc) {
        if (lane == 0) ptx::mbarrier_init(&barrier, 1);
        __syncwarp();
    }
    while (task < tasks) {
        int next = tasks;
        if (lane == 0) {
            if constexpr (use_clc) {
                ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);
                cg::invoke_one(cg::coalesced_threads(), [&] {
                    ptx::clusterlaunchcontrol_try_cancel(&reply, &barrier);
                });
                ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &barrier, sizeof(reply));
            } else {
                next = atomicAdd(head, 1);
            }
        }
        unsigned sum = __reduce_add_sync(0xffffffffu, work(task, lane));
        if (lane == 0) {
            output[task] = sum;
            if (visits) {  // Separate, untimed correctness run.
                atomicAdd(&visits[task], 1);
                owner[task] = blockIdx.x;
            }
            if constexpr (use_clc) {
                while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &barrier, phase)) {}
                phase ^= 1;
                if (ptx::clusterlaunchcontrol_query_cancel_is_canceled(reply)) {
                    next = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(reply);
                    ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
                }
            }
        }
        task = __shfl_sync(0xffffffffu, next, 0);
    }
}

int main() {
    cudaDeviceProp gpu{};
    CUDA(cudaGetDeviceProperties(&gpu, 0));
    std::printf("GPU: %s, SM %d.%d, %d SMs\n", gpu.name, gpu.major, gpu.minor, gpu.multiProcessorCount);
    if (gpu.major != 10 || gpu.minor != 0) return 1;

    int *head, *visits, *owner;
    unsigned* output;
    CUDA(cudaMalloc(&visits, tasks * sizeof(int)));
    dsm_check<<<4, 1>>>(visits);
    CUDA(cudaGetLastError());
    int observed[2];
    CUDA(cudaMemcpy(observed, visits, sizeof(observed), cudaMemcpyDeviceToHost));
    bool dsm_ok = observed[0] == 103 && observed[1] == 777;
    std::printf("DSM  CTA0 read CTA3: %d (expect 103); CTA3 saw write: %d (expect 777)  %s\n",
                observed[0], observed[1], dsm_ok ? "PASS" : "FAIL");
    if (!dsm_ok) return 1;

    CUDA(cudaMalloc(&head, sizeof(int)));
    CUDA(cudaMalloc(&output, tasks * sizeof(unsigned)));
    CUDA(cudaMalloc(&owner, tasks * sizeof(int)));
    int per_sm;
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, schedule<false>, lanes, 0));
    int workers = std::min(tasks, per_sm * gpu.multiProcessorCount);
    std::printf("Tasks: %d, threads/CTA: %d, atomic worker CTAs: %d, CLC logical CTAs: %d\n",
                tasks, lanes, workers, tasks);

    cudaEvent_t start, stop;
    CUDA(cudaEventCreate(&start));
    CUDA(cudaEventCreate(&stop));
    std::vector<float> times[2];
    std::vector<int> counts(tasks), owners(tasks);
    std::vector<unsigned> actual(tasks), expected(tasks);
    for (int t = 0; t < tasks; ++t)
        for (int l = 0; l < lanes; ++l) expected[t] += work(t, l);

    // Run 0 checks every task, run 1 warms up, runs 2..12 measure the median.
    // Alternate order so one scheduler is not always the first to run.
    for (int run = 0; run < 13; ++run) {
        for (int slot = 0; slot < 2; ++slot) {
            int clc = (run + slot) % 2;
            if (!clc) CUDA(cudaMemcpy(head, &workers, sizeof(workers), cudaMemcpyHostToDevice));
            if (run == 0) CUDA(cudaMemset(visits, 0, tasks * sizeof(int)));
            int* audit = run == 0 ? visits : nullptr;
            CUDA(cudaEventRecord(start));
            if (clc) schedule<true><<<tasks, lanes>>>(head, output, audit, owner);
            else schedule<false><<<workers, lanes>>>(head, output, audit, owner);
            CUDA(cudaEventRecord(stop));
            CUDA(cudaGetLastError());
            CUDA(cudaEventSynchronize(stop));
            if (run >= 2) {
                float ms;
                CUDA(cudaEventElapsedTime(&ms, start, stop));
                times[clc].push_back(ms);
            }
            if (run == 0) {
                CUDA(cudaMemcpy(counts.data(), visits, tasks * sizeof(int), cudaMemcpyDeviceToHost));
                CUDA(cudaMemcpy(owners.data(), owner, tasks * sizeof(int), cudaMemcpyDeviceToHost));
                CUDA(cudaMemcpy(actual.data(), output, tasks * sizeof(unsigned), cudaMemcpyDeviceToHost));
                int taken = 0;
                for (int t = 0; t < tasks; ++t) {
                    if (counts[t] != 1 || actual[t] != expected[t]) {
                        std::fprintf(stderr, "%s FAIL task=%d visits=%d value=%u expected=%u\n",
                                     clc ? "CLC" : "atomic", t, counts[t], actual[t], expected[t]);
                        return 1;
                    }
                    taken += owners[t] != t;
                }
                std::printf("%-6s PASS  %d/%d correct, exactly once; %s=%d\n",
                            clc ? "CLC" : "atomic", tasks, tasks, clc ? "stolen" : "queue claims", taken);
                if (clc && !taken) std::puts("CLC: no successful cancellation observed in this audit.");
            }
        }
    }
    for (auto& samples : times) std::sort(samples.begin(), samples.end());
    float atomic_ms = times[0][5], clc_ms = times[1][5];
    std::printf("GPU median of 11: atomic %.4f ms | CLC %.4f ms | atomic/CLC %.3fx (>1: CLC faster)\n",
                atomic_ms, clc_ms, atomic_ms / clc_ms);
    CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(stop));
    CUDA(cudaFree(head)); CUDA(cudaFree(output)); CUDA(cudaFree(visits)); CUDA(cudaFree(owner));
}
