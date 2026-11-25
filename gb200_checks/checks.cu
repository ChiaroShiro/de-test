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

constexpr int ring_size = 1024, hop = 67, steps = 4097, samples = 11;
constexpr int exchanges = 31;
struct DsmResult {
    unsigned sm[4], end_sm[4];
    int observed[2];
    unsigned long long ticks[2][2][samples + 1]; // direction, local/peer, sample
    unsigned long long bulk_ns[2][samples + 1];
    unsigned bulk_error;
};

__device__ unsigned sm_id() {
    unsigned id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return id;
}

__device__ unsigned long long timer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}

// Two active CTAs simultaneously copy one 4 KiB block each, then synchronize.
// Each round reads the previous round's result. Only rank 0 times the exchange.
__device__ void bulk_exchange(unsigned* buf, DsmResult* result) {
    auto cluster = cg::this_cluster();
    int rank = cluster.block_rank(), tid = threadIdx.x;
    bool active = rank == 0 || rank == 3;
    for (int s = 0; s <= samples; ++s) {
        for (int turn = 0; turn < 2; ++turn) {
            int remote = (s + turn) % 2;
            volatile unsigned* dst = remote ? cluster.map_shared_rank(buf, 3-rank) : buf;
            for (int i = tid; i < ring_size; i += blockDim.x) buf[i] = rank * ring_size + i;
            cluster.sync();
            unsigned long long start = 0;
            if (rank == 0 && tid == 0) start = timer_ns();
            cluster.sync();
            for (int r = 0; r < exchanges; ++r) {
                int src_page = (r % 2) * ring_size;
                if (active)
                    for (int i = tid; i < ring_size; i += blockDim.x)
                        dst[ring_size - src_page + i] = buf[src_page + i] + 1;
                cluster.sync();
            }
            if (rank == 0 && tid == 0) result->bulk_ns[remote][s] = timer_ns() - start;
            int source = remote ? 3-rank : rank; // An odd number of exchanges.
            if (active)
                for (int i = tid; i < ring_size; i += blockDim.x)
                    if (buf[ring_size + i] != unsigned(source * ring_size + i + exchanges))
                        atomicOr(&result->bulk_error, 1u);
        }
    }
}

// Each loaded index determines the next address. Start and end clocks both
// have data dependencies so the last memory access is included.
template<bool remote>
__device__ unsigned long long chase(unsigned base) {
    unsigned long long begin, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(begin) :: "memory");
    unsigned first = unsigned(begin) & (ring_size - 1), index = first;
    #pragma unroll 1
    for (int i = 0; i < steps; ++i) {
        unsigned address = base + 4 * index;
        if constexpr (remote)
            asm volatile("ld.shared::cluster.u32 %0,[%1];" : "=r"(index) : "r"(address) : "memory");
        else
            asm volatile("ld.shared::cta.u32 %0,[%1];" : "=r"(index) : "r"(address) : "memory");
    }
    asm volatile("{ .reg .pred ok; setp.eq.u32 ok,%1,%2; mov.u64 %0,0; @ok mov.u64 %0,%%clock64; }"
                 : "=l"(end) : "r"(index), "r"((first + steps * hop) & (ring_size - 1)) : "memory");
    return end ? end - begin : 0;
}

__global__ void __cluster_dims__(4, 1, 1) dsm_check(DsmResult* result) {
    __shared__ int value;
    __shared__ unsigned ring[2 * ring_size];
    auto cluster = cg::this_cluster();
    int rank = cluster.block_rank();
    if (threadIdx.x == 0) {
        result->sm[rank] = sm_id();
        value = 100 + rank;
    }
    for (int i = threadIdx.x; i < ring_size; i += blockDim.x) ring[i] = (i + hop) & (ring_size - 1);
    cluster.sync();
    if (rank == 0 && threadIdx.x == 0) {
        volatile int* remote = cluster.map_shared_rank(&value, 3);
        result->observed[0] = *remote;
        *remote = 777;
    }
    cluster.sync();
    if (rank == 3 && threadIdx.x == 0) result->observed[1] = value;
    for (int direction = 0; direction < 2; ++direction) {
        int src = direction * 3, dst = 3 - src;
        if (rank == src && threadIdx.x == 0) {
            unsigned local = __cvta_generic_to_shared(ring), peer;
            asm("mapa.shared::cluster.u32 %0,%1,%2;" : "=r"(peer) : "r"(local), "r"(dst));
            for (int s = 0; s <= samples; ++s) {
                for (int turn = 0; turn < 2; ++turn) {
                    int remote = (s + turn) % 2;
                    result->ticks[direction][remote][s] = remote ? chase<true>(peer) : chase<false>(local);
                }
            }
        }
        cluster.sync(); // Only one active reader; keep every remote allocation alive.
    }
    bulk_exchange(ring, result);
    if (threadIdx.x == 0) result->end_sm[rank] = sm_id();
}

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
__global__ void schedule(int tasks, int* head, unsigned* output, int* visits, int* owner) {
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

int run_dsm() {
    DsmResult dsm{}, *device_dsm;
    CUDA(cudaMalloc(&device_dsm, sizeof(dsm)));
    CUDA(cudaMemset(device_dsm, 0, sizeof(dsm)));
    dsm_check<<<4, 128>>>(device_dsm);
    CUDA(cudaGetLastError());
    CUDA(cudaMemcpy(&dsm, device_dsm, sizeof(dsm), cudaMemcpyDeviceToHost));
    CUDA(cudaFree(device_dsm));
    for (int i = 0; i < 4; ++i)
        std::printf("DSM  CTA%d: SM %u (end %u)\n", i, dsm.sm[i], dsm.end_sm[i]);
    const int* observed = dsm.observed;
    bool dsm_ok = observed[0] == 103 && observed[1] == 777;
    std::printf("DSM  CTA0 read CTA3: %d (expect 103); CTA3 saw write: %d (expect 777)  %s\n",
                observed[0], observed[1], dsm_ok ? "PASS" : "FAIL");
    if (!dsm_ok) return 1;
    for (int dir = 0; dir < 2; ++dir) {
        double cycles[2];
        for (int mode = 0; mode < 2; ++mode) {
            auto* t = dsm.ticks[dir][mode] + 1; // Drop warm-up.
            std::sort(t, t + samples);
            if (!t[0]) { std::fprintf(stderr, "DSM pointer chain FAIL\n"); return 1; }
            cycles[mode] = double(t[samples / 2]) / steps;
        }
        bool stable = dsm.sm[0] == dsm.end_sm[0] && dsm.sm[3] == dsm.end_sm[3];
        const char* placement = !stable ? "SM ID changed" : dsm.sm[0] == dsm.sm[3] ? "same SM" : "different SMs";
        std::printf("DSM  CTA%d -> CTA%d (%s): local %.1f | peer %.1f cycles/load | peer/local %.2fx\n",
                    3*dir, 3-3*dir, placement, cycles[0], cycles[1], cycles[1]/cycles[0]);
    }

    if (dsm.bulk_error) { std::fprintf(stderr, "DSM block exchange FAIL\n"); return 1; }
    double payload = 2.0 * ring_size * sizeof(unsigned) * exchanges;
    for (int remote = 0; remote < 2; ++remote) {
        auto* t = dsm.bulk_ns[remote] + 1;
        std::sort(t, t + samples);
        if (!t[0]) return 1;
        std::printf("DSM  4 KiB per CTA, 2 senders: %-5s %.2f GB/s (payload, includes cluster sync) PASS\n",
                    remote ? "peer" : "local", payload / t[samples / 2]);
    }
    return 0;
}

int run_schedulers(int tasks, int sms) {
    int *head, *visits, *owner;
    unsigned* output;
    CUDA(cudaMalloc(&visits, tasks * sizeof(int)));
    CUDA(cudaMalloc(&head, sizeof(int)));
    CUDA(cudaMalloc(&output, tasks * sizeof(unsigned)));
    CUDA(cudaMalloc(&owner, tasks * sizeof(int)));
    int per_sm;
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, schedule<false>, lanes, 0));
    int workers = std::min(tasks, per_sm * sms);
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
            if (clc) schedule<true><<<tasks, lanes>>>(tasks, head, output, audit, owner);
            else schedule<false><<<workers, lanes>>>(tasks, head, output, audit, owner);
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
    return 0;
}

int main() {
    cudaDeviceProp gpu{};
    CUDA(cudaGetDeviceProperties(&gpu, 0));
    std::printf("GPU: %s, SM %d.%d, %d SMs\n", gpu.name, gpu.major, gpu.minor, gpu.multiProcessorCount);
    if (gpu.major != 10 || gpu.minor != 0 || run_dsm()) return 1;
    for (int tasks : {32768, 327680})
        if (run_schedulers(tasks, gpu.multiProcessorCount)) return 1;
    return 0;
}
