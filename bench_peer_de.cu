// GPU0 raw LZ4 chunks -> GPU1 hardware DE reads peer addresses -> GPU1 output.
// No payload cudaMemcpy in peer_decode(). CUDA >= 12.8; nvCOMP >= 4.2.
// Build/run through bench_peer_de.py. stdout: one JSON record; stderr: diagnostics.
// API: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvcomp/lz4.h>
#include <nvcomp/version.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if CUDA_VERSION < 12080
#error "peer-DE requires CUDA Toolkit headers >= 12.8 (cuMemBatchDecompressAsync)."
#endif
#if NVCOMP_VER_MAJOR < 4 || (NVCOMP_VER_MAJOR == 4 && NVCOMP_VER_MINOR < 2)
#error "Use nvCOMP >= 4.2 with Blackwell support."
#endif

// nvCOMP 4.x uses the older options name; 5.x also renamed the scratch query.
// Always compile and link against the same nvCOMP installation.
#if NVCOMP_VER_MAJOR < 5
using BenchLz4CompressOpts = nvcompBatchedLZ4Opts_t;
#define BENCH_LZ4_DEFAULT_OPTS nvcompBatchedLZ4DefaultOpts
#define BENCH_LZ4_TEMP_SIZE nvcompBatchedLZ4CompressGetTempSize
#elif NVCOMP_VER_MAJOR == 5
using BenchLz4CompressOpts = nvcompBatchedLZ4CompressOpts_t;
#define BENCH_LZ4_DEFAULT_OPTS nvcompBatchedLZ4CompressDefaultOpts
#define BENCH_LZ4_TEMP_SIZE nvcompBatchedLZ4CompressGetTempSizeAsync
#else
using BenchLz4CompressOpts = nvcompBatchedLZ4CompressOpts_t;
#define BENCH_LZ4_DEFAULT_OPTS nvcompBatchedLZ4CompressDefaultOpts
#define BENCH_LZ4_TEMP_SIZE nvcompBatchedLZ4CompressGetTempSize
#endif

static void rt(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));
}
static void drv(CUresult e, const char* where) {
    if (e == CUDA_SUCCESS) return;
    const char *name = nullptr, *desc = nullptr;
    cuGetErrorName(e, &name); cuGetErrorString(e, &desc);
    throw std::runtime_error(std::string(where)+": "+
        (name ? name : "CUDA_ERROR")+" / "+(desc ? desc : ""));
}
static void nc(nvcompStatus_t e, const char* where) {
    if (e != nvcompSuccess)
        throw std::runtime_error(std::string(where)+": nvCOMP status "+std::to_string(int(e)));
}
#define RT(x) rt((x), #x)
#define DR(x) drv((x), #x)
#define NC(x) nc((x), #x)

static size_t checked_mul(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max()/b)
        throw std::runtime_error("allocation size overflow");
    return a*b;
}
static size_t align16(size_t n) {
    if (n > std::numeric_limits<size_t>::max()-15)
        throw std::runtime_error("aligned size overflow");
    return (n+15)&~size_t(15);
}
struct Buffer {
    int dev;
    void* p = nullptr;
    size_t bytes;
    Buffer(int device, size_t n):dev(device),bytes(n) {
        RT(cudaSetDevice(dev));
        RT(cudaMalloc(&p, std::max(size_t(1), n)));
    }
    ~Buffer() {
        if (p) { cudaSetDevice(dev); cudaFree(p); }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* as() { return static_cast<T*>(p); }
    uint8_t* at(size_t n) { return as<uint8_t>()+n; }
};
struct Stream {
    int dev;
    cudaStream_t s = nullptr;
    explicit Stream(int d):dev(d) {
        RT(cudaSetDevice(d)); RT(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
    }
    ~Stream() { if(s) {cudaSetDevice(dev);cudaStreamSynchronize(s);cudaStreamDestroy(s);} }
    void wait() {RT(cudaSetDevice(dev));RT(cudaStreamSynchronize(s));}
};

// nvCOMP changed the LLIF signatures between releases. Compile against installed
// headers; never guess struct layout or C ABI from Python ctypes.
template<class F>
static nvcompStatus_t temp_size(F f, size_t count, size_t chunk,
        BenchLz4CompressOpts opts, size_t* bytes, size_t total, cudaStream_t s) {
    if constexpr (std::is_invocable_v<F,size_t,size_t,decltype(opts),size_t*,size_t,cudaStream_t>)
        return f(count,chunk,opts,bytes,total,s);
    else if constexpr (std::is_invocable_v<F,size_t,size_t,decltype(opts),size_t*,size_t>)
        return f(count,chunk,opts,bytes,total);
    else if constexpr (std::is_invocable_v<F,size_t,size_t,decltype(opts),size_t*>)
        return f(count,chunk,opts,bytes);
    else {
        static_assert(std::is_invocable_v<F,size_t,size_t,decltype(opts),size_t*>,
                      "Unrecognized nvCOMP LZ4 CompressGetTempSize signature");
    }
}
template<class F>
static nvcompStatus_t compress_call(F f, const void* const* input,
        const size_t* sizes, size_t chunk, size_t count, void* temp, size_t temp_bytes,
        void* const* output, size_t* compressed_sizes,
        BenchLz4CompressOpts opts, nvcompStatus_t* statuses, cudaStream_t s) {
    if constexpr (std::is_invocable_v<F,const void* const*,const size_t*,size_t,size_t,
        void*,size_t,void* const*,size_t*,decltype(opts),nvcompStatus_t*,cudaStream_t>)
        return f(input,sizes,chunk,count,temp,temp_bytes,output,compressed_sizes,opts,statuses,s);
    else if constexpr (std::is_invocable_v<F,const void* const*,const size_t*,size_t,size_t,
        void*,size_t,void* const*,size_t*,decltype(opts),cudaStream_t>)
        return f(input,sizes,chunk,count,temp,temp_bytes,output,compressed_sizes,opts,s);
    else {
        static_assert(std::is_invocable_v<F,const void* const*,const size_t*,size_t,size_t,
            void*,size_t,void* const*,size_t*,decltype(opts),cudaStream_t>,
            "Unrecognized nvCOMP LZ4 CompressAsync signature");
    }
}

__global__ static void pack_chunks(const uint8_t* slots, size_t stride,
        const size_t* lengths, const size_t* offsets, uint8_t* packed, size_t count) {
    for(size_t c=blockIdx.x;c<count;c+=gridDim.x) {
        const uint8_t* src=slots+c*stride;
        uint8_t* dst=packed+offsets[c];
        size_t n=lengths[c], words=n/16;
        for(size_t k=threadIdx.x;k<words;k+=blockDim.x)
            reinterpret_cast<uint4*>(dst)[k]=reinterpret_cast<const uint4*>(src)[k];
        for(size_t k=words*16+threadIdx.x;k<n;k+=blockDim.x) dst[k]=src[k];
        // Padding is transmitted by the staged baseline. Define it explicitly.
        for(size_t k=n+threadIdx.x;k<((n+15)&~size_t(15));k+=blockDim.x) dst[k]=0;
    }
}

static void require_de_ptr(void* p, const char* label) {
    // Zero the whole slot: accepts the boolean attribute's byte-sized result.
    unsigned int capable=0;
    drv(cuPointerGetAttribute(&capable,CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE,
        reinterpret_cast<CUdeviceptr>(p)),label);
    if (!capable) throw std::runtime_error(std::string(label)+" is not HW_DECOMPRESS_CAPABLE");
}
struct Options {
    int src=0,dst=1,iters=5,warmup=2;
    size_t bytes=0,chunk=65536;
    bool e2e=true;
};
static size_t positive(const std::string& s) {
    if(s.empty() || s[0]=='-') throw std::runtime_error("expected positive integer: "+s);
    size_t pos=0; unsigned long long n=std::stoull(s,&pos);
    if(pos!=s.size() || !n || n>std::numeric_limits<size_t>::max())
        throw std::runtime_error("invalid positive integer: "+s);
    return size_t(n);
}
static Options parse(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        std::string a=argv[i];
        if(a=="--no-e2e") {o.e2e=false;continue;}
        if(a=="--help") {
            std::cout<<"bench_peer_de --stdin-bytes N --src 0 --dst 1 --chunk 65536 "
                        "--iters 5 --warmup 2 [--no-e2e]\n"; std::exit(0);
        }
        if(i+1>=argc) throw std::runtime_error("missing value: "+a);
        std::string v=argv[++i];
        if(a=="--stdin-bytes") o.bytes=positive(v);
        else if(a=="--chunk") o.chunk=positive(v);
        else if(a=="--src" || a=="--dst" || a=="--warmup" || a=="--iters") {
            size_t pos=0; long val=std::stol(v,&pos);
            if(pos!=v.size() || val<0 || val>100000) throw std::runtime_error("bad value: "+a);
            if(a=="--src")o.src=int(val);else if(a=="--dst")o.dst=int(val);
            else if(a=="--iters")o.iters=int(val);else o.warmup=int(val);
        } else throw std::runtime_error("unknown option: "+a);
    }
    if(!o.bytes || o.src==o.dst || o.iters<1 || o.chunk%16 || o.chunk>4*1024*1024)
        throw std::runtime_error("need nonempty input, distinct GPUs, iters>=1, "
                                 "and 16-byte-aligned chunk <=4 MiB");
    return o;
}
struct Times { double median=0,min=0,max=0; };
static Times measure(const Options& o, const std::function<void()>& fn) {
    for(int k=0;k<o.warmup;++k) fn();
    std::vector<double> samples;
    for(int k=0;k<o.iters;++k) {
        auto a=std::chrono::steady_clock::now();fn();
        auto b=std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double>(b-a).count());
    }
    std::sort(samples.begin(),samples.end());
    double med=samples[samples.size()/2];
    if(samples.size()%2==0) med=(med+samples[samples.size()/2-1])/2;
    return {med,samples.front(),samples.back()};
}
static std::string quote(const std::string& s) {
    std::string out="\"";
    for(unsigned char c:s) {
        if(c=='"' || c=='\\') {out+='\\';out+=char(c);}
        else if(c=='\n') out+="\\n";
        else if(c=='\r') out+="\\r";
        else if(c=='\t') out+="\\t";
        else if(c<32) out+='?';
        else out+=char(c);
    }
    return out+"\"";
}
static void verify_bytes(int device, Buffer& output, const std::vector<uint8_t>& ref,
        Buffer* actual, const std::vector<size_t>& raw_sizes, const char* label) {
    RT(cudaSetDevice(device));
    std::vector<uint8_t> got(ref.size());
    RT(cudaMemcpy(got.data(),output.p,got.size(),cudaMemcpyDeviceToHost));
    auto mismatch=std::mismatch(got.begin(),got.end(),ref.begin());
    if(mismatch.first!=got.end())
        throw std::runtime_error(std::string(label)+" byte mismatch at "+
                                 std::to_string(mismatch.first-got.begin()));
    if(actual) {
        std::vector<uint32_t> sizes(raw_sizes.size());
        RT(cudaMemcpy(sizes.data(),actual->p,sizes.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
        for(size_t i=0;i<sizes.size();++i)
            if(sizes[i]!=raw_sizes[i])
                throw std::runtime_error(std::string(label)+" output length mismatch, chunk "+std::to_string(i));
    }
}
static void poison(int device,Buffer& out,const std::vector<uint8_t>& inv,Buffer* actual=nullptr) {
    RT(cudaSetDevice(device));
    RT(cudaMemcpy(out.p,inv.data(),inv.size(),cudaMemcpyHostToDevice));
    if(actual) RT(cudaMemset(actual->p,0xff,actual->bytes));
}
static void submit_de(int device,cudaStream_t stream,std::vector<CUmemDecompressParams>& req,
                      const char* label) {
    RT(cudaSetDevice(device));
    size_t error=SIZE_MAX;
    CUresult result=cuMemBatchDecompressAsync(req.data(),req.size(),0,&error,stream);
    if(result!=CUDA_SUCCESS) {
        std::string detail=std::string(label)+" (chunk "+
            (error==SIZE_MAX?std::string("batch"):std::to_string(error))+")";
        drv(result,detail.c_str());
    }
    // Submission success alone does not certify completion or correct bytes.
    rt(cudaStreamSynchronize(stream),label);
}

int main(int argc,char** argv) {
    std::string stage="arguments";
    try {
        Options o=parse(argc,argv);
        stage="device checks"; DR(cuInit(0));
        int devices=0;RT(cudaGetDeviceCount(&devices));
        if(o.src>=devices || o.dst>=devices) throw std::runtime_error("GPU index out of range");
        cudaDeviceProp sp{},dp{};
        RT(cudaGetDeviceProperties(&sp,o.src));RT(cudaGetDeviceProperties(&dp,o.dst));
        int driver=0,runtime=0;RT(cudaDriverGetVersion(&driver));RT(cudaRuntimeGetVersion(&runtime));
        const std::string nvcomp_headers=std::to_string(NVCOMP_VER_MAJOR)+"."+
            std::to_string(NVCOMP_VER_MINOR)+"."+std::to_string(NVCOMP_VER_PATCH);
        std::cerr<<"nvCOMP headers="<<nvcomp_headers<<" CUDA headers="<<CUDA_VERSION<<" runtime="<<runtime<<" driver="<<driver
                 <<"\nGPU"<<o.src<<"="<<sp.name<<" GPU"<<o.dst<<"="<<dp.name<<"\n";
        int access=0;RT(cudaDeviceCanAccessPeer(&access,o.dst,o.src));
        if(!access) throw std::runtime_error("destination GPU cannot read source GPU");
        RT(cudaSetDevice(o.src));RT(cudaFree(nullptr));
        RT(cudaSetDevice(o.dst));RT(cudaFree(nullptr));
        auto pe=cudaDeviceEnablePeerAccess(o.src,0);
        if(pe==cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError(); else RT(pe);
        size_t de_limit=std::numeric_limits<size_t>::max();
        for(int dev:{o.src,o.dst}) {
            int mask=0,max_length=0;
            DR(cuDeviceGetAttribute(&mask,CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK,dev));
            DR(cuDeviceGetAttribute(&max_length,CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH,dev));
            if(!(mask & CU_MEM_DECOMPRESS_ALGORITHM_LZ4))
                throw std::runtime_error("LZ4 DE is unavailable on GPU"+std::to_string(dev));
            if(max_length<=0 || o.chunk>size_t(max_length))
                throw std::runtime_error("chunk exceeds DE limit on GPU"+std::to_string(dev));
            de_limit=std::min(de_limit,size_t(max_length));
            std::cerr<<"GPU"<<dev<<" DE LZ4=yes max_length="<<max_length<<"\n";
        }
        std::cerr<<"Peer access is enabled; inspect nvidia-smi topo -m for actual NVLink topology.\n";
        stage="read input";
        std::vector<uint8_t> host(o.bytes), inv(o.bytes);
        if(o.bytes>size_t(std::numeric_limits<std::streamsize>::max()))
            throw std::runtime_error("stdin size too large");
        std::cin.read(reinterpret_cast<char*>(host.data()),std::streamsize(host.size()));
        if(size_t(std::cin.gcount())!=host.size()) throw std::runtime_error("short stdin payload");
        for(size_t i=0;i<o.bytes;++i) inv[i]=uint8_t(~host[i]);
        const size_t count=(o.bytes-1)/o.chunk+1;
        if(count>std::numeric_limits<unsigned>::max()) throw std::runtime_error("too many chunks");
        std::vector<size_t> raw_sizes(count),lengths(count),offsets(count);
        for(size_t i=0;i<count;++i)raw_sizes[i]=std::min(o.chunk,o.bytes-i*o.chunk);
        auto opts=BENCH_LZ4_DEFAULT_OPTS; // byte data; no bitshuffle
        size_t maximum=0,scratch_bytes=0;
        NC(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(o.chunk,opts,&maximum));
        const size_t stride=align16(maximum),pool_bytes=checked_mul(count,stride);
        Stream s0(o.src),s1(o.dst);
        stage="allocate";
        Buffer raw(o.src,o.bytes),slots(o.src,pool_bytes),packed(o.src,pool_bytes);
        Buffer input_ptrs(o.src,count*sizeof(void*)),output_ptrs(o.src,count*sizeof(void*));
        Buffer dsizes(o.src,count*sizeof(size_t)),dlengths(o.src,count*sizeof(size_t));
        Buffer doffsets(o.src,count*sizeof(size_t)),cstatus(o.src,count*sizeof(nvcompStatus_t));
        Buffer out(o.dst,o.bytes),actual(o.dst,count*sizeof(uint32_t));
        std::vector<const void*> ip(count);std::vector<void*> op(count);
        for(size_t i=0;i<count;++i){ip[i]=raw.at(i*o.chunk);op[i]=slots.at(i*stride);}
        RT(cudaSetDevice(o.src));
        RT(cudaMemcpy(raw.p,host.data(),o.bytes,cudaMemcpyHostToDevice));
        RT(cudaMemcpy(input_ptrs.p,ip.data(),count*sizeof(void*),cudaMemcpyHostToDevice));
        RT(cudaMemcpy(output_ptrs.p,op.data(),count*sizeof(void*),cudaMemcpyHostToDevice));
        RT(cudaMemcpy(dsizes.p,raw_sizes.data(),count*sizeof(size_t),cudaMemcpyHostToDevice));
        NC(temp_size(&BENCH_LZ4_TEMP_SIZE,count,o.chunk,opts,
                     &scratch_bytes,o.bytes,s0.s));
        Buffer scratch(o.src,scratch_bytes);
        std::vector<nvcompStatus_t> status(count);
        size_t payload_bytes=0,wire_bytes=0;
        auto prepare=[&] {
            RT(cudaSetDevice(o.src));
            RT(cudaMemsetAsync(cstatus.p,0,cstatus.bytes,s0.s));
            NC(compress_call(&nvcompBatchedLZ4CompressAsync,input_ptrs.as<const void*>(),
                dsizes.as<size_t>(),o.chunk,count,scratch.p,scratch_bytes,
                output_ptrs.as<void*>(),dlengths.as<size_t>(),opts,cstatus.as<nvcompStatus_t>(),s0.s));
            RT(cudaMemcpyAsync(lengths.data(),dlengths.p,count*sizeof(size_t),cudaMemcpyDeviceToHost,s0.s));
            RT(cudaMemcpyAsync(status.data(),cstatus.p,count*sizeof(nvcompStatus_t),cudaMemcpyDeviceToHost,s0.s));
            s0.wait();
            payload_bytes=wire_bytes=0;
            for(size_t i=0;i<count;++i) {
                if(status[i]!=nvcompSuccess || lengths[i]==0 || lengths[i]>maximum)
                    throw std::runtime_error("invalid compressor output at chunk "+std::to_string(i));
                if(lengths[i]>de_limit)
                    throw std::runtime_error("compressed chunk exceeds DE limit; use a smaller --chunk");
                offsets[i]=wire_bytes;payload_bytes+=lengths[i];wire_bytes+=align16(lengths[i]);
            }
            if(wire_bytes>pool_bytes)throw std::runtime_error("packed pool overflow");
            RT(cudaMemcpyAsync(doffsets.p,offsets.data(),count*sizeof(size_t),cudaMemcpyHostToDevice,s0.s));
            pack_chunks<<<unsigned(std::min(count,size_t(65535))),256,0,s0.s>>>(
                slots.as<uint8_t>(),stride,dlengths.as<size_t>(),doffsets.as<size_t>(),
                packed.as<uint8_t>(),count);
            RT(cudaGetLastError());s0.wait();
        };
        std::vector<CUmemDecompressParams> req(count);
        auto configure=[&](uint8_t* input,Buffer& output,Buffer& act) {
            for(size_t i=0;i<count;++i) {
                auto& p=req[i];p={};
                p.algo=CU_MEM_DECOMPRESS_ALGORITHM_LZ4;
                p.src=input+offsets[i];p.srcNumBytes=lengths[i];
                p.dst=output.at(i*o.chunk);p.dstNumBytes=raw_sizes[i];
                p.dstActBytes=act.as<uint32_t>()+i;
            }
        };
        stage="compress and local DE control";prepare();
        // Validate raw LZ4 format locally on GPU0, without a GPU1 compressed buffer.
        {
            Buffer local(o.src,o.bytes),local_actual(o.src,count*sizeof(uint32_t));
            RT(cudaSetDevice(o.src));
            require_de_ptr(packed.p,"GPU0 compressed pool");
            require_de_ptr(local.p,"GPU0 local output");
            require_de_ptr(local_actual.p,"GPU0 actual lengths");
            configure(packed.as<uint8_t>(),local,local_actual);
            poison(o.src,local,inv,&local_actual);
            submit_de(o.src,s0.s,req,"local DE control");
            verify_bytes(o.src,local,host,&local_actual,raw_sizes,"local DE control");
        }
        std::cerr<<"Local DE control: PASS\n";
        stage="peer DE input capability";
        RT(cudaSetDevice(o.dst));
        require_de_ptr(packed.p,"GPU0 compressed pool viewed from destination");
        require_de_ptr(out.p,"destination output");
        require_de_ptr(actual.p,"destination actual lengths");
        // C path: no destination compressed payload allocation, no peer payload memcpy.
        auto peer_decode=[&] {
            configure(packed.as<uint8_t>(),out,actual);
            submit_de(o.dst,s1.s,req,"peer-input DE");
        };
        stage="peer DE correctness";
        poison(o.dst,out,inv,&actual);peer_decode();
        verify_bytes(o.dst,out,host,&actual,raw_sizes,"peer-input DE");
        std::cerr<<"Peer-input DE: PASS (no application payload staging / no payload memcpy)\n";
        stage="peer DE timing";
        Times peer=measure(o,peer_decode);
        stage="raw copy correctness";
        auto raw_copy=[&] {
            RT(cudaSetDevice(o.dst));
            RT(cudaMemcpyPeerAsync(out.p,o.dst,raw.p,o.src,o.bytes,s1.s));s1.wait();
        };
        poison(o.dst,out,inv);raw_copy();verify_bytes(o.dst,out,host,nullptr,raw_sizes,"raw copy");
        Times raw_t=measure(o,raw_copy);
        stage="compression preparation timing";
        Times prep=measure(o,prepare);
        Times peer_e2e{};
        if(o.e2e) {
            auto online_peer=[&]{prepare();peer_decode();};
            stage="online peer DE correctness";
            poison(o.dst,out,inv,&actual);online_peer();
            verify_bytes(o.dst,out,host,&actual,raw_sizes,"online peer DE");
            stage="online peer DE timing";peer_e2e=measure(o,online_peer);
        }
        // Only now allocate destination compressed staging for the B baseline.
        stage="staged control allocation";Buffer staging(o.dst,pool_bytes);
        require_de_ptr(staging.p,"destination staged input");
        auto compressed_copy=[&] {
            RT(cudaSetDevice(o.dst));
            RT(cudaMemcpyPeerAsync(staging.p,o.dst,packed.p,o.src,wire_bytes,s1.s));s1.wait();
        };
        auto local_decode=[&] {
            configure(staging.as<uint8_t>(),out,actual);
            submit_de(o.dst,s1.s,req,"destination local DE");
        };
        auto staged=[&]{compressed_copy();local_decode();};
        stage="staged correctness";
        poison(o.dst,out,inv,&actual);staged();
        verify_bytes(o.dst,out,host,&actual,raw_sizes,"staged DE");
        stage="staged timing";Times staged_t=measure(o,staged);
        Times xfer=measure(o,compressed_copy),local=measure(o,local_decode),staged_e2e{};
        if(o.e2e) {
            auto online_staged=[&]{prepare();staged();};
            stage="online staged correctness";
            poison(o.dst,out,inv,&actual);online_staged();
            verify_bytes(o.dst,out,host,&actual,raw_sizes,"online staged DE");
            stage="online staged timing";staged_e2e=measure(o,online_staged);
        }
        std::cerr<<"All byte/length checks: PASS. Times include host submission/synchronization;"
                    " online times also include sizes readback, offsets and GPU packing.\n";
        std::ostringstream json;json<<std::setprecision(12);
        json<<"{\"status\":\"ok\",\"algorithm\":\"LZ4\",\"src\":"<<o.src<<",\"dst\":"<<o.dst
            <<",\"nvcomp_headers\":"<<quote(nvcomp_headers)
            <<",\"src_gpu\":"<<quote(sp.name)<<",\"dst_gpu\":"<<quote(dp.name)
            <<",\"cuda_headers\":"<<CUDA_VERSION<<",\"cuda_runtime\":"<<runtime
            <<",\"cuda_driver\":"<<driver<<",\"raw_bytes\":"<<o.bytes
            <<",\"payload_bytes\":"<<payload_bytes<<",\"staged_wire_bytes\":"<<wire_bytes
            <<",\"chunk_bytes\":"<<o.chunk<<",\"chunks\":"<<count
            <<",\"iters\":"<<o.iters<<",\"warmup\":"<<o.warmup
            <<",\"ratio\":"<<double(o.bytes)/payload_bytes
            <<",\"peer_payload_staging_bytes\":0,\"peer_payload_memcpy_calls\":0"
              ",\"topology\":\"peer_access; NVLink route must be confirmed externally\"";
        auto emit=[&](const char* name,Times t) {
            json<<",\""<<name<<"_s\":"<<t.median
                <<",\""<<name<<"_min_s\":"<<t.min<<",\""<<name<<"_max_s\":"<<t.max;
        };
        emit("raw_copy",raw_t);emit("prepare",prep);emit("peer_de",peer);
        emit("staged_de",staged_t);emit("compressed_copy",xfer);emit("local_de",local);
        if(o.e2e){emit("peer_e2e",peer_e2e);emit("staged_e2e",staged_e2e);}
        json<<",\"peer_de_output_GBps\":"<<o.bytes/peer.median/1e9
            <<",\"raw_copy_GBps\":"<<o.bytes/raw_t.median/1e9
            <<",\"peer_receive_speedup\":"<<raw_t.median/peer.median
            <<",\"verified\":true}";
        std::cout<<json.str()<<"\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"FAIL ["<<stage<<"]: "<<e.what()<<"\nNo software or staged fallback was performed.\n";
        std::cout<<"{\"status\":\"error\",\"stage\":"<<quote(stage)
                 <<",\"message\":"<<quote(e.what())<<"}\n";
        return 2;
    }
}

