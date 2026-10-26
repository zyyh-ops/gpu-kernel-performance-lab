#include "double_buffer_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t status = (call); \
    if (status != cudaSuccess) throw std::runtime_error( \
        std::string(#call) + ": " + cudaGetErrorString(status)); \
} while(0)

using Gemm = void(*)(const float*,const float*,float*,int);
constexpr int TILE = kGemmTile;

struct Variant {
    const char* name;
    Gemm kernel;
    std::size_t base_bytes;
};
const Variant SINGLE_FAST{"single-fast16", matmul_single_buffer_kernel<TILE,true>, single_base_smem_bytes};
const Variant DOUBLE_FAST{"double-fast16", matmul_double_buffer_kernel<TILE,true>, double_base_smem_bytes};
const Variant SINGLE_GENERAL{"single-masked", matmul_single_buffer_kernel<TILE,false>, single_base_smem_bytes};
const Variant DOUBLE_GENERAL{"double-async4-masked", matmul_double_buffer_kernel<TILE,false>, double_base_smem_bytes};

// cudaMalloc allocations are sufficiently aligned; row-start 16B alignment
// follows from contiguous FP32 matrix N % TILE == 0, and all vector starts
// lie on 4-float boundaries in those rows.
const Variant& select_variant(bool use_double, int n) {
    const bool fast = (n % TILE == 0);
    return use_double ? (fast ? DOUBLE_FAST : DOUBLE_GENERAL)
                      : (fast ? SINGLE_FAST : SINGLE_GENERAL);
}

struct Options {
    std::string mode = "test";
    std::string json_path;
    int n = 4096;
    int warmup = 5;
    int repeat = 20;
    int samples = 5;
    std::vector<int> extra_kib{0,16,24,32,44};
    std::vector<int> test_sizes{1,7,15,16,17,31,32,33,47,48,65,129,256};
};

void validate_n(int n) {
    if (n <= 0) throw std::invalid_argument("N must be positive");
}

std::vector<int> parse_csv(const std::string& in) {
    std::vector<int> out;
    size_t start=0;
    while (start < in.size()) {
        const size_t end=in.find(',',start);
        const std::string x=in.substr(start, end==std::string::npos ? end : end-start);
        if (x.empty()) throw std::invalid_argument("empty extra-kib entry");
        size_t consumed=0;
        const int v=std::stoi(x,&consumed);
        if (consumed!=x.size() || v<0) throw std::invalid_argument("invalid extra-kib entry: "+x);
        out.push_back(v);
        if (end==std::string::npos) break;
        start=end+1;
    }
    if (out.empty()) throw std::invalid_argument("extra-kib must be nonempty");
    return out;
}

Options parse(int argc, char** argv) {
    Options opt;
    for (int i=1; i<argc; ++i) {
        auto arg=std::string(argv[i]);
        auto val=[&]() -> std::string {
            if(i+1>=argc) throw std::invalid_argument("missing value for "+arg);
            return argv[++i];
        };
        if(arg=="--mode") opt.mode=val();
        else if(arg=="--n") opt.n=std::stoi(val());
        else if(arg=="--warmup") opt.warmup=std::stoi(val());
        else if(arg=="--repeat") opt.repeat=std::stoi(val());
        else if(arg=="--samples") opt.samples=std::stoi(val());
        else if(arg=="--extra-kib") opt.extra_kib=parse_csv(val());
        else if(arg=="--test-sizes") opt.test_sizes=parse_csv(val());
        else if(arg=="--json") opt.json_path=val();
        else if(arg=="--help") {
            std::cout << "--mode test|bench|both --n 4096 --test-sizes 1,17,33,65 "
                         "--extra-kib 0,16,24,32,44 "
                         "--warmup 5 --repeat 20 --samples 5 --json results/summary/double_buffer.json\n";
            std::exit(0);
        }
        else throw std::invalid_argument("unknown option: "+arg);
    }
    if (opt.mode!="test" && opt.mode!="bench" && opt.mode!="both")
        throw std::invalid_argument("--mode must be test, bench, or both");
    validate_n(opt.n);
    for (int sz:opt.test_sizes) validate_n(sz);
    if(opt.warmup<0 || opt.repeat<=0 || opt.samples<=0) throw std::invalid_argument("invalid timing repetitions");
    return opt;
}

struct DeviceMemory {
    float *a=nullptr, *b=nullptr, *c=nullptr;
    explicit DeviceMemory(int n) {
        const size_t bytes=size_t(n)*size_t(n)*sizeof(float);
        try {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&a),bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&b),bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c),bytes));
        } catch (...) { cudaFree(a); cudaFree(b); cudaFree(c); throw; }
    }
    ~DeviceMemory() {cudaFree(a); cudaFree(b); cudaFree(c);}
    DeviceMemory(const DeviceMemory&)=delete;
    DeviceMemory& operator=(const DeviceMemory&)=delete;
};

void load_inputs(DeviceMemory& d, int n) {
    const size_t cells=size_t(n)*size_t(n);
    std::vector<float> A(cells),B(cells);
    for (size_t i=0;i<cells;++i) {
        A[i]=float(int(i%17)-8)/8.0f;
        B[i]=float(int(i%13)-6)/6.0f;
    }
    CUDA_CHECK(cudaMemcpy(d.a,A.data(),cells*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.b,B.data(),cells*sizeof(float),cudaMemcpyHostToDevice));
}

std::size_t required_bytes(const Variant& v, int extra_kib) {
    return v.base_bytes + size_t(extra_kib)*1024;
}

void launch(const Variant& v, const DeviceMemory& d, int n, int extra_kib) {
    const dim3 block(TILE,TILE);
    const dim3 grid((n-1)/TILE+1,(n-1)/TILE+1);
    v.kernel<<<grid,block,required_bytes(v,extra_kib)>>>(d.a,d.b,d.c,n);
    CUDA_CHECK(cudaGetLastError());
}

void test_one(int n, const Variant& v) {
    validate_n(n);
    DeviceMemory d(n);
    const size_t cells=size_t(n)*size_t(n);
    std::vector<float> A(cells), B(cells), got(cells);
    for(size_t i=0;i<cells;++i) {
        A[i]=float(int(i%17)-8)/8.0f;
        B[i]=float(int(i%13)-6)/6.0f;
    }
    CUDA_CHECK(cudaMemcpy(d.a,A.data(),cells*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.b,B.data(),cells*sizeof(float),cudaMemcpyHostToDevice));
    launch(v,d,n,0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(got.data(),d.c,cells*sizeof(float),cudaMemcpyDeviceToHost));
    double max_abs=0.;
    for(int i=0;i<n;++i) {
        for(int j=0;j<n;++j) {
            double expected=0.;
            for(int k=0;k<n;++k) expected += double(A[size_t(i)*n+k])*double(B[size_t(k)*n+j]);
            const double error=std::abs(expected-double(got[size_t(i)*n+j]));
            max_abs=std::max(max_abs,error);
            if (!std::isfinite(got[size_t(i)*n+j]) || error > 5e-3 + 1e-3 * std::abs(expected))
                throw std::runtime_error(std::string(v.name)+" correctness FAIL N="+std::to_string(n)+
                    " at ("+std::to_string(i)+","+std::to_string(j)+") error="+std::to_string(error));
        }
    }
    std::cout<<"[PASS] "<<v.name<<" N="<<n<<" max_abs="<<max_abs<<"\n";
}

struct Events {
    cudaEvent_t start=nullptr,stop=nullptr;
    Events() {
        CUDA_CHECK(cudaEventCreate(&start));
        try {CUDA_CHECK(cudaEventCreate(&stop));}
        catch(...) {cudaEventDestroy(start);throw;}
    }
    ~Events(){if(start)cudaEventDestroy(start);if(stop)cudaEventDestroy(stop);}
};

// Each timed observation is kernel-only elapsed time per launch; compilations and allocation excluded.
double time_sample(const Variant& v, const DeviceMemory& d, int n, int extra_kib, int warmup, int repeat) {
    for(int i=0;i<warmup;++i) launch(v,d,n,extra_kib);
    CUDA_CHECK(cudaDeviceSynchronize());
    Events e;
    CUDA_CHECK(cudaEventRecord(e.start));
    for(int i=0;i<repeat;++i) launch(v,d,n,extra_kib);
    CUDA_CHECK(cudaEventRecord(e.stop));
    CUDA_CHECK(cudaEventSynchronize(e.stop));
    float elapsed=0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed,e.start,e.stop));
    return double(elapsed)/repeat;
}

double median(std::vector<double> values) {
    std::sort(values.begin(),values.end());
    const size_t m=values.size()/2;
    return values.size()%2 ? values[m] : 0.5*(values[m-1]+values[m]);
}

double occupancy_pct(const Variant& v, int extra_kib, const cudaDeviceProp& prop) {
    int active_blocks=0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,v.kernel,
        TILE*TILE,required_bytes(v,extra_kib)));
    return 100.0*active_blocks*TILE*TILE / prop.maxThreadsPerMultiProcessor;
}

struct Record {
    int extra_kib;
    std::size_t single_shared_bytes, double_shared_bytes;
    double single_ms,double_ms,single_gflops,double_gflops,single_occ,double_occ,speedup;
};

double gflops(int n, double ms) { return 2.0*double(n)*n*n/(ms*1e-3)/1e9; }

void write_json(const Options& opt, const cudaDeviceProp& prop, const std::vector<Record>& records) {
    if(opt.json_path.empty()) return;
    std::ofstream out(opt.json_path);
    if (!out) throw std::runtime_error("cannot write JSON: "+opt.json_path+
        " (create parent folder first)");
    int runtime_version=0,driver_version=0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    out<<std::setprecision(10);
    out<<"{\n  \"status\": \"gpu_measured_not_automatically_validated\",\n"
       <<"  \"variant\": \"single_vs_cp_async_double_buffer\",\n"
       <<"  \"N\": "<<opt.n<<", \"tile\": "<<TILE
       <<", \"dtype\": \"float32\",\n"
       <<"  \"copy_path\": \""<<(opt.n%TILE==0?"cp.async16-fast":"cp.async4-masked")<<"\",\n"
       <<"  \"gpu\": \""<<prop.name<<"\", \"sm\": \""<<prop.major<<prop.minor<<"\",\n"
       <<"  \"cuda_runtime_version\": "<<runtime_version
       <<", \"cuda_driver_version\": "<<driver_version<<",\n"
       <<"  \"nvcc_version\": \""<<__CUDACC_VER_MAJOR__<<"."<<__CUDACC_VER_MINOR__<<"\",\n"
       <<"  \"warmup\": "<<opt.warmup<<", \"repeat\": "<<opt.repeat
       <<", \"samples\": "<<opt.samples<<",\n"
       <<"  \"units\": {\"latency\": \"ms/launch\", \"throughput\": \"GFLOP/s\", \"occupancy\": \"theoretical_pct\"},\n"
       <<"  \"results\": [\n";
    for(size_t i=0;i<records.size();++i) {
        const auto&r=records[i];
        out<<"    {\"extra_shared_kib\": "<<r.extra_kib
           <<", \"single_total_shared_bytes\": "<<r.single_shared_bytes
           <<", \"double_total_shared_bytes\": "<<r.double_shared_bytes
           <<", \"single_ms\": "<<r.single_ms<<", \"double_ms\": "<<r.double_ms
           <<", \"single_gflops\": "<<r.single_gflops<<", \"double_gflops\": "<<r.double_gflops
           <<", \"single_theoretical_occupancy_pct\": "<<r.single_occ
           <<", \"double_theoretical_occupancy_pct\": "<<r.double_occ
           <<", \"speedup_double_over_single\": "<<r.speedup<<"}"
           <<(i+1==records.size()?"\n":",\n");
    }
    out<<"  ]\n}\n";
    if(!out)throw std::runtime_error("writing benchmark JSON failed");
    std::cout<<"[saved] "<<opt.json_path<<"\n";
}

void run_benchmark(const Options& opt, const cudaDeviceProp& prop) {
    validate_n(opt.n);
    DeviceMemory d(opt.n);
    load_inputs(d,opt.n);
    // Benchmark mode does not replace correctness; run --mode test beforehand.
    std::vector<Record> records;
    const Variant& single=select_variant(false,opt.n);
    const Variant& dbl=select_variant(true,opt.n);
    std::cout<<"GPU: "<<prop.name<<" N="<<opt.n<<" tile="<<TILE
             <<" path="<<(opt.n%TILE==0?"cp.async16-fast":"cp.async4-masked")<<"\n";
    std::cout<<"extra_KiB | single_total_B | double_total_B | single_GF | single_occ% | double_GF | double_occ% | speedup\n";
    for(int extra:opt.extra_kib) {
        const size_t sb=required_bytes(single,extra),db=required_bytes(dbl,extra);
        if(sb>prop.sharedMemPerBlock || db>prop.sharedMemPerBlock)
            throw std::invalid_argument("Dynamic shared memory beyond default per-block device limit; extra_KiB="+std::to_string(extra));
        std::vector<double> ts,td;
        // Paired, alternating A/B order to reduce run-order bias.
        for(int i=0;i<opt.samples;++i) {
            if(i%2==0) {
                ts.push_back(time_sample(single,d,opt.n,extra,opt.warmup,opt.repeat));
                td.push_back(time_sample(dbl,d,opt.n,extra,opt.warmup,opt.repeat));
            } else {
                td.push_back(time_sample(dbl,d,opt.n,extra,opt.warmup,opt.repeat));
                ts.push_back(time_sample(single,d,opt.n,extra,opt.warmup,opt.repeat));
            }
        }
        const double sm=median(ts),dm=median(td);
        Record r{extra,sb,db,sm,dm,gflops(opt.n,sm),gflops(opt.n,dm),
            occupancy_pct(single,extra,prop),occupancy_pct(dbl,extra,prop),sm/dm};
        records.push_back(r);
        std::cout<<extra<<" | "<<sb<<" | "<<db<<" | "<<r.single_gflops<<" | "<<r.single_occ
                 <<" | "<<r.double_gflops<<" | "<<r.double_occ<<" | "<<r.speedup<<"x\n";
    }
    write_json(opt,prop,records);
}

int main(int argc,char** argv) {
    try {
        const Options opt=parse(argc,argv);
        int device=0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
        if(prop.major<8) throw std::runtime_error("cp.async requires sm_80 or later");
        if(opt.mode=="test"||opt.mode=="both") {
            for(int n:opt.test_sizes) {
                test_one(n,select_variant(false,n));
                test_one(n,select_variant(true,n));
            }
            std::cout<<"[PASS] CUDA correctness for all requested shapes (including tails)\n";
        }
        if(opt.mode=="bench"||opt.mode=="both") run_benchmark(opt,prop);
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"[FAIL] "<<e.what()<<"\n";
        return 1;
    }
}
