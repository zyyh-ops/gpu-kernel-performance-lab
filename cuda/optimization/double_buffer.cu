#include "double_buffer_kernels.cuh"
#include "gemm_validation.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t status = (call); \
    if (status != cudaSuccess) throw std::runtime_error( \
        std::string(#call) + ": " + cudaGetErrorString(status)); \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error( \
        std::string(#call) + ": cuBLAS status " + std::to_string(int(status))); \
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
    std::string reference = "auto";
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
        else if(arg=="--reference") opt.reference=val();
        else if(arg=="--help") {
            std::cout << "--mode test|bench|both --n 4096 --test-sizes 1,17,33,65 "
                         "--extra-kib 0,16,24,32,44 --reference auto|cpu|cublas "
                         "--warmup 5 --repeat 20 --samples 5 --json results/summary/double_buffer.json\n";
            std::exit(0);
        }
        else throw std::invalid_argument("unknown option: "+arg);
    }
    if (opt.mode!="test" && opt.mode!="bench" && opt.mode!="both")
        throw std::invalid_argument("--mode must be test, bench, or both");
    if (opt.reference!="auto" && opt.reference!="cpu" && opt.reference!="cublas")
        throw std::invalid_argument("--reference must be auto, cpu, or cublas");
    // Historical evidence must never be silently overwritten.
    if (!opt.json_path.empty() && std::filesystem::exists(opt.json_path))
        throw std::invalid_argument("JSON already exists; choose a new --json path: "+opt.json_path);
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

struct Reference {
    std::string name;
    int cublas_version = 0;
    std::vector<double> values;
};

struct BlasHandle {
    cublasHandle_t handle = nullptr;
    BlasHandle() { CUBLAS_CHECK(cublasCreate(&handle)); }
    ~BlasHandle() { if (handle) cublasDestroy(handle); }
    BlasHandle(const BlasHandle&)=delete;
    BlasHandle& operator=(const BlasHandle&)=delete;
};

Reference make_reference(const DeviceMemory& d, int n, const std::string& requested) {
    const size_t cells=size_t(n)*size_t(n);
    Reference result;
    const bool cpu = requested=="cpu" || (requested=="auto" && n<=256);
    if (cpu) {
        std::vector<float> a(cells), b(cells);
        CUDA_CHECK(cudaMemcpy(a.data(),d.a,cells*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(),d.b,cells*sizeof(float),cudaMemcpyDeviceToHost));
        result.name="cpu_fp64";
        result.values=cpu_gemm_reference(a,b,n);
        return result;
    }
    BlasHandle blas;
    CUBLAS_CHECK(cublasGetVersion(blas.handle,&result.cublas_version));
    CUBLAS_CHECK(cublasSetMathMode(blas.handle,CUBLAS_PEDANTIC_MATH));
    const float alpha=1.f,beta=0.f;
    // Row-major C=A*B is column-major C^T=B^T*A^T: swap pointers, no transpose.
    CUBLAS_CHECK(cublasGemmEx(blas.handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,
        &alpha,d.b,CUDA_R_32F,n,d.a,CUDA_R_32F,n,&beta,d.c,CUDA_R_32F,n,
        CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(cells);
    CUDA_CHECK(cudaMemcpy(got.data(),d.c,cells*sizeof(float),cudaMemcpyDeviceToHost));
    result.values.assign(got.begin(),got.end());
    result.name="cublas_fp32_pedantic";
    // Cross-check row/column-major interpretation on small GPU test cases.
    if (n<=256) {
        const auto cpu_ref=make_reference(d,n,"cpu");
        if (!compare_gemm(got,cpu_ref.values,5e-3,1e-3).passed())
            throw std::runtime_error("cuBLAS reference disagrees with CPU FP64 reference");
    }
    return result;
}

struct Validation {
    int n,extra_kib;
    std::string variant,reference;
    int cublas_version;
    ErrorMetrics metrics;
};

void validate_shared(const Variant& v, int extra, const cudaDeviceProp& prop) {
    if (required_bytes(v,extra)>prop.sharedMemPerBlock)
        throw std::invalid_argument("Dynamic shared memory beyond default per-block device limit; extra_KiB="+std::to_string(extra));
}

bool test_one(const DeviceMemory& d, int n, const Variant& v, int extra,
              const Reference& ref, const cudaDeviceProp& prop,
              std::vector<Validation>& validations) {
    validate_shared(v,extra,prop);
    const size_t cells=size_t(n)*n;
    // Poison output so missing stores cannot reuse cuBLAS or the other variant's output.
    CUDA_CHECK(cudaMemset(d.c,0xff,cells*sizeof(float)));
    launch(v,d,n,extra);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(cells);
    CUDA_CHECK(cudaMemcpy(got.data(),d.c,cells*sizeof(float),cudaMemcpyDeviceToHost));
    const auto metrics=compare_gemm(got,ref.values,5e-3,1e-3);
    validations.push_back({n,extra,v.name,ref.name,ref.cublas_version,metrics});
    std::cout<<(metrics.passed()?"[PASS] ":"[FAIL] ")<<v.name<<" N="<<n
             <<" extra_KiB="<<extra<<" reference="<<ref.name
             <<" max_abs="<<metrics.max_abs<<" max_scaled_error="<<metrics.max_scaled_error
             <<" mismatches="<<metrics.mismatches<<" nonfinite="<<metrics.nonfinite<<"\n";
    return metrics.passed();
}

void validate_pair(const DeviceMemory& d, int n, int extra, const Reference& ref,
                   const cudaDeviceProp& prop, std::vector<Validation>& validations) {
    const bool single_ok=test_one(d,n,select_variant(false,n),extra,ref,prop,validations);
    const bool double_ok=test_one(d,n,select_variant(true,n),extra,ref,prop,validations);
    if (!single_ok || !double_ok)
        throw std::runtime_error("Correctness failed; benchmark aborted for N="+
                                 std::to_string(n)+" extra_KiB="+std::to_string(extra));
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

std::string json_string(const std::string& value) {
    std::ostringstream out;
    out<<'"';
    for (unsigned char c:value) {
        if (c=='"' || c=='\\') out<<'\\'<<char(c);
        else if (c<0x20) out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
        else out<<char(c);
    }
    out<<'"';
    return out.str();
}

void write_json(const Options& opt, const cudaDeviceProp& prop, const std::vector<Record>& records,
                const std::vector<Validation>& validations, const std::string& error,
                const std::vector<std::string>& command) {
    if(opt.json_path.empty()) return;
    std::ofstream out(opt.json_path);
    if (!out) throw std::runtime_error("cannot write JSON: "+opt.json_path+
        " (create parent folder first)");
    int runtime_version=0,driver_version=0;
    // Best effort even after a CUDA error: preserve failure diagnostics.
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);
    out<<std::setprecision(10);
    const char* status=!error.empty()?"FAILED":
        (records.empty()?"gpu_correctness_passed":"gpu_correctness_and_benchmark_passed");
    out<<"{\n  \"schema_version\": 2,\n  \"status\": "<<json_string(status)<<",\n"
       <<"  \"error\": "<<json_string(error)<<",\n"
       <<"  \"mode\": "<<json_string(opt.mode)<<",\n"
       <<"  \"reference_requested\": "<<json_string(opt.reference)<<",\n"
       <<"  \"variant\": \"single_vs_cp_async_double_buffer\",\n"
       <<"  \"N\": "<<(opt.mode=="test"?"null":std::to_string(opt.n))<<", \"tile\": "<<TILE
       <<", \"dtype\": \"float32\",\n"
       <<"  \"copy_path\": \""<<(opt.mode=="test"?"per_correctness_record":
                                   (opt.n%TILE==0?"cp.async16-fast":"cp.async4-masked"))<<"\",\n"
       <<"  \"gpu\": "<<json_string(prop.name)<<", \"sm\": \""<<prop.major<<prop.minor<<"\",\n"
       <<"  \"cuda_runtime_version\": "<<runtime_version
       <<", \"cuda_driver_version\": "<<driver_version<<",\n"
       <<"  \"nvcc_version\": \""<<__CUDACC_VER_MAJOR__<<"."<<__CUDACC_VER_MINOR__<<"\",\n"
       <<"  \"warmup\": "<<opt.warmup<<", \"repeat\": "<<opt.repeat
       <<", \"samples\": "<<opt.samples<<",\n"
       <<"  \"units\": {\"latency\": \"ms/launch\", \"throughput\": \"GFLOP/s\", \"occupancy\": \"theoretical_pct\"},\n"
       <<"  \"input_pattern\": \"A[i]=(i%17-8)/8; B[i]=(i%13-6)/6\",\n"
       <<"  \"validation_policy\": \"each variant independently vs reference; validation excluded from timing\",\n"
       <<"  \"command\": [";
    for (size_t i=0;i<command.size();++i) out<<(i?", ":"")<<json_string(command[i]);
    out<<"],\n  \"correctness\": [\n";
    for (size_t i=0;i<validations.size();++i) {
        const auto& v=validations[i];
        out<<"    {\"N\": "<<v.n<<", \"extra_shared_kib\": "<<v.extra_kib
           <<", \"variant\": "<<json_string(v.variant)
           <<", \"reference\": "<<json_string(v.reference)
           <<", \"cublas_version\": "<<v.cublas_version
           <<", \"atol\": 0.005, \"rtol\": 0.001"
           <<", \"max_abs_error_finite\": "<<v.metrics.max_abs
           <<", \"max_scaled_error_finite\": "<<v.metrics.max_scaled_error
           <<", \"mismatch_count\": "<<v.metrics.mismatches
           <<", \"nonfinite_count\": "<<v.metrics.nonfinite
           <<", \"elements_checked\": "<<size_t(v.n)*v.n
           <<", \"status\": "<<json_string(v.metrics.passed()?"PASS":"FAIL")<<"}"
           <<(i+1==validations.size()?"\n":",\n");
    }
    out<<"  ],\n  \"results\": [\n";
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

void run_benchmark(const Options& opt, const cudaDeviceProp& prop,
                   std::vector<Record>& records, std::vector<Validation>& validations) {
    validate_n(opt.n);
    DeviceMemory d(opt.n);
    load_inputs(d,opt.n);
    // Reference generation and validation are strictly outside event timing.
    const auto ref=make_reference(d,opt.n,opt.reference);
    const Variant& single=select_variant(false,opt.n);
    const Variant& dbl=select_variant(true,opt.n);
    std::cout<<"GPU: "<<prop.name<<" N="<<opt.n<<" tile="<<TILE
             <<" path="<<(opt.n%TILE==0?"cp.async16-fast":"cp.async4-masked")<<"\n";
    std::cout<<"extra_KiB | single_total_B | double_total_B | single_GF | single_occ% | double_GF | double_occ% | speedup\n";
    for(int extra:opt.extra_kib) {
        const size_t sb=required_bytes(single,extra),db=required_bytes(dbl,extra);
        validate_pair(d,opt.n,extra,ref,prop,validations);
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
        if (!std::isfinite(sm) || !std::isfinite(dm) || sm<=0. || dm<=0.)
            throw std::runtime_error("Invalid benchmark latency; no throughput recorded");
        Record r{extra,sb,db,sm,dm,gflops(opt.n,sm),gflops(opt.n,dm),
            occupancy_pct(single,extra,prop),occupancy_pct(dbl,extra,prop),sm/dm};
        records.push_back(r);
        std::cout<<extra<<" | "<<sb<<" | "<<db<<" | "<<r.single_gflops<<" | "<<r.single_occ
                 <<" | "<<r.double_gflops<<" | "<<r.double_occ<<" | "<<r.speedup<<"x\n";
    }
}

int main(int argc,char** argv) {
    try {
        const Options opt=parse(argc,argv);
        int device=0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
        if(prop.major<8) throw std::runtime_error("cp.async requires sm_80 or later");
        std::vector<Record> records;
        std::vector<Validation> validations;
        std::vector<std::string> command(argv,argv+argc);
        std::string error;
        try {
            if(opt.mode=="test"||opt.mode=="both") {
                for(int n:opt.test_sizes) {
                    DeviceMemory d(n);
                    load_inputs(d,n);
                    const auto ref=make_reference(d,n,opt.reference);
                    for(int extra:opt.extra_kib)
                        validate_pair(d,n,extra,ref,prop,validations);
                }
                std::cout<<"[PASS] CUDA correctness for all requested shapes and reservations\n";
            }
            if(opt.mode=="bench"||opt.mode=="both") run_benchmark(opt,prop,records,validations);
        } catch (const std::exception& e) {
            error=e.what();
        }
        write_json(opt,prop,records,validations,error,command);
        if (!error.empty()) std::cerr<<"[FAIL] "<<error<<"\n";
        return error.empty()?0:1;
    } catch(const std::exception& e) {
        std::cerr<<"[FAIL] "<<e.what()<<"\n";
        return 1;
    }
}
