// CUDA FP32 transpose: correctness and structured benchmark CLI.
// Does NOT establish GPU correctness until compiled and run on an NVIDIA GPU.
#include "transpose_kernels.cuh"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr) do {                                         \
    cudaError_t err__ = (expr);                                      \
    if (err__ != cudaSuccess) {                                      \
        std::ostringstream os__;                                     \
        os__ << #expr << " failed: " << cudaGetErrorString(err__);   \
        throw std::runtime_error(os__.str());                       \
    }                                                                \
} while (false)

namespace {
struct Args {
    std::string mode = "test";
    std::string sizes = "1,7,31,32,33,47,63,65,129";
    std::string json_path;
    int n = 8192;
    int warmup = 5;
    int repeat = 50;
    int samples = 5;
};

int parse_positive(const std::string& s) {
    auto x = std::stoll(s);
    if (x < 1 || x > std::numeric_limits<int>::max())
        throw std::invalid_argument("integer must be within [1, INT_MAX]");
    return static_cast<int>(x);
}
Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + argv[i]);
            return argv[++i];
        };
        const std::string arg = argv[i];
        if (arg == "--mode") args.mode = next();
        else if (arg == "--sizes") args.sizes = next();
        else if (arg == "--n") args.n = parse_positive(next());
        else if (arg == "--warmup") args.warmup = parse_positive(next());
        else if (arg == "--repeat") args.repeat = parse_positive(next());
        else if (arg == "--samples") args.samples = parse_positive(next());
        else if (arg == "--json") args.json_path = next();
        else if (arg == "--help") {
            std::cout << "--mode test|bench|all --sizes 1,31,32,33,65 "
                         "--n 8192 --warmup 5 --repeat 50 --samples 5 "
                         "--json results/summary/transpose_8192.json\n";
            std::exit(0);
        } else throw std::invalid_argument("unknown option: " + arg);
    }
    if (args.mode != "test" && args.mode != "bench" && args.mode != "all")
        throw std::invalid_argument("mode must be test, bench, or all");
    return args;
}
std::vector<int> parse_sizes(const std::string& s) {
    std::vector<int> result;
    std::istringstream is(s);
    std::string part;
    while (std::getline(is, part, ',')) result.push_back(parse_positive(part));
    if (result.empty()) throw std::invalid_argument("sizes cannot be empty");
    return result;
}

struct DeviceBuffers {
    float* input = nullptr;
    float* output = nullptr;
    explicit DeviceBuffers(std::size_t bytes) {
        try {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&input), bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), bytes));
        } catch (...) {
            if (input) cudaFree(input);
            if (output) cudaFree(output);
            input = nullptr;
            output = nullptr;
            throw;
        }
    }
    DeviceBuffers(const DeviceBuffers&) = delete;
    DeviceBuffers& operator=(const DeviceBuffers&) = delete;
    ~DeviceBuffers() {
        if (input) cudaFree(input);
        if (output) cudaFree(output);
    }
};

struct Events {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    Events() {
        CUDA_CHECK(cudaEventCreate(&start));
        try { CUDA_CHECK(cudaEventCreate(&stop)); }
        catch (...) { cudaEventDestroy(start); throw; }
    }
    ~Events() {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
    }
};

std::size_t count(int n) {
    return static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
}
std::vector<float> make_input(std::size_t elements) {
    std::vector<float> h(elements);
    for (std::size_t i = 0; i < elements; ++i) {
        std::uint32_t v = static_cast<std::uint32_t>(i);
        v ^= v >> 16;
        v *= 0x7feb352du;
        v ^= v >> 15;
        h[i] = static_cast<float>(v & 0x00ffffffu);
    }
    return h;
}

enum class Variant { Naive = 0, Shared = 1, Padded = 2, Copy = 3 };
constexpr int kVariantCount = 4;
const char* name(Variant v) {
    switch (v) {
        case Variant::Naive: return "naive";
        case Variant::Shared: return "shared";
        case Variant::Padded: return "padded";
        case Variant::Copy: return "copy_baseline";
    }
    return "unknown";
}
void launch(Variant variant, float* input, float* output, int n) {
    dim3 block(gpu_lab::TILE_DIM, gpu_lab::BLOCK_ROWS);
    dim3 grid((n + gpu_lab::TILE_DIM - 1) / gpu_lab::TILE_DIM,
              (n + gpu_lab::TILE_DIM - 1) / gpu_lab::TILE_DIM);
    switch (variant) {
        case Variant::Naive:
            gpu_lab::transpose_naive<<<grid, block>>>(input, output, n); break;
        case Variant::Shared:
            gpu_lab::transpose_shared<<<grid, block>>>(input, output, n); break;
        case Variant::Padded:
            gpu_lab::transpose_shared_padded<<<grid, block>>>(input, output, n); break;
        case Variant::Copy: {
            const int threads = 256;
            const std::size_t blocks = (count(n) + threads - 1) / threads;
            if (blocks > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()))
                throw std::invalid_argument("copy grid too large");
            gpu_lab::copy_kernel<<<static_cast<unsigned int>(blocks), threads>>>(
                input, output, count(n));
            break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void verify_on_gpu(int n) {
    auto elements = count(n);
    auto bytes = elements * sizeof(float);
    auto input = make_input(elements);
    std::vector<float> output(elements);
    DeviceBuffers d(bytes);
    CUDA_CHECK(cudaMemcpy(d.input, input.data(), bytes, cudaMemcpyHostToDevice));
    for (Variant v : {Variant::Naive, Variant::Shared, Variant::Padded, Variant::Copy}) {
        CUDA_CHECK(cudaMemset(d.output, 0xFF, bytes));
        launch(v, d.input, d.output, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(output.data(), d.output, bytes, cudaMemcpyDeviceToHost));
        for (std::size_t row = 0; row < static_cast<std::size_t>(n); ++row) {
            for (std::size_t col = 0; col < static_cast<std::size_t>(n); ++col) {
                const auto out_index = row * n + col;
                const auto in_index = v == Variant::Copy ? out_index : col * n + row;
                if (output[out_index] != input[in_index]) {
                    std::ostringstream os;
                    os << "FAIL " << name(v) << " n=" << n
                       << " at (" << row << "," << col << ")"
                       << " expected=" << input[in_index] << " actual=" << output[out_index];
                    throw std::runtime_error(os.str());
                }
            }
        }
        std::cout << "PASS correctness: " << name(v) << " n=" << n << "\n";
    }
}

double median(std::vector<double> x) {
    std::sort(x.begin(), x.end());
    const auto m = x.size() / 2;
    return x.size() % 2 ? x[m] : (x[m - 1] + x[m]) / 2.0;
}
struct BenchResult {
    std::vector<double> samples_ms;
    double median_ms = 0;
    double effective_gbps = 0;
};
struct AllResults {
    BenchResult variants[kVariantCount];
    cudaDeviceProp gpu{};
    int runtime_version = 0;
    int driver_version = 0;
};

AllResults benchmark(const Args& args) {
    // Correctness at the *same size* is required before timing.
    verify_on_gpu(args.n);
    const auto elements = count(args.n);
    const auto bytes = elements * sizeof(float);
    auto host = make_input(elements);
    DeviceBuffers d(bytes);
    CUDA_CHECK(cudaMemcpy(d.input, host.data(), bytes, cudaMemcpyHostToDevice));

    AllResults result;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&result.gpu, device));
    CUDA_CHECK(cudaRuntimeGetVersion(&result.runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&result.driver_version));
    for (int v = 0; v < kVariantCount; ++v) {
        for (int i = 0; i < args.warmup; ++i)
            launch(static_cast<Variant>(v), d.input, d.output, args.n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    Events events;
    // Rotate the A/B order across samples to moderate order and clock drift.
    for (int sample = 0; sample < args.samples; ++sample) {
        for (int shift = 0; shift < kVariantCount; ++shift) {
            int index = (sample + shift) % kVariantCount;
            auto v = static_cast<Variant>(index);
            CUDA_CHECK(cudaEventRecord(events.start));
            for (int iteration = 0; iteration < args.repeat; ++iteration)
                launch(v, d.input, d.output, args.n);
            CUDA_CHECK(cudaEventRecord(events.stop));
            CUDA_CHECK(cudaEventSynchronize(events.stop));
            float elapsed = 0;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, events.start, events.stop));
            result.variants[index].samples_ms.push_back(elapsed / args.repeat);
        }
    }
    const double useful_bytes = 2.0 * static_cast<double>(bytes); // one read + one write
    for (int v = 0; v < kVariantCount; ++v) {
        auto& r = result.variants[v];
        r.median_ms = median(r.samples_ms);
        r.effective_gbps = useful_bytes / (r.median_ms * 1e-3) / 1e9;
        std::cout << name(static_cast<Variant>(v)) << ": "
                  << std::fixed << std::setprecision(4) << r.median_ms << " ms, "
                  << std::setprecision(2) << r.effective_gbps << " effective GB/s\n";
    }
    auto& results = result.variants;
    std::cout << "Shared/Naive speedup: "
              << results[static_cast<int>(Variant::Naive)].median_ms /
                 results[static_cast<int>(Variant::Shared)].median_ms << "x\n";
    std::cout << "Padded/Shared speedup: "
              << results[static_cast<int>(Variant::Shared)].median_ms /
                 results[static_cast<int>(Variant::Padded)].median_ms << "x\n";
    return result;
}

void save_json(const Args& args, const AllResults& results) {
    if (args.json_path.empty()) return;
    std::ofstream o(args.json_path);
    if (!o) throw std::runtime_error("cannot open --json output path: " + args.json_path);
    o << std::fixed << std::setprecision(6);
    o << "{\n  \"record_type\": \"current_gpu_benchmark\",\n"
      << "  \"validation_status\": \"gpu_correctness_and_benchmark_passed\",\n"
      << "  \"benchmark\": \"cuda_fp32_square_transpose\",\n"
      << "  \"device\": {\"name\": \"" << results.gpu.name << "\", "
      << "\"compute_capability\": \"sm_" << results.gpu.major << results.gpu.minor << "\", "
      << "\"cuda_runtime_version\": " << results.runtime_version << ", "
      << "\"cuda_driver_version\": " << results.driver_version << ", "
      << "\"nvcc_version\": \"" << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "\"},\n"
      << "  \"shape\": {\"n\": " << args.n << ", \"dtype\": \"fp32\"},\n"
      << "  \"methodology\": {\"warmup_launches\": " << args.warmup
      << ", \"repeats_per_sample\": " << args.repeat
      << ", \"samples\": " << args.samples
      << ", \"timing\": \"cuda_events_per_batch_divided_by_repeat\", "
      << "\"effective_bandwidth_definition\": \"2*n*n*sizeof(float)/median_latency\"},\n"
      << "  \"results\": {\n";
    for (int idx = 0; idx < kVariantCount; ++idx) {
        const auto& r = results.variants[idx];
        o << "    \"" << name(static_cast<Variant>(idx)) << "\": {"
          << "\"median_latency_ms\": " << r.median_ms << ", "
          << "\"effective_gbps\": " << r.effective_gbps << ", "
          << "\"sample_latency_ms\": [";
        for (std::size_t i = 0; i < r.samples_ms.size(); ++i) {
            if (i) o << ", ";
            o << r.samples_ms[i];
        }
        o << "]}" << (idx == kVariantCount - 1 ? "\n" : ",\n");
    }
    o << "  },\n  \"notes\": "
      << "\"The copy kernel is a streaming reference, not an equivalent transpose algorithm. "
         "Bandwidth is useful-byte effective throughput, not a DRAM hardware counter.\"\n}\n";
    if (!o) throw std::runtime_error("failed to write JSON: " + args.json_path);
    std::cout << "Wrote: " << args.json_path << "\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        if (args.mode == "test" || args.mode == "all") {
            for (int n : parse_sizes(args.sizes)) verify_on_gpu(n);
        }
        if (args.mode == "bench" || args.mode == "all") {
            auto results = benchmark(args);
            save_json(args, results);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
