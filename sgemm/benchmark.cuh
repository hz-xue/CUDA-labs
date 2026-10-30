#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::cerr << #call << ": " << cudaGetErrorString(error) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

namespace sgemm_bench {
struct Shape { int m, n, k; };

// Allocate outside timing; RAII also releases buffers if validation fails.
struct Buffer {
    float* data = nullptr;
    explicit Buffer(size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(float)));
    }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

inline void upload(float* dst, const std::vector<float>& src) {
    CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
}

// Validate every output for small cases, and deterministic samples for large cases.
// CPU dot products accumulate in double; tolerance accounts for FP32 accumulation.
inline bool verify(const Shape& s, const std::vector<float>& a,
                   const std::vector<float>& b, const std::vector<float>& initial,
                   const std::vector<float>& result, float alpha, float beta,
                   bool full) {
    const size_t count = size_t(s.m) * s.n;
    const size_t checks = full ? count : std::min<size_t>(64, count);
    std::mt19937 rng(20261031);
    for (size_t q = 0; q < checks; ++q) {
        size_t index = full ? q : (q == 0 ? 0 : (q == 1 ? count - 1 : rng() % count));
        const size_t row = index / s.n, col = index % s.n;
        double sum = 0;
        for (int k = 0; k < s.k; ++k)
            sum += double(a[row * s.k + k]) * b[size_t(k) * s.n + col];
        const double expected = alpha * sum + beta * initial[index];
        const double tolerance = 1e-3 + 1e-3 * std::abs(expected);
        if (!std::isfinite(result[index]) || std::abs(result[index] - expected) > tolerance) {
            std::cerr << "FAIL at (" << row << ',' << col << "): got "
                      << result[index] << ", expected " << expected << '\n';
            return false;
        }
    }
    return true;
}

template<class Launch>
bool run_case(Shape s, Launch launch, bool timing, int warmup, int repeats, int rounds) {
    const size_t na = size_t(s.m) * s.k, nb = size_t(s.k) * s.n, nc = size_t(s.m) * s.n;
    std::vector<float> a(na), b(nb), initial(nc), result(nc);
    std::mt19937 rng(20261031);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : a) v = dist(rng);
    for (float& v : b) v = dist(rng);
    for (float& v : initial) v = dist(rng);
    Buffer da(na), db(nb), dc(nc);
    upload(da.data, a); upload(db.data, b);
    // Nonzero beta and nonunit alpha catch writeback bugs. Restore C for each test.
    for (int test = 0; test < 2; ++test) {
        const float alpha = test == 0 ? 0.5f : 1.0f;
        const float beta = test == 0 ? 0.75f : 0.0f;
        upload(dc.data, initial);
        launch(s, alpha, da.data, db.data, beta, dc.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(result.data(), dc.data, nc * sizeof(float), cudaMemcpyDeviceToHost));
        if (!verify(s, a, b, initial, result, alpha, beta, !timing)) return false;
    }
    std::cout << "M=" << s.m << " N=" << s.n << " K=" << s.k
              << (timing ? " sampled validation PASS" : " full validation PASS");
    if (!timing) { std::cout << '\n'; return true; }

    // beta=0 keeps repeated launches mathematically independent.
    for (int i = 0; i < warmup; ++i) launch(s, 1.0f, da.data, db.data, 0.0f, dc.data);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> samples;
    for (int r = 0; r < rounds; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeats; ++i) launch(s, 1.0f, da.data, db.data, 0.0f, dc.data);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (!(ms > 0) || !std::isfinite(ms)) {
            std::cerr << "Invalid timer result\n";
            CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
            return false;
        }
        samples.push_back(ms / repeats);
    }
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(samples.begin(), samples.end());
    const float median = rounds % 2 ? samples[rounds / 2]
                                    : (samples[rounds / 2 - 1] + samples[rounds / 2]) / 2;
    const double gflops = 2.0 * s.m * s.n * s.k / (median * 1e6);
    std::cout << std::fixed << std::setprecision(4) << ": median=" << median
              << " ms min=" << samples.front() << " max=" << samples.back()
              << " GFLOPS=" << gflops << '\n';
    return true;
}

inline int positive(const char* arg) {
    size_t used;
    const long long value = std::stoll(arg, &used);
    if (arg[used] != '\0' || value <= 0 || value > 16384)
        throw std::invalid_argument("expected an integer in [1,16384]");
    return static_cast<int>(value);
}

template<class Launch>
int run(int argc, char** argv, const char* name, Launch launch) {
    if (argc != 1 && argc != 4 && argc != 7) {
        std::cerr << "Usage: " << argv[0] << " [M N K [warmup repeats rounds]]\n";
        return EXIT_FAILURE;
    }
    try {
        int warmup = 10, repeats = 50, rounds = 5;
        std::vector<Shape> sizes = {{128,128,128}, {256,256,256}, {512,512,512},
                                   {1024,1024,1024}, {2048,2048,2048},
                                   {4096,4096,4096}, {4092,4092,4092}};
        if (argc >= 4) sizes = {{positive(argv[1]), positive(argv[2]), positive(argv[3])}};
        if (argc == 7) { warmup=positive(argv[4]); repeats=positive(argv[5]); rounds=positive(argv[6]); }
        int device; cudaDeviceProp props;
        CUDA_CHECK(cudaGetDevice(&device)); CUDA_CHECK(cudaGetDeviceProperties(&props, device));
        std::cout << name << " on " << props.name << ": warmup=" << warmup
                  << " repeats=" << repeats << " rounds=" << rounds << '\n';
        // Small full checks include rectangular, edge, and partial K tiles.
        for (Shape s : {Shape{1,1,1}, Shape{17,33,9}, Shape{65,67,17}, Shape{129,131,15}})
            if (!run_case(s, launch, false, warmup, repeats, rounds)) return EXIT_FAILURE;
        for (Shape s : sizes)
            if (!run_case(s, launch, true, warmup, repeats, rounds)) return EXIT_FAILURE;
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return EXIT_FAILURE;
    }
}
} // namespace sgemm_bench
