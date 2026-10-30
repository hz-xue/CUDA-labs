#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        cerr << #call << ": " << cudaGetErrorString(error) << endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

const int blockSize = 32;

__global__ void sgemm_coalesce_2D(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C){
        int col = blockDim.x * blockIdx.x + threadIdx.x;
        int row = blockDim.y * blockIdx.y + threadIdx.y;

        if(row < M && col < N){
               float tem = 0.0f;
                for(int i = 0; i < K; i++){
                        tem += A[row * K + i] * B[i * N + col];
                }
                C[row * N + col] = alpha * tem + beta * C[row * N + col];
        }
}

int main() {
    const int M = 4092;
    const int N = 4092;
    const int K = 4092;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    vector<float> h_A(M * K, 1.0f);
    vector<float> h_B(K * N, 2.0f);
    vector<float> h_C(M * N, 0.0f);

    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), sizeC, cudaMemcpyHostToDevice));

    // 每个 block 有多少个threads
    dim3 block(blockSize, blockSize);
    dim3 grid((N + blockSize - 1) / blockSize,
              (M + blockSize - 1) / blockSize);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    sgemm_coalesce_2D<<<grid, block>>>(
        M, N, K, alpha, d_A, d_B, beta, d_C
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));

    // 验证结果：每个元素应为 4092 * 1 * 2 = 8184
    const float expected = alpha * K * 2.0f;
    bool correct = true;
    for (int i = 0; i < M * N; ++i) {
        if (!std::isfinite(h_C[i]) || std::fabs(h_C[i] - expected) > 1e-4f) {
            correct = false;
            break;
        }
    }

    cout << "Correct: " << (correct ? "YES" : "NO") << endl;
    cout << "Kernel time: " << milliseconds << " ms" << endl;

    double gflops = (2.0 * M * N * K) / (milliseconds * 1e6);
    cout << "Performance: " << gflops << " GFLOPS" << endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}