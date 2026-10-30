#include "benchmark.cuh"

const int blockSize = 32;

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C){
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        int col = blockDim.y * blockIdx.y + threadIdx.y;

        if(row < M && col < N){
               float tem = 0.0f;
                for(int i = 0; i < K; i++){
                        tem += A[row * K + i] * B[i * N + col];
                }
                C[row * N + col] = alpha * tem + beta * C[row * N + col];
        }
}

int main(int argc, char** argv) {
    return sgemm_bench::run(argc, argv, "sgemm_naive",
        [](sgemm_bench::Shape s, float alpha, const float* A, const float* B, float beta, float* C) {
            dim3 block(blockSize, blockSize);
            dim3 grid((s.m + blockSize - 1) / blockSize, (s.n + blockSize - 1) / blockSize);
            sgemm_naive<<<grid, block>>>(s.m, s.n, s.k, alpha, A, B, beta, C);
        });
}
