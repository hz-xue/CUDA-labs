#include "benchmark.cuh"

const int blockSize = 32;

__global__ void sgemm_coalesce(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C){
        int row = blockSize * blockIdx.x + threadIdx.x / blockSize;
        int col = blockSize * blockIdx.y + threadIdx.x % blockSize;

        if(row < M && col < N){
               float tem = 0.0f;
                for(int i = 0; i < K; i++){
                        tem += A[row * K + i] * B[i * N + col];
                }
                C[row * N + col] = alpha * tem + beta * C[row * N + col];
        }
}

int main(int argc, char** argv) {
    return sgemm_bench::run(argc, argv, "sgemm_coalesce",
        [](sgemm_bench::Shape s, float alpha, const float* A, const float* B, float beta, float* C) {
            dim3 block(blockSize * blockSize);
            dim3 grid((s.m + blockSize - 1) / blockSize, (s.n + blockSize - 1) / blockSize);
            sgemm_coalesce<<<grid, block>>>(s.m, s.n, s.k, alpha, A, B, beta, C);
        });
}
