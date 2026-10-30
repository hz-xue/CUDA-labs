#include "benchmark.cuh"

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

int main(int argc, char** argv) {
    return sgemm_bench::run(argc, argv, "sgemm_coalesce_2D",
        [](sgemm_bench::Shape s, float alpha, const float* A, const float* B, float beta, float* C) {
            dim3 block(blockSize, blockSize);
            dim3 grid((s.n + blockSize - 1) / blockSize, (s.m + blockSize - 1) / blockSize);
            sgemm_coalesce_2D<<<grid, block>>>(s.m, s.n, s.k, alpha, A, B, beta, C);
        });
}
