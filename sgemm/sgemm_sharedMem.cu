#include "benchmark.cuh"

const int blockSize = 32;

__global__ void sgemm_sharedMem(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C){
        __shared__ float As[blockSize][blockSize], Bs[blockSize][blockSize];
        int row = blockSize * blockIdx.x + threadIdx.x / blockSize;
        int col = blockSize * blockIdx.y + threadIdx.x % blockSize;

        float tem = 0.0f;
        for(int t = 0; t < (K + blockSize - 1) / blockSize; t++){
                int block_row = t * blockSize + threadIdx.x / blockSize;//其实是thread的row以及col
                int block_col = t * blockSize + threadIdx.x % blockSize;
                if(row < M && block_col < K){
                        As[threadIdx.x / blockSize][threadIdx.x % blockSize] = A[row * K + block_col];
                }
                else
                        As[threadIdx.x / blockSize][threadIdx.x % blockSize] = 0;
                if(col < N && block_row < K){
                        Bs[threadIdx.x / blockSize][threadIdx.x % blockSize] = B[block_row * N + col];
                }
                else
                        Bs[threadIdx.x / blockSize][threadIdx.x % blockSize] = 0;
                __syncthreads();
                for(int i = 0; i < blockSize; i++)
                        tem += As[threadIdx.x / blockSize][i] * Bs[i][threadIdx.x % blockSize];

                __syncthreads();
        }
        if (row < M && col < N) {
    C[row * N + col] =
        alpha * tem + beta * C[row * N + col];
}
}

int main(int argc, char** argv) {
    return sgemm_bench::run(argc, argv, "sgemm_sharedMem",
        [](sgemm_bench::Shape s, float alpha, const float* A, const float* B, float beta, float* C) {
            dim3 block(blockSize * blockSize);
            dim3 grid((s.m + blockSize - 1) / blockSize, (s.n + blockSize - 1) / blockSize);
            sgemm_sharedMem<<<grid, block>>>(s.m, s.n, s.k, alpha, A, B, beta, C);
        });
}
