#include "benchmark.cuh"

const int blockSize = 32;

__global__ void sgemm_sharedMem_2D(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C){
        __shared__ float As[blockSize][blockSize], Bs[blockSize][blockSize];
        int col = blockDim.x * blockIdx.x + threadIdx.x;
        int row = blockDim.y * blockIdx.y + threadIdx.y;
        float tem = 0.0f;
        for(int t = 0; t < (K + blockSize - 1) / blockSize; t++){
                if( t * blockSize + threadIdx.x < K && row < M)
                        As[threadIdx.y][threadIdx.x] = A[row * K + t * blockSize + threadIdx.x];
                else
                        As[threadIdx.y][threadIdx.x] = 0.0f;
                if(col < N && t * blockSize + threadIdx.y < K)
                        Bs[threadIdx.y][threadIdx.x] = B[(t * blockSize + threadIdx.y) * N + col];
                else
                        Bs[threadIdx.y][threadIdx.x] = 0.0f;

                __syncthreads();

                for(int i = 0; i < blockSize; i++){
                        tem += As[threadIdx.y][i] * Bs[i][threadIdx.x];
                }
                __syncthreads(); //防止其他thread没计算完就覆盖了shared memory
        }
        if (row < M && col < N)
                C[row * N + col] = tem * alpha + C[row * N + col] * beta;
}

int main(int argc, char** argv) {
    return sgemm_bench::run(argc, argv, "sgemm_sharedMem_2D",
        [](sgemm_bench::Shape s, float alpha, const float* A, const float* B, float beta, float* C) {
            dim3 block(blockSize, blockSize);
            dim3 grid((s.n + blockSize - 1) / blockSize, (s.m + blockSize - 1) / blockSize);
            sgemm_sharedMem_2D<<<grid, block>>>(s.m, s.n, s.k, alpha, A, B, beta, C);
        });
}
