#include "benchmark.cuh"

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

__global__ void kernel5(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C){
        __shared__ float As[BM][BK], Bs[BK][BN];
        int strideA = 32, strideB = 2;
        float threadResult[TM][TN] = {0.0};
        float regM[TM] = {0.0f};
                float regN[TN] = {0.0f};

                int threadRow = threadIdx.x / (BN / TN) * TM;
                int threadCol = threadIdx.x % (BN / TN) * TN;
        for(int t = 0; t < (K + BK - 1) / BK; t++){
                for(int i = 0; i < 4; i ++){
                        int loadARow = blockIdx.x * BM + threadIdx.x / BK + i * strideA;
                        int loadACol = t * BK + threadIdx.x % BK;
                        if(loadARow < M && loadACol < K){
                                As[threadIdx.x / BK + i * strideA][threadIdx.x % BK] = A[loadARow * K + loadACol];
                        }
                        else
                                 As[threadIdx.x / BK + i * strideA][threadIdx.x % BK] = 0;

                        int loadBRow = t * BK + threadIdx.x / BN + i * strideB;
                        int loadBCol = blockIdx.y * BN + threadIdx.x % BN;

                        if(loadBRow < K && loadBCol < N){
                                Bs[threadIdx.x / BN + i * strideB][threadIdx.x % BN] = B[loadBRow * N + loadBCol];
                        }
                        else
                                Bs[threadIdx.x / BN + i * strideB][threadIdx.x % BN] = 0;
                }
                __syncthreads();

                for(int k = 0; k < BK; k++){
                        for(int m = 0; m < TM; m++){
                                regM[m] = As[threadRow + m][k];
                        }
                        for(int n = 0; n < TN; n++){
                                regN[n] = Bs[k][threadCol + n];
                        }

                        for(int m = 0; m < TM; m++){
                                for(int n = 0; n < TN; n++){
                                        threadResult[m][n] += regM[m] * regN[n];
                                }
                        }
                }
                __syncthreads();
        }

        for(int m = 0; m < TM; m++){
                for(int n = 0; n < TN; n++){
                if(blockIdx.x * BM + threadRow + m < M && blockIdx.y * BN + threadCol + n < N)
                        C[(blockIdx.x * BM + threadRow + m) * N + blockIdx.y * BN + threadCol + n] = alpha * threadResult[m][n] + beta * C[(blockIdx.x * BM + threadRow + m) * N + blockIdx.y * BN + threadCol + n];
                }
        }
}

int main(int argc, char** argv) {
    return sgemm_bench::run(argc, argv, "kernel5",
        [](sgemm_bench::Shape s, float alpha, const float* A, const float* B, float beta, float* C) {
            dim3 block((BM * BN) / (TM * TN));
            dim3 grid((s.m + BM - 1) / BM, (s.n + BN - 1) / BN);
            kernel5<<<grid, block>>>(s.m, s.n, s.k, alpha, A, B, beta, C);
        });
}
