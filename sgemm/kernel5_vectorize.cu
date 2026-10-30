#include "benchmark.cuh"

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

__global__ void kernel5_vectorize(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C){
        //尝试在GMEM -> SMEM的搬运时一个线程搬4个连续的float而非一个thread搬一个，循环4次
        __shared__ float As[BM][BK], Bs[BK][BN];
        float threadResult[TM][TN] = {0.0};
        float regM[TM] = {0.0f};
                float regN[TN] = {0.0f};

                int threadRow = threadIdx.x / (BN / TN) * TM;
                int threadCol = threadIdx.x % (BN / TN) * TN;
        for(int t = 0; t < (K + BK - 1) / BK; t++){
                int threadARow = threadIdx.x / (BK / 4);
                int threadACol = threadIdx.x % (BK / 4) * 4;
                int threadBRow = threadIdx.x / (BN / 4);
                int threadBCol = threadIdx.x % (BN / 4) * 4;

                for(int j = 0; j < 4; j++){
                if(blockIdx.x * BM + threadARow < M && t * BK + threadACol + j < K)
                        As[threadARow][threadACol + j] = A[(blockIdx.x * BM + threadARow) * K + t * BK + threadACol + j];
                else
                        As[threadARow][threadACol + j] = 0;
                if(blockIdx.y * BN + threadBCol + j < N && t * BK + threadBRow < K)
                        Bs[threadBRow][threadBCol + j] = B[(t * BK + threadBRow) * N + blockIdx.y * BN + threadBCol + j];
                else
                        Bs[threadBRow][threadBCol + j] = 0;
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
    return sgemm_bench::run(argc, argv, "kernel5_vectorize",
        [](sgemm_bench::Shape s, float alpha, const float* A, const float* B, float beta, float* C) {
            dim3 block((BM * BN) / (TM * TN));
            dim3 grid((s.m + BM - 1) / BM, (s.n + BN - 1) / BN);
            kernel5_vectorize<<<grid, block>>>(s.m, s.n, s.k, alpha, A, B, beta, C);
        });
}
