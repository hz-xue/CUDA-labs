#include "benchmark.cuh"

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8;

__global__ void kernel4(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C){
        //1D blockTile 一个thread计算C里的同一列的8个元素
        __shared__ float As[BM][BK], Bs[BK][BN];
        int ARow = blockIdx.x * BM + threadIdx.x / BK;
        int BCol = blockIdx.y * BN + threadIdx.x % BN;

        float threadResult[TM] = {0.0f};
        int col = threadIdx.x % BN;
                int row = threadIdx.x / BN * TM;

        for(int t = 0; t < (K + BK - 1) / BK; t++){
                int ACol = t * BK + threadIdx.x % BK;
                int BRow = t * BK + threadIdx.x / BN;
                if(ARow < M && ACol < K){
                        As[threadIdx.x / BK][threadIdx.x % BK] = A[ARow * K + ACol];
                }
                else
                        As[threadIdx.x / BK][threadIdx.x % BK] = 0;
                if(BCol < N && BRow < K){
                        Bs[threadIdx.x / BN][threadIdx.x % BN] = B[BRow * N + BCol];
                }
                else
                        Bs[threadIdx.x / BN][threadIdx.x % BN] = 0;
                __syncthreads();
                for(int k = 0; k < BK; k++){
                        float Btmp = Bs[k][col];
                        for(int i = 0; i < TM; i++){
                                threadResult[i] += As[row + i][k] * Btmp;
                        }
                }
                __syncthreads();
        }
        for(int i = 0; i < TM; i++){
                if(blockIdx.x * BM + row + i < M && blockIdx.y * BN + col < N)
                        C[(blockIdx.x * BM + row + i) * N + blockIdx.y * BN + col] = threadResult[i] * alpha + C[(blockIdx.x * BM + row + i) * N + blockIdx.y * BN + col] * beta;
                }
}

int main(int argc, char** argv) {
    return sgemm_bench::run(argc, argv, "kernel4",
        [](sgemm_bench::Shape s, float alpha, const float* A, const float* B, float beta, float* C) {
            dim3 block(512);
            dim3 grid((s.m + BM - 1) / BM, (s.n + BN - 1) / BN);
            kernel4<<<grid, block>>>(s.m, s.n, s.k, alpha, A, B, beta, C);
        });
}
