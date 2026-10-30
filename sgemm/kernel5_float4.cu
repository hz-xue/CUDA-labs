#include "benchmark.cuh"
#include <cstdint>

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

// p points to at least one valid float. Only issue a 16-byte load when
// all four values exist in this row and the actual address is aligned.
__device__ __forceinline__ float4 load_float4_or_tail(const float* p, int remaining) {
    if (remaining >= 4 && (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0) {
        return *reinterpret_cast<const float4*>(p);
    }
    float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (remaining > 0) value.x = p[0];
    if (remaining > 1) value.y = p[1];
    if (remaining > 2) value.z = p[2];
    if (remaining > 3) value.w = p[3];
    return value;
}

__global__ void kernel5_float4(int M, int N, int K, float alpha,
                              const float* A, const float* B, float beta, float* C) {
    __shared__ float As[BM][BK], Bs[BK][BN];
    const int tid = threadIdx.x;
    const int threadRow = tid / (BN / TN) * TM;
    const int threadCol = tid % (BN / TN) * TN;
    // Same element ownership as kernel5_vectorize: 256 threads load
    // 256 groups of four floats from each input tile.
    const int aRow = tid / (BK / 4);
    const int aCol = tid % (BK / 4) * 4;
    const int bRow = tid / (BN / 4);
    const int bCol = tid % (BN / 4) * 4;
    float results[TM][TN] = {};
    float regM[TM], regN[TN];

    for (int tile = 0; tile < (K + BK - 1) / BK; ++tile) {
        const int globalARow = blockIdx.x * BM + aRow;
        const int globalACol = tile * BK + aCol;
        const int globalBRow = tile * BK + bRow;
        const int globalBCol = blockIdx.y * BN + bCol;
        float4 av = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 bv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        // Check bounds before forming a pointer; padding is zero-filled.
        if (globalARow < M && globalACol < K)
            av = load_float4_or_tail(A + globalARow * K + globalACol, K - globalACol);
        if (globalBRow < K && globalBCol < N)
            bv = load_float4_or_tail(B + globalBRow * N + globalBCol, N - globalBCol);

        As[aRow][aCol + 0] = av.x;
        As[aRow][aCol + 1] = av.y;
        As[aRow][aCol + 2] = av.z;
        As[aRow][aCol + 3] = av.w;
        Bs[bRow][bCol + 0] = bv.x;
        Bs[bRow][bCol + 1] = bv.y;
        Bs[bRow][bCol + 2] = bv.z;
        Bs[bRow][bCol + 3] = bv.w;
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            for (int m = 0; m < TM; ++m) regM[m] = As[threadRow + m][k];
            for (int n = 0; n < TN; ++n) regN[n] = Bs[k][threadCol + n];
            for (int m = 0; m < TM; ++m)
                for (int n = 0; n < TN; ++n)
                    results[m][n] += regM[m] * regN[n];
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; ++m) {
        const int row = blockIdx.x * BM + threadRow + m;
        for (int n = 0; n < TN; ++n) {
            const int col = blockIdx.y * BN + threadCol + n;
            if (row < M && col < N)
                C[row * N + col] = alpha * results[m][n] + beta * C[row * N + col];
        }
    }
}

int main(int argc, char** argv) {
    return sgemm_bench::run(argc, argv, "kernel5_float4",
        [](sgemm_bench::Shape s, float alpha, const float* A, const float* B, float beta, float* C) {
            dim3 block((BM * BN) / (TM * TN));
            dim3 grid((s.m + BM - 1) / BM, (s.n + BN - 1) / BN);
            kernel5_float4<<<grid, block>>>(s.m, s.n, s.k, alpha, A, B, beta, C);
        });
}
