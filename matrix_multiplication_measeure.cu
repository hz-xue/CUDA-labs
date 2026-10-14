#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>

constexpr int TILE_DIM = 32;

void check(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA 错误：%s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

__global__ void simpleMultiply(
    const float *a, const float *b, float *c,
    int M, int W, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int i = 0; i < W; ++i) {
        sum += a[row * W + i] * b[i * N + col];
    }
    c[row * N + col] = sum;
}

__global__ void coalescedMultiply(
    const float *a, const float *b, float *c,
    int M, int W, int N
) {
    __shared__ float aTile[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    aTile[ty][tx] = a[row * W + tx];
    __syncwarp();

    float sum = 0.0f;
    for (int i = 0; i < TILE_DIM; ++i) {
        sum += aTile[ty][i] * b[i * N + col];
    }
    c[row * N + col] = sum;
}

__global__ void sharedABMultiply(
    const float *a, const float *b, float *c,
    int M, int W, int N
) {
    __shared__ float aTile[TILE_DIM][TILE_DIM];
    __shared__ float bTile[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    aTile[ty][tx] = a[row * W + tx];
    bTile[ty][tx] = b[ty * N + col];
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < TILE_DIM; ++i) {
        sum += aTile[ty][i] * bTile[i][tx];
    }
    c[row * N + col] = sum;
}

// 根据编号启动对应 kernel
void launch(
    int version,
    dim3 grid, dim3 block,
    const float *a, const float *b, float *c,
    int M, int W, int N
) {
    switch (version) {
        case 0:
            simpleMultiply<<<grid, block>>>(a, b, c, M, W, N);
            break;
        case 1:
            coalescedMultiply<<<grid, block>>>(a, b, c, M, W, N);
            break;
        case 2:
            sharedABMultiply<<<grid, block>>>(a, b, c, M, W, N);
            break;
    }
}

int main() {
    // 可以先改回 M=64、N=96。
    // 小矩阵通常主要受启动开销影响，不容易观察优化收益。
    constexpr int M = 2048;
    constexpr int W = 32;
    constexpr int N = 2048;

    static_assert(W == TILE_DIM, "本版本要求 W == TILE_DIM");
    static_assert(M % TILE_DIM == 0 && N % TILE_DIM == 0,
                  "本版本要求 M、N 是 TILE_DIM 的倍数");

    std::vector<float> h_a(M * W);
    std::vector<float> h_b(W * N);
    std::vector<float> h_c(M * N);

    // 使用不完全相同的数据，避免全为常量时掩盖索引错误。
    for (int row = 0; row < M; ++row) {
        for (int k = 0; k < W; ++k) {
            h_a[row * W + k] =
                static_cast<float>((row + 3 * k) % 7 - 3);
        }
    }

    for (int k = 0; k < W; ++k) {
        for (int col = 0; col < N; ++col) {
            h_b[k * N + col] =
                static_cast<float>((2 * k + col) % 5 - 2);
        }
    }

    // CPU 参考结果，不计入 GPU 时间。
    std::vector<float> reference(M * N);
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < W; ++k) {
                sum += h_a[row * W + k] * h_b[k * N + col];
            }
            reference[row * N + col] = sum;
        }
    }

    size_t bytesA = h_a.size() * sizeof(float);
    size_t bytesB = h_b.size() * sizeof(float);
    size_t bytesC = h_c.size() * sizeof(float);

    float *a = nullptr, *b = nullptr, *c = nullptr;
    check(cudaMalloc(&a, bytesA));
    check(cudaMalloc(&b, bytesB));
    check(cudaMalloc(&c, bytesC));

    check(cudaMemcpy(a, h_a.data(), bytesA, cudaMemcpyHostToDevice));
    check(cudaMemcpy(b, h_b.data(), bytesB, cudaMemcpyHostToDevice));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(N / TILE_DIM, M / TILE_DIM);

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    const char *names[] = {
        "simpleMultiply",
        "coalescedMultiply",
        "sharedABMultiply"
    };

    constexpr int warmups = 10;
    constexpr int repeats = 100;

    printf("M=%d, W=%d, N=%d\n", M, W, N);

    for (int version = 0; version < 3; ++version) {
        // 先用 NaN 填充输出，帮助发现未写入的元素。
        std::fill(h_c.begin(), h_c.end(), NAN);
        check(cudaMemcpy(c, h_c.data(), bytesC, cudaMemcpyHostToDevice));

        for (int j = 0; j < warmups; ++j) {
            launch(version, grid, block, a, b, c, M, W, N);
        }
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());

        check(cudaEventRecord(start));

        for (int j = 0; j < repeats; ++j) {
            launch(version, grid, block, a, b, c, M, W, N);
        }

        check(cudaGetLastError());
        check(cudaEventRecord(stop));
        check(cudaEventSynchronize(stop));

        float totalMs = 0.0f;
        check(cudaEventElapsedTime(&totalMs, start, stop));

        check(cudaMemcpy(
            h_c.data(), c, bytesC, cudaMemcpyDeviceToHost
        ));

        for (size_t index = 0; index < h_c.size(); ++index) {
            if (!std::isfinite(h_c[index]) ||
                std::fabs(h_c[index] - reference[index]) > 1e-4f) {
                fprintf(stderr,
                        "%s 验证失败：元素 %zu，GPU=%f，CPU=%f\n",
                        names[version], index,
                        h_c[index], reference[index]);
                std::exit(1);
            }
        }

        printf("%-20s：平均 %9.3f 微秒/次，验证通过\n",
               names[version], totalMs * 1000.0f / repeats);
    }

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaFree(a));
    check(cudaFree(b));
    check(cudaFree(c));
    return 0;
}
