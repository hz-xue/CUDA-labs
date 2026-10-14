#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
constexpr int TILE_DIM = 32;
__global__ void simpleMultiply(
    const float *a,
    const float *b,
    float *c,
    int M, int W, int N
) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;

    for (int i = 0; i < W; ++i) {
        sum += a[row * W + i] * b[i * N + col];
    }

    c[row * N + col] = sum;
}

__global__ void coalescedMultiply(
    const float *a,
    const float *b,
    float *c,
    int M, int W, int N
) {
    __shared__ float aTile[TILE_DIM][TILE_DIM];


    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }
    
    aTile[threadIdx.y][threadIdx.x] = a[row * TILE_DIM + threadIdx.x];
    __syncwarp();
    float sum = 0.0f;

    for (int i = 0; i < TILE_DIM; ++i) {
        sum += aTile[threadIdx.y][i] * b[i * N + col];
    }

    c[row * N + col] = sum;
}

__global__ void sharedABMultiply(
    const float *a,
    const float *b,
    float *c,
    int M, int W, int N
) {
    __shared__ float aTile[TILE_DIM][TILE_DIM],
    		     bTile[TILE_DIM][TILE_DIM];


    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    aTile[threadIdx.y][threadIdx.x] = a[row * TILE_DIM + threadIdx.x];
    bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y * N + col];
    __syncthreads();
    float sum = 0.0f;

    for (int i = 0; i < TILE_DIM; ++i) {
        sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
    }

    c[row * N + col] = sum;
}

int main(){
	const int M = 64;
    	const int W = 32;
    	const int N = 96;
	std::vector<float> h_a(M * W, 1.0f);
    	std::vector<float> h_b(W * N, 2.0f);
    	std::vector<float> h_c(M * N);

	float *a, *b, *c;
	cudaMalloc(&a, M * W * sizeof(float));
	cudaMalloc(&b, N * W * sizeof(float));
	cudaMalloc(&c, M * N * sizeof(float));

	cudaMemcpy(a, h_a.data(), M * W * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b, h_b.data(), N * W * sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 block(32, 32);
	dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
	coalescedMultiply<<<grid, block>>>(a, b, c, M, W, N);

	cudaDeviceSynchronize();

	cudaMemcpy(h_c.data(), c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < h_c.size(); ++i) {
        if (h_c[i] != 64.0f) {
            fprintf(stderr, "验证失败：元素 %zu = %f\n", i, h_c[i]);
            return 1;
        }
    }

    printf("验证通过，C 中所有元素都等于 %.1f\n", h_c[0]);

    	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	return 0;
}
