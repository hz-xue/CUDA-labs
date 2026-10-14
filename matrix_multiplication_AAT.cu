#include <cstdio>
#include <vector>
const int TILE_DIM = 32;
__global__ void simpleMultiply(float *a, float *c, int M){
       int row = blockDim.y * blockIdx.y + threadIdx.y;
       int col = blockDim.x * blockIdx.x + threadIdx.x;
	
       float sum = 0.0f;
	for(int i = 0; i < TILE_DIM; i++){
		sum += a[row * TILE_DIM + i] * a[col * TILE_DIM + i];
	}
	c[row * M + col] = sum;
}

__global__ void coalescedMultiply(float *a, float *c, int M){
	int row = blockDim.y * blockIdx.y + threadIdx.y;
        int col = blockDim.x * blockIdx.x + threadIdx.x;
	
	__shared__ float aTile[TILE_DIM][TILE_DIM],
			 transposedTile[TILE_DIM][TILE_DIM];
	int row_begin = blockDim.y * blockIdx.y;
	int col_begin = blockDim.x * blockIdx.x;
	aTile[threadIdx.y][threadIdx.x] = a[row * TILE_DIM + threadIdx.x];
	transposedTile[threadIdx.x][threadIdx.y] = a[(col_begin + threadIdx.y) * TILE_DIM + threadIdx.x];
	
	__syncthreads();
        float sum = 0.0f;
        for(int i = 0; i < TILE_DIM; i++){
		sum += aTile[threadIdx.y][i] * transposedTile[i][threadIdx.x];
	}
	c[row * M + col] = sum;
}

__global__ void coalescedWithoutBankConflictMultiply(float *a, float *c, int M){
        int row = blockDim.y * blockIdx.y + threadIdx.y;
        int col = blockDim.x * blockIdx.x + threadIdx.x;

        __shared__ float aTile[TILE_DIM][TILE_DIM],
                         transposedTile[TILE_DIM][TILE_DIM + 1];
        int row_begin = blockDim.y * blockIdx.y;
        int col_begin = blockDim.x * blockIdx.x;
        aTile[threadIdx.y][threadIdx.x] = a[row * TILE_DIM + threadIdx.x];
        transposedTile[threadIdx.x][threadIdx.y] = a[(col_begin + threadIdx.y) * TILE_DIM + threadIdx.x];

        __syncthreads();
        float sum = 0.0f;
        for(int i = 0; i < TILE_DIM; i++){
                sum += aTile[threadIdx.y][i] * transposedTile[i][threadIdx.x];
        }
        c[row * M + col] = sum;
}
int main(){
	const int M = 64;
        const int W = 32;
        std::vector<float> h_a(M * W, 1.0f);
        std::vector<float> h_c(M * M);
    float *a,*c;
    cudaMalloc(&a, M * W * sizeof(float));
    cudaMalloc(&c, M * M * sizeof(float));

    cudaMemcpy(a, h_a.data(), M * W * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((M + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    coalescedWithoutBankConflictMultiply<<<grid, block>>>(a, c, M);

    cudaDeviceSynchronize();

    cudaMemcpy(h_c.data(), c, M * M * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < h_c.size(); ++i) {
    if (h_c[i] != 32.0f) {
        fprintf(stderr, "验证失败：元素 %zu = %f\n", i, h_c[i]);
        return 1;
    }
}

printf("验证通过，C 中所有元素都等于 %.1f\n", h_c[0]);

    cudaFree(a);
    cudaFree(c);
    return 0;
}
