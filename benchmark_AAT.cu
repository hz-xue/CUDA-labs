#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

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

    __shared__ float aTile[TILE_DIM][TILE_DIM];
    __shared__ float transposedTile[TILE_DIM][TILE_DIM];

    int col_begin = blockDim.x * blockIdx.x;

    aTile[threadIdx.y][threadIdx.x] =
        a[row * TILE_DIM + threadIdx.x];

    transposedTile[threadIdx.x][threadIdx.y] =
        a[(col_begin + threadIdx.y) * TILE_DIM + threadIdx.x];

    __syncthreads();

    float sum = 0.0f;
    for(int i = 0; i < TILE_DIM; i++){
        sum += aTile[threadIdx.y][i]
             * transposedTile[i][threadIdx.x];
    }

    c[row * M + col] = sum;
}

__global__ void coalescedWithoutBankConflictMultiply(float *a, float *c, int M){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float aTile[TILE_DIM][TILE_DIM];
    __shared__ float transposedTile[TILE_DIM][TILE_DIM + 1];

    int col_begin = blockDim.x * blockIdx.x;

    aTile[threadIdx.y][threadIdx.x] =
        a[row * TILE_DIM + threadIdx.x];

    transposedTile[threadIdx.x][threadIdx.y] =
        a[(col_begin + threadIdx.y) * TILE_DIM + threadIdx.x];

    __syncthreads();

    float sum = 0.0f;
    for(int i = 0; i < TILE_DIM; i++){
        sum += aTile[threadIdx.y][i]
             * transposedTile[i][threadIdx.x];
    }

    c[row * M + col] = sum;
}

void benchmark(const char *name,
               int version,
               float *a,
               float *c,
               dim3 grid,
               dim3 block,
               int M,
               int repeat)
{
    // warm up
    for(int i = 0; i < 100; i++){
        if(version == 0)
            simpleMultiply<<<grid, block>>>(a, c, M);
        else if(version == 1)
            coalescedMultiply<<<grid, block>>>(a, c, M);
        else
            coalescedWithoutBankConflictMultiply<<<grid, block>>>(a, c, M);
    }

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for(int i = 0; i < repeat; i++){
        if(version == 0)
            simpleMultiply<<<grid, block>>>(a, c, M);
        else if(version == 1)
            coalescedMultiply<<<grid, block>>>(a, c, M);
        else
            coalescedWithoutBankConflictMultiply<<<grid, block>>>(a, c, M);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    printf("%-40s total = %10.3f ms   avg = %10.3f us\n",
           name,
           ms,
           ms * 1000.0f / repeat);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(){
    const int M = 4096;
    const int W = 32;
    const int REPEAT = 100;

    std::vector<float> h_a(M * W, 1.0f);

    float *a, *c;

    cudaMalloc(&a, M * W * sizeof(float));
    cudaMalloc(&c, M * M * sizeof(float));

    cudaMemcpy(a,
               h_a.data(),
               M * W * sizeof(float),
               cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(M / TILE_DIM, M / TILE_DIM);

    benchmark("simpleMultiply",
              0, a, c, grid, block, M, REPEAT);

    benchmark("coalescedMultiply",
              1, a, c, grid, block, M, REPEAT);

    benchmark("coalescedWithoutBankConflictMultiply",
              2, a, c, grid, block, M, REPEAT);

    cudaFree(a);
    cudaFree(c);

    return 0;
}
