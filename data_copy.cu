#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call) do {                                  \
    cudaError_t err = (call);                             \
    if (err != cudaSuccess) {                            \
        fprintf(stderr, "Line %d: %s\n", __LINE__,        \
                cudaGetErrorString(err));                \
        exit(EXIT_FAILURE);                              \
    }                                                    \
} while (0)

__global__ void strideCopy(float *odata, const float *idata, int stride)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t xid = tid * stride;
    odata[xid] = idata[xid];
}

int main()
{
    const int N = 1024 * 1024;  // 每种 stride 都复制 N 个 float
    const int max_stride = 32;
    const int threads = 256;
    const int blocks = N / threads;
    const int repeats = 50;

    const size_t count = (size_t)N * max_stride;
    const size_t size = count * sizeof(float);

    float *odata, *idata, *h_data;
    CHECK(cudaMalloc(&odata, size));
    CHECK(cudaMalloc(&idata, size));
    CHECK(cudaMallocHost(&h_data, size));

    for (size_t i = 0; i < count; ++i) {
        h_data[i] = (float)(i % 1024);
    }

    CHECK(cudaMemcpy(idata, h_data, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("stride,time_ms,effective_GB_s,ratio_to_stride1\n");

    double baseline_bandwidth = 0.0;

    for (int stride = 1; stride <= max_stride; ++stride) {
        // 预热，不计入时间
        for (int i = 0; i < 5; ++i) {
            strideCopy<<<blocks, threads>>>(odata, idata, stride);
        }
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(start));

        for (int i = 0; i < repeats; ++i) {
            strideCopy<<<blocks, threads>>>(odata, idata, stride);
        }

        CHECK(cudaEventRecord(stop));
        CHECK(cudaGetLastError());
        CHECK(cudaEventSynchronize(stop));

        float total_ms;
        CHECK(cudaEventElapsedTime(&total_ms, start, stop));

        double average_ms = total_ms / repeats;

        // 只计算实际需要的 N 个 float：读取一次、写入一次
        double bandwidth =
            (2.0 * N * sizeof(float)) / (average_ms * 1e6);

        if (stride == 1) {
            baseline_bandwidth = bandwidth;
        }

        printf("%d,%.6f,%.3f,%.4f\n",
               stride, average_ms, bandwidth,
               bandwidth / baseline_bandwidth);

        // 校验输出；不计入时间
        CHECK(cudaMemcpy(h_data, odata, size, cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < (size_t)N; ++i) {
            size_t xid = i * stride;
            float expected = (float)(xid % 1024);

            if (h_data[xid] != expected) {
                fprintf(stderr, "Mismatch: stride=%d, index=%zu\n",
                        stride, xid);
                return EXIT_FAILURE;
            }
        }
    }

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(odata));
    CHECK(cudaFree(idata));
    CHECK(cudaFreeHost(h_data));

    return 0;
}
