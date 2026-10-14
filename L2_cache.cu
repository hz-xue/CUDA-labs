#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

void check(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA 错误：%s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

// 每个线程循环处理多个元素，覆盖整个流式数组。
// 持久化数组只读，流式数组的每个元素由唯一线程读写。
__global__ void kernel(
    const int *__restrict__ data_persistent,
    int *__restrict__ data_streaming,
    int dataSize,
    int freqSize
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = tid; index < dataSize; index += stride) {
        data_streaming[index] ^=
            data_persistent[index % freqSize];
    }
}

// 返回平均每次 kernel 的耗时，单位为毫秒。
float measure(
    int *data_persistent,
    int *data_streaming,
    int dataSize,
    int freqSize,
    size_t cacheSize,
    bool enablePersisting,
    int blocks,
    int threads,
    cudaStream_t stream
) {
    size_t windowBytes =
        static_cast<size_t>(freqSize) * sizeof(int);
    size_t streamingBytes =
        static_cast<size_t>(dataSize) * sizeof(int);

    // 确保前一组工作完成，再调整缓存配置。
    check(cudaStreamSynchronize(stream));
    check(cudaCtxResetPersistingL2Cache());

    // 对照组不预留持久化空间；实验组预留 cacheSize。
    check(cudaDeviceSetLimit(
        cudaLimitPersistingL2CacheSize,
        enablePersisting ? cacheSize : 0
    ));

    cudaStreamAttrValue attr{};

    if (enablePersisting) {
        attr.accessPolicyWindow.base_ptr = data_persistent;
        attr.accessPolicyWindow.num_bytes = windowBytes;
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    } else {
        // num_bytes 为 0：禁用访问策略窗口。
        attr.accessPolicyWindow.num_bytes = 0;
    }

    check(cudaStreamSetAttribute(
        stream, cudaStreamAttributeAccessPolicyWindow, &attr
    ));

    // 两组使用相同的初始数据；初始化不计时。
    // 按字节填充 1，每个 int 的值为 0x01010101。
    check(cudaMemsetAsync(
        data_persistent, 1, windowBytes, stream
    ));
    check(cudaMemsetAsync(
        data_streaming, 0, streamingBytes, stream
    ));

    constexpr int warmups = 5;
    constexpr int repeats = 100;

    // 使用与正式测试相同的启动配置预热。
    for (int j = 0; j < warmups; ++j) {
        kernel<<<blocks, threads, 0, stream>>>(
            data_persistent, data_streaming, dataSize, freqSize
        );
    }
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    check(cudaEventRecord(start, stream));

    for (int j = 0; j < repeats; ++j) {
        kernel<<<blocks, threads, 0, stream>>>(
            data_persistent, data_streaming, dataSize, freqSize
        );
    }

    check(cudaGetLastError());
    check(cudaEventRecord(stop, stream));
    check(cudaEventSynchronize(stop));

    float totalMs = 0;
    check(cudaEventElapsedTime(&totalMs, start, stop));

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));

    // 清除窗口，避免下一组继续使用旧地址。
    attr.accessPolicyWindow.num_bytes = 0;
    check(cudaStreamSetAttribute(
        stream, cudaStreamAttributeAccessPolicyWindow, &attr
    ));
    check(cudaCtxResetPersistingL2Cache());

    return totalMs / repeats;
}

int main() {
    check(cudaSetDevice(0));

    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0));

    printf("GPU：%s\n", prop.name);
    printf("计算能力：%d.%d\n", prop.major, prop.minor);
    printf("L2 总容量：%.2f MiB\n",
           prop.l2CacheSize / (1024.0 * 1024.0));
    printf("持久化 L2 最大容量：%.2f MiB\n",
           prop.persistingL2CacheMaxSize / (1024.0 * 1024.0));
    printf("最大访问窗口：%.2f MiB\n",
           prop.accessPolicyMaxWindowSize / (1024.0 * 1024.0));

    if (prop.persistingL2CacheMaxSize <= 0 ||
        prop.accessPolicyMaxWindowSize <= 0) {
        fprintf(stderr, "当前设备不支持本实验所需的持久化配置。\n");
        return 1;
    }

    // 保存原来的预留容量，结束时恢复。
    size_t originalSize = 0;
    check(cudaDeviceGetLimit(
        &originalSize, cudaLimitPersistingL2CacheSize
    ));

    size_t cacheSize = std::min(
        30ULL * 1024 * 1024,
        static_cast<unsigned long long>(prop.persistingL2CacheMaxSize)
    );

    check(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, cacheSize));

    size_t actualSize = 0;
    check(cudaDeviceGetLimit(
        &actualSize, cudaLimitPersistingL2CacheSize
    ));
    printf("实验组实际预留容量：%.2f MiB\n",
           actualSize / (1024.0 * 1024.0));

    cudaStream_t stream;
    check(cudaStreamCreate(&stream));

    const size_t streamingBytes = 1024ULL * 1024 * 1024;
    const int dataSize =
        static_cast<int>(streamingBytes / sizeof(int));

    const int threads = 256;
    const int blocks = prop.multiProcessorCount * 8;

    printf("启动配置：%d blocks × %d threads\n\n", blocks, threads);
    printf("窗口大小    关闭策略(ms)    开启策略(ms)    加速比\n");

    for (int sizeMiB = 10; sizeMiB <= 60; sizeMiB += 10) {
        size_t windowBytes = sizeMiB * 1024ULL * 1024;

        if (windowBytes >
            static_cast<size_t>(prop.accessPolicyMaxWindowSize)) {
            printf("%2d MiB：超过最大窗口，跳过\n", sizeMiB);
            continue;
        }

        int freqSize =
            static_cast<int>(windowBytes / sizeof(int));

        int *data_persistent = nullptr;
        int *data_streaming = nullptr;

        check(cudaMalloc(&data_persistent, windowBytes));
        check(cudaMalloc(&data_streaming, streamingBytes));

        float normalMs = measure(
            data_persistent, data_streaming,
            dataSize, freqSize, cacheSize, false,
            blocks, threads, stream
        );

        float persistingMs = measure(
            data_persistent, data_streaming,
            dataSize, freqSize, cacheSize, true,
            blocks, threads, stream
        );

        printf("%2d MiB      %10.4f      %10.4f      %.3fx\n",
               sizeMiB, normalMs, persistingMs,
               normalMs / persistingMs);

        check(cudaFree(data_persistent));
        check(cudaFree(data_streaming));
    }

    check(cudaStreamDestroy(stream));
    check(cudaDeviceSetLimit(
        cudaLimitPersistingL2CacheSize, originalSize
    ));

    return 0;
}
