# SGEMM kernels

从原始代码拆分而来，每个 `.cu` 文件包含一个 kernel 和独立的测试入口，计算行主序矩阵 `C = alpha * A * B + beta * C`。

| 文件 | 实现 | block | grid 的 x / y |
| --- | --- | --- | --- |
| `sgemm_naive.cu` | 朴素实现 | `(32, 32)` | `ceil(M/32)` / `ceil(N/32)` |
| `sgemm_coalesce.cu` | 一维线程布局、合并访存 | `1024` | `ceil(M/32)` / `ceil(N/32)` |
| `sgemm_coalesce_2D.cu` | 二维线程布局、合并访存 | `(32, 32)` | `ceil(N/32)` / `ceil(M/32)` |
| `sgemm_sharedMem.cu` | 一维线程布局、共享内存分块 | `1024` | `ceil(M/32)` / `ceil(N/32)` |
| `sgemm_sharedMem_2D.cu` | 二维线程布局、共享内存分块 | `(32, 32)` | `ceil(N/32)` / `ceil(M/32)` |
| `kernel4.cu` | 1D block tiling，每线程计算同列 8 个元素 | `512` | `ceil(M/64)` / `ceil(N/64)` |

## 编译运行

需要 CUDA Toolkit、兼容的 C++ 主机编译器和 NVIDIA GPU。在此目录下逐个编译，例如：

```sh
nvcc -O3 sgemm_naive.cu -o sgemm_naive
./sgemm_naive
nvcc -O3 kernel4.cu -o kernel4
./kernel4
```

Windows 下生成 `.exe`，例如 `nvcc -O3 kernel4.cu -o kernel4.exe` 后运行 `.\kernel4.exe`。

默认沿用原始测试：`M=N=K=4092`、`A=1`、`B=2`、初始 `C=0`、`alpha=1`、`beta=0`，每个结果应为 `8184`。输出正确性、单次 kernel 耗时和 GFLOPS；验证失败或 CUDA 调用失败时返回非零状态。单次计时仅供练习参考。

保留原始 kernel 函数名、计算逻辑和注释，仅拆分文件、补齐头文件，并为各实现配置对应的启动参数和 CUDA 错误检查。
