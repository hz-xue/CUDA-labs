# SGEMM kernels

每个 `.cu` 文件包含一个 kernel 和独立入口，公共测试逻辑位于 `benchmark.cuh`。
均计算行主序矩阵 `C = alpha * A * B + beta * C`。

| 文件 | 实现 | 每个 block 的线程数 |
| --- | --- | --- |
| `sgemm_naive.cu` | 朴素实现 | 32 × 32 |
| `sgemm_coalesce.cu` | 一维线程布局、合并访存 | 1024 |
| `sgemm_coalesce_2D.cu` | 二维线程布局、合并访存 | 32 × 32 |
| `sgemm_sharedMem.cu` | 一维线程布局、共享内存分块 | 1024 |
| `sgemm_sharedMem_2D.cu` | 二维线程布局、共享内存分块 | 32 × 32 |
| `kernel4.cu` | 1D thread tiling，每线程 8 × 1 个结果 | 512 |
| `kernel5.cu` | 2D thread tiling，每线程 8 × 8 个结果 | 256 |
| `kernel5_vectorize.cu` | 每线程搬运 4 个连续标量的实验 | 256 |
| `kernel5_float4.cu` | 显式 float4 全局内存加载，含对齐与尾部回退 | 256 |

三个 kernel5 使用 BM=BN=128，BK=TM=TN=8。修复原始代码写回时遗漏 alpha/beta 的问题。
`kernel5_vectorize` 保留原始标量读取循环，没有显式使用 `float4`；不能仅凭名称认定生成了 128-bit 加载指令，需检查 SASS。

`kernel5_float4` 保持同样的线程映射、共享内存布局和计算循环，只将 A/B 的全局内存读取改为显式 `float4`。共享内存读写和 C 写回仍使用标量。
仅当实际地址满足 16 字节对齐且行内至少剩余 4 个元素时才使用向量读取，其余情况逐元素读取并对越界部分补零。N/K 不能被 4 整除时，部分行会走标量路径；4092 与 4096 均能被 4 整除。
该实验不包含文章 Kernel 6 的 As 转置优化，也不保证比其他版本更快。

## 编译运行

需要 CUDA Toolkit、兼容的主机 C++ 编译器和 NVIDIA GPU，无需链接 cuBLAS。
在此目录下编译，保留同目录的公共头文件：

```sh
nvcc -O3 -std=c++14 kernel5.cu -o kernel5
./kernel5
nvcc -O3 -std=c++14 kernel5_vectorize.cu -o kernel5_vectorize
./kernel5_vectorize 4092 4092 4092
nvcc -O3 -std=c++14 kernel5_float4.cu -o kernel5_float4
./kernel5_float4 4092 4092 4092 10 50 5
```

Windows 可使用 `-o kernel5.exe`，再运行 `.\kernel5.exe`。

```sh
# 自定义 M N K，以及预热次数、每轮重复次数、轮数
./kernel5 257 511 129 10 50 5
# 快速检查与短计时
./kernel5 128 128 128 2 3 2
```

参数必须为 1～16384 的整数；大矩阵的实际上限仍受可用显存和内存限制。
朴素 kernel 的默认完整测试可能耗时较长，可先指定小尺寸。

## 正确性与计时

- 全部 9 个实现使用同一流程。固定随机种子 20261031，A、B、初始 C 为 [-1,1] 随机数据。
- 小尺寸 `(1,1,1)`、`(17,33,9)`、`(65,67,17)`、`(129,131,15)` 全量对照 CPU double 累加参考，覆盖非方阵和边界分块。
- 每个尺寸分别验证 `(alpha,beta)=(0.5,0.75)` 和 `(1,0)`，每次恢复初始 C。
- 性能尺寸默认是 128、256、512、1024、2048、4096、4092 的方阵；各尺寸在计时前校验 64 个确定性采样位置（含首尾）。这是抽样检查，不是大矩阵的全量正确性证明。
- 误差条件：`abs(actual-reference) <= 1e-3 + 1e-3*abs(reference)`，拒绝非有限输出。
- 正式计时前预热 10 次并同步；每轮使用 CUDA Event 包住 50 次连续启动，共 5 轮。
- 输出各轮单次平均耗时的中位数、最小值、最大值，以及按中位数计算的 GFLOPS。
- 计时使用 alpha=1、beta=0，避免重复启动累积 C；分配、传输、正确性检查均排除在计时之外。
- 连续重复使用同一批数据，结果反映这种缓存复用场景；未清空缓存或锁定 GPU 时钟。比较版本时应使用相同 GPU、编译参数和输入尺寸。

流程参考 [Si Boehm 的 SGEMM 测试入口](https://github.com/siboehm/SGEMM_CUDA/blob/master/sgemm.cu)，这里使用 CPU 参考，并增加显式预热、多轮统计和非整齐尺寸检查。

## float4 对照与指令检查（H100）

```sh
for name in kernel5 kernel5_vectorize kernel5_float4; do
    nvcc -O3 -std=c++14 -arch=sm_90 "${name}.cu" -o "$name" || break
    "./$name" 4092 4092 4092 10 50 5 || break
done

# 同时测试完整分块，和触发部分标量回退的非对齐行宽
./kernel5_float4 4096 4096 4096 10 50 5
./kernel5_float4 257 511 129 10 50 5

# 输出 SASS，查看全局内存加载是否使用 LDG...128 指令
cuobjdump --dump-sass kernel5_float4 > kernel5_float4.sass
grep -E 'LDG.*128' kernel5_float4.sass
```

`sm_90` 针对 H100，其他 GPU 请使用对应架构。显式向量路径和标量回退会同时存在于二进制中，需结合目标 kernel 的反汇编确认；运行更快与否以相同条件下的实际计时为准。
