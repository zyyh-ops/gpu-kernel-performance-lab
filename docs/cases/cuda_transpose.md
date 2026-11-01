# CUDA Transpose：全局内存合并访问与共享内存重排

更新日期：2026-11-01。当前已有 RTX 5090 的[测试日志](../../results/logs/gpu_smoke_5090.log)和 [N=8192 测量 JSON](../../results/summary/transpose_8192.json)，不再处于“未编译、未上 GPU”的状态。

## 工作负载与实现

输入是连续 FP32 方阵，输出写入另一块内存。TILE_DIM=32、BLOCK_ROWS=8，每个 block 256 线程。

- Naive：线程沿输入行连续读取，输出转置位置导致跨步写入。
- Shared：先合并读取到共享内存，经过 block 同步后重排线程访问，使输出写入连续。
- Padded：共享 tile 从 [32][32] 改为 [32][33]，改变列访问的 bank 映射。
- Copy：只进行复制，作为流式读写参考，不是语义等价的转置实现。

源码位于 [matrix_transpose.cu](../../cuda/memory/matrix_transpose.cu) 与 [transpose_kernels.cuh](../../cuda/memory/transpose_kernels.cuh)。

## 当前性能

| 版本 | 延迟中位数 | 有效带宽 |
|---|---:|---:|
| Naive | 0.650996 ms | 824.69 GB/s |
| Shared | 0.369149 ms | 1454.35 GB/s |
| Padded | 0.366836 ms | 1463.52 GB/s |
| Copy 参考 | 0.350753 ms | 1530.63 GB/s |

Shared 相对 Naive 为 **1.76×**；Padded 相对 Shared 的有效带宽仅高 **0.63%**，与本轮样本波动相近，尚不足以宣称稳定的额外收益。Padded 达到 Copy 参考有效带宽的约 95.6%，这不表示两者具有相同功能。

计时使用 5 次预热、每个样本 50 次 launch、共 5 个样本并轮换版本顺序。有效带宽为 `2*N*N*sizeof(float)/latency`；N=8192 时每个矩阵 256 MiB。该指标不等于实测 DRAM 流量。

## 正确性证据

smoke 日志覆盖 N=1、7、31、32、33、47、63、65、129，每个尺寸对 Naive/Shared/Padded/Copy 都记录 PASS，共 36 条。benchmark 程序还会先校验实际基准尺寸，N=8192 JSON 记录 `gpu_correctness_and_benchmark_passed`。

CPU 模型检查索引映射与 padding 的地址关系，不能单独验证 GPU 同步或越界行为。现有性能数据也不能直接量化 bank conflict 的独立延迟。

## 复现

以下为 RTX 5090 的构建与测试命令：

```bash
make cuda-transpose-test CUDA_ARCH=sm_120
make cuda-transpose-bench CUDA_ARCH=sm_120
```

如需保留原结果，可直接指定新的输出名：

```bash
./build/matrix_transpose --mode bench --n 8192 --warmup 5 --repeat 50 --samples 5 --json results/summary/transpose_8192_rerun.json
```

若更换架构或需要确认重编译，给 make 增加 `-B`。后续应通过跨会话交替重复观察 Shared/Padded 小差异，不能从单轮结果断言 padding 无效或稳定更快。
