# CUDA GEMM：资源受限条件下的 cp.async 双缓冲

更新日期：2026-11-01。当前证据包括 [N=4096 基准](../../results/summary/double_buffer_4096.json)、[边界尺寸正确性日志](../../results/logs/gpu_smoke_5090.log)和 [memcheck 日志](../../results/logs/double_buffer_memcheck_5090.log)。

## 实验问题

双缓冲在计算 tile t 时预取 tile t+1，用异步复制尝试隐藏加载延迟，也增加共享内存和同步开销。本实验对连续 FP32 方阵做 TILE=16 的 GEMM，给每个 block 额外预留共享内存，以控制理论驻留容量，比较 Single Buffer 与 Double Buffer。

额外预留空间不参与矩阵计算；因此这是受控的资源压力实验，不能直接代表真实模型或所有 GEMM 的收益。

## 关键实现

[当前 kernel](../../cuda/optimization/double_buffer_kernels.cuh)从一块 16B 对齐的动态共享内存中显式划分 [A0][A1][B0][B1]，避免依赖多个动态共享声明形成独立分配。

- N 能被 16 整除：使用每次 16B 的 `cp.async` 快路径。
- 其他正 N：每个元素使用 4B 异步复制，越界元素 zero-fill；无效源地址使用安全的分配基址。
- Single Buffer 同样对非整 tile 做 mask；两种实现都保护输出边界。
- `wait_group 0` 之后使用 `__syncthreads()`，完成整个 block 的数据交接。
- 基础共享内存分别为 Single 2 KiB、Double 4 KiB，再加相同的实验额外预留量。

流程为：预取首 tile → 等待并同步 → 预取下一 tile 同时计算当前 tile → 等待并同步 → 切换 ping/pong → 输出。

## 当前 RTX 5090 结果

N=4096，FP32，16B 快路径；5 次预热、每个样本 20 次 launch、5 个样本，A/B 顺序交替。

| 额外共享内存 | Single 理论 occupancy | Double 理论 occupancy | Single GFLOP/s | Double GFLOP/s | 加速比 |
|---:|---:|---:|---:|---:|---:|
| 0 KiB | 100.0% | 100.0% | 10288.46 | 10633.33 | 1.034× |
| 16 KiB | 83.3% | 66.7% | 10116.73 | 10524.15 | 1.040× |
| 24 KiB | 50.0% | 50.0% | 7913.60 | 8760.00 | 1.107× |
| 32 KiB | 33.3% | 33.3% | 6751.35 | 8704.09 | 1.289× |
| 44 KiB | 33.3% | 33.3% | 6751.72 | 8701.20 | 1.289× |

在本实验较低理论驻留容量下，双缓冲相对收益更明显；32 KiB 行约为 **1.29×**。occupancy 由 API 理论计算得出，没有运行时 occupancy 硬件计数。不能将“理论 occupancy 相同”解释成所有资源条件完全相同。

## 正确性与检查范围

- smoke 覆盖 N=1、7、15、16、17、31、32、33、47、48、65、129、256；两种实现各 13 条 PASS，共 26 条，包含对齐 16B 和不规则 4B 路径。
- 校验对照为 CPU FP64 累加，GPU FP32 允许误差 `5e-3 + 1e-3*abs(reference)`；当前测试以零额外预留量执行。
- 已归档 memcheck 覆盖 N=1、17、33、65，两种实现输出 PASS，末尾为 `ERROR SUMMARY: 0 errors`。
- 尚无 racecheck 结果；memcheck 零错误不能替代竞态检查。
- benchmark JSON 保留 `gpu_measured_not_automatically_validated` 原始状态，因为直接运行 bench 不自动做正确性验收。独立测试日志提供补充证据，不能倒改 JSON 状态。
- 小尺寸测试并未逐项验证 N=4096 及所有额外共享预留组合的输出；当前大尺寸性能记录应与这个限制一起阅读。

## 复现

```bash
make cuda-double-buffer-test CUDA_ARCH=sm_120
make cuda-double-buffer-bench CUDA_ARCH=sm_120
compute-sanitizer --tool memcheck ./build/double_buffer --mode test --test-sizes 1,17,33,65
```

Makefile 的 benchmark 目标依赖正确性目标；直接 `--mode bench` 不包含此前置条件。重跑会覆盖默认 JSON，应先留存旧结果。

后续可补 racecheck、更多不规则尺寸及 N=4097 的独立性能基线。N=4096 与 N=4097 的负载不同，不能把两者之差当作纯粹的向量化收益。
