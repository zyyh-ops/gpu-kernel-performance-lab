# RTX 5090 实验结果与证据综述

审查日期：2026-11-01。当前版本标识为 `v0.1.0-rc2-gpu-evidence`，GPU 为 NVIDIA GeForce RTX 5090（sm_120）。本页依据工作区已有的四份原始测量 JSON、四份测试日志和两份 SASS 更新；本次审查没有独立重跑 GPU。

结论：四组核心性能实验及部分扩展检查已有可公开证据；正式验收仍有未完成项。Matmul 新增完整补测 5/5 通过、无跳过；旧 smoke 的两项跳过记录仍保留，完整补测并未重新执行所有 CUDA/KV 测试。

## 核心性能结果

| 实验与负载 | 基线 | 对照 | 结果 |
|---|---:|---:|---|
| FP32 Transpose，8192² | Naive 824.69 GB/s | Shared 1454.35 GB/s | **1.76×** |
| FP32 Transpose，8192² | Shared 1454.35 GB/s | Padded 1463.52 GB/s | **+0.63%**，尚不足以支持稳定增益 |
| FP16 Matmul，4096³，BM128/BN256/BK32/stages3 | w4 55.68 TFLOP/s | w8 184.76 TFLOP/s | **3.32×**，n_spills 178→0 |
| FP16 Paged KV，32768 tokens×512 features，Page16 | 顺序 1473.52 GB/s | 随机 1327.71 GB/s | 随机带宽 **低 9.90%**，延迟高 10.98% |
| FP32 Double Buffer GEMM，4096²，额外共享内存 32 KiB | Single 6751.35 GFLOP/s | Double 8704.09 GFLOP/s | **1.29×**，双方理论 occupancy 33.3% |

GB/s 为标称有用字节数除以时间，不是 DRAM 硬件计数。所有加速比均在同一实验内比较，不表示超过 PyTorch/cuBLAS/vLLM。Transpose 的 Copy 参考为 1530.63 GB/s，仅用作流式访问参照。

## 证据文件与覆盖范围

| 文件 | 已确认内容 | 边界 |
|---|---|---|
| [Matmul JSON](../../results/summary/matmul_spill_20261101_070422.json) | w4/w8 两组正确性 PASS；形状、配置、资源和延迟齐备 | 不覆盖所有参数扫描 |
| [KV JSON](../../results/summary/kv_locality_20261101_070449.json) | 顺序/随机两组 PASS；5 轮交替计时 | 没有 cache/TLB 硬件计数器 |
| [Transpose JSON](../../results/summary/transpose_8192.json) | N=8192，状态为正确性与 benchmark 通过 | padding 小差异需重复检验 |
| [Double Buffer JSON](../../results/summary/double_buffer_4096.json) | N=4096，16B 路径，5 档共享预留扫描 | 原状态为 gpu_measured_not_automatically_validated，不自动携带 PASS |
| [GPU smoke](../../results/logs/gpu_smoke_5090.log) | Transpose 36 条 PASS；Double Buffer 26 条 PASS；Matmul 3 项通过、2 项跳过；KV 8 项通过 | 未附 nvcc 构建输出、源码版本及完整环境 |
| [Matmul 完整补测](../../results/logs/matmul_full_5090.log) | 5/5 通过、无跳过，含 4096³ 大尺寸和 256³ autotune，耗时 2.138 秒 | 日志未附源码版本与完整环境 |
| [Double Buffer memcheck](../../results/logs/double_buffer_memcheck_5090.log) | N=1/17/33/65 两种实现 PASS，ERROR SUMMARY: 0 errors | 无完整调用命令及工具版本；不证明 racecheck 通过 |
| [Matmul autotune](../../results/logs/matmul_autotune_5090.log) | 独立运行 PASS，选中 BM64/BN128/BK32、warps4、stages3 | 未附形状与完整命令；完整单元测试另有补测日志 |
| [w4 SASS](../../results/assembly/matmul_BM128_BN256_BK32_w4_s3.sass)、[w8 SASS](../../results/assembly/matmul_BM128_BN256_BK32_w8_s3.sass) | 文件目标 sm_120；静态 LDL/STL 为 403/0 | 文件名不编码 M/N/K，需补生成命令与版本关联 |

早期 Matmul smoke 跳过 `test_large_4096` 和 `test_autotuned_path`；新增完整补测中两项均显示 ok。当前 5 项 Matmul 单元测试均有执行通过的证据；旧 smoke 文件保持原样。

Double Buffer 测试覆盖 13 个尺寸：1/7/15/16/17/31/32/33/47/48/65/129/256，同时涉及 4B mask 与 16B 对齐路径。大尺寸和全部额外共享预留组合未在这些小尺寸日志中逐项校验。

## 测量方法与解释

Matmul/KV 记录 PyTorch `2.6.0a0+ecf3bae40a.nv25.01`、Triton 3.1.0、CUDA 12.8。两者用 `triton.testing.do_bench`，预热 25 ms、测量预算 100 ms；Matmul 外层重复 3 次，KV 5 次。Matmul 按配置依次计时，KV 交替顺序。

CUDA 记录 nvcc 12.8、Runtime API 12080、Driver API 支持版本 13030；最后一项不是驱动包版本。Transpose 为 5 次预热、50 次 launch/样本、5 个样本；Double Buffer 为 5/20/5。Double Buffer JSON 只保存聚合延迟。

Matmul 的寄存器、spill 与 SASS 差异支持资源压力解释，但静态 LDL/STL 403→0 不是运行时事务计数。KV 随机映射的性能下降也不能唯一归因于 L2/TLB/DRAM 行局部性。

## Double Buffer 共享内存扫描

| 额外共享内存 | Single 理论 occupancy | Double 理论 occupancy | Single GFLOP/s | Double GFLOP/s | 加速比 |
|---:|---:|---:|---:|---:|---:|
| 0 KiB | 100.0% | 100.0% | 10288.46 | 10633.33 | 1.034× |
| 16 KiB | 83.3% | 66.7% | 10116.73 | 10524.15 | 1.040× |
| 24 KiB | 50.0% | 50.0% | 7913.60 | 8760.00 | 1.107× |
| 32 KiB | 33.3% | 33.3% | 6751.35 | 8704.09 | 1.289× |
| 44 KiB | 33.3% | 33.3% | 6751.72 | 8701.20 | 1.289× |

人工预留共享内存导致的低理论驻留容量下，相对收益更明显，不能推广为所有实际 GEMM 都有约 29% 提升。

## 本次审查执行的核对

- 四份原始 JSON 均可解析，与[派生摘要](../../results/summary/computed_rtx5090_summary.json)记录的 SHA-256 一致。
- 按延迟、标称字节数和运算量核对公开数值，差异在显示舍入范围内。
- 重新统计 SASS 的 LDL/STL、HMMA/MMA、LDSM，分别为 w4 的 403/128/36 和 w8 的 0/64/24，与[计数摘要](../../results/assembly/sass_instruction_counts.txt)一致。
- 本轮完整离线发布检查退出码为 0，包含 15 项 CPU 映射测试；此前历史 JSON 依赖已移除。

派生摘要中的旧文字说明仅针对最初四份 JSON，不代表后续日志尚未归档。原始文件保持原样，跨文件验收状态由本页和[机器状态](RELEASE_STATUS.json)说明。

## 当前剩余事项

Matmul 无跳过测试已补齐；后续补完整环境、源码/构建关联，并完成最终提交的 CI 与发布整理；具体见[验收清单](GPU_ACCEPTANCE.md)。racecheck 和更多扫描属于后续加深验证。

旧 RTX 4090 的历史数据目前不在工作区中，本页不继续引用旧数值作为可追溯证据，也不构造跨 GPU 的受控加速比。
