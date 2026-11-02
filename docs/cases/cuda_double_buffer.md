# CUDA GEMM：资源受限条件下的 cp.async 双缓冲

更新日期：2026-11-02。当前主结果为[带正确性验证的 N=4096 JSON](../../results/summary/double_buffer_4096.json)，辅以[集成测试](../../results/summary/double_buffer_integration.log)、[旧边界尺寸 smoke](../../results/logs/gpu_smoke_5090.log)、[memcheck](../../results/logs/double_buffer_memcheck_5090.log)和 [racecheck](../../results/logs/double_buffer_racecheck_5090.log)。

## 实验问题与实现

对连续 FP32 方阵做 TILE=16 的 GEMM，在计算 tile t 时预取 tile t+1，以异步复制隐藏部分加载延迟。额外预留共享内存不参与矩阵计算，用于构造资源压力；因此不能直接代表真实模型或所有 GEMM 的收益。

[Kernel](../../cuda/optimization/double_buffer_kernels.cuh)使用一块 16B 对齐的动态共享内存，显式划分 [A0][A1][B0][B1]。N 能被 16 整除时走 16B `cp.async`；其余正 N 走 4B mask/zero-fill，Single 也处理边界。`wait_group 0` 后使用 `__syncthreads()` 完成交接。基础共享内存分别为 Single 2 KiB、Double 4 KiB，再加同一额外预留量。

## 当前 RTX 5090 结果

N=4096，FP32，16B 快路径；5 次预热、每样本 20 次 launch、5 个样本，A/B 顺序交替。每档配置先分别对照 cuBLAS 验证两种实现，再使用 CUDA events 计时；参考计算、传输和比较不计入耗时。

| 额外共享内存 | Single 理论 occupancy | Double 理论 occupancy | Single GFLOP/s | Double GFLOP/s | 加速比 |
|---:|---:|---:|---:|---:|---:|
| 0 KiB | 100.0% | 100.0% | 10115.24 | 10435.49 | 1.032× |
| 16 KiB | 83.3% | 66.7% | 10042.15 | 10305.02 | 1.026× |
| 24 KiB | 50.0% | 50.0% | 7870.63 | 8604.46 | 1.093× |
| 32 KiB | 33.3% | 33.3% | 6836.51 | 8248.01 | 1.206× |
| 44 KiB | 33.3% | 33.3% | 6834.60 | 8245.90 | 1.206× |

32 KiB 行为 **1.206×（约 1.21×，吞吐增加约 20.6%）**，Single / Double 延迟分别为 20.1037 / 16.6633 ms。此前约 1.29× 属于另一轮测量，不再作为本次已验证配置的数值。当前 JSON 在工作区中已被新结果替换；旧内容可从 Git 历史追溯，不能将两次运行拼成同一条曲线。

在本次资源压力扫描中，低理论驻留容量下相对收益更明显。occupancy 是 API 理论值；相同 occupancy 不表示所有资源条件相同，也不是运行时硬件计数。JSON 保存聚合延迟，没有逐样本时钟/温度和延迟序列，不据此确定不同运行间差异的原因。

## 正确性与检查范围

- 新 JSON：`schema_version=2`、`gpu_correctness_and_benchmark_passed`。N=4096 × 0/16/24/32/44 KiB × Single/Double，共 **10 条 PASS**，对应 **5 条性能记录**。
- 两种实现分别对照 `cublas_fp32_pedantic`；每条检查全部 16,777,216 个元素。容差为 `0.005 + 0.001*abs(reference)`，最大绝对误差约 `3.03e-5`，最大容差归一化误差约 `0.004964`；失配数及非有限值数均为 0。
- 输入模式由 JSON 明确记录，本结论只适用于该模式和已测配置，不代表所有输入都经过验证。
- 集成测试 **4/4 通过，无跳过**：3 项 GPU 集成测试（参考路径/边界/预留量、benchmark 同配置检查及输出防覆盖、无效资源配置失败）与 1 项主机数值测试。
- 早期 smoke 覆盖 13 个尺寸 × 2 实现，共 26 条 PASS，CPU FP64 参考、零额外预留量，覆盖 16B 和 4B 路径。这与新大尺寸 cuBLAS 验证是不同记录。
- memcheck：N=1/17/33/65，0 errors。racecheck：同组尺寸，0 errors、0 warnings。Sanitizer 日志没有记录额外预留量和完整构建来源，不能外推为 N=4096 五档都已通过 sanitizer，也不自动归到本次新二进制。

## 复现与来源

完整命令和源码/环境归档见[验证流程](../double-buffer-validation.md)。直接 `--mode bench` 也会先验证同配置；`--json` 拒绝覆盖已有文件。Makefile 可通过 `DOUBLE_BUFFER_JSON` 指定新路径。

已记录基础提交与未提交修改状态，完整源码快照/构建关联仍待补齐，见[源码来源记录](../release/SOURCE_PROVENANCE.md)。本次核心补测已完成，其他形状、随机输入、更多 sanitizer 路径可选，不作为当前结果成立的追加要求。
