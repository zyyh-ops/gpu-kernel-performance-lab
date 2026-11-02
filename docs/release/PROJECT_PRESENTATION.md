# 项目技术概览

GPU Kernel Performance Lab 是一个 CUDA/Triton GPU 优化实验仓库，通过四个可复现案例连接地址映射、访存布局、寄存器资源和异步流水线。项目提供独立正确性测试、CPU 参考模型、结构化性能结果与部分编译产物，定位为教学与工程实验。

## 四个实验回答什么问题

| 实验 | 工程问题 | RTX 5090 当前观察 |
|---|---|---|
| [Transpose](../cases/cuda_transpose.md) | 如何通过共享内存重排改善全局写入？ | 8192² FP32：Shared 相对 Naive 1.76× |
| [Matmul](../cases/matmul_spill.md) | tile/warp 分配如何影响寄存器 spill？ | 固定 4096³ FP16、BM128/BN256/BK32/stages3：w4→w8 为 3.32×，n_spills 178→0 |
| [Paged KV](../cases/paged_kv_write.md) | token→物理槽位顺序如何影响写入？ | 32768×512 FP16、Page16：随机映射有效带宽低 9.90% |
| [Double Buffer](../cases/cuda_double_buffer.md) | 资源受限时异步预取能带来多少收益？ | 4096² FP32、额外共享内存 32 KiB、双方理论 occupancy 33.3%：约 1.21×；五档 × 两实现均通过 cuBLAS 正确性对照 |

表内性能比均为各实验内部对照；Matmul 另有独立 cuBLAS 性能对照，见项目 README。Double Buffer 的 cuBLAS 在本轮只用作正确性参考，性能仍比较 Single 与 Double。有效带宽按有用字节数计算，不能描述为硬件 DRAM 测量值。

## 工程组织

`triton/` 与 `cuda/` 放置实现；`tests/` 包含 GPU 校验和 CPU 模型；`benchmarks/` 与 `scripts/` 分别承担测量、编译检查和诊断；`results/` 保存测量、日志及精选汇编。

Matmul 将固定配置实验与自动调优分离。Paged KV 显式约束槽位唯一性和布局。Transpose 区分共享内存重排与 padding。Double Buffer 显式划分 ping/pong 缓冲区，并为对齐和尾块尺寸提供不同异步复制路径。

## 与推理系统的关系

Paged KV 原语展示上层 token-to-slot 映射如何落到 GPU 写入，但不管理物理块生命周期，也不执行 attention。Matmul 案例展示计算路径中的 tile/warp/resource 选择。系统级调度、批处理与 kernel 局部优化应分别评估，不能把局部加速比直接迁移为端到端推理收益。

## 当前成熟度

四组性能记录已归档。Matmul 完整补测 5/5 通过；Double Buffer 新增 N=4096 五档 10 条正确性 PASS、集成 4/4 无跳过，以及限定尺寸 racecheck 零错误/警告。环境与基础提交状态已补录；实际源码快照/构建关联及最终发布整理仍待完成。覆盖范围见[证据综述](RTX5090_GPU_RESULTS.md)，仍以候选版本公开，不宣称生产可用。
