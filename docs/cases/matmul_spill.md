# Triton Matmul：寄存器 spill 与固定配置性能差异

更新日期：2026-11-01。当前案例依据 RTX 5090 的[测量 JSON](../../results/summary/matmul_spill_20261101_070422.json)、[GPU 测试日志](../../results/logs/gpu_smoke_5090.log)和 [SASS 文件](../../results/assembly/)。本次文档审查只核对已有证据，没有重跑 GPU。

## 问题与控制变量

对于 C=A@B，扩大输出 tile 可以增加数据复用，也会增加累加器规模与寄存器压力。本实验固定 M=N=K=4096、BM=128、BN=256、BK=32、stages=3，只比较 warps=4 与 8。输入和输出为 FP16，累加为 FP32。

| warps | 延迟中位数 | 吞吐量 | 每线程寄存器指标 | n_spills | 共享内存 |
|---:|---:|---:|---:|---:|---:|
| 4 | 2.4685 ms | 55.68 TFLOP/s | 255 | 178 | 49152 B |
| 8 | 0.7439 ms | 184.76 TFLOP/s | 210 | 0 | 49152 B |

这组配置中 w8/w4 吞吐量之比为 **3.32×**，两组均记录 `correctness=PASS`。这是同一 kernel 的配置对照，不是相对 PyTorch 或 cuBLAS 的加速比。

## 编译与性能证据

对已保存的两份 SASS 按指令助记符重新计数，结果与[指令摘要](../../results/assembly/sass_instruction_counts.txt)一致：

| 配置 | LDL/STL | HMMA/MMA | LDSM |
|---|---:|---:|---:|
| w4 | 403 | 128 | 36 |
| w8 | 0 | 64 | 24 |

在相同 tile 下，w8 的寄存器指标降低、spill 指标归零，SASS 中也不再出现 LDL/STL。这些证据支持“较高的每线程资源压力及 spill 与 w4 的显著性能损失有关”的解释。

计数是静态汇编条目数，不是执行次数；`n_spills` 也不是 DRAM 事务数。现有结果未隔离 warp 数变化的全部作用，不能宣称 occupancy 翻倍或把全部差异精确归因于 spill。

## 正确性与自动调优

旧 smoke 中 Matmul 为 3 项通过、2 项跳过。新增[完整补测日志](../../results/logs/matmul_full_5090.log)记录 5 项全部通过、无跳过（2.138 秒），包含 4096³ 固定配置专项和 256³ autotune 专项。旧日志保留原样；当前测试覆盖以两份日志共同说明。4096³ spill benchmark 的 w4/w8 两组 PASS 仍为独立配置对照证据。

[独立 autotune 日志](../../results/logs/matmul_autotune_5090.log)记录 PASS，选中 BM64/BN128/BK32、warps4、stages3。日志未记录运行形状和完整命令，因此只能说“该次独立运行的选择”，不能把它当作所有尺寸的最优配置，更不能沿用旧文档的 BM128/BN256/warps8 为当前 autotune 赢家。

## 实现与复现

[实现](../../triton/matmul.py)将固定配置 JIT kernel 与 autotune 包装分离，网格按实际 tile 计算。接口限定同设备、连续的二维 FP16 张量；M/N/K 尾块使用 mask。

在已配置 GPU 环境的仓库根目录执行：

```bash
python benchmarks/bench_matmul.py --experiment spill
python scripts/inspect_matmul.py --M 4096 --N 4096 --K 4096 --BM 128 --BN 256 --BK 32 --warps 4 --stages 3 --save-dir results/raw
python scripts/inspect_matmul.py --M 4096 --N 4096 --K 4096 --BM 128 --BN 256 --BK 32 --warps 8 --stages 3 --save-dir results/raw
```

当前 Matmul 基准按配置顺序测量，每组外层重复 3 次，并未交替 w4/w8。stages、BK、tile 扫描尚无本轮已归档结果；见[后续队列](../gpu_validation_queue.md)。旧 4090 的 7.58× 不作为本案例当前结论。
