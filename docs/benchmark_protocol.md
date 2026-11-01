# 基准测试与证据规范

本规范约束性能结论如何产生和归档。当前已归档实验见 [RTX 5090 实验证据](release/RTX5090_GPU_RESULTS.md)，不再使用“等待获得 GPU 后首次测试”的旧状态。

## 每次实验应记录什么

- GPU 型号、计算能力、PyTorch/Triton/CUDA/nvcc 版本；同时记录 Python、操作系统、驱动包版本和源码提交号。
- 输入形状、数据类型、连续性和输出语义；GEMM 的 BM/BN/BK、warps、stages，KV 的页大小和特征块大小。
- 预热、重复次数、样本聚合方式、计时器、同步和 stream 边界。
- 正确性参考、误差阈值、通过/失败/跳过情况。
- 每次样本及中位数；硬件允许时记录寄存器、spill、共享内存、PTX/SASS。
- 条件允许时记录功耗限制、温度、时钟和其他负载，减少跨会话小幅差异的误判。

以上是归档目标，不代表现有文件已包含全部字段。当前环境缺项见[环境说明](environment.md)。

## 对照与计时

先编译和校验，再计时。JIT、参考实现以及 KV 槽位范围/唯一性检查不计入 kernel 延迟。不同实现须使用相同输入形状、精度和输出定义。

优先在同一设备、同一会话交替执行 A/B。当前 KV 与两个 CUDA 基准轮换测试顺序；Matmul 脚本按配置依次完成各自重复测量，尚未交替 A/B，不能描述为已消除顺序偏差。

当前 Matmul/KV 使用 `triton.testing.do_bench`，记录预热 25 ms、测量预算 100 ms；Matmul 外层重复 3 次，KV 5 次。两个 CUDA 程序使用 CUDA events 测量一批 launch 后折算单次延迟。Double Buffer JSON 保存聚合结果，未保存每个原始延迟样本。

## 指标定义

| 指标 | 计算方式 | 解释边界 |
|---|---|---|
| GEMM TFLOP/s | `2*M*N*K / (latency_ms * 1e9)` | 标称运算量除以耗时，不是硬件计数器 |
| FP32 Transpose GB/s | `2*N*N*4 / (latency_ms * 1e6)` | 输入读取与输出写入的有效字节数 |
| FP16 KV Write GB/s | `4*num_tokens*n_features*2 / (latency_ms * 1e6)` | K/V 各一次读写；不含 mapping 和额外实际流量 |
| A→B 加速比 | `latency_A / latency_B` | 仅适用于相同工作负载 |
| 随机映射带宽损失 | `1 - BW_random/BW_sequential` | 与延迟增加百分比不同 |
| 理论 occupancy | CUDA occupancy API 计算的驻留容量 | 不等于运行时测得的活跃 warp 比例 |

`n_spills` 是编译资源指标；SASS 中 `LDL/STL` 的数量是静态指令条目数。二者都不能直接换算为运行时 DRAM 事务数。没有 profiler 证据时，不给出确定的缓存命中率、TLB 或 DRAM 行冲突归因。

## 文件归档与发布

- [results/summary](../results/summary/) 保存结构化测量及派生摘要；保留原始值、元数据和文件散列。
- [results/logs](../results/logs/) 保存精简的测试/检查输出。
- [results/assembly](../results/assembly/) 保存用于结论的精选汇编及计数。
- `results/raw/` 默认被 Git 忽略，适合临时 profiler 输出；需要公开的证据应检查内容后另行归档。
- 当前工作区没有旧历史数据目录。以后若引入旧数据，应注明设备、版本和来源，不能与本次结果混成一条性能曲线。
- 测量文件应绑定源码提交、构建命令和环境清单；当前这些关联信息不完整，文档保留限制。

日志中出现 `skipped` 不等于通过。某一形状的 benchmark PASS 也不能替代整个测试套件。CPU 参考模型只检查映射和数学约定，不验证 CUDA 指令、同步或 GPU 性能。
