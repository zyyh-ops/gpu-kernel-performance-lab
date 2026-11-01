# Matmul 实现审查

更新日期：2026-11-01。本页记录当前接口与工程设计，性能分析见 [Matmul 案例](cases/matmul_spill.md)。

## 实现结构

[triton/matmul.py](../triton/matmul.py)定义固定配置 kernel、自动调优包装和输入校验。固定配置路径显式传入 BM/BN/BK、warps、stages；autotune 通过独立包装选择配置，避免固定参数与自动选择冲突。

固定网格按调用配置计算，自动调优网格按实际元数据计算；均使用 `ceil(M/BM)*ceil(N/BN)` 个 program。原先基于固定小 tile 的网格假设不再适用。

正确性、性能测量、汇编检查和自动调优拆分到 `tests/`、`benchmarks/`、`scripts/`。基准保存形状、环境、配置、计时、资源指标和每组 PASS；检查脚本分别标记 PTX/SASS，并允许工具链无法提供某类汇编。

## 接口与数值契约

- 输入为同一 CUDA 设备上的非空、连续二维 FP16 张量，内维匹配。
- 内部 FP32 累加，输出 FP16；A/B 读取和 C 写入均保护尾块。
- 不支持 BF16/FP32 输入、任意 strides 或通用布局。
- 公开 `matmul_triton` 执行输入校验；底层 `launch_fixed` 用于受控实验，调用者应确保配置和输出有效。
- 固定配置与 `autotune=True` 不可同时指定；保留 7 个候选配置。

## 已归档的验证

旧 [GPU smoke](../results/logs/gpu_smoke_5090.log)记录 3 项通过、2 项跳过。新增[完整补测](../results/logs/matmul_full_5090.log)记录 5/5 通过、无跳过，耗时 2.138 秒，覆盖固定边界形状、大 tile/warp 配置、CPU 输入拒绝、4096³ 大尺寸与 256³ autotune。

[4096³ spill 基准](../results/summary/matmul_spill_20261101_070422.json)的 w4/w8 两组均 PASS。[独立 autotune](../results/logs/matmul_autotune_5090.log)也报告 PASS，选中 BM64/BN128/BK32、warps4、stages3，但日志缺少运行形状及命令；它与完整补测中的 256³ autotune 用例是不同运行，不用该配置填充完整补测的选择结果。

[两份 SASS](../results/assembly/)及资源数据支持固定配置下的 spill 分析；没有 Nsight 硬件计数器，不能声称测得缓存命中率或实际 occupancy。

## 待补内容

1. 完整补测已归档；后续变更源码时继续显式开启 `RUN_LARGE=1 RUN_AUTOTUNE=1` 验证。
2. 将构建/运行命令、形状、源码版本与 SASS/autotune 产物关联。
3. 后续按需归档 stages、BK 和 tile 扫描，并考虑交替配置顺序测量。

不再以旧 RTX 4090 的 22.86→173.34 TFLOP/s 作为当前结果；当前固定配置测量为 55.68→184.76 TFLOP/s，解释边界见案例页。
