# 源码清单与验证范围

更新日期：2026-11-02。清单依据当前文件实际存在情况编写。源码存在、测试通过和性能得到验证是不同状态。

## 核心实验

| 源码 | 当前证据 | 入口 |
|---|---|---|
| `triton/matmul.py` | 固定配置 benchmark 两组 PASS；完整补测 5/5 PASS、无跳过；另有独立 autotune PASS 与 w4/w8 SASS | [Matmul 案例](cases/matmul_spill.md) |
| `triton/paged_kv_write.py` | GPU 8 项测试通过；顺序/随机映射 benchmark 两组 PASS | [Paged KV 案例](cases/paged_kv_write.md) |
| `cuda/memory/matrix_transpose.cu`、`transpose_kernels.cuh` | 9 个尺寸×4 个版本共 36 条 PASS；N=8192 基准含正确性状态 | [Transpose 案例](cases/cuda_transpose.md) |
| `cuda/optimization/double_buffer.cu`、`double_buffer_kernels.cuh` | 旧 smoke 26 条 PASS；N=1/17/33/65 memcheck/racecheck 零错误；新 N=4096 五档 × 两实现 10 条 PASS；集成 4/4 无跳过 | [Double Buffer 案例](cases/cuda_double_buffer.md) |

上述是仓库已归档证据的覆盖范围，尚未形成完整的源码提交—构建—运行关联，详见[验证综述](release/RTX5090_GPU_RESULTS.md)。

## 已存在的教学示例

| 类别 | 当前源码路径 | 验证范围 |
|---|---|---|
| CUDA 基础 | `cuda/basics/vector_add.cu` | 源码已存在，未纳入四个核心实验的验收 |
| CUDA 内存 | `cuda/memory/matrix_add.cu`、`memory_coalescing.cu`、`shared_memory.cu`、`vectorized_memory_access.cu` | 源码已存在，未归档独立验证报告 |
| CUDA 计算 | `cuda/compute/matrix_matmul.cu`、`warp_reduce_sum.cu` | 源码已存在，未归档独立验证报告 |
| CUDA 优化 | `cuda/optimization/saxpy_ilp.cu`、`roofline_bench.cu` | 源码已存在，未归档独立验证报告 |
| Triton 基础 | `triton/vector_add.py`、`triton/softmax.py` | 源码已存在，未纳入核心 smoke 测试 |

不再将这 11 份教学源码标记为“尚未迁移”，也不据其存在声称已通过 GPU 测试。

## 工程辅助材料

- `benchmarks/`：Matmul 参数扫描、Paged KV 对照实验。
- `scripts/`：汇编检查、autotune、数值诊断、CPU 映射模型和离线检查。
- `tests/`：Triton GPU 测试、CPU 映射模型，以及新增 Double Buffer 主机数值与可选 GPU 集成测试。
- [triton/requirements.txt](../triton/requirements.txt)：未锁版本的依赖清单。
- [results/summary](../results/summary/)、[results/logs](../results/logs/)、[results/assembly](../results/assembly/)：当前公开结论的证据。
- `docs/archive/` 和旧 `results/historical/` 材料在当前工作区缺失；不再宣称仓库保留了完整原始教学源码档案。
