# GPU Kernel Performance Lab

新增：[Double Buffer 同配置验证与补测](docs/double-buffer-validation.md)。
该流程已获得 GPU 补测记录：N=4096 五档预留量 × 两实现共 **10 条正确性 PASS**，集成测试 **4/4 通过、无跳过**；下文 Double Buffer 数据已同步为本轮结果。

**基于 CUDA C++ 与 Triton 的 GPU Kernel 实现、正确性验证和性能分析实验项目**

当前为 `v0.1.0-rc2-gpu-evidence` 候选版本（2026-11-02 更新）。四组 GPU 测量已归档；Matmul 完整补测 5/5 通过，Double Buffer 新增同配置 cuBLAS 验证和 GPU 集成记录。环境及基础提交/未提交状态已收到，完整源码快照与构建关联仍待补齐。文档入口见[中文文档导航与 GitHub 收录建议](docs/README.md)，完整状态见[发布审查](docs/release/RELEASE_REVIEW.md)。

本项目围绕 GPU 性能工程中的几个典型问题展开：Global Memory 访问模式、Shared Memory 数据重排、Tensor Core GEMM 的寄存器压力、Paged KV Cache 的物理地址映射，以及异步数据搬运与计算重叠。

项目不是生产级算子库，也不以单纯追求峰值数字为目标。每个核心案例尽量形成一条完整的工程证据链：

**问题建模 → Kernel 实现 → Correctness 验证 → 控制变量 Benchmark → 编译器 / 资源分析 → 结果与限制说明。**

目前包含四个重点案例：CUDA Matrix Transpose、CUDA `cp.async` Double Buffering、Triton Tensor-Core Matmul 和 Triton Paged KV Write。

---

## 1. RTX 5090 实验结果

以下结果均来自 **NVIDIA GeForce RTX 5090（Compute Capability 12.0，`sm_120`）**。表中加速比仅表示同一实验内、相同工作负载下的实现对照，不代表相对 PyTorch 或其他框架的加速比；唯一的框架对照是表末的 cuBLAS 一行，按同 shape、同 dtype、同计时口径单独测量。

| 案例 | 实验条件 | 对照结果 | 观察 |
|---|---|---|---|
| CUDA Transpose | FP32，8192 × 8192 | Naive：824.69 GB/s；Shared：1454.35 GB/s | Shared 相比 Naive 提升约 **1.76×** |
| Triton Matmul | FP16 输入、FP32 累加，4096³；BM=128、BN=256、BK=32、Stages=3 | 4 warps：55.68 TFLOP/s；8 warps：184.76 TFLOP/s | 8-warps 配置约 **3.32×**；`n_spills` 从 178 降至 0 |
| Triton Paged KV Write | FP16，32768 Tokens × 512 Features，Page Size=16，BLOCK_F=512 | Sequential：1473.52 GB/s；Random：1327.71 GB/s | Random 映射的有效带宽低约 **9.90%** |
| CUDA Double Buffering | FP32 GEMM，N=4096，TILE=16，额外预留 Shared Memory=32 KiB | Single：6836.51 GFLOP/s；Double：8248.01 GFLOP/s | 理论 Occupancy 均为 33.3%，Double Buffer 约 **1.21×** |
| Triton vs cuBLAS | 4096³，FP16 输入 / FP32 累加，同一 `do_bench` 口径 | Triton：184.78 TFLOP/s；cuBLAS：186.41 TFLOP/s | Triton 达到 cuBLAS 约 **99%**；2048³ 以上输出与 cuBLAS 逐位相同；1024³ 时 Triton 仅 44%（cuBLAS 用 split-K 填满 SM） |

### 指标口径与解读

- **Effective Bandwidth**：按名义有效读写字节数除以测量耗时计算，不等同于硬件性能计数器测得的 DRAM 实际流量。
- **TFLOP/s / GFLOP/s**：按约定的运算量除以 Kernel 耗时计算，是派生吞吐指标，不是硬件计数器直接读数。
- **cuBLAS 对照口径**：参考值取 cuBLAS 在 `allow_fp16_reduced_precision_reduction=False`（FP32 累加）下的输出，与 Triton kernel 的累加口径一致。若改用默认的 FP16 归约，在小 shape 上 split-K 的部分和会引入约 1–2 ulp 误差，1024³ 时会出现约 0.05% 元素失配，因此不把它作为参考真值。
- **未归一化到峰值**：项目不把 TFLOP/s 折算成“峰值百分比”。RTX 5090 的 FP16（FP32 累加）参考时钟峰值为约 209.7 TFLOP/s（21760 核 × 4 × 2.41 GHz），而本机实测 SM 时钟更高（8192³ 下约 2.66 GHz，对应约 231.6 TFLOP/s），按参考峰值归一化会得出超过 100% 的误导性数字。跨实现比较一律用同 shape 的吞吐比值。
- **Transpose Padding**：Padded 版本为 1463.52 GB/s，相比 Shared 版本的 1454.35 GB/s 高约 **0.63%**。这一差异较小，不将其作为稳定、普遍的额外加速结论。
- **Transpose Copy Baseline**：本次为 1530.63 effective GB/s，仅作为连续读写参考，不是语义等价的 Transpose 实现。
- **Double Buffering Occupancy**：通过额外动态 Shared Memory 预留构造资源约束；表中的 Occupancy 是理论 Occupancy，不是运行时硬件计数器观测值。约 1.21× 的结果仅适用于该实验配置，不应推广为所有 GEMM 工作负载的收益。
- **Paged KV Locality**：结果说明物理 Slot 的跨 Program 排列会影响该工作负载的性能，但当前没有硬件计数器证据将差异唯一归因于 L2、TLB 或 DRAM 分区。

原始结果文件：

- [CUDA Transpose Benchmark](results/summary/transpose_8192.json)
- [CUDA Double Buffering Benchmark](results/summary/double_buffer_4096.json)
- [Triton Matmul Register Spill Benchmark](results/summary/matmul_spill_20261101_070422.json)
- [Triton Paged KV Locality Benchmark](results/summary/kv_locality_20261101_070449.json)
- [cuBLAS 对照 Benchmark](results/summary/cublas_vs_triton_20261101_122628.json)
- [Matmul SASS 指令统计](results/assembly/sass_instruction_counts.txt)

---

## 2. 核心案例

### 2.1 CUDA Matrix Transpose：Global Memory Coalescing 与 Shared Memory

实现并比较以下版本：

- **Naive Transpose**：全局内存读取连续，但转置后的写入存在跨列步长访问。
- **Shared Memory Transpose**：先将数据以合并访问方式写入 Shared Memory，再通过 Tile 内索引重排改善全局写入访问模式。
- **Shared Memory + Padding**：将 Shared Tile 的列维度从 32 调整为 33，改变 Bank 映射，缓解转置访问中的典型 Bank Conflict。
- **Copy Baseline**：提供连续读写带宽参考，不作为等价的 Transpose 算法。

本案例重点分析：修复全局内存访问模式带来的收益，与进一步消除 Shared Memory Bank Conflict 的增量收益之间有什么区别。

详细说明：[CUDA Transpose Case Study](docs/cases/cuda_transpose.md)

### 2.2 CUDA Double Buffering：`cp.async` 与 Ping-Pong Shared Memory

实现 Single Buffer 与基于 `cp.async` 的 Double Buffer GEMM，对比异步数据搬运与计算重叠在不同资源条件下的效果。

核心设计：

- 使用一块连续的动态 Shared Memory，并显式划分 A/B 矩阵的 Ping-Pong Buffer。
- 通过 `cp.async`、`commit_group`、`wait_group` 和 Block 级同步组织数据加载与计算。
- 对齐尺寸使用 16-byte 异步拷贝快速路径。
- 非 16 整除尺寸使用带边界检查和 Zero-Fill 的 4-byte 异步拷贝路径，并对输出写入进行边界 Mask。
- 通过额外 Shared Memory 预留改变理论 Occupancy，观察 Single / Double Buffer 的相对表现。

早期 RTX 5090 smoke 中，13 种尺寸 × 两实现共 **26 项通过**。本次另补 N=4096、0/16/24/32/44 KiB 五档预留量，两实现分别对照 cuBLAS FP32，**10 条 PASS**，每条完整检查所有输出元素。独立集成测试 **4/4 通过、无跳过**。已归档 N=1/17/33/65 的 Memcheck 零错误和 Racecheck 零错误/警告；Sanitizer 范围不等同于大尺寸五档。

本案例说明 Double Buffering 的收益取决于可被隐藏的访存延迟和资源约束，并非启用异步拷贝就一定更快。

详细说明：[CUDA Double Buffering Case Study](docs/cases/cuda_double_buffer.md)

### 2.3 Triton Tensor-Core Matmul：Register Pressure 与 Spill

实现 FP16 输入、FP32 累加的 Triton GEMM，并围绕 Tile、Warp、Stage 等参数开展控制变量实验。

固定 `BLOCK_M=128`、`BLOCK_N=256`、`BLOCK_K=32`、`num_stages=3`，比较 `num_warps=4/8`：

- 4 warps：55.68 TFLOP/s，`n_regs=255`，`n_spills=178`。
- 8 warps：184.76 TFLOP/s，`n_regs=210`，`n_spills=0`。

除 Benchmark 元数据外，还保存了两个配置的 SASS。静态反汇编中，4-warps 版本出现 212 条 `LDL` 与 191 条 `STL` 指令；8-warps 版本未出现这两类指令。该证据与 Spill 指标和性能变化相互印证。

注意：SASS 中的指令出现次数是静态指令计数，不是动态执行次数或实际内存事务数量。

此外，独立 Autotune 日志记录 Correctness 通过，该次运行选中的配置为：

```text
BLOCK_M=64
BLOCK_N=128
BLOCK_K=32
num_warps=4
num_ctas=1
num_stages=3
maxnreg=None
```

该日志未记录形状和完整命令，也未提供与手工固定配置的配对性能比较，因此不据此宣称它对所有尺寸最优或更快，256³ Autotune 单元测试的通过证据见另行归档的完整补测日志。

详细说明：[Triton Matmul Spill Case Study](docs/cases/matmul_spill.md)

### 2.4 Triton Paged KV Write：Slot Mapping 与访存局部性

实现 KV Cache 写入 Kernel，将输入 Token 的 K/V 向量写入物理分页 Cache。

地址映射过程：

```text
token_id
   │
   ▼
slot_mapping[token_id]
   │
   ▼
physical_slot
   ├── physical_block = physical_slot // PAGE_SIZE
   └── physical_offset = physical_slot % PAGE_SIZE
   │
   ▼
cache[physical_block, physical_offset, feature_offset]
```

当前实现的范围是 **Paged KV Cache Write**，不是完整的 PagedAttention、Attention Kernel 或 KV Block Manager。

输入约束包括：

- K/V 输入为连续的二维 FP16 Tensor。
- Cache 为连续的三维 FP16 Tensor。
- `slot_mapping` 为同设备上的连续 `int32` Tensor。
- Slot 必须在有效物理容量范围内，且映射不能重复；重复 Slot 会导致写入冲突。
- 安全校验发生在 Benchmark 计时区间之外，避免将 GPU 到 CPU 的同步开销混入 Kernel 计时。

本案例对比 Sequential 与 Random 物理 Slot 排列，研究单个 Token 内 Feature 连续访问与跨 Program 物理地址局部性之间的关系。

详细说明：[Triton Paged KV Write Case Study](docs/cases/paged_kv_write.md)

---

## 3. Correctness 与验证状态

以下状态对应 RTX 5090、`sm_120` 的实际测试记录。

| 模块 | 验证内容 | 结果 |
|---|---|---|
| CUDA Transpose | 9 种尺寸 × 4 个实现（Naive / Shared / Padded / Copy） | **36 项 PASS** |
| CUDA Double Buffering | 13 种尺寸 × 2 个实现（Single / Double），覆盖对齐与非对齐路径 | **26 项 PASS** |
| CUDA Double Buffering | Compute Sanitizer Memcheck，N=1、17、33、65 | **0 errors** |
| CUDA Double Buffering | N=4096，五档预留量 × 两实现，独立 cuBLAS 对照 | **10 条 PASS，五条性能结果** |
| CUDA Double Buffering | 3 项 GPU 集成测试 + 1 项主机数值测试 | **4/4 PASS，无跳过** |
| CUDA Double Buffering | Racecheck，N=1、17、33、65 | **0 errors、0 warnings** |
| Triton Matmul | 4-warps / 8-warps 固定配置 Benchmark 中的 Correctness 检查 | **均 PASS** |
| Triton Matmul | 独立 Autotune 路径 Correctness | **PASS** |
| Triton Matmul | 完整单元测试，含大尺寸与 Autotune | **5/5 PASS，无跳过** |
| Triton Paged KV Write | GPU 单元测试 | **8 项 PASS** |
| CPU Reference Models | Slot Mapping、Transpose Mapping、Double Buffering 映射模型 | **15 项 PASS** |

补充说明：

- 旧 smoke 中 Matmul 为 3 项通过、2 项跳过；新增[完整补测日志](results/logs/matmul_full_5090.log)记录 5/5 通过、无跳过，耗时 2.138 秒，已覆盖 4096³ 大尺寸和 256³ Autotune 用例。
- Compute Sanitizer 的 0 errors 结论仅对应本次执行的 N=1、17、33、65 和当前测试路径，不代表所有输入尺寸、所有配置均经过 Sanitizer。
- CPU Reference Model 测试用于验证地址映射和索引语义，不能替代 GPU Kernel Correctness。

证据：[GPU Smoke](results/logs/gpu_smoke_5090.log)、[Double Buffer 集成](results/summary/double_buffer_integration.log)、[Racecheck](results/logs/double_buffer_racecheck_5090.log)、[源码来源](docs/release/SOURCE_PROVENANCE.md)。

---

## 4. 项目结构

```text
gpu-kernel-performance-lab/
├── cuda/
│   ├── basics/                 # CUDA 基础算子
│   ├── compute/                # GEMM、Warp Reduction
│   ├── memory/                 # Matrix Transpose、访存实验
│   └── optimization/            # Double Buffer、Roofline、ILP
├── triton/
│   ├── matmul.py                # Tensor-Core GEMM
│   ├── paged_kv_write.py        # Paged KV Cache Write
│   ├── softmax.py
│   ├── vector_add.py
│   └── requirements.txt
├── benchmarks/                  # Matmul、Paged KV Benchmark
├── tests/                       # GPU Correctness 与 CPU 参考模型测试
├── scripts/                     # Autotune、PTX/SASS 检查、Release Gate
├── docs/
│   ├── README.md                # 中文文档导航与收录建议
│   ├── cases/                   # 四个核心案例分析
│   ├── release/                 # Release Review 与 GPU 验收记录
│   ├── benchmark_protocol.md
│   └── environment.md
├── results/
│   ├── summary/                 # RTX 5090 结构化 Benchmark 结果
│   ├── assembly/                # Matmul SASS 反汇编
│   └── logs/                    # GPU Smoke、Memcheck、Autotune 日志
├── Makefile
└── README.md
```

---

## 5. 环境与运行

### 5.1 已记录的实验环境

| 项目 | 版本 / 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| Compute Capability | 12.0（`sm_120`） |
| CUDA Runtime | 12.8 |
| CUDA Driver API 支持版本 | 13030 |
| 驱动版本（NVIDIA-SMI / KMD） | 610.43.02 |
| Python / 系统内核 | 3.12.3 / Linux 5.15.0-187-generic |
| 宿主编译器 | GCC 13.3.0 |
| Compute Sanitizer | 2025.1.0.0 |
| NVCC | 12.8 |
| PyTorch | `2.6.0a0+ecf3bae40a.nv25.01` |
| Triton | 3.1.0 |

上述补录版本见[环境日志](results/logs/environment_5090.txt)，不倒推为每份早期产物的确定环境，也不代表所有平台都必须使用相同版本。复现实验时，应根据目标 GPU、驱动、CUDA Toolkit、PyTorch 和 Triton 的兼容关系配置环境。

`triton/requirements.txt` 当前列出 `torch` 和 `triton`，未锁定具体版本；它不是完整的环境锁文件。安装时请确保使用支持目标 CUDA 环境的 PyTorch 发行包。

### 5.2 CPU-only Release Gate

不需要 NVIDIA GPU、CUDA Toolkit 或 Triton 即可执行 CPU 参考模型检查：

```bash
python scripts/release_check.py
```

该命令检查结构、文档链接、候选状态、Python 语法、15 项 CPU 映射测试及新增主机数值测试。后者需要 g++/clang++；未设置 `DOUBLE_BUFFER_BINARY` 时三项 GPU 集成测试会跳过。仅在 GPU 机器执行该命令，不等于完成 GPU 验证；本轮 GPU 结论来自独立无跳过日志。

2026-11-01 本轮复查中，该命令退出码为 0，完整离线检查及 15 项 CPU 模型测试通过。脚本已移除对旧历史 JSON 的依赖；GPU 测量与日志证据另行核对，不属于该脚本的自动验证范围。

### 5.3 GPU 编译与 Correctness Smoke Test

RTX 5090 对应 `sm_120`：

```bash
make gpu-smoke CUDA_ARCH=sm_120
```

该入口会编译并测试 CUDA Transpose、CUDA Double Buffering，并运行 Triton Matmul 与 Paged KV 的 GPU 测试。

如果使用其他 GPU，请根据目标设备调整 `CUDA_ARCH`。Makefile 当前默认值为 `sm_89`，适用于 RTX 4090；不要将默认架构直接用于 RTX 5090。

### 5.4 Benchmark

先创建结果目录：

```bash
mkdir -p results/summary
```

运行四个核心实验：

```bash
# Triton Matmul：固定 Tile 下比较 4 / 8 Warps
python benchmarks/bench_matmul.py \
  --experiment spill \
  --output results/summary/matmul_spill.json

# Triton Paged KV：Sequential / Random Locality
python benchmarks/bench_paged_kv.py \
  --experiment locality \
  --output results/summary/kv_locality.json

# CUDA Transpose
make cuda-transpose-bench CUDA_ARCH=sm_120

# CUDA Double Buffering
# 每档配置先独立对照参考，再计时；已有输出文件会被拒绝，请指定新文件名。
make cuda-double-buffer-bench CUDA_ARCH=sm_120 DOUBLE_BUFFER_JSON=results/summary/double_buffer_next_run.json
```

### 5.5 Matmul Autotune 与 PTX/SASS 检查

独立运行 Autotune 并检查输出正确性：

```bash
python scripts/autotune_matmul.py
```

导出 4-warps 和 8-warps 配置的 PTX/SASS：

```bash
mkdir -p results/assembly

python scripts/inspect_matmul.py \
  --M 4096 --N 4096 --K 4096 \
  --BM 128 --BN 256 --BK 32 \
  --warps 4 --stages 3 \
  --show both \
  --save-dir results/assembly

python scripts/inspect_matmul.py \
  --M 4096 --N 4096 --K 4096 \
  --BM 128 --BN 256 --BK 32 \
  --warps 8 --stages 3 \
  --show both \
  --save-dir results/assembly
```

Benchmark 与编译器检查应分开记录：前者用于比较运行性能，后者用于分析编译资源和生成指令。

---

## 6. Benchmark 方法与结果解释

本项目遵循以下基本原则：

1. **Correctness 优先**：先验证输出，再进行性能测量。
2. **控制变量**：A/B 对照尽量固定输入形状、数据类型、Tile 和计时方法，只改变目标参数。
3. **编译与计时分离**：Triton JIT 编译和参考结果计算不计入 Kernel Benchmark。
4. **保留原始结果**：记录 GPU、软件版本、输入形状、配置、预热、重复次数和统计口径。
5. **区分有效吞吐与硬件计数器**：Effective Bandwidth、TFLOP/s 是按名义字节数或运算量计算的派生指标。
6. **不跨环境直接比较**：不同 GPU、驱动、编译器和 Kernel 版本的结果不能拼接成严格的跨 GPU 加速比。

更完整的实验规范见 [Benchmark Protocol](docs/benchmark_protocol.md)。

---

## 7. 项目范围与已知限制

- 本项目是 GPU Kernel 学习与性能分析实验室，不是生产级 CUDA 算子库，也不是完整 LLM Serving 系统。
- Triton Paged KV 模块只负责将 K/V Token 向量写入分页物理 Cache，不实现 Attention、Block 分配、引用计数或完整 KV Cache 生命周期管理。
- 当前性能数据对应指定 RTX 5090、指定输入形状与软件环境，不保证在其他 GPU 或工作负载上得到相同收益。
- Transpose 的 Padding 版本仅比 Shared 版本高约 0.63%，不将其描述为稳定的普遍加速。
- Paged KV 的 Random Slot 性能差异尚未通过硬件计数器定位到某个特定缓存或地址转换组件。
- Double Buffering 的理论 Occupancy 实验使用额外 Shared Memory 预留构造资源约束，不能直接推广到所有 GEMM。
- SASS 中 `LDL` / `STL` 的静态出现次数不能当作动态执行次数或实际内存事务数量。
- 另有 11 份教学示例源码已存在，尚未纳入四个核心实验的 GPU 验收；清单见 [inventory](docs/inventory.md)。当前工作区没有旧版 `docs/archive/` 源码归档。

---

## 8. 进一步阅读

- [中文文档导航与 GitHub 收录建议](docs/README.md)
- [CUDA Transpose 案例分析](docs/cases/cuda_transpose.md)
- [CUDA Double Buffering 案例分析](docs/cases/cuda_double_buffer.md)
- [Triton Matmul Register Spill 案例分析](docs/cases/matmul_spill.md)
- [Triton Paged KV Write 案例分析](docs/cases/paged_kv_write.md)
- [Benchmark Protocol](docs/benchmark_protocol.md)
- [Environment Notes](docs/environment.md)
- [GPU Smoke 日志](results/logs/gpu_smoke_5090.log)

---

## 9. License

当前仓库快照未包含 `LICENSE` 文件。添加并确认许可证之前，请不要将本项目描述为已采用 MIT、Apache-2.0 或其他开源许可证。
