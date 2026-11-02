# 环境与复现说明

更新日期：2026-11-02。版本来自测量文件与新增[环境日志](../results/logs/environment_5090.txt)。本次未重新配置或运行 GPU；补录环境不自动证明每份早期产物的运行环境。

## 已记录的 RTX 5090 环境

| 项目 | 已归档值 | 来源与限制 |
|---|---|---|
| GPU / 计算能力 | NVIDIA GeForce RTX 5090 / 12.0（sm_120） | 四组基准记录 |
| PyTorch | 2.6.0a0+ecf3bae40a.nv25.01 | Matmul、KV JSON |
| Triton | 3.1.0 | Matmul、KV JSON |
| PyTorch CUDA | 12.8 | Matmul、KV JSON |
| nvcc | 12.8 | 两个 CUDA JSON |
| CUDA Runtime API 版本 | 12080 | 两个 CUDA JSON |
| CUDA Driver API 支持版本 | 13030 | 两个 CUDA JSON；不是 nvidia-smi 显示的驱动包版本 |
| Python | 3.12.3 | 环境日志 |
| 系统内核 | Linux 5.15.0-187-generic，x86_64 | 环境日志 |
| 宿主编译器 | GCC 13.3.0（Ubuntu 13.3.0-6ubuntu2~24.04） | 环境日志 |
| 驱动版本 | NVIDIA-SMI / KMD 610.43.02 | 环境日志，与 Driver API 支持版本区分 |
| NVCC 完整版本 | 12.8.61 | 环境日志 |
| Compute Sanitizer | 2025.1.0.0 | 环境日志，不自动绑定每份旧 Sanitizer 日志 |
| 基础提交 | `3eb445b1bd83719205a4e8ea81d093e51eed236e` | 用户提供；工作区有未提交修改 |
| 完整源码/构建关联 | 仍待快照/差异、编译输出及二进制散列 | 见[来源记录](release/SOURCE_PROVENANCE.md) |

[triton/requirements.txt](../triton/requirements.txt) 已存在，内容为 `torch`、`triton`，没有版本约束。它是依赖名称清单，不是锁文件。不要将上述测量环境当作可在任意机器直接安装的通用依赖组合。

## CPU 检查

CPU 映射测试只需 Python 3.10 或更新版本及标准库：

```bash
python -m unittest discover -s tests -p 'test_*mapping_cpu.py' -v
python scripts/release_check.py
```

本轮本地复查：完整离线发布检查退出码为 0，映射测试 15/15 通过；脚本已移除旧历史 JSON 依赖。详见[发布审查](release/RELEASE_REVIEW.md)。默认未设置 `DOUBLE_BUFFER_BINARY` 时 GPU 集成测试跳过；新增主机数值测试需 g++/clang++。独立 GPU 集成记录已 4/4 无跳过，不能与离线 gate 混为同一个结论。

## GPU 构建与测试

以下命令面向配置好 NVIDIA CUDA、PyTorch、Triton、Make 和 C++17 编译器的 Linux/Bash 环境。Double Buffer 的 `cp.async` 要求 sm_80 或更新架构；Makefile 默认 `sm_89`，RTX 5090 必须显式使用 `sm_120`。

```bash
make gpu-smoke CUDA_ARCH=sm_120
python benchmarks/bench_matmul.py --experiment spill
python benchmarks/bench_paged_kv.py --experiment locality
make cuda-transpose-bench CUDA_ARCH=sm_120
make cuda-double-buffer-bench CUDA_ARCH=sm_120
```

Transpose 默认输出仍需自行防覆盖；Double Buffer 已拒绝覆盖现有 JSON，应以 `DOUBLE_BUFFER_JSON=results/summary/double_buffer_next_run.json` 或 `--json` 指定新路径。更换 `CUDA_ARCH` 后，Make 不会仅因变量变化自动重编译已有目标；请用 `make -B gpu-smoke CUDA_ARCH=sm_120` 等方式强制重建。

默认 smoke 未设置开关时仍会跳过 Matmul 的两个可选用例；本轮显式启用后的[完整补测](../results/logs/matmul_full_5090.log)已 5/5 通过、无跳过。复现命令见[后续验证步骤](release/NEXT_GPU_CHECKLIST.md)。

## 补录环境

在实际运行 GPU 的机器上执行：

```bash
python --version
python -c "import torch, triton; print('torch', torch.__version__, 'torch cuda', torch.version.cuda, 'triton', triton.__version__, 'cuda available', torch.cuda.is_available())"
nvidia-smi
nvcc --version
git rev-parse HEAD
git status --short
```

将命令、退出状态、运行日期、环境输出与测量一同保存。若运行时存在未提交改动，还应记录差异，不能只靠提交号标识代码。公开前去除机器账户路径等无关信息。
