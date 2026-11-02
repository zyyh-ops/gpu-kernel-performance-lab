# 后续验证与归档步骤

更新日期：2026-11-02。Double Buffer GPU 集成测试 4/4 无跳过、大尺寸五档 10/10 正确性和五条性能记录已经齐备；环境与限定尺寸 racecheck 已归档。**目前主要剩归档。**

## 1. 补存实际实验源码

基础提交和未提交修改已记录于 [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md)。若 GPU 机器自实验后未修改源码，可补录当时保留的文件；否则不要把今天的文件直接标成当时版本。

在 GPU 项目根目录执行（Linux/Bash）：

```bash
set -euo pipefail
archive_dir=$(mktemp -d results/double_buffer_source_XXXXXXXX)
date -u > "$archive_dir/archive_time.txt"
git rev-parse HEAD > "$archive_dir/base_commit.txt"
git status --short > "$archive_dir/git_status.txt"
git diff HEAD --binary > "$archive_dir/working_tree.patch"
mkdir -p "$archive_dir/source/cuda/optimization"
cp Makefile "$archive_dir/source/"
cp cuda/optimization/{double_buffer.cu,double_buffer_kernels.cuh,gemm_validation.h} \
  "$archive_dir/source/cuda/optimization/"
sha256sum Makefile cuda/optimization/{double_buffer.cu,double_buffer_kernels.cuh,gemm_validation.h} \
  > "$archive_dir/source_sha256.txt"
# 仅在该程序确认为实验使用的二进制时，补存其散列。
sha256sum build/double_buffer > "$archive_dir/binary_sha256.txt"
```

`git diff` 不包含未跟踪的新文件，因此需要单独保存完整源码。另附说明将快照关联到本次结果和补录日期。若原构建日志仍在，应直接归档；新编译的日志只证明新构建，不能倒填旧运行。

## 2. 同步公开材料与最终提交

- 保持本轮 JSON/日志原样，引用当前 32 KiB 约 **1.21×** 的结果。
- 收录新代码、测试、环境/racecheck/集成记录与文档，核对最终 Git 差异。
- 运行最终提交的远端 CI，并明确许可证。离线 gate 与 GPU 实测结论分开表述。
- 早期 Matmul autotune/SASS 的命令、形状、版本来源仍可从可靠记录补录；缺失就标明未知。

## 3. 后续代码变更时复测

Double Buffer 的构建、GPU 集成与同配置 benchmark 命令见[完整流程](../double-buffer-validation.md)。该流程现在用于复现/回归，不再是首次验收待办。

Matmul 源码变更后可在新日志路径运行：

```bash
set -euo pipefail
run_dir=$(mktemp -d results/regression_XXXXXXXX)
RUN_LARGE=1 RUN_AUTOTUNE=1 python -m unittest discover -s tests -p 'test_matmul.py' -v \
  2>&1 | tee "$run_dir/matmul_full.log"
```

更广泛实验见[验证队列](../gpu_validation_queue.md)，均为可选加深项。不要覆盖现有实验文件制造看似同一次运行的记录。
