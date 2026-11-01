# 后续验证与归档步骤

更新日期：2026-11-01。四组 benchmark、GPU smoke、Double Buffer memcheck 和 Matmul SASS 已归档。本页仅列当前缺项及进一步分析，不再要求仅为补文档重复已有四组性能实验。

## 1. 本地发布检查已通过

`release_check.py` 与 `offline_audit.py` 已移除旧历史 JSON 依赖。本轮 `python scripts/release_check.py` 退出码为 0，包含 15 项 CPU 模型测试。该项不再阻塞候选版本；最终提交仍需跑同一入口的 GitHub CI，下面的 GPU 验收缺项继续保留。

## 2. Matmul 完整补测已通过

已归档[完整补测日志](../../results/logs/matmul_full_5090.log)：5/5 通过、无跳过，耗时 2.138 秒。大尺寸与 autotune 两项缺口已解除。以下保留为后续源码变更时的复现命令，新日志使用不同文件名：

```bash
set -o pipefail
mkdir -p results/logs
RUN_LARGE=1 RUN_AUTOTUNE=1 python -m unittest discover -s tests -p 'test_matmul.py' -v 2>&1 | tee results/logs/matmul_full_5090_rerun.log
```

检查退出码及结尾，应为 5 项执行且无 skipped；若失败，保留原始输出并调查，不能只看出现过 PASS。现有 spill benchmark 与独立 autotune 日志不能代替此项状态。

## 3. 补录完整运行上下文

```bash
{
  date -u
  git rev-parse HEAD
  git status --short
  python --version
  python -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__)"
  uname -a
  c++ --version
  nvidia-smi
  nvcc --version
  compute-sanitizer --version
} > results/logs/environment_5090.txt 2>&1
```

检查每项命令的输出，缺失工具应如实标记。若需为新提交建立完整构建—测试证据，可强制重编译并启用全部 Matmul 用例：

```bash
set -o pipefail
RUN_LARGE=1 RUN_AUTOTUNE=1 make -B gpu-smoke CUDA_ARCH=sm_120 2>&1 | tee results/logs/gpu_smoke_full_5090.log
```

新环境输出不能倒推为旧测量发生时的环境。对旧产物只有能确认来源时才补关联，否则保留未知并在下次运行时完整记录。

## 4. 补录独立 autotune 与 SASS 来源

```bash
set -o pipefail
python scripts/autotune_matmul.py --M 4096 --N 4096 --K 4096 2>&1 | tee results/logs/matmul_autotune_4096_rerun.log
python scripts/inspect_matmul.py --M 4096 --N 4096 --K 4096 --BM 128 --BN 256 --BK 32 --warps 4 --stages 3 --save-dir results/raw/matmul_rerun 2>&1 | tee results/logs/matmul_inspect_w4_rerun.log
python scripts/inspect_matmul.py --M 4096 --N 4096 --K 4096 --BM 128 --BN 256 --BK 32 --warps 8 --stages 3 --save-dir results/raw/matmul_rerun 2>&1 | tee results/logs/matmul_inspect_w8_rerun.log
```

同时保存完整命令与源码版本。现有 SASS 已足以支撑“静态 LDL/STL 不同”的观察；这里的目的在于补全可复现来源，不是把已有文件重新标为缺失。若旧生成命令能够可靠恢复，也可直接记录而无需重复运行。

## 5. 可选的额外检查

memcheck 已有 N=1/17/33/65 零错误日志；下一步可以补 racecheck：

```bash
set -o pipefail
compute-sanitizer --tool racecheck --error-exitcode 1 ./build/double_buffer --mode test --test-sizes 1,17,33,65 2>&1 | tee results/logs/double_buffer_racecheck_5090.log
```

工具不支持或未安装应记录限制，不算通过。更多参数扫描见[验证队列](../gpu_validation_queue.md)。

以上日志名为新文件；再次使用同一命令前应保留此前结果。`results/raw/` 默认不入 Git，需要公开的汇编应检查后选取归档；不要覆盖旧结果来制造看似连续的同一次实验。
