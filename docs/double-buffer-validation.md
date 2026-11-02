# Double Buffer：同配置正确性验证与性能补测

本次改动只扩展验证、报告和构建流程，未修改计算 Kernel。
**新流程已获得 GPU 实测证据：集成测试 4/4 无跳过，N=4096 五档 × 两实现共 10 条 PASS 和 5 条性能记录。32 KiB 的当前加速比为约 1.21×。**

证据：[集成日志](../results/summary/double_buffer_integration.log)、[验证与性能 JSON](../results/summary/double_buffer_4096.json)。本页命令保留作后续复现/回归，不再表示首次运行待办。源码来源归档范围见[来源记录](release/SOURCE_PROVENANCE.md)。

## 行为

- `--mode test` 对 `--test-sizes` 与 `--extra-kib` 的每个组合分别验证 Single / Double。
- `--reference auto`（默认）：N ≤ 256 使用 CPU FP64 累加参考；更大尺寸使用 cuBLAS。
- `--reference cpu|cublas` 可强制选择。不要为 N=4096 强制 CPU：三重循环很慢。
- cuBLAS 使用 FP32 输入/输出和 `CUBLAS_COMPUTE_32F_PEDANTIC`，不使用 TF32 快速计算模式。
  行主序 C=A×B 通过列主序 Cᵀ=Bᵀ×Aᵀ 实现，调用时交换 A/B 指针。
  小尺寸显式选择 cuBLAS 时，还会与 CPU FP64 参考交叉验证。
- 每个 Kernel 启动前将输出填为 NaN，防止漏写位置复用参考值或另一个 Kernel 的结果。
- 每个元素要求有限且满足 `abs(got-ref) <= 0.005 + 0.001*abs(ref)`，沿用原测试容差。
  这是容差内一致性，不是逐位相等；报告包含所有元素的比较结果。
- `--mode bench` 对每个实际测量的 N / 额外预留量先验证两种实现，通过后才计时。
  参考计算、输出传输和比较均不计入 CUDA Event 时间；失败返回非零退出码。
- `--mode both` 先跑指定测试矩阵，然后执行上述带验证的 benchmark。
- `--json` 同时支持 test / bench / both，记录命令、环境、参考类型、容差、误差、
  非有限值和失配计数，以及已完成的性能结果。数值失败也保存报告，顶层状态为 `FAILED`。
  初始化设备前的错误（例如无 GPU、参数错误）只写终端并返回非零。
- 已存在的 JSON 路径会被拒绝，避免覆盖历史数据；每次运行使用新目录。

精度模式依据：[NVIDIA cuBLAS 12.8 文档](https://docs.nvidia.com/cuda/archive/12.8.2/cublas/index.html#cublascomputetype-t)。

## 最小 GPU 复测（Linux / Bash，RTX 5090）

在项目根目录执行。先保留当前修改，再将该源码版本上传/同步到 GPU 机器。
下面的命令自动生成新目录，并记录提交、未提交差异及实际源码散列。

```bash
set -euo pipefail
run_dir=$(mktemp -d results/double_buffer_run_XXXXXXXX)
echo "本次结果目录：$run_dir"
{
  date -u
  git rev-parse HEAD
  git status --short
  uname -a
  c++ --version
  python --version
  nvidia-smi
  nvcc --version
} > "$run_dir/environment.txt" 2>&1
git diff HEAD --binary > "$run_dir/working_tree.patch"
# 新增且未提交的文件不在 git diff 中，因此另外保存本次构建的完整源码。
mkdir -p "$run_dir/source/cuda/optimization"
cp Makefile "$run_dir/source/"
cp cuda/optimization/{double_buffer.cu,double_buffer_kernels.cuh,gemm_validation.h} \
  "$run_dir/source/cuda/optimization/"
sha256sum Makefile cuda/optimization/{double_buffer.cu,double_buffer_kernels.cuh,gemm_validation.h} \
  > "$run_dir/source_sha256.txt"
make -B build/double_buffer CUDA_ARCH=sm_120 2>&1 | tee "$run_dir/build.log"
sha256sum build/double_buffer > "$run_dir/binary_sha256.txt"

# 小尺寸交叉验证 cuBLAS 的布局和边界路径，以及 0/32 KiB 两档资源预留。
./build/double_buffer --mode test --test-sizes 16,17,33 \
  --extra-kib 0,32 --reference cublas --json "$run_dir/reference_check.json" \
  2>&1 | tee "$run_dir/reference_check.log"

# 简历所用配置：两种实现分别验证成功后才测量性能。
./build/double_buffer --mode bench --n 4096 --extra-kib 32 \
  --reference cublas --warmup 5 --repeat 20 --samples 5 \
  --json "$run_dir/benchmark_4096_32kib.json" 2>&1 | tee "$run_dir/benchmark.log"
```

预期：第二份 JSON 顶层为 `gpu_correctness_and_benchmark_passed`；`correctness`
包含 Single / Double 两条 N=4096、extra_shared_kib=32、status=PASS 的记录。
`results` 含该配置的新性能值。若失败，不沿用旧加速比宣称新版本通过。

完整五档扫描只需在上述 benchmark 命令中改为 `--extra-kib 0,16,24,32,44`，
并选择新的输出文件名。无需重复做原来所有学习实验。

## 自动检查

```bash
# 主机数值检查（需要 g++ 或 clang++）；没有显式指定 GPU 程序时 GPU 检查跳过。
python -m unittest discover -s tests -p 'test_double_buffer_validation.py' -v

# 真实 GPU 集成检查：小尺寸 CPU/cuBLAS 对照、边界、资源预留、自动选参考、
# 无效资源配置失败、已有 JSON 防覆盖。程序路径错/无 GPU 时应失败，不能算通过。
DOUBLE_BUFFER_BINARY=./build/double_buffer \
  python -m unittest discover -s tests -p 'test_double_buffer_validation.py' -v
```

上述小尺寸自动检查不能替代 N=4096 的实际补测。
本流程只覆盖记录的输入模式与配置，不表示所有输入、同步竞争或其他硬件均已验证。
