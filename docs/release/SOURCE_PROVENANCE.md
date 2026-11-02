# 本次 GPU 补测的源码来源记录

记录日期：2026-11-02。以下 Git 信息由用户在对话中提供，表示 GPU 运行机器上报告的工作区状态；不是本次文档整理后重新采集的源码快照。

## 已确认的信息

- 基础提交：`3eb445b1bd83719205a4e8ea81d093e51eed236e`。
- 工作区存在未提交修改和新增文件，实际实验源码不能仅由基础提交唯一标识。
- 用户报告的系统：`Linux f74025d42834 5.15.0-187-generic #197-Ubuntu SMP Fri Jul 17 19:17:01 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux`。
- [环境日志](../../results/logs/environment_5090.txt)记录 Python、系统、编译器、GPU、驱动与工具版本。
- [Double Buffer JSON](../../results/summary/double_buffer_4096.json)记录完整运行参数、参考实现、cuBLAS 版本与验证/性能结果。
- JSON 的命令中输出名为 `results/summary/double_buffer_4096_validated.json`，当前归档名为 `results/summary/double_buffer_4096.json`。归档文件名不同不改变原始记录；本次不修改其中的命令。
- [集成日志](../../results/summary/double_buffer_integration.log)记录 4/4 通过、无跳过。

## 用户提供的 Git 状态

```text
 M Makefile
 M README.md
 M cuda/optimization/double_buffer.cu
 M docs/release/GPU_ACCEPTANCE.md
 M docs/release/NEXT_GPU_CHECKLIST.md
 M docs/release/RELEASE_REVIEW.md
 M results/summary/double_buffer_4096.json
 M scripts/offline_audit.py
 M scripts/release_check.py
?? cuda/optimization/gemm_validation.h
?? docs/double-buffer-validation.md
?? results/logs/double_buffer_racecheck_5090.log
?? results/logs/environment_5090.txt
?? results/summary/double_buffer_integration.log
?? tests/gemm_validation_host.cpp
?? tests/test_double_buffer_validation.py
```

## 仍待归档的关联材料

实际运行时的源码快照或差异（含未跟踪的新文件）、编译命令和输出、二进制散列尚未收到。
`git status` 仅列文件，不能恢复这些修改的内容；`git diff` 也不会自动包含未跟踪文件。
若 GPU 机器自实验后未改源码，可补存当时保留的文件并注明补录时间；否则应保留关联未知，不能把当前文件散列倒填成实验版本。

当前已确认的结果可按其记录范围使用；无需仅为文档整理重跑现有实验。
后续新实验采用[同配置验证与归档流程](../double-buffer-validation.md)。
