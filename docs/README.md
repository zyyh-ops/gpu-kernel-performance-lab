# 文档导航与 GitHub 收录建议

更新日期：2026-11-02。本文按当前工作区的源码、测试日志、汇编和测量 JSON 审查文档，适合作为 GitHub 文档入口。

项目目前已有 RTX 5090 上四组核心实验的测量记录，以及部分独立正确性和内存检查证据。正式发布仍有待办，详见[发布审查](release/RELEASE_REVIEW.md)。本次文档审查没有重新运行 GPU 实验。

## 建议提交的文档

| 文档 | 用途与处理结果 |
|---|---|
| [实验与证据规范](benchmark_protocol.md) | 统一计时、对照、指标和证据归档规则，已改为中文 |
| [环境与复现](environment.md) | 更新实际记录的 RTX 5090 软件栈，区分依赖清单和版本锁定 |
| [源码清单](inventory.md) | 按实际文件更新，消除“源码尚未迁移”的旧描述 |
| [GPU 验证队列](gpu_validation_queue.md) | 区分已归档证据、部分覆盖和后续实验 |
| [Matmul 实现审查](matmul_review.md) | 保留接口、网格、精度和固定配置/自动调优分离的设计依据 |
| [Paged KV 实现审查](paged_kv_review.md) | 保留地址映射、输入契约和计时边界的设计依据 |
| [Matmul 案例](cases/matmul_spill.md) | 以当前 5090 数据和 SASS 为主，移除旧 4090 数值作为当前结论的表述 |
| [Paged KV 案例](cases/paged_kv_write.md) | 更新当前带宽和测试覆盖，保留因果解释的边界 |
| [Transpose 案例](cases/cuda_transpose.md) | 更新正确性、性能和 padding 差异 |
| [Double Buffer 案例](cases/cuda_double_buffer.md) | 更新大尺寸同配置 cuBLAS 验证、五档性能、GPU 集成及限定范围 racecheck |
| [Double Buffer 验证流程](double-buffer-validation.md) | 保留后续复现/回归和归档命令，首次验证已完成 |
| [源码来源记录](release/SOURCE_PROVENANCE.md) | 保存用户提供的基础提交/工作区状态，区分已收到信息与待归档快照 |
| [RTX 5090 实验证据](release/RTX5090_GPU_RESULTS.md) | 汇总当前结果，并链接到对应原始文件 |
| [发布审查](release/RELEASE_REVIEW.md) | 替换早期离线候选结论，记录当前发布阻碍 |
| [GPU 验收清单](release/GPU_ACCEPTANCE.md) | 按实际证据勾选，跳过的测试不计为通过 |
| [后续验证步骤](release/NEXT_GPU_CHECKLIST.md) | 只保留尚需补齐的验证和归档步骤 |
| [项目技术概览](release/PROJECT_PRESENTATION.md) | 将原简历/面试话术改为面向仓库读者的项目说明 |
| [机器可读发布状态](release/RELEASE_STATUS.json) | 保留稳定字段和状态标识，说明文字改为中文并同步证据 |

上述文档与状态文件适合随对应源码和证据一起提交。技术文档可公开，不等于已满足正式稳定版本的发布条件。

## 不建议随本次公开提交的文件

`release/OFFLINE_GATE_RERUN_20261101.txt` 是较早工作区的原始离线运行记录，包含“历史 JSON 存在、两份 Triton 源码未迁移”等旧状态，与当前目录不符。建议仅本地留档，不作为当前验收证据；本次未删除或改写它。

旧 `docs/archive/` 和 `results/historical/` 所指的材料在当前工作区缺失，因此公开文档不再把它们当作现有证据链接，也不据此发布旧 4090 的性能结论。

## 提交时需要一并包含的证据

- [测量 JSON](../results/summary/)：四组原始实验记录和派生计算摘要。
- [测试与环境日志](../results/logs/)：GPU smoke、Double Buffer memcheck/racecheck、环境及 Matmul 日志。
- [Double Buffer 集成日志](../results/summary/double_buffer_integration.log)：当前实际保存在 summary 目录，4/4 无跳过。
- [Matmul 汇编](../results/assembly/)：w4/w8 SASS 和静态指令计数。

`computed_rtx5090_summary.json` 已同步当前四份 JSON 的散列、Double Buffer 新数值和验证边界；跨文件日志及 SASS 状态见[证据综述](release/RTX5090_GPU_RESULTS.md)。本次只修改派生摘要和文档，不改写原始结果。

## 当前提交前问题

Double Buffer 同配置 GPU 验证已闭合；用户另报告 RTX 5090 机器上的 offline_audit/release_check 通过，具体 GPU 结论来自独立无跳过日志。环境与基础提交/未提交清单已收到，剩余为实际源码快照/差异、构建关联、早期 autotune/SASS 来源及最终提交的远端 CI/许可整理。无需仅为文档整理重复现有实验。
