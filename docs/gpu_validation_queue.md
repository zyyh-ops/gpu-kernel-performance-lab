# GPU 验证与后续实验队列

更新日期：2026-11-01。状态依据当前归档文件；“已有证据”不表示本次文档审查重新运行了 GPU。正式验收见[验收清单](release/GPU_ACCEPTANCE.md)。

## 当前覆盖

| 项目 | 当前状态 | 证据或剩余动作 |
|---|---|---|
| Matmul 完整测试 | 已归档通过 | 完整补测 5/5 PASS、无跳过；4096³ benchmark 两组 PASS |
| Matmul w4/w8 spill 对照 | 已测量 | 55.68→184.76 TFLOP/s，配置和指标见[案例](cases/matmul_spill.md) |
| Matmul SASS | 已归档 | w4/w8 LDL/STL 静态计数 403/0；仍需补源码版本和完整生成命令 |
| Matmul 独立 autotune | 有 PASS 日志 | 已记录选中配置；未记录运行形状/命令；完整单元测试另有 5/5 PASS 日志 |
| Paged KV 正确性 | 已归档通过 | 8 项测试通过 |
| Paged KV 顺序/随机 | 已测量 | 两组 PASS，有效带宽 1473.52/1327.71 GB/s |
| Transpose 正确性/性能 | 已归档 | smoke 36 条 PASS，另有 N=8192 benchmark 正确性状态 |
| Double Buffer 两条复制路径 | 已归档通过 | 13 个尺寸×2 版本，含 16B 快路径与 4B 尾块路径 |
| Double Buffer memcheck | 已归档零错误 | N=1/17/33/65，不能代替 racecheck |
| Double Buffer occupancy 扫描 | 已测量 | N=4096、额外共享内存 0/16/24/32/44 KiB；JSON 自身不包含正确性 PASS |

详见[原始日志](../results/logs/)、[测量文件](../results/summary/)和[汇编](../results/assembly/)。

## 发布前应补齐

1. 离线检查已通过：历史 JSON 依赖已移除，本轮完整检查退出码为 0，包含 15 项 CPU 模型测试。最终提交后仍需确认远端 CI。
2. Matmul 完整补测已完成；仍需补录早期独立 autotune 运行的形状、命令。
3. 将 GPU 记录与源码版本、编译命令、完整环境关联。当前 smoke 没有保存 nvcc 编译输出。
4. 完成最终 Git 差异与 CI 检查，并明确许可选择。当前状态仍为有 GPU 证据的候选版本。

具体命令见[后续验证步骤](release/NEXT_GPU_CHECKLIST.md)。已经归档的四组性能结果无需仅为补文档而重复生成。

## 可选深入分析

| 方向 | 命令或动作 | 尚未支持的结论 |
|---|---|---|
| Matmul stages/BK/tile | `python benchmarks/bench_matmul.py --experiment stages`；另选 `block-k`、`tile` | 当前最优参数的全面搜索 |
| KV BLOCK_F/warps/scale | `python benchmarks/bench_paged_kv.py --experiment block-f`；另选 `warps`、`scale`、`compare` | 所有规模和配置的规律 |
| KV 汇编 | `python scripts/inspect_paged_kv.py --kind both --show ptx --save-dir results/raw` | 当前编译器下的指令级地址开销 |
| Double Buffer racecheck | 对 N=1/17/33/65 保存检查输出 | 已完成竞态检查 |
| Double Buffer 扩展形状/配置 | N=19/34/66；为实际大尺寸与预留组合增加适当正确性覆盖 | 所有 benchmark 组合均逐项验证 |
| Transpose 小差异 | Shared/Padded 跨会话配对复测 | padding 稳定提升 0.63% |
| Double Buffer 不规则性能 | 单独测 N=4097 | 与 N=4096 之差等于纯 4B/16B 路径成本 |

未执行或没有归档的项目继续保留为待办；不使用旧历史数字填充本轮空缺。
