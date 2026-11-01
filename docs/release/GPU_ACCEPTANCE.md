# GPU 与发布验收清单

更新日期：2026-11-01。当前 `v0.1.0-rc2-gpu-evidence` 仍为候选版本。勾选项只代表已有归档证据；未勾选项不隐含失败，也不计为通过。

## A：正确性证据

- [x] Transpose：9 个尺寸×4 个版本，36 条 PASS；N=8192 benchmark 另有正确性状态。
- [x] Double Buffer：13 个尺寸×2 个版本，26 条 PASS，覆盖 16B/4B 路径。
- [x] Double Buffer memcheck：N=1/17/33/65，日志零错误。
- [x] Paged KV：8 项 GPU 测试通过。
- [x] Matmul：smoke 3 项测试通过，4096³ spill benchmark 两个配置 PASS。
- [x] Matmul 独立 autotune：有 PASS 与选择配置日志。
- [x] Matmul 完整测试：新增[补测日志](../../results/logs/matmul_full_5090.log)，5/5 通过、无跳过，包含大尺寸与 autotune（2.138 秒）。
- [ ] 构建追溯：保存 nvcc 编译命令/输出，关联运行时源码提交及未提交差异。
- [ ] 完整环境：补 Python、系统、宿主编译器、驱动包版本和工具版本。
- [ ] 独立 autotune/SASS：补实际形状、命令与生成环境，避免仅靠文件名推断。

证据见 [GPU smoke](../../results/logs/gpu_smoke_5090.log)、[memcheck](../../results/logs/double_buffer_memcheck_5090.log)和 [autotune](../../results/logs/matmul_autotune_5090.log)。日志缺少版本关联不应通过猜测补齐。

## B：核心性能与编译证据

- [x] Matmul 固定 BM128/BN256/BK32/stages3 的 w4/w8 对照，延迟/资源/正确性齐备。
- [x] Matmul 两份 sm_120 SASS 已归档，LDL/STL 静态计数复核为 403/0。
- [x] KV 顺序/随机同负载对照，含正确性与计时元数据。
- [x] Transpose Naive/Shared/Padded 与 Copy 参考已记录。
- [x] Double Buffer N=4096、5 档额外共享预留的性能已记录。

Double Buffer 的勾选仅表示性能归档完成，原始 JSON 状态仍为 `gpu_measured_not_automatically_validated`。测试模式以小尺寸和零额外预留量验证，不代表大尺寸各组合逐项验算。

## C：公开提交前检查

- [x] 将四个案例、复现说明和审查文档更新为中文当前状态。
- [x] 公开数值与测量文件核对；区分有效带宽、理论 occupancy、编译指标和静态指令数。
- [x] 本次本地 15 项 CPU 模型测试通过。
- [x] 已移除历史 JSON 依赖；本轮完整离线发布检查退出码为 0。
- [ ] 最终 Git 差异检查：包含被文档引用的证据，排除旧离线日志及无关产物。
- [ ] 完成最终提交对应的 GitHub CI，不能用旧 PASS 日志代替。
- [ ] 明确代码来源与许可选择；当前没有 LICENSE 文件。

## D：后续加深验证

- [ ] Double Buffer racecheck；不能由 memcheck 零错误推定通过。
- [ ] 大尺寸与各额外共享预留配置的适当正确性覆盖。
- [ ] Matmul stages/BK/tile、KV block-f/warps/scale 和 KV PTX/SASS。
- [ ] Transpose Shared/Padded 跨会话重复测量。

D 类是继续完善实验的方向，不必为了当前四组已限定范围的观察全部执行。正式验收状态保存在 [RELEASE_STATUS.json](RELEASE_STATUS.json)，具体补测命令见[后续步骤](NEXT_GPU_CHECKLIST.md)。
