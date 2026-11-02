# GPU 与发布验收清单

更新日期：2026-11-02。`v0.1.0-rc2-gpu-evidence` 仍为候选版本。勾选代表已有归档证据，不代表本次文档整理重新执行 GPU。

## A：已归档验证

- [x] Transpose：9 尺寸 × 4 版本，36 条 PASS；N=8192 benchmark 另有正确性状态。
- [x] Matmul：4096³ spill 两配置 PASS，完整补测 5/5 通过、无跳过。
- [x] Paged KV：8 项 GPU 测试通过，顺序/随机 benchmark 两组 PASS。
- [x] Double Buffer 边界：早期 13 尺寸 × 2 实现，26 条 PASS，覆盖 16B/4B 路径。
- [x] Double Buffer 大尺寸：N=4096、额外预留 0/16/24/32/44 KiB，两实现各自对照 cuBLAS，共 10 条 PASS。
- [x] Double Buffer 性能：上述五档各有验证后的性能记录；32 KiB 加速比为约 1.21×。
- [x] Double Buffer 集成测试：4/4 通过，无跳过，含 3 项 GPU 集成测试与 1 项主机数值测试。
- [x] Double Buffer memcheck：N=1/17/33/65，0 errors。
- [x] Double Buffer racecheck：N=1/17/33/65，0 errors、0 warnings；日志未绑定完整源码/预留量，不能外推到所有配置。
- [x] 环境日志：Python、系统、宿主编译器、驱动、GPU、CUDA、PyTorch/Triton、Sanitizer 版本已记录。
- [x] 基础提交与工作区状态：用户提供 `3eb445b1bd83719205a4e8ea81d093e51eed236e` 和未提交修改清单，已写入[来源记录](SOURCE_PROVENANCE.md)。

证据入口：[Double Buffer JSON](../../results/summary/double_buffer_4096.json)、[集成日志](../../results/summary/double_buffer_integration.log)、[racecheck](../../results/logs/double_buffer_racecheck_5090.log)、[环境](../../results/logs/environment_5090.txt)、[完整综述](RTX5090_GPU_RESULTS.md)。

## B：公开提交前整理

- [x] README、案例、概览及派生摘要同步本轮 Double Buffer 数字与状态。
- [x] 已有本地离线检查通过记录；GPU 机器上 offline_audit/release_check 通过。
- [x] 最终 Git 差异：包含新增代码、测试和证据，排除无关产物。
- [x] 最终提交对应的 GitHub CI。
- [ ] 明确代码来源与许可选择；当前没有 LICENSE。

offline/release 检查在 GPU 机器运行也不自动等于 GPU 测试；以具体测试项和 skipped 状态为准。本轮独立集成日志已明确无跳过。

## C：可选深入分析

- [ ] Double Buffer 更多输入模式、不规则尺寸、额外 sanitizer 路径与跨会话稳定性。
- [ ] Matmul stages/BK/tile、KV block-f/warps/scale 及 KV 汇编。
- [ ] Transpose Shared/Padded 小差异的跨会话配对复测。

这些是扩展研究方向，不要求为当前已验证结论全部完成。后续归档步骤见 [NEXT_GPU_CHECKLIST.md](NEXT_GPU_CHECKLIST.md)。
