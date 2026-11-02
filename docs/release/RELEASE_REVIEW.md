# 当前发布审查

更新日期：2026-11-02。版本标识保留 `v0.1.0-rc2-gpu-evidence`。同步GPU 补测、环境与 Git 信息，并核对 JSON、日志和派生摘要。


## 核查结果

| 检查 | 当前结论 |
|---|---|
| 四个核心实验 | 源码、测量和对应测试证据已存在；各自范围见[证据综述](RTX5090_GPU_RESULTS.md) |
| Matmul 完整测试 | 5/5 通过、无跳过；4096³ spill 两组 PASS |
| Double Buffer 大尺寸验证 | N=4096、五档预留量、两个实现独立对照 cuBLAS，10/10 PASS |
| Double Buffer 性能 | 同配置五条性能记录；32 KiB 为约 1.21× |
| Double Buffer 集成测试 | 4/4 通过，无跳过：3 项 GPU 集成测试与 1 项主机测试 |
| Sanitizer | 已有 N=1/17/33/65 memcheck 零错误、racecheck 零错误/警告；范围与新大尺寸测试不同 |
| 环境 | Python、系统、宿主编译器、驱动、NVCC、PyTorch、Triton、Sanitizer 已归档 |
| 源码来源 | 基础提交 `3eb445b1bd83719205a4e8ea81d093e51eed236e` 与未提交修改；完整快照/差异及构建日志仍待归档 |
| 离线检查 | 本次本地 release_check 通过：15 项 CPU 映射、1 项主机数值通过，3 项 GPU 集成跳过；用户另报告 GPU 机器上的离线检查通过。独立 GPU 集成日志为另一运行，4/4 无跳过 |
| GitHub CI / 许可 | 最终提交对应的远端 CI 尚未确认；当前未提供 LICENSE |

## 已关闭的缺项

旧的“尚无 racecheck”“环境版本未完整记录”“N=4096 benchmark 不携带正确性”不再适用于当前已归档证据。
新 [Double Buffer JSON](../../results/summary/double_buffer_4096.json)为 schema v2，保存参考类型、命令、误差、容差、逐配置 PASS 与性能。
新 [集成日志](../../results/summary/double_buffer_integration.log)没有 skipped。
原有离线检查的历史文件依赖、Matmul 可选测试跳过问题此前已解决。

## 剩余事项

1. 补存本次实验使用的完整源码快照或差异（含新增文件）、实际构建命令/输出和二进制散列。
2. 早期 Matmul 独立 autotune/SASS 的形状、生成命令与来源仍需补录，不能用新环境倒推全部旧记录。
3. 整理最终 Git 差异、核对最终提交的远端 CI，并明确代码许可。

## 数值与文件处理

当前 Double Buffer 主文件已替换为新补测结果。本次文档同步不改写原始 JSON 或日志，派生摘要按当前文件重新计算散列和指标。
机器状态见 [RELEASE_STATUS.json](RELEASE_STATUS.json)，正式清单见 [GPU_ACCEPTANCE.md](GPU_ACCEPTANCE.md)。
