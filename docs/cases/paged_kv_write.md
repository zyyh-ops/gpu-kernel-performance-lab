# Triton Paged KV Write：地址映射与访存局部性

更新日期：2026-11-01。当前已归档 RTX 5090 的[局部性基准](../../results/summary/kv_locality_20261101_070449.json)及 [8 项 GPU 测试通过日志](../../results/logs/gpu_smoke_5090.log)。

## 功能与输入契约

该 kernel 只负责将 K/V 写入物理缓存槽。连续 FP16 输入为 `[num_tokens, n_features]`，缓存为 `[num_blocks, page_size, n_features]`。

```text
token
  → slot_mapping[token]
  → physical_block = slot // page_size
  → slot_in_block  = slot % page_size
  → cache[physical_block, slot_in_block, feature]
```

每个 Triton program 处理一个 token 的一个 feature chunk，使用二维网格；尾部 feature 使用 mask。槽位必须唯一、在合法范围内；mapping 为同设备、连续 int32 张量。

公开 launcher 默认检查槽位范围和唯一性，检查包含 GPU→CPU 同步。benchmark 在计时外完成一次验证，再对同一有效 mapping 使用 `check_slots=False`。重复槽位、-1 跳过标记、任意 strides、共享前缀与引用计数不在接口契约内。

## 当前测量

固定 32768 tokens、512 features、PAGE_SIZE=16、BLOCK_F=512、warps=4，使用相同 kernel 对比两种物理槽位顺序。

| 槽位顺序 | 延迟中位数 | 有效带宽 | 正确性 |
|---|---:|---:|---|
| 顺序 | 0.09109 ms | 1473.52 GB/s | PASS |
| 随机且唯一 | 0.10109 ms | 1327.71 GB/s | PASS |

随机映射的有效带宽低 **9.90%**，延迟高 **10.98%**。5 轮外层重复交替执行顺序/随机版本。带宽按 K/V 各一次读取和写入的标称有效字节计算，排除 mapping 元数据等额外流量，不是硬件测得的 DRAM 吞吐量。

每个 program 内的特征片段仍连续，随机物理槽位主要改变 program 之间的访问关系。现有数据不能唯一确定 L2、TLB 或 DRAM 行局部性中的哪一项造成差异。

## 验证覆盖与边界

GPU 测试覆盖 feature 尾部、页边界、稀疏槽位、多个 feature program、线性写入对照、随机唯一映射、非法槽位拒绝和非连续输入拒绝；CPU 模型另检验地址代数。

当前没有已归档的新版 KV PTX/SASS、BLOCK_F/warps/scale 性能扫描结果。因此不再把旧版编译观察描述为当前产物，也不宣称随机映射总会损失固定比例。

该原语没有 attention 计算、物理块分配、请求调度或生命周期管理，不应描述为完整 PagedAttention。

## 复现

```bash
python -m unittest discover -s tests -p 'test_paged_kv.py' -v
python benchmarks/bench_paged_kv.py --experiment locality
python benchmarks/bench_paged_kv.py --experiment compare --block-f 128
python benchmarks/bench_paged_kv.py --experiment block-f
python benchmarks/bench_paged_kv.py --experiment warps
python benchmarks/bench_paged_kv.py --experiment scale
python scripts/inspect_paged_kv.py --kind both --block-f 512 --show ptx --save-dir results/raw
```

首个 locality 实验已有结果，其余命令是进一步分析入口，不表示这些扫描已通过验收。实现约束详见[工程审查](../paged_kv_review.md)。
