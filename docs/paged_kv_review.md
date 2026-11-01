# Paged KV Write 实现审查

更新日期：2026-11-01。审查对象为[当前实现](../triton/paged_kv_write.py)，当前工作区没有可链接的旧版源码归档，因此本页以现有代码可验证的行为为准。

## 地址映射与网格

每个 program 负责 `(token, feature_chunk)`。先从 `slot_mapping[token]` 得到物理槽位，再用页大小求 block/slot，最后按缓存 strides 和 feature offset 写入。读取与写入均保持 feature 维连续，并用 mask 保护尾部。

线性写入对照与分页写入分开，便于研究地址计算和物理槽位顺序的影响；两者不是 attention 实现。

## 输入校验

`KVWriteConfig` 显式描述 page_size、block_f、num_warps。输入和缓存必须是同一 CUDA 设备的连续 FP16 张量，mapping 为连续 int32。

公开分页 launcher 默认执行实际槽位范围和唯一性检查，防止非法索引及重复写入。该检查会同步并将 mapping 复制到 CPU，不适合计入 kernel 延迟。benchmark 先验证固定 mapping，再关闭重复槽位检查；使用 `check_slots=False` 的调用者承担维持这一前提的责任。

不支持 -1 跳过槽位、重复槽位、任意 strides 或共享块所有权语义。

## 验证与测量

[GPU 日志](../results/logs/gpu_smoke_5090.log)中的 8 项 KV 测试均通过，覆盖边界、尾部、稀疏映射、线性对照、非法输入等；CPU 模型额外验证页地址代数，包括非 2 的幂页大小。

[当前 locality JSON](../results/summary/kv_locality_20261101_070449.json)对顺序/随机两组都记录 PASS，正确性参考使用 `index_copy_`。计时排除编译、参考计算及槽位校验，外层 5 轮交替顺序。有效带宽只统计 K/V 的标称读取和写入字节数。

## 分析边界

当前数据支持给定 workload 下顺序/随机槽位的性能差异，未提供新版 KV PTX/SASS 或 profiler 计数器，因此不将差异直接归因于某一种缓存、TLB 或 DRAM 行行为。

该项目没有实现 block 分配、请求调度、前缀共享或引用计数。完整结果及复现命令见 [Paged KV 案例](cases/paged_kv_write.md)；扩展扫描列入[验证队列](gpu_validation_queue.md)。
