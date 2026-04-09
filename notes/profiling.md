# Profiling 记录规范

每个新 kernel 版本都建议使用同一套检查清单。

## 先看正确性

- 先和 CPU reference 对比。
- 至少测试一个非常小的 case 和一个非平凡 case。
- 至少测试一个不是 block size 整数倍的输入长度。

## Benchmark 基本规范

- 计时前先 warm up。
- 至少测多次，避免单次波动。
- 同时打印平均时间和该算子对应的吞吐指标。
- 比较不同 variant 时保持输入 shape 不变。
- 注意：如果重复在同一块 buffer 上运行，cache 友好的 kernel 可能会表现得比真实 DRAM 带宽更“漂亮”。

## 观察什么

- 这个 kernel 更像是 memory-bound 还是 compute-bound？
- 全局内存加载是否 coalesced？
- shared memory 是否真的减少了重复读取？
- occupancy 会不会被寄存器或 shared memory 限制？
- 更复杂的版本是否真的比简单版本更快？

## 要记录什么

- launch 配置
- 测得的运行时间
- 你判断的主要瓶颈
- 新版本为什么有效，或者为什么失败
