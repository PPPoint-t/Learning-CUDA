# TopK

`topk` 会把学习平台重新接回归档作业中的 `kthLargest`，但也会把设计空间进一步打开。

建议的探索方向：

- `naive_sort_like`
- `block_select`
- `quickselect`
- `heap_based`

核心学习点：

- 局部有序和完全排序的区别
- 局部候选生成与合并
- partition 带来的额外内存访问开销

继续补全时，建议重点观察：

- `warp_topk`
  小 `k` 情况下，能否把候选维护在寄存器或单个 warp 内部
- `block_select`
  一个 block 如何生成局部候选，再交给下一阶段合并
- `radix_or_bucket_style`
  当值域特征合适时，是否能避开比较排序式实现
