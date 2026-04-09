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
