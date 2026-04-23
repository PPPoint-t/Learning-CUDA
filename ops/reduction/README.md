# Reduction

`reduction` 是第一个真正需要你关注同步和 shared memory 的算子。

建议的版本演进：

- `naive`
- `shared_mem`
- `warp`
- `two_stage`

学习这个算子时，重点想清楚这些问题：

- shared memory 什么时候真的减少了全局内存访问？
- 输入长度超过一个 block 时，应该如何组织多阶段归约？
- warp-level reduction 什么时候能让实现更简单、性能更好？

进一步优化时，建议优先关注这些点：

- `multiple_elements_per_thread`
  每个线程先在寄存器里累加多个元素，减少 block 数和 shared memory 压力。
- `warp_shuffle_tail`
  在 block 内最后 32 个线程的收尾阶段用 `__shfl_down_sync`，减少同步。
- `vectorized_load`
  对齐满足时用 `float4` 等宽加载，提高带宽利用率。
- `two_stage_global_reduce`
  明确第一阶段输出中间结果、第二阶段继续归约的组织方式。
