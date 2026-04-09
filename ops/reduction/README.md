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
