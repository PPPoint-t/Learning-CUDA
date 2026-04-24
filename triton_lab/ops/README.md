# Triton Ops

这里放你后续自己写的 Triton kernel 实现。

建议等完成 `Triton-Puzzles` 之后，再按本仓库的学习顺序逐步补：

1. `vector_add`
2. `silu`
3. `softmax`
4. `rmsnorm`
5. `gemm`
6. `attention`

建议每个算子先只保留一个最小可运行版本，确认你已经把 Triton 的索引和 mask 写明白了，再考虑自动调参、融合和更复杂的版本。
