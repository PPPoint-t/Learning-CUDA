# GEMM

`gemm` 是进入 tiling 和数据复用世界的关键算子。

建议的版本演进：

- `naive`
- `tiled`
- `vectorized`
- `double_buffered`

核心学习点：

- coalesced load
- shared memory tiling
- 算术强度
- occupancy 和寄存器压力之间的平衡
