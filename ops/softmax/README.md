# Softmax

`softmax` 是平台里第一个对数值稳定性比较敏感的按行算子。

建议的版本演进：

- `naive`
- `shared_mem`
- `warp`
- `online`

核心学习点：

- 稳定的 max-subtract 写法
- 一行数据如何并行化
- max reduction 和 sum reduction 的组织方式
- 带宽开销和重复计算之间的取舍

继续推进时建议优先关注：

- `warp_per_row` 和 `block_per_row` 在不同 `cols` 下的取舍
- `vectorized_load`
  对按行读取的输入做向量化搬运
- `online_softmax`
  用在线更新的 `max` / `sum` 减少额外的一趟内存读写
- 以后与 `attention` 融合时，尽量避免物化完整的 score matrix
