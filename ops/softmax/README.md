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
