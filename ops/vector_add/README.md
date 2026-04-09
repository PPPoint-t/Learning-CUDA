# Vector Add

`vector_add` 是整个学习平台里的第一个算子，因为它可以把最基础的 CUDA 知识单独拎出来练：

- kernel launch 配置
- 线程索引和数据索引映射
- 越界检查
- Host 和 Device 之间的数据搬运
- 正确性测试
- 第一份性能 benchmark

## 当前版本

- `naive`
  一个线程负责一个输出元素。
- `strided_loop`
  通过 grid-stride loop 让每个线程处理多个元素。

## 在这里要学会什么

- 线程如何映射到一维数组。
- 为什么 grid-stride loop 是一个非常常见的基线写法。
- 在开始做复杂优化之前，先把测试和 benchmark 建起来。
