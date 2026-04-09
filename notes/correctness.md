# Correctness 说明

这是学习 CUDA 时最容易被忽略、但最关键的一件事：你怎么知道自己写的 CPU reference 是对的？

短答案是：你不能只“相信自己写的那份 CPU 代码”，你要主动建立一套验证路径。

## 推荐的验证顺序

1. 先写最朴素、最容易读懂的 CPU reference。
2. 用手算得出的极小样例检查它。
3. 用边界 case 检查它。
4. 用算子本身的数学性质检查它。
5. 如果条件允许，再和可信外部实现做交叉验证。

## 1. CPU reference 要故意写得“慢但清楚”

CPU reference 的目标不是快，而是可信。

例如：

- `vector_add`
  直接一层 for 循环。
- `reduction`
  直接从头累加。
- `softmax`
  明确分成 `max -> exp -> sum -> normalize` 四步。
- `topk`
  可以直接用 CPU 上的排序或部分排序思路做基准。

不要一上来就在 CPU reference 里做复杂优化，否则它会失去“参考答案”的价值。

## 2. 先写能手算的极小样例

例如：

- `vector_add`
  `[1, 2] + [3, 4] = [4, 6]`
- `reduction`
  `[1, 2, 3, 4] -> 10`
- `softmax`
  `[0, 0] -> [0.5, 0.5]`
- `topk`
  `[5, 1, 4, 2], k=2 -> [5, 4]`

如果 CPU reference 连这些 case 都不稳定，那 CUDA 版本的测试就没有意义。

## 3. 要主动覆盖边界 case

至少包括：

- 空输入或最小输入
- 只有 1 个元素
- 不是 block size 整数倍的长度
- 含有负数、零、重复值
- 很大和很小的数值
- 对 attention/softmax 这类算子，要覆盖 mask、不同 shape、不同 head 配置

## 4. 利用算子本身的性质做自检

有些算子即使没有外部库，也能做很强的自检。

例如：

- `softmax`
  每一行输出之和应接近 1；每个元素应大于 0。
- `reduction(sum)`
  改变输入切分方式，最终和应一致。
- `vector_add`
  `out[i] - a[i]` 应等于 `b[i]`。
- `topk`
  输出应有序，且每个输出都应来自输入集合。

这些性质不能替代 reference，但能快速发现明显错误。

## 5. 可以和外部实现交叉验证

可以，而且很推荐。

常见选择有：

- `PyTorch`
- `NumPy`
- `CuPy`
- CPU 标准库算法

例如：

- `softmax` 可以和 `torch.softmax` 对比
- `matmul/gemm` 可以和 `torch.matmul` 或 `numpy.matmul` 对比
- `topk` 可以和 `torch.topk` 对比

注意：

- 外部实现是“交叉验证”，不是唯一真理。
- 比较时要保持 shape、dtype、layout 和语义一致。
- 测试正确性时，不需要要求完全相同的 bitwise 结果，浮点数通常要用误差阈值比较。

## 实际建议

每个算子都至少准备三层验证：

- 手写小样例
- CPU reference 对比
- 外部实现交叉验证（如果容易获得）

如果这三层都过了，再去做性能优化，方向才不会跑偏。

