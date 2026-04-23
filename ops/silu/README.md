# SiLU

`silu` 是一个典型的 element-wise 激活算子。
它和 `vector_add` 一样几乎没有跨元素的数据依赖，但每个元素都会调用一次 `expf`，因此能帮助你同时感受访存开销和数学指令开销。

## 当前版本

- `naive`
  一个线程负责一个输出元素，直接计算 `x / (1 + exp(-x))`。
- `strided_loop`
  在一维索引上加入 grid-stride loop，让线程数量和输入长度解耦。
- `vectorized`
  用 `float4` 一次读取 4 个输入元素，减少访存指令和地址计算开销；尾部不足 4 个元素时退回标量路径。

## 在这里要学会什么

- element-wise 算子的基线往往就是 `naive -> grid-stride -> vectorized`。
- `float4` 向量化优化的是访存和寻址，不会消除 `expf` 本身的计算开销。
- 数学函数算子要特别注意正确性验证和误差阈值，而不是只盯着吞吐量。

## 后续可补全的优化

- `fast_math`
  试验 `__expf` 或更激进的近似 sigmoid，比较吞吐和数值误差。
- `half2`
  当后续切到 FP16 / BF16 时，用 `half2` / `bfloat162` 进一步提高吞吐。
- `fused_bias_silu`
  把 bias add 和激活融合，减少一次全局内存往返。
- `alignment_specialized`
  根据指针对齐情况选择不同的向量化路径，继续打磨尾部和对齐分支。

## 常用命令

```bash
make test OP=silu VARIANT=naive
make test OP=silu VARIANT=vectorized
make bench OP=silu VARIANT=vectorized
```
