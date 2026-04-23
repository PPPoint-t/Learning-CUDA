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
- `vectorized`
  用 `float4` 一次搬运 4 个元素，并对尾部不足 4 个元素的部分做标量回退。

## 在这里要学会什么

- 线程如何映射到一维数组。
- 为什么 grid-stride loop 是一个非常常见的基线写法。
- `float4` 向量化为什么能减少访存指令和地址计算开销。
- 为什么这个算子通常不值得引入 shared memory：它几乎没有数据复用，只会增加中转成本。
- 在开始做复杂优化之前，先把测试和 benchmark 建起来。

## 后续可补全的优化

- `aligned_vectorized`
  更严格地区分对齐路径和非对齐路径，进一步打磨 vectorized baseline。
- `half2`
  如果后续开始练 FP16 / BF16，可以尝试 `half2` / `bfloat162` 版本。
- `fused_elementwise`
  把 add 和后续激活或 bias 操作融合，减少一次 global memory round-trip。
- `launch_tuning`
  系统比较 block size、occupancy 和带宽利用率，而不是只固定在 256 线程。

## 常用命令

```bash
make test OP=vector_add VARIANT=naive
make test OP=vector_add VARIANT=vectorized
make bench OP=vector_add VARIANT=vectorized
```
