# RMSNorm

`rmsnorm` 是这个仓库里第一个把 row-wise reduction 和逐元素缩放真正绑在一起的算子。
它既不是纯粹的 element-wise，也不像 `reduction` 那样只有一个标量输出，非常适合练“每一行该由一个线程、一个 warp，还是一个 block 来负责”。

## 当前版本

- `naive`
  一个线程负责一整行，串行算完整个平方和，再回头做归一化和加权。
- `shared_reduce`
  一个 block 负责一行，每个线程处理部分列，在 shared memory 中做树状规约。
- `warp_reduce`
  一个 warp 负责一行，用 `__shfl_down_sync` 做寄存器级规约，减少 shared memory 和同步开销。

## 在这里要学会什么

- row-wise 算子通常要根据 `cols` 决定映射粒度，而不是盲目地一线程一行。
- `block-per-row` 和 `warp-per-row` 都是常见设计，适合的 hidden size 不一样。
- shared memory reduction 和 warp shuffle reduction 的收益点不同：前者更通用，后者更轻量。
- 这类算子的瓶颈往往来自“读取一整行 + 做一次归约 + 再写回一整行”的总流量和同步开销。

## 后续可补全的优化

- `vectorized`
  用 `float4` 或 `half2` 读取 `input / weight / output`，减少访存指令数。
- `adaptive_row_mapping`
  根据 `cols` 自动选择 `warp-per-row`、`block-per-row` 甚至 `multi-warp-per-row`。
- `mixed_precision`
  输入使用 FP16 / BF16，归约累加保持 FP32，贴近实际训练和推理实现。
- `fused_residual_bias_rmsnorm`
  把 residual add、bias add 和 RMSNorm 融合，减少多次 global memory 往返。
- `layernorm_extension`
  以后如果扩展到 LayerNorm，可以顺带练习 `Welford` 这类更稳定的方差计算。

## 常用命令

```bash
make test OP=rmsnorm VARIANT=naive
make test OP=rmsnorm VARIANT=shared_reduce
make test OP=rmsnorm VARIANT=warp_reduce
make bench OP=rmsnorm VARIANT=warp_reduce
```
