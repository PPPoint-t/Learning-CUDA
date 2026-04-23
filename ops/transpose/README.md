# Transpose

`transpose` 是这个仓库里第一个明显依赖二维访存模式的算子。
它几乎没有计算量，性能主要取决于 global memory 是否合并访问，以及 shared memory 是否发生 bank conflict。

## 当前版本

- `naive`
  一个线程负责一个输出元素。输入读取基本连续，但输出写回是跨行跳写，global store 不合并。
- `strided_loop`
  在 `naive` 的索引逻辑上加 2D grid-stride loop，主要是为了练熟二维线程映射；在当前 benchmark 尺寸下，性能通常和 `naive` 接近。
- `shared_mem_tiled`
  先把一个 tile 搬到 shared memory，再转置后写回 global memory，把原本糟糕的 global 写访问改成合并写。
- `bank_conflict_free`
  在 shared tile 的第二维补 `+1` padding，消掉转置读取时的 shared memory bank conflict。
- `vectorized_shared_free`
  在 bank-conflict-free tiled transpose 的基础上，用 `float4` 做 128-bit 搬运；当 `rows` 或 `cols` 不是 4 的倍数时，自动退回标量 tiled 路径。

## 在这里要学会什么

- 为什么 transpose 常常是典型的内存带宽受限算子。
- global memory coalescing 和 shared memory bank conflict 是两层不同的问题。
- shared memory 中 `tile[T][T + 1]` 这一个 padding 为什么能带来真实收益。
- 向量化不只是把 `float` 改成 `float4`，还必须处理对齐和尾部边界。

## 后续可补全的优化

- `thread_coarsening`
  让一个线程负责 tile 中多个元素，摊薄索引计算和同步开销。
- `larger_tiles`
  针对具体 GPU 调整 tile 大小，在 shared memory 占用和吞吐之间找平衡。
- `async_tiled`
  在较新的架构上尝试 `cp.async` 或双缓冲 tile，重叠 global -> shared 搬运和计算。
- `diagonal_reordering`
  对很大的方阵，探索 diagonal block reordering 之类的调度方式，减少访存热点。

## 常用命令

```bash
make test OP=transpose VARIANT=naive
make test OP=transpose VARIANT=bank_conflict_free
make test OP=transpose VARIANT=vectorized_shared_free
make bench OP=transpose VARIANT=shared_mem_tiled
make bench OP=transpose VARIANT=bank_conflict_free
make bench OP=transpose VARIANT=vectorized_shared_free
```
