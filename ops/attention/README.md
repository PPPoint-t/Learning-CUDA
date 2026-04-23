# Attention

`attention` 是这个学习平台里的最终综合题。

建议的推进顺序：

1. 先只做 score 计算
2. 再补稳定 softmax
3. 再补 mask
4. 再补 value 累加
5. 再做 tiling 和 online softmax
6. 最后再回头研究 FlashAttention 风格的 memory-efficient 写法

这个算子应当放在学习路线后半段，因为它依赖你先掌握 reduction、softmax、topk 类选择思路，以及 GEMM 风格的 tiling。

真正开始实现时，建议重点盯住这些问题：

- QK^T 的 tile 应该如何放进 shared memory / registers
- softmax 如何在线更新，避免物化完整 attention score matrix
- mask 和 dropout 应该插在哪一步，才不会破坏数值稳定性
- value 累加如何和 softmax 归一化流水化地结合
- 哪些中间量值得留在寄存器里，哪些必须落 shared memory

等你回到这里时，归档作业和你单独保留的 FlashAttention 分支都会是很有价值的参考。
