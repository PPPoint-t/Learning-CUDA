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

等你回到这里时，归档作业和你单独保留的 FlashAttention 分支都会是很有价值的参考。
