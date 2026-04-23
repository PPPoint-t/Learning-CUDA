# Scan

`scan` 是第一个逼着你从“逐元素思维”转到“并行树结构思维”的算子。

建议的版本演进：

- `hillis_steele`
- `blelloch`
- `block_scan`
- `multi_block`

核心学习点：

- up-sweep 和 down-sweep 的结构
- block 内 scan 和多 block 组合方式
- 非 2 的幂长度输入下的正确性处理

进一步优化时，建议继续补这些内容：

- shared memory 树结构中的 bank conflict 处理
- warp-scan 和 block-scan 的混合写法
- block 间前缀和传播的组织方式
- 更高阶的 `decoupled_lookback` 思路
