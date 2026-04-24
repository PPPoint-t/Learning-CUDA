# Triton Lab

这个目录是 `Learning-CUDA` 的 Triton 学习支线。

它的目标分成两个阶段：

1. 先系统学习 Triton 语法、程序模型和调试方式。
2. 学完之后，再为当前仓库里的算子补 Triton 实现，并逐步接入 Triton `ref / test / bench`。

第一阶段刻意不接入当前的 CUDA 主线。
也就是说：

- 不修改 `ops/<op>/tests/*.cu`
- 不修改 `ops/<op>/bench/*.cu`
- 不要求 Triton 立即参与 `make test` 或 `make bench`

这样做的原因很直接：先把 Triton 学会，再决定如何作为独立后端接入，不要在语法还不熟的时候把工程结构一起搅复杂。

## 当前目录结构

```text
triton_lab/
|- upstream/
|  `- Triton-Puzzles/
|- notes/
|- common/
|- ops/
|- tests/
|- bench/
|- README.md
`- requirements.txt
```

## 当前阶段要做什么

- 先完整走一遍 `upstream/Triton-Puzzles/`
- 在 `notes/` 里记录每个 puzzle 学到的 Triton 语法点
- 把 puzzle 中的模式和本仓库算子对应起来

建议优先建立下面这组映射：

- `vector_add` / `silu`
  对应一维索引、mask、grid 和 element-wise kernel
- `softmax` / `rmsnorm`
  对应按行处理、block 内协作和 reduction 思维
- `transpose` / `gemm`
  对应 tile、pointer arithmetic、二维布局和向量化搬运
- `attention`
  对应在线归一化、融合和 memory-efficient 设计

## 上游快照来源

- upstream repo: `https://github.com/gpu-mode/Triton-Puzzles`
- vendored commit: `4d794abed02292081500ebf4b1e35cd5fc5fb019`
- license: Apache 2.0

上游快照当前按原样放在 `upstream/Triton-Puzzles/`，便于你先学习，不急着做本地改造。

## 后续接入规划

等你写完 `Triton-Puzzles` 之后，再按下面的顺序推进：

1. 在 `triton_lab/ops/` 中为 `vector_add` 写第一个 Triton kernel
2. 在 `triton_lab/tests/` 中做 `Triton vs CPU reference`
3. 在 `triton_lab/bench/` 中做 Triton benchmark
4. 再考虑引入 `CUDA vs Triton vs CPU` 的统一比较脚手架

这里要坚持一个原则：

- CPU reference 是最基础的语义基准
- Triton reference 是未来新增的 GPU 侧并列验证后端
- 现有 CUDA kernel 仍然是本仓库当前的主优化对象

## 环境

第一阶段只提供一个最小 `requirements.txt`。
真正开始跑 Triton kernel 前，仍然建议你按自己的 CUDA / PyTorch / Triton 环境实际情况做安装和校验。
