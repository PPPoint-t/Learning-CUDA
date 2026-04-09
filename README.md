# Learning-CUDA Lab

这个分支把仓库整理成一个长期使用的 CUDA 学习平台。

原来的 2025 夏季作业已经保存在 `archive/summer2025-assignment/` 中。
当前根目录用于日常学习、测试、benchmark 和算子迭代。

## 目标

- 保持一条清晰、稳定、可长期积累的学习主线。
- 一次专注一个算子，但允许同一算子保留多个 kernel 版本并排对比。
- 将历史作业和新的学习基础设施分开管理。
- 养成“先校验正确性，再看性能”的习惯。

## 目录结构

```text
Learning-CUDA/
|- archive/
|  `- summer2025-assignment/
|- common/
|  |- benchmark_utils.h
|  |- cuda_check.h
|  |- data_utils.h
|  |- device_info.h
|  |- test_utils.h
|  `- timer.h
|- notes/
|  |- correctness.md
|  |- profiling.md
|  `- roadmap.md
|- ops/
|  |- _template/
|  |- attention/
|  |- gemm/
|  |- reduction/
|  |- scan/
|  |- softmax/
|  |- topk/
|  `- vector_add/
|- LICENSE
`- Makefile
```

## 开发节奏

小而确定的改动可以直接提交到 `lab`。

建议每个算子都按下面的顺序推进：

1. 写或更新该算子的 README。
2. 写一个 CPU reference。
3. 写第一个 CUDA 版本，通常是 `naive`。
4. 加入正确性测试。
5. 加入 benchmark。
6. 再继续做 `warp`、`block`、`tiled` 等改进版本。
7. 把观察到的瓶颈和结论记录到算子 README 或 `notes/` 里。

只有当实验很乱、周期很长、或者不确定是否保留时，再使用临时 `exp/*` 分支。

## 常用命令

列出当前规划中的算子：

```bash
make list-ops
```

运行归档作业：

```bash
make assignment
make assignment VERBOSE=true
```

运行示例算子的测试：

```bash
make test OP=vector_add VARIANT=naive
make test OP=vector_add VARIANT=strided_loop
```

运行示例算子的 benchmark：

```bash
make bench OP=vector_add VARIANT=naive
```

建议配合阅读：

- `notes/correctness.md`
- `notes/profiling.md`

## 学习顺序

建议按下面的顺序推进：

1. `vector_add`
2. `reduction`
3. `scan`
4. `softmax`
5. `topk`
6. `gemm`
7. `attention`

`attention` 被故意放在最后，因为它是一个综合性课题，会同时考验内存访问、tiling、reduction、数值稳定性和整体性能理解。
