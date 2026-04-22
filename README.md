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
|  |- silu/
|  |- softmax/
|  |- transpose/
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

## 代码规范

这个项目统一使用 `Google C++ Style` 作为基线，并通过根目录的 `.clang-format`
自动格式化。原因很直接：规则成熟、`clang-format` 支持稳定、对 `.h` / `.cu`
都适用，适合这种长期积累的 CUDA 学习仓库。

约定如下：

- 新改的 C++ / CUDA 文件，提交前都执行一次 `clang-format -i <file>`。
- 元素个数、索引、stride 统一使用 `size_t`；block / thread 数量继续使用 `int`。
- 对外接口放在 `ops/<op>/include/*.h`；kernel 实现放在 `variants/*.cu`；实现文件要包含自己的公开头文件，避免声明和定义漂移。
- kernel 命名统一为 `<op>_<variant>_kernel`；对外 launcher 统一为 `launch_<op>`；variant 名称函数返回纯 variant 名，例如 `naive`、`strided_loop`、`vectorized`。
- 注释优先解释“为什么这样写”或性能相关假设，不重复代码表面语义。
- 先跑正确性测试，再跑 benchmark；benchmark 结论写回算子 README 或 `notes/`。

常用格式化命令：

```bash
clang-format -i ops/silu/variants/vectorized.cu
clang-format -i common/*.h ops/*/include/*.h ops/*/variants/*.cu ops/*/tests/*.cu ops/*/bench/*.cu
```

每新增一个算子时，同步检查三处是否更新：

- `ops/<op>/`
- 根目录 `README` 的目录结构和学习顺序
- `Makefile` 里的 `OPS`

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
make test OP=silu VARIANT=vectorized
make test OP=transpose VARIANT=bank_conflict_free
```

运行示例算子的 benchmark：

```bash
make bench OP=vector_add VARIANT=naive
make bench OP=silu VARIANT=vectorized
make bench OP=transpose VARIANT=bank_conflict_free
```

建议配合阅读：

- `notes/correctness.md`
- `notes/profiling.md`

## 学习顺序

建议按下面的顺序推进：

1. `vector_add (naive -> strided_loop -> vectorized)` 感受纯粹的并行，打满内存带宽。

2. `silu (naive -> strided_loop -> vectorized)` 掌握 CUDA 数学指令，典型的 Element-wise 访存密集型。

3. `transpose (naive -> shared_mem_tiled -> bank_conflict_free)` 引入 2D 视角，理解 Shared Memory 如何拯救糟糕的全局内存非合并访问。

4. `rmsnorm (naive -> vectorized)` 单线程内的串行数学计算落地，为大规模归约做热身。

5. `reduction (naive -> shared_mem -> warp -> block)` 跨越鸿沟，掌握线程同步与基础协作。

6. `scan (naive -> shared_mem -> block)`  解决跨线程的数据依赖积累问题。

7. `softmax (naive -> online) 将 reduction` 与数学公式结合，理解 Online 算法如何消除一趟多余的内存读取。

8. `topk (naive -> warp -> block)`  掌握并发数据筛选与局部排序。

9. `gemm (naive -> tiled -> vectorized -> tensor_core)` 真正的计算密集型巅峰，将计算管线与内存管线重叠 (Pipeline)。

10. `attention (FlashAttention basic)` 集大成者，融合 Softmax、GEMM 和寄存器级的数据复用。

`attention` 被故意放在最后，因为它是一个综合性课题，会同时考验内存访问、tiling、reduction、数值稳定性和整体性能理解。
