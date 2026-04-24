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
|  |- rmsnorm/
|  |- scan/
|  |- silu/
|  |- softmax/
|  |- transpose/
|  |- topk/
|  `- vector_add/
|- triton_lab/
|  |- upstream/
|  |  `- Triton-Puzzles/
|  |- notes/
|  |- common/
|  |- ops/
|  |- tests/
|  |- bench/
|  |- README.md
|  `- requirements.txt
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
make test OP=vector_add VARIANT=vectorized
make test OP=silu VARIANT=vectorized
make test OP=transpose VARIANT=vectorized_shared_free
make test OP=rmsnorm VARIANT=warp_reduce
```

运行示例算子的 benchmark：

```bash
make bench OP=vector_add VARIANT=vectorized
make bench OP=silu VARIANT=vectorized
make bench OP=transpose VARIANT=vectorized_shared_free
make bench OP=rmsnorm VARIANT=warp_reduce
```

建议配合阅读：

- `notes/correctness.md`
- `notes/profiling.md`
- `triton_lab/README.md`

## 当前已落地的多版本算子

- `vector_add`
  `naive`、`strided_loop`、`vectorized`
- `silu`
  `naive`、`strided_loop`、`vectorized`
- `transpose`
  `naive`、`strided_loop`、`shared_mem_tiled`、`bank_conflict_free`、`vectorized_shared_free`
- `rmsnorm`
  `naive`、`shared_reduce`、`warp_reduce`

其余算子当前主要以 README 规划和学习路线为主，后续再逐步补实现、测试和 benchmark。

## Triton 学习支线

仓库现在额外维护一条独立的 `triton_lab/` 支线，用来完成两件事：

- 先系统学习 Triton 语法、索引方式和 tile 思维。
- 等学习稳定后，再为本仓库中的算子补 Triton 实现，并把它作为和 CPU reference 并列的验证后端。

当前第一阶段只做骨架和学习材料引入：

- vendoring 一份 `Triton-Puzzles` 快照到 `triton_lab/upstream/Triton-Puzzles/`
- 为后续的 Triton `ops / tests / bench / notes` 预留目录
- 不接入当前的 `make test` / `make bench`
- 不改变现有 CUDA 主线的目录和工作流

设计原则是：

- CPU reference 仍然保留为最基础的语义基准
- Triton 实现以后作为 GPU 侧的并列验证后端
- 等 Triton 学习阶段完成后，再单独引入 CUDA vs Triton vs CPU 的比较脚手架

### Triton Lab 环境

第一阶段建议给 `triton_lab/` 单独准备一个 Python 环境，而不是和当前 `nvcc + .cu` 的 CUDA 主线混在一起。

推荐步骤：

```bash
cd /home/swy/infini/Learning-CUDA
python -m pip install --upgrade pip
python -m pip install -r triton_lab/requirements.txt
```

如果你后续想在本地打开 puzzle notebook，可以继续安装：

```bash
python -m pip install jupyter
```

如果你想试 `triton-viz`，它是可选依赖，建议等基础 puzzle 跑通之后再装：

```bash
python -m pip install git+https://github.com/Deep-Learning-Profiling-Tools/triton-viz
```

启动 notebook 的最小命令：

```bash
cd /home/swy/infini/Learning-CUDA
source .venv-triton/bin/activate
jupyter notebook triton_lab/upstream/Triton-Puzzles/Triton-Puzzles.ipynb
```

这里有几个约定要记住：

- `triton_lab/requirements.txt` 目前只提供第一阶段的最小依赖集合，后续你可以按本机 CUDA / PyTorch / Triton 版本再细化。
- 第一阶段只要求你能顺利学习 `Triton-Puzzles`，不要求 Triton 参与当前 `make test` / `make bench`。
- 等你完成 `Triton-Puzzles` 之后，再开始在 `triton_lab/ops/` 中写 Triton kernel，并逐步补 `tests/` 和 `bench/`。

## 学习顺序

建议按下面的顺序推进：

1. `vector_add (naive -> strided_loop -> vectorized)` 感受纯粹的一维并行和内存带宽上限；后续可补 `aligned_vectorized`、`half2` 和简单的 element-wise fuse。

2. `silu (naive -> strided_loop -> vectorized)` 在 element-wise 框架里引入 `expf` 这类数学指令；后续可补 `fast_math`、`half2`、`bias + silu` 融合。

3. `transpose (naive -> strided_loop -> shared_mem_tiled -> bank_conflict_free -> vectorized_shared_free)` 引入 2D 视角，先修 global memory coalescing，再修 shared memory bank conflict；后续可补 thread coarsening、更大 tile 和 `cp.async`。

4. `rmsnorm (naive -> shared_reduce -> warp_reduce)` 把 row-wise reduction 真正落地；后续可补 `vectorized`、mixed precision 和 `fused residual + bias + rmsnorm`。

5. `reduction (naive -> shared_mem -> warp -> two_stage)` 跨越鸿沟，掌握 block 内树状规约、warp shuffle 和多阶段全局归约。

6. `scan (hillis_steele -> blelloch -> block_scan -> multi_block)` 解决跨线程的数据依赖积累问题，学会并行树结构和 block / grid 级拼接。

7. `softmax (naive -> shared_mem -> warp -> online)` 将 reduction 与数学公式结合，理解 `max-subtract` 稳定写法和 online 算法如何消除一趟多余的内存读取。

8. `topk (naive_sort_like -> block_select -> quickselect -> heap_based)` 掌握并发数据筛选、局部排序和“部分有序”与“完全排序”的区别。

9. `gemm (naive -> tiled -> vectorized -> double_buffered -> tensor_core)` 真正进入计算密集型核心课题，理解 tiling、register blocking、pipeline 和 Tensor Core。

10. `attention (score -> stable_softmax -> mask -> value_accumulate -> tiled_online_softmax -> flashattention_basic)` 集大成者，融合 softmax、GEMM、tiling、masking 和 memory-efficient 设计。

`attention` 被故意放在最后，因为它是一个综合性课题，会同时考验内存访问、tiling、reduction、数值稳定性和整体性能理解。
