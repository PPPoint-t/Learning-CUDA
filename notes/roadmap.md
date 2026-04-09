# 学习路线

这个实验仓库按阶段组织学习内容。

## 第一阶段：CUDA 基础

- `vector_add`
  学习 launch 配置、越界检查、Host-Device 数据拷贝，以及第一份 benchmark。
- `reduction`
  学习 shared memory、线程同步和分层归约。
- `scan`
  学习树形并行模式，以及 block 内与 grid 级别的组合方式。

## 第二阶段：按行处理且对数值稳定性敏感的算子

- `softmax`
  学习稳定的 max-subtract 归一化，以及按行并行化的选择。
- `topk`
  学习选择类 kernel，以及简洁实现和内存流量之间的取舍。

## 第三阶段：计算密集型算子

- `gemm`
  学习 tiling、coalescing、shared memory 复用、向量化加载和吞吐量思维。

## 第四阶段：综合型 attention

- `attention`
  把前面的内容串起来，包括 reduction、softmax、tiling、masking 和内存效率。

建议规则：不要在前一个算子还没有 CPU reference、至少一个测试和至少一个 benchmark 之前，就急着开始下一阶段。
