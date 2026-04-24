# Triton Tests

这里预留给后续 Triton correctness 测试。

未来建议的验证顺序：

1. `Triton vs CPU reference`
2. 稳定之后，再补 `CUDA vs Triton vs CPU`

第一阶段不接入现有 `ops/<op>/tests/*.cu`。
这样可以避免在 Triton 学习阶段把 C++ / CUDA 和 Python / Triton 两套测试体系强耦合在一起。
