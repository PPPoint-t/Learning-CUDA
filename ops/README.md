# Operators

这个目录按算子组织学习内容。

每个算子目录最终建议包含：

- `README.md`
- `include/`
- `reference/`
- `variants/`
- `tests/`
- `bench/`

建议同时利用根目录下的 `common/` 和 `notes/`：

- `common/` 放可复用的小工具
- `notes/correctness.md` 说明如何验证 reference 的可信度
- `notes/profiling.md` 说明如何记录 benchmark 和 profiling 结论

当前已经进入“可持续迭代”状态的算子包括：

- `vector_add/`
- `silu/`
- `transpose/`
- `rmsnorm/`

这些目录都已经具备至少一部分 `reference / variants / tests / bench`。
其余目录当前主要作为后续学习路线的占位和规划，但 README 已经给出了建议的版本演进和学习重点。
