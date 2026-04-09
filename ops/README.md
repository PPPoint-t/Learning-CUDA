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

当前第一个完整示例是 `vector_add/`。
其余目录先作为后续学习路线的占位和规划。
