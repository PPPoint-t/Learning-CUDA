# Operator Template

开始一个新算子时，建议使用下面的目录结构：

```text
ops/<op_name>/
|- README.md
|- include/
|- reference/
|- variants/
|- tests/
`- bench/
```

推荐的推进顺序：

1. 先写一个短 README，说明输入输出、shape 约定和学习目标。
2. 补一个小型 CPU reference。
3. 实现 `variants/naive.cu`。
4. 编写 `tests/test_<op_name>.cu`。
5. 编写 `bench/bench_<op_name>.cu`。
6. 在 `naive` 正确后，再继续加高级版本。

建议的 variant 命名：

- `naive`
- `strided_loop`
- `vectorized`
- `shared_mem`
- `warp`
- `block`
- `tiled`
- `online`
