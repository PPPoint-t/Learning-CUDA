# Triton Bench

这里预留给后续 Triton benchmark。

建议等你已经能稳定写出正确的 Triton kernel 之后，再开始建 benchmark。

原因是：

- Triton benchmark 最终最好和 CUDA benchmark 放在同一层级比较
- 如果太早做，很容易把环境、数据搬运和接口开销混进 kernel 性能里

第一阶段只保留目录和说明，不放实际 benchmark 脚本。
