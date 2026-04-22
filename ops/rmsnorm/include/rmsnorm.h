#pragma once

const char* rmsnorm_variant_name();

void launch_rmsnorm(const float* d_input, float* d_output, const float* d_weight, size_t rows, size_t cols, float eps);