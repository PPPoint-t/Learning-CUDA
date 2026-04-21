#pragma once

const char* transpose_variant_name();

void launch_transpose(const float* d_input, float* output, size_t rows, size_t cols);