#pragma once

const char* silu_variant_name();

void launch_silu(const float* d_input, float* d_output, size_t n);
