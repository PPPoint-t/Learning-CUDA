#pragma once

const char* vector_add_variant_name();

void launch_vector_add(const float* d_a,
                       const float* d_b,
                       float* d_out,
                       int n);

