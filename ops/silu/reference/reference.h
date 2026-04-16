#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

inline void silu_reference(const std::vector<float>& input, std::vector<float>& output) {
  output.resize(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = input[i] / (1.0f + std::exp(-input[i]));
  }
}
