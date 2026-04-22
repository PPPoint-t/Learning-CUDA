#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

inline void rmsnorm_reference(const std::vector<float>& input, std::vector<float>& output,
                              const std::vector<float>& weight, size_t rows, size_t cols,
                              float eps) {
  // 均方根层归一化一般是二维输入矩阵，每一行有自己的 RMS
  if (input.size() != rows * cols || output.size() != rows * cols) {
    throw std::runtime_error("rmsnorm_reference expects input / output size to match rows * cols");
  }
  if (weight.size() != cols) {
    throw std::runtime_error("rmsnorm_reference expects weight size to match cols");
  }

  for (size_t r = 0; r < rows; r++) {
    float mean_square = 0.0f;
    for (size_t c = 0; c < cols; c++) {
      float val = input[r * cols + c];
      mean_square += val * val;
    }
    mean_square /= cols;
    float inv_rms = 1.0f / std::sqrt(mean_square + eps);

    for (size_t c = 0; c < cols; c++) {
      output[r * cols + c] = input[r * cols + c] * inv_rms * weight[c];
    }
  }
}
