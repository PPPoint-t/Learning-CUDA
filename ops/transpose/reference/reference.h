#pragma once

#include <stdexcept>
#include <vector>

inline void transpose_reference(const std::vector<float>& input, std::vector<float>& output, size_t rows, size_t cols) {
  if(input.size() != rows * cols) {
    throw std::runtime_error("transpose_reference expects input size to match rows * cols");
  }
  output.resize(input.size());
  for(size_t r =0;r<rows;++r){
    for(size_t c=0;c<cols;++c){
      output[c*rows+r] = input[r*cols+c];
    }
  }
}
