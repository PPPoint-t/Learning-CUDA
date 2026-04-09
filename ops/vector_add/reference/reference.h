#pragma once

#include <stdexcept>
#include <vector>

inline void vector_add_reference(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 std::vector<float>& out) {
  if (a.size() != b.size()) {
    throw std::runtime_error("vector_add_reference expects equal-sized inputs");
  }

  out.resize(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = a[i] + b[i];
  }
}

