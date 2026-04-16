#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

struct CompareResult {
  bool pass = false;
  // 绝对误差和相对误差初始值设为无穷大，这样任何实际计算的误差都会更新它们
  float max_abs_error = std::numeric_limits<float>::infinity();
  float max_rel_error = std::numeric_limits<float>::infinity();
  size_t worst_index = 0;
};

template <typename T>
bool exact_match(const std::vector<T>& out, const std::vector<T>& ref) {
  if (out.size() != ref.size()) {
    return false;
  }
  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] != ref[i]) {
      return false;
    }
  }
  return true;
}

inline CompareResult compare_allclose(const std::vector<float>& out, const std::vector<float>& ref,
                                      float atol = 1e-5f, float rtol = 1e-5f) {
  CompareResult result;
  if (out.size() != ref.size()) {
    return result;
  }

  result.pass = true;
  result.max_abs_error = 0.0f;
  result.max_rel_error = 0.0f;
  for (size_t i = 0; i < out.size(); ++i) {
    float abs_err = std::fabs(out[i] - ref[i]);

    // 求相对误差，分母不能太小，否则会导致相对误差过大，加一个小的常数来避免除以零
    float denom = std::max(std::fabs(ref[i]), 1e-12f);
    float rel_err = abs_err / denom;
    float bound = atol + rtol * std::fabs(ref[i]);

    // 更新最大误差信息
    if (abs_err > result.max_abs_error) {
      result.max_abs_error = abs_err;
      result.max_rel_error = rel_err;
      result.worst_index = i;
    }

    if (abs_err > bound) {
      result.pass = false;
    }
  }
  return result;
}

inline void print_check(const std::string& label, bool pass) {
  std::cout << label << ": " << (pass ? "PASS" : "FAIL") << std::endl;
}

inline void print_compare_result(const std::string& label, const CompareResult& result) {
  std::cout << label << ": " << (result.pass ? "PASS" : "FAIL")
            << " | max_abs_error=" << result.max_abs_error
            << " | max_rel_error=" << result.max_rel_error
            << " | worst_index=" << result.worst_index << std::endl;
}
