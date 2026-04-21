#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

inline uint32_t lcg_next(uint32_t state) {
  return state * 1664525u + 1013904223u;
}

inline std::vector<float> make_deterministic_floats(size_t n, uint32_t seed = 1u,
                                                    float scale = 1.0f, float bias = 0.0f) {
  std::vector<float> out(n);
  uint32_t state = seed;
  for (size_t i = 0; i < n; ++i) {
    state = lcg_next(state);
    // 取低16位
    int centered = static_cast<int>(state & 0xffffu) - 32768;
    // 归一化、缩放、偏移
    out[i] = bias + scale * static_cast<float>(centered) / 4096.0f;
  }
  return out;
}

inline std::vector<int> make_deterministic_ints(size_t n, uint32_t seed = 1u, int low = -100,
                                                int high = 100) {
  if (low > high) {
    std::swap(low, high);
  }

  std::vector<int> out(n);
  uint32_t state = seed;
  const uint32_t range = static_cast<uint32_t>(high - low + 1);
  for (size_t i = 0; i < n; ++i) {
    state = lcg_next(state);
    out[i] = low + static_cast<int>(state % range);
  }
  return out;
}

inline std::vector<float> make_constant_floats(size_t n, float value) {
  return std::vector<float>(n, value);
}
