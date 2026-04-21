#include <iostream>
#include <vector>

#include "common/cuda_check.h"
#include "common/data_utils.h"
#include "common/test_utils.h"
#include "ops/transpose/include/transpose.h"
#include "ops/transpose/reference/reference.h"

namespace {

bool run_case(size_t rows, size_t cols) {
  std::vector<float> input = make_deterministic_floats(rows * cols, 7u, 0.5f, -1.0f);
  std::vector<float> ref;
  std::vector<float> out(rows * cols, 0.0f);

  transpose_reference(input, ref, rows, cols);

  float* d_input = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, out.size() * sizeof(float)));

  CUDA_CHECK(
      cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  launch_transpose(d_input, d_out, rows, cols);
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_out));

  CompareResult result = compare_allclose(ref, out);
  std::cout << "  rows=" << rows << " cols=" << cols << " ";
  print_compare_result("compare", result);
  return result.pass;
}

}  // namespace

int main() {
  std::cout << "Testing transpose variant: " << transpose_variant_name() << std::endl;

  bool pass = true;
  pass = run_case(1, 6) && pass;
  pass = run_case(257, 1) && pass;
  pass = run_case(1 << 15, 1 << 10) && pass;

  print_check("transpose", pass);
  return pass ? 0 : 1;
}