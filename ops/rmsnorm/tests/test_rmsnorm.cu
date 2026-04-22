#include <iostream>
#include <vector>

#include "common/cuda_check.h"
#include "common/data_utils.h"
#include "common/test_utils.h"
#include "ops/rmsnorm/include/rmsnorm.h"
#include "ops/rmsnorm/reference/reference.h"

namespace {

bool run_case(size_t rows, size_t cols, float eps) {
  size_t n = rows * cols;
  std::vector<float> input = make_deterministic_floats(n, 7u, 0.5f, -1.0f);
  std::vector<float> weight = make_deterministic_floats(cols, 17u, 0.25f, 2.0f);
  std::vector<float> ref(n, 0.0f);
  std::vector<float> out(n, 0.0f);

  rmsnorm_reference(input, ref, weight, rows, cols, eps);

  float* d_input = nullptr;
  float* d_weight = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weight, weight.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, out.size() * sizeof(float)));

  CUDA_CHECK(
      cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice));
  launch_rmsnorm(d_input, d_out, d_weight, rows, cols, eps);
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_weight));
  CUDA_CHECK(cudaFree(d_out));

  CompareResult result = compare_allclose(ref, out);
  std::cout << "  rows=" << rows << " cols=" << cols << " eps=" << eps << " ";
  print_compare_result("compare", result);
  return result.pass;
}

}

int main(){
  std::cout << "Testing RMSNorm variant: " << rmsnorm_variant_name() << std::endl;

  bool pass = true;
  pass = run_case(4, 8, 1e-5f) && pass;
  pass = run_case(16, 64, 1e-5f) && pass;
  pass = run_case(32, 128, 1e-5f) && pass;
  return pass ? 0 : 1;
}