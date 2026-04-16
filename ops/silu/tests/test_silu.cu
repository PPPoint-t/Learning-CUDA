#include <iostream>
#include <vector>

#include "common/cuda_check.h"
#include "common/data_utils.h"
#include "common/test_utils.h"
#include "ops/silu/include/silu.h"
#include "ops/silu/reference/reference.h"

namespace {

bool run_case(size_t n) {
  std::vector<float> input = make_deterministic_floats(n, 7u, 0.5f, -1.0f);
  std::vector<float> ref;
  std::vector<float> out(n, 0.0f);

  silu_reference(input, ref);

  float* d_input = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  launch_silu(d_input, d_out, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_out));

  CompareResult result = compare_allclose(out, ref);
  std::cout << "  n=" << n << " ";
  print_compare_result("compare", result);
  return result.pass;
}

}  // namespace

int main() {
  std::cout << "Testing silu variant: " << silu_variant_name() << std::endl;

  bool pass = true;
  pass = run_case(1) && pass;
  pass = run_case(257) && pass;
  pass = run_case(1 << 20) && pass;

  print_check("silu", pass);
  return pass ? 0 : 1;
}
