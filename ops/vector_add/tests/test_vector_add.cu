#include <iostream>
#include <vector>

#include "common/cuda_check.h"
#include "common/data_utils.h"
#include "common/test_utils.h"
#include "ops/vector_add/include/vector_add.h"
#include "ops/vector_add/reference/reference.h"

namespace {

bool run_case(int n) {
  std::vector<float> a = make_deterministic_floats(n, 7u, 0.5f, -1.0f);
  std::vector<float> b = make_deterministic_floats(n, 17u, 0.25f, 2.0f);
  std::vector<float> ref;
  std::vector<float> out(n, 0.0f);

  vector_add_reference(a, b, ref);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  launch_vector_add(d_a, d_b, d_out, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_out));

  CompareResult result = compare_allclose(out, ref);
  std::cout << "  n=" << n << " ";
  print_compare_result("compare", result);
  return result.pass;
}

}  // namespace

int main() {
  std::cout << "Testing vector_add variant: " << vector_add_variant_name()
            << std::endl;

  bool pass = true;
  pass = run_case(1) && pass;
  pass = run_case(257) && pass;
  pass = run_case(1 << 20) && pass;

  print_check("vector_add", pass);
  return pass ? 0 : 1;
}
