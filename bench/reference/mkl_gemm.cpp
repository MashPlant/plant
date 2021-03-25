// clang++ mkl_gemm.cpp -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -fopenmp -Ofast -march=native && ./a.out 500 500
#include "common.h"
#include <mkl.h>

int main(int argc, char **argv) {
  float *a = alloc(sizeof(float) * M * K);
  float *b = alloc(sizeof(float) * K * N);
  float *c = alloc(sizeof(float) * M * N);
  float *c1 = alloc(sizeof(float) * M * N);

  for (int i = 0; i < M * K; ++i) a[i] = gen();
  for (int i = 0; i < K * N; ++i) b[i] = gen();

  bool transposeA = false, transposeB = false;
  float alpha = 1.0, beta = 0.0;
  int ldA = transposeA ? M : K, ldB = transposeB ? K : N, ldC = N;

  run(argc, argv, [=](int rep) {
    for (int i = 0; i < rep; ++i)
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a, ldA, b, ldB, beta, c, ldC);
  });

  gemm_naive(a, b, c1);
  print_diff(c, c1, M * N);
}
