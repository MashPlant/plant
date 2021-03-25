// nvcc cublas_gemm.cu -lcublas -O3 && ./a.out 500 500
#include "common.h"
#include <cublas_v2.h>

int main(int argc, char **argv) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  float *a = alloc(sizeof(float) * M * K);
  float *b = alloc(sizeof(float) * K * N);
  float *c = alloc(sizeof(float) * M * N);
  float *c1 = alloc(sizeof(float) * M * N);

  for (int i = 0; i < M * K; ++i) a[i] = gen();
  for (int i = 0; i < K * N; ++i) b[i] = gen();

  bool transposeA = false, transposeB = false;
  float alpha = 1, beta = 0;
  int ldA = transposeA ? M : K, ldB = transposeB ? K : N, ldC = N;

  float *d_a, *d_b, *d_c;
  cudaMalloc((void **) &d_a, sizeof(float) * M * K);
  cudaMalloc((void **) &d_b, sizeof(float) * K * N);
  cudaMalloc((void **) &d_c, sizeof(float) * M * N);
  cudaMemcpy(d_a, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

  run(argc, argv, [=](int rep) {
    for (int i = 0; i < rep; ++i) {
      // cuBLAS gemm默认列主序，这里传入B，A，则计算B^T x A^T = (A x B)^T
      // 而保存时也是列主序，所以实际上得到了A x B
      cublasSgemm(handle, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N, transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
                  N, M, K, &alpha, d_b, ldB, d_a, ldA, &beta, d_c, ldC);
      cudaStreamSynchronize(0);
    }
  });

  gemm_naive(a, b, c1);
  cudaMemcpy(c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
  print_diff(c, c1, M * N);
}
