// CPU: ppcg ppcg_gemm.c --target=c --openmp && clang ppcg_gemm.ppcg.c -fopenmp -Ofast -march=native && ./a.out 10 10
// GPU: ppcg ppcg_gemm.c && nvcc ppcg_gemm_host.cu ppcg_gemm_kernel.cu -Xcompiler -fopenmp -O3 -use_fast_math && ./a.out 500 500
#include "common.h"

void gemm_ppcg(float a[STATIC_RESTRICT 2048][2048], float b[STATIC_RESTRICT 2048][2048], float c[STATIC_RESTRICT 2048][2048]) {
#pragma scop
  for (int i = 0; i < 2048; ++i) {
    for (int j = 0; j < 2048; ++j) {
      c[i][j] = 0;
      for (int k = 0; k < 2048; ++k) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
#pragma endscop
}

typedef struct { float *a, *b, *c; } Capture;

void fn(void *f_data, int rep) {
  Capture c = *(Capture *) f_data;
  for (int i = 0; i < rep; ++i) {
#if __CUDACC__
    // 由PPCG生成的代码复制而来
    dim3 k0_dimBlock(16, 32);
    dim3 k0_dimGrid(64, 64);
    kernel0<<<k0_dimGrid, k0_dimBlock>>>(c.a, c.b, c.c);
    cudaStreamSynchronize(0);
#else
    gemm_ppcg(c.a, c.b, c.c);
#endif
  }
}

int main(int argc, char **argv) {
  float *a = alloc(M * K);
  float *b = alloc(K * N);
  float *c = alloc(M * N);
  float *c1 = alloc(M * N);

  for (int i = 0; i < M * K; ++i) a[i] = gen();
  for (int i = 0; i < K * N; ++i) b[i] = gen();

#if __CUDACC__
  float *d_a, *d_b, *d_c;
  cudaMalloc((void **) &d_a, sizeof(float) * M * K);
  cudaMalloc((void **) &d_b, sizeof(float) * K * N);
  cudaMalloc((void **) &d_c, sizeof(float) * M * N);
  cudaMemcpy(d_a, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

  Capture cap = {d_a, d_b, d_c};
  run_c(argc, argv, fn, &cap);

  cudaMemcpy(c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
#else
  Capture cap = {a, b, c};
  run_c(argc, argv, fn, &cap);
#endif

  gemm_naive(a, b, c1);
  print_diff(c, c1, M * N);
}
