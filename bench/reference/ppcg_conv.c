// CPU: ppcg ppcg_conv.c --target=c --openmp && clang ppcg_conv.ppcg.c -fopenmp -Ofast -march=native && ./a.out 10 10
// GPU: ppcg ppcg_conv.c && nvcc ppcg_conv_host.cu ppcg_conv_kernel.cu -O3 -use_fast_math && ./a.out 1 1
#include "common.h"

void conv_ppcg(float a[STATIC_RESTRICT 256][256][14][14], float w[STATIC_RESTRICT 512][256][3][3],
               float b[STATIC_RESTRICT 256][512][14][14]) {
#pragma scop
  for (int nn = 0; nn < 256; ++nn) {
    for (int cc = 0; cc < 512; ++cc) {
      for (int xx = 0; xx < 14; ++xx) {
        for (int yy = 0; yy < 14; ++yy) {
          b[nn][cc][xx][yy] = 0;
          for (int rc = 0; rc < 256; ++rc) {
            for (int rx = 0; rx < 3; ++rx) {
              for (int ry = 0; ry < 3; ++ry) {
                b[nn][cc][xx][yy] += ((0 <= xx + rx - 1 && xx + rx - 1 < 14 && 0 <= yy + ry - 1 && yy + ry - 1 < 14) ? a[nn][rc][xx + rx - 1][yy + ry - 1] : 0)
                                     * w[cc][rc][rx][ry];
              }
            }
          }
        }
      }
    }
  }
#pragma endscop
}

typedef struct { float *a, *w, *b; } Capture;

void fn(void *f_data, int rep) {
  Capture c = *(Capture *) f_data;
  for (int i = 0; i < rep; ++i) {
#if __CUDACC__
    // 由PPCG生成的代码复制而来
    dim3 k0_dimBlock(4, 4, 32);
    dim3 k0_dimGrid(16, 8);
    kernel0<<<k0_dimGrid, k0_dimBlock>>>(c.a, c.b, c.w);
    cudaStreamSynchronize(0);
#else
    conv_ppcg(c.a, c.w, c.b);
#endif
  }
}

int main(int argc, char **argv) {
  float *a = alloc(sizeof(float) * 256 * 256 * 14 * 14);
  float *w = alloc(sizeof(float) * 512 * 256 * 3 * 3);
  float *b = alloc(sizeof(float) * 256 * 512 * 14 * 14);
  float *b1 = alloc(sizeof(float) * 256 * 512 * 14 * 14);

  for (int i = 0; i < 256 * 256 * 14 * 14; ++i) a[i] = gen();
  for (int i = 0; i < 512 * 256 * 3 * 3; ++i) w[i] = gen();

#if __CUDACC__
  float *d_a, *d_w, *d_b;
  cudaMalloc((void **) &d_a, sizeof(float) * 256 * 256 * 14 * 14);
  cudaMalloc((void **) &d_w, sizeof(float) * 512 * 256 * 3 * 3);
  cudaMalloc((void **) &d_b, sizeof(float) * 256 * 512 * 14 * 14);
  cudaMemcpy(d_a, a, sizeof(float) * 256 * 256 * 14 * 14, cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, w, sizeof(float) * 512 * 256 * 3 * 3, cudaMemcpyHostToDevice);

  Capture cap = {d_a, d_w, d_b};
  run_c(argc, argv, fn, &cap);

  cudaMemcpy(b, d_b, sizeof(float) * 256 * 512 * 14 * 14, cudaMemcpyDeviceToHost);
#else
  Capture cap = {a, w, b};
  run_c(argc, argv, fn, &cap);
#endif

  conv_naive(a, w, b1);
  print_diff(b, b1, 256 * 512 * 14 * 14);
}
