#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

unsigned seed = 19260817;

float gen() {
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  return (seed / float(-1u)) - 0.5;
}

float *alloc(size_t size) { return (float *) aligned_alloc(128, size); }

void print_diff(const float *x, const float *y, int n) {
  float diff = 0;
  for (int i = 0; i < n; ++i)
    diff = std::max(diff, std::abs(((float *) x)[i] - ((float *) y)[i]));
  printf("max diff = %.12f\n", diff);
}

template<typename F>
void run(int argc, char **argv, F f) {
  if (argc < 2) {
    printf("usage: %s <repeat times>", argv[0]);
    exit(1);
  }
  for (int i = 1; i < argc; ++i) {
    using namespace std::chrono;
    int rep = atoi(argv[i]);
    auto beg = high_resolution_clock::now();
    f(rep);
    auto end = high_resolution_clock::now();
    printf("avg time: %8lfs\n", duration_cast<duration<double>>(end - beg).count() / rep);
  }
}

const int M = 2048;
const int N = 2048;
const int K = 2048;

void gemm_naive(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c) {
  memset(c, 0, sizeof(float) * M * N);
#pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < N; ++j) {
        c[i * N + j] += a[i * K + k] * b[k * N + j];
      }
    }
  }
}

const int batch = 256;
const int in_channel = 256;
const int out_channel = 512;
const int in_size = 14;
const int kernel = 3;
const int pad = 1;
const int out_size = in_size - kernel + 2 * pad + 1;

void conv_naive(const float *__restrict__ a, const float *__restrict__ w, float *__restrict__ b) {
#pragma omp parallel for
  for (int nn = 0; nn < batch; ++nn) {
    for (int cc = 0; cc < out_channel; ++cc) {
      for (int xx = 0; xx < out_size; ++xx) {
        for (int yy = 0; yy < out_size; ++yy) {
          float s = 0;
          for (int rc = 0; rc < in_channel; ++rc) {
            for (int rx = 0; rx < kernel; ++rx) {
              for (int ry = 0; ry < kernel; ++ry) {
                int x = xx + rx - pad, y = yy + ry - pad;
                s += ((0 <= x && x < in_size && 0 <= y && y < in_size) ? a[nn * (in_channel * in_size * in_size) + rc * (in_size * in_size) + x * in_size + y] : 0)
                     * w[cc * (in_channel * kernel * kernel) + rc * (kernel * kernel) + rx * kernel + ry];
              }
            }
          }
          b[nn * (out_channel * out_size * out_size) + cc * (out_size * out_size) + xx * out_size + yy] = s;
        }
      }
    }
  }
}
