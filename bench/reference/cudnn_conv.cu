// nvcc cudnn_conv.cu -lcudnn -Xcompiler -fopenmp -O3 && ./a.out 100 100
#include "common.h"
#include <cudnn.h>

int main(int argc, char **argv) {
  cudnnHandle_t handle;
  cudnnCreate(&handle);

  float *x = alloc(batch * in_channel * in_size * in_size);
  float *w = alloc(out_channel * in_channel * kernel * kernel);
  float *y = alloc(batch * out_channel * out_size * out_size);
  float *y1 = alloc(batch * out_channel * out_size * out_size);

  for (int i = 0; i < batch * in_channel * in_size * in_size; ++i) x[i] = gen();
  for (int i = 0; i < out_channel * in_channel * kernel * kernel; ++i) w[i] = gen();

  float *d_x, *d_w, *d_y;
  cudaMalloc((void **) &d_x, sizeof(float) * batch * in_channel * in_size * in_size);
  cudaMalloc((void **) &d_w, sizeof(float) * out_channel * in_channel * kernel * kernel);
  cudaMalloc((void **) &d_y, sizeof(float) * batch * out_channel * out_size * out_size);
  cudaMemcpy(d_x, x, sizeof(float) * batch * in_channel * in_size * in_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, w, sizeof(float) * out_channel * in_channel * kernel * kernel, cudaMemcpyHostToDevice);

  cudnnTensorDescriptor_t x_desc;
  cudnnCreateTensorDescriptor(&x_desc);
  cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, in_channel, in_size, in_size);
  cudnnFilterDescriptor_t w_desc;
  cudnnCreateFilterDescriptor(&w_desc);
  cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channel, in_channel, kernel, kernel);
  cudnnTensorDescriptor_t y_desc;
  cudnnCreateTensorDescriptor(&y_desc);
  cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, out_channel, out_size, out_size);

  cudnnConvolutionDescriptor_t conv_desc;
  cudnnCreateConvolutionDescriptor(&conv_desc);
  // stride = dilation = (1, 1)
  cudnnSetConvolution2dDescriptor(conv_desc, pad, pad, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

  cudnnConvolutionFwdAlgoPerf_t algo;
#ifdef ALGO
  algo.algo = ALGO;
#else
  int _;
  cudnnFindConvolutionForwardAlgorithm(handle, x_desc, w_desc, conv_desc, y_desc, 1, &_, &algo);
#endif
  printf("algo = %d\n", (int) algo.algo);

  size_t workspace_size;
  cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc, algo.algo, &workspace_size);
  void *workspace;
  cudaMalloc(&workspace, workspace_size);
  printf("workspace_size = %d, workspace = %p\n", (int) workspace_size, workspace);

  float alpha = 1, beta = 0;

  run(argc, argv, [=](int rep) {
    for (int i = 0; i < rep; ++i) {
      cudnnConvolutionForward(handle, &alpha, x_desc, d_x, w_desc, d_w, conv_desc, algo.algo,
                              workspace, workspace_size, &beta, y_desc, d_y);
      cudaStreamSynchronize(0);
    }
  });

  conv_naive(x, w, y1);
  cudaMemcpy(y, d_y, sizeof(float) * batch * out_channel * out_size * out_size, cudaMemcpyDeviceToHost);
  print_diff(y, y1, batch * out_channel * out_size * out_size);
}
