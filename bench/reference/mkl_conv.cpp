// clang++ mkl_conv.cpp -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkldnn -fopenmp -Ofast -march=native && ./a.out 200 200
#include "common.hpp"
#include <mkldnn.hpp>

using namespace mkldnn;

int main(int argc, char **argv) {
  engine cpu_engine(engine::kind::cpu, 0);
  stream cpu_stream(cpu_engine);

  float *a = alloc(sizeof(float) * batch * in_channel * in_size * in_size);
  float *w = alloc(sizeof(float) * out_channel * in_channel * kernel * kernel);
  float *bias = alloc(sizeof(float) * out_channel);
  float *b1 = alloc(sizeof(float) * batch * out_channel * out_size * out_size);

  for (int i = 0; i < batch * in_channel * in_size * in_size; ++i) a[i] = gen();
  for (int i = 0; i < out_channel * in_channel * kernel * kernel; ++i) w[i] = gen();
  for (int i = 0; i < out_channel; ++i) bias[i] = 0;

  auto a_md = memory::desc({batch, in_channel, in_size, in_size}, memory::data_type::f32, memory::format_tag::nchw);
  auto w_md = memory::desc({out_channel, in_channel, kernel, kernel}, memory::data_type::f32, memory::format_tag::oihw);
  auto bias_md = memory::desc({out_channel}, memory::data_type::f32, memory::format_tag::x);
  auto b_md = memory::desc({batch, out_channel, out_size, out_size}, memory::data_type::f32, memory::format_tag::nchw);

  auto a_mem = memory(a_md, cpu_engine, a);
  auto w_mem = memory(w_md, cpu_engine, w);
  auto bias_mem = memory(bias_md, cpu_engine, bias);

  auto conv_d = convolution_forward::desc(prop_kind::forward_inference,
                                          algorithm::convolution_direct, a_md, w_md, bias_md, b_md,
                                          {1, 1}, {pad, pad}, {pad, pad});
  auto conv_pd = convolution_forward::primitive_desc(conv_d, cpu_engine);
  auto b_mem = memory(conv_pd.dst_desc(), cpu_engine);
  auto task = convolution_forward(conv_pd);
  std::unordered_map<int, memory> args = {{MKLDNN_ARG_SRC,     a_mem},
                                          {MKLDNN_ARG_WEIGHTS, w_mem},
                                          {MKLDNN_ARG_BIAS,    bias_mem},
                                          {MKLDNN_ARG_DST,     b_mem}};

  run(argc, argv, [&](int rep) {
    for (int i = 0; i < rep; ++i) {
      task.execute(cpu_stream, args);
      cpu_stream.wait();
    }
  });

  conv_naive(a, w, b1);
  print_diff((float *) b_mem.get_data_handle(), b1, batch * out_channel * out_size * out_size);
}
