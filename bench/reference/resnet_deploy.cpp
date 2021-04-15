#include "common.h"
#include <algorithm>
#include <cassert>
#include <dlfcn.h>
#include <fcntl.h>
#include <functional>
#include <optional>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

using u32 = unsigned;
using f32 = float;

void parallel_init(u32);
void parallel_launch(void (*)(void *, u32), void *);

void *load_lib(char *path, size_t len) {
  void *lib = dlopen(path, RTLD_NOW);
  assert(lib && "lib not found");
  *(size_t *) dlsym(lib, "parallel_launch") = (size_t) parallel_launch;
  path[len - 3] = '\0'; // .so -> \0so
  return dlsym(lib, path);
}

int data_dir = open("resnet_data", O_RDONLY);

f32 *load_weight(const char *path) {
  int fd = openat(data_dir, path, O_RDONLY);
  assert(fd != -1 && "file not found");
  struct stat st;
  fstat(fd, &st);
  return (f32 *) mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
}

auto conv(u32 ic, u32 oc, u32 size, u32 kern, u32 stride, u32 pad, u32 add, u32 relu) {
  char buf[128];
  size_t len = sprintf(buf, "ic%d_oc%d_size%d_kern%d_stride%d_pad%d_add%d_relu%d.so", ic, oc, size, kern, stride, pad, add, relu);
  void *f = load_lib(buf, len);
  static u32 id = 0;
  len = sprintf(buf, "conv%d_w", id++);
  f32 *w = load_weight(buf);
  buf[len - 1] = 'b';
  f32 *bias = load_weight(buf);
  u32 osize = (size - kern + 2 * pad) / stride + 1;
  f32 *b = alloc(oc * osize * osize);
  return std::pair([=](f32 *i, f32 *add) {
    if (add) {
      ((void (*)(f32 *, f32 *, f32 *, f32 *, f32 *)) f)(i, w, bias, add, b);
    } else {
      ((void (*)(f32 *, f32 *, f32 *, f32 *)) f)(i, w, bias, b);
    }
  }, b);
}

auto maxpool(u32 chan, u32 size, u32 kern, u32 stride, u32 pad) {
  char buf[] = "maxpool.so";
  void *f = load_lib(buf, sizeof(buf) - 1);
  u32 osize = (size - kern + 2 * pad) / stride + 1;
  f32 *b = alloc(chan * osize * osize);
  return std::pair([=](f32 *i) { ((void (*)(f32 *, f32 *)) f)(i, b); }, b);
}

auto avgpool(u32 chan, u32 size) {
  char buf[] = "avgpool.so";
  void *f = load_lib(buf, sizeof(buf) - 1);
  f32 *b = alloc(chan * size * size);
  return std::pair([=](f32 *i) { ((void (*)(f32 *, f32 *)) f)(i, b); }, b);
}

auto gemv(u32 m, u32 n) {
  char buf[] = "gemv.so";
  void *f = load_lib(buf, sizeof(buf) - 1);
  f32 *w = load_weight("gemv_w");
  f32 *c = load_weight("gemv_b");
  f32 *b = alloc(m);
  return std::pair([=](f32 *i) { ((void (*)(f32 *, f32 *, f32 *, f32 *)) f)(i, w, c, b); }, b);
}

std::pair<std::function<void(f32 *)>, f32 *> block(u32 inplanes, u32 planes, u32 size, u32 stride, bool bottleneck) {
  u32 expansion = bottleneck ? 4 : 1;
  bool downsample = stride != 1 || inplanes != planes * expansion;
  if (bottleneck) {
    auto f1 = conv(inplanes, planes, size, 1, stride, 0, 0, 1);
    auto f2 = conv(planes, planes, size / stride, 3, 1, 1, 0, 1);
    auto f3 = conv(planes, planes * expansion, size / stride, 1, 1, 0, 1, 1);
    auto f4 = downsample ? std::optional(conv(inplanes, planes * expansion, size, 1, stride, 0, 0, 0)) : std::nullopt;
    return std::pair([=](f32 *i) {
      if (f4) { f4->first(i, nullptr); }
      f1.first(i, nullptr);
      f2.first(f1.second, nullptr);
      f3.first(f2.second, f4 ? f4->second : i);
    }, f3.second);
  } else {
    auto f1 = conv(inplanes, planes, size, 3, stride, 1, 0, 1);
    auto f2 = conv(planes, planes, size / stride, 3, 1, 1, 1, 1);
    auto f3 = downsample ? std::optional(conv(inplanes, planes * expansion, size, 1, stride, 0, 0, 0)) : std::nullopt;
    return std::pair([=](f32 *i) {
      if (f3) { f3->first(i, nullptr); }
      f1.first(i, nullptr);
      f2.first(f1.second, f3 ? f3->second : i);
    }, f2.second);
  }
}

auto layer(u32 inplanes, u32 planes, u32 blocks, u32 size, u32 stride, bool bottleneck) {
  u32 expansion = bottleneck ? 4 : 1;
  std::vector < std::pair < std::function < void(f32 * ) > , f32 * >> layers;
  layers.reserve(blocks);
  layers.push_back(block(inplanes, planes, size, stride, bottleneck));
  for (u32 _ = 1; _ < blocks; ++_) {
    layers.push_back(block(planes * expansion, planes, size / stride, 1, bottleneck));
  }
  f32 *b = layers.back().second;
  return std::pair([=, layers{std::move(layers)}](f32 *i) {
    for (auto &f : layers) { f.first(i), i = f.second; }
  }, b);
}

void softmax(f32 *p, u32 n) {
  f32 m = *std::max_element(p, p + n);
  f32 s = 0.0;
  for (u32 i = 0; i < n; ++i) { s += (p[i] = exp(p[i] - m)); }
  for (u32 i = 0; i < n; ++i) { p[i] /= s; }
}

int main(int argc, char **argv) {
  parallel_init(0);

  if (argc != 3) { puts("usage: cargo run --bin resnet <layer> <repeat>"), exit(1); }
  u32 repeat = atoi(argv[2]);
  std::array<u32, 4> blocks;
  bool bottleneck;
  std::string_view size(argv[1]);
  if (size == "18") { blocks = {2, 2, 2, 2}, bottleneck = false; }
  else if (size == "34") { blocks = {3, 4, 6, 3}, bottleneck = false; }
  else if (size == "50") { blocks = {3, 4, 6, 3}, bottleneck = true; }
  else if (size == "101") { blocks = {3, 4, 23, 3}, bottleneck = true; }
  else if (size == "152") { blocks = {3, 8, 36, 3}, bottleneck = true; }
  else { printf("expect 1st argument to be [18, 34, 50, 101, 152], found %s\n", argv[1]), exit(1); }

  u32 expansion = bottleneck ? 4 : 1;
  f32 *input = load_weight("input");
  auto f1 = conv(3, 64, 224, 7, 2, 3, 0, 1);
  auto f2 = maxpool(64, 112, 3, 2, 1);
  auto f3 = layer(64, 64, blocks[0], 56, 1, bottleneck);
  auto f4 = layer(64 * expansion, 128, blocks[1], 56, 2, bottleneck);
  auto f5 = layer(128 * expansion, 256, blocks[2], 28, 2, bottleneck);
  auto f6 = layer(256 * expansion, 512, blocks[3], 14, 2, bottleneck);
  auto f7 = avgpool(512 * expansion, 7);
  auto f8 = gemv(1000, 512 * expansion);

  for (u32 _ = 0; _ < 4; ++_) {
    double beg = get_time();
    for (u32 _ = 0; _ < repeat; ++_) {
      f1.first(input, nullptr);
      f2.first(f1.second);
      f3.first(f2.second);
      f4.first(f3.second);
      f5.first(f4.second);
      f6.first(f5.second);
      f7.first(f6.second);
      f8.first(f7.second);
    }
    printf("%.5lfs\n", (get_time() - beg) / repeat);
  }

  softmax(f8.second, 1000);
  std::pair<u32, f32> result[1000];
  for (u32 i = 0; i < 1000; ++i) { result[i] = {i, f8.second[i]}; }
  std::sort(result, result + 1000, [](auto l, auto r) { return l.second > r.second; });
  for (u32 i = 0; i < 5; ++i) { printf("class = %d, prob = %f\n", result[i].first, result[i].second); }
}
