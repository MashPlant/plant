#include <cstdlib>
using i8 = char;
using u8 = unsigned char;
using i16 = short;
using u16 = unsigned short;
using i32 = int;
using u32 = unsigned;
using i64 = long long;
using u64 = unsigned long long;
using f32 = float;
using f64 = double;
#define max(x, y) ({ auto _x = x; auto _y = y; _x > _y ? _x : _y; })
#define min(x, y) ({ auto _x = x; auto _y = y; _x > _y ? _y : _x; })
#define floord(x, y) ((x) / (y))
#define vec(t, n) t __attribute__((vector_size(n * sizeof(t))))
extern "C" { void (*parallel_launch)(void (*)(void *, i32, i32), void *); }
#ifdef __CUDACC__
template <typename F> __global__ void exec_kern(F f) { f(); }
#define cuda_malloc(size) ({ void *_x; cudaMalloc(&_x, size); _x; })
[[noreturn]] __device__ void unreachable() {}
#define assume(x) if (!(x)) unreachable()
#endif
