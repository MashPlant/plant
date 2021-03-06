typedef char i8;
typedef unsigned char u8;
typedef short i16;
typedef unsigned short u16;
typedef int i32;
typedef unsigned u32;
typedef long long i64;
typedef unsigned long long u64;
typedef float f32;
typedef double f64;
#define max(x, y) ({ __typeof__(x) _x = x; __typeof__(y) _y = y; _x > _y ? _x : _y; })
#define min(x, y) ({ __typeof__(x) _x = x; __typeof__(y) _y = y; _x > _y ? _y : _x; })
#define floord(x, y) ((x) / (y))
#ifdef __CUDACC__
template <typename F, typename... Args> __global__ void exec_kern(F f, Args... args) { f(args...); }
#define cuda_malloc(size) ({ void *_x; cudaMalloc(&_x, size); _x; })
[[noreturn]]  __device__ void unreachable() {}
#define assume(x) if (!(x)) unreachable()
#endif
