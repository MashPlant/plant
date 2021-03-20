#include <cstdlib>
#ifdef __CUDACC__
template <typename F, typename... Args> __global__ void exec_kern(F f, Args... args) { f(args...); }
#define cuda_malloc(size) ({ void *_x; cudaMalloc(&_x, size); _x; })
[[noreturn]] __device__ void unreachable() {}
#define assume(x) if (!(x)) unreachable()
using i8x1 = char1; using u8x1 = uchar1; using i8x2 = char2; using u8x2 = uchar2; using i8x3 = char3; using u8x3 = uchar3; using i8x4 = char4; using u8x4 = uchar4;
using i16x1 = short1; using u16x1 = ushort1; using i16x2 = short2; using u16x2 = ushort2; using i16x3 = short3; using u16x3 = ushort3; using i16x4 = short4; using u16x4 = ushort4;
using i32x1 = int1; using u32x1 = uint1; using i32x2 = int2; using u32x2 = uint2; using i32x3 = int3; using u32x3 = uint3; using i32x4 = int4; using u32x4 = uint4;
using i64x1 = longlong1; using u64x1 = ulonglong1; using i64x2 = longlong2; using u64x2 = ulonglong2; using i64x3 = longlong3; using u64x3 = ulonglong3; using i64x4 = longlong4; using u64x4 = ulonglong4;
using f32x1 = float1; using f64x1 = double1; using f32x2 = float2; using f64x2 = double2; using f32x3 = float3; using f64x3 = double3; using f32x4 = float4; using f64x4 = double4;
#include <helper_math.h>
#else
#define assume(x) __builtin_assume(x)
#endif
using i8 = char; using u8 = unsigned char; using i16 = short; using u16 = unsigned short; using i32 = int; using u32 = unsigned; using i64 = long long; using u64 = unsigned long long; using f32 = float; using f64 = double;
#define max(x, y) ({ auto _x = x; auto _y = y; _x > _y ? _x : _y; })
#define min(x, y) ({ auto _x = x; auto _y = y; _x > _y ? _y : _x; })
#define floord(x, y) ((x) / (y))
extern "C" { void (*parallel_launch)(void (*)(void *, u32), void *); }
