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
#define floord(x, y) ({ __typeof__(x) _x = x; __typeof__(y) _y = y; _x < 0 ? -((-_x + _y - 1) / _y) : _x / _y; })
