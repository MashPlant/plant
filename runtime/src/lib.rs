pub mod array;

pub use array::*;
pub use Backend::*;
pub use Type::*;

use std::{fmt::{*, Result as FmtResult}, sync::Once};

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum Backend { CPU, GPU }

impl Default for Backend { fn default() -> Self { CPU } }

#[repr(u8)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum Type { I8, U8, I16, U16, I32, U32, I64, U64, F32, F64, Void }

impl Type {
  pub fn size(self) -> usize {
    match self {
      I8 | U8 => 1, I16 | U16 => 2,
      I32 | U32 | F32 => 4, I64 | U64 | Void | F64 => 8, // Void应该是不可能的
    }
  }

  // 将Expr::Val中的值转化为i64
  pub fn val_i64(self, x: u64) -> i64 {
    match self {
      I8 => x as i8 as _, U8 => x as u8 as _, I16 => x as i16 as _, U16 => x as u16 as _,
      I32 => x as i32 as _, U32 => x as u32 as _, I64 | U64 | Void => x as _, // Void应该是不可能的
      F32 => f32::from_bits(x as _) as _, F64 => f64::from_bits(x) as _,
    }
  }

  pub fn as_str(self) -> &'static str {
    match self {
      I8 => "i8", U8 => "u8", I16 => "i16", U16 => "u16",
      I32 => "i32", U32 => "u32", I64 => "i64", U64 => "u64",
      F32 => "f32", F64 => "f64", Void => "void"
    }
  }
}

impl Display for Type {
  fn fmt(&self, f: &mut Formatter) -> FmtResult { f.write_str(self.as_str()) }
}

// 虽然有很多开源的随机数实现，但用自己的还是方便一点
#[derive(Debug, Clone, Copy)]
pub struct XorShiftRng(pub u64);

impl XorShiftRng {
  pub fn gen(&self) -> u64 {
    let mut x = self.0;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    unsafe { *(&self.0 as *const _ as *mut _) = x; }
    x
  }

  // 返回0.0~1.0间的浮点数
  pub fn gen_f32(&self) -> f32 {
    self.gen() as u32 as f32 * (1.0 / u32::MAX as f32)
  }

  pub unsafe fn fill(&self, ty: Type, p: *mut u8) {
    let x = self.gen();
    match ty {
      I8 | U8 => *p = x as _, I16 | U16 => *(p as *mut u16) = x as _,
      I32 | U32 => *(p as *mut u32) = x as _, I64 | U64 | Void => *(p as *mut u64) = x as _, // Void应该是不可能的
      F32 => *(p as *mut f32) = x as u32 as f32 * (1.0 / u32::MAX as f32) - 0.5,
      F64 => *(p as *mut f64) = x as f64 * (1.0 / u64::MAX as f64) - 0.5,
    }
  }
}

// 返回parallel线程数，在调用parallel_init前是0
pub fn parallel_num_thread() -> u32 {
  extern "C" { static num_thread: u32; }
  unsafe { num_thread }
}

// 传0则自动检测系统核心数，传非0值则配置线程数为参数值
// 用户需保证调用parallel_launch前调用了parallel_init
pub fn parallel_init(th: u32) {
  extern "C" { fn parallel_init(th: u32); }
  unsafe { parallel_init(th); }
}

// 允许多次调用，但至少调用一次
pub fn parallel_init_default() {
  static INIT: Once = Once::new();
  INIT.call_once(|| parallel_init(0));
}

extern "C" { pub fn parallel_launch(f: extern "C" fn(*mut u8, u32), args: *mut u8); }
