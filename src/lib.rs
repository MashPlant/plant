#![feature(box_syntax, box_patterns, try_trait, bindings_after_at)]

#[macro_use]
extern crate log;

macro_rules! impl_setter {
  ($fn: ident $field: ident $ty: ty) => {
    pub fn $fn(&self, $field: $ty) -> &Self {
      self.p().$field = $field;
      self
    }
  };
}

pub mod expr;
pub mod comp;
pub mod buf;
pub mod func;
pub mod fmt;
pub mod tuner;
pub mod feature;

pub use expr::*;
pub use comp::*;
pub use buf::*;
pub use func::*;
pub use fmt::*;
pub use tuner::*;
pub use feature::*;

use ptr::*;
use isl::*;
use std::fmt::{*, Result as FmtResult};
pub use expr::{Type::*, Expr::*};
pub use comp::DimTag::*;
pub use buf::{BufKind::*, BufLoc::*};
pub use Backend::*;
pub use tuner::{Loss::*, TunerPolicy::*};
pub use feature::Feature::*;

pub type HashMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<ahash::AHasher>>;
pub type HashSet<K> = std::collections::HashSet<K, std::hash::BuildHasherDefault<ahash::AHasher>>;

// 这两个字符串可以传给init_log作为log的过滤器，也可以传入其它字符串
// 只使用Func，Comp等时推荐使用FUNC_FILTER，输出所有debug log
pub const FUNC_FILTER: &str = "debug";
// 使用了Tuner时推荐使用，Tuner中都是info级别的log，而其它mod中都是debug级别的log
// 同时关闭其它crate的log，因为xgboost会输出一些无用的log
pub const TUNER_FILTER: &str = "off,plant=info";

pub fn init_log(filter: &str) {
  env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(filter))
    .format(|buf, record| {
      use std::io::Write;
      writeln!(buf, "{}:{}:{}: {}", record.level(), record.file().unwrap(), record.line().unwrap(), record.args())
    }).init();
}

// Unit与()完全一样，但是没法为()实现Try，所以定义一个新类型来实现Try
#[derive(Copy, Clone)]
pub struct Unit;
impl_try!(Unit);

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum Backend { C, CUDA }

pub const CC: &str = "clang";
pub const NVCC: &str = "nvcc";

// 虽然有很多开源的随机数实现，但用自己的还是方便一点
#[derive(Debug, Clone, Copy)]
pub struct XorShiftRng(pub u64);

impl XorShiftRng {
  pub fn gen(&self) -> u64 {
    let mut x = self.p().get().0;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    self.p().get().0 = x;
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
      F32 => *(p as *mut f32) = x as u32 as f32 * (1.0 / u32::MAX as f32), F64 => *(p as *mut f64) = x as f64 * (1.0 / u64::MAX as f64),
    }
  }
}
