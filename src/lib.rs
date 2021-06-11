#![feature(box_syntax, box_patterns, try_trait, try_trait_v2, control_flow_enum)]

#[macro_use]
extern crate log;

macro_rules! impl_setter {
  ($fn: ident $field: ident $ty: ty) => {
    pub fn $fn(&self, $field: $ty) -> P<Self> {
      self.p().$field = $field;
      self.p()
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

pub use expr::Expr::*;
pub use comp::DimTag::*;
pub use buf::{BufKind::*, BufLoc::*};
pub use Backend::*;
pub use tuner::{Loss::*, TunerPolicy::*};
pub use feature::Feature::*;

pub use isl::*;
pub use tools::P;
pub use plant_macros::*;
pub use plant_runtime::*;

use std::fmt::{*, Result as FmtResult};
use tools::{*, fmt::*};

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

pub const CC: &str = "clang++";
pub const NVCC: &str = "nvcc";
