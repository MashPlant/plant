#![feature(box_syntax, box_patterns, try_trait)]

#[macro_use]
extern crate log;

pub mod expr;
pub mod comp;
pub mod buf;
pub mod func;
pub mod fmt;

pub use expr::*;
pub use comp::*;
pub use buf::*;
pub use func::*;
pub use fmt::*;

use ptr::*;
use isl::*;
use std::fmt::{*, Result as FmtResult};
pub use expr::{Type::*, Expr::*};
pub use comp::DimTag::*;
pub use buf::BufKind::*;

pub type HashMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<ahash::AHasher>>;
pub type HashSet<K> = std::collections::HashSet<K, std::hash::BuildHasherDefault<ahash::AHasher>>;

pub fn init_log() {
  env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
    .format(|buf, record| {
      use std::io::Write;
      writeln!(buf, "{}:{}:{}: {}", record.level(), record.file().unwrap(), record.line().unwrap(), record.args())
    }).init();
}

// Unit与()完全一样，但是没法为()实现Try，所以定义一个新类型来实现Try
#[derive(Copy, Clone)]
pub struct Unit;
impl_try!(Unit);

pub enum Backend { C, CUDA }
