#![feature(box_syntax, box_patterns)]

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
use isl::cstr;
pub use expr::{Type::*, Expr::*};
pub use comp::DimTag::*;
pub use buf::BufKind::*;

use std::hash::BuildHasherDefault;
use ahash::AHasher;

pub type HashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<AHasher>>;
pub type HashSet<K> = std::collections::HashSet<K, BuildHasherDefault<AHasher>>;

pub fn init_log() {
  env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
    .format(|buf, record| {
      use std::io::Write;
      writeln!(buf, "{}:{}:{}: {}", record.level(), record.file().unwrap(), record.line().unwrap(), record.args())
    }).init();
}

pub enum Backend { C, CUDA }
