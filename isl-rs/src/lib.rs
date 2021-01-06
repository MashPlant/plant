#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

pub mod aff;
pub mod aff_type;
pub mod ast_build;
pub mod ast;
pub mod ast_type;
pub mod constraint;
pub mod ctx;
pub mod flow;
pub mod id;
pub mod id_to_ast_expr;
pub mod id_to_id;
pub mod id_to_pw_aff;
pub mod ilp;
pub mod local_space;
pub mod lp;
pub mod map;
pub mod map_to_basic_set;
pub mod map_type;
pub mod mat;
pub mod maybe_ast_expr;
pub mod maybe_basic_set;
pub mod maybe_id;
pub mod maybe_pw_aff;
pub mod obj;
pub mod point;
pub mod polynomial;
pub mod polynomial_type;
pub mod printer;
pub mod printer_type;
pub mod schedule_node;
pub mod schedule;
pub mod schedule_type;
pub mod set;
pub mod space;
pub mod stream;
pub mod union_map;
pub mod union_map_type;
pub mod union_set;
pub mod val;
pub mod vec;
pub mod version;
pub mod vertices;

pub use aff::*;
pub use aff_type::*;
pub use ast_build::*;
pub use ast::*;
pub use ast_type::*;
pub use constraint::*;
pub use ctx::*;
pub use flow::*;
pub use id::*;
pub use id_to_ast_expr::*;
pub use id_to_id::*;
pub use id_to_pw_aff::*;
pub use ilp::*;
pub use local_space::*;
pub use lp::*;
pub use map::*;
pub use map_to_basic_set::*;
pub use map_type::*;
pub use mat::*;
pub use maybe_ast_expr::*;
pub use maybe_basic_set::*;
pub use maybe_id::*;
pub use maybe_pw_aff::*;
pub use obj::*;
pub use point::*;
pub use polynomial::*;
pub use polynomial_type::*;
pub use printer::*;
pub use printer_type::*;
pub use schedule_node::*;
pub use schedule::*;
pub use schedule_type::*;
pub use set::*;
pub use space::*;
pub use stream::*;
pub use union_map::*;
pub use union_map_type::*;
pub use union_set::*;
pub use val::*;
pub use vec::*;
pub use version::*;
pub use vertices::*;

impl Ctx {
  #[inline(always)]
  pub fn new() -> Option<Ctx> { unsafe { isl_ctx_alloc() } }
}

#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum Error { None = 0, Abort, Alloc, Unknown, Internal, Invalid, Quota, Unsupported }

#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum Stat { Error = -1, Ok }

#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum Bool { Error = -1, False, True }

#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum DimType { Cst = 0, Param, In, Out, Div, All }

impl DimType { pub const Set: DimType = DimType::Out; }

#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum Fold { Min = 0, Max, List }

#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum AstOpType { Error = -1, And, AndThen, Or, OrElse, Max, Min, Minus, Add, Sub, Mul, Div, FDivQ, PDivQ, PDivR, ZDivR, Cond, Select, Eq, Le, Lt, Ge, Gt, Call, Access, Member, AddressOf }

#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum AstExprType { Error = -1, Op, Id, Int }

#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum AstNodeType { Error = -1, For = 1, If, Block, Mark, User }

#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum AstLoopType { Error = -1, Default, Atomic, Unroll, Separate }

#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum ScheduleNodeType { Error = -1, Band, Context, Domain, Expansion, Extension, Filter, Leaf, Guard, Mark, Sequence, Set }

#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum TokenType { Error = -1, Unknown = 256, Value = 257, Ident, Ge, Le, Gt, Lt, Ne, EqEq, LexGe, LexLe, LexGt, LexLt, To, And, Or, Exists, Not, Def, Infty, Nan, Min, Max, Rat, True, False, Ceild, Floord, Mod, String, Map, Aff, Ceil, Floor, Implies, Last }

use std::{os::raw::{c_int, c_uint, c_long, c_ulong, c_double, c_char, c_void}, ptr::{self, NonNull}, mem, fmt};
use libc::FILE;

// a CStr/CString implementation with same layout as `*const c_char`, and use malloc/free to manage CString memory
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct CStr(NonNull<c_char>);

// ISL参数接受Option<CStr>，但Rust不允许实现From<&str> for Option<CStr>，为方便提供这样一个函数
#[inline(always)]
pub fn cstr(s: &str) -> Option<CStr> { Some(CStr::from_str(s)) }

impl CStr {
  #[inline(always)]
  pub unsafe fn from_ptr(ptr: *const c_char) -> CStr { CStr(NonNull::new_unchecked(ptr as _)) }

  // `from_str` is also unsafe, but not marked as unsafe for convenience
  #[inline(always)]
  pub fn from_str(s: &str) -> CStr { unsafe { CStr(NonNull::new_unchecked(s.as_ptr() as _)) } }

  #[inline(always)]
  pub fn as_str<'a>(self) -> &'a str { unsafe { mem::transmute(std::slice::from_raw_parts(self.0.as_ptr(), libc::strlen(self.0.as_ptr()))) } }
}

impl From<&str> for CStr {
  #[inline(always)]
  fn from(s: &str) -> Self { CStr::from_str(s) }
}

impl AsRef<str> for CStr {
  #[inline(always)]
  fn as_ref(&self) -> &str { self.as_str() }
}

impl std::ops::Deref for CStr {
  type Target = str;
  #[inline(always)]
  fn deref(&self) -> &str { self.as_str() }
}

impl fmt::Debug for CStr {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{:?}", self.as_str()) }
}

impl fmt::Display for CStr {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(self.as_str()) }
}

#[repr(transparent)]
pub struct CString(NonNull<c_char>);

impl CString {
  #[inline(always)]
  pub unsafe fn from_ptr(ptr: *mut c_char) -> CString { CString(NonNull::new_unchecked(ptr)) }
}

impl std::ops::Deref for CString {
  type Target = CStr;
  #[inline(always)]
  fn deref(&self) -> &CStr { unsafe { mem::transmute(self) } }
}

impl fmt::Debug for CString {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{:?}", self.as_str()) }
}

impl fmt::Display for CString {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", self.as_str()) }
}

impl Drop for CString {
  #[inline(always)]
  fn drop(&mut self) { unsafe { libc::free(self.0.as_ptr() as _) } }
}

trait To<T> { unsafe fn to(self) -> T; }

impl<T> To<T> for T {
  #[inline(always)]
  unsafe fn to(self) -> T { self }
}

impl To<*const c_char> for &CStr {
  #[inline(always)]
  unsafe fn to(self) -> *const c_char { self.as_ptr() as _ }
}

impl To<Option<CStr>> for *const c_char {
  #[inline(always)]
  unsafe fn to(self) -> Option<CStr> { if self.is_null() { None } else { Some(CStr::from_ptr(self)) } }
}

impl To<Option<CString>> for *mut c_char {
  #[inline(always)]
  unsafe fn to(self) -> Option<CString> { if self.is_null() { None } else { Some(CString::from_ptr(self)) } }
}

impl To<Option<bool>> for Bool {
  #[inline(always)]
  unsafe fn to(self) -> Option<bool> { match self { Bool::Error => None, Bool::False => Some(false), Bool::True => Some(true) } }
}

impl To<Bool> for Option<bool> {
  #[inline(always)]
  unsafe fn to(self) -> Bool { match self { None => Bool::Error, Some(false) => Bool::False, Some(true) => Bool::True } }
}

impl To<Bool> for bool {
  #[inline(always)]
  unsafe fn to(self) -> Bool { if self { Bool::True } else { Bool::False } }
}

impl To<Option<()>> for Stat {
  #[inline(always)]
  unsafe fn to(self) -> Option<()> { match self { Stat::Error => None, Stat::Ok => Some(()) } }
}

impl To<Stat> for Option<()> {
  #[inline(always)]
  unsafe fn to(self) -> Stat { match self { None => Stat::Error, Some(()) => Stat::Ok } }
}

impl To<()> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> () {}
}

impl<T1: To<Option<U1>>, T2: To<Option<U2>>, U1, U2> To<Option<(U1, U2)>> for (T1, T2) {
  #[inline(always)]
  unsafe fn to(self) -> Option<(U1, U2)> { match (self.0.to(), self.1.to()) { (Some(a), Some(b)) => Some((a, b)), _ => None } }
}

impl<T1: To<Option<U1>>, T2: To<Option<U2>>, T3: To<Option<U3>>, U1, U2, U3> To<Option<(U1, U2, U3)>> for (T1, T2, T3) {
  #[inline(always)]
  unsafe fn to(self) -> Option<(U1, U2, U3)> { match (self.0.to(), self.1.to(), self.2.to()) { (Some(a), Some(b), Some(c)) => Some((a, b, c)), _ => None } }
}

impl<T1: To<Option<()>>, T2: To<Option<U2>>, T3: To<Option<U3>>, U2, U3> To<Option<(U2, U3)>> for (T1, T2, T3) {
  #[inline(always)]
  unsafe fn to(self) -> Option<(U2, U3)> { match (self.0.to(), self.1.to(), self.2.to()) { (Some(_), Some(b), Some(c)) => Some((b, c)), _ => None } }
}

impl<T2: To<Option<U2>>, T3: To<Option<U3>>, T4: To<Option<U4>>, T5: To<Option<U5>>, U2, U3, U4, U5> To<Option<(c_int, U2, U3, U4, U5)>> for (c_int, T2, T3, T4, T5) {
  #[inline(always)]
  unsafe fn to(self) -> Option<(c_int, U2, U3, U4, U5)> { match (self.1.to(), self.2.to(), self.3.to(), self.4.to()) { (Some(b), Some(c), Some(d), Some(e)) => Some((self.0, b, c, d, e)), _ => None } }
}

impl<T> To<c_uint> for &mut [T] {
  #[inline(always)]
  unsafe fn to(self) -> c_uint { self.len() as _ }
}

impl<T> To<*mut T> for &mut [T] {
  #[inline(always)]
  unsafe fn to(self) -> *mut T { self.as_mut_ptr() }
}
