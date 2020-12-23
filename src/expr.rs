use crate::*;

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum Type { U8, U16, U32, U64, I8, I16, I32, I64, F32, F64, Bool, Ptr }

#[derive(Debug, Clone)]
pub struct Expr(pub Type, pub ExprKind);

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum UnOp { LNot, Cast, Floor, Ceil, Round, Trunc, Sin, Cos, Tan, Abs, Sqrt, Exp, Log }

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum BinOp { Add, Sub, Mul, Div, Rem, LAnd, LOr, Eq, Ne, Le, Lt, Ge, Gt, Max, Min, Memcpy }

#[derive(Debug, Clone)]
pub struct Var {
  // 在Expt::Var中，Expr::ty必和Var::ty相等
  pub ty: Type,
  pub name: Box<str>,
  pub range: Box<[Option<Expr>; 2]>
}

impl Var {
  pub fn expr(self) -> Expr { Expr(self.ty, Var(self)) }
}

#[derive(Debug, Clone)]
pub enum ExprKind {
  // 实际存放的值根据Expr::ty来，可以表示浮点数
  Val(u64),
  Var(Var),
  Unary(UnOp, Box<Expr>),
  Binary(BinOp, Box<[Expr; 2]>),
  Call(Box<str>, Box<[Expr]>),
  Access(Box<str>, Box<[Expr]>),
  Alloc(Box<str>),
  Free(Box<str>),
}

impl Expr {
  pub fn cast(self, to: Type) -> Expr {
    if self.0 == to { self } else { Expr(to, Unary(UnOp::Cast, box self)) }
  }

  pub fn args(&self) -> &[Expr] {
    match &self.1 {
      Unary(_, x) => std::slice::from_ref(x),
      Binary(_, lr) => lr.as_ref(),
      Call(_, args) | Access(_, args) => args,
      Val(..) | Var(..) | Alloc(..) | Free(..) => &[],
    }
  }

  pub fn args_mut(&mut self) -> &mut [Expr] {
    match &mut self.1 {
      Unary(_, x) => std::slice::from_mut(x),
      Binary(_, lr) => lr.as_mut(),
      Call(_, args) | Access(_, args) => args,
      Val(..) | Var(..) | Alloc(..) | Free(..) => &mut [],
    }
  }

  pub fn visit<'a>(&'a self, f: &mut impl FnMut(&'a Expr)) {
    f(self);
    for x in self.args() { x.visit(f); }
  }

  pub fn visit_mut(&mut self, f: &mut impl FnMut(&mut Expr)) {
    f(self);
    for x in self.args_mut() { x.visit_mut(f); }
  }
}

macro_rules! impl_primitive {
  ($($val: ident $ty: ident),*) => {
    $(impl From<$ty> for Expr {
      fn from(t: $ty) -> Expr { Expr($val, Val(t as _)) }
    })*
  };
}

impl_primitive!(U8 u8, U16 u16, U32 u32, U64 u64, I8 i8, I16 i16, I32 i32, I64 i64, Bool bool);

impl From<f32> for Expr {
  fn from(t: f32) -> Expr { Expr(F32, Val(t.to_bits() as _)) }
}

impl From<f64> for Expr {
  fn from(t: f64) -> Expr { Expr(F64, Val(t.to_bits())) }
}

macro_rules! impl_op {
  ($($op: ident $fn: ident),*) => {
    $(impl std::ops::$op for Expr {
      type Output = Expr;
      fn $fn(self, rhs: Expr) -> Expr { Expr(self.0, Binary(BinOp::$op, box [self, rhs])) }
    })*
  };
}

impl_op!(Add add, Sub sub, Mul mul, Div div, Rem rem);

macro_rules! impl_other {
  ($($op: ident $fn: ident),*) => {
    impl Expr {
      $(pub fn $fn(self, rhs: Expr) -> Expr { Expr(self.0, Binary(BinOp::$op, box [self, rhs])) })*
    }
  };
}

impl_other!(LAnd land, LOr lor, Eq eq, Ne ne, Le le, Lt lt, Ge ge, Gt gt);
