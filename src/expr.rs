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
pub enum ExprKind {
  // 实际存放的值根据Expr::ty来，可以表示浮点数
  Val(u64),
  Iter(u32),
  Param(Box<str>),
  Unary(UnOp, Box<Expr>),
  Binary(BinOp, Box<[Expr; 2]>),
  Call(Box<str>, Box<[Expr]>),
  Access(P<Comp>, Box<[Expr]>),
  Load(P<Buf>, Box<[Expr]>),
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
      Call(_, args) | Access(_, args) | Load(_, args) => args,
      Val(..) | Iter(..) | Param(..) | Alloc(..) | Free(..) => &[],
    }
  }

  pub fn args_mut(&mut self) -> &mut [Expr] {
    match &mut self.1 {
      Unary(_, x) => std::slice::from_mut(x),
      Binary(_, lr) => lr.as_mut(),
      Call(_, args) | Access(_, args) | Load(_, args) => args,
      Val(..) | Iter(..) | Param(..) | Alloc(..) | Free(..) => &mut [],
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

// 用于实现一系列不能用Rust的operator traits实现的operator，把它放在IntoExpr trait里
// Rust中涉及比较的trait返回值类型都是定死的，不能改成Expr
macro_rules! impl_other {
  ($($op: ident $fn: ident),*) => {
    $(fn $fn(self, rhs: impl IntoExpr) -> Expr {
      let l = self.expr();
      Expr(l.0, Binary(BinOp::$op, box [l, rhs.expr()]))
    })*
  };
}

pub trait IntoExpr: Sized + Clone {
  fn expr(self) -> Expr;

  fn clone_expr(&self) -> Expr { self.clone().expr() }

  impl_other!(LAnd land, LOr lor, Eq eq, Ne ne, Le le, Lt lt, Ge ge, Gt gt);
}

impl IntoExpr for Expr { fn expr(self) -> Expr { self } }

impl IntoExpr for &Expr { fn expr(self) -> Expr { self.clone() } }

macro_rules! impl_primitive {
  ($($val: ident $ty: ident),*) => {
    $(impl IntoExpr for $ty { fn expr(self) -> Expr { Expr($val, Val(self as _)) } })*
  };
}

impl_primitive!(U8 u8, U16 u16, U32 u32, U64 u64, I8 i8, I16 i16, I32 i32, I64 i64, Bool bool);

impl IntoExpr for f32 { fn expr(self) -> Expr { Expr(F32, Val(self.to_bits() as _)) } }

impl IntoExpr for f64 { fn expr(self) -> Expr { Expr(F64, Val(self.to_bits())) } }

macro_rules! impl_op {
  ($($op: ident $fn: ident),*) => {
    $(impl<R: IntoExpr> std::ops::$op<R> for Expr {
      type Output = Expr;
      fn $fn(self, rhs: R) -> Expr { Expr(self.0, Binary(BinOp::$op, box [self, rhs.expr()])) }
    }

    impl<R: IntoExpr> std::ops::$op<R> for &Expr {
      type Output = Expr;
      fn $fn(self, rhs: R) -> Expr { Expr(self.0, Binary(BinOp::$op, box [self.clone(), rhs.expr()])) }
    })*
  };
}

impl_op!(Add add, Sub sub, Mul mul, Div div, Rem rem);
