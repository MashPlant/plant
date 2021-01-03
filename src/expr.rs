use isl::{AstExpr, AstExprType, AstOpType};
use crate::*;

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum Type { U8, U16, U32, U64, I8, I16, I32, I64, F32, F64, Bool, Ptr }

impl Type {
  // 将ExprKind::Val中的值转化为i64
  pub fn val_i64(self, x: u64) -> i64 {
    match self {
      U8 => x as u8 as _, U16 => x as u16 as _, U32 => x as u32 as _, U64 | Ptr => x as _,
      I8 => x as i8 as _, I16 => x as i16 as _, I32 => x as i32 as _, I64 => x as i64 as _,
      F32 => f32::from_bits(x as _) as _,
      F64 => f64::from_bits(x) as _,
      Bool => (x != 0) as _,
    }
  }
}

#[derive(Debug, Clone)]
pub struct Expr(pub Type, pub ExprKind);

// 可用于Func::comp，Comp::at等接受impl Expr的slice的函数，直接传&[]会报错无法推断类型
pub const EMPTY: &[Expr] = &[];
pub const EMPTY2: &[(Expr, Expr)] = &[];

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum UnOp { LNot, Cast, Floor, Ceil, Round, Trunc, Sin, Cos, Tan, Abs, Sqrt, Exp, Log }

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum BinOp { Add, Sub, Mul, Div, Rem, LAnd, LOr, Eq, Ne, Le, Lt, Ge, Gt, Max, Min, Memcpy }

#[derive(Debug, Clone)]
pub enum ExprKind {
  // 实际存放的值根据Expr::ty来，可以表示浮点数
  Val(u64),
  Iter(u32),
  // R<str>一般引用Comp::name
  Param(R<str>),
  Unary(UnOp, Box<Expr>),
  Binary(BinOp, Box<[Expr; 2]>),
  Call(Box<str>, Box<[Expr]>),
  Access(P<Comp>, Box<[Expr]>),
  Load(P<Buf>, Box<[Expr]>),
  Alloc(Box<str>),
  Free(Box<str>),
}

// 用在Expr::visit和visit_mut中，返回true表示继续访问Expr的children，否则不访问
pub trait VisitChildren { fn visit(self) -> bool; }

impl VisitChildren for () { fn visit(self) -> bool { true } }

impl VisitChildren for bool { fn visit(self) -> bool { self } }

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

  pub fn visit<'a, R: VisitChildren>(&'a self, f: &mut impl FnMut(&'a Expr) -> R) {
    if f(self).visit() {
      for x in self.args() { x.visit(f); }
    }
  }

  pub fn visit_mut<R: VisitChildren>(&mut self, f: &mut impl FnMut(&mut Expr) -> R) {
    if f(self).visit() {
      for x in self.args_mut() { x.visit_mut(f); }
    }
  }

  pub fn from_isl(f: &Func, e: AstExpr) -> Option<Expr> {
    Some(match e.get_type() {
      // iter_ty是I32或I64都可以直接用as转换
      AstExprType::Int => Expr(f.iter_ty, Val(e.get_val()?.get_num_si() as _)),
      AstExprType::Id => {
        let name = e.get_id()?.get_name()?.as_str();
        // from_isl只在代码生成阶段用到，ISL AST中的循环迭代器名字已经被设置成_i0, i0, _i1, i1...的形式
        Expr(f.iter_ty, if name.starts_with("i") {
          Iter(name.get(1..)?.parse().ok()?)
        } else {
          // 严格来说param不一定是iter类型，但这种情形没有意义，param一般用来表示非仿射访问的下标
          Param(name.into())
        })
      }
      AstExprType::Op => {
        use AstOpType::*;
        let (n, op) = (e.get_op_n_arg(), e.get_op_type());
        let op0 = Expr::from_isl(f, e.get_op_arg(0)?)?;
        match e.get_op_type() {
          Minus => Expr(op0.0, Binary(BinOp::Sub, box [0.expr(), op0])),
          Access => {
            // 不使用op0和处理AstExprType::Id的逻辑
            let name = e.get_op_arg(0)?.get_id()?.get_name()?.as_str();
            let buf = f.find_buf(name)?;
            let mut idx = Vec::with_capacity(n as usize - 1);
            for i in 1..n { idx.push(Expr::from_isl(f, e.get_op_arg(i)?)?); }
            Expr(buf.ty, Load(buf, idx.into()))
          }
          _ => {
            let op1 = Expr::from_isl(f, e.get_op_arg(1)?)?;
            let op = match op {
              Max => BinOp::Max, Min => BinOp::Min, Add => BinOp::Add, Sub => BinOp::Sub, Mul => BinOp::Mul,
              FDivQ => BinOp::Div, PDivQ => BinOp::Div, PDivR => BinOp::Rem, ZDivR => BinOp::Rem,
              Eq => BinOp::Eq, Le => BinOp::Le, Lt => BinOp::Lt, Ge => BinOp::Ge, Gt => BinOp::Gt,
              _ => return None,
            };
            Expr(op0.0, Binary(op, box [op0, op1]))
          }
        }
      }
      _ => return None,
    })
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
