use std::{mem, sync::atomic::{AtomicU8, Ordering}};
use crate::*;

static ITER_TY: AtomicU8 = AtomicU8::new(I32 as _);

pub fn iter_ty() -> Type { unsafe { mem::transmute(ITER_TY.load(Ordering::Relaxed)) } }

pub fn set_iter_ty(ty: Type) { ITER_TY.store(ty as _, Ordering::Relaxed) }

pub fn iter(i: u32) -> Expr { Iter(iter_ty(), i) }

#[derive(Debug, Clone)]
pub enum Expr {
  // 实际存放的值根据Type来，可以表示浮点数
  Val(Type, u64),
  Iter(Type, u32),
  Param(P<Comp>),
  Cast(Type, Box<Expr>),
  Unary(UnOp, Box<Expr>),
  Binary(BinOp, Box<[Expr; 2]>),
  // :?操作符
  Select(Box<[Expr; 3]>),
  // 用Box包一层以减小整个enum的大小
  Call(Box<Call>),
  Access(P<Comp>, Box<[Expr]>),
  // idx表示成(i0 * size1) + i1 ...的形式
  Load(P<Buf>, Box<Expr>),
  Memcpy(P<Buf>, P<Buf>),
  Alloc(P<Buf>),
  Free(P<Buf>),
  Ramp(Type, i32, u32, Box<Expr>),
  // 一段直接用于输出的字符串，不考虑它的内部结构
  Verbatim(Type, R<str>),
}

impl_try!(Expr);

#[derive(Debug, Clone)]
pub struct Call {
  pub ret: Type,
  pub name: R<str>,
  pub args: Box<[Expr]>,
}

pub fn call(ret: Type, name: &str, args: Box<[Expr]>) -> Expr {
  Call(box Call { ret, name: name.into(), args })
}

pub fn syncthreads() -> Expr { Verbatim(Void, "__syncthreads()".r()) }

// 逻辑非用x != 0表示，取负用0 - x表示，不在Unary中提供
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum UnOp { Floor, Ceil, Round, Trunc, Sin, Cos, Tan, Abs, Sqrt, Exp, Log }

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum BinOp { Add, Sub, Mul, Div, Rem, LAnd, LOr, Eq, Ne, Le, Lt, Ge, Gt, Max, Min }

// 用在Expr::visit和visit_mut中，返回true表示继续访问Expr的children，否则不访问
pub trait VisitChildren { fn visit(self) -> bool; }

impl VisitChildren for () { fn visit(self) -> bool { true } }

impl VisitChildren for bool { fn visit(self) -> bool { self } }

impl Expr {
  pub fn ty(&self) -> Type {
    use BinOp::*;
    match self {
      &Val(ty, _) | &Iter(ty, _) | &Cast(ty, _) | &Call(box Call { ret: ty, .. }) | &Ramp(ty, ..) | &Verbatim(ty, _) => ty,
      Param(comp) | Access(comp, _) => comp.expr.ty(),
      Unary(_, x) => x.ty(),
      Binary(op, box [l, _]) => if (Add <= *op && *op <= Rem) || (Max <= *op && *op <= Min) { l.ty() } else { I32 }
      Select(box [_, t, _]) => t.ty(),
      Load(buf, _) => buf.ty,
      Memcpy(..) | Alloc(..) | Free(..) => Void,
    }
  }

  pub fn cast(self, to: Type) -> Expr {
    if self.ty() == to { self } else { Cast(to, box self) }
  }

  pub fn args(&self) -> &[Expr] {
    match self {
      Unary(_, x) | Cast(_, x) | Load(_, x) | Ramp(.., x) => std::slice::from_ref(x),
      Binary(_, x) => x.as_ref(),
      Select(x) => x.as_ref(),
      Call(box Call { args, .. }) | Access(_, args) => args,
      Val(..) | Iter(..) | Param(..) | Memcpy(..) | Alloc(..) | Free(..) | Verbatim(..) => &[],
    }
  }

  pub fn args_mut(&mut self) -> &mut [Expr] { self.args().p().get() }

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

  pub fn modify<R: VisitChildren>(&self, mut f: impl FnMut(&mut Expr) -> R) -> Expr {
    let mut e = self.clone();
    e.visit_mut(&mut f);
    e
  }

  pub fn replace_iter(&mut self, iter_map: &[Expr]) {
    self.visit_mut(&mut |e| if let Iter(_, x) = e {
      *e = iter_map[*x as usize].clone();
      false // 停止访问，iter_map中的Iter不应该再被替换
    } else { true })
  }

  // 将自身替换为0，返回自身原来的值，可以为无用但无法移动的Expr省去一次拷贝
  pub fn replace0(&mut self) -> Expr { mem::replace(self, Val(I8, 0)) }

  pub(crate) fn collect_params(&self, params: &mut HashSet<P<Comp>>) {
    self.visit(&mut |e| if let &Param(x) = e { params.insert(x); })
  }

  pub fn from_isl(f: &Func, e: AstExpr) -> Expr {
    match e.get_type() {
      // iter_ty是I32或I64都可以直接用as转换
      AstExprType::Int => Val(iter_ty(), e.get_val()?.get_num_si() as _),
      AstExprType::Id => {
        let name = e.get_id()?.get_name()?.as_str();
        // from_isl只在代码生成阶段用到，ISL AST中的循环迭代器名字已经被设置成_i0, i0, _i1, i1...的形式
        if name.starts_with("i") {
          Iter(iter_ty(), name.get(1..)?.parse().ok()?)
        } else {
          Param(f.find_comp(name)?.into())
        }
      }
      AstExprType::Op => {
        use AstExprOpType::*;
        let (n, op) = (e.get_op_n_arg(), e.get_op_type());
        match e.get_op_type() {
          Access => {
            // 不使用处理AstExprType::Id的逻辑
            let name = e.get_op_arg(0)?.get_id()?.get_name()?.as_str();
            let buf = f.find_buf(name)?;
            let mut idx = Vec::with_capacity(n as usize - 1);
            for i in 1..n { idx.push(Expr::from_isl(f, e.get_op_arg(i)?)); }
            buf.at(idx.into())
          }
          Cond | Select => Expr::Select(box [Expr::from_isl(f, e.get_op_arg(0)?),
            Expr::from_isl(f, e.get_op_arg(1)?), Expr::from_isl(f, e.get_op_arg(2)?)]),
          _ => {
            let op0 = Expr::from_isl(f, e.get_op_arg(0)?);
            let op = match op {
              Max => BinOp::Max, Min => BinOp::Min, Add => BinOp::Add, Sub => BinOp::Sub, Mul => BinOp::Mul,
              FDivQ => BinOp::Div, PDivQ => BinOp::Div, PDivR => BinOp::Rem, ZDivR => BinOp::Rem,
              And => BinOp::LAnd, Or => BinOp::LOr, Eq => BinOp::Eq, Le => BinOp::Le, Lt => BinOp::Lt, Ge => BinOp::Ge, Gt => BinOp::Gt,
              Minus => return Binary(BinOp::Sub, box [0.expr(), op0]),
              _ => debug_panic!("invalid expr op: {:?}", op),
            };
            let op1 = Expr::from_isl(f, e.get_op_arg(1)?);
            Binary(op, box [op0, op1])
          }
        }
      }
      ty => debug_panic!("invalid expr type: {:?}", ty),
    }
  }
}

macro_rules! impl_ops {
  ($($op: ident $fn: ident),*) => {
    $(fn $fn(self, rhs: impl IntoExpr) -> Expr { Binary(BinOp::$op, box [self.expr(), rhs.expr()]) })*
  };
}

pub trait IntoExpr: Sized + Clone {
  fn expr(self) -> Expr;

  // max_，min_是为了和std::cmp::Ord::max做出区分
  impl_ops!(Add add, Sub sub, Mul mul, Div div, Rem rem,
    LAnd land, LOr lor, Eq eq, Ne ne, Le le, Lt lt, Ge ge, Gt gt, Max max_, Min min_);

  fn neg(self) -> Expr { Binary(BinOp::Sub, box [0.expr(), self.expr()]) }

  fn not(self) -> Expr { Binary(BinOp::Ne, box [0.expr(), self.expr()]) }

  fn select(self, t: impl IntoExpr, f: impl IntoExpr) -> Expr { Select(box [self.expr(), t.expr(), f.expr()]) }
}

impl IntoExpr for Expr { fn expr(self) -> Expr { self } }

impl IntoExpr for &Expr { fn expr(self) -> Expr { self.clone() } }

macro_rules! impl_primitive {
  ($($val: ident $ty: ident),*) => {
    $(
      impl IntoExpr for $ty { fn expr(self) -> Expr { Val($val, self as _) } }
      impl IntoExpr for &$ty { fn expr(self) -> Expr { Val($val, *self as _) } }
    )*
  };
}

impl_primitive!(U8 u8, U16 u16, U32 u32, U64 u64, I8 i8, I16 i16, I32 i32, I64 i64);

impl IntoExpr for f32 { fn expr(self) -> Expr { Val(F32, self.to_bits() as _) } }

impl IntoExpr for f64 { fn expr(self) -> Expr { Val(F64, self.to_bits()) } }
