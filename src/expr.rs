use crate::*;

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum Type { I8, U8, I16, U16, I32, U32, I64, U64, F32, F64, Void }

impl Type {
  // 将Expr::Val中的值转化为i64
  pub fn val_i64(self, x: u64) -> i64 {
    match self {
      I8 => x as i8 as _, U8 => x as u8 as _, I16 => x as i16 as _, U16 => x as u16 as _,
      I32 => x as i32 as _, U32 => x as u32 as _, I64 | U64 | Void => x as _, // Void应该是不可能的
      F32 => f32::from_bits(x as _) as _, F64 => f64::from_bits(x) as _,
    }
  }
}

#[derive(Debug, Clone)]
pub enum Expr {
  // 实际存放的值根据Type来，可以表示浮点数
  Val(Type, u64),
  Iter(Type, u32),
  Param(P<Comp>),
  Cast(Type, Box<Expr>),
  Unary(UnOp, Box<Expr>),
  Binary(BinOp, Box<[Expr; 2]>),
  Call(Box<str>, Box<[Expr]>),
  Access(P<Comp>, Box<[Expr]>),
  Load(P<Buf>, Box<[Expr]>),
  Memcpy(P<Buf>, P<Buf>),
  Alloc(P<Buf>),
  Free(P<Buf>),
  Sync,
}

impl_try!(Expr);

// 可用于Func::comp，Comp::at等接受impl Expr的slice的函数，直接传&[]会报错无法推断类型
pub const EMPTY: &[Expr] = &[];
pub const EMPTY2: &[(Expr, Expr)] = &[];

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
      &Val(ty, _) | &Iter(ty, _) | &Cast(ty, _) => ty,
      Param(comp) | Access(comp, _) => comp.expr.ty(),
      Unary(_, x) => x.ty(),
      Binary(op, box [l, r]) => if (Add <= *op && *op <= Rem) || (Max <= *op && *op <= Min) {
        // 这是我随意规定的类型规则，不完全符合C语言的规则
        // 两个整数运算，若位数不同则结果是位数高的类型，若位数相同，有符号和无符号的运算结果是无符号
        // 整数和浮点数运算结果总是浮点数，两个浮点数运算也是取位数高的类型
        l.ty().max(r.ty())
      } else { I32 }
      Load(buf, _) => buf.ty,
      Call(..) | Memcpy(..) | Alloc(..) | Free(..) | Sync => Void,
    }
  }

  pub fn cast(self, to: Type) -> Expr {
    if self.ty() == to { self } else { Cast(to, box self) }
  }

  pub fn args(&self) -> &[Expr] {
    match self {
      Unary(_, x) | Cast(_, x) => std::slice::from_ref(x),
      Binary(_, lr) => lr.as_ref(),
      Call(_, args) | Access(_, args) | Load(_, args) => args,
      Val(..) | Iter(..) | Param(..) | Memcpy(..) | Alloc(..) | Free(..) | Sync => &[],
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

  pub fn from_isl(f: &Func, e: AstExpr) -> Expr {
    match e.get_type() {
      // iter_ty是I32或I64都可以直接用as转换
      AstExprType::Int => Val(f.iter_ty, e.get_val()?.get_num_si() as _),
      AstExprType::Id => {
        let name = e.get_id()?.get_name()?.as_str();
        // from_isl只在代码生成阶段用到，ISL AST中的循环迭代器名字已经被设置成_i0, i0, _i1, i1...的形式
        if name.starts_with("i") {
          Iter(f.iter_ty, name.get(1..)?.parse().ok()?)
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
            Load(buf.into(), idx.into())
          }
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

// 用于实现一系列不能用Rust的operator traits实现的operator，把它放在IntoExpr trait里
// Rust中涉及比较的trait返回值类型都是定死的，不能改成Expr
macro_rules! impl_other {
  ($($op: ident $fn: ident),*) => {
    $(fn $fn(self, rhs: impl IntoExpr) -> Expr { Binary(BinOp::$op, box [self.expr(), rhs.expr()]) })*
  };
}

pub trait IntoExpr: Sized + Clone {
  fn expr(self) -> Expr;

  fn clone_expr(&self) -> Expr { self.clone().expr() }

  // max_，min_是为了和std::cmp::Ord::max做出区分
  impl_other!(LAnd land, LOr lor, Eq eq, Ne ne, Le le, Lt lt, Ge ge, Gt gt, Max max_, Min min_);
}

impl IntoExpr for Expr { fn expr(self) -> Expr { self } }

impl IntoExpr for &Expr { fn expr(self) -> Expr { self.clone() } }

impl IntoExpr for &&Expr { fn expr(self) -> Expr { (*self).clone() } }

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

macro_rules! impl_op {
  ($($op: ident $fn: ident $op_assign: ident $fn_assign: ident),*) => {
    $(impl<R: IntoExpr> std::ops::$op<R> for Expr {
      type Output = Expr;
      fn $fn(self, rhs: R) -> Expr { Binary(BinOp::$op, box [self, rhs.expr()]) }
    }

    impl<R: IntoExpr> std::ops::$op<R> for &Expr {
      type Output = Expr;
      fn $fn(self, rhs: R) -> Expr { Binary(BinOp::$op, box [self.clone(), rhs.expr()]) }
    }

    impl<R: IntoExpr> std::ops::$op_assign<R> for Expr {
      fn $fn_assign(&mut self, rhs: R) { *self = Binary(BinOp::$op, box [self.clone(), rhs.expr()]) }
    })*
  };
}

impl_op!(Add add AddAssign add_assign, Sub sub SubAssign sub_assign,
  Mul mul MulAssign mul_assign, Div div DivAssign div_assign, Rem rem RemAssign rem_assign);
