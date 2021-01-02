use std::fmt::{Display, Formatter, Result};
use crate::{*, UnOp::*, BinOp::*};

pub fn fn2display(f: impl Fn(&mut Formatter) -> Result) -> impl Display {
  struct S<F>(F);
  impl<F: Fn(&mut Formatter) -> Result> Display for S<F> {
    fn fmt(&self, f: &mut Formatter) -> Result { (self.0)(f) }
  }
  S(f)
}

pub fn sep<T: Display>(it: impl Iterator<Item=T> + Clone, sep: &'static str) -> impl Display {
  fn2display(move |f| {
    let mut first = true;
    for t in it.clone() {
      write!(f, "{}{}", if first { "" } else { sep }, t)?;
      first = false;
    }
    Ok(())
  })
}

pub fn comma_sep<T: Display>(it: impl Iterator<Item=T> + Clone) -> impl Display { sep(it, ", ") }

impl UnOp {
  pub fn op_str(self) -> &'static str {
    match self {
      LNot => "!", Cast => "cast",
      Floor => "floor", Ceil => "ceil", Round => "round", Trunc => "trunc",
      Sin => "sin", Cos => "cos", Tan => "tan", Abs => "abs", Sqrt => "sqrt", Exp => "exp", Log => "log",
    }
  }
}

impl BinOp {
  pub fn op_str(self) -> &'static str {
    match self {
      Add => "+", Sub => "-", Mul => "*", Div => "/", Rem => "%",
      LAnd => "&&", LOr => "||", Eq => "==", Ne => "!=", Le => "<=", Lt => "<", Ge => ">=", Gt => ">",
      Max => "max", Min => "min", Memcpy => "memcpy", // 以call格式输出
    }
  }
}

// 这个实现主要是为了上下界传给ISL，但其中很多写法其实ISL并不支持
impl Display for Expr {
  fn fmt(&self, f: &mut Formatter) -> Result {
    match &self.1 {
      &Val(x) => match self.0 {
        U8 => write!(f, "{}", x as u8),
        U16 => write!(f, "{}", x as u16),
        U32 => write!(f, "{}", x as u32),
        U64 | Ptr => write!(f, "{}", x), // Ptr应该是不可能的
        I8 => write!(f, "{}", x as i8),
        I16 => write!(f, "{}", x as i16),
        I32 => write!(f, "{}", x as i32),
        I64 => write!(f, "{}", x as i64),
        F32 => write!(f, "{}", f32::from_bits(x as _)),
        F64 => write!(f, "{}", f64::from_bits(x)),
        Bool => write!(f, "{}", x != 0),
      },
      Iter(x) => write!(f, "i{}", x),
      Param(x) => f.pad(x),
      Unary(op, x) =>
        if *op == Cast { write!(f, "cast({:?}, {})", self.0, x) } else { write!(f, "({}({}))", op.op_str(), x) }
      Binary(op, box [l, r]) => {
        let s = op.op_str();
        if *op >= Max { write!(f, "{}({}, {})", s, l, r) } else { write!(f, "({} {} {})", l, s, r) }
      }
      Call(x, args) => write!(f, "{}({})", x, comma_sep(args.iter())),
      Access(x, args) => write!(f, "{}[{}]", x.name(), comma_sep(args.iter())),
      Load(x, args) => write!(f, "{}[{}]", x.name, comma_sep(args.iter())),
      Alloc(x) => write!(f, "allocate({})", x),
      Free(x) => write!(f, "free({})", x),
    }
  }
}
