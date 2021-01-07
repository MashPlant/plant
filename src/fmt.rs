use crate::{*, UnOp::*, BinOp::*};

pub fn fn2display(f: impl Fn(&mut Formatter) -> FmtResult) -> impl Display {
  struct S<F>(F);
  impl<F: Fn(&mut Formatter) -> FmtResult> Display for S<F> {
    fn fmt(&self, f: &mut Formatter) -> FmtResult { (self.0)(f) }
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

impl Type {
  pub fn as_str(self) -> &'static str {
    match self {
      I8 => "i8", U8 => "u8", I16 => "i16", U16 => "u16",
      I32 => "i32", U32 => "u32", I64 => "i64", U64 => "u64",
      F32 => "f32", F64 => "f64", Ptr => "void *"
    }
  }
}

impl UnOp {
  pub fn as_str(self) -> &'static str {
    match self {
      Floor => "floor", Ceil => "ceil", Round => "round", Trunc => "trunc",
      Sin => "sin", Cos => "cos", Tan => "tan", Abs => "abs", Sqrt => "sqrt", Exp => "exp", Log => "log",
    }
  }
}

impl BinOp {
  pub fn as_str(self) -> &'static str {
    match self {
      Add => "+", Sub => "-", Mul => "*", Div => "/", Rem => "%",
      LAnd => "&&", LOr => "||", Eq => "==", Ne => "!=", Le => "<=", Lt => "<", Ge => ">=", Gt => ">",
      Max => "max", Min => "min", Memcpy => "memcpy", // 以call格式输出
    }
  }
}

impl Display for Type {
  fn fmt(&self, f: &mut Formatter) -> FmtResult { f.write_str(self.as_str()) }
}

impl Display for UnOp {
  fn fmt(&self, f: &mut Formatter) -> FmtResult { f.write_str(self.as_str()) }
}

impl Display for BinOp {
  fn fmt(&self, f: &mut Formatter) -> FmtResult { f.write_str(self.as_str()) }
}

// 在Func::comp，Func::set_constraint，access_to_load，Comp::store中将表达式传递给ISL，但其中很多写法ISL并不支持，用户自己负责
impl Display for Expr {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    match self {
      &Val(ty, x) => match ty {
        I8 => write!(f, "{}", x as i8),
        U8 => write!(f, "{}", x as u8),
        I16 => write!(f, "{}", x as i16),
        U16 => write!(f, "{}", x as u16),
        I32 => write!(f, "{}", x as i32),
        U32 => write!(f, "{}", x as u32),
        I64 => write!(f, "{}", x as i64),
        U64 | Ptr => write!(f, "{}", x), // Ptr应该是不可能的
        F32 => write!(f, "{}", f32::from_bits(x as _)),
        F64 => write!(f, "{}", f64::from_bits(x)),
      },
      &Iter(_, x) => write!(f, "i{}", x),
      &Param(x) => f.write_str(x.name()),
      Cast(ty, x) => write!(f, "({})({})", ty, x),
      Unary(op, x) => write!(f, "{}({})", op, x),
      Binary(op, box [l, r]) =>
        if *op >= Max { write!(f, "{}({}, {})", op, l, r) } else { write!(f, "({} {} {})", l, op, r) },
      Call(x, args) => write!(f, "{}({})", x, comma_sep(args.iter())),
      Access(x, args) => write!(f, "{}[{}]", x.name(), comma_sep(args.iter())),
      Load(buf, idx) => {
        let first = idx.first().expect("empty index");
        write!(f, "{}[", buf.name)?;
        for _ in 2..idx.len() { f.write_str("(")?; }
        write!(f, "{}{}]", first, sep(idx.iter().zip(buf.sizes.iter()).skip(1)
          .map(|(idx, size)| fn2display(move |f| write!(f, "*{}+{}", size, idx))), ")"))
      }
      Alloc(x) => write!(f, "allocate({})", x),
      Free(x) => write!(f, "free({})", x),
    }
  }
}
