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

pub fn comma_sep<T: Display>(it: impl Iterator<Item=T> + Clone) -> impl Display { sep(it, ",") }

pub fn opt<'a, T: Display>(x: &'a Option<T>) -> impl Display + 'a {
  fn2display(move |f| if let Some(x) = x { x.fmt(f) } else { f.write_str("None") })
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
      Max => "max", Min => "min", // 以call格式输出
    }
  }
}

impl Display for UnOp {
  fn fmt(&self, f: &mut Formatter) -> FmtResult { f.write_str(self.as_str()) }
}

impl Display for BinOp {
  fn fmt(&self, f: &mut Formatter) -> FmtResult { f.write_str(self.as_str()) }
}

impl Expr {
  // 用于生成代码和传递给ISL，但其中一些写法不是合法的C/ISL语法，用户自己负责
  // pred按照https://en.cppreference.com/w/c/language/operator_precedence，值越大优先级越低
  // 名字用show而非fmt只是为了避免和Display::fmt冲突
  pub fn show<'a>(&'a self, pred: u32) -> impl Display + 'a {
    fn2display(move |f| {
      let p1;
      macro_rules! par {
        ($($arg:tt)*) => {
          if p1 >= pred { f.write_str("(")?; }
          $($arg)*?;
          if p1 >= pred { f.write_str(")")?; }
          return Ok(());
        }
      }
      match self {
        &Val(ty, x) => match ty {
          I8 => Display::fmt(&(x as i8), f),
          U8 => Display::fmt(&(x as u8), f),
          I16 => Display::fmt(&(x as i16), f),
          U16 => Display::fmt(&(x as u16), f),
          I32 => Display::fmt(&(x as i32), f),
          U32 => Display::fmt(&(x as u32), f),
          I64 => Display::fmt(&(x as i64), f),
          U64 | Void => Display::fmt(&x, f), // Void应该是不可能的
          F32 => Display::fmt(&f32::from_bits(x as _), f),
          F64 => Display::fmt(&f64::from_bits(x), f),
        },
        &Iter(_, x) => write!(f, "i{}", x),
        &Param(x) => f.write_str(x.name()),
        Cast(ty, x) => {
          p1 = 2;
          par!(write!(f, "({}){}", ty, x.show(p1)));
        }
        Unary(op, x) => {
          p1 = 2;
          par!(write!(f, "{}{}", op, x.show(p1)));
        }
        Binary(op, box [l, r]) => match *op {
          Div => write!(f, "floord({},{})", l, r),
          Max | Min => write!(f, "{}({},{})", op, l, r),
          op => {
            p1 = match op { Add | Sub => 4, Mul | Rem => 3, LAnd => 11, LOr => 12, Eq | Ne => 7, _ /* Le | Lt | Ge | Gt */ => 6 };
            par!(write!(f, "{}{}{}", l.show(p1), op, r.show(p1)));
          }
        }
        Select(box [cond, t, f1]) => {
          p1 = 13;
          par!(write!(f, "{}?{}:{}", cond, t, f1));
        }
        Call(x) => write!(f, "{}({})", x.name, comma_sep(x.args.iter())),
        Access(x, args) => {
          debug_assert_eq!(x.orig_dim(), args.len() as _);
          write!(f, "{}[{}]", x.name(), comma_sep(args.iter()))
        }
        Load(buf, idx) => write!(f, "{}[{}]", buf.name, idx),
        Memcpy(to, from) => {
          debug_assert!(to.sizes.len() == from.sizes.len() && to.ty == from.ty);
          match (to.loc, from.loc) {
            (Host, Host) => write!(f, "memcpy({},{},{})", to.name, from.name, to.bytes()),
            _ => write!(f, "cudaMemcpy({},{},{},cudaMemcpy{})", to.name, from.name, to.bytes(), match (to.loc, from.loc) {
              (Host, Global) => "DeviceToHost",
              (Global, Host) => "HostToDevice",
              (Global, Global) => "DeviceToDevice",
              _ => debug_panic!("invalid memcpy type"),
            }),
          }
        }
        Alloc(x) => match x.loc {
          Host => {
            write!(f, "{}=({}*)", x.arg(), x.ty)?;
            if let Some(a) = x.align { write!(f, "aligned_alloc({},{})", a, x.bytes()) } else { write!(f, "malloc({})", x.bytes()) }
          }
          Global => write!(f, "{}=({ty}*)cuda_malloc({})", x.arg(), x.bytes(), ty = x.ty),
          Local | Shared => write!(f, "{} {} {}[{}]{}", if x.loc == Shared { "__shared__" } else { "" }, x.ty, x.name, x.elems(),
            fn2display(move |f| if let Some(a) = x.align { write!(f, "__attribute__((aligned({})))", a) } else { Ok(()) })),
        }
        Free(x) => match x.loc {
          Host | Global => write!(f, "{}({})", if x.loc == Host { "free" } else { "cudaFree" }, x.name),
          _ => write!(f, "/*free({})*/", x.name), // 不实际执行free
        }
        Sync => f.write_str("__syncthreads()"),
        Vector(ty, n, x) => {
          match **x { Load(..) => {} _ => debug_panic!("vector operand must be load: {}", x) }
          write!(f, "*(vec({},{})*)&{}", ty, n, x)
        }
        Opaque(_, x) => f.write_str(x),
      }
    })
  }
}

impl Display for Expr {
  fn fmt(&self, f: &mut Formatter) -> FmtResult { self.show(14).fmt(f) }
}

impl DimTag {
  pub fn gpu_idx(self) -> &'static str {
    match self {
      GPUBlockX => "blockIdx.x", GPUBlockY => "blockIdx.y", GPUBlockZ => "blockIdx.z",
      GPUThreadX => "threadIdx.x", GPUThreadY => "threadIdx.y", GPUThreadZ => "threadIdx.z",
      _ => "", // 应该是不可能的
    }
  }
}
