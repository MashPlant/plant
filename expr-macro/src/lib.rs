#![feature(box_syntax)]

use quote::ToTokens;
use proc_macro::TokenStream;
use proc_macro2::Span;
use syn::{*, spanned::Spanned, punctuated::Punctuated};

#[proc_macro]
pub fn x(input: TokenStream) -> TokenStream {
  if let Ok(e) = parse(input.clone()) { expr(e) } else {
    array(parse_macro_input!(input with Punctuated::parse_terminated))
  }.into_token_stream().into()
}

fn expr(e: Expr) -> Expr {
  macro_rules! punct {
    ($($e: expr),*) => {{
      let mut p = Punctuated::new();
      $(p.push($e);)*
      p
    }};
  }
  use Expr::*;
  match e {
    Assign(x) => {
      // 随便设计的一个语法，_ = e会忽略e的结构，调用(e).expr()
      // 这里必须加括号，否则可能产生a + b.expr()这样的表达式，看起来和一般理解的AST不一样
      let span = x.span();
      method_call(x.attrs, ExprParen { attrs: <_>::default(), paren_token: <_>::default(), expr: x.right }.into(),
        "expr", span, <_>::default())
    }
    Binary(x) => {
      use BinOp::*;
      let op = match x.op {
        Add(_) => "add", Sub(_) => "sub", Mul(_) => "mul", Div(_) => "div", Rem(_) => "rem",
        And(_) => "land", Or(_) => "lor",
        Eq(_) => "eq", Lt(_) => "lt", Le(_) => "le", Ne(_) => "ne", Ge(_) => "ge", Gt(_) => "gt",
        _ => panic!("unknown op {:?}", x.op),
      };
      method_call(x.attrs, expr(*x.left), op, x.op.span(), punct![expr(*x.right)])
    }
    Block(mut x) => {
      assert_eq!(x.block.stmts.len(), 1, "block expr size must be 1");
      match x.block.stmts.pop() {
        Some(Stmt::Expr(x)) => expr(x),
        _ => panic!("block expr size must be 1"),
      }
    }
    Call(x) => {
      let span = x.span();
      let f = x.func.as_ref();
      if let Some((f, ret)) = (|| match f {
        Path(f) if f.path.segments.len() == 1 => {
          let seg = &f.path.segments[0];
          match &seg.arguments {
            PathArguments::AngleBracketed(arg) if arg.args.len() == 1 =>
              if let GenericArgument::Type(ty) = &arg.args[0] { Some((&seg.ident, ty)) } else { None },
            _ => None,
          }
        }
        _ => None,
      })() {
        ExprCall {
          attrs: x.attrs,
          func: box ident("call", span),
          paren_token: x.paren_token,
          args: punct![ty(ret), ExprLit { attrs: <_>::default(), lit: LitStr::new(&f.to_string(), span).into() }.into(), array(x.args)],
        }.into()
      } else {
        method_call(x.attrs, *x.func, "at", span, punct![array(x.args)])
      }
    }
    Cast(x) => method_call(x.attrs, expr(*x.expr), "cast", x.as_token.span, punct![ty(&x.ty)]),
    If(x) => method_call(x.attrs, expr(*x.cond), "select", x.if_token.span,
      punct![expr(ExprBlock { attrs: <_>::default(), label: <_>::default(), block: x.then_branch }.into()),
        expr(*x.else_branch.expect("if expr must have else branch").1)]),
    Lit(x) => {
      let span = x.span();
      method_call(x.attrs.clone(), x.into(), "expr", span, <_>::default())
    }
    Paren(x) => ExprParen {
      attrs: x.attrs,
      paren_token: x.paren_token,
      expr: box expr(*x.expr),
    }.into(),
    Path(x) => {
      let id = x.path.get_ident().expect("path expr must be single variable name").to_string();
      let span = x.span();
      if id.starts_with("i") && id[1..].parse::<u32>().is_ok() {
        ExprCall {
          attrs: x.attrs,
          func: box ident("iter", span),
          paren_token: <_>::default(),
          args: punct![ExprLit { attrs: <_>::default(), lit: LitInt::new(&id[1..], span).into() }.into()],
        }.into()
      } else {
        method_call(x.attrs.clone(), x.into(), "expr", span, <_>::default())
      }
    }
    Unary(x) => {
      let op = match x.op { UnOp::Not(_) => "not", UnOp::Neg(_) => "neg", _ => panic!("unknown op {:?}", x.op) };
      method_call(x.attrs, expr(*x.expr), op, x.op.span(), <_>::default())
    }
    _ => panic!("unknown expr {:?}", e),
  }
}

fn ident(s: &str, span: Span) -> Expr {
  ExprPath { attrs: <_>::default(), qself: <_>::default(), path: Ident::new(s, span).into() }.into()
}

fn array(mut idx: Punctuated<Expr, Token![,]>) -> Expr {
  for e in &mut idx { unsafe { std::ptr::write(e, expr(std::ptr::read(e))); } }
  method_call(<_>::default(), ExprArray { attrs: <_>::default(), bracket_token: <_>::default(), elems: idx }.into(),
    "into", Span::call_site(), <_>::default())
}

fn method_call(attrs: Vec<Attribute>, receiver: Expr, s: &str, span: Span, args: Punctuated<Expr, Token![,]>) -> Expr {
  ExprMethodCall {
    attrs,
    receiver: box receiver,
    dot_token: <_>::default(),
    method: Ident::new(s, span),
    turbofish: <_>::default(),
    paren_token: <_>::default(),
    args,
  }.into()
}

fn ty(ty: &Type) -> Expr {
  macro_rules! err { () => { panic!("unknown type {:?}", ty) }; }
  match ty {
    Type::Path(x) => {
      let ty = match x.path.get_ident().unwrap_or_else(|| err!()).to_string().as_str() {
        "i8" => "I8", "u8" => "U8", "i16" => "I16", "u16" => "U16", "i32" => "I32", "u32" => "U32",
        "i64" => "I64", "u64" => "U64", "f32" => "F32", "f64" => "F64", "void" => "Void", _ => err!(),
      };
      ident(ty, x.span())
    }
    _ => err!(),
  }
}
