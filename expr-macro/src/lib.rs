#![feature(box_syntax, box_patterns)]

use quote::ToTokens;
use proc_macro::TokenStream;
use proc_macro2::Span;
use syn::{*, spanned::Spanned, punctuated::Punctuated};
use std::{ptr, sync::atomic::{AtomicUsize, Ordering}};

macro_rules! punct {
  ($($e: expr),*) => {{
    let mut p = Punctuated::new();
    $(p.push($e);)*
    p
  }};
}

macro_rules! d { () => { <_>::default() }; }

#[proc_macro]
pub fn c(input: TokenStream) -> TokenStream {
  static COMP_CNT: AtomicUsize = AtomicUsize::new(0);
  fn err() -> ! { panic!("expect for loop in form `for i in 0..ub for/expr`") }
  let mut f = parse_macro_input!(input as ExprForLoop);
  let span = f.span();
  let mut iters = Vec::new();
  loop {
    let iter = if let Pat::Ident(x) = f.pat { x.ident.to_string() } else { err() };
    let ub = match *f.expr {
      Expr::Range(ExprRange {
        from: Some(box Expr::Lit(ExprLit { lit: Lit::Int(f), .. })),
        to: Some(t), limits: RangeLimits::HalfOpen(_), ..
      }) if f.base10_digits() == "0" => *t,
      _ => err(),
    };
    iters.push((iter, ub));
    if f.body.stmts.len() != 1 { err(); }
    match f.body.stmts.remove(0) {
      Stmt::Expr(Expr::ForLoop(f1)) => f = f1,
      Stmt::Expr(e) => {
        let id = COMP_CNT.fetch_add(1, Ordering::Relaxed);
        let mut args = Punctuated::new();
        for (idx, (_, ub)) in iters.iter().enumerate() {
          args.push(expr(ub.clone(), &iters[..idx]));
        }
        return method_call(d!(), ident("f", span), "comp", span, punct![
          ExprLit { attrs: d!(), lit: LitStr::new(&format!("__generated_comp{}", id), span).into() }.into(),
          method_call(d!(), ExprArray { attrs: d!(), bracket_token: d!(), elems: args }.into(), "into", span, d!()),
          expr(e, &iters)
        ]).into_token_stream().into();
      }
      _ => err(),
    }
  }
}

#[proc_macro]
pub fn x(input: TokenStream) -> TokenStream {
  if let Ok(e) = parse(input.clone()) { expr(e, &[]) } else {
    array(parse_macro_input!(input with Punctuated::parse_terminated), &[])
  }.into_token_stream().into()
}

fn expr(e: Expr, iters: &[(String, Expr)]) -> Expr {
  use Expr::*;
  match e {
    Binary(x) => {
      use BinOp::*;
      let op = match x.op {
        Add(_) => "add", Sub(_) => "sub", Mul(_) => "mul", Div(_) => "div", Rem(_) => "rem",
        And(_) => "land", Or(_) => "lor",
        Eq(_) => "eq", Lt(_) => "lt", Le(_) => "le", Ne(_) => "ne", Ge(_) => "ge", Gt(_) => "gt",
        _ => panic!("unknown op {:?}", x.op),
      };
      method_call(x.attrs, expr(*x.left, iters), op, x.op.span(), punct![expr(*x.right, iters)])
    }
    Block(mut x) => {
      assert_eq!(x.block.stmts.len(), 1, "block expr size must be 1");
      match x.block.stmts.pop() {
        Some(Stmt::Expr(x)) => expr(x, iters),
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
          args: punct![ty(ret), ExprLit { attrs: d!(), lit: LitStr::new(&f.to_string(), span).into() }.into(), array(x.args, iters)],
        }.into()
      } else {
        method_call(x.attrs, *x.func, "at", span, punct![array(x.args, iters)])
      }
    }
    Cast(x) => method_call(x.attrs, expr(*x.expr, iters), "cast", x.as_token.span, punct![ty(&x.ty)]),
    If(x) => method_call(x.attrs, expr(*x.cond, iters), "select", x.if_token.span,
      punct![expr(ExprBlock { attrs: d!(), label: d!(), block: x.then_branch }.into(), iters),
        expr(*x.else_branch.expect("if expr must have else branch").1, iters)]),
    Lit(x) => {
      let span = x.span();
      method_call(x.attrs.clone(), x.into(), "expr", span, d!())
    }
    Paren(x) => if let Paren(_) = *x.expr {
      // 随便设计的一个语法，((e))会忽略e的结构，调用(e).expr()
      // 这里必须加括号，否则可能产生a + b.expr()这样的表达式，和一般理解的AST不一样
      let span = x.span();
      method_call(x.attrs, *x.expr, "expr", span, d!())
    } else {
      ExprParen {
        attrs: x.attrs,
        paren_token: x.paren_token,
        expr: box expr(*x.expr, iters),
      }.into()
    }
    Path(x) => {
      let id = x.path.get_ident().expect("path expr must be single variable name").to_string();
      let span = x.span();
      let it_str;
      let it = if let Some(i) = iters.iter().position(|(i, _)| &id == i) {
        it_str = format!("{}", i);
        Some(it_str.as_str())
      } else if id.starts_with("i") && id[1..].parse::<u32>().is_ok() { Some(&id[1..]) } else { None };
      if let Some(it) = it {
        ExprCall {
          attrs: x.attrs,
          func: box ident("iter", span),
          paren_token: d!(),
          args: punct![ExprLit { attrs: d!(), lit: LitInt::new(it, span).into() }.into()],
        }.into()
      } else {
        method_call(x.attrs.clone(), x.into(), "expr", span, d!())
      }
    }
    Unary(x) => {
      let op = match x.op { UnOp::Not(_) => "not", UnOp::Neg(_) => "neg", _ => panic!("unknown op {:?}", x.op) };
      method_call(x.attrs, expr(*x.expr, iters), op, x.op.span(), d!())
    }
    _ => panic!("unknown expr {:?}", e),
  }
}

fn ident(s: &str, span: Span) -> Expr {
  ExprPath { attrs: d!(), qself: d!(), path: Ident::new(s, span).into() }.into()
}

fn array(mut idx: Punctuated<Expr, Token![,]>, iters: &[(String, Expr)]) -> Expr {
  for e in &mut idx { unsafe { ptr::write(e, expr(ptr::read(e), iters)); } }
  method_call(d!(), ExprArray { attrs: d!(), bracket_token: d!(), elems: idx }.into(),
    "into", Span::call_site(), d!())
}

fn method_call(attrs: Vec<Attribute>, receiver: Expr, s: &str, span: Span, args: Punctuated<Expr, Token![,]>) -> Expr {
  ExprMethodCall {
    attrs,
    receiver: box receiver,
    dot_token: d!(),
    method: Ident::new(s, span),
    turbofish: d!(),
    paren_token: d!(),
    args,
  }.into()
}

fn ty(ty: &Type) -> Expr {
  macro_rules! err { () => { panic!("unknown type {:?}", ty) }; }
  match ty {
    Type::Tuple(x) if x.elems.is_empty() => ident("Void", x.span()),
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
