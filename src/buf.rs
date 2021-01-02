use crate::*;

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum BufKind { In, Out, Temp }

#[derive(Debug)]
pub struct Buf {
  pub name: Box<str>,
  pub ty: Type,
  pub kind: BufKind,
  pub sizes: Vec<Expr>,
}

impl Func {
  pub fn buf(&self, name: &str, ty: Type, kind: BufKind, sizes: &[impl IntoExpr]) -> R<Buf> {
    assert!(self.find_buf(name).is_none());
    let buf = box Buf { name: name.into(), ty, kind, sizes: sizes.iter().map(|e| e.clone_expr()).collect() };
    let ret = R::new(&*buf);
    P::new(self).bufs.push(buf);
    ret
  }
}

impl Buf {
  pub fn at(&self, idx: &[impl IntoExpr]) -> Expr {
    assert_eq!(idx.len(), self.sizes.len());
    Expr(self.ty, Load(self.into(), idx.iter().map(|e| e.clone_expr()).collect()))
  }
}
