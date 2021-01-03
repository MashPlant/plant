use crate::*;

extern "C" {
  pub fn isl_id_to_ast_expr_alloc(ctx: CtxRef, min_size: c_int) -> Option<IdToAstExpr>;
  pub fn isl_id_to_ast_expr_copy(hmap: IdToAstExprRef) -> Option<IdToAstExpr>;
  pub fn isl_id_to_ast_expr_free(hmap: IdToAstExpr) -> *mut c_void;
  pub fn isl_id_to_ast_expr_get_ctx(hmap: IdToAstExprRef) -> Option<CtxRef>;
  pub fn isl_id_to_ast_expr_try_get(hmap: IdToAstExprRef, key: IdRef) -> MaybeIslAstExpr;
  pub fn isl_id_to_ast_expr_has(hmap: IdToAstExprRef, key: IdRef) -> Bool;
  pub fn isl_id_to_ast_expr_get(hmap: IdToAstExprRef, key: Id) -> Option<AstExpr>;
  pub fn isl_id_to_ast_expr_set(hmap: IdToAstExpr, key: Id, val: AstExpr) -> Option<IdToAstExpr>;
  pub fn isl_id_to_ast_expr_drop(hmap: IdToAstExpr, key: Id) -> Option<IdToAstExpr>;
  pub fn isl_id_to_ast_expr_foreach(hmap: IdToAstExprRef, fn_: unsafe extern "C" fn(key: Id, val: AstExpr, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_printer_print_id_to_ast_expr(p: Printer, hmap: IdToAstExprRef) -> Option<Printer>;
  pub fn isl_id_to_ast_expr_dump(hmap: IdToAstExprRef) -> ();
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MaybeIslAstExpr(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct MaybeIslAstExprRef(pub NonNull<c_void>);

impl MaybeIslAstExpr {
  #[inline(always)]
  pub fn read(&self) -> MaybeIslAstExpr { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: MaybeIslAstExpr) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MaybeIslAstExprRef> for MaybeIslAstExpr {
  #[inline(always)]
  fn as_ref(&self) -> &MaybeIslAstExprRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for MaybeIslAstExpr {
  type Target = MaybeIslAstExprRef;
  #[inline(always)]
  fn deref(&self) -> &MaybeIslAstExprRef { self.as_ref() }
}

impl To<Option<MaybeIslAstExpr>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MaybeIslAstExpr> { NonNull::new(self).map(MaybeIslAstExpr) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct IdToAstExpr(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct IdToAstExprRef(pub NonNull<c_void>);

impl IdToAstExpr {
  #[inline(always)]
  pub fn read(&self) -> IdToAstExpr { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: IdToAstExpr) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<IdToAstExprRef> for IdToAstExpr {
  #[inline(always)]
  fn as_ref(&self) -> &IdToAstExprRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for IdToAstExpr {
  type Target = IdToAstExprRef;
  #[inline(always)]
  fn deref(&self) -> &IdToAstExprRef { self.as_ref() }
}

impl To<Option<IdToAstExpr>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<IdToAstExpr> { NonNull::new(self).map(IdToAstExpr) }
}

impl CtxRef {
  #[inline(always)]
  pub fn id_to_ast_expr_alloc(self, min_size: c_int) -> Option<IdToAstExpr> {
    unsafe {
      let ret = isl_id_to_ast_expr_alloc(self.to(), min_size.to());
      (ret).to()
    }
  }
}

impl IdToAstExpr {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_id_to_ast_expr_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set(self, key: Id, val: AstExpr) -> Option<IdToAstExpr> {
    unsafe {
      let ret = isl_id_to_ast_expr_set(self.to(), key.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, key: Id) -> Option<IdToAstExpr> {
    unsafe {
      let ret = isl_id_to_ast_expr_drop(self.to(), key.to());
      (ret).to()
    }
  }
}

impl IdToAstExprRef {
  #[inline(always)]
  pub fn copy(self) -> Option<IdToAstExpr> {
    unsafe {
      let ret = isl_id_to_ast_expr_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_id_to_ast_expr_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn try_get(self, key: IdRef) -> MaybeIslAstExpr {
    unsafe {
      let ret = isl_id_to_ast_expr_try_get(self.to(), key.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has(self, key: IdRef) -> Option<bool> {
    unsafe {
      let ret = isl_id_to_ast_expr_has(self.to(), key.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get(self, key: Id) -> Option<AstExpr> {
    unsafe {
      let ret = isl_id_to_ast_expr_get(self.to(), key.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(Id, AstExpr) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(Id, AstExpr) -> Option<()>>(key: Id, val: AstExpr, user: *mut c_void) -> Stat { (*(user as *mut F))(key.to(), val.to()).to() }
    unsafe {
      let ret = isl_id_to_ast_expr_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_id_to_ast_expr_dump(self.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_id_to_ast_expr(self, hmap: IdToAstExprRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_id_to_ast_expr(self.to(), hmap.to());
      (ret).to()
    }
  }
}

impl Drop for IdToAstExpr {
  fn drop(&mut self) { IdToAstExpr(self.0).free() }
}

