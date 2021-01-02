use crate::*;

extern "C" {
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MaybeIslAstExpr(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
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

