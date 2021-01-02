use crate::*;

extern "C" {
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MaybeIslPwAff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct MaybeIslPwAffRef(pub NonNull<c_void>);

impl MaybeIslPwAff {
  #[inline(always)]
  pub fn read(&self) -> MaybeIslPwAff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: MaybeIslPwAff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MaybeIslPwAffRef> for MaybeIslPwAff {
  #[inline(always)]
  fn as_ref(&self) -> &MaybeIslPwAffRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for MaybeIslPwAff {
  type Target = MaybeIslPwAffRef;
  #[inline(always)]
  fn deref(&self) -> &MaybeIslPwAffRef { self.as_ref() }
}

impl To<Option<MaybeIslPwAff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MaybeIslPwAff> { NonNull::new(self).map(MaybeIslPwAff) }
}

