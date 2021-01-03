use crate::*;

extern "C" {
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MaybeIslId(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct MaybeIslIdRef(pub NonNull<c_void>);

impl MaybeIslId {
  #[inline(always)]
  pub fn read(&self) -> MaybeIslId { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: MaybeIslId) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MaybeIslIdRef> for MaybeIslId {
  #[inline(always)]
  fn as_ref(&self) -> &MaybeIslIdRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for MaybeIslId {
  type Target = MaybeIslIdRef;
  #[inline(always)]
  fn deref(&self) -> &MaybeIslIdRef { self.as_ref() }
}

impl To<Option<MaybeIslId>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MaybeIslId> { NonNull::new(self).map(MaybeIslId) }
}

