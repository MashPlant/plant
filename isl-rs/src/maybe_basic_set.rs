use crate::*;

extern "C" {
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MaybeIslBasicSet(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct MaybeIslBasicSetRef(pub NonNull<c_void>);

impl MaybeIslBasicSet {
  #[inline(always)]
  pub fn read(&self) -> MaybeIslBasicSet { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&mut self, x: MaybeIslBasicSet) { unsafe { ptr::write(self, x) } }
}

impl AsRef<MaybeIslBasicSetRef> for MaybeIslBasicSet {
  #[inline(always)]
  fn as_ref(&self) -> &MaybeIslBasicSetRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for MaybeIslBasicSet {
  type Target = MaybeIslBasicSetRef;
  #[inline(always)]
  fn deref(&self) -> &MaybeIslBasicSetRef { self.as_ref() }
}

impl To<Option<MaybeIslBasicSet>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MaybeIslBasicSet> { NonNull::new(self).map(MaybeIslBasicSet) }
}

