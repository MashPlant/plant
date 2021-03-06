use crate::*;

extern "C" {
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MaybeIslBasicSet(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct MaybeIslBasicSetRef(pub NonNull<c_void>);

impl_try!(MaybeIslBasicSet);
impl_try!(MaybeIslBasicSetRef);

impl MaybeIslBasicSet {
  #[inline(always)]
  pub fn read(&self) -> MaybeIslBasicSet { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: MaybeIslBasicSet) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MaybeIslBasicSetRef> for MaybeIslBasicSet {
  #[inline(always)]
  fn as_ref(&self) -> &MaybeIslBasicSetRef { unsafe { mem::transmute(self) } }
}

impl Deref for MaybeIslBasicSet {
  type Target = MaybeIslBasicSetRef;
  #[inline(always)]
  fn deref(&self) -> &MaybeIslBasicSetRef { self.as_ref() }
}

impl To<Option<MaybeIslBasicSet>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MaybeIslBasicSet> { NonNull::new(self).map(MaybeIslBasicSet) }
}

