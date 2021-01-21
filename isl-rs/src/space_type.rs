use crate::*;

extern "C" {
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Space(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct SpaceRef(pub NonNull<c_void>);

impl_try!(Space);
impl_try!(SpaceRef);

impl Space {
  #[inline(always)]
  pub fn read(&self) -> Space { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Space) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<SpaceRef> for Space {
  #[inline(always)]
  fn as_ref(&self) -> &SpaceRef { unsafe { mem::transmute(self) } }
}

impl Deref for Space {
  type Target = SpaceRef;
  #[inline(always)]
  fn deref(&self) -> &SpaceRef { self.as_ref() }
}

impl To<Option<Space>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Space> { NonNull::new(self).map(Space) }
}

