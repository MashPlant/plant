use crate::*;

extern "C" {
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Val(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct ValRef(pub NonNull<c_void>);

impl_try!(Val);
impl_try!(ValRef);

impl Val {
  #[inline(always)]
  pub fn read(&self) -> Val { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Val) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<ValRef> for Val {
  #[inline(always)]
  fn as_ref(&self) -> &ValRef { unsafe { mem::transmute(self) } }
}

impl Deref for Val {
  type Target = ValRef;
  #[inline(always)]
  fn deref(&self) -> &ValRef { self.as_ref() }
}

impl To<Option<Val>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Val> { NonNull::new(self).map(Val) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct ValList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct ValListRef(pub NonNull<c_void>);

impl_try!(ValList);
impl_try!(ValListRef);

impl ValList {
  #[inline(always)]
  pub fn read(&self) -> ValList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: ValList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<ValListRef> for ValList {
  #[inline(always)]
  fn as_ref(&self) -> &ValListRef { unsafe { mem::transmute(self) } }
}

impl Deref for ValList {
  type Target = ValListRef;
  #[inline(always)]
  fn deref(&self) -> &ValListRef { self.as_ref() }
}

impl To<Option<ValList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<ValList> { NonNull::new(self).map(ValList) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MultiVal(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct MultiValRef(pub NonNull<c_void>);

impl_try!(MultiVal);
impl_try!(MultiValRef);

impl MultiVal {
  #[inline(always)]
  pub fn read(&self) -> MultiVal { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: MultiVal) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MultiValRef> for MultiVal {
  #[inline(always)]
  fn as_ref(&self) -> &MultiValRef { unsafe { mem::transmute(self) } }
}

impl Deref for MultiVal {
  type Target = MultiValRef;
  #[inline(always)]
  fn deref(&self) -> &MultiValRef { self.as_ref() }
}

impl To<Option<MultiVal>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MultiVal> { NonNull::new(self).map(MultiVal) }
}

