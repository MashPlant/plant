use crate::*;

extern "C" {
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Id(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct IdRef(pub NonNull<c_void>);

impl_try!(Id);
impl_try!(IdRef);

impl Id {
  #[inline(always)]
  pub fn read(&self) -> Id { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Id) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<IdRef> for Id {
  #[inline(always)]
  fn as_ref(&self) -> &IdRef { unsafe { mem::transmute(self) } }
}

impl Deref for Id {
  type Target = IdRef;
  #[inline(always)]
  fn deref(&self) -> &IdRef { self.as_ref() }
}

impl To<Option<Id>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Id> { NonNull::new(self).map(Id) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct IdList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct IdListRef(pub NonNull<c_void>);

impl_try!(IdList);
impl_try!(IdListRef);

impl IdList {
  #[inline(always)]
  pub fn read(&self) -> IdList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: IdList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<IdListRef> for IdList {
  #[inline(always)]
  fn as_ref(&self) -> &IdListRef { unsafe { mem::transmute(self) } }
}

impl Deref for IdList {
  type Target = IdListRef;
  #[inline(always)]
  fn deref(&self) -> &IdListRef { self.as_ref() }
}

impl To<Option<IdList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<IdList> { NonNull::new(self).map(IdList) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MultiId(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct MultiIdRef(pub NonNull<c_void>);

impl_try!(MultiId);
impl_try!(MultiIdRef);

impl MultiId {
  #[inline(always)]
  pub fn read(&self) -> MultiId { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: MultiId) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MultiIdRef> for MultiId {
  #[inline(always)]
  fn as_ref(&self) -> &MultiIdRef { unsafe { mem::transmute(self) } }
}

impl Deref for MultiId {
  type Target = MultiIdRef;
  #[inline(always)]
  fn deref(&self) -> &MultiIdRef { self.as_ref() }
}

impl To<Option<MultiId>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MultiId> { NonNull::new(self).map(MultiId) }
}

