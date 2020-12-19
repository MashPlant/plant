use crate::*;

extern "C" {
}

#[repr(transparent)]
#[derive(Debug)]
pub struct ObjType(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct ObjTypeRef(pub NonNull<c_void>);

impl ObjType {
  #[inline(always)]
  pub fn read(&self) -> ObjType { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&mut self, x: ObjType) { unsafe { ptr::write(self, x) } }
}

impl AsRef<ObjTypeRef> for ObjType {
  #[inline(always)]
  fn as_ref(&self) -> &ObjTypeRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for ObjType {
  type Target = ObjTypeRef;
  #[inline(always)]
  fn deref(&self) -> &ObjTypeRef { self.as_ref() }
}

impl To<Option<ObjType>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<ObjType> { NonNull::new(self).map(ObjType) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Obj(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct ObjRef(pub NonNull<c_void>);

impl Obj {
  #[inline(always)]
  pub fn read(&self) -> Obj { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&mut self, x: Obj) { unsafe { ptr::write(self, x) } }
}

impl AsRef<ObjRef> for Obj {
  #[inline(always)]
  fn as_ref(&self) -> &ObjRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for Obj {
  type Target = ObjRef;
  #[inline(always)]
  fn deref(&self) -> &ObjRef { self.as_ref() }
}

impl To<Option<Obj>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Obj> { NonNull::new(self).map(Obj) }
}

