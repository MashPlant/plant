use crate::*;

extern "C" {
}

#[repr(transparent)]
#[derive(Debug)]
pub struct ObjType(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct ObjTypeRef(pub NonNull<c_void>);

impl_try!(ObjType);
impl_try!(ObjTypeRef);

impl ObjType {
  #[inline(always)]
  pub fn read(&self) -> ObjType { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: ObjType) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<ObjTypeRef> for ObjType {
  #[inline(always)]
  fn as_ref(&self) -> &ObjTypeRef { unsafe { mem::transmute(self) } }
}

impl Deref for ObjType {
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
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct ObjRef(pub NonNull<c_void>);

impl_try!(Obj);
impl_try!(ObjRef);

impl Obj {
  #[inline(always)]
  pub fn read(&self) -> Obj { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Obj) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<ObjRef> for Obj {
  #[inline(always)]
  fn as_ref(&self) -> &ObjRef { unsafe { mem::transmute(self) } }
}

impl Deref for Obj {
  type Target = ObjRef;
  #[inline(always)]
  fn deref(&self) -> &ObjRef { self.as_ref() }
}

impl To<Option<Obj>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Obj> { NonNull::new(self).map(Obj) }
}

