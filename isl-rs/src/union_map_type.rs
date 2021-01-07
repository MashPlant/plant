use crate::*;

extern "C" {
}

#[repr(transparent)]
#[derive(Debug)]
pub struct UnionMap(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct UnionMapRef(pub NonNull<c_void>);

impl_try!(UnionMap);
impl_try!(UnionMapRef);

impl UnionMap {
  #[inline(always)]
  pub fn read(&self) -> UnionMap { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: UnionMap) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<UnionMapRef> for UnionMap {
  #[inline(always)]
  fn as_ref(&self) -> &UnionMapRef { unsafe { mem::transmute(self) } }
}

impl Deref for UnionMap {
  type Target = UnionMapRef;
  #[inline(always)]
  fn deref(&self) -> &UnionMapRef { self.as_ref() }
}

impl To<Option<UnionMap>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<UnionMap> { NonNull::new(self).map(UnionMap) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct UnionMapList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct UnionMapListRef(pub NonNull<c_void>);

impl_try!(UnionMapList);
impl_try!(UnionMapListRef);

impl UnionMapList {
  #[inline(always)]
  pub fn read(&self) -> UnionMapList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: UnionMapList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<UnionMapListRef> for UnionMapList {
  #[inline(always)]
  fn as_ref(&self) -> &UnionMapListRef { unsafe { mem::transmute(self) } }
}

impl Deref for UnionMapList {
  type Target = UnionMapListRef;
  #[inline(always)]
  fn deref(&self) -> &UnionMapListRef { self.as_ref() }
}

impl To<Option<UnionMapList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<UnionMapList> { NonNull::new(self).map(UnionMapList) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct UnionSet(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct UnionSetRef(pub NonNull<c_void>);

impl_try!(UnionSet);
impl_try!(UnionSetRef);

impl UnionSet {
  #[inline(always)]
  pub fn read(&self) -> UnionSet { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: UnionSet) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<UnionSetRef> for UnionSet {
  #[inline(always)]
  fn as_ref(&self) -> &UnionSetRef { unsafe { mem::transmute(self) } }
}

impl Deref for UnionSet {
  type Target = UnionSetRef;
  #[inline(always)]
  fn deref(&self) -> &UnionSetRef { self.as_ref() }
}

impl To<Option<UnionSet>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<UnionSet> { NonNull::new(self).map(UnionSet) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct UnionSetList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct UnionSetListRef(pub NonNull<c_void>);

impl_try!(UnionSetList);
impl_try!(UnionSetListRef);

impl UnionSetList {
  #[inline(always)]
  pub fn read(&self) -> UnionSetList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: UnionSetList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<UnionSetListRef> for UnionSetList {
  #[inline(always)]
  fn as_ref(&self) -> &UnionSetListRef { unsafe { mem::transmute(self) } }
}

impl Deref for UnionSetList {
  type Target = UnionSetListRef;
  #[inline(always)]
  fn deref(&self) -> &UnionSetListRef { self.as_ref() }
}

impl To<Option<UnionSetList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<UnionSetList> { NonNull::new(self).map(UnionSetList) }
}

