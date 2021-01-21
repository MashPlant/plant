use crate::*;

extern "C" {
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Aff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct AffRef(pub NonNull<c_void>);

impl_try!(Aff);
impl_try!(AffRef);

impl Aff {
  #[inline(always)]
  pub fn read(&self) -> Aff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Aff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<AffRef> for Aff {
  #[inline(always)]
  fn as_ref(&self) -> &AffRef { unsafe { mem::transmute(self) } }
}

impl Deref for Aff {
  type Target = AffRef;
  #[inline(always)]
  fn deref(&self) -> &AffRef { self.as_ref() }
}

impl To<Option<Aff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Aff> { NonNull::new(self).map(Aff) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct AffList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct AffListRef(pub NonNull<c_void>);

impl_try!(AffList);
impl_try!(AffListRef);

impl AffList {
  #[inline(always)]
  pub fn read(&self) -> AffList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: AffList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<AffListRef> for AffList {
  #[inline(always)]
  fn as_ref(&self) -> &AffListRef { unsafe { mem::transmute(self) } }
}

impl Deref for AffList {
  type Target = AffListRef;
  #[inline(always)]
  fn deref(&self) -> &AffListRef { self.as_ref() }
}

impl To<Option<AffList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<AffList> { NonNull::new(self).map(AffList) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct PwAff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct PwAffRef(pub NonNull<c_void>);

impl_try!(PwAff);
impl_try!(PwAffRef);

impl PwAff {
  #[inline(always)]
  pub fn read(&self) -> PwAff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: PwAff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<PwAffRef> for PwAff {
  #[inline(always)]
  fn as_ref(&self) -> &PwAffRef { unsafe { mem::transmute(self) } }
}

impl Deref for PwAff {
  type Target = PwAffRef;
  #[inline(always)]
  fn deref(&self) -> &PwAffRef { self.as_ref() }
}

impl To<Option<PwAff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<PwAff> { NonNull::new(self).map(PwAff) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct PwAffList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct PwAffListRef(pub NonNull<c_void>);

impl_try!(PwAffList);
impl_try!(PwAffListRef);

impl PwAffList {
  #[inline(always)]
  pub fn read(&self) -> PwAffList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: PwAffList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<PwAffListRef> for PwAffList {
  #[inline(always)]
  fn as_ref(&self) -> &PwAffListRef { unsafe { mem::transmute(self) } }
}

impl Deref for PwAffList {
  type Target = PwAffListRef;
  #[inline(always)]
  fn deref(&self) -> &PwAffListRef { self.as_ref() }
}

impl To<Option<PwAffList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<PwAffList> { NonNull::new(self).map(PwAffList) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct UnionPwAff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct UnionPwAffRef(pub NonNull<c_void>);

impl_try!(UnionPwAff);
impl_try!(UnionPwAffRef);

impl UnionPwAff {
  #[inline(always)]
  pub fn read(&self) -> UnionPwAff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: UnionPwAff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<UnionPwAffRef> for UnionPwAff {
  #[inline(always)]
  fn as_ref(&self) -> &UnionPwAffRef { unsafe { mem::transmute(self) } }
}

impl Deref for UnionPwAff {
  type Target = UnionPwAffRef;
  #[inline(always)]
  fn deref(&self) -> &UnionPwAffRef { self.as_ref() }
}

impl To<Option<UnionPwAff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<UnionPwAff> { NonNull::new(self).map(UnionPwAff) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct UnionPwAffList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct UnionPwAffListRef(pub NonNull<c_void>);

impl_try!(UnionPwAffList);
impl_try!(UnionPwAffListRef);

impl UnionPwAffList {
  #[inline(always)]
  pub fn read(&self) -> UnionPwAffList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: UnionPwAffList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<UnionPwAffListRef> for UnionPwAffList {
  #[inline(always)]
  fn as_ref(&self) -> &UnionPwAffListRef { unsafe { mem::transmute(self) } }
}

impl Deref for UnionPwAffList {
  type Target = UnionPwAffListRef;
  #[inline(always)]
  fn deref(&self) -> &UnionPwAffListRef { self.as_ref() }
}

impl To<Option<UnionPwAffList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<UnionPwAffList> { NonNull::new(self).map(UnionPwAffList) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MultiAff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct MultiAffRef(pub NonNull<c_void>);

impl_try!(MultiAff);
impl_try!(MultiAffRef);

impl MultiAff {
  #[inline(always)]
  pub fn read(&self) -> MultiAff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: MultiAff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MultiAffRef> for MultiAff {
  #[inline(always)]
  fn as_ref(&self) -> &MultiAffRef { unsafe { mem::transmute(self) } }
}

impl Deref for MultiAff {
  type Target = MultiAffRef;
  #[inline(always)]
  fn deref(&self) -> &MultiAffRef { self.as_ref() }
}

impl To<Option<MultiAff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MultiAff> { NonNull::new(self).map(MultiAff) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct PwMultiAff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct PwMultiAffRef(pub NonNull<c_void>);

impl_try!(PwMultiAff);
impl_try!(PwMultiAffRef);

impl PwMultiAff {
  #[inline(always)]
  pub fn read(&self) -> PwMultiAff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: PwMultiAff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<PwMultiAffRef> for PwMultiAff {
  #[inline(always)]
  fn as_ref(&self) -> &PwMultiAffRef { unsafe { mem::transmute(self) } }
}

impl Deref for PwMultiAff {
  type Target = PwMultiAffRef;
  #[inline(always)]
  fn deref(&self) -> &PwMultiAffRef { self.as_ref() }
}

impl To<Option<PwMultiAff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<PwMultiAff> { NonNull::new(self).map(PwMultiAff) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct PwMultiAffList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct PwMultiAffListRef(pub NonNull<c_void>);

impl_try!(PwMultiAffList);
impl_try!(PwMultiAffListRef);

impl PwMultiAffList {
  #[inline(always)]
  pub fn read(&self) -> PwMultiAffList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: PwMultiAffList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<PwMultiAffListRef> for PwMultiAffList {
  #[inline(always)]
  fn as_ref(&self) -> &PwMultiAffListRef { unsafe { mem::transmute(self) } }
}

impl Deref for PwMultiAffList {
  type Target = PwMultiAffListRef;
  #[inline(always)]
  fn deref(&self) -> &PwMultiAffListRef { self.as_ref() }
}

impl To<Option<PwMultiAffList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<PwMultiAffList> { NonNull::new(self).map(PwMultiAffList) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct UnionPwMultiAff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct UnionPwMultiAffRef(pub NonNull<c_void>);

impl_try!(UnionPwMultiAff);
impl_try!(UnionPwMultiAffRef);

impl UnionPwMultiAff {
  #[inline(always)]
  pub fn read(&self) -> UnionPwMultiAff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: UnionPwMultiAff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<UnionPwMultiAffRef> for UnionPwMultiAff {
  #[inline(always)]
  fn as_ref(&self) -> &UnionPwMultiAffRef { unsafe { mem::transmute(self) } }
}

impl Deref for UnionPwMultiAff {
  type Target = UnionPwMultiAffRef;
  #[inline(always)]
  fn deref(&self) -> &UnionPwMultiAffRef { self.as_ref() }
}

impl To<Option<UnionPwMultiAff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<UnionPwMultiAff> { NonNull::new(self).map(UnionPwMultiAff) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct UnionPwMultiAffList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct UnionPwMultiAffListRef(pub NonNull<c_void>);

impl_try!(UnionPwMultiAffList);
impl_try!(UnionPwMultiAffListRef);

impl UnionPwMultiAffList {
  #[inline(always)]
  pub fn read(&self) -> UnionPwMultiAffList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: UnionPwMultiAffList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<UnionPwMultiAffListRef> for UnionPwMultiAffList {
  #[inline(always)]
  fn as_ref(&self) -> &UnionPwMultiAffListRef { unsafe { mem::transmute(self) } }
}

impl Deref for UnionPwMultiAffList {
  type Target = UnionPwMultiAffListRef;
  #[inline(always)]
  fn deref(&self) -> &UnionPwMultiAffListRef { self.as_ref() }
}

impl To<Option<UnionPwMultiAffList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<UnionPwMultiAffList> { NonNull::new(self).map(UnionPwMultiAffList) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MultiPwAff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct MultiPwAffRef(pub NonNull<c_void>);

impl_try!(MultiPwAff);
impl_try!(MultiPwAffRef);

impl MultiPwAff {
  #[inline(always)]
  pub fn read(&self) -> MultiPwAff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: MultiPwAff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MultiPwAffRef> for MultiPwAff {
  #[inline(always)]
  fn as_ref(&self) -> &MultiPwAffRef { unsafe { mem::transmute(self) } }
}

impl Deref for MultiPwAff {
  type Target = MultiPwAffRef;
  #[inline(always)]
  fn deref(&self) -> &MultiPwAffRef { self.as_ref() }
}

impl To<Option<MultiPwAff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MultiPwAff> { NonNull::new(self).map(MultiPwAff) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MultiUnionPwAff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct MultiUnionPwAffRef(pub NonNull<c_void>);

impl_try!(MultiUnionPwAff);
impl_try!(MultiUnionPwAffRef);

impl MultiUnionPwAff {
  #[inline(always)]
  pub fn read(&self) -> MultiUnionPwAff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: MultiUnionPwAff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MultiUnionPwAffRef> for MultiUnionPwAff {
  #[inline(always)]
  fn as_ref(&self) -> &MultiUnionPwAffRef { unsafe { mem::transmute(self) } }
}

impl Deref for MultiUnionPwAff {
  type Target = MultiUnionPwAffRef;
  #[inline(always)]
  fn deref(&self) -> &MultiUnionPwAffRef { self.as_ref() }
}

impl To<Option<MultiUnionPwAff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MultiUnionPwAff> { NonNull::new(self).map(MultiUnionPwAff) }
}

