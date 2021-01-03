use crate::*;

extern "C" {
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Qpolynomial(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct QpolynomialRef(pub NonNull<c_void>);

impl Qpolynomial {
  #[inline(always)]
  pub fn read(&self) -> Qpolynomial { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Qpolynomial) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<QpolynomialRef> for Qpolynomial {
  #[inline(always)]
  fn as_ref(&self) -> &QpolynomialRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for Qpolynomial {
  type Target = QpolynomialRef;
  #[inline(always)]
  fn deref(&self) -> &QpolynomialRef { self.as_ref() }
}

impl To<Option<Qpolynomial>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Qpolynomial> { NonNull::new(self).map(Qpolynomial) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Term(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct TermRef(pub NonNull<c_void>);

impl Term {
  #[inline(always)]
  pub fn read(&self) -> Term { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Term) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<TermRef> for Term {
  #[inline(always)]
  fn as_ref(&self) -> &TermRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for Term {
  type Target = TermRef;
  #[inline(always)]
  fn deref(&self) -> &TermRef { self.as_ref() }
}

impl To<Option<Term>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Term> { NonNull::new(self).map(Term) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct PwQpolynomial(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct PwQpolynomialRef(pub NonNull<c_void>);

impl PwQpolynomial {
  #[inline(always)]
  pub fn read(&self) -> PwQpolynomial { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: PwQpolynomial) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<PwQpolynomialRef> for PwQpolynomial {
  #[inline(always)]
  fn as_ref(&self) -> &PwQpolynomialRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for PwQpolynomial {
  type Target = PwQpolynomialRef;
  #[inline(always)]
  fn deref(&self) -> &PwQpolynomialRef { self.as_ref() }
}

impl To<Option<PwQpolynomial>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<PwQpolynomial> { NonNull::new(self).map(PwQpolynomial) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct QpolynomialFold(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct QpolynomialFoldRef(pub NonNull<c_void>);

impl QpolynomialFold {
  #[inline(always)]
  pub fn read(&self) -> QpolynomialFold { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: QpolynomialFold) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<QpolynomialFoldRef> for QpolynomialFold {
  #[inline(always)]
  fn as_ref(&self) -> &QpolynomialFoldRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for QpolynomialFold {
  type Target = QpolynomialFoldRef;
  #[inline(always)]
  fn deref(&self) -> &QpolynomialFoldRef { self.as_ref() }
}

impl To<Option<QpolynomialFold>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<QpolynomialFold> { NonNull::new(self).map(QpolynomialFold) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct PwQpolynomialFold(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct PwQpolynomialFoldRef(pub NonNull<c_void>);

impl PwQpolynomialFold {
  #[inline(always)]
  pub fn read(&self) -> PwQpolynomialFold { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: PwQpolynomialFold) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<PwQpolynomialFoldRef> for PwQpolynomialFold {
  #[inline(always)]
  fn as_ref(&self) -> &PwQpolynomialFoldRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for PwQpolynomialFold {
  type Target = PwQpolynomialFoldRef;
  #[inline(always)]
  fn deref(&self) -> &PwQpolynomialFoldRef { self.as_ref() }
}

impl To<Option<PwQpolynomialFold>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<PwQpolynomialFold> { NonNull::new(self).map(PwQpolynomialFold) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct UnionPwQpolynomial(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct UnionPwQpolynomialRef(pub NonNull<c_void>);

impl UnionPwQpolynomial {
  #[inline(always)]
  pub fn read(&self) -> UnionPwQpolynomial { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: UnionPwQpolynomial) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<UnionPwQpolynomialRef> for UnionPwQpolynomial {
  #[inline(always)]
  fn as_ref(&self) -> &UnionPwQpolynomialRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for UnionPwQpolynomial {
  type Target = UnionPwQpolynomialRef;
  #[inline(always)]
  fn deref(&self) -> &UnionPwQpolynomialRef { self.as_ref() }
}

impl To<Option<UnionPwQpolynomial>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<UnionPwQpolynomial> { NonNull::new(self).map(UnionPwQpolynomial) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct UnionPwQpolynomialFold(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct UnionPwQpolynomialFoldRef(pub NonNull<c_void>);

impl UnionPwQpolynomialFold {
  #[inline(always)]
  pub fn read(&self) -> UnionPwQpolynomialFold { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: UnionPwQpolynomialFold) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<UnionPwQpolynomialFoldRef> for UnionPwQpolynomialFold {
  #[inline(always)]
  fn as_ref(&self) -> &UnionPwQpolynomialFoldRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for UnionPwQpolynomialFold {
  type Target = UnionPwQpolynomialFoldRef;
  #[inline(always)]
  fn deref(&self) -> &UnionPwQpolynomialFoldRef { self.as_ref() }
}

impl To<Option<UnionPwQpolynomialFold>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<UnionPwQpolynomialFold> { NonNull::new(self).map(UnionPwQpolynomialFold) }
}

