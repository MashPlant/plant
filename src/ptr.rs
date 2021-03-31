use std::{ptr::NonNull, cmp::Ordering, hash::{Hash, Hasher}, ops::{Deref, DerefMut}, fmt::{Debug, Display, Formatter, Result}};

// the difference between P<T> & R<T> is:
// the PartialEq & PartialOrd & Hash & Debug & Display implementation of P is based on T,
// while those of R is based on object
// in the most cases using both is okay, and I recommend to use P, and use R only you need T-based operation

#[repr(transparent)]
pub struct P<T: ?Sized>(pub NonNull<T>);

#[repr(transparent)]
pub struct R<T: ?Sized>(pub NonNull<T>);

impl<T: ?Sized> P<T> {
  #[inline(always)]
  pub fn new(ptr: *const T) -> P<T> { unsafe { P(NonNull::new_unchecked(ptr as _)) } }

  #[inline(always)]
  pub fn get<'a>(self) -> &'a mut T { unsafe { &mut *self.0.as_ptr() } }
}

impl<T: ?Sized> Deref for P<T> {
  type Target = T;
  #[inline(always)]
  fn deref(&self) -> &Self::Target { self.get() }
}

impl<T: ?Sized> DerefMut for P<T> {
  #[inline(always)]
  fn deref_mut(&mut self) -> &mut Self::Target { self.get() }
}

impl<T: ?Sized> Clone for P<T> {
  #[inline(always)]
  fn clone(&self) -> Self { P(self.0) }
}

impl<T: ?Sized> Copy for P<T> {}

unsafe impl<T: ?Sized> Send for P<T> {}

unsafe impl<T: ?Sized> Sync for P<T> {}

impl<T: ?Sized> From<&T> for P<T> {
  #[inline(always)]
  fn from(x: &T) -> Self { P(x.into()) }
}

impl<T: ?Sized> From<&mut T> for P<T> {
  #[inline(always)]
  fn from(x: &mut T) -> Self { P(x.into()) }
}

impl<T: ?Sized> From<R<T>> for P<T> {
  #[inline(always)]
  fn from(x: R<T>) -> Self { P(x.0) }
}

impl<T: ?Sized> PartialEq for P<T> {
  #[inline(always)]
  fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl<T: ?Sized> Eq for P<T> {}

impl<T: ?Sized> PartialOrd for P<T> {
  #[inline(always)]
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> { self.0.partial_cmp(&other.0) }
}

impl<T: ?Sized> Ord for P<T> {
  #[inline(always)]
  fn cmp(&self, other: &Self) -> Ordering { self.0.cmp(&other.0) }
}

impl<T: ?Sized> Hash for P<T> {
  #[inline(always)]
  fn hash<H: Hasher>(&self, state: &mut H) { self.0.hash(state) }
}

impl<T: ?Sized> Debug for P<T> {
  #[inline(always)]
  fn fmt(&self, f: &mut Formatter) -> Result { self.0.fmt(f) }
}

impl<T: ?Sized> Display for P<T> {
  #[inline(always)]
  fn fmt(&self, f: &mut Formatter) -> Result { self.0.fmt(f) }
}

pub trait IntoP<T: ?Sized> {
  fn p(self) -> P<T>;
}

impl<T: ?Sized, U: Into<P<T>>> IntoP<T> for U {
  #[inline(always)]
  fn p(self) -> P<T> { self.into() }
}

impl<T: ?Sized> R<T> {
  #[inline(always)]
  pub fn new(ptr: *const T) -> R<T> { unsafe { R(NonNull::new_unchecked(ptr as _)) } }

  #[inline(always)]
  pub fn get<'a>(self) -> &'a mut T { unsafe { &mut *self.0.as_ptr() } }
}

impl<T: ?Sized> Deref for R<T> {
  type Target = T;
  #[inline(always)]
  fn deref(&self) -> &Self::Target { self.get() }
}

impl<T: ?Sized> DerefMut for R<T> {
  #[inline(always)]
  fn deref_mut(&mut self) -> &mut Self::Target { self.get() }
}

impl<T: ?Sized> Clone for R<T> {
  #[inline(always)]
  fn clone(&self) -> Self { R(self.0) }
}

impl<T: ?Sized> Copy for R<T> {}

unsafe impl<T: ?Sized> Send for R<T> {}

unsafe impl<T: ?Sized> Sync for R<T> {}

impl<T: ?Sized> From<&T> for R<T> {
  #[inline(always)]
  fn from(x: &T) -> Self { R(x.into()) }
}

impl<T: ?Sized> From<&mut T> for R<T> {
  #[inline(always)]
  fn from(x: &mut T) -> Self { R(x.into()) }
}

impl<T: ?Sized> From<P<T>> for R<T> {
  #[inline(always)]
  fn from(x: P<T>) -> Self { R(x.0) }
}

impl<T: ?Sized + PartialEq> PartialEq for R<T> {
  #[inline(always)]
  fn eq(&self, other: &Self) -> bool { self.get() == other.get() }
}

impl<T: ?Sized + Eq> Eq for R<T> {}

impl<T: ?Sized + PartialOrd> PartialOrd for R<T> {
  #[inline(always)]
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> { (*self.get()).partial_cmp(other.get()) }
}

impl<T: ?Sized + Ord> Ord for R<T> {
  #[inline(always)]
  fn cmp(&self, other: &Self) -> Ordering { (*self.get()).cmp(other.get()) }
}

impl<T: ?Sized + Hash> Hash for R<T> {
  #[inline(always)]
  fn hash<H: Hasher>(&self, state: &mut H) { self.get().hash(state) }
}

impl<T: ?Sized + Debug> Debug for R<T> {
  #[inline(always)]
  fn fmt(&self, f: &mut Formatter) -> Result { self.get().fmt(f) }
}

impl<T: ?Sized + Display> Display for R<T> {
  #[inline(always)]
  fn fmt(&self, f: &mut Formatter) -> Result { self.get().fmt(f) }
}

pub trait IntoR<T: ?Sized> {
  fn r(self) -> R<T>;
}

impl<T: ?Sized, U: Into<R<T>>> IntoR<T> for U {
  #[inline(always)]
  fn r(self) -> R<T> { self.into() }
}
