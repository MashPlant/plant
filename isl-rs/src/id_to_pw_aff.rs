use crate::*;

extern "C" {
  pub fn isl_id_to_pw_aff_alloc(ctx: CtxRef, min_size: c_int) -> Option<IdToPwAff>;
  pub fn isl_id_to_pw_aff_copy(hmap: IdToPwAffRef) -> Option<IdToPwAff>;
  pub fn isl_id_to_pw_aff_free(hmap: IdToPwAff) -> *mut c_void;
  pub fn isl_id_to_pw_aff_get_ctx(hmap: IdToPwAffRef) -> Option<CtxRef>;
  pub fn isl_id_to_pw_aff_try_get(hmap: IdToPwAffRef, key: IdRef) -> MaybeIslPwAff;
  pub fn isl_id_to_pw_aff_has(hmap: IdToPwAffRef, key: IdRef) -> Bool;
  pub fn isl_id_to_pw_aff_get(hmap: IdToPwAffRef, key: Id) -> Option<PwAff>;
  pub fn isl_id_to_pw_aff_set(hmap: IdToPwAff, key: Id, val: PwAff) -> Option<IdToPwAff>;
  pub fn isl_id_to_pw_aff_drop(hmap: IdToPwAff, key: Id) -> Option<IdToPwAff>;
  pub fn isl_id_to_pw_aff_foreach(hmap: IdToPwAffRef, fn_: unsafe extern "C" fn(key: Id, val: PwAff, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_printer_print_id_to_pw_aff(p: Printer, hmap: IdToPwAffRef) -> Option<Printer>;
  pub fn isl_id_to_pw_aff_dump(hmap: IdToPwAffRef) -> ();
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MaybeIslPwAff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct MaybeIslPwAffRef(pub NonNull<c_void>);

impl_try!(MaybeIslPwAff);
impl_try!(MaybeIslPwAffRef);

impl MaybeIslPwAff {
  #[inline(always)]
  pub fn read(&self) -> MaybeIslPwAff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: MaybeIslPwAff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MaybeIslPwAffRef> for MaybeIslPwAff {
  #[inline(always)]
  fn as_ref(&self) -> &MaybeIslPwAffRef { unsafe { mem::transmute(self) } }
}

impl Deref for MaybeIslPwAff {
  type Target = MaybeIslPwAffRef;
  #[inline(always)]
  fn deref(&self) -> &MaybeIslPwAffRef { self.as_ref() }
}

impl To<Option<MaybeIslPwAff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MaybeIslPwAff> { NonNull::new(self).map(MaybeIslPwAff) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct IdToPwAff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct IdToPwAffRef(pub NonNull<c_void>);

impl_try!(IdToPwAff);
impl_try!(IdToPwAffRef);

impl IdToPwAff {
  #[inline(always)]
  pub fn read(&self) -> IdToPwAff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: IdToPwAff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<IdToPwAffRef> for IdToPwAff {
  #[inline(always)]
  fn as_ref(&self) -> &IdToPwAffRef { unsafe { mem::transmute(self) } }
}

impl Deref for IdToPwAff {
  type Target = IdToPwAffRef;
  #[inline(always)]
  fn deref(&self) -> &IdToPwAffRef { self.as_ref() }
}

impl To<Option<IdToPwAff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<IdToPwAff> { NonNull::new(self).map(IdToPwAff) }
}

impl CtxRef {
  #[inline(always)]
  pub fn id_to_pw_aff_alloc(self, min_size: c_int) -> Option<IdToPwAff> {
    unsafe {
      let ret = isl_id_to_pw_aff_alloc(self.to(), min_size.to());
      (ret).to()
    }
  }
}

impl IdToPwAff {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_id_to_pw_aff_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set(self, key: Id, val: PwAff) -> Option<IdToPwAff> {
    unsafe {
      let ret = isl_id_to_pw_aff_set(self.to(), key.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, key: Id) -> Option<IdToPwAff> {
    unsafe {
      let ret = isl_id_to_pw_aff_drop(self.to(), key.to());
      (ret).to()
    }
  }
}

impl IdToPwAffRef {
  #[inline(always)]
  pub fn copy(self) -> Option<IdToPwAff> {
    unsafe {
      let ret = isl_id_to_pw_aff_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_id_to_pw_aff_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn try_get(self, key: IdRef) -> MaybeIslPwAff {
    unsafe {
      let ret = isl_id_to_pw_aff_try_get(self.to(), key.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has(self, key: IdRef) -> Bool {
    unsafe {
      let ret = isl_id_to_pw_aff_has(self.to(), key.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get(self, key: Id) -> Option<PwAff> {
    unsafe {
      let ret = isl_id_to_pw_aff_get(self.to(), key.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(Id, PwAff) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(Id, PwAff) -> Stat>(key: Id, val: PwAff, user: *mut c_void) -> Stat { (*(user as *mut F))(key.to(), val.to()) }
    unsafe {
      let ret = isl_id_to_pw_aff_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_id_to_pw_aff_dump(self.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_id_to_pw_aff(self, hmap: IdToPwAffRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_id_to_pw_aff(self.to(), hmap.to());
      (ret).to()
    }
  }
}

impl Drop for IdToPwAff {
  fn drop(&mut self) { IdToPwAff(self.0).free() }
}

