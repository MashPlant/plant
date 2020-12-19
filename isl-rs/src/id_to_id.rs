use crate::*;

extern "C" {
  pub fn isl_id_to_id_alloc(ctx: CtxRef, min_size: c_int) -> Option<IdToId>;
  pub fn isl_id_to_id_copy(hmap: IdToIdRef) -> Option<IdToId>;
  pub fn isl_id_to_id_free(hmap: IdToId) -> *mut c_void;
  pub fn isl_id_to_id_get_ctx(hmap: IdToIdRef) -> Option<CtxRef>;
  pub fn isl_id_to_id_try_get(hmap: IdToIdRef, key: IdRef) -> MaybeIslId;
  pub fn isl_id_to_id_has(hmap: IdToIdRef, key: IdRef) -> Bool;
  pub fn isl_id_to_id_get(hmap: IdToIdRef, key: Id) -> Option<Id>;
  pub fn isl_id_to_id_set(hmap: IdToId, key: Id, val: Id) -> Option<IdToId>;
  pub fn isl_id_to_id_drop(hmap: IdToId, key: Id) -> Option<IdToId>;
  pub fn isl_id_to_id_foreach(hmap: IdToIdRef, fn_: unsafe extern "C" fn(key: Id, val: Id, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_printer_print_id_to_id(p: Printer, hmap: IdToIdRef) -> Option<Printer>;
  pub fn isl_id_to_id_dump(hmap: IdToIdRef) -> ();
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MaybeIslId(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct MaybeIslIdRef(pub NonNull<c_void>);

impl MaybeIslId {
  #[inline(always)]
  pub fn read(&self) -> MaybeIslId { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&mut self, x: MaybeIslId) { unsafe { ptr::write(self, x) } }
}

impl AsRef<MaybeIslIdRef> for MaybeIslId {
  #[inline(always)]
  fn as_ref(&self) -> &MaybeIslIdRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for MaybeIslId {
  type Target = MaybeIslIdRef;
  #[inline(always)]
  fn deref(&self) -> &MaybeIslIdRef { self.as_ref() }
}

impl To<Option<MaybeIslId>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MaybeIslId> { NonNull::new(self).map(MaybeIslId) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct IdToId(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct IdToIdRef(pub NonNull<c_void>);

impl IdToId {
  #[inline(always)]
  pub fn read(&self) -> IdToId { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&mut self, x: IdToId) { unsafe { ptr::write(self, x) } }
}

impl AsRef<IdToIdRef> for IdToId {
  #[inline(always)]
  fn as_ref(&self) -> &IdToIdRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for IdToId {
  type Target = IdToIdRef;
  #[inline(always)]
  fn deref(&self) -> &IdToIdRef { self.as_ref() }
}

impl To<Option<IdToId>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<IdToId> { NonNull::new(self).map(IdToId) }
}

impl CtxRef {
  #[inline(always)]
  pub fn id_to_id_alloc(self, min_size: c_int) -> Option<IdToId> {
    unsafe {
      let ret = isl_id_to_id_alloc(self.to(), min_size.to());
      (ret).to()
    }
  }
}

impl IdToId {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_id_to_id_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set(self, key: Id, val: Id) -> Option<IdToId> {
    unsafe {
      let ret = isl_id_to_id_set(self.to(), key.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, key: Id) -> Option<IdToId> {
    unsafe {
      let ret = isl_id_to_id_drop(self.to(), key.to());
      (ret).to()
    }
  }
}

impl IdToIdRef {
  #[inline(always)]
  pub fn copy(self) -> Option<IdToId> {
    unsafe {
      let ret = isl_id_to_id_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_id_to_id_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn try_get(self, key: IdRef) -> MaybeIslId {
    unsafe {
      let ret = isl_id_to_id_try_get(self.to(), key.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has(self, key: IdRef) -> Option<bool> {
    unsafe {
      let ret = isl_id_to_id_has(self.to(), key.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get(self, key: Id) -> Option<Id> {
    unsafe {
      let ret = isl_id_to_id_get(self.to(), key.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(Id, Id) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(Id, Id) -> Option<()>>(key: Id, val: Id, user: *mut c_void) -> Stat { (*(user as *mut F))(key.to(), val.to()).to() }
    unsafe {
      let ret = isl_id_to_id_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_id_to_id_dump(self.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_id_to_id(self, hmap: IdToIdRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_id_to_id(self.to(), hmap.to());
      (ret).to()
    }
  }
}

impl Drop for IdToId {
  fn drop(&mut self) { IdToId(self.0).free() }
}

