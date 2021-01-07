use crate::*;

extern "C" {
  pub fn isl_map_to_basic_set_alloc(ctx: CtxRef, min_size: c_int) -> Option<MapToBasicSet>;
  pub fn isl_map_to_basic_set_copy(hmap: MapToBasicSetRef) -> Option<MapToBasicSet>;
  pub fn isl_map_to_basic_set_free(hmap: MapToBasicSet) -> *mut c_void;
  pub fn isl_map_to_basic_set_get_ctx(hmap: MapToBasicSetRef) -> Option<CtxRef>;
  pub fn isl_map_to_basic_set_try_get(hmap: MapToBasicSetRef, key: MapRef) -> MaybeIslBasicSet;
  pub fn isl_map_to_basic_set_has(hmap: MapToBasicSetRef, key: MapRef) -> Bool;
  pub fn isl_map_to_basic_set_get(hmap: MapToBasicSetRef, key: Map) -> Option<BasicSet>;
  pub fn isl_map_to_basic_set_set(hmap: MapToBasicSet, key: Map, val: BasicSet) -> Option<MapToBasicSet>;
  pub fn isl_map_to_basic_set_drop(hmap: MapToBasicSet, key: Map) -> Option<MapToBasicSet>;
  pub fn isl_map_to_basic_set_foreach(hmap: MapToBasicSetRef, fn_: unsafe extern "C" fn(key: Map, val: BasicSet, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_printer_print_map_to_basic_set(p: Printer, hmap: MapToBasicSetRef) -> Option<Printer>;
  pub fn isl_map_to_basic_set_dump(hmap: MapToBasicSetRef) -> ();
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MaybeIslBasicSet(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct MaybeIslBasicSetRef(pub NonNull<c_void>);

impl_try!(MaybeIslBasicSet);
impl_try!(MaybeIslBasicSetRef);

impl MaybeIslBasicSet {
  #[inline(always)]
  pub fn read(&self) -> MaybeIslBasicSet { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: MaybeIslBasicSet) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MaybeIslBasicSetRef> for MaybeIslBasicSet {
  #[inline(always)]
  fn as_ref(&self) -> &MaybeIslBasicSetRef { unsafe { mem::transmute(self) } }
}

impl Deref for MaybeIslBasicSet {
  type Target = MaybeIslBasicSetRef;
  #[inline(always)]
  fn deref(&self) -> &MaybeIslBasicSetRef { self.as_ref() }
}

impl To<Option<MaybeIslBasicSet>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MaybeIslBasicSet> { NonNull::new(self).map(MaybeIslBasicSet) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MapToBasicSet(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct MapToBasicSetRef(pub NonNull<c_void>);

impl_try!(MapToBasicSet);
impl_try!(MapToBasicSetRef);

impl MapToBasicSet {
  #[inline(always)]
  pub fn read(&self) -> MapToBasicSet { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: MapToBasicSet) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MapToBasicSetRef> for MapToBasicSet {
  #[inline(always)]
  fn as_ref(&self) -> &MapToBasicSetRef { unsafe { mem::transmute(self) } }
}

impl Deref for MapToBasicSet {
  type Target = MapToBasicSetRef;
  #[inline(always)]
  fn deref(&self) -> &MapToBasicSetRef { self.as_ref() }
}

impl To<Option<MapToBasicSet>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MapToBasicSet> { NonNull::new(self).map(MapToBasicSet) }
}

impl CtxRef {
  #[inline(always)]
  pub fn map_to_basic_set_alloc(self, min_size: c_int) -> Option<MapToBasicSet> {
    unsafe {
      let ret = isl_map_to_basic_set_alloc(self.to(), min_size.to());
      (ret).to()
    }
  }
}

impl MapToBasicSet {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_map_to_basic_set_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set(self, key: Map, val: BasicSet) -> Option<MapToBasicSet> {
    unsafe {
      let ret = isl_map_to_basic_set_set(self.to(), key.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, key: Map) -> Option<MapToBasicSet> {
    unsafe {
      let ret = isl_map_to_basic_set_drop(self.to(), key.to());
      (ret).to()
    }
  }
}

impl MapToBasicSetRef {
  #[inline(always)]
  pub fn copy(self) -> Option<MapToBasicSet> {
    unsafe {
      let ret = isl_map_to_basic_set_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_map_to_basic_set_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn try_get(self, key: MapRef) -> MaybeIslBasicSet {
    unsafe {
      let ret = isl_map_to_basic_set_try_get(self.to(), key.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has(self, key: MapRef) -> Bool {
    unsafe {
      let ret = isl_map_to_basic_set_has(self.to(), key.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get(self, key: Map) -> Option<BasicSet> {
    unsafe {
      let ret = isl_map_to_basic_set_get(self.to(), key.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(Map, BasicSet) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(Map, BasicSet) -> Stat>(key: Map, val: BasicSet, user: *mut c_void) -> Stat { (*(user as *mut F))(key.to(), val.to()) }
    unsafe {
      let ret = isl_map_to_basic_set_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_map_to_basic_set_dump(self.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_map_to_basic_set(self, hmap: MapToBasicSetRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_map_to_basic_set(self.to(), hmap.to());
      (ret).to()
    }
  }
}

impl Drop for MapToBasicSet {
  fn drop(&mut self) { MapToBasicSet(self.0).free() }
}

