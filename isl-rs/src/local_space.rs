use crate::*;

extern "C" {
  pub fn isl_local_space_get_ctx(ls: LocalSpaceRef) -> Option<CtxRef>;
  pub fn isl_local_space_from_space(dim: Space) -> Option<LocalSpace>;
  pub fn isl_local_space_copy(ls: LocalSpaceRef) -> Option<LocalSpace>;
  pub fn isl_local_space_free(ls: LocalSpace) -> *mut c_void;
  pub fn isl_local_space_is_params(ls: LocalSpaceRef) -> Bool;
  pub fn isl_local_space_is_set(ls: LocalSpaceRef) -> Bool;
  pub fn isl_local_space_set_tuple_id(ls: LocalSpace, type_: DimType, id: Id) -> Option<LocalSpace>;
  pub fn isl_local_space_dim(ls: LocalSpaceRef, type_: DimType) -> c_int;
  pub fn isl_local_space_has_dim_name(ls: LocalSpaceRef, type_: DimType, pos: c_uint) -> Bool;
  pub fn isl_local_space_get_dim_name(ls: LocalSpaceRef, type_: DimType, pos: c_uint) -> Option<CStr>;
  pub fn isl_local_space_set_dim_name(ls: LocalSpace, type_: DimType, pos: c_uint, s: CStr) -> Option<LocalSpace>;
  pub fn isl_local_space_has_dim_id(ls: LocalSpaceRef, type_: DimType, pos: c_uint) -> Bool;
  pub fn isl_local_space_get_dim_id(ls: LocalSpaceRef, type_: DimType, pos: c_uint) -> Option<Id>;
  pub fn isl_local_space_set_dim_id(ls: LocalSpace, type_: DimType, pos: c_uint, id: Id) -> Option<LocalSpace>;
  pub fn isl_local_space_get_space(ls: LocalSpaceRef) -> Option<Space>;
  pub fn isl_local_space_get_div(ls: LocalSpaceRef, pos: c_int) -> Option<Aff>;
  pub fn isl_local_space_find_dim_by_name(ls: LocalSpaceRef, type_: DimType, name: CStr) -> c_int;
  pub fn isl_local_space_domain(ls: LocalSpace) -> Option<LocalSpace>;
  pub fn isl_local_space_range(ls: LocalSpace) -> Option<LocalSpace>;
  pub fn isl_local_space_from_domain(ls: LocalSpace) -> Option<LocalSpace>;
  pub fn isl_local_space_add_dims(ls: LocalSpace, type_: DimType, n: c_uint) -> Option<LocalSpace>;
  pub fn isl_local_space_drop_dims(ls: LocalSpace, type_: DimType, first: c_uint, n: c_uint) -> Option<LocalSpace>;
  pub fn isl_local_space_insert_dims(ls: LocalSpace, type_: DimType, first: c_uint, n: c_uint) -> Option<LocalSpace>;
  pub fn isl_local_space_set_from_params(ls: LocalSpace) -> Option<LocalSpace>;
  pub fn isl_local_space_intersect(ls1: LocalSpace, ls2: LocalSpace) -> Option<LocalSpace>;
  pub fn isl_local_space_wrap(ls: LocalSpace) -> Option<LocalSpace>;
  pub fn isl_local_space_is_equal(ls1: LocalSpaceRef, ls2: LocalSpaceRef) -> Bool;
  pub fn isl_local_space_lifting(ls: LocalSpace) -> Option<BasicMap>;
  pub fn isl_local_space_flatten_domain(ls: LocalSpace) -> Option<LocalSpace>;
  pub fn isl_local_space_flatten_range(ls: LocalSpace) -> Option<LocalSpace>;
  pub fn isl_printer_print_local_space(p: Printer, ls: LocalSpaceRef) -> Option<Printer>;
  pub fn isl_local_space_dump(ls: LocalSpaceRef) -> ();
}

#[repr(transparent)]
#[derive(Debug)]
pub struct LocalSpace(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct LocalSpaceRef(pub NonNull<c_void>);

impl LocalSpace {
  #[inline(always)]
  pub fn read(&self) -> LocalSpace { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: LocalSpace) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<LocalSpaceRef> for LocalSpace {
  #[inline(always)]
  fn as_ref(&self) -> &LocalSpaceRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for LocalSpace {
  type Target = LocalSpaceRef;
  #[inline(always)]
  fn deref(&self) -> &LocalSpaceRef { self.as_ref() }
}

impl To<Option<LocalSpace>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<LocalSpace> { NonNull::new(self).map(LocalSpace) }
}

impl LocalSpace {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_local_space_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_id(self, type_: DimType, id: Id) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_local_space_set_tuple_id(self.to(), type_.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_name(self, type_: DimType, pos: c_uint, s: CStr) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_local_space_set_dim_name(self.to(), type_.to(), pos.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_id(self, type_: DimType, pos: c_uint, id: Id) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_local_space_set_dim_id(self.to(), type_.to(), pos.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain(self) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_local_space_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range(self) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_local_space_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn from_domain(self) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_local_space_from_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_dims(self, type_: DimType, n: c_uint) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_local_space_add_dims(self.to(), type_.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_local_space_drop_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_local_space_insert_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_from_params(self) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_local_space_set_from_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect(self, ls2: LocalSpace) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_local_space_intersect(self.to(), ls2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn wrap(self) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_local_space_wrap(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lifting(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_local_space_lifting(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten_domain(self) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_local_space_flatten_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten_range(self) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_local_space_flatten_range(self.to());
      (ret).to()
    }
  }
}

impl LocalSpaceRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_local_space_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_local_space_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_params(self) -> Option<bool> {
    unsafe {
      let ret = isl_local_space_is_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_set(self) -> Option<bool> {
    unsafe {
      let ret = isl_local_space_is_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_int {
    unsafe {
      let ret = isl_local_space_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_dim_name(self, type_: DimType, pos: c_uint) -> Option<bool> {
    unsafe {
      let ret = isl_local_space_has_dim_name(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_name(self, type_: DimType, pos: c_uint) -> Option<CStr> {
    unsafe {
      let ret = isl_local_space_get_dim_name(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_dim_id(self, type_: DimType, pos: c_uint) -> Option<bool> {
    unsafe {
      let ret = isl_local_space_has_dim_id(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_id(self, type_: DimType, pos: c_uint) -> Option<Id> {
    unsafe {
      let ret = isl_local_space_get_dim_id(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_local_space_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_div(self, pos: c_int) -> Option<Aff> {
    unsafe {
      let ret = isl_local_space_get_div(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: CStr) -> c_int {
    unsafe {
      let ret = isl_local_space_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_equal(self, ls2: LocalSpaceRef) -> Option<bool> {
    unsafe {
      let ret = isl_local_space_is_equal(self.to(), ls2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_local_space_dump(self.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_local_space(self, ls: LocalSpaceRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_local_space(self.to(), ls.to());
      (ret).to()
    }
  }
}

impl Space {
  #[inline(always)]
  pub fn local_space_from_space(self) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_local_space_from_space(self.to());
      (ret).to()
    }
  }
}

impl Drop for LocalSpace {
  fn drop(&mut self) { LocalSpace(self.0).free() }
}

