use crate::*;

extern "C" {
  pub fn isl_vec_alloc(ctx: CtxRef, size: c_uint) -> Option<Vec>;
  pub fn isl_vec_zero(ctx: CtxRef, size: c_uint) -> Option<Vec>;
  pub fn isl_vec_copy(vec: VecRef) -> Option<Vec>;
  pub fn isl_vec_free(vec: Vec) -> *mut c_void;
  pub fn isl_vec_get_ctx(vec: VecRef) -> Option<CtxRef>;
  pub fn isl_vec_size(vec: VecRef) -> c_int;
  pub fn isl_vec_get_element_val(vec: VecRef, pos: c_int) -> Option<Val>;
  pub fn isl_vec_set_element_si(vec: Vec, pos: c_int, v: c_int) -> Option<Vec>;
  pub fn isl_vec_set_element_val(vec: Vec, pos: c_int, v: Val) -> Option<Vec>;
  pub fn isl_vec_is_equal(vec1: VecRef, vec2: VecRef) -> Bool;
  pub fn isl_vec_cmp_element(vec1: VecRef, vec2: VecRef, pos: c_int) -> c_int;
  pub fn isl_vec_dump(vec: VecRef) -> ();
  pub fn isl_printer_print_vec(printer: Printer, vec: VecRef) -> Option<Printer>;
  pub fn isl_vec_ceil(vec: Vec) -> Option<Vec>;
  pub fn isl_vec_normalize(vec: Vec) -> Option<Vec>;
  pub fn isl_vec_set_si(vec: Vec, v: c_int) -> Option<Vec>;
  pub fn isl_vec_set_val(vec: Vec, v: Val) -> Option<Vec>;
  pub fn isl_vec_clr(vec: Vec) -> Option<Vec>;
  pub fn isl_vec_neg(vec: Vec) -> Option<Vec>;
  pub fn isl_vec_add(vec1: Vec, vec2: Vec) -> Option<Vec>;
  pub fn isl_vec_extend(vec: Vec, size: c_uint) -> Option<Vec>;
  pub fn isl_vec_zero_extend(vec: Vec, size: c_uint) -> Option<Vec>;
  pub fn isl_vec_concat(vec1: Vec, vec2: Vec) -> Option<Vec>;
  pub fn isl_vec_sort(vec: Vec) -> Option<Vec>;
  pub fn isl_vec_read_from_file(ctx: CtxRef, input: *mut FILE) -> Option<Vec>;
  pub fn isl_vec_drop_els(vec: Vec, pos: c_uint, n: c_uint) -> Option<Vec>;
  pub fn isl_vec_add_els(vec: Vec, n: c_uint) -> Option<Vec>;
  pub fn isl_vec_insert_els(vec: Vec, pos: c_uint, n: c_uint) -> Option<Vec>;
  pub fn isl_vec_insert_zero_els(vec: Vec, pos: c_uint, n: c_uint) -> Option<Vec>;
  pub fn isl_vec_move_els(vec: Vec, dst_col: c_uint, src_col: c_uint, n: c_uint) -> Option<Vec>;
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Vec(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct VecRef(pub NonNull<c_void>);

impl_try!(Vec);
impl_try!(VecRef);

impl Vec {
  #[inline(always)]
  pub fn read(&self) -> Vec { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Vec) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<VecRef> for Vec {
  #[inline(always)]
  fn as_ref(&self) -> &VecRef { unsafe { mem::transmute(self) } }
}

impl Deref for Vec {
  type Target = VecRef;
  #[inline(always)]
  fn deref(&self) -> &VecRef { self.as_ref() }
}

impl To<Option<Vec>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Vec> { NonNull::new(self).map(Vec) }
}

impl CtxRef {
  #[inline(always)]
  pub fn vec_alloc(self, size: c_uint) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_alloc(self.to(), size.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn vec_zero(self, size: c_uint) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_zero(self.to(), size.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn vec_read_from_file(self, input: *mut FILE) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_read_from_file(self.to(), input.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_vec(self, vec: VecRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_vec(self.to(), vec.to());
      (ret).to()
    }
  }
}

impl Vec {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_vec_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_element_si(self, pos: c_int, v: c_int) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_set_element_si(self.to(), pos.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_element_val(self, pos: c_int, v: Val) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_set_element_val(self.to(), pos.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ceil(self) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_ceil(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn normalize(self) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_normalize(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_si(self, v: c_int) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_set_si(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_val(self, v: Val) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_set_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn clr(self) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_clr(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg(self) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, vec2: Vec) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_add(self.to(), vec2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn extend(self, size: c_uint) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_extend(self.to(), size.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn zero_extend(self, size: c_uint) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_zero_extend(self.to(), size.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, vec2: Vec) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_concat(self.to(), vec2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort(self) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_sort(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_els(self, pos: c_uint, n: c_uint) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_drop_els(self.to(), pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_els(self, n: c_uint) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_add_els(self.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_els(self, pos: c_uint, n: c_uint) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_insert_els(self.to(), pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_zero_els(self, pos: c_uint, n: c_uint) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_insert_zero_els(self.to(), pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn move_els(self, dst_col: c_uint, src_col: c_uint, n: c_uint) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_move_els(self.to(), dst_col.to(), src_col.to(), n.to());
      (ret).to()
    }
  }
}

impl VecRef {
  #[inline(always)]
  pub fn copy(self) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_vec_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn size(self) -> c_int {
    unsafe {
      let ret = isl_vec_size(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_element_val(self, pos: c_int) -> Option<Val> {
    unsafe {
      let ret = isl_vec_get_element_val(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_equal(self, vec2: VecRef) -> Bool {
    unsafe {
      let ret = isl_vec_is_equal(self.to(), vec2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn cmp_element(self, vec2: VecRef, pos: c_int) -> c_int {
    unsafe {
      let ret = isl_vec_cmp_element(self.to(), vec2.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_vec_dump(self.to());
      (ret).to()
    }
  }
}

impl Drop for Vec {
  fn drop(&mut self) { Vec(self.0).free() }
}

