use crate::*;

extern "C" {
  pub fn isl_val_list_get_ctx(list: ValListRef) -> Option<CtxRef>;
  pub fn isl_val_list_from_val(el: Val) -> Option<ValList>;
  pub fn isl_val_list_alloc(ctx: CtxRef, n: c_int) -> Option<ValList>;
  pub fn isl_val_list_copy(list: ValListRef) -> Option<ValList>;
  pub fn isl_val_list_free(list: ValList) -> *mut c_void;
  pub fn isl_val_list_add(list: ValList, el: Val) -> Option<ValList>;
  pub fn isl_val_list_insert(list: ValList, pos: c_uint, el: Val) -> Option<ValList>;
  pub fn isl_val_list_drop(list: ValList, first: c_uint, n: c_uint) -> Option<ValList>;
  pub fn isl_val_list_concat(list1: ValList, list2: ValList) -> Option<ValList>;
  pub fn isl_val_list_n_val(list: ValListRef) -> c_int;
  pub fn isl_val_list_get_val(list: ValListRef, index: c_int) -> Option<Val>;
  pub fn isl_val_list_set_val(list: ValList, index: c_int, el: Val) -> Option<ValList>;
  pub fn isl_val_list_foreach(list: ValListRef, fn_: unsafe extern "C" fn(el: Val, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_val_list_map(list: ValList, fn_: unsafe extern "C" fn(el: Val, user: *mut c_void) -> Option<Val>, user: *mut c_void) -> Option<ValList>;
  pub fn isl_val_list_sort(list: ValList, cmp: unsafe extern "C" fn(a: ValRef, b: ValRef, user: *mut c_void) -> c_int, user: *mut c_void) -> Option<ValList>;
  pub fn isl_val_list_foreach_scc(list: ValListRef, follows: unsafe extern "C" fn(a: ValRef, b: ValRef, user: *mut c_void) -> Bool, follows_user: *mut c_void, fn_: unsafe extern "C" fn(scc: ValList, user: *mut c_void) -> Stat, fn_user: *mut c_void) -> Stat;
  pub fn isl_printer_print_val_list(p: Printer, list: ValListRef) -> Option<Printer>;
  pub fn isl_val_list_dump(list: ValListRef) -> ();
  pub fn isl_multi_val_dim(multi: MultiValRef, type_: DimType) -> c_uint;
  pub fn isl_multi_val_get_ctx(multi: MultiValRef) -> Option<CtxRef>;
  pub fn isl_multi_val_get_space(multi: MultiValRef) -> Option<Space>;
  pub fn isl_multi_val_get_domain_space(multi: MultiValRef) -> Option<Space>;
  pub fn isl_multi_val_find_dim_by_name(multi: MultiValRef, type_: DimType, name: Option<CStr>) -> c_int;
  pub fn isl_multi_val_from_val_list(space: Space, list: ValList) -> Option<MultiVal>;
  pub fn isl_multi_val_zero(space: Space) -> Option<MultiVal>;
  pub fn isl_multi_val_copy(multi: MultiValRef) -> Option<MultiVal>;
  pub fn isl_multi_val_free(multi: MultiVal) -> *mut c_void;
  pub fn isl_multi_val_plain_is_equal(multi1: MultiValRef, multi2: MultiValRef) -> Bool;
  pub fn isl_multi_val_involves_nan(multi: MultiValRef) -> Bool;
  pub fn isl_multi_val_find_dim_by_id(multi: MultiValRef, type_: DimType, id: IdRef) -> c_int;
  pub fn isl_multi_val_get_dim_id(multi: MultiValRef, type_: DimType, pos: c_uint) -> Option<Id>;
  pub fn isl_multi_val_set_dim_name(multi: MultiVal, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<MultiVal>;
  pub fn isl_multi_val_set_dim_id(multi: MultiVal, type_: DimType, pos: c_uint, id: Id) -> Option<MultiVal>;
  pub fn isl_multi_val_get_tuple_name(multi: MultiValRef, type_: DimType) -> Option<CStr>;
  pub fn isl_multi_val_has_tuple_id(multi: MultiValRef, type_: DimType) -> Bool;
  pub fn isl_multi_val_get_tuple_id(multi: MultiValRef, type_: DimType) -> Option<Id>;
  pub fn isl_multi_val_set_tuple_name(multi: MultiVal, type_: DimType, s: Option<CStr>) -> Option<MultiVal>;
  pub fn isl_multi_val_set_tuple_id(multi: MultiVal, type_: DimType, id: Id) -> Option<MultiVal>;
  pub fn isl_multi_val_reset_tuple_id(multi: MultiVal, type_: DimType) -> Option<MultiVal>;
  pub fn isl_multi_val_reset_user(multi: MultiVal) -> Option<MultiVal>;
  pub fn isl_multi_val_drop_dims(multi: MultiVal, type_: DimType, first: c_uint, n: c_uint) -> Option<MultiVal>;
  pub fn isl_multi_val_get_val(multi: MultiValRef, pos: c_int) -> Option<Val>;
  pub fn isl_multi_val_set_val(multi: MultiVal, pos: c_int, el: Val) -> Option<MultiVal>;
  pub fn isl_multi_val_range_splice(multi1: MultiVal, pos: c_uint, multi2: MultiVal) -> Option<MultiVal>;
  pub fn isl_multi_val_flatten_range(multi: MultiVal) -> Option<MultiVal>;
  pub fn isl_multi_val_flat_range_product(multi1: MultiVal, multi2: MultiVal) -> Option<MultiVal>;
  pub fn isl_multi_val_range_product(multi1: MultiVal, multi2: MultiVal) -> Option<MultiVal>;
  pub fn isl_multi_val_factor_range(multi: MultiVal) -> Option<MultiVal>;
  pub fn isl_multi_val_range_is_wrapping(multi: MultiValRef) -> Bool;
  pub fn isl_multi_val_range_factor_domain(multi: MultiVal) -> Option<MultiVal>;
  pub fn isl_multi_val_range_factor_range(multi: MultiVal) -> Option<MultiVal>;
  pub fn isl_multi_val_scale_val(multi: MultiVal, v: Val) -> Option<MultiVal>;
  pub fn isl_multi_val_scale_down_val(multi: MultiVal, v: Val) -> Option<MultiVal>;
  pub fn isl_multi_val_scale_multi_val(multi: MultiVal, mv: MultiVal) -> Option<MultiVal>;
  pub fn isl_multi_val_scale_down_multi_val(multi: MultiVal, mv: MultiVal) -> Option<MultiVal>;
  pub fn isl_multi_val_mod_multi_val(multi: MultiVal, mv: MultiVal) -> Option<MultiVal>;
  pub fn isl_multi_val_add(multi1: MultiVal, multi2: MultiVal) -> Option<MultiVal>;
  pub fn isl_multi_val_sub(multi1: MultiVal, multi2: MultiVal) -> Option<MultiVal>;
  pub fn isl_multi_val_align_params(multi: MultiVal, model: Space) -> Option<MultiVal>;
  pub fn isl_multi_val_from_range(multi: MultiVal) -> Option<MultiVal>;
  pub fn isl_multi_val_neg(multi: MultiVal) -> Option<MultiVal>;
  pub fn isl_multi_val_involves_dims(multi: MultiValRef, type_: DimType, first: c_uint, n: c_uint) -> Bool;
  pub fn isl_multi_val_insert_dims(multi: MultiVal, type_: DimType, first: c_uint, n: c_uint) -> Option<MultiVal>;
  pub fn isl_multi_val_add_dims(multi: MultiVal, type_: DimType, n: c_uint) -> Option<MultiVal>;
  pub fn isl_multi_val_project_domain_on_params(multi: MultiVal) -> Option<MultiVal>;
  pub fn isl_multi_val_product(multi1: MultiVal, multi2: MultiVal) -> Option<MultiVal>;
  pub fn isl_multi_val_splice(multi1: MultiVal, in_pos: c_uint, out_pos: c_uint, multi2: MultiVal) -> Option<MultiVal>;
  pub fn isl_val_zero(ctx: CtxRef) -> Option<Val>;
  pub fn isl_val_one(ctx: CtxRef) -> Option<Val>;
  pub fn isl_val_negone(ctx: CtxRef) -> Option<Val>;
  pub fn isl_val_nan(ctx: CtxRef) -> Option<Val>;
  pub fn isl_val_infty(ctx: CtxRef) -> Option<Val>;
  pub fn isl_val_neginfty(ctx: CtxRef) -> Option<Val>;
  pub fn isl_val_int_from_si(ctx: CtxRef, i: c_long) -> Option<Val>;
  pub fn isl_val_int_from_ui(ctx: CtxRef, u: c_ulong) -> Option<Val>;
  pub fn isl_val_int_from_chunks(ctx: CtxRef, n: c_int, size: c_int, chunks: *mut c_void) -> Option<Val>;
  pub fn isl_val_copy(v: ValRef) -> Option<Val>;
  pub fn isl_val_free(v: Val) -> *mut c_void;
  pub fn isl_val_get_ctx(val: ValRef) -> Option<CtxRef>;
  pub fn isl_val_get_hash(val: ValRef) -> c_uint;
  pub fn isl_val_get_num_si(v: ValRef) -> c_long;
  pub fn isl_val_get_den_si(v: ValRef) -> c_long;
  pub fn isl_val_get_den_val(v: ValRef) -> Option<Val>;
  pub fn isl_val_get_d(v: ValRef) -> c_double;
  pub fn isl_val_n_abs_num_chunks(v: ValRef, size: c_int) -> c_int;
  pub fn isl_val_get_abs_num_chunks(v: ValRef, size: c_int, chunks: *mut c_void) -> c_int;
  pub fn isl_val_set_si(v: Val, i: c_long) -> Option<Val>;
  pub fn isl_val_abs(v: Val) -> Option<Val>;
  pub fn isl_val_neg(v: Val) -> Option<Val>;
  pub fn isl_val_inv(v: Val) -> Option<Val>;
  pub fn isl_val_floor(v: Val) -> Option<Val>;
  pub fn isl_val_ceil(v: Val) -> Option<Val>;
  pub fn isl_val_trunc(v: Val) -> Option<Val>;
  pub fn isl_val_2exp(v: Val) -> Option<Val>;
  pub fn isl_val_min(v1: Val, v2: Val) -> Option<Val>;
  pub fn isl_val_max(v1: Val, v2: Val) -> Option<Val>;
  pub fn isl_val_add(v1: Val, v2: Val) -> Option<Val>;
  pub fn isl_val_add_ui(v1: Val, v2: c_ulong) -> Option<Val>;
  pub fn isl_val_sub(v1: Val, v2: Val) -> Option<Val>;
  pub fn isl_val_sub_ui(v1: Val, v2: c_ulong) -> Option<Val>;
  pub fn isl_val_mul(v1: Val, v2: Val) -> Option<Val>;
  pub fn isl_val_mul_ui(v1: Val, v2: c_ulong) -> Option<Val>;
  pub fn isl_val_div(v1: Val, v2: Val) -> Option<Val>;
  pub fn isl_val_div_ui(v1: Val, v2: c_ulong) -> Option<Val>;
  pub fn isl_val_mod(v1: Val, v2: Val) -> Option<Val>;
  pub fn isl_val_gcd(v1: Val, v2: Val) -> Option<Val>;
  pub fn isl_val_gcdext(v1: Val, v2: Val, x: *mut Val, y: *mut Val) -> Option<Val>;
  pub fn isl_val_sgn(v: ValRef) -> c_int;
  pub fn isl_val_is_zero(v: ValRef) -> Bool;
  pub fn isl_val_is_one(v: ValRef) -> Bool;
  pub fn isl_val_is_negone(v: ValRef) -> Bool;
  pub fn isl_val_is_nonneg(v: ValRef) -> Bool;
  pub fn isl_val_is_nonpos(v: ValRef) -> Bool;
  pub fn isl_val_is_pos(v: ValRef) -> Bool;
  pub fn isl_val_is_neg(v: ValRef) -> Bool;
  pub fn isl_val_is_int(v: ValRef) -> Bool;
  pub fn isl_val_is_rat(v: ValRef) -> Bool;
  pub fn isl_val_is_nan(v: ValRef) -> Bool;
  pub fn isl_val_is_infty(v: ValRef) -> Bool;
  pub fn isl_val_is_neginfty(v: ValRef) -> Bool;
  pub fn isl_val_cmp_si(v: ValRef, i: c_long) -> c_int;
  pub fn isl_val_lt(v1: ValRef, v2: ValRef) -> Bool;
  pub fn isl_val_le(v1: ValRef, v2: ValRef) -> Bool;
  pub fn isl_val_gt(v1: ValRef, v2: ValRef) -> Bool;
  pub fn isl_val_ge(v1: ValRef, v2: ValRef) -> Bool;
  pub fn isl_val_eq(v1: ValRef, v2: ValRef) -> Bool;
  pub fn isl_val_ne(v1: ValRef, v2: ValRef) -> Bool;
  pub fn isl_val_abs_eq(v1: ValRef, v2: ValRef) -> Bool;
  pub fn isl_val_is_divisible_by(v1: ValRef, v2: ValRef) -> Bool;
  pub fn isl_val_read_from_str(ctx: CtxRef, str: Option<CStr>) -> Option<Val>;
  pub fn isl_printer_print_val(p: Printer, v: ValRef) -> Option<Printer>;
  pub fn isl_val_dump(v: ValRef) -> ();
  pub fn isl_val_to_str(v: ValRef) -> Option<CString>;
  pub fn isl_multi_val_add_val(mv: MultiVal, v: Val) -> Option<MultiVal>;
  pub fn isl_multi_val_mod_val(mv: MultiVal, v: Val) -> Option<MultiVal>;
  pub fn isl_multi_val_read_from_str(ctx: CtxRef, str: Option<CStr>) -> Option<MultiVal>;
  pub fn isl_printer_print_multi_val(p: Printer, mv: MultiValRef) -> Option<Printer>;
  pub fn isl_multi_val_dump(mv: MultiValRef) -> ();
  pub fn isl_multi_val_to_str(mv: MultiValRef) -> Option<CString>;
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Val(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct ValRef(pub NonNull<c_void>);

impl_try!(Val);
impl_try!(ValRef);

impl Val {
  #[inline(always)]
  pub fn read(&self) -> Val { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Val) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<ValRef> for Val {
  #[inline(always)]
  fn as_ref(&self) -> &ValRef { unsafe { mem::transmute(self) } }
}

impl Deref for Val {
  type Target = ValRef;
  #[inline(always)]
  fn deref(&self) -> &ValRef { self.as_ref() }
}

impl To<Option<Val>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Val> { NonNull::new(self).map(Val) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct ValList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct ValListRef(pub NonNull<c_void>);

impl_try!(ValList);
impl_try!(ValListRef);

impl ValList {
  #[inline(always)]
  pub fn read(&self) -> ValList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: ValList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<ValListRef> for ValList {
  #[inline(always)]
  fn as_ref(&self) -> &ValListRef { unsafe { mem::transmute(self) } }
}

impl Deref for ValList {
  type Target = ValListRef;
  #[inline(always)]
  fn deref(&self) -> &ValListRef { self.as_ref() }
}

impl To<Option<ValList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<ValList> { NonNull::new(self).map(ValList) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MultiVal(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct MultiValRef(pub NonNull<c_void>);

impl_try!(MultiVal);
impl_try!(MultiValRef);

impl MultiVal {
  #[inline(always)]
  pub fn read(&self) -> MultiVal { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: MultiVal) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MultiValRef> for MultiVal {
  #[inline(always)]
  fn as_ref(&self) -> &MultiValRef { unsafe { mem::transmute(self) } }
}

impl Deref for MultiVal {
  type Target = MultiValRef;
  #[inline(always)]
  fn deref(&self) -> &MultiValRef { self.as_ref() }
}

impl To<Option<MultiVal>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MultiVal> { NonNull::new(self).map(MultiVal) }
}

impl CtxRef {
  #[inline(always)]
  pub fn val_list_alloc(self, n: c_int) -> Option<ValList> {
    unsafe {
      let ret = isl_val_list_alloc(self.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn val_zero(self) -> Option<Val> {
    unsafe {
      let ret = isl_val_zero(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn val_one(self) -> Option<Val> {
    unsafe {
      let ret = isl_val_one(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn val_negone(self) -> Option<Val> {
    unsafe {
      let ret = isl_val_negone(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn val_nan(self) -> Option<Val> {
    unsafe {
      let ret = isl_val_nan(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn val_infty(self) -> Option<Val> {
    unsafe {
      let ret = isl_val_infty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn val_neginfty(self) -> Option<Val> {
    unsafe {
      let ret = isl_val_neginfty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn val_int_from_si(self, i: c_long) -> Option<Val> {
    unsafe {
      let ret = isl_val_int_from_si(self.to(), i.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn val_int_from_ui(self, u: c_ulong) -> Option<Val> {
    unsafe {
      let ret = isl_val_int_from_ui(self.to(), u.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn val_int_from_chunks(self, n: c_int, size: c_int, chunks: *mut c_void) -> Option<Val> {
    unsafe {
      let ret = isl_val_int_from_chunks(self.to(), n.to(), size.to(), chunks.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn val_read_from_str(self, str: Option<CStr>) -> Option<Val> {
    unsafe {
      let ret = isl_val_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_val_read_from_str(self, str: Option<CStr>) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
}

impl MultiVal {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_multi_val_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_name(self, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_set_dim_name(self.to(), type_.to(), pos.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_id(self, type_: DimType, pos: c_uint, id: Id) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_set_dim_id(self.to(), type_.to(), pos.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_name(self, type_: DimType, s: Option<CStr>) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_set_tuple_name(self.to(), type_.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_id(self, type_: DimType, id: Id) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_set_tuple_id(self.to(), type_.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_tuple_id(self, type_: DimType) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_reset_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_drop_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_val(self, pos: c_int, el: Val) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_set_val(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_splice(self, pos: c_uint, multi2: MultiVal) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_range_splice(self.to(), pos.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten_range(self) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_flatten_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flat_range_product(self, multi2: MultiVal) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_flat_range_product(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_product(self, multi2: MultiVal) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_range_product(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn factor_range(self) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_factor_domain(self) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_range_factor_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_factor_range(self) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_range_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_val(self, v: Val) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_scale_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_val(self, v: Val) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_scale_down_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_multi_val(self, mv: MultiVal) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_scale_multi_val(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_multi_val(self, mv: MultiVal) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_scale_down_multi_val(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mod_multi_val(self, mv: MultiVal) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_mod_multi_val(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, multi2: MultiVal) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_add(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sub(self, multi2: MultiVal) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_sub(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_align_params(self.to(), model.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn from_range(self) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_from_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg(self) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_insert_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_dims(self, type_: DimType, n: c_uint) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_add_dims(self.to(), type_.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn project_domain_on_params(self) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_project_domain_on_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn product(self, multi2: MultiVal) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_product(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn splice(self, in_pos: c_uint, out_pos: c_uint, multi2: MultiVal) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_splice(self.to(), in_pos.to(), out_pos.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_val(self, v: Val) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_add_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mod_val(self, v: Val) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_mod_val(self.to(), v.to());
      (ret).to()
    }
  }
}

impl MultiValRef {
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_uint {
    unsafe {
      let ret = isl_multi_val_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_multi_val_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_multi_val_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_domain_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_multi_val_get_domain_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: Option<CStr>) -> c_int {
    unsafe {
      let ret = isl_multi_val_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, multi2: MultiValRef) -> Bool {
    unsafe {
      let ret = isl_multi_val_plain_is_equal(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_nan(self) -> Bool {
    unsafe {
      let ret = isl_multi_val_involves_nan(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_id(self, type_: DimType, id: IdRef) -> c_int {
    unsafe {
      let ret = isl_multi_val_find_dim_by_id(self.to(), type_.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_id(self, type_: DimType, pos: c_uint) -> Option<Id> {
    unsafe {
      let ret = isl_multi_val_get_dim_id(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_name(self, type_: DimType) -> Option<CStr> {
    unsafe {
      let ret = isl_multi_val_get_tuple_name(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_tuple_id(self, type_: DimType) -> Bool {
    unsafe {
      let ret = isl_multi_val_has_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_id(self, type_: DimType) -> Option<Id> {
    unsafe {
      let ret = isl_multi_val_get_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_val(self, pos: c_int) -> Option<Val> {
    unsafe {
      let ret = isl_multi_val_get_val(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_is_wrapping(self) -> Bool {
    unsafe {
      let ret = isl_multi_val_range_is_wrapping(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Bool {
    unsafe {
      let ret = isl_multi_val_involves_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_multi_val_dump(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_multi_val_to_str(self.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_val_list(self, list: ValListRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_val_list(self.to(), list.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_val(self, v: ValRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_multi_val(self, mv: MultiValRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_multi_val(self.to(), mv.to());
      (ret).to()
    }
  }
}

impl Space {
  #[inline(always)]
  pub fn multi_val_from_val_list(self, list: ValList) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_from_val_list(self.to(), list.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_val_zero(self) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_val_zero(self.to());
      (ret).to()
    }
  }
}

impl Val {
  #[inline(always)]
  pub fn list_from_val(self) -> Option<ValList> {
    unsafe {
      let ret = isl_val_list_from_val(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_val_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_si(self, i: c_long) -> Option<Val> {
    unsafe {
      let ret = isl_val_set_si(self.to(), i.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn abs(self) -> Option<Val> {
    unsafe {
      let ret = isl_val_abs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg(self) -> Option<Val> {
    unsafe {
      let ret = isl_val_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn inv(self) -> Option<Val> {
    unsafe {
      let ret = isl_val_inv(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn floor(self) -> Option<Val> {
    unsafe {
      let ret = isl_val_floor(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ceil(self) -> Option<Val> {
    unsafe {
      let ret = isl_val_ceil(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn trunc(self) -> Option<Val> {
    unsafe {
      let ret = isl_val_trunc(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn exp2(self) -> Option<Val> {
    unsafe {
      let ret = isl_val_2exp(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn min(self, v2: Val) -> Option<Val> {
    unsafe {
      let ret = isl_val_min(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn max(self, v2: Val) -> Option<Val> {
    unsafe {
      let ret = isl_val_max(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, v2: Val) -> Option<Val> {
    unsafe {
      let ret = isl_val_add(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_ui(self, v2: c_ulong) -> Option<Val> {
    unsafe {
      let ret = isl_val_add_ui(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sub(self, v2: Val) -> Option<Val> {
    unsafe {
      let ret = isl_val_sub(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sub_ui(self, v2: c_ulong) -> Option<Val> {
    unsafe {
      let ret = isl_val_sub_ui(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mul(self, v2: Val) -> Option<Val> {
    unsafe {
      let ret = isl_val_mul(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mul_ui(self, v2: c_ulong) -> Option<Val> {
    unsafe {
      let ret = isl_val_mul_ui(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn div(self, v2: Val) -> Option<Val> {
    unsafe {
      let ret = isl_val_div(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn div_ui(self, v2: c_ulong) -> Option<Val> {
    unsafe {
      let ret = isl_val_div_ui(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mod_(self, v2: Val) -> Option<Val> {
    unsafe {
      let ret = isl_val_mod(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gcd(self, v2: Val) -> Option<Val> {
    unsafe {
      let ret = isl_val_gcd(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gcdext(self, v2: Val) -> Option<(Val, Val, Val)> {
    unsafe {
      let ref mut x = 0 as *mut c_void;
      let ref mut y = 0 as *mut c_void;
      let ret = isl_val_gcdext(self.to(), v2.to(), x as *mut _ as _, y as *mut _ as _);
      (ret, *x, *y).to()
    }
  }
}

impl ValList {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_val_list_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, el: Val) -> Option<ValList> {
    unsafe {
      let ret = isl_val_list_add(self.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert(self, pos: c_uint, el: Val) -> Option<ValList> {
    unsafe {
      let ret = isl_val_list_insert(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, first: c_uint, n: c_uint) -> Option<ValList> {
    unsafe {
      let ret = isl_val_list_drop(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, list2: ValList) -> Option<ValList> {
    unsafe {
      let ret = isl_val_list_concat(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_val(self, index: c_int, el: Val) -> Option<ValList> {
    unsafe {
      let ret = isl_val_list_set_val(self.to(), index.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map<F1: FnMut(Val) -> Option<Val>>(self, fn_: &mut F1) -> Option<ValList> {
    unsafe extern "C" fn fn1<F: FnMut(Val) -> Option<Val>>(el: Val, user: *mut c_void) -> Option<Val> { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_val_list_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort<F1: FnMut(ValRef, ValRef) -> c_int>(self, cmp: &mut F1) -> Option<ValList> {
    unsafe extern "C" fn fn1<F: FnMut(ValRef, ValRef) -> c_int>(a: ValRef, b: ValRef, user: *mut c_void) -> c_int { (*(user as *mut F))(a.to(), b.to()) }
    unsafe {
      let ret = isl_val_list_sort(self.to(), fn1::<F1>, cmp as *mut _ as _);
      (ret).to()
    }
  }
}

impl ValListRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_val_list_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<ValList> {
    unsafe {
      let ret = isl_val_list_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_val(self) -> c_int {
    unsafe {
      let ret = isl_val_list_n_val(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_val(self, index: c_int) -> Option<Val> {
    unsafe {
      let ret = isl_val_list_get_val(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(Val) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(Val) -> Stat>(el: Val, user: *mut c_void) -> Stat { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_val_list_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_scc<F1: FnMut(ValRef, ValRef) -> Bool, F2: FnMut(ValList) -> Stat>(self, follows: &mut F1, fn_: &mut F2) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(ValRef, ValRef) -> Bool>(a: ValRef, b: ValRef, user: *mut c_void) -> Bool { (*(user as *mut F))(a.to(), b.to()) }
    unsafe extern "C" fn fn2<F: FnMut(ValList) -> Stat>(scc: ValList, user: *mut c_void) -> Stat { (*(user as *mut F))(scc.to()) }
    unsafe {
      let ret = isl_val_list_foreach_scc(self.to(), fn1::<F1>, follows as *mut _ as _, fn2::<F2>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_val_list_dump(self.to());
      (ret).to()
    }
  }
}

impl ValRef {
  #[inline(always)]
  pub fn copy(self) -> Option<Val> {
    unsafe {
      let ret = isl_val_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_val_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_hash(self) -> c_uint {
    unsafe {
      let ret = isl_val_get_hash(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_num_si(self) -> c_long {
    unsafe {
      let ret = isl_val_get_num_si(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_den_si(self) -> c_long {
    unsafe {
      let ret = isl_val_get_den_si(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_den_val(self) -> Option<Val> {
    unsafe {
      let ret = isl_val_get_den_val(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_d(self) -> c_double {
    unsafe {
      let ret = isl_val_get_d(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_abs_num_chunks(self, size: c_int) -> c_int {
    unsafe {
      let ret = isl_val_n_abs_num_chunks(self.to(), size.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_abs_num_chunks(self, size: c_int, chunks: *mut c_void) -> c_int {
    unsafe {
      let ret = isl_val_get_abs_num_chunks(self.to(), size.to(), chunks.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sgn(self) -> c_int {
    unsafe {
      let ret = isl_val_sgn(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_zero(self) -> Bool {
    unsafe {
      let ret = isl_val_is_zero(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_one(self) -> Bool {
    unsafe {
      let ret = isl_val_is_one(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_negone(self) -> Bool {
    unsafe {
      let ret = isl_val_is_negone(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_nonneg(self) -> Bool {
    unsafe {
      let ret = isl_val_is_nonneg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_nonpos(self) -> Bool {
    unsafe {
      let ret = isl_val_is_nonpos(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_pos(self) -> Bool {
    unsafe {
      let ret = isl_val_is_pos(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_neg(self) -> Bool {
    unsafe {
      let ret = isl_val_is_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_int(self) -> Bool {
    unsafe {
      let ret = isl_val_is_int(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_rat(self) -> Bool {
    unsafe {
      let ret = isl_val_is_rat(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_nan(self) -> Bool {
    unsafe {
      let ret = isl_val_is_nan(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_infty(self) -> Bool {
    unsafe {
      let ret = isl_val_is_infty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_neginfty(self) -> Bool {
    unsafe {
      let ret = isl_val_is_neginfty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn cmp_si(self, i: c_long) -> c_int {
    unsafe {
      let ret = isl_val_cmp_si(self.to(), i.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lt(self, v2: ValRef) -> Bool {
    unsafe {
      let ret = isl_val_lt(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn le(self, v2: ValRef) -> Bool {
    unsafe {
      let ret = isl_val_le(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gt(self, v2: ValRef) -> Bool {
    unsafe {
      let ret = isl_val_gt(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ge(self, v2: ValRef) -> Bool {
    unsafe {
      let ret = isl_val_ge(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eq(self, v2: ValRef) -> Bool {
    unsafe {
      let ret = isl_val_eq(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ne(self, v2: ValRef) -> Bool {
    unsafe {
      let ret = isl_val_ne(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn abs_eq(self, v2: ValRef) -> Bool {
    unsafe {
      let ret = isl_val_abs_eq(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_divisible_by(self, v2: ValRef) -> Bool {
    unsafe {
      let ret = isl_val_is_divisible_by(self.to(), v2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_val_dump(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_val_to_str(self.to());
      (ret).to()
    }
  }
}

impl Drop for MultiVal {
  fn drop(&mut self) { MultiVal(self.0).free() }
}

impl fmt::Display for MultiValRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for MultiVal {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl Drop for Val {
  fn drop(&mut self) { Val(self.0).free() }
}

impl Drop for ValList {
  fn drop(&mut self) { ValList(self.0).free() }
}

impl fmt::Display for ValRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for Val {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

