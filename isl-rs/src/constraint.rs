use crate::*;

extern "C" {
  pub fn isl_constraint_list_get_ctx(list: ConstraintListRef) -> Option<CtxRef>;
  pub fn isl_constraint_list_from_constraint(el: Constraint) -> Option<ConstraintList>;
  pub fn isl_constraint_list_alloc(ctx: CtxRef, n: c_int) -> Option<ConstraintList>;
  pub fn isl_constraint_list_copy(list: ConstraintListRef) -> Option<ConstraintList>;
  pub fn isl_constraint_list_free(list: ConstraintList) -> *mut c_void;
  pub fn isl_constraint_list_add(list: ConstraintList, el: Constraint) -> Option<ConstraintList>;
  pub fn isl_constraint_list_insert(list: ConstraintList, pos: c_uint, el: Constraint) -> Option<ConstraintList>;
  pub fn isl_constraint_list_drop(list: ConstraintList, first: c_uint, n: c_uint) -> Option<ConstraintList>;
  pub fn isl_constraint_list_clear(list: ConstraintList) -> Option<ConstraintList>;
  pub fn isl_constraint_list_swap(list: ConstraintList, pos1: c_uint, pos2: c_uint) -> Option<ConstraintList>;
  pub fn isl_constraint_list_reverse(list: ConstraintList) -> Option<ConstraintList>;
  pub fn isl_constraint_list_concat(list1: ConstraintList, list2: ConstraintList) -> Option<ConstraintList>;
  pub fn isl_constraint_list_size(list: ConstraintListRef) -> c_int;
  pub fn isl_constraint_list_n_constraint(list: ConstraintListRef) -> c_int;
  pub fn isl_constraint_list_get_at(list: ConstraintListRef, index: c_int) -> Option<Constraint>;
  pub fn isl_constraint_list_get_constraint(list: ConstraintListRef, index: c_int) -> Option<Constraint>;
  pub fn isl_constraint_list_set_constraint(list: ConstraintList, index: c_int, el: Constraint) -> Option<ConstraintList>;
  pub fn isl_constraint_list_foreach(list: ConstraintListRef, fn_: unsafe extern "C" fn(el: Constraint, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_constraint_list_every(list: ConstraintListRef, test: unsafe extern "C" fn(el: ConstraintRef, user: *mut c_void) -> Bool, user: *mut c_void) -> Bool;
  pub fn isl_constraint_list_map(list: ConstraintList, fn_: unsafe extern "C" fn(el: Constraint, user: *mut c_void) -> Option<Constraint>, user: *mut c_void) -> Option<ConstraintList>;
  pub fn isl_constraint_list_sort(list: ConstraintList, cmp: unsafe extern "C" fn(a: ConstraintRef, b: ConstraintRef, user: *mut c_void) -> c_int, user: *mut c_void) -> Option<ConstraintList>;
  pub fn isl_constraint_list_foreach_scc(list: ConstraintListRef, follows: unsafe extern "C" fn(a: ConstraintRef, b: ConstraintRef, user: *mut c_void) -> Bool, follows_user: *mut c_void, fn_: unsafe extern "C" fn(scc: ConstraintList, user: *mut c_void) -> Stat, fn_user: *mut c_void) -> Stat;
  pub fn isl_constraint_list_to_str(list: ConstraintListRef) -> Option<CString>;
  pub fn isl_printer_print_constraint_list(p: Printer, list: ConstraintListRef) -> Option<Printer>;
  pub fn isl_constraint_list_dump(list: ConstraintListRef) -> ();
  pub fn isl_constraint_get_ctx(c: ConstraintRef) -> Option<CtxRef>;
  pub fn isl_constraint_alloc_equality(ls: LocalSpace) -> Option<Constraint>;
  pub fn isl_constraint_alloc_inequality(ls: LocalSpace) -> Option<Constraint>;
  pub fn isl_equality_alloc(ls: LocalSpace) -> Option<Constraint>;
  pub fn isl_inequality_alloc(ls: LocalSpace) -> Option<Constraint>;
  pub fn isl_constraint_copy(c: ConstraintRef) -> Option<Constraint>;
  pub fn isl_constraint_free(c: Constraint) -> *mut c_void;
  pub fn isl_basic_map_n_constraint(bmap: BasicMapRef) -> c_int;
  pub fn isl_basic_set_n_constraint(bset: BasicSetRef) -> c_int;
  pub fn isl_basic_map_foreach_constraint(bmap: BasicMapRef, fn_: unsafe extern "C" fn(c: Constraint, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_basic_set_foreach_constraint(bset: BasicSetRef, fn_: unsafe extern "C" fn(c: Constraint, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_basic_map_get_constraint_list(bmap: BasicMapRef) -> Option<ConstraintList>;
  pub fn isl_basic_set_get_constraint_list(bset: BasicSetRef) -> Option<ConstraintList>;
  pub fn isl_constraint_is_equal(constraint1: ConstraintRef, constraint2: ConstraintRef) -> c_int;
  pub fn isl_basic_set_foreach_bound_pair(bset: BasicSetRef, type_: DimType, pos: c_uint, fn_: unsafe extern "C" fn(lower: Constraint, upper: Constraint, bset: BasicSet, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_basic_map_add_constraint(bmap: BasicMap, constraint: Constraint) -> Option<BasicMap>;
  pub fn isl_basic_set_add_constraint(bset: BasicSet, constraint: Constraint) -> Option<BasicSet>;
  pub fn isl_map_add_constraint(map: Map, constraint: Constraint) -> Option<Map>;
  pub fn isl_set_add_constraint(set: Set, constraint: Constraint) -> Option<Set>;
  pub fn isl_basic_map_has_defining_equality(bmap: BasicMapRef, type_: DimType, pos: c_int, c: *mut Constraint) -> Bool;
  pub fn isl_basic_set_has_defining_equality(bset: BasicSetRef, type_: DimType, pos: c_int, constraint: *mut Constraint) -> Bool;
  pub fn isl_basic_set_has_defining_inequalities(bset: BasicSetRef, type_: DimType, pos: c_int, lower: *mut Constraint, upper: *mut Constraint) -> Bool;
  pub fn isl_constraint_get_space(constraint: ConstraintRef) -> Option<Space>;
  pub fn isl_constraint_get_local_space(constraint: ConstraintRef) -> Option<LocalSpace>;
  pub fn isl_constraint_dim(constraint: ConstraintRef, type_: DimType) -> c_int;
  pub fn isl_constraint_involves_dims(constraint: ConstraintRef, type_: DimType, first: c_uint, n: c_uint) -> Bool;
  pub fn isl_constraint_get_dim_name(constraint: ConstraintRef, type_: DimType, pos: c_uint) -> Option<CStr>;
  pub fn isl_constraint_get_constant_val(constraint: ConstraintRef) -> Option<Val>;
  pub fn isl_constraint_get_coefficient_val(constraint: ConstraintRef, type_: DimType, pos: c_int) -> Option<Val>;
  pub fn isl_constraint_set_constant_si(constraint: Constraint, v: c_int) -> Option<Constraint>;
  pub fn isl_constraint_set_constant_val(constraint: Constraint, v: Val) -> Option<Constraint>;
  pub fn isl_constraint_set_coefficient_si(constraint: Constraint, type_: DimType, pos: c_int, v: c_int) -> Option<Constraint>;
  pub fn isl_constraint_set_coefficient_val(constraint: Constraint, type_: DimType, pos: c_int, v: Val) -> Option<Constraint>;
  pub fn isl_constraint_get_div(constraint: ConstraintRef, pos: c_int) -> Option<Aff>;
  pub fn isl_constraint_negate(constraint: Constraint) -> Option<Constraint>;
  pub fn isl_constraint_is_equality(constraint: ConstraintRef) -> Bool;
  pub fn isl_constraint_is_div_constraint(constraint: ConstraintRef) -> Bool;
  pub fn isl_constraint_is_lower_bound(constraint: ConstraintRef, type_: DimType, pos: c_uint) -> Bool;
  pub fn isl_constraint_is_upper_bound(constraint: ConstraintRef, type_: DimType, pos: c_uint) -> Bool;
  pub fn isl_basic_map_from_constraint(constraint: Constraint) -> Option<BasicMap>;
  pub fn isl_basic_set_from_constraint(constraint: Constraint) -> Option<BasicSet>;
  pub fn isl_constraint_get_bound(constraint: ConstraintRef, type_: DimType, pos: c_int) -> Option<Aff>;
  pub fn isl_constraint_get_aff(constraint: ConstraintRef) -> Option<Aff>;
  pub fn isl_equality_from_aff(aff: Aff) -> Option<Constraint>;
  pub fn isl_inequality_from_aff(aff: Aff) -> Option<Constraint>;
  pub fn isl_constraint_plain_cmp(c1: ConstraintRef, c2: ConstraintRef) -> c_int;
  pub fn isl_constraint_cmp_last_non_zero(c1: ConstraintRef, c2: ConstraintRef) -> c_int;
  pub fn isl_printer_print_constraint(p: Printer, c: ConstraintRef) -> Option<Printer>;
  pub fn isl_constraint_dump(c: ConstraintRef) -> ();
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Constraint(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct ConstraintRef(pub NonNull<c_void>);

impl_try!(Constraint);
impl_try!(ConstraintRef);

impl Constraint {
  #[inline(always)]
  pub fn read(&self) -> Constraint { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Constraint) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<ConstraintRef> for Constraint {
  #[inline(always)]
  fn as_ref(&self) -> &ConstraintRef { unsafe { mem::transmute(self) } }
}

impl Deref for Constraint {
  type Target = ConstraintRef;
  #[inline(always)]
  fn deref(&self) -> &ConstraintRef { self.as_ref() }
}

impl To<Option<Constraint>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Constraint> { NonNull::new(self).map(Constraint) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct ConstraintList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct ConstraintListRef(pub NonNull<c_void>);

impl_try!(ConstraintList);
impl_try!(ConstraintListRef);

impl ConstraintList {
  #[inline(always)]
  pub fn read(&self) -> ConstraintList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: ConstraintList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<ConstraintListRef> for ConstraintList {
  #[inline(always)]
  fn as_ref(&self) -> &ConstraintListRef { unsafe { mem::transmute(self) } }
}

impl Deref for ConstraintList {
  type Target = ConstraintListRef;
  #[inline(always)]
  fn deref(&self) -> &ConstraintListRef { self.as_ref() }
}

impl To<Option<ConstraintList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<ConstraintList> { NonNull::new(self).map(ConstraintList) }
}

impl Aff {
  #[inline(always)]
  pub fn equality_from_aff(self) -> Option<Constraint> {
    unsafe {
      let ret = isl_equality_from_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn inequality_from_aff(self) -> Option<Constraint> {
    unsafe {
      let ret = isl_inequality_from_aff(self.to());
      (ret).to()
    }
  }
}

impl BasicMap {
  #[inline(always)]
  pub fn add_constraint(self, constraint: Constraint) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_add_constraint(self.to(), constraint.to());
      (ret).to()
    }
  }
}

impl BasicMapRef {
  #[inline(always)]
  pub fn n_constraint(self) -> c_int {
    unsafe {
      let ret = isl_basic_map_n_constraint(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_constraint<F1: FnMut(Constraint) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(Constraint) -> Stat>(c: Constraint, user: *mut c_void) -> Stat { (*(user as *mut F))(c.to()) }
    unsafe {
      let ret = isl_basic_map_foreach_constraint(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_constraint_list(self) -> Option<ConstraintList> {
    unsafe {
      let ret = isl_basic_map_get_constraint_list(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_defining_equality(self, type_: DimType, pos: c_int) -> Option<(Bool, Constraint)> {
    unsafe {
      let ref mut c = 0 as *mut c_void;
      let ret = isl_basic_map_has_defining_equality(self.to(), type_.to(), pos.to(), c as *mut _ as _);
      (ret, *c).to()
    }
  }
}

impl BasicSet {
  #[inline(always)]
  pub fn add_constraint(self, constraint: Constraint) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_add_constraint(self.to(), constraint.to());
      (ret).to()
    }
  }
}

impl BasicSetRef {
  #[inline(always)]
  pub fn n_constraint(self) -> c_int {
    unsafe {
      let ret = isl_basic_set_n_constraint(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_constraint<F1: FnMut(Constraint) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(Constraint) -> Stat>(c: Constraint, user: *mut c_void) -> Stat { (*(user as *mut F))(c.to()) }
    unsafe {
      let ret = isl_basic_set_foreach_constraint(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_constraint_list(self) -> Option<ConstraintList> {
    unsafe {
      let ret = isl_basic_set_get_constraint_list(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_bound_pair<F1: FnMut(Constraint, Constraint, BasicSet) -> Stat>(self, type_: DimType, pos: c_uint, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(Constraint, Constraint, BasicSet) -> Stat>(lower: Constraint, upper: Constraint, bset: BasicSet, user: *mut c_void) -> Stat { (*(user as *mut F))(lower.to(), upper.to(), bset.to()) }
    unsafe {
      let ret = isl_basic_set_foreach_bound_pair(self.to(), type_.to(), pos.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_defining_equality(self, type_: DimType, pos: c_int) -> Option<(Bool, Constraint)> {
    unsafe {
      let ref mut constraint = 0 as *mut c_void;
      let ret = isl_basic_set_has_defining_equality(self.to(), type_.to(), pos.to(), constraint as *mut _ as _);
      (ret, *constraint).to()
    }
  }
  #[inline(always)]
  pub fn has_defining_inequalities(self, type_: DimType, pos: c_int) -> Option<(Bool, Constraint, Constraint)> {
    unsafe {
      let ref mut lower = 0 as *mut c_void;
      let ref mut upper = 0 as *mut c_void;
      let ret = isl_basic_set_has_defining_inequalities(self.to(), type_.to(), pos.to(), lower as *mut _ as _, upper as *mut _ as _);
      (ret, *lower, *upper).to()
    }
  }
}

impl Constraint {
  #[inline(always)]
  pub fn list_from_constraint(self) -> Option<ConstraintList> {
    unsafe {
      let ret = isl_constraint_list_from_constraint(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_constraint_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_constant_si(self, v: c_int) -> Option<Constraint> {
    unsafe {
      let ret = isl_constraint_set_constant_si(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_constant_val(self, v: Val) -> Option<Constraint> {
    unsafe {
      let ret = isl_constraint_set_constant_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_coefficient_si(self, type_: DimType, pos: c_int, v: c_int) -> Option<Constraint> {
    unsafe {
      let ret = isl_constraint_set_coefficient_si(self.to(), type_.to(), pos.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_coefficient_val(self, type_: DimType, pos: c_int, v: Val) -> Option<Constraint> {
    unsafe {
      let ret = isl_constraint_set_coefficient_val(self.to(), type_.to(), pos.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn negate(self) -> Option<Constraint> {
    unsafe {
      let ret = isl_constraint_negate(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_map_from_constraint(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_from_constraint(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_set_from_constraint(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_from_constraint(self.to());
      (ret).to()
    }
  }
}

impl ConstraintList {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_constraint_list_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, el: Constraint) -> Option<ConstraintList> {
    unsafe {
      let ret = isl_constraint_list_add(self.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert(self, pos: c_uint, el: Constraint) -> Option<ConstraintList> {
    unsafe {
      let ret = isl_constraint_list_insert(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, first: c_uint, n: c_uint) -> Option<ConstraintList> {
    unsafe {
      let ret = isl_constraint_list_drop(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn clear(self) -> Option<ConstraintList> {
    unsafe {
      let ret = isl_constraint_list_clear(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn swap(self, pos1: c_uint, pos2: c_uint) -> Option<ConstraintList> {
    unsafe {
      let ret = isl_constraint_list_swap(self.to(), pos1.to(), pos2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reverse(self) -> Option<ConstraintList> {
    unsafe {
      let ret = isl_constraint_list_reverse(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, list2: ConstraintList) -> Option<ConstraintList> {
    unsafe {
      let ret = isl_constraint_list_concat(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_constraint(self, index: c_int, el: Constraint) -> Option<ConstraintList> {
    unsafe {
      let ret = isl_constraint_list_set_constraint(self.to(), index.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map<F1: FnMut(Constraint) -> Option<Constraint>>(self, fn_: &mut F1) -> Option<ConstraintList> {
    unsafe extern "C" fn fn1<F: FnMut(Constraint) -> Option<Constraint>>(el: Constraint, user: *mut c_void) -> Option<Constraint> { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_constraint_list_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort<F1: FnMut(ConstraintRef, ConstraintRef) -> c_int>(self, cmp: &mut F1) -> Option<ConstraintList> {
    unsafe extern "C" fn fn1<F: FnMut(ConstraintRef, ConstraintRef) -> c_int>(a: ConstraintRef, b: ConstraintRef, user: *mut c_void) -> c_int { (*(user as *mut F))(a.to(), b.to()) }
    unsafe {
      let ret = isl_constraint_list_sort(self.to(), fn1::<F1>, cmp as *mut _ as _);
      (ret).to()
    }
  }
}

impl ConstraintListRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_constraint_list_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<ConstraintList> {
    unsafe {
      let ret = isl_constraint_list_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn size(self) -> c_int {
    unsafe {
      let ret = isl_constraint_list_size(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_constraint(self) -> c_int {
    unsafe {
      let ret = isl_constraint_list_n_constraint(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_at(self, index: c_int) -> Option<Constraint> {
    unsafe {
      let ret = isl_constraint_list_get_at(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_constraint(self, index: c_int) -> Option<Constraint> {
    unsafe {
      let ret = isl_constraint_list_get_constraint(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(Constraint) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(Constraint) -> Stat>(el: Constraint, user: *mut c_void) -> Stat { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_constraint_list_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn every<F1: FnMut(ConstraintRef) -> Bool>(self, test: &mut F1) -> Bool {
    unsafe extern "C" fn fn1<F: FnMut(ConstraintRef) -> Bool>(el: ConstraintRef, user: *mut c_void) -> Bool { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_constraint_list_every(self.to(), fn1::<F1>, test as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_scc<F1: FnMut(ConstraintRef, ConstraintRef) -> Bool, F2: FnMut(ConstraintList) -> Stat>(self, follows: &mut F1, fn_: &mut F2) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(ConstraintRef, ConstraintRef) -> Bool>(a: ConstraintRef, b: ConstraintRef, user: *mut c_void) -> Bool { (*(user as *mut F))(a.to(), b.to()) }
    unsafe extern "C" fn fn2<F: FnMut(ConstraintList) -> Stat>(scc: ConstraintList, user: *mut c_void) -> Stat { (*(user as *mut F))(scc.to()) }
    unsafe {
      let ret = isl_constraint_list_foreach_scc(self.to(), fn1::<F1>, follows as *mut _ as _, fn2::<F2>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_constraint_list_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_constraint_list_dump(self.to());
      (ret).to()
    }
  }
}

impl ConstraintRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_constraint_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<Constraint> {
    unsafe {
      let ret = isl_constraint_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_equal(self, constraint2: ConstraintRef) -> c_int {
    unsafe {
      let ret = isl_constraint_is_equal(self.to(), constraint2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_constraint_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_local_space(self) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_constraint_get_local_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_int {
    unsafe {
      let ret = isl_constraint_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Bool {
    unsafe {
      let ret = isl_constraint_involves_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_name(self, type_: DimType, pos: c_uint) -> Option<CStr> {
    unsafe {
      let ret = isl_constraint_get_dim_name(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_constant_val(self) -> Option<Val> {
    unsafe {
      let ret = isl_constraint_get_constant_val(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_coefficient_val(self, type_: DimType, pos: c_int) -> Option<Val> {
    unsafe {
      let ret = isl_constraint_get_coefficient_val(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_div(self, pos: c_int) -> Option<Aff> {
    unsafe {
      let ret = isl_constraint_get_div(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_equality(self) -> Bool {
    unsafe {
      let ret = isl_constraint_is_equality(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_div_constraint(self) -> Bool {
    unsafe {
      let ret = isl_constraint_is_div_constraint(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_lower_bound(self, type_: DimType, pos: c_uint) -> Bool {
    unsafe {
      let ret = isl_constraint_is_lower_bound(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_upper_bound(self, type_: DimType, pos: c_uint) -> Bool {
    unsafe {
      let ret = isl_constraint_is_upper_bound(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_bound(self, type_: DimType, pos: c_int) -> Option<Aff> {
    unsafe {
      let ret = isl_constraint_get_bound(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_aff(self) -> Option<Aff> {
    unsafe {
      let ret = isl_constraint_get_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_cmp(self, c2: ConstraintRef) -> c_int {
    unsafe {
      let ret = isl_constraint_plain_cmp(self.to(), c2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn cmp_last_non_zero(self, c2: ConstraintRef) -> c_int {
    unsafe {
      let ret = isl_constraint_cmp_last_non_zero(self.to(), c2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_constraint_dump(self.to());
      (ret).to()
    }
  }
}

impl CtxRef {
  #[inline(always)]
  pub fn constraint_list_alloc(self, n: c_int) -> Option<ConstraintList> {
    unsafe {
      let ret = isl_constraint_list_alloc(self.to(), n.to());
      (ret).to()
    }
  }
}

impl LocalSpace {
  #[inline(always)]
  pub fn constraint_alloc_equality(self) -> Option<Constraint> {
    unsafe {
      let ret = isl_constraint_alloc_equality(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn constraint_alloc_inequality(self) -> Option<Constraint> {
    unsafe {
      let ret = isl_constraint_alloc_inequality(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn equality_alloc(self) -> Option<Constraint> {
    unsafe {
      let ret = isl_equality_alloc(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn inequality_alloc(self) -> Option<Constraint> {
    unsafe {
      let ret = isl_inequality_alloc(self.to());
      (ret).to()
    }
  }
}

impl Map {
  #[inline(always)]
  pub fn add_constraint(self, constraint: Constraint) -> Option<Map> {
    unsafe {
      let ret = isl_map_add_constraint(self.to(), constraint.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_constraint_list(self, list: ConstraintListRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_constraint_list(self.to(), list.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_constraint(self, c: ConstraintRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_constraint(self.to(), c.to());
      (ret).to()
    }
  }
}

impl Set {
  #[inline(always)]
  pub fn add_constraint(self, constraint: Constraint) -> Option<Set> {
    unsafe {
      let ret = isl_set_add_constraint(self.to(), constraint.to());
      (ret).to()
    }
  }
}

impl Drop for Constraint {
  fn drop(&mut self) { Constraint(self.0).free() }
}

impl Drop for ConstraintList {
  fn drop(&mut self) { ConstraintList(self.0).free() }
}

impl fmt::Display for ConstraintListRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for ConstraintList {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

