use crate::*;

extern "C" {
  pub fn isl_union_set_dim(uset: UnionSetRef, type_: DimType) -> c_uint;
  pub fn isl_union_set_from_basic_set(bset: BasicSet) -> Option<UnionSet>;
  pub fn isl_union_set_from_set(set: Set) -> Option<UnionSet>;
  pub fn isl_union_set_empty(space: Space) -> Option<UnionSet>;
  pub fn isl_union_set_copy(uset: UnionSetRef) -> Option<UnionSet>;
  pub fn isl_union_set_free(uset: UnionSet) -> *mut c_void;
  pub fn isl_union_set_get_ctx(uset: UnionSetRef) -> Option<CtxRef>;
  pub fn isl_union_set_get_space(uset: UnionSetRef) -> Option<Space>;
  pub fn isl_union_set_reset_user(uset: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_universe(uset: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_params(uset: UnionSet) -> Option<Set>;
  pub fn isl_union_set_detect_equalities(uset: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_affine_hull(uset: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_polyhedral_hull(uset: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_remove_redundancies(uset: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_simple_hull(uset: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_coalesce(uset: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_compute_divs(uset: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_lexmin(uset: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_lexmax(uset: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_add_set(uset: UnionSet, set: Set) -> Option<UnionSet>;
  pub fn isl_union_set_union(uset1: UnionSet, uset2: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_subtract(uset1: UnionSet, uset2: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_intersect(uset1: UnionSet, uset2: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_intersect_params(uset: UnionSet, set: Set) -> Option<UnionSet>;
  pub fn isl_union_set_product(uset1: UnionSet, uset2: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_gist(uset: UnionSet, context: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_gist_params(uset: UnionSet, set: Set) -> Option<UnionSet>;
  pub fn isl_union_set_apply(uset: UnionSet, umap: UnionMap) -> Option<UnionSet>;
  pub fn isl_union_set_preimage_multi_aff(uset: UnionSet, ma: MultiAff) -> Option<UnionSet>;
  pub fn isl_union_set_preimage_pw_multi_aff(uset: UnionSet, pma: PwMultiAff) -> Option<UnionSet>;
  pub fn isl_union_set_preimage_union_pw_multi_aff(uset: UnionSet, upma: UnionPwMultiAff) -> Option<UnionSet>;
  pub fn isl_union_set_project_out(uset: UnionSet, type_: DimType, first: c_uint, n: c_uint) -> Option<UnionSet>;
  pub fn isl_union_set_remove_divs(bset: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_is_params(uset: UnionSetRef) -> Bool;
  pub fn isl_union_set_is_empty(uset: UnionSetRef) -> Bool;
  pub fn isl_union_set_is_subset(uset1: UnionSetRef, uset2: UnionSetRef) -> Bool;
  pub fn isl_union_set_is_equal(uset1: UnionSetRef, uset2: UnionSetRef) -> Bool;
  pub fn isl_union_set_is_disjoint(uset1: UnionSetRef, uset2: UnionSetRef) -> Bool;
  pub fn isl_union_set_is_strict_subset(uset1: UnionSetRef, uset2: UnionSetRef) -> Bool;
  pub fn isl_union_set_get_hash(uset: UnionSetRef) -> c_uint;
  pub fn isl_union_set_n_set(uset: UnionSetRef) -> c_int;
  pub fn isl_union_set_foreach_set(uset: UnionSetRef, fn_: unsafe extern "C" fn(set: Set, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_union_set_get_basic_set_list(uset: UnionSetRef) -> Option<BasicSetList>;
  pub fn isl_union_set_contains(uset: UnionSetRef, space: SpaceRef) -> Bool;
  pub fn isl_union_set_extract_set(uset: UnionSetRef, dim: Space) -> Option<Set>;
  pub fn isl_set_from_union_set(uset: UnionSet) -> Option<Set>;
  pub fn isl_union_set_foreach_point(uset: UnionSetRef, fn_: unsafe extern "C" fn(pnt: Point, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_union_set_sample(uset: UnionSet) -> Option<BasicSet>;
  pub fn isl_union_set_sample_point(uset: UnionSet) -> Option<Point>;
  pub fn isl_union_set_from_point(pnt: Point) -> Option<UnionSet>;
  pub fn isl_union_set_lift(uset: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_lex_lt_union_set(uset1: UnionSet, uset2: UnionSet) -> Option<UnionMap>;
  pub fn isl_union_set_lex_le_union_set(uset1: UnionSet, uset2: UnionSet) -> Option<UnionMap>;
  pub fn isl_union_set_lex_gt_union_set(uset1: UnionSet, uset2: UnionSet) -> Option<UnionMap>;
  pub fn isl_union_set_lex_ge_union_set(uset1: UnionSet, uset2: UnionSet) -> Option<UnionMap>;
  pub fn isl_union_set_coefficients(bset: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_solutions(bset: UnionSet) -> Option<UnionSet>;
  pub fn isl_union_set_read_from_file(ctx: CtxRef, input: *mut FILE) -> Option<UnionSet>;
  pub fn isl_union_set_read_from_str(ctx: CtxRef, str: CStr) -> Option<UnionSet>;
  pub fn isl_union_set_to_str(uset: UnionSetRef) -> Option<CString>;
  pub fn isl_printer_print_union_set(p: Printer, uset: UnionSetRef) -> Option<Printer>;
  pub fn isl_union_set_dump(uset: UnionSetRef) -> ();
  pub fn isl_union_set_list_get_ctx(list: UnionSetListRef) -> Option<CtxRef>;
  pub fn isl_union_set_list_from_union_set(el: UnionSet) -> Option<UnionSetList>;
  pub fn isl_union_set_list_alloc(ctx: CtxRef, n: c_int) -> Option<UnionSetList>;
  pub fn isl_union_set_list_copy(list: UnionSetListRef) -> Option<UnionSetList>;
  pub fn isl_union_set_list_free(list: UnionSetList) -> *mut c_void;
  pub fn isl_union_set_list_add(list: UnionSetList, el: UnionSet) -> Option<UnionSetList>;
  pub fn isl_union_set_list_insert(list: UnionSetList, pos: c_uint, el: UnionSet) -> Option<UnionSetList>;
  pub fn isl_union_set_list_drop(list: UnionSetList, first: c_uint, n: c_uint) -> Option<UnionSetList>;
  pub fn isl_union_set_list_concat(list1: UnionSetList, list2: UnionSetList) -> Option<UnionSetList>;
  pub fn isl_union_set_list_n_union_set(list: UnionSetListRef) -> c_int;
  pub fn isl_union_set_list_get_union_set(list: UnionSetListRef, index: c_int) -> Option<UnionSet>;
  pub fn isl_union_set_list_set_union_set(list: UnionSetList, index: c_int, el: UnionSet) -> Option<UnionSetList>;
  pub fn isl_union_set_list_foreach(list: UnionSetListRef, fn_: unsafe extern "C" fn(el: UnionSet, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_union_set_list_map(list: UnionSetList, fn_: unsafe extern "C" fn(el: UnionSet, user: *mut c_void) -> Option<UnionSet>, user: *mut c_void) -> Option<UnionSetList>;
  pub fn isl_union_set_list_sort(list: UnionSetList, cmp: unsafe extern "C" fn(a: UnionSetRef, b: UnionSetRef, user: *mut c_void) -> c_int, user: *mut c_void) -> Option<UnionSetList>;
  pub fn isl_union_set_list_foreach_scc(list: UnionSetListRef, follows: unsafe extern "C" fn(a: UnionSetRef, b: UnionSetRef, user: *mut c_void) -> Bool, follows_user: *mut c_void, fn_: unsafe extern "C" fn(scc: UnionSetList, user: *mut c_void) -> Stat, fn_user: *mut c_void) -> Stat;
  pub fn isl_printer_print_union_set_list(p: Printer, list: UnionSetListRef) -> Option<Printer>;
  pub fn isl_union_set_list_dump(list: UnionSetListRef) -> ();
  pub fn isl_union_set_list_union(list: UnionSetList) -> Option<UnionSet>;
}

impl BasicSet {
  #[inline(always)]
  pub fn union_set_from_basic_set(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_from_basic_set(self.to());
      (ret).to()
    }
  }
}

impl CtxRef {
  #[inline(always)]
  pub fn union_set_read_from_file(self, input: *mut FILE) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_read_from_file(self.to(), input.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_set_read_from_str(self, str: CStr) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_set_list_alloc(self, n: c_int) -> Option<UnionSetList> {
    unsafe {
      let ret = isl_union_set_list_alloc(self.to(), n.to());
      (ret).to()
    }
  }
}

impl Point {
  #[inline(always)]
  pub fn union_set_from_point(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_from_point(self.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_union_set(self, uset: UnionSetRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_union_set(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_union_set_list(self, list: UnionSetListRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_union_set_list(self.to(), list.to());
      (ret).to()
    }
  }
}

impl Set {
  #[inline(always)]
  pub fn union_set_from_set(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_from_set(self.to());
      (ret).to()
    }
  }
}

impl Space {
  #[inline(always)]
  pub fn union_set_empty(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_empty(self.to());
      (ret).to()
    }
  }
}

impl UnionSet {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_union_set_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn universe(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_universe(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn params(self) -> Option<Set> {
    unsafe {
      let ret = isl_union_set_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn detect_equalities(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_detect_equalities(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn affine_hull(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_affine_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn polyhedral_hull(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_polyhedral_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_redundancies(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_remove_redundancies(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn simple_hull(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_simple_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coalesce(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_coalesce(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn compute_divs(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_compute_divs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lexmin(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_lexmin(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lexmax(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_lexmax(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_set(self, set: Set) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_add_set(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union(self, uset2: UnionSet) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_union(self.to(), uset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract(self, uset2: UnionSet) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_subtract(self.to(), uset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect(self, uset2: UnionSet) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_intersect(self.to(), uset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_params(self, set: Set) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_intersect_params(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn product(self, uset2: UnionSet) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_product(self.to(), uset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, context: UnionSet) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_gist(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_params(self, set: Set) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_gist_params(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn apply(self, umap: UnionMap) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_apply(self.to(), umap.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_multi_aff(self, ma: MultiAff) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_preimage_multi_aff(self.to(), ma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_pw_multi_aff(self, pma: PwMultiAff) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_preimage_pw_multi_aff(self.to(), pma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_union_pw_multi_aff(self, upma: UnionPwMultiAff) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_preimage_union_pw_multi_aff(self.to(), upma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn project_out(self, type_: DimType, first: c_uint, n: c_uint) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_project_out(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_divs(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_remove_divs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_from_union_set(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_from_union_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sample(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_union_set_sample(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sample_point(self) -> Option<Point> {
    unsafe {
      let ret = isl_union_set_sample_point(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lift(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_lift(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_lt_union_set(self, uset2: UnionSet) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_set_lex_lt_union_set(self.to(), uset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_le_union_set(self, uset2: UnionSet) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_set_lex_le_union_set(self.to(), uset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_gt_union_set(self, uset2: UnionSet) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_set_lex_gt_union_set(self.to(), uset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_ge_union_set(self, uset2: UnionSet) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_set_lex_ge_union_set(self.to(), uset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coefficients(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_coefficients(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn solutions(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_solutions(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn list_from_union_set(self) -> Option<UnionSetList> {
    unsafe {
      let ret = isl_union_set_list_from_union_set(self.to());
      (ret).to()
    }
  }
}

impl UnionSetList {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_union_set_list_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, el: UnionSet) -> Option<UnionSetList> {
    unsafe {
      let ret = isl_union_set_list_add(self.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert(self, pos: c_uint, el: UnionSet) -> Option<UnionSetList> {
    unsafe {
      let ret = isl_union_set_list_insert(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, first: c_uint, n: c_uint) -> Option<UnionSetList> {
    unsafe {
      let ret = isl_union_set_list_drop(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, list2: UnionSetList) -> Option<UnionSetList> {
    unsafe {
      let ret = isl_union_set_list_concat(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_union_set(self, index: c_int, el: UnionSet) -> Option<UnionSetList> {
    unsafe {
      let ret = isl_union_set_list_set_union_set(self.to(), index.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map<F1: FnMut(UnionSet) -> Option<UnionSet>>(self, fn_: &mut F1) -> Option<UnionSetList> {
    unsafe extern "C" fn fn1<F: FnMut(UnionSet) -> Option<UnionSet>>(el: UnionSet, user: *mut c_void) -> Option<UnionSet> { (*(user as *mut F))(el.to()).to() }
    unsafe {
      let ret = isl_union_set_list_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort<F1: FnMut(UnionSetRef, UnionSetRef) -> c_int>(self, cmp: &mut F1) -> Option<UnionSetList> {
    unsafe extern "C" fn fn1<F: FnMut(UnionSetRef, UnionSetRef) -> c_int>(a: UnionSetRef, b: UnionSetRef, user: *mut c_void) -> c_int { (*(user as *mut F))(a.to(), b.to()).to() }
    unsafe {
      let ret = isl_union_set_list_sort(self.to(), fn1::<F1>, cmp as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_list_union(self.to());
      (ret).to()
    }
  }
}

impl UnionSetListRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_union_set_list_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<UnionSetList> {
    unsafe {
      let ret = isl_union_set_list_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_union_set(self) -> c_int {
    unsafe {
      let ret = isl_union_set_list_n_union_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_union_set(self, index: c_int) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_list_get_union_set(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(UnionSet) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(UnionSet) -> Option<()>>(el: UnionSet, user: *mut c_void) -> Stat { (*(user as *mut F))(el.to()).to() }
    unsafe {
      let ret = isl_union_set_list_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_scc<F1: FnMut(UnionSetRef, UnionSetRef) -> Option<bool>, F2: FnMut(UnionSetList) -> Option<()>>(self, follows: &mut F1, fn_: &mut F2) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(UnionSetRef, UnionSetRef) -> Option<bool>>(a: UnionSetRef, b: UnionSetRef, user: *mut c_void) -> Bool { (*(user as *mut F))(a.to(), b.to()).to() }
    unsafe extern "C" fn fn2<F: FnMut(UnionSetList) -> Option<()>>(scc: UnionSetList, user: *mut c_void) -> Stat { (*(user as *mut F))(scc.to()).to() }
    unsafe {
      let ret = isl_union_set_list_foreach_scc(self.to(), fn1::<F1>, follows as *mut _ as _, fn2::<F2>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_union_set_list_dump(self.to());
      (ret).to()
    }
  }
}

impl UnionSetRef {
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_uint {
    unsafe {
      let ret = isl_union_set_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_union_set_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_union_set_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_params(self) -> Option<bool> {
    unsafe {
      let ret = isl_union_set_is_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_empty(self) -> Option<bool> {
    unsafe {
      let ret = isl_union_set_is_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_subset(self, uset2: UnionSetRef) -> Option<bool> {
    unsafe {
      let ret = isl_union_set_is_subset(self.to(), uset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_equal(self, uset2: UnionSetRef) -> Option<bool> {
    unsafe {
      let ret = isl_union_set_is_equal(self.to(), uset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_disjoint(self, uset2: UnionSetRef) -> Option<bool> {
    unsafe {
      let ret = isl_union_set_is_disjoint(self.to(), uset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_strict_subset(self, uset2: UnionSetRef) -> Option<bool> {
    unsafe {
      let ret = isl_union_set_is_strict_subset(self.to(), uset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_hash(self) -> c_uint {
    unsafe {
      let ret = isl_union_set_get_hash(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_set(self) -> c_int {
    unsafe {
      let ret = isl_union_set_n_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_set<F1: FnMut(Set) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(Set) -> Option<()>>(set: Set, user: *mut c_void) -> Stat { (*(user as *mut F))(set.to()).to() }
    unsafe {
      let ret = isl_union_set_foreach_set(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_basic_set_list(self) -> Option<BasicSetList> {
    unsafe {
      let ret = isl_union_set_get_basic_set_list(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn contains(self, space: SpaceRef) -> Option<bool> {
    unsafe {
      let ret = isl_union_set_contains(self.to(), space.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn extract_set(self, dim: Space) -> Option<Set> {
    unsafe {
      let ret = isl_union_set_extract_set(self.to(), dim.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_point<F1: FnMut(Point) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(Point) -> Option<()>>(pnt: Point, user: *mut c_void) -> Stat { (*(user as *mut F))(pnt.to()).to() }
    unsafe {
      let ret = isl_union_set_foreach_point(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_union_set_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_union_set_dump(self.to());
      (ret).to()
    }
  }
}

impl Drop for UnionSet {
  fn drop(&mut self) { UnionSet(self.0).free() }
}

impl Drop for UnionSetList {
  fn drop(&mut self) { UnionSetList(self.0).free() }
}

impl fmt::Display for UnionSetRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for UnionSet {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

