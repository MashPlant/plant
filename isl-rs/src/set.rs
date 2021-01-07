use crate::*;

extern "C" {
  pub fn isl_basic_set_n_dim(bset: BasicSetRef) -> c_uint;
  pub fn isl_basic_set_n_param(bset: BasicSetRef) -> c_uint;
  pub fn isl_basic_set_total_dim(bset: BasicSetRef) -> c_uint;
  pub fn isl_basic_set_dim(bset: BasicSetRef, type_: DimType) -> c_uint;
  pub fn isl_set_n_dim(set: SetRef) -> c_uint;
  pub fn isl_set_n_param(set: SetRef) -> c_uint;
  pub fn isl_set_dim(set: SetRef, type_: DimType) -> c_uint;
  pub fn isl_basic_set_get_ctx(bset: BasicSetRef) -> Option<CtxRef>;
  pub fn isl_set_get_ctx(set: SetRef) -> Option<CtxRef>;
  pub fn isl_basic_set_get_space(bset: BasicSetRef) -> Option<Space>;
  pub fn isl_set_get_space(set: SetRef) -> Option<Space>;
  pub fn isl_set_reset_space(set: Set, dim: Space) -> Option<Set>;
  pub fn isl_basic_set_get_div(bset: BasicSetRef, pos: c_int) -> Option<Aff>;
  pub fn isl_basic_set_get_local_space(bset: BasicSetRef) -> Option<LocalSpace>;
  pub fn isl_basic_set_get_tuple_name(bset: BasicSetRef) -> Option<CStr>;
  pub fn isl_set_has_tuple_name(set: SetRef) -> Bool;
  pub fn isl_set_get_tuple_name(set: SetRef) -> Option<CStr>;
  pub fn isl_basic_set_set_tuple_name(set: BasicSet, s: Option<CStr>) -> Option<BasicSet>;
  pub fn isl_set_set_tuple_name(set: Set, s: Option<CStr>) -> Option<Set>;
  pub fn isl_basic_set_get_dim_name(bset: BasicSetRef, type_: DimType, pos: c_uint) -> Option<CStr>;
  pub fn isl_basic_set_set_dim_name(bset: BasicSet, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<BasicSet>;
  pub fn isl_set_has_dim_name(set: SetRef, type_: DimType, pos: c_uint) -> Bool;
  pub fn isl_set_get_dim_name(set: SetRef, type_: DimType, pos: c_uint) -> Option<CStr>;
  pub fn isl_set_set_dim_name(set: Set, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<Set>;
  pub fn isl_basic_set_get_dim_id(bset: BasicSetRef, type_: DimType, pos: c_uint) -> Option<Id>;
  pub fn isl_basic_set_set_tuple_id(bset: BasicSet, id: Id) -> Option<BasicSet>;
  pub fn isl_set_set_dim_id(set: Set, type_: DimType, pos: c_uint, id: Id) -> Option<Set>;
  pub fn isl_set_has_dim_id(set: SetRef, type_: DimType, pos: c_uint) -> Bool;
  pub fn isl_set_get_dim_id(set: SetRef, type_: DimType, pos: c_uint) -> Option<Id>;
  pub fn isl_set_set_tuple_id(set: Set, id: Id) -> Option<Set>;
  pub fn isl_set_reset_tuple_id(set: Set) -> Option<Set>;
  pub fn isl_set_has_tuple_id(set: SetRef) -> Bool;
  pub fn isl_set_get_tuple_id(set: SetRef) -> Option<Id>;
  pub fn isl_set_reset_user(set: Set) -> Option<Set>;
  pub fn isl_set_find_dim_by_id(set: SetRef, type_: DimType, id: IdRef) -> c_int;
  pub fn isl_set_find_dim_by_name(set: SetRef, type_: DimType, name: Option<CStr>) -> c_int;
  pub fn isl_basic_set_is_rational(bset: BasicSetRef) -> c_int;
  pub fn isl_basic_set_free(bset: BasicSet) -> *mut c_void;
  pub fn isl_basic_set_copy(bset: BasicSetRef) -> Option<BasicSet>;
  pub fn isl_basic_set_empty(space: Space) -> Option<BasicSet>;
  pub fn isl_basic_set_universe(space: Space) -> Option<BasicSet>;
  pub fn isl_basic_set_nat_universe(dim: Space) -> Option<BasicSet>;
  pub fn isl_basic_set_positive_orthant(space: Space) -> Option<BasicSet>;
  pub fn isl_basic_set_print_internal(bset: BasicSetRef, out: *mut FILE, indent: c_int) -> ();
  pub fn isl_basic_set_intersect(bset1: BasicSet, bset2: BasicSet) -> Option<BasicSet>;
  pub fn isl_basic_set_intersect_params(bset1: BasicSet, bset2: BasicSet) -> Option<BasicSet>;
  pub fn isl_basic_set_apply(bset: BasicSet, bmap: BasicMap) -> Option<BasicSet>;
  pub fn isl_basic_set_preimage_multi_aff(bset: BasicSet, ma: MultiAff) -> Option<BasicSet>;
  pub fn isl_basic_set_affine_hull(bset: BasicSet) -> Option<BasicSet>;
  pub fn isl_basic_set_remove_dims(bset: BasicSet, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicSet>;
  pub fn isl_basic_set_sample(bset: BasicSet) -> Option<BasicSet>;
  pub fn isl_basic_set_detect_equalities(bset: BasicSet) -> Option<BasicSet>;
  pub fn isl_basic_set_remove_redundancies(bset: BasicSet) -> Option<BasicSet>;
  pub fn isl_set_remove_redundancies(set: Set) -> Option<Set>;
  pub fn isl_basic_set_list_intersect(list: BasicSetList) -> Option<BasicSet>;
  pub fn isl_set_list_union(list: SetList) -> Option<Set>;
  pub fn isl_basic_set_read_from_file(ctx: CtxRef, input: *mut FILE) -> Option<BasicSet>;
  pub fn isl_basic_set_read_from_str(ctx: CtxRef, str: Option<CStr>) -> Option<BasicSet>;
  pub fn isl_set_read_from_file(ctx: CtxRef, input: *mut FILE) -> Option<Set>;
  pub fn isl_set_read_from_str(ctx: CtxRef, str: Option<CStr>) -> Option<Set>;
  pub fn isl_basic_set_dump(bset: BasicSetRef) -> ();
  pub fn isl_set_dump(set: SetRef) -> ();
  pub fn isl_printer_print_basic_set(printer: Printer, bset: BasicSetRef) -> Option<Printer>;
  pub fn isl_printer_print_set(printer: Printer, map: SetRef) -> Option<Printer>;
  pub fn isl_basic_set_fix_si(bset: BasicSet, type_: DimType, pos: c_uint, value: c_int) -> Option<BasicSet>;
  pub fn isl_basic_set_fix_val(bset: BasicSet, type_: DimType, pos: c_uint, v: Val) -> Option<BasicSet>;
  pub fn isl_set_fix_si(set: Set, type_: DimType, pos: c_uint, value: c_int) -> Option<Set>;
  pub fn isl_set_lower_bound_si(set: Set, type_: DimType, pos: c_uint, value: c_int) -> Option<Set>;
  pub fn isl_basic_set_lower_bound_val(bset: BasicSet, type_: DimType, pos: c_uint, value: Val) -> Option<BasicSet>;
  pub fn isl_set_lower_bound_val(set: Set, type_: DimType, pos: c_uint, value: Val) -> Option<Set>;
  pub fn isl_set_upper_bound_si(set: Set, type_: DimType, pos: c_uint, value: c_int) -> Option<Set>;
  pub fn isl_basic_set_upper_bound_val(bset: BasicSet, type_: DimType, pos: c_uint, value: Val) -> Option<BasicSet>;
  pub fn isl_set_upper_bound_val(set: Set, type_: DimType, pos: c_uint, value: Val) -> Option<Set>;
  pub fn isl_set_equate(set: Set, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<Set>;
  pub fn isl_basic_set_is_equal(bset1: BasicSetRef, bset2: BasicSetRef) -> Bool;
  pub fn isl_basic_set_is_disjoint(bset1: BasicSetRef, bset2: BasicSetRef) -> Bool;
  pub fn isl_basic_set_partial_lexmin(bset: BasicSet, dom: BasicSet, empty: *mut Set) -> Option<Set>;
  pub fn isl_basic_set_partial_lexmax(bset: BasicSet, dom: BasicSet, empty: *mut Set) -> Option<Set>;
  pub fn isl_set_partial_lexmin(set: Set, dom: Set, empty: *mut Set) -> Option<Set>;
  pub fn isl_set_partial_lexmax(set: Set, dom: Set, empty: *mut Set) -> Option<Set>;
  pub fn isl_basic_set_lexmin(bset: BasicSet) -> Option<Set>;
  pub fn isl_basic_set_lexmax(bset: BasicSet) -> Option<Set>;
  pub fn isl_set_lexmin(set: Set) -> Option<Set>;
  pub fn isl_set_lexmax(set: Set) -> Option<Set>;
  pub fn isl_basic_set_partial_lexmin_pw_multi_aff(bset: BasicSet, dom: BasicSet, empty: *mut Set) -> Option<PwMultiAff>;
  pub fn isl_basic_set_partial_lexmax_pw_multi_aff(bset: BasicSet, dom: BasicSet, empty: *mut Set) -> Option<PwMultiAff>;
  pub fn isl_set_lexmin_pw_multi_aff(set: Set) -> Option<PwMultiAff>;
  pub fn isl_set_lexmax_pw_multi_aff(set: Set) -> Option<PwMultiAff>;
  pub fn isl_basic_set_union(bset1: BasicSet, bset2: BasicSet) -> Option<Set>;
  pub fn isl_basic_set_compare_at(bset1: BasicSetRef, bset2: BasicSetRef, pos: c_int) -> c_int;
  pub fn isl_set_follows_at(set1: SetRef, set2: SetRef, pos: c_int) -> c_int;
  pub fn isl_basic_set_params(bset: BasicSet) -> Option<BasicSet>;
  pub fn isl_basic_set_from_params(bset: BasicSet) -> Option<BasicSet>;
  pub fn isl_set_params(set: Set) -> Option<Set>;
  pub fn isl_set_from_params(set: Set) -> Option<Set>;
  pub fn isl_basic_set_dims_get_sign(bset: BasicSetRef, type_: DimType, pos: c_uint, n: c_uint, signs: *mut c_int) -> Stat;
  pub fn isl_basic_set_plain_is_universe(bset: BasicSetRef) -> Bool;
  pub fn isl_basic_set_is_universe(bset: BasicSetRef) -> Bool;
  pub fn isl_basic_set_plain_is_empty(bset: BasicSetRef) -> Bool;
  pub fn isl_basic_set_is_empty(bset: BasicSetRef) -> Bool;
  pub fn isl_basic_set_is_bounded(bset: BasicSetRef) -> Bool;
  pub fn isl_basic_set_is_subset(bset1: BasicSetRef, bset2: BasicSetRef) -> Bool;
  pub fn isl_basic_set_plain_is_equal(bset1: BasicSetRef, bset2: BasicSetRef) -> Bool;
  pub fn isl_set_empty(space: Space) -> Option<Set>;
  pub fn isl_set_universe(space: Space) -> Option<Set>;
  pub fn isl_set_nat_universe(dim: Space) -> Option<Set>;
  pub fn isl_set_copy(set: SetRef) -> Option<Set>;
  pub fn isl_set_free(set: Set) -> *mut c_void;
  pub fn isl_set_from_basic_set(bset: BasicSet) -> Option<Set>;
  pub fn isl_set_sample(set: Set) -> Option<BasicSet>;
  pub fn isl_basic_set_sample_point(bset: BasicSet) -> Option<Point>;
  pub fn isl_set_sample_point(set: Set) -> Option<Point>;
  pub fn isl_set_detect_equalities(set: Set) -> Option<Set>;
  pub fn isl_set_affine_hull(set: Set) -> Option<BasicSet>;
  pub fn isl_set_convex_hull(set: Set) -> Option<BasicSet>;
  pub fn isl_set_polyhedral_hull(set: Set) -> Option<BasicSet>;
  pub fn isl_set_simple_hull(set: Set) -> Option<BasicSet>;
  pub fn isl_set_unshifted_simple_hull(set: Set) -> Option<BasicSet>;
  pub fn isl_set_plain_unshifted_simple_hull(set: Set) -> Option<BasicSet>;
  pub fn isl_set_unshifted_simple_hull_from_set_list(set: Set, list: SetList) -> Option<BasicSet>;
  pub fn isl_set_bounded_simple_hull(set: Set) -> Option<BasicSet>;
  pub fn isl_set_union_disjoint(set1: Set, set2: Set) -> Option<Set>;
  pub fn isl_set_union(set1: Set, set2: Set) -> Option<Set>;
  pub fn isl_set_product(set1: Set, set2: Set) -> Option<Set>;
  pub fn isl_basic_set_flat_product(bset1: BasicSet, bset2: BasicSet) -> Option<BasicSet>;
  pub fn isl_set_flat_product(set1: Set, set2: Set) -> Option<Set>;
  pub fn isl_set_intersect(set1: Set, set2: Set) -> Option<Set>;
  pub fn isl_set_intersect_params(set: Set, params: Set) -> Option<Set>;
  pub fn isl_set_subtract(set1: Set, set2: Set) -> Option<Set>;
  pub fn isl_set_complement(set: Set) -> Option<Set>;
  pub fn isl_set_apply(set: Set, map: Map) -> Option<Set>;
  pub fn isl_set_preimage_multi_aff(set: Set, ma: MultiAff) -> Option<Set>;
  pub fn isl_set_preimage_pw_multi_aff(set: Set, pma: PwMultiAff) -> Option<Set>;
  pub fn isl_set_preimage_multi_pw_aff(set: Set, mpa: MultiPwAff) -> Option<Set>;
  pub fn isl_set_fix_val(set: Set, type_: DimType, pos: c_uint, v: Val) -> Option<Set>;
  pub fn isl_set_fix_dim_si(set: Set, dim: c_uint, value: c_int) -> Option<Set>;
  pub fn isl_basic_set_insert_dims(bset: BasicSet, type_: DimType, pos: c_uint, n: c_uint) -> Option<BasicSet>;
  pub fn isl_set_insert_dims(set: Set, type_: DimType, pos: c_uint, n: c_uint) -> Option<Set>;
  pub fn isl_basic_set_add_dims(bset: BasicSet, type_: DimType, n: c_uint) -> Option<BasicSet>;
  pub fn isl_set_add_dims(set: Set, type_: DimType, n: c_uint) -> Option<Set>;
  pub fn isl_basic_set_move_dims(bset: BasicSet, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<BasicSet>;
  pub fn isl_set_move_dims(set: Set, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<Set>;
  pub fn isl_basic_set_project_out(bset: BasicSet, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicSet>;
  pub fn isl_set_project_out(set: Set, type_: DimType, first: c_uint, n: c_uint) -> Option<Set>;
  pub fn isl_set_project_onto_map(set: Set, type_: DimType, first: c_uint, n: c_uint) -> Option<Map>;
  pub fn isl_basic_set_remove_divs(bset: BasicSet) -> Option<BasicSet>;
  pub fn isl_basic_set_eliminate(bset: BasicSet, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicSet>;
  pub fn isl_set_eliminate(set: Set, type_: DimType, first: c_uint, n: c_uint) -> Option<Set>;
  pub fn isl_set_eliminate_dims(set: Set, first: c_uint, n: c_uint) -> Option<Set>;
  pub fn isl_set_remove_dims(bset: Set, type_: DimType, first: c_uint, n: c_uint) -> Option<Set>;
  pub fn isl_basic_set_remove_divs_involving_dims(bset: BasicSet, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicSet>;
  pub fn isl_set_remove_divs_involving_dims(set: Set, type_: DimType, first: c_uint, n: c_uint) -> Option<Set>;
  pub fn isl_basic_set_remove_unknown_divs(bset: BasicSet) -> Option<BasicSet>;
  pub fn isl_set_remove_unknown_divs(set: Set) -> Option<Set>;
  pub fn isl_set_remove_divs(set: Set) -> Option<Set>;
  pub fn isl_set_split_dims(set: Set, type_: DimType, first: c_uint, n: c_uint) -> Option<Set>;
  pub fn isl_basic_set_drop_constraints_involving_dims(bset: BasicSet, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicSet>;
  pub fn isl_basic_set_drop_constraints_not_involving_dims(bset: BasicSet, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicSet>;
  pub fn isl_set_drop_constraints_involving_dims(set: Set, type_: DimType, first: c_uint, n: c_uint) -> Option<Set>;
  pub fn isl_set_drop_constraints_not_involving_dims(set: Set, type_: DimType, first: c_uint, n: c_uint) -> Option<Set>;
  pub fn isl_basic_set_involves_dims(bset: BasicSetRef, type_: DimType, first: c_uint, n: c_uint) -> Bool;
  pub fn isl_set_involves_dims(set: SetRef, type_: DimType, first: c_uint, n: c_uint) -> Bool;
  pub fn isl_set_print_internal(set: SetRef, out: *mut FILE, indent: c_int) -> ();
  pub fn isl_set_plain_is_empty(set: SetRef) -> Bool;
  pub fn isl_set_plain_is_universe(set: SetRef) -> Bool;
  pub fn isl_set_is_params(set: SetRef) -> Bool;
  pub fn isl_set_is_empty(set: SetRef) -> Bool;
  pub fn isl_set_is_bounded(set: SetRef) -> Bool;
  pub fn isl_set_is_subset(set1: SetRef, set2: SetRef) -> Bool;
  pub fn isl_set_is_strict_subset(set1: SetRef, set2: SetRef) -> Bool;
  pub fn isl_set_is_equal(set1: SetRef, set2: SetRef) -> Bool;
  pub fn isl_set_is_disjoint(set1: SetRef, set2: SetRef) -> Bool;
  pub fn isl_set_is_singleton(set: SetRef) -> Bool;
  pub fn isl_set_is_box(set: SetRef) -> Bool;
  pub fn isl_set_has_equal_space(set1: SetRef, set2: SetRef) -> Bool;
  pub fn isl_set_sum(set1: Set, set2: Set) -> Option<Set>;
  pub fn isl_basic_set_neg(bset: BasicSet) -> Option<BasicSet>;
  pub fn isl_set_neg(set: Set) -> Option<Set>;
  pub fn isl_set_make_disjoint(set: Set) -> Option<Set>;
  pub fn isl_basic_set_compute_divs(bset: BasicSet) -> Option<Set>;
  pub fn isl_set_compute_divs(set: Set) -> Option<Set>;
  pub fn isl_set_align_divs(set: Set) -> Option<Set>;
  pub fn isl_set_plain_get_val_if_fixed(set: SetRef, type_: DimType, pos: c_uint) -> Option<Val>;
  pub fn isl_set_dim_is_bounded(set: SetRef, type_: DimType, pos: c_uint) -> Bool;
  pub fn isl_set_dim_has_lower_bound(set: SetRef, type_: DimType, pos: c_uint) -> Bool;
  pub fn isl_set_dim_has_upper_bound(set: SetRef, type_: DimType, pos: c_uint) -> Bool;
  pub fn isl_set_dim_has_any_lower_bound(set: SetRef, type_: DimType, pos: c_uint) -> Bool;
  pub fn isl_set_dim_has_any_upper_bound(set: SetRef, type_: DimType, pos: c_uint) -> Bool;
  pub fn isl_basic_set_gist(bset: BasicSet, context: BasicSet) -> Option<BasicSet>;
  pub fn isl_set_gist_basic_set(set: Set, context: BasicSet) -> Option<Set>;
  pub fn isl_set_gist(set: Set, context: Set) -> Option<Set>;
  pub fn isl_set_gist_params(set: Set, context: Set) -> Option<Set>;
  pub fn isl_set_dim_residue_class_val(set: SetRef, pos: c_int, modulo: *mut Val, residue: *mut Val) -> Stat;
  pub fn isl_stride_info_get_stride(si: StrideInfoRef) -> Option<Val>;
  pub fn isl_stride_info_get_offset(si: StrideInfoRef) -> Option<Aff>;
  pub fn isl_stride_info_free(si: StrideInfo) -> *mut c_void;
  pub fn isl_set_get_stride_info(set: SetRef, pos: c_int) -> Option<StrideInfo>;
  pub fn isl_set_get_stride(set: SetRef, pos: c_int) -> Option<Val>;
  pub fn isl_set_coalesce(set: Set) -> Option<Set>;
  pub fn isl_set_plain_cmp(set1: SetRef, set2: SetRef) -> c_int;
  pub fn isl_set_plain_is_equal(set1: SetRef, set2: SetRef) -> Bool;
  pub fn isl_set_plain_is_disjoint(set1: SetRef, set2: SetRef) -> Bool;
  pub fn isl_set_get_hash(set: SetRef) -> c_uint;
  pub fn isl_set_n_basic_set(set: SetRef) -> c_int;
  pub fn isl_set_foreach_basic_set(set: SetRef, fn_: unsafe extern "C" fn(bset: BasicSet, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_set_get_basic_set_list(set: SetRef) -> Option<BasicSetList>;
  pub fn isl_set_foreach_point(set: SetRef, fn_: unsafe extern "C" fn(pnt: Point, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_set_count_val(set: SetRef) -> Option<Val>;
  pub fn isl_basic_set_from_point(pnt: Point) -> Option<BasicSet>;
  pub fn isl_set_from_point(pnt: Point) -> Option<Set>;
  pub fn isl_basic_set_box_from_points(pnt1: Point, pnt2: Point) -> Option<BasicSet>;
  pub fn isl_set_box_from_points(pnt1: Point, pnt2: Point) -> Option<Set>;
  pub fn isl_basic_set_lift(bset: BasicSet) -> Option<BasicSet>;
  pub fn isl_set_lift(set: Set) -> Option<Set>;
  pub fn isl_set_lex_le_set(set1: Set, set2: Set) -> Option<Map>;
  pub fn isl_set_lex_lt_set(set1: Set, set2: Set) -> Option<Map>;
  pub fn isl_set_lex_ge_set(set1: Set, set2: Set) -> Option<Map>;
  pub fn isl_set_lex_gt_set(set1: Set, set2: Set) -> Option<Map>;
  pub fn isl_set_size(set: SetRef) -> c_int;
  pub fn isl_basic_set_align_params(bset: BasicSet, model: Space) -> Option<BasicSet>;
  pub fn isl_set_align_params(set: Set, model: Space) -> Option<Set>;
  pub fn isl_basic_set_equalities_matrix(bset: BasicSetRef, c1: DimType, c2: DimType, c3: DimType, c4: DimType) -> Option<Mat>;
  pub fn isl_basic_set_inequalities_matrix(bset: BasicSetRef, c1: DimType, c2: DimType, c3: DimType, c4: DimType) -> Option<Mat>;
  pub fn isl_basic_set_from_constraint_matrices(dim: Space, eq: Mat, ineq: Mat, c1: DimType, c2: DimType, c3: DimType, c4: DimType) -> Option<BasicSet>;
  pub fn isl_basic_set_reduced_basis(bset: BasicSetRef) -> Option<Mat>;
  pub fn isl_basic_set_coefficients(bset: BasicSet) -> Option<BasicSet>;
  pub fn isl_basic_set_list_coefficients(list: BasicSetList) -> Option<BasicSetList>;
  pub fn isl_set_coefficients(set: Set) -> Option<BasicSet>;
  pub fn isl_basic_set_solutions(bset: BasicSet) -> Option<BasicSet>;
  pub fn isl_set_solutions(set: Set) -> Option<BasicSet>;
  pub fn isl_set_dim_max(set: Set, pos: c_int) -> Option<PwAff>;
  pub fn isl_set_dim_min(set: Set, pos: c_int) -> Option<PwAff>;
  pub fn isl_basic_set_to_str(bset: BasicSetRef) -> Option<CString>;
  pub fn isl_set_to_str(set: SetRef) -> Option<CString>;
}

#[repr(transparent)]
#[derive(Debug)]
pub struct StrideInfo(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct StrideInfoRef(pub NonNull<c_void>);

impl_try!(StrideInfo);
impl_try!(StrideInfoRef);

impl StrideInfo {
  #[inline(always)]
  pub fn read(&self) -> StrideInfo { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: StrideInfo) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<StrideInfoRef> for StrideInfo {
  #[inline(always)]
  fn as_ref(&self) -> &StrideInfoRef { unsafe { mem::transmute(self) } }
}

impl Deref for StrideInfo {
  type Target = StrideInfoRef;
  #[inline(always)]
  fn deref(&self) -> &StrideInfoRef { self.as_ref() }
}

impl To<Option<StrideInfo>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<StrideInfo> { NonNull::new(self).map(StrideInfo) }
}

impl BasicSet {
  #[inline(always)]
  pub fn set_tuple_name(self, s: Option<CStr>) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_set_tuple_name(self.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_name(self, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_set_dim_name(self.to(), type_.to(), pos.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_id(self, id: Id) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_set_tuple_id(self.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_basic_set_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect(self, bset2: BasicSet) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_intersect(self.to(), bset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_params(self, bset2: BasicSet) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_intersect_params(self.to(), bset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn apply(self, bmap: BasicMap) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_apply(self.to(), bmap.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_multi_aff(self, ma: MultiAff) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_preimage_multi_aff(self.to(), ma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn affine_hull(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_affine_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_remove_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sample(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_sample(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn detect_equalities(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_detect_equalities(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_redundancies(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_remove_redundancies(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn fix_si(self, type_: DimType, pos: c_uint, value: c_int) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_fix_si(self.to(), type_.to(), pos.to(), value.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn fix_val(self, type_: DimType, pos: c_uint, v: Val) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_fix_val(self.to(), type_.to(), pos.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lower_bound_val(self, type_: DimType, pos: c_uint, value: Val) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_lower_bound_val(self.to(), type_.to(), pos.to(), value.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn upper_bound_val(self, type_: DimType, pos: c_uint, value: Val) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_upper_bound_val(self.to(), type_.to(), pos.to(), value.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn partial_lexmin(self, dom: BasicSet) -> Option<(Set, Set)> {
    unsafe {
      let ref mut empty = 0 as *mut c_void;
      let ret = isl_basic_set_partial_lexmin(self.to(), dom.to(), empty as *mut _ as _);
      (ret, *empty).to()
    }
  }
  #[inline(always)]
  pub fn partial_lexmax(self, dom: BasicSet) -> Option<(Set, Set)> {
    unsafe {
      let ref mut empty = 0 as *mut c_void;
      let ret = isl_basic_set_partial_lexmax(self.to(), dom.to(), empty as *mut _ as _);
      (ret, *empty).to()
    }
  }
  #[inline(always)]
  pub fn lexmin(self) -> Option<Set> {
    unsafe {
      let ret = isl_basic_set_lexmin(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lexmax(self) -> Option<Set> {
    unsafe {
      let ret = isl_basic_set_lexmax(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn partial_lexmin_pw_multi_aff(self, dom: BasicSet) -> Option<(PwMultiAff, Set)> {
    unsafe {
      let ref mut empty = 0 as *mut c_void;
      let ret = isl_basic_set_partial_lexmin_pw_multi_aff(self.to(), dom.to(), empty as *mut _ as _);
      (ret, *empty).to()
    }
  }
  #[inline(always)]
  pub fn partial_lexmax_pw_multi_aff(self, dom: BasicSet) -> Option<(PwMultiAff, Set)> {
    unsafe {
      let ref mut empty = 0 as *mut c_void;
      let ret = isl_basic_set_partial_lexmax_pw_multi_aff(self.to(), dom.to(), empty as *mut _ as _);
      (ret, *empty).to()
    }
  }
  #[inline(always)]
  pub fn union(self, bset2: BasicSet) -> Option<Set> {
    unsafe {
      let ret = isl_basic_set_union(self.to(), bset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn params(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn from_params(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_from_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_from_basic_set(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_from_basic_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sample_point(self) -> Option<Point> {
    unsafe {
      let ret = isl_basic_set_sample_point(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flat_product(self, bset2: BasicSet) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_flat_product(self.to(), bset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_dims(self, type_: DimType, pos: c_uint, n: c_uint) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_insert_dims(self.to(), type_.to(), pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_dims(self, type_: DimType, n: c_uint) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_add_dims(self.to(), type_.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn move_dims(self, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_move_dims(self.to(), dst_type.to(), dst_pos.to(), src_type.to(), src_pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn project_out(self, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_project_out(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_divs(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_remove_divs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eliminate(self, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_eliminate(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_divs_involving_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_remove_divs_involving_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_unknown_divs(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_remove_unknown_divs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_constraints_involving_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_drop_constraints_involving_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_constraints_not_involving_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_drop_constraints_not_involving_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn compute_divs(self) -> Option<Set> {
    unsafe {
      let ret = isl_basic_set_compute_divs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, context: BasicSet) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_gist(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lift(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_lift(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_align_params(self.to(), model.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coefficients(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_coefficients(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn solutions(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_solutions(self.to());
      (ret).to()
    }
  }
}

impl BasicSetList {
  #[inline(always)]
  pub fn intersect(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_list_intersect(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coefficients(self) -> Option<BasicSetList> {
    unsafe {
      let ret = isl_basic_set_list_coefficients(self.to());
      (ret).to()
    }
  }
}

impl BasicSetRef {
  #[inline(always)]
  pub fn n_dim(self) -> c_uint {
    unsafe {
      let ret = isl_basic_set_n_dim(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_param(self) -> c_uint {
    unsafe {
      let ret = isl_basic_set_n_param(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn total_dim(self) -> c_uint {
    unsafe {
      let ret = isl_basic_set_total_dim(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_uint {
    unsafe {
      let ret = isl_basic_set_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_basic_set_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_basic_set_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_div(self, pos: c_int) -> Option<Aff> {
    unsafe {
      let ret = isl_basic_set_get_div(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_local_space(self) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_basic_set_get_local_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_name(self) -> Option<CStr> {
    unsafe {
      let ret = isl_basic_set_get_tuple_name(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_name(self, type_: DimType, pos: c_uint) -> Option<CStr> {
    unsafe {
      let ret = isl_basic_set_get_dim_name(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_id(self, type_: DimType, pos: c_uint) -> Option<Id> {
    unsafe {
      let ret = isl_basic_set_get_dim_id(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_rational(self) -> c_int {
    unsafe {
      let ret = isl_basic_set_is_rational(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_internal(self, out: *mut FILE, indent: c_int) -> () {
    unsafe {
      let ret = isl_basic_set_print_internal(self.to(), out.to(), indent.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_basic_set_dump(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_equal(self, bset2: BasicSetRef) -> Bool {
    unsafe {
      let ret = isl_basic_set_is_equal(self.to(), bset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_disjoint(self, bset2: BasicSetRef) -> Bool {
    unsafe {
      let ret = isl_basic_set_is_disjoint(self.to(), bset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn compare_at(self, bset2: BasicSetRef, pos: c_int) -> c_int {
    unsafe {
      let ret = isl_basic_set_compare_at(self.to(), bset2.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dims_get_sign(self, type_: DimType, pos: c_uint, n: c_uint, signs: &mut c_int) -> Stat {
    unsafe {
      let ret = isl_basic_set_dims_get_sign(self.to(), type_.to(), pos.to(), n.to(), signs.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_universe(self) -> Bool {
    unsafe {
      let ret = isl_basic_set_plain_is_universe(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_universe(self) -> Bool {
    unsafe {
      let ret = isl_basic_set_is_universe(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_empty(self) -> Bool {
    unsafe {
      let ret = isl_basic_set_plain_is_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_empty(self) -> Bool {
    unsafe {
      let ret = isl_basic_set_is_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_bounded(self) -> Bool {
    unsafe {
      let ret = isl_basic_set_is_bounded(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_subset(self, bset2: BasicSetRef) -> Bool {
    unsafe {
      let ret = isl_basic_set_is_subset(self.to(), bset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, bset2: BasicSetRef) -> Bool {
    unsafe {
      let ret = isl_basic_set_plain_is_equal(self.to(), bset2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Bool {
    unsafe {
      let ret = isl_basic_set_involves_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn equalities_matrix(self, c1: DimType, c2: DimType, c3: DimType, c4: DimType) -> Option<Mat> {
    unsafe {
      let ret = isl_basic_set_equalities_matrix(self.to(), c1.to(), c2.to(), c3.to(), c4.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn inequalities_matrix(self, c1: DimType, c2: DimType, c3: DimType, c4: DimType) -> Option<Mat> {
    unsafe {
      let ret = isl_basic_set_inequalities_matrix(self.to(), c1.to(), c2.to(), c3.to(), c4.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reduced_basis(self) -> Option<Mat> {
    unsafe {
      let ret = isl_basic_set_reduced_basis(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_basic_set_to_str(self.to());
      (ret).to()
    }
  }
}

impl CtxRef {
  #[inline(always)]
  pub fn basic_set_read_from_file(self, input: *mut FILE) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_read_from_file(self.to(), input.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_set_read_from_str(self, str: Option<CStr>) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_read_from_file(self, input: *mut FILE) -> Option<Set> {
    unsafe {
      let ret = isl_set_read_from_file(self.to(), input.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_read_from_str(self, str: Option<CStr>) -> Option<Set> {
    unsafe {
      let ret = isl_set_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
}

impl Point {
  #[inline(always)]
  pub fn basic_set_from_point(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_from_point(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_from_point(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_from_point(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_set_box_from_points(self, pnt2: Point) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_box_from_points(self.to(), pnt2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_box_from_points(self, pnt2: Point) -> Option<Set> {
    unsafe {
      let ret = isl_set_box_from_points(self.to(), pnt2.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_basic_set(self, bset: BasicSetRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_basic_set(self.to(), bset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_set(self, map: SetRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_set(self.to(), map.to());
      (ret).to()
    }
  }
}

impl Set {
  #[inline(always)]
  pub fn reset_space(self, dim: Space) -> Option<Set> {
    unsafe {
      let ret = isl_set_reset_space(self.to(), dim.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_name(self, s: Option<CStr>) -> Option<Set> {
    unsafe {
      let ret = isl_set_set_tuple_name(self.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_name(self, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<Set> {
    unsafe {
      let ret = isl_set_set_dim_name(self.to(), type_.to(), pos.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_id(self, type_: DimType, pos: c_uint, id: Id) -> Option<Set> {
    unsafe {
      let ret = isl_set_set_dim_id(self.to(), type_.to(), pos.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_id(self, id: Id) -> Option<Set> {
    unsafe {
      let ret = isl_set_set_tuple_id(self.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_tuple_id(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_reset_tuple_id(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_redundancies(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_remove_redundancies(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn fix_si(self, type_: DimType, pos: c_uint, value: c_int) -> Option<Set> {
    unsafe {
      let ret = isl_set_fix_si(self.to(), type_.to(), pos.to(), value.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lower_bound_si(self, type_: DimType, pos: c_uint, value: c_int) -> Option<Set> {
    unsafe {
      let ret = isl_set_lower_bound_si(self.to(), type_.to(), pos.to(), value.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lower_bound_val(self, type_: DimType, pos: c_uint, value: Val) -> Option<Set> {
    unsafe {
      let ret = isl_set_lower_bound_val(self.to(), type_.to(), pos.to(), value.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn upper_bound_si(self, type_: DimType, pos: c_uint, value: c_int) -> Option<Set> {
    unsafe {
      let ret = isl_set_upper_bound_si(self.to(), type_.to(), pos.to(), value.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn upper_bound_val(self, type_: DimType, pos: c_uint, value: Val) -> Option<Set> {
    unsafe {
      let ret = isl_set_upper_bound_val(self.to(), type_.to(), pos.to(), value.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn equate(self, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<Set> {
    unsafe {
      let ret = isl_set_equate(self.to(), type1.to(), pos1.to(), type2.to(), pos2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn partial_lexmin(self, dom: Set) -> Option<(Set, Set)> {
    unsafe {
      let ref mut empty = 0 as *mut c_void;
      let ret = isl_set_partial_lexmin(self.to(), dom.to(), empty as *mut _ as _);
      (ret, *empty).to()
    }
  }
  #[inline(always)]
  pub fn partial_lexmax(self, dom: Set) -> Option<(Set, Set)> {
    unsafe {
      let ref mut empty = 0 as *mut c_void;
      let ret = isl_set_partial_lexmax(self.to(), dom.to(), empty as *mut _ as _);
      (ret, *empty).to()
    }
  }
  #[inline(always)]
  pub fn lexmin(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_lexmin(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lexmax(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_lexmax(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lexmin_pw_multi_aff(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_set_lexmin_pw_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lexmax_pw_multi_aff(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_set_lexmax_pw_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn params(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn from_params(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_from_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_set_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sample(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_set_sample(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sample_point(self) -> Option<Point> {
    unsafe {
      let ret = isl_set_sample_point(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn detect_equalities(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_detect_equalities(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn affine_hull(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_set_affine_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn convex_hull(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_set_convex_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn polyhedral_hull(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_set_polyhedral_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn simple_hull(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_set_simple_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn unshifted_simple_hull(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_set_unshifted_simple_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_unshifted_simple_hull(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_set_plain_unshifted_simple_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn unshifted_simple_hull_from_set_list(self, list: SetList) -> Option<BasicSet> {
    unsafe {
      let ret = isl_set_unshifted_simple_hull_from_set_list(self.to(), list.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn bounded_simple_hull(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_set_bounded_simple_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_disjoint(self, set2: Set) -> Option<Set> {
    unsafe {
      let ret = isl_set_union_disjoint(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union(self, set2: Set) -> Option<Set> {
    unsafe {
      let ret = isl_set_union(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn product(self, set2: Set) -> Option<Set> {
    unsafe {
      let ret = isl_set_product(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flat_product(self, set2: Set) -> Option<Set> {
    unsafe {
      let ret = isl_set_flat_product(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect(self, set2: Set) -> Option<Set> {
    unsafe {
      let ret = isl_set_intersect(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_params(self, params: Set) -> Option<Set> {
    unsafe {
      let ret = isl_set_intersect_params(self.to(), params.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract(self, set2: Set) -> Option<Set> {
    unsafe {
      let ret = isl_set_subtract(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn complement(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_complement(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn apply(self, map: Map) -> Option<Set> {
    unsafe {
      let ret = isl_set_apply(self.to(), map.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_multi_aff(self, ma: MultiAff) -> Option<Set> {
    unsafe {
      let ret = isl_set_preimage_multi_aff(self.to(), ma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_pw_multi_aff(self, pma: PwMultiAff) -> Option<Set> {
    unsafe {
      let ret = isl_set_preimage_pw_multi_aff(self.to(), pma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_multi_pw_aff(self, mpa: MultiPwAff) -> Option<Set> {
    unsafe {
      let ret = isl_set_preimage_multi_pw_aff(self.to(), mpa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn fix_val(self, type_: DimType, pos: c_uint, v: Val) -> Option<Set> {
    unsafe {
      let ret = isl_set_fix_val(self.to(), type_.to(), pos.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn fix_dim_si(self, dim: c_uint, value: c_int) -> Option<Set> {
    unsafe {
      let ret = isl_set_fix_dim_si(self.to(), dim.to(), value.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_dims(self, type_: DimType, pos: c_uint, n: c_uint) -> Option<Set> {
    unsafe {
      let ret = isl_set_insert_dims(self.to(), type_.to(), pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_dims(self, type_: DimType, n: c_uint) -> Option<Set> {
    unsafe {
      let ret = isl_set_add_dims(self.to(), type_.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn move_dims(self, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<Set> {
    unsafe {
      let ret = isl_set_move_dims(self.to(), dst_type.to(), dst_pos.to(), src_type.to(), src_pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn project_out(self, type_: DimType, first: c_uint, n: c_uint) -> Option<Set> {
    unsafe {
      let ret = isl_set_project_out(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn project_onto_map(self, type_: DimType, first: c_uint, n: c_uint) -> Option<Map> {
    unsafe {
      let ret = isl_set_project_onto_map(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eliminate(self, type_: DimType, first: c_uint, n: c_uint) -> Option<Set> {
    unsafe {
      let ret = isl_set_eliminate(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eliminate_dims(self, first: c_uint, n: c_uint) -> Option<Set> {
    unsafe {
      let ret = isl_set_eliminate_dims(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<Set> {
    unsafe {
      let ret = isl_set_remove_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_divs_involving_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<Set> {
    unsafe {
      let ret = isl_set_remove_divs_involving_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_unknown_divs(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_remove_unknown_divs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_divs(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_remove_divs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn split_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<Set> {
    unsafe {
      let ret = isl_set_split_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_constraints_involving_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<Set> {
    unsafe {
      let ret = isl_set_drop_constraints_involving_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_constraints_not_involving_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<Set> {
    unsafe {
      let ret = isl_set_drop_constraints_not_involving_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sum(self, set2: Set) -> Option<Set> {
    unsafe {
      let ret = isl_set_sum(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn make_disjoint(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_make_disjoint(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn compute_divs(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_compute_divs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_divs(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_align_divs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_basic_set(self, context: BasicSet) -> Option<Set> {
    unsafe {
      let ret = isl_set_gist_basic_set(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, context: Set) -> Option<Set> {
    unsafe {
      let ret = isl_set_gist(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_params(self, context: Set) -> Option<Set> {
    unsafe {
      let ret = isl_set_gist_params(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coalesce(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_coalesce(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lift(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_lift(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_le_set(self, set2: Set) -> Option<Map> {
    unsafe {
      let ret = isl_set_lex_le_set(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_lt_set(self, set2: Set) -> Option<Map> {
    unsafe {
      let ret = isl_set_lex_lt_set(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_ge_set(self, set2: Set) -> Option<Map> {
    unsafe {
      let ret = isl_set_lex_ge_set(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_gt_set(self, set2: Set) -> Option<Map> {
    unsafe {
      let ret = isl_set_lex_gt_set(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<Set> {
    unsafe {
      let ret = isl_set_align_params(self.to(), model.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coefficients(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_set_coefficients(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn solutions(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_set_solutions(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim_max(self, pos: c_int) -> Option<PwAff> {
    unsafe {
      let ret = isl_set_dim_max(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim_min(self, pos: c_int) -> Option<PwAff> {
    unsafe {
      let ret = isl_set_dim_min(self.to(), pos.to());
      (ret).to()
    }
  }
}

impl SetList {
  #[inline(always)]
  pub fn union(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_list_union(self.to());
      (ret).to()
    }
  }
}

impl SetRef {
  #[inline(always)]
  pub fn n_dim(self) -> c_uint {
    unsafe {
      let ret = isl_set_n_dim(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_param(self) -> c_uint {
    unsafe {
      let ret = isl_set_n_param(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_uint {
    unsafe {
      let ret = isl_set_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_set_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_set_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_tuple_name(self) -> Bool {
    unsafe {
      let ret = isl_set_has_tuple_name(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_name(self) -> Option<CStr> {
    unsafe {
      let ret = isl_set_get_tuple_name(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_dim_name(self, type_: DimType, pos: c_uint) -> Bool {
    unsafe {
      let ret = isl_set_has_dim_name(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_name(self, type_: DimType, pos: c_uint) -> Option<CStr> {
    unsafe {
      let ret = isl_set_get_dim_name(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_dim_id(self, type_: DimType, pos: c_uint) -> Bool {
    unsafe {
      let ret = isl_set_has_dim_id(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_id(self, type_: DimType, pos: c_uint) -> Option<Id> {
    unsafe {
      let ret = isl_set_get_dim_id(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_tuple_id(self) -> Bool {
    unsafe {
      let ret = isl_set_has_tuple_id(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_id(self) -> Option<Id> {
    unsafe {
      let ret = isl_set_get_tuple_id(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_id(self, type_: DimType, id: IdRef) -> c_int {
    unsafe {
      let ret = isl_set_find_dim_by_id(self.to(), type_.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: Option<CStr>) -> c_int {
    unsafe {
      let ret = isl_set_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_set_dump(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn follows_at(self, set2: SetRef, pos: c_int) -> c_int {
    unsafe {
      let ret = isl_set_follows_at(self.to(), set2.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Bool {
    unsafe {
      let ret = isl_set_involves_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_internal(self, out: *mut FILE, indent: c_int) -> () {
    unsafe {
      let ret = isl_set_print_internal(self.to(), out.to(), indent.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_empty(self) -> Bool {
    unsafe {
      let ret = isl_set_plain_is_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_universe(self) -> Bool {
    unsafe {
      let ret = isl_set_plain_is_universe(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_params(self) -> Bool {
    unsafe {
      let ret = isl_set_is_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_empty(self) -> Bool {
    unsafe {
      let ret = isl_set_is_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_bounded(self) -> Bool {
    unsafe {
      let ret = isl_set_is_bounded(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_subset(self, set2: SetRef) -> Bool {
    unsafe {
      let ret = isl_set_is_subset(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_strict_subset(self, set2: SetRef) -> Bool {
    unsafe {
      let ret = isl_set_is_strict_subset(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_equal(self, set2: SetRef) -> Bool {
    unsafe {
      let ret = isl_set_is_equal(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_disjoint(self, set2: SetRef) -> Bool {
    unsafe {
      let ret = isl_set_is_disjoint(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_singleton(self) -> Bool {
    unsafe {
      let ret = isl_set_is_singleton(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_box(self) -> Bool {
    unsafe {
      let ret = isl_set_is_box(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_equal_space(self, set2: SetRef) -> Bool {
    unsafe {
      let ret = isl_set_has_equal_space(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_get_val_if_fixed(self, type_: DimType, pos: c_uint) -> Option<Val> {
    unsafe {
      let ret = isl_set_plain_get_val_if_fixed(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim_is_bounded(self, type_: DimType, pos: c_uint) -> Bool {
    unsafe {
      let ret = isl_set_dim_is_bounded(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim_has_lower_bound(self, type_: DimType, pos: c_uint) -> Bool {
    unsafe {
      let ret = isl_set_dim_has_lower_bound(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim_has_upper_bound(self, type_: DimType, pos: c_uint) -> Bool {
    unsafe {
      let ret = isl_set_dim_has_upper_bound(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim_has_any_lower_bound(self, type_: DimType, pos: c_uint) -> Bool {
    unsafe {
      let ret = isl_set_dim_has_any_lower_bound(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim_has_any_upper_bound(self, type_: DimType, pos: c_uint) -> Bool {
    unsafe {
      let ret = isl_set_dim_has_any_upper_bound(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim_residue_class_val(self, pos: c_int) -> Option<(Stat, Val, Val)> {
    unsafe {
      let ref mut modulo = 0 as *mut c_void;
      let ref mut residue = 0 as *mut c_void;
      let ret = isl_set_dim_residue_class_val(self.to(), pos.to(), modulo as *mut _ as _, residue as *mut _ as _);
      (ret, *modulo, *residue).to()
    }
  }
  #[inline(always)]
  pub fn get_stride_info(self, pos: c_int) -> Option<StrideInfo> {
    unsafe {
      let ret = isl_set_get_stride_info(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_stride(self, pos: c_int) -> Option<Val> {
    unsafe {
      let ret = isl_set_get_stride(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_cmp(self, set2: SetRef) -> c_int {
    unsafe {
      let ret = isl_set_plain_cmp(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, set2: SetRef) -> Bool {
    unsafe {
      let ret = isl_set_plain_is_equal(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_disjoint(self, set2: SetRef) -> Bool {
    unsafe {
      let ret = isl_set_plain_is_disjoint(self.to(), set2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_hash(self) -> c_uint {
    unsafe {
      let ret = isl_set_get_hash(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_basic_set(self) -> c_int {
    unsafe {
      let ret = isl_set_n_basic_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_basic_set<F1: FnMut(BasicSet) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(BasicSet) -> Stat>(bset: BasicSet, user: *mut c_void) -> Stat { (*(user as *mut F))(bset.to()) }
    unsafe {
      let ret = isl_set_foreach_basic_set(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_basic_set_list(self) -> Option<BasicSetList> {
    unsafe {
      let ret = isl_set_get_basic_set_list(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_point<F1: FnMut(Point) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(Point) -> Stat>(pnt: Point, user: *mut c_void) -> Stat { (*(user as *mut F))(pnt.to()) }
    unsafe {
      let ret = isl_set_foreach_point(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn count_val(self) -> Option<Val> {
    unsafe {
      let ret = isl_set_count_val(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn size(self) -> c_int {
    unsafe {
      let ret = isl_set_size(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_set_to_str(self.to());
      (ret).to()
    }
  }
}

impl Space {
  #[inline(always)]
  pub fn basic_set_empty(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_set_universe(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_universe(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_set_nat_universe(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_nat_universe(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_set_positive_orthant(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_positive_orthant(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_empty(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_universe(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_universe(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_nat_universe(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_nat_universe(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_set_from_constraint_matrices(self, eq: Mat, ineq: Mat, c1: DimType, c2: DimType, c3: DimType, c4: DimType) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_from_constraint_matrices(self.to(), eq.to(), ineq.to(), c1.to(), c2.to(), c3.to(), c4.to());
      (ret).to()
    }
  }
}

impl StrideInfo {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_stride_info_free(self.to());
      (ret).to()
    }
  }
}

impl StrideInfoRef {
  #[inline(always)]
  pub fn get_stride(self) -> Option<Val> {
    unsafe {
      let ret = isl_stride_info_get_stride(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_offset(self) -> Option<Aff> {
    unsafe {
      let ret = isl_stride_info_get_offset(self.to());
      (ret).to()
    }
  }
}

impl Drop for BasicSet {
  fn drop(&mut self) { BasicSet(self.0).free() }
}

impl fmt::Display for BasicSetRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for BasicSet {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl Drop for Set {
  fn drop(&mut self) { Set(self.0).free() }
}

impl fmt::Display for SetRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for Set {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl Drop for StrideInfo {
  fn drop(&mut self) { StrideInfo(self.0).free() }
}

