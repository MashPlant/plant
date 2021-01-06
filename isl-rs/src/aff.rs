use crate::*;

extern "C" {
  pub fn isl_aff_zero_on_domain(ls: LocalSpace) -> Option<Aff>;
  pub fn isl_aff_val_on_domain(ls: LocalSpace, val: Val) -> Option<Aff>;
  pub fn isl_aff_var_on_domain(ls: LocalSpace, type_: DimType, pos: c_uint) -> Option<Aff>;
  pub fn isl_aff_nan_on_domain(ls: LocalSpace) -> Option<Aff>;
  pub fn isl_aff_param_on_domain_space_id(space: Space, id: Id) -> Option<Aff>;
  pub fn isl_aff_copy(aff: AffRef) -> Option<Aff>;
  pub fn isl_aff_free(aff: Aff) -> *mut c_void;
  pub fn isl_aff_get_ctx(aff: AffRef) -> Option<CtxRef>;
  pub fn isl_aff_get_hash(aff: AffRef) -> c_uint;
  pub fn isl_aff_dim(aff: AffRef, type_: DimType) -> c_int;
  pub fn isl_aff_involves_dims(aff: AffRef, type_: DimType, first: c_uint, n: c_uint) -> Bool;
  pub fn isl_aff_get_domain_space(aff: AffRef) -> Option<Space>;
  pub fn isl_aff_get_space(aff: AffRef) -> Option<Space>;
  pub fn isl_aff_get_domain_local_space(aff: AffRef) -> Option<LocalSpace>;
  pub fn isl_aff_get_local_space(aff: AffRef) -> Option<LocalSpace>;
  pub fn isl_aff_get_dim_name(aff: AffRef, type_: DimType, pos: c_uint) -> Option<CStr>;
  pub fn isl_aff_get_constant_val(aff: AffRef) -> Option<Val>;
  pub fn isl_aff_get_coefficient_val(aff: AffRef, type_: DimType, pos: c_int) -> Option<Val>;
  pub fn isl_aff_coefficient_sgn(aff: AffRef, type_: DimType, pos: c_int) -> c_int;
  pub fn isl_aff_get_denominator_val(aff: AffRef) -> Option<Val>;
  pub fn isl_aff_set_constant_si(aff: Aff, v: c_int) -> Option<Aff>;
  pub fn isl_aff_set_constant_val(aff: Aff, v: Val) -> Option<Aff>;
  pub fn isl_aff_set_coefficient_si(aff: Aff, type_: DimType, pos: c_int, v: c_int) -> Option<Aff>;
  pub fn isl_aff_set_coefficient_val(aff: Aff, type_: DimType, pos: c_int, v: Val) -> Option<Aff>;
  pub fn isl_aff_add_constant_si(aff: Aff, v: c_int) -> Option<Aff>;
  pub fn isl_aff_add_constant_val(aff: Aff, v: Val) -> Option<Aff>;
  pub fn isl_aff_add_constant_num_si(aff: Aff, v: c_int) -> Option<Aff>;
  pub fn isl_aff_add_coefficient_si(aff: Aff, type_: DimType, pos: c_int, v: c_int) -> Option<Aff>;
  pub fn isl_aff_add_coefficient_val(aff: Aff, type_: DimType, pos: c_int, v: Val) -> Option<Aff>;
  pub fn isl_aff_is_cst(aff: AffRef) -> Bool;
  pub fn isl_aff_set_tuple_id(aff: Aff, type_: DimType, id: Id) -> Option<Aff>;
  pub fn isl_aff_set_dim_name(aff: Aff, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<Aff>;
  pub fn isl_aff_set_dim_id(aff: Aff, type_: DimType, pos: c_uint, id: Id) -> Option<Aff>;
  pub fn isl_aff_find_dim_by_name(aff: AffRef, type_: DimType, name: Option<CStr>) -> c_int;
  pub fn isl_aff_plain_is_equal(aff1: AffRef, aff2: AffRef) -> Bool;
  pub fn isl_aff_plain_is_zero(aff: AffRef) -> Bool;
  pub fn isl_aff_is_nan(aff: AffRef) -> Bool;
  pub fn isl_aff_get_div(aff: AffRef, pos: c_int) -> Option<Aff>;
  pub fn isl_aff_from_range(aff: Aff) -> Option<Aff>;
  pub fn isl_aff_neg(aff: Aff) -> Option<Aff>;
  pub fn isl_aff_ceil(aff: Aff) -> Option<Aff>;
  pub fn isl_aff_floor(aff: Aff) -> Option<Aff>;
  pub fn isl_aff_mod_val(aff: Aff, mod_: Val) -> Option<Aff>;
  pub fn isl_aff_mul(aff1: Aff, aff2: Aff) -> Option<Aff>;
  pub fn isl_aff_div(aff1: Aff, aff2: Aff) -> Option<Aff>;
  pub fn isl_aff_add(aff1: Aff, aff2: Aff) -> Option<Aff>;
  pub fn isl_aff_sub(aff1: Aff, aff2: Aff) -> Option<Aff>;
  pub fn isl_aff_scale_val(aff: Aff, v: Val) -> Option<Aff>;
  pub fn isl_aff_scale_down_ui(aff: Aff, f: c_uint) -> Option<Aff>;
  pub fn isl_aff_scale_down_val(aff: Aff, v: Val) -> Option<Aff>;
  pub fn isl_aff_insert_dims(aff: Aff, type_: DimType, first: c_uint, n: c_uint) -> Option<Aff>;
  pub fn isl_aff_add_dims(aff: Aff, type_: DimType, n: c_uint) -> Option<Aff>;
  pub fn isl_aff_move_dims(aff: Aff, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<Aff>;
  pub fn isl_aff_drop_dims(aff: Aff, type_: DimType, first: c_uint, n: c_uint) -> Option<Aff>;
  pub fn isl_aff_project_domain_on_params(aff: Aff) -> Option<Aff>;
  pub fn isl_aff_align_params(aff: Aff, model: Space) -> Option<Aff>;
  pub fn isl_aff_gist(aff: Aff, context: Set) -> Option<Aff>;
  pub fn isl_aff_gist_params(aff: Aff, context: Set) -> Option<Aff>;
  pub fn isl_aff_pullback_aff(aff1: Aff, aff2: Aff) -> Option<Aff>;
  pub fn isl_aff_pullback_multi_aff(aff: Aff, ma: MultiAff) -> Option<Aff>;
  pub fn isl_aff_zero_basic_set(aff: Aff) -> Option<BasicSet>;
  pub fn isl_aff_neg_basic_set(aff: Aff) -> Option<BasicSet>;
  pub fn isl_aff_eq_basic_set(aff1: Aff, aff2: Aff) -> Option<BasicSet>;
  pub fn isl_aff_eq_set(aff1: Aff, aff2: Aff) -> Option<Set>;
  pub fn isl_aff_ne_set(aff1: Aff, aff2: Aff) -> Option<Set>;
  pub fn isl_aff_le_basic_set(aff1: Aff, aff2: Aff) -> Option<BasicSet>;
  pub fn isl_aff_le_set(aff1: Aff, aff2: Aff) -> Option<Set>;
  pub fn isl_aff_lt_basic_set(aff1: Aff, aff2: Aff) -> Option<BasicSet>;
  pub fn isl_aff_lt_set(aff1: Aff, aff2: Aff) -> Option<Set>;
  pub fn isl_aff_ge_basic_set(aff1: Aff, aff2: Aff) -> Option<BasicSet>;
  pub fn isl_aff_ge_set(aff1: Aff, aff2: Aff) -> Option<Set>;
  pub fn isl_aff_gt_basic_set(aff1: Aff, aff2: Aff) -> Option<BasicSet>;
  pub fn isl_aff_gt_set(aff1: Aff, aff2: Aff) -> Option<Set>;
  pub fn isl_aff_read_from_str(ctx: CtxRef, str: Option<CStr>) -> Option<Aff>;
  pub fn isl_aff_to_str(aff: AffRef) -> Option<CString>;
  pub fn isl_printer_print_aff(p: Printer, aff: AffRef) -> Option<Printer>;
  pub fn isl_aff_dump(aff: AffRef) -> ();
  pub fn isl_pw_aff_get_ctx(pwaff: PwAffRef) -> Option<CtxRef>;
  pub fn isl_pw_aff_get_hash(pa: PwAffRef) -> c_uint;
  pub fn isl_pw_aff_get_domain_space(pwaff: PwAffRef) -> Option<Space>;
  pub fn isl_pw_aff_get_space(pwaff: PwAffRef) -> Option<Space>;
  pub fn isl_pw_aff_from_aff(aff: Aff) -> Option<PwAff>;
  pub fn isl_pw_aff_empty(dim: Space) -> Option<PwAff>;
  pub fn isl_pw_aff_alloc(set: Set, aff: Aff) -> Option<PwAff>;
  pub fn isl_pw_aff_zero_on_domain(ls: LocalSpace) -> Option<PwAff>;
  pub fn isl_pw_aff_var_on_domain(ls: LocalSpace, type_: DimType, pos: c_uint) -> Option<PwAff>;
  pub fn isl_pw_aff_nan_on_domain(ls: LocalSpace) -> Option<PwAff>;
  pub fn isl_pw_aff_val_on_domain(domain: Set, v: Val) -> Option<PwAff>;
  pub fn isl_set_indicator_function(set: Set) -> Option<PwAff>;
  pub fn isl_pw_aff_get_dim_name(pa: PwAffRef, type_: DimType, pos: c_uint) -> Option<CStr>;
  pub fn isl_pw_aff_has_dim_id(pa: PwAffRef, type_: DimType, pos: c_uint) -> Bool;
  pub fn isl_pw_aff_get_dim_id(pa: PwAffRef, type_: DimType, pos: c_uint) -> Option<Id>;
  pub fn isl_pw_aff_set_dim_id(pma: PwAff, type_: DimType, pos: c_uint, id: Id) -> Option<PwAff>;
  pub fn isl_pw_aff_find_dim_by_name(pa: PwAffRef, type_: DimType, name: Option<CStr>) -> c_int;
  pub fn isl_pw_aff_is_empty(pwaff: PwAffRef) -> Bool;
  pub fn isl_pw_aff_involves_nan(pa: PwAffRef) -> Bool;
  pub fn isl_pw_aff_plain_cmp(pa1: PwAffRef, pa2: PwAffRef) -> c_int;
  pub fn isl_pw_aff_plain_is_equal(pwaff1: PwAffRef, pwaff2: PwAffRef) -> Bool;
  pub fn isl_pw_aff_is_equal(pa1: PwAffRef, pa2: PwAffRef) -> Bool;
  pub fn isl_pw_aff_union_min(pwaff1: PwAff, pwaff2: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_union_max(pwaff1: PwAff, pwaff2: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_union_add(pwaff1: PwAff, pwaff2: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_copy(pwaff: PwAffRef) -> Option<PwAff>;
  pub fn isl_pw_aff_free(pwaff: PwAff) -> *mut c_void;
  pub fn isl_pw_aff_dim(pwaff: PwAffRef, type_: DimType) -> c_uint;
  pub fn isl_pw_aff_involves_dims(pwaff: PwAffRef, type_: DimType, first: c_uint, n: c_uint) -> Bool;
  pub fn isl_pw_aff_is_cst(pwaff: PwAffRef) -> Bool;
  pub fn isl_pw_aff_project_domain_on_params(pa: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_align_params(pwaff: PwAff, model: Space) -> Option<PwAff>;
  pub fn isl_pw_aff_has_tuple_id(pa: PwAffRef, type_: DimType) -> Bool;
  pub fn isl_pw_aff_get_tuple_id(pa: PwAffRef, type_: DimType) -> Option<Id>;
  pub fn isl_pw_aff_set_tuple_id(pwaff: PwAff, type_: DimType, id: Id) -> Option<PwAff>;
  pub fn isl_pw_aff_reset_tuple_id(pa: PwAff, type_: DimType) -> Option<PwAff>;
  pub fn isl_pw_aff_reset_user(pa: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_params(pwa: PwAff) -> Option<Set>;
  pub fn isl_pw_aff_domain(pwaff: PwAff) -> Option<Set>;
  pub fn isl_pw_aff_from_range(pwa: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_min(pwaff1: PwAff, pwaff2: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_max(pwaff1: PwAff, pwaff2: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_mul(pwaff1: PwAff, pwaff2: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_div(pa1: PwAff, pa2: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_add(pwaff1: PwAff, pwaff2: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_sub(pwaff1: PwAff, pwaff2: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_neg(pwaff: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_ceil(pwaff: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_floor(pwaff: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_mod_val(pa: PwAff, mod_: Val) -> Option<PwAff>;
  pub fn isl_pw_aff_tdiv_q(pa1: PwAff, pa2: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_tdiv_r(pa1: PwAff, pa2: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_intersect_params(pa: PwAff, set: Set) -> Option<PwAff>;
  pub fn isl_pw_aff_intersect_domain(pa: PwAff, set: Set) -> Option<PwAff>;
  pub fn isl_pw_aff_subtract_domain(pa: PwAff, set: Set) -> Option<PwAff>;
  pub fn isl_pw_aff_cond(cond: PwAff, pwaff_true: PwAff, pwaff_false: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_scale_val(pa: PwAff, v: Val) -> Option<PwAff>;
  pub fn isl_pw_aff_scale_down_val(pa: PwAff, f: Val) -> Option<PwAff>;
  pub fn isl_pw_aff_insert_dims(pwaff: PwAff, type_: DimType, first: c_uint, n: c_uint) -> Option<PwAff>;
  pub fn isl_pw_aff_add_dims(pwaff: PwAff, type_: DimType, n: c_uint) -> Option<PwAff>;
  pub fn isl_pw_aff_move_dims(pa: PwAff, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<PwAff>;
  pub fn isl_pw_aff_drop_dims(pwaff: PwAff, type_: DimType, first: c_uint, n: c_uint) -> Option<PwAff>;
  pub fn isl_pw_aff_coalesce(pwqp: PwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_gist(pwaff: PwAff, context: Set) -> Option<PwAff>;
  pub fn isl_pw_aff_gist_params(pwaff: PwAff, context: Set) -> Option<PwAff>;
  pub fn isl_pw_aff_pullback_multi_aff(pa: PwAff, ma: MultiAff) -> Option<PwAff>;
  pub fn isl_pw_aff_pullback_pw_multi_aff(pa: PwAff, pma: PwMultiAff) -> Option<PwAff>;
  pub fn isl_pw_aff_pullback_multi_pw_aff(pa: PwAff, mpa: MultiPwAff) -> Option<PwAff>;
  pub fn isl_pw_aff_n_piece(pwaff: PwAffRef) -> c_int;
  pub fn isl_pw_aff_foreach_piece(pwaff: PwAffRef, fn_: unsafe extern "C" fn(set: Set, aff: Aff, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_set_from_pw_aff(pwaff: PwAff) -> Option<Set>;
  pub fn isl_map_from_pw_aff(pwaff: PwAff) -> Option<Map>;
  pub fn isl_pw_aff_pos_set(pa: PwAff) -> Option<Set>;
  pub fn isl_pw_aff_nonneg_set(pwaff: PwAff) -> Option<Set>;
  pub fn isl_pw_aff_zero_set(pwaff: PwAff) -> Option<Set>;
  pub fn isl_pw_aff_non_zero_set(pwaff: PwAff) -> Option<Set>;
  pub fn isl_pw_aff_eq_set(pwaff1: PwAff, pwaff2: PwAff) -> Option<Set>;
  pub fn isl_pw_aff_ne_set(pwaff1: PwAff, pwaff2: PwAff) -> Option<Set>;
  pub fn isl_pw_aff_le_set(pwaff1: PwAff, pwaff2: PwAff) -> Option<Set>;
  pub fn isl_pw_aff_lt_set(pwaff1: PwAff, pwaff2: PwAff) -> Option<Set>;
  pub fn isl_pw_aff_ge_set(pwaff1: PwAff, pwaff2: PwAff) -> Option<Set>;
  pub fn isl_pw_aff_gt_set(pwaff1: PwAff, pwaff2: PwAff) -> Option<Set>;
  pub fn isl_pw_aff_eq_map(pa1: PwAff, pa2: PwAff) -> Option<Map>;
  pub fn isl_pw_aff_lt_map(pa1: PwAff, pa2: PwAff) -> Option<Map>;
  pub fn isl_pw_aff_gt_map(pa1: PwAff, pa2: PwAff) -> Option<Map>;
  pub fn isl_pw_aff_read_from_str(ctx: CtxRef, str: Option<CStr>) -> Option<PwAff>;
  pub fn isl_pw_aff_to_str(pa: PwAffRef) -> Option<CString>;
  pub fn isl_printer_print_pw_aff(p: Printer, pwaff: PwAffRef) -> Option<Printer>;
  pub fn isl_pw_aff_dump(pwaff: PwAffRef) -> ();
  pub fn isl_pw_aff_list_min(list: PwAffList) -> Option<PwAff>;
  pub fn isl_pw_aff_list_max(list: PwAffList) -> Option<PwAff>;
  pub fn isl_pw_aff_list_eq_set(list1: PwAffList, list2: PwAffList) -> Option<Set>;
  pub fn isl_pw_aff_list_ne_set(list1: PwAffList, list2: PwAffList) -> Option<Set>;
  pub fn isl_pw_aff_list_le_set(list1: PwAffList, list2: PwAffList) -> Option<Set>;
  pub fn isl_pw_aff_list_lt_set(list1: PwAffList, list2: PwAffList) -> Option<Set>;
  pub fn isl_pw_aff_list_ge_set(list1: PwAffList, list2: PwAffList) -> Option<Set>;
  pub fn isl_pw_aff_list_gt_set(list1: PwAffList, list2: PwAffList) -> Option<Set>;
  pub fn isl_multi_aff_dim(multi: MultiAffRef, type_: DimType) -> c_uint;
  pub fn isl_multi_aff_get_ctx(multi: MultiAffRef) -> Option<CtxRef>;
  pub fn isl_multi_aff_get_space(multi: MultiAffRef) -> Option<Space>;
  pub fn isl_multi_aff_get_domain_space(multi: MultiAffRef) -> Option<Space>;
  pub fn isl_multi_aff_find_dim_by_name(multi: MultiAffRef, type_: DimType, name: Option<CStr>) -> c_int;
  pub fn isl_multi_aff_from_aff_list(space: Space, list: AffList) -> Option<MultiAff>;
  pub fn isl_multi_aff_zero(space: Space) -> Option<MultiAff>;
  pub fn isl_multi_aff_copy(multi: MultiAffRef) -> Option<MultiAff>;
  pub fn isl_multi_aff_free(multi: MultiAff) -> *mut c_void;
  pub fn isl_multi_aff_plain_is_equal(multi1: MultiAffRef, multi2: MultiAffRef) -> Bool;
  pub fn isl_multi_aff_involves_nan(multi: MultiAffRef) -> Bool;
  pub fn isl_multi_aff_find_dim_by_id(multi: MultiAffRef, type_: DimType, id: IdRef) -> c_int;
  pub fn isl_multi_aff_get_dim_id(multi: MultiAffRef, type_: DimType, pos: c_uint) -> Option<Id>;
  pub fn isl_multi_aff_set_dim_name(multi: MultiAff, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<MultiAff>;
  pub fn isl_multi_aff_set_dim_id(multi: MultiAff, type_: DimType, pos: c_uint, id: Id) -> Option<MultiAff>;
  pub fn isl_multi_aff_get_tuple_name(multi: MultiAffRef, type_: DimType) -> Option<CStr>;
  pub fn isl_multi_aff_has_tuple_id(multi: MultiAffRef, type_: DimType) -> Bool;
  pub fn isl_multi_aff_get_tuple_id(multi: MultiAffRef, type_: DimType) -> Option<Id>;
  pub fn isl_multi_aff_set_tuple_name(multi: MultiAff, type_: DimType, s: Option<CStr>) -> Option<MultiAff>;
  pub fn isl_multi_aff_set_tuple_id(multi: MultiAff, type_: DimType, id: Id) -> Option<MultiAff>;
  pub fn isl_multi_aff_reset_tuple_id(multi: MultiAff, type_: DimType) -> Option<MultiAff>;
  pub fn isl_multi_aff_reset_user(multi: MultiAff) -> Option<MultiAff>;
  pub fn isl_multi_aff_drop_dims(multi: MultiAff, type_: DimType, first: c_uint, n: c_uint) -> Option<MultiAff>;
  pub fn isl_multi_aff_get_aff(multi: MultiAffRef, pos: c_int) -> Option<Aff>;
  pub fn isl_multi_aff_set_aff(multi: MultiAff, pos: c_int, el: Aff) -> Option<MultiAff>;
  pub fn isl_multi_aff_range_splice(multi1: MultiAff, pos: c_uint, multi2: MultiAff) -> Option<MultiAff>;
  pub fn isl_multi_aff_flatten_range(multi: MultiAff) -> Option<MultiAff>;
  pub fn isl_multi_aff_flat_range_product(multi1: MultiAff, multi2: MultiAff) -> Option<MultiAff>;
  pub fn isl_multi_aff_range_product(multi1: MultiAff, multi2: MultiAff) -> Option<MultiAff>;
  pub fn isl_multi_aff_factor_range(multi: MultiAff) -> Option<MultiAff>;
  pub fn isl_multi_aff_range_is_wrapping(multi: MultiAffRef) -> Bool;
  pub fn isl_multi_aff_range_factor_domain(multi: MultiAff) -> Option<MultiAff>;
  pub fn isl_multi_aff_range_factor_range(multi: MultiAff) -> Option<MultiAff>;
  pub fn isl_multi_aff_scale_val(multi: MultiAff, v: Val) -> Option<MultiAff>;
  pub fn isl_multi_aff_scale_down_val(multi: MultiAff, v: Val) -> Option<MultiAff>;
  pub fn isl_multi_aff_scale_multi_val(multi: MultiAff, mv: MultiVal) -> Option<MultiAff>;
  pub fn isl_multi_aff_scale_down_multi_val(multi: MultiAff, mv: MultiVal) -> Option<MultiAff>;
  pub fn isl_multi_aff_mod_multi_val(multi: MultiAff, mv: MultiVal) -> Option<MultiAff>;
  pub fn isl_multi_aff_add(multi1: MultiAff, multi2: MultiAff) -> Option<MultiAff>;
  pub fn isl_multi_aff_sub(multi1: MultiAff, multi2: MultiAff) -> Option<MultiAff>;
  pub fn isl_multi_aff_align_params(multi: MultiAff, model: Space) -> Option<MultiAff>;
  pub fn isl_multi_aff_from_range(multi: MultiAff) -> Option<MultiAff>;
  pub fn isl_multi_aff_plain_cmp(multi1: MultiAffRef, multi2: MultiAffRef) -> c_int;
  pub fn isl_multi_aff_neg(multi: MultiAff) -> Option<MultiAff>;
  pub fn isl_multi_aff_involves_dims(multi: MultiAffRef, type_: DimType, first: c_uint, n: c_uint) -> Bool;
  pub fn isl_multi_aff_insert_dims(multi: MultiAff, type_: DimType, first: c_uint, n: c_uint) -> Option<MultiAff>;
  pub fn isl_multi_aff_add_dims(multi: MultiAff, type_: DimType, n: c_uint) -> Option<MultiAff>;
  pub fn isl_multi_aff_project_domain_on_params(multi: MultiAff) -> Option<MultiAff>;
  pub fn isl_multi_aff_product(multi1: MultiAff, multi2: MultiAff) -> Option<MultiAff>;
  pub fn isl_multi_aff_splice(multi1: MultiAff, in_pos: c_uint, out_pos: c_uint, multi2: MultiAff) -> Option<MultiAff>;
  pub fn isl_multi_aff_from_aff(aff: Aff) -> Option<MultiAff>;
  pub fn isl_multi_aff_identity(space: Space) -> Option<MultiAff>;
  pub fn isl_multi_aff_domain_map(space: Space) -> Option<MultiAff>;
  pub fn isl_multi_aff_range_map(space: Space) -> Option<MultiAff>;
  pub fn isl_multi_aff_project_out_map(space: Space, type_: DimType, first: c_uint, n: c_uint) -> Option<MultiAff>;
  pub fn isl_multi_aff_multi_val_on_space(space: Space, mv: MultiVal) -> Option<MultiAff>;
  pub fn isl_multi_aff_floor(ma: MultiAff) -> Option<MultiAff>;
  pub fn isl_multi_aff_gist_params(maff: MultiAff, context: Set) -> Option<MultiAff>;
  pub fn isl_multi_aff_gist(maff: MultiAff, context: Set) -> Option<MultiAff>;
  pub fn isl_multi_aff_lift(maff: MultiAff, ls: *mut LocalSpace) -> Option<MultiAff>;
  pub fn isl_multi_aff_pullback_multi_aff(ma1: MultiAff, ma2: MultiAff) -> Option<MultiAff>;
  pub fn isl_multi_aff_move_dims(ma: MultiAff, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<MultiAff>;
  pub fn isl_multi_aff_lex_lt_set(ma1: MultiAff, ma2: MultiAff) -> Option<Set>;
  pub fn isl_multi_aff_lex_le_set(ma1: MultiAff, ma2: MultiAff) -> Option<Set>;
  pub fn isl_multi_aff_lex_gt_set(ma1: MultiAff, ma2: MultiAff) -> Option<Set>;
  pub fn isl_multi_aff_lex_ge_set(ma1: MultiAff, ma2: MultiAff) -> Option<Set>;
  pub fn isl_multi_aff_to_str(ma: MultiAffRef) -> Option<CString>;
  pub fn isl_printer_print_multi_aff(p: Printer, maff: MultiAffRef) -> Option<Printer>;
  pub fn isl_multi_aff_read_from_str(ctx: CtxRef, str: Option<CStr>) -> Option<MultiAff>;
  pub fn isl_multi_aff_dump(maff: MultiAffRef) -> ();
  pub fn isl_multi_pw_aff_dim(multi: MultiPwAffRef, type_: DimType) -> c_uint;
  pub fn isl_multi_pw_aff_get_ctx(multi: MultiPwAffRef) -> Option<CtxRef>;
  pub fn isl_multi_pw_aff_get_space(multi: MultiPwAffRef) -> Option<Space>;
  pub fn isl_multi_pw_aff_get_domain_space(multi: MultiPwAffRef) -> Option<Space>;
  pub fn isl_multi_pw_aff_find_dim_by_name(multi: MultiPwAffRef, type_: DimType, name: Option<CStr>) -> c_int;
  pub fn isl_multi_pw_aff_from_pw_aff_list(space: Space, list: PwAffList) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_zero(space: Space) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_copy(multi: MultiPwAffRef) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_free(multi: MultiPwAff) -> *mut c_void;
  pub fn isl_multi_pw_aff_plain_is_equal(multi1: MultiPwAffRef, multi2: MultiPwAffRef) -> Bool;
  pub fn isl_multi_pw_aff_involves_nan(multi: MultiPwAffRef) -> Bool;
  pub fn isl_multi_pw_aff_find_dim_by_id(multi: MultiPwAffRef, type_: DimType, id: IdRef) -> c_int;
  pub fn isl_multi_pw_aff_get_dim_id(multi: MultiPwAffRef, type_: DimType, pos: c_uint) -> Option<Id>;
  pub fn isl_multi_pw_aff_set_dim_name(multi: MultiPwAff, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_set_dim_id(multi: MultiPwAff, type_: DimType, pos: c_uint, id: Id) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_get_tuple_name(multi: MultiPwAffRef, type_: DimType) -> Option<CStr>;
  pub fn isl_multi_pw_aff_has_tuple_id(multi: MultiPwAffRef, type_: DimType) -> Bool;
  pub fn isl_multi_pw_aff_get_tuple_id(multi: MultiPwAffRef, type_: DimType) -> Option<Id>;
  pub fn isl_multi_pw_aff_set_tuple_name(multi: MultiPwAff, type_: DimType, s: Option<CStr>) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_set_tuple_id(multi: MultiPwAff, type_: DimType, id: Id) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_reset_tuple_id(multi: MultiPwAff, type_: DimType) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_reset_user(multi: MultiPwAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_drop_dims(multi: MultiPwAff, type_: DimType, first: c_uint, n: c_uint) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_get_pw_aff(multi: MultiPwAffRef, pos: c_int) -> Option<PwAff>;
  pub fn isl_multi_pw_aff_set_pw_aff(multi: MultiPwAff, pos: c_int, el: PwAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_range_splice(multi1: MultiPwAff, pos: c_uint, multi2: MultiPwAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_flatten_range(multi: MultiPwAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_flat_range_product(multi1: MultiPwAff, multi2: MultiPwAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_range_product(multi1: MultiPwAff, multi2: MultiPwAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_factor_range(multi: MultiPwAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_range_is_wrapping(multi: MultiPwAffRef) -> Bool;
  pub fn isl_multi_pw_aff_range_factor_domain(multi: MultiPwAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_range_factor_range(multi: MultiPwAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_scale_val(multi: MultiPwAff, v: Val) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_scale_down_val(multi: MultiPwAff, v: Val) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_scale_multi_val(multi: MultiPwAff, mv: MultiVal) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_scale_down_multi_val(multi: MultiPwAff, mv: MultiVal) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_mod_multi_val(multi: MultiPwAff, mv: MultiVal) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_add(multi1: MultiPwAff, multi2: MultiPwAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_sub(multi1: MultiPwAff, multi2: MultiPwAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_align_params(multi: MultiPwAff, model: Space) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_from_range(multi: MultiPwAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_neg(multi: MultiPwAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_involves_dims(multi: MultiPwAffRef, type_: DimType, first: c_uint, n: c_uint) -> Bool;
  pub fn isl_multi_pw_aff_insert_dims(multi: MultiPwAff, type_: DimType, first: c_uint, n: c_uint) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_add_dims(multi: MultiPwAff, type_: DimType, n: c_uint) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_project_domain_on_params(multi: MultiPwAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_product(multi1: MultiPwAff, multi2: MultiPwAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_splice(multi1: MultiPwAff, in_pos: c_uint, out_pos: c_uint, multi2: MultiPwAff) -> Option<MultiPwAff>;
  pub fn isl_pw_multi_aff_zero(space: Space) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_identity(space: Space) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_range_map(space: Space) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_project_out_map(space: Space, type_: DimType, first: c_uint, n: c_uint) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_from_multi_aff(ma: MultiAff) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_from_pw_aff(pa: PwAff) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_alloc(set: Set, maff: MultiAff) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_copy(pma: PwMultiAffRef) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_free(pma: PwMultiAff) -> *mut c_void;
  pub fn isl_pw_multi_aff_dim(pma: PwMultiAffRef, type_: DimType) -> c_uint;
  pub fn isl_pw_multi_aff_get_pw_aff(pma: PwMultiAffRef, pos: c_int) -> Option<PwAff>;
  pub fn isl_pw_multi_aff_set_pw_aff(pma: PwMultiAff, pos: c_uint, pa: PwAff) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_get_ctx(pma: PwMultiAffRef) -> Option<CtxRef>;
  pub fn isl_pw_multi_aff_get_domain_space(pma: PwMultiAffRef) -> Option<Space>;
  pub fn isl_pw_multi_aff_get_space(pma: PwMultiAffRef) -> Option<Space>;
  pub fn isl_pw_multi_aff_has_tuple_name(pma: PwMultiAffRef, type_: DimType) -> Bool;
  pub fn isl_pw_multi_aff_get_tuple_name(pma: PwMultiAffRef, type_: DimType) -> Option<CStr>;
  pub fn isl_pw_multi_aff_get_tuple_id(pma: PwMultiAffRef, type_: DimType) -> Option<Id>;
  pub fn isl_pw_multi_aff_has_tuple_id(pma: PwMultiAffRef, type_: DimType) -> Bool;
  pub fn isl_pw_multi_aff_set_tuple_id(pma: PwMultiAff, type_: DimType, id: Id) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_reset_tuple_id(pma: PwMultiAff, type_: DimType) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_reset_user(pma: PwMultiAff) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_find_dim_by_name(pma: PwMultiAffRef, type_: DimType, name: Option<CStr>) -> c_int;
  pub fn isl_pw_multi_aff_drop_dims(pma: PwMultiAff, type_: DimType, first: c_uint, n: c_uint) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_domain(pma: PwMultiAff) -> Option<Set>;
  pub fn isl_pw_multi_aff_empty(space: Space) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_from_domain(set: Set) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_multi_val_on_domain(domain: Set, mv: MultiVal) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_get_dim_name(pma: PwMultiAffRef, type_: DimType, pos: c_uint) -> Option<CStr>;
  pub fn isl_pw_multi_aff_get_dim_id(pma: PwMultiAffRef, type_: DimType, pos: c_uint) -> Option<Id>;
  pub fn isl_pw_multi_aff_set_dim_id(pma: PwMultiAff, type_: DimType, pos: c_uint, id: Id) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_involves_nan(pma: PwMultiAffRef) -> Bool;
  pub fn isl_pw_multi_aff_plain_is_equal(pma1: PwMultiAffRef, pma2: PwMultiAffRef) -> Bool;
  pub fn isl_pw_multi_aff_is_equal(pma1: PwMultiAffRef, pma2: PwMultiAffRef) -> Bool;
  pub fn isl_pw_multi_aff_fix_si(pma: PwMultiAff, type_: DimType, pos: c_uint, value: c_int) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_union_add(pma1: PwMultiAff, pma2: PwMultiAff) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_neg(pma: PwMultiAff) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_add(pma1: PwMultiAff, pma2: PwMultiAff) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_sub(pma1: PwMultiAff, pma2: PwMultiAff) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_scale_val(pma: PwMultiAff, v: Val) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_scale_down_val(pma: PwMultiAff, v: Val) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_scale_multi_val(pma: PwMultiAff, mv: MultiVal) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_union_lexmin(pma1: PwMultiAff, pma2: PwMultiAff) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_union_lexmax(pma1: PwMultiAff, pma2: PwMultiAff) -> Option<PwMultiAff>;
  pub fn isl_multi_aff_flatten_domain(ma: MultiAff) -> Option<MultiAff>;
  pub fn isl_pw_multi_aff_range_product(pma1: PwMultiAff, pma2: PwMultiAff) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_flat_range_product(pma1: PwMultiAff, pma2: PwMultiAff) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_product(pma1: PwMultiAff, pma2: PwMultiAff) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_intersect_params(pma: PwMultiAff, set: Set) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_intersect_domain(pma: PwMultiAff, set: Set) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_subtract_domain(pma: PwMultiAff, set: Set) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_project_domain_on_params(pma: PwMultiAff) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_align_params(pma: PwMultiAff, model: Space) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_coalesce(pma: PwMultiAff) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_gist_params(pma: PwMultiAff, set: Set) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_gist(pma: PwMultiAff, set: Set) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_pullback_multi_aff(pma: PwMultiAff, ma: MultiAff) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_pullback_pw_multi_aff(pma1: PwMultiAff, pma2: PwMultiAff) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_n_piece(pma: PwMultiAffRef) -> c_int;
  pub fn isl_pw_multi_aff_foreach_piece(pma: PwMultiAffRef, fn_: unsafe extern "C" fn(set: Set, maff: MultiAff, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_map_from_pw_multi_aff(pma: PwMultiAff) -> Option<Map>;
  pub fn isl_set_from_pw_multi_aff(pma: PwMultiAff) -> Option<Set>;
  pub fn isl_pw_multi_aff_to_str(pma: PwMultiAffRef) -> Option<CString>;
  pub fn isl_printer_print_pw_multi_aff(p: Printer, pma: PwMultiAffRef) -> Option<Printer>;
  pub fn isl_pw_multi_aff_from_set(set: Set) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_from_map(map: Map) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_read_from_str(ctx: CtxRef, str: Option<CStr>) -> Option<PwMultiAff>;
  pub fn isl_pw_multi_aff_dump(pma: PwMultiAffRef) -> ();
  pub fn isl_union_pw_multi_aff_empty(space: Space) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_from_aff(aff: Aff) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_from_pw_multi_aff(pma: PwMultiAff) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_from_domain(uset: UnionSet) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_multi_val_on_domain(domain: UnionSet, mv: MultiVal) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_aff_param_on_domain_id(domain: UnionSet, id: Id) -> Option<UnionPwAff>;
  pub fn isl_union_pw_multi_aff_copy(upma: UnionPwMultiAffRef) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_free(upma: UnionPwMultiAff) -> *mut c_void;
  pub fn isl_union_set_identity_union_pw_multi_aff(uset: UnionSet) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_get_union_pw_aff(upma: UnionPwMultiAffRef, pos: c_int) -> Option<UnionPwAff>;
  pub fn isl_union_pw_multi_aff_add_pw_multi_aff(upma: UnionPwMultiAff, pma: PwMultiAff) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_get_ctx(upma: UnionPwMultiAffRef) -> Option<CtxRef>;
  pub fn isl_union_pw_multi_aff_get_space(upma: UnionPwMultiAffRef) -> Option<Space>;
  pub fn isl_union_pw_multi_aff_dim(upma: UnionPwMultiAffRef, type_: DimType) -> c_uint;
  pub fn isl_union_pw_multi_aff_set_dim_name(upma: UnionPwMultiAff, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_find_dim_by_name(upma: UnionPwMultiAffRef, type_: DimType, name: Option<CStr>) -> c_int;
  pub fn isl_union_pw_multi_aff_drop_dims(upma: UnionPwMultiAff, type_: DimType, first: c_uint, n: c_uint) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_reset_user(upma: UnionPwMultiAff) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_coalesce(upma: UnionPwMultiAff) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_gist_params(upma: UnionPwMultiAff, context: Set) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_gist(upma: UnionPwMultiAff, context: UnionSet) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_pullback_union_pw_multi_aff(upma1: UnionPwMultiAff, upma2: UnionPwMultiAff) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_align_params(upma: UnionPwMultiAff, model: Space) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_n_pw_multi_aff(upma: UnionPwMultiAffRef) -> c_int;
  pub fn isl_union_pw_multi_aff_foreach_pw_multi_aff(upma: UnionPwMultiAffRef, fn_: unsafe extern "C" fn(pma: PwMultiAff, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_union_pw_multi_aff_extract_pw_multi_aff(upma: UnionPwMultiAffRef, space: Space) -> Option<PwMultiAff>;
  pub fn isl_union_pw_multi_aff_involves_nan(upma: UnionPwMultiAffRef) -> Bool;
  pub fn isl_union_pw_multi_aff_plain_is_equal(upma1: UnionPwMultiAffRef, upma2: UnionPwMultiAffRef) -> Bool;
  pub fn isl_union_pw_multi_aff_domain(upma: UnionPwMultiAff) -> Option<UnionSet>;
  pub fn isl_union_pw_multi_aff_neg(upma: UnionPwMultiAff) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_add(upma1: UnionPwMultiAff, upma2: UnionPwMultiAff) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_union_add(upma1: UnionPwMultiAff, upma2: UnionPwMultiAff) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_sub(upma1: UnionPwMultiAff, upma2: UnionPwMultiAff) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_scale_val(upma: UnionPwMultiAff, val: Val) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_scale_down_val(upma: UnionPwMultiAff, val: Val) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_scale_multi_val(upma: UnionPwMultiAff, mv: MultiVal) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_flat_range_product(upma1: UnionPwMultiAff, upma2: UnionPwMultiAff) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_intersect_params(upma: UnionPwMultiAff, set: Set) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_intersect_domain(upma: UnionPwMultiAff, uset: UnionSet) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_subtract_domain(upma: UnionPwMultiAff, uset: UnionSet) -> Option<UnionPwMultiAff>;
  pub fn isl_union_map_from_union_pw_multi_aff(upma: UnionPwMultiAff) -> Option<UnionMap>;
  pub fn isl_printer_print_union_pw_multi_aff(p: Printer, upma: UnionPwMultiAffRef) -> Option<Printer>;
  pub fn isl_union_pw_multi_aff_from_union_set(uset: UnionSet) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_from_union_map(umap: UnionMap) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_read_from_str(ctx: CtxRef, str: Option<CStr>) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_dump(upma: UnionPwMultiAffRef) -> ();
  pub fn isl_union_pw_multi_aff_to_str(upma: UnionPwMultiAffRef) -> Option<CString>;
  pub fn isl_multi_pw_aff_get_hash(mpa: MultiPwAffRef) -> c_uint;
  pub fn isl_multi_pw_aff_identity(space: Space) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_from_multi_aff(ma: MultiAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_from_pw_aff(pa: PwAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_domain(mpa: MultiPwAff) -> Option<Set>;
  pub fn isl_multi_pw_aff_intersect_params(mpa: MultiPwAff, set: Set) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_intersect_domain(mpa: MultiPwAff, domain: Set) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_coalesce(mpa: MultiPwAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_gist(mpa: MultiPwAff, set: Set) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_gist_params(mpa: MultiPwAff, set: Set) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_is_cst(mpa: MultiPwAffRef) -> Bool;
  pub fn isl_multi_pw_aff_is_equal(mpa1: MultiPwAffRef, mpa2: MultiPwAffRef) -> Bool;
  pub fn isl_multi_pw_aff_pullback_multi_aff(mpa: MultiPwAff, ma: MultiAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_pullback_pw_multi_aff(mpa: MultiPwAff, pma: PwMultiAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_pullback_multi_pw_aff(mpa1: MultiPwAff, mpa2: MultiPwAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_move_dims(pma: MultiPwAff, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<MultiPwAff>;
  pub fn isl_set_from_multi_pw_aff(mpa: MultiPwAff) -> Option<Set>;
  pub fn isl_map_from_multi_pw_aff(mpa: MultiPwAff) -> Option<Map>;
  pub fn isl_pw_multi_aff_from_multi_pw_aff(mpa: MultiPwAff) -> Option<PwMultiAff>;
  pub fn isl_multi_pw_aff_from_pw_multi_aff(pma: PwMultiAff) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_eq_map(mpa1: MultiPwAff, mpa2: MultiPwAff) -> Option<Map>;
  pub fn isl_multi_pw_aff_lex_lt_map(mpa1: MultiPwAff, mpa2: MultiPwAff) -> Option<Map>;
  pub fn isl_multi_pw_aff_lex_gt_map(mpa1: MultiPwAff, mpa2: MultiPwAff) -> Option<Map>;
  pub fn isl_multi_pw_aff_read_from_str(ctx: CtxRef, str: Option<CStr>) -> Option<MultiPwAff>;
  pub fn isl_multi_pw_aff_to_str(mpa: MultiPwAffRef) -> Option<CString>;
  pub fn isl_printer_print_multi_pw_aff(p: Printer, mpa: MultiPwAffRef) -> Option<Printer>;
  pub fn isl_multi_pw_aff_dump(mpa: MultiPwAffRef) -> ();
  pub fn isl_union_pw_aff_copy(upa: UnionPwAffRef) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_free(upa: UnionPwAff) -> *mut c_void;
  pub fn isl_union_pw_aff_get_ctx(upa: UnionPwAffRef) -> Option<CtxRef>;
  pub fn isl_union_pw_aff_get_space(upa: UnionPwAffRef) -> Option<Space>;
  pub fn isl_union_pw_aff_dim(upa: UnionPwAffRef, type_: DimType) -> c_uint;
  pub fn isl_union_pw_aff_set_dim_name(upa: UnionPwAff, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_find_dim_by_name(upa: UnionPwAffRef, type_: DimType, name: Option<CStr>) -> c_int;
  pub fn isl_union_pw_aff_drop_dims(upa: UnionPwAff, type_: DimType, first: c_uint, n: c_uint) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_reset_user(upa: UnionPwAff) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_empty(space: Space) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_from_pw_aff(pa: PwAff) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_val_on_domain(domain: UnionSet, v: Val) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_aff_on_domain(domain: UnionSet, aff: Aff) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_pw_aff_on_domain(domain: UnionSet, pa: PwAff) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_add_pw_aff(upa: UnionPwAff, pa: PwAff) -> Option<UnionPwAff>;
  pub fn isl_union_pw_multi_aff_from_union_pw_aff(upa: UnionPwAff) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_aff_n_pw_aff(upa: UnionPwAffRef) -> c_int;
  pub fn isl_union_pw_aff_foreach_pw_aff(upa: UnionPwAffRef, fn_: unsafe extern "C" fn(pa: PwAff, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_union_pw_aff_extract_pw_aff(upa: UnionPwAffRef, space: Space) -> Option<PwAff>;
  pub fn isl_union_pw_aff_involves_nan(upa: UnionPwAffRef) -> Bool;
  pub fn isl_union_pw_aff_plain_is_equal(upa1: UnionPwAffRef, upa2: UnionPwAffRef) -> Bool;
  pub fn isl_union_pw_aff_domain(upa: UnionPwAff) -> Option<UnionSet>;
  pub fn isl_union_pw_aff_neg(upa: UnionPwAff) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_add(upa1: UnionPwAff, upa2: UnionPwAff) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_union_add(upa1: UnionPwAff, upa2: UnionPwAff) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_sub(upa1: UnionPwAff, upa2: UnionPwAff) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_coalesce(upa: UnionPwAff) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_gist(upa: UnionPwAff, context: UnionSet) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_gist_params(upa: UnionPwAff, context: Set) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_pullback_union_pw_multi_aff(upa: UnionPwAff, upma: UnionPwMultiAff) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_floor(upa: UnionPwAff) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_scale_val(upa: UnionPwAff, v: Val) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_scale_down_val(upa: UnionPwAff, v: Val) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_mod_val(upa: UnionPwAff, f: Val) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_align_params(upa: UnionPwAff, model: Space) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_intersect_params(upa: UnionPwAff, set: Set) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_intersect_domain(upa: UnionPwAff, uset: UnionSet) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_subtract_domain(upa: UnionPwAff, uset: UnionSet) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_zero_union_set(upa: UnionPwAff) -> Option<UnionSet>;
  pub fn isl_union_map_from_union_pw_aff(upa: UnionPwAff) -> Option<UnionMap>;
  pub fn isl_union_pw_aff_read_from_str(ctx: CtxRef, str: Option<CStr>) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_to_str(upa: UnionPwAffRef) -> Option<CString>;
  pub fn isl_printer_print_union_pw_aff(p: Printer, upa: UnionPwAffRef) -> Option<Printer>;
  pub fn isl_union_pw_aff_dump(upa: UnionPwAffRef) -> ();
  pub fn isl_multi_union_pw_aff_dim(multi: MultiUnionPwAffRef, type_: DimType) -> c_uint;
  pub fn isl_multi_union_pw_aff_get_ctx(multi: MultiUnionPwAffRef) -> Option<CtxRef>;
  pub fn isl_multi_union_pw_aff_get_space(multi: MultiUnionPwAffRef) -> Option<Space>;
  pub fn isl_multi_union_pw_aff_get_domain_space(multi: MultiUnionPwAffRef) -> Option<Space>;
  pub fn isl_multi_union_pw_aff_find_dim_by_name(multi: MultiUnionPwAffRef, type_: DimType, name: Option<CStr>) -> c_int;
  pub fn isl_multi_union_pw_aff_from_union_pw_aff_list(space: Space, list: UnionPwAffList) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_zero(space: Space) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_copy(multi: MultiUnionPwAffRef) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_free(multi: MultiUnionPwAff) -> *mut c_void;
  pub fn isl_multi_union_pw_aff_plain_is_equal(multi1: MultiUnionPwAffRef, multi2: MultiUnionPwAffRef) -> Bool;
  pub fn isl_multi_union_pw_aff_involves_nan(multi: MultiUnionPwAffRef) -> Bool;
  pub fn isl_multi_union_pw_aff_find_dim_by_id(multi: MultiUnionPwAffRef, type_: DimType, id: IdRef) -> c_int;
  pub fn isl_multi_union_pw_aff_get_dim_id(multi: MultiUnionPwAffRef, type_: DimType, pos: c_uint) -> Option<Id>;
  pub fn isl_multi_union_pw_aff_set_dim_name(multi: MultiUnionPwAff, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_set_dim_id(multi: MultiUnionPwAff, type_: DimType, pos: c_uint, id: Id) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_get_tuple_name(multi: MultiUnionPwAffRef, type_: DimType) -> Option<CStr>;
  pub fn isl_multi_union_pw_aff_has_tuple_id(multi: MultiUnionPwAffRef, type_: DimType) -> Bool;
  pub fn isl_multi_union_pw_aff_get_tuple_id(multi: MultiUnionPwAffRef, type_: DimType) -> Option<Id>;
  pub fn isl_multi_union_pw_aff_set_tuple_name(multi: MultiUnionPwAff, type_: DimType, s: Option<CStr>) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_set_tuple_id(multi: MultiUnionPwAff, type_: DimType, id: Id) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_reset_tuple_id(multi: MultiUnionPwAff, type_: DimType) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_reset_user(multi: MultiUnionPwAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_drop_dims(multi: MultiUnionPwAff, type_: DimType, first: c_uint, n: c_uint) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_get_union_pw_aff(multi: MultiUnionPwAffRef, pos: c_int) -> Option<UnionPwAff>;
  pub fn isl_multi_union_pw_aff_set_union_pw_aff(multi: MultiUnionPwAff, pos: c_int, el: UnionPwAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_range_splice(multi1: MultiUnionPwAff, pos: c_uint, multi2: MultiUnionPwAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_flatten_range(multi: MultiUnionPwAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_flat_range_product(multi1: MultiUnionPwAff, multi2: MultiUnionPwAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_range_product(multi1: MultiUnionPwAff, multi2: MultiUnionPwAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_factor_range(multi: MultiUnionPwAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_range_is_wrapping(multi: MultiUnionPwAffRef) -> Bool;
  pub fn isl_multi_union_pw_aff_range_factor_domain(multi: MultiUnionPwAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_range_factor_range(multi: MultiUnionPwAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_scale_val(multi: MultiUnionPwAff, v: Val) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_scale_down_val(multi: MultiUnionPwAff, v: Val) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_scale_multi_val(multi: MultiUnionPwAff, mv: MultiVal) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_scale_down_multi_val(multi: MultiUnionPwAff, mv: MultiVal) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_mod_multi_val(multi: MultiUnionPwAff, mv: MultiVal) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_add(multi1: MultiUnionPwAff, multi2: MultiUnionPwAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_sub(multi1: MultiUnionPwAff, multi2: MultiUnionPwAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_align_params(multi: MultiUnionPwAff, model: Space) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_from_range(multi: MultiUnionPwAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_neg(multi: MultiUnionPwAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_from_multi_aff(ma: MultiAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_from_union_pw_aff(upa: UnionPwAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_from_multi_pw_aff(mpa: MultiPwAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_multi_val_on_domain(domain: UnionSet, mv: MultiVal) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_multi_aff_on_domain(domain: UnionSet, ma: MultiAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_pw_multi_aff_on_domain(domain: UnionSet, pma: PwMultiAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_floor(mupa: MultiUnionPwAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_intersect_domain(mupa: MultiUnionPwAff, uset: UnionSet) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_intersect_params(mupa: MultiUnionPwAff, params: Set) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_intersect_range(mupa: MultiUnionPwAff, set: Set) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_domain(mupa: MultiUnionPwAff) -> Option<UnionSet>;
  pub fn isl_multi_union_pw_aff_coalesce(aff: MultiUnionPwAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_gist(aff: MultiUnionPwAff, context: UnionSet) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_gist_params(aff: MultiUnionPwAff, context: Set) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_apply_aff(mupa: MultiUnionPwAff, aff: Aff) -> Option<UnionPwAff>;
  pub fn isl_multi_union_pw_aff_apply_multi_aff(mupa: MultiUnionPwAff, ma: MultiAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_apply_pw_aff(mupa: MultiUnionPwAff, pa: PwAff) -> Option<UnionPwAff>;
  pub fn isl_multi_union_pw_aff_apply_pw_multi_aff(mupa: MultiUnionPwAff, pma: PwMultiAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_pullback_union_pw_multi_aff(mupa: MultiUnionPwAff, upma: UnionPwMultiAff) -> Option<MultiUnionPwAff>;
  pub fn isl_union_pw_multi_aff_from_multi_union_pw_aff(mupa: MultiUnionPwAff) -> Option<UnionPwMultiAff>;
  pub fn isl_multi_union_pw_aff_union_add(mupa1: MultiUnionPwAff, mupa2: MultiUnionPwAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_from_union_pw_multi_aff(upma: UnionPwMultiAff) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_from_union_map(umap: UnionMap) -> Option<MultiUnionPwAff>;
  pub fn isl_union_map_from_multi_union_pw_aff(mupa: MultiUnionPwAff) -> Option<UnionMap>;
  pub fn isl_multi_union_pw_aff_zero_union_set(mupa: MultiUnionPwAff) -> Option<UnionSet>;
  pub fn isl_multi_union_pw_aff_extract_multi_pw_aff(mupa: MultiUnionPwAffRef, space: Space) -> Option<MultiPwAff>;
  pub fn isl_multi_union_pw_aff_read_from_str(ctx: CtxRef, str: Option<CStr>) -> Option<MultiUnionPwAff>;
  pub fn isl_multi_union_pw_aff_to_str(mupa: MultiUnionPwAffRef) -> Option<CString>;
  pub fn isl_printer_print_multi_union_pw_aff(p: Printer, mupa: MultiUnionPwAffRef) -> Option<Printer>;
  pub fn isl_multi_union_pw_aff_dump(mupa: MultiUnionPwAffRef) -> ();
  pub fn isl_union_pw_aff_list_get_ctx(list: UnionPwAffListRef) -> Option<CtxRef>;
  pub fn isl_union_pw_aff_list_from_union_pw_aff(el: UnionPwAff) -> Option<UnionPwAffList>;
  pub fn isl_union_pw_aff_list_alloc(ctx: CtxRef, n: c_int) -> Option<UnionPwAffList>;
  pub fn isl_union_pw_aff_list_copy(list: UnionPwAffListRef) -> Option<UnionPwAffList>;
  pub fn isl_union_pw_aff_list_free(list: UnionPwAffList) -> *mut c_void;
  pub fn isl_union_pw_aff_list_add(list: UnionPwAffList, el: UnionPwAff) -> Option<UnionPwAffList>;
  pub fn isl_union_pw_aff_list_insert(list: UnionPwAffList, pos: c_uint, el: UnionPwAff) -> Option<UnionPwAffList>;
  pub fn isl_union_pw_aff_list_drop(list: UnionPwAffList, first: c_uint, n: c_uint) -> Option<UnionPwAffList>;
  pub fn isl_union_pw_aff_list_concat(list1: UnionPwAffList, list2: UnionPwAffList) -> Option<UnionPwAffList>;
  pub fn isl_union_pw_aff_list_n_union_pw_aff(list: UnionPwAffListRef) -> c_int;
  pub fn isl_union_pw_aff_list_get_union_pw_aff(list: UnionPwAffListRef, index: c_int) -> Option<UnionPwAff>;
  pub fn isl_union_pw_aff_list_set_union_pw_aff(list: UnionPwAffList, index: c_int, el: UnionPwAff) -> Option<UnionPwAffList>;
  pub fn isl_union_pw_aff_list_foreach(list: UnionPwAffListRef, fn_: unsafe extern "C" fn(el: UnionPwAff, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_union_pw_aff_list_map(list: UnionPwAffList, fn_: unsafe extern "C" fn(el: UnionPwAff, user: *mut c_void) -> Option<UnionPwAff>, user: *mut c_void) -> Option<UnionPwAffList>;
  pub fn isl_union_pw_aff_list_sort(list: UnionPwAffList, cmp: unsafe extern "C" fn(a: UnionPwAffRef, b: UnionPwAffRef, user: *mut c_void) -> c_int, user: *mut c_void) -> Option<UnionPwAffList>;
  pub fn isl_union_pw_aff_list_foreach_scc(list: UnionPwAffListRef, follows: unsafe extern "C" fn(a: UnionPwAffRef, b: UnionPwAffRef, user: *mut c_void) -> Bool, follows_user: *mut c_void, fn_: unsafe extern "C" fn(scc: UnionPwAffList, user: *mut c_void) -> Stat, fn_user: *mut c_void) -> Stat;
  pub fn isl_printer_print_union_pw_aff_list(p: Printer, list: UnionPwAffListRef) -> Option<Printer>;
  pub fn isl_union_pw_aff_list_dump(list: UnionPwAffListRef) -> ();
  pub fn isl_union_pw_multi_aff_list_get_ctx(list: UnionPwMultiAffListRef) -> Option<CtxRef>;
  pub fn isl_union_pw_multi_aff_list_from_union_pw_multi_aff(el: UnionPwMultiAff) -> Option<UnionPwMultiAffList>;
  pub fn isl_union_pw_multi_aff_list_alloc(ctx: CtxRef, n: c_int) -> Option<UnionPwMultiAffList>;
  pub fn isl_union_pw_multi_aff_list_copy(list: UnionPwMultiAffListRef) -> Option<UnionPwMultiAffList>;
  pub fn isl_union_pw_multi_aff_list_free(list: UnionPwMultiAffList) -> *mut c_void;
  pub fn isl_union_pw_multi_aff_list_add(list: UnionPwMultiAffList, el: UnionPwMultiAff) -> Option<UnionPwMultiAffList>;
  pub fn isl_union_pw_multi_aff_list_insert(list: UnionPwMultiAffList, pos: c_uint, el: UnionPwMultiAff) -> Option<UnionPwMultiAffList>;
  pub fn isl_union_pw_multi_aff_list_drop(list: UnionPwMultiAffList, first: c_uint, n: c_uint) -> Option<UnionPwMultiAffList>;
  pub fn isl_union_pw_multi_aff_list_concat(list1: UnionPwMultiAffList, list2: UnionPwMultiAffList) -> Option<UnionPwMultiAffList>;
  pub fn isl_union_pw_multi_aff_list_n_union_pw_multi_aff(list: UnionPwMultiAffListRef) -> c_int;
  pub fn isl_union_pw_multi_aff_list_get_union_pw_multi_aff(list: UnionPwMultiAffListRef, index: c_int) -> Option<UnionPwMultiAff>;
  pub fn isl_union_pw_multi_aff_list_set_union_pw_multi_aff(list: UnionPwMultiAffList, index: c_int, el: UnionPwMultiAff) -> Option<UnionPwMultiAffList>;
  pub fn isl_union_pw_multi_aff_list_foreach(list: UnionPwMultiAffListRef, fn_: unsafe extern "C" fn(el: UnionPwMultiAff, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_union_pw_multi_aff_list_map(list: UnionPwMultiAffList, fn_: unsafe extern "C" fn(el: UnionPwMultiAff, user: *mut c_void) -> Option<UnionPwMultiAff>, user: *mut c_void) -> Option<UnionPwMultiAffList>;
  pub fn isl_union_pw_multi_aff_list_sort(list: UnionPwMultiAffList, cmp: unsafe extern "C" fn(a: UnionPwMultiAffRef, b: UnionPwMultiAffRef, user: *mut c_void) -> c_int, user: *mut c_void) -> Option<UnionPwMultiAffList>;
  pub fn isl_union_pw_multi_aff_list_foreach_scc(list: UnionPwMultiAffListRef, follows: unsafe extern "C" fn(a: UnionPwMultiAffRef, b: UnionPwMultiAffRef, user: *mut c_void) -> Bool, follows_user: *mut c_void, fn_: unsafe extern "C" fn(scc: UnionPwMultiAffList, user: *mut c_void) -> Stat, fn_user: *mut c_void) -> Stat;
  pub fn isl_printer_print_union_pw_multi_aff_list(p: Printer, list: UnionPwMultiAffListRef) -> Option<Printer>;
  pub fn isl_union_pw_multi_aff_list_dump(list: UnionPwMultiAffListRef) -> ();
}

impl Aff {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_aff_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_constant_si(self, v: c_int) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_set_constant_si(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_constant_val(self, v: Val) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_set_constant_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_coefficient_si(self, type_: DimType, pos: c_int, v: c_int) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_set_coefficient_si(self.to(), type_.to(), pos.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_coefficient_val(self, type_: DimType, pos: c_int, v: Val) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_set_coefficient_val(self.to(), type_.to(), pos.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_constant_si(self, v: c_int) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_add_constant_si(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_constant_val(self, v: Val) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_add_constant_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_constant_num_si(self, v: c_int) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_add_constant_num_si(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_coefficient_si(self, type_: DimType, pos: c_int, v: c_int) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_add_coefficient_si(self.to(), type_.to(), pos.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_coefficient_val(self, type_: DimType, pos: c_int, v: Val) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_add_coefficient_val(self.to(), type_.to(), pos.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_id(self, type_: DimType, id: Id) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_set_tuple_id(self.to(), type_.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_name(self, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_set_dim_name(self.to(), type_.to(), pos.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_id(self, type_: DimType, pos: c_uint, id: Id) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_set_dim_id(self.to(), type_.to(), pos.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn from_range(self) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_from_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg(self) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ceil(self) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_ceil(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn floor(self) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_floor(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mod_val(self, mod_: Val) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_mod_val(self.to(), mod_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mul(self, aff2: Aff) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_mul(self.to(), aff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn div(self, aff2: Aff) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_div(self.to(), aff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, aff2: Aff) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_add(self.to(), aff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sub(self, aff2: Aff) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_sub(self.to(), aff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_val(self, v: Val) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_scale_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_ui(self, f: c_uint) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_scale_down_ui(self.to(), f.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_val(self, v: Val) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_scale_down_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_insert_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_dims(self, type_: DimType, n: c_uint) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_add_dims(self.to(), type_.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn move_dims(self, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_move_dims(self.to(), dst_type.to(), dst_pos.to(), src_type.to(), src_pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_drop_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn project_domain_on_params(self) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_project_domain_on_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_align_params(self.to(), model.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, context: Set) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_gist(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_params(self, context: Set) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_gist_params(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pullback_aff(self, aff2: Aff) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_pullback_aff(self.to(), aff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pullback_multi_aff(self, ma: MultiAff) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_pullback_multi_aff(self.to(), ma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn zero_basic_set(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_aff_zero_basic_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg_basic_set(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_aff_neg_basic_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eq_basic_set(self, aff2: Aff) -> Option<BasicSet> {
    unsafe {
      let ret = isl_aff_eq_basic_set(self.to(), aff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eq_set(self, aff2: Aff) -> Option<Set> {
    unsafe {
      let ret = isl_aff_eq_set(self.to(), aff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ne_set(self, aff2: Aff) -> Option<Set> {
    unsafe {
      let ret = isl_aff_ne_set(self.to(), aff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn le_basic_set(self, aff2: Aff) -> Option<BasicSet> {
    unsafe {
      let ret = isl_aff_le_basic_set(self.to(), aff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn le_set(self, aff2: Aff) -> Option<Set> {
    unsafe {
      let ret = isl_aff_le_set(self.to(), aff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lt_basic_set(self, aff2: Aff) -> Option<BasicSet> {
    unsafe {
      let ret = isl_aff_lt_basic_set(self.to(), aff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lt_set(self, aff2: Aff) -> Option<Set> {
    unsafe {
      let ret = isl_aff_lt_set(self.to(), aff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ge_basic_set(self, aff2: Aff) -> Option<BasicSet> {
    unsafe {
      let ret = isl_aff_ge_basic_set(self.to(), aff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ge_set(self, aff2: Aff) -> Option<Set> {
    unsafe {
      let ret = isl_aff_ge_set(self.to(), aff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gt_basic_set(self, aff2: Aff) -> Option<BasicSet> {
    unsafe {
      let ret = isl_aff_gt_basic_set(self.to(), aff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gt_set(self, aff2: Aff) -> Option<Set> {
    unsafe {
      let ret = isl_aff_gt_set(self.to(), aff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_aff_from_aff(self) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_from_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_aff_from_aff(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_from_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_multi_aff_from_aff(self) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_from_aff(self.to());
      (ret).to()
    }
  }
}

impl AffRef {
  #[inline(always)]
  pub fn copy(self) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_aff_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_hash(self) -> c_uint {
    unsafe {
      let ret = isl_aff_get_hash(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_int {
    unsafe {
      let ret = isl_aff_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<bool> {
    unsafe {
      let ret = isl_aff_involves_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_domain_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_aff_get_domain_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_aff_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_domain_local_space(self) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_aff_get_domain_local_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_local_space(self) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_aff_get_local_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_name(self, type_: DimType, pos: c_uint) -> Option<CStr> {
    unsafe {
      let ret = isl_aff_get_dim_name(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_constant_val(self) -> Option<Val> {
    unsafe {
      let ret = isl_aff_get_constant_val(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_coefficient_val(self, type_: DimType, pos: c_int) -> Option<Val> {
    unsafe {
      let ret = isl_aff_get_coefficient_val(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coefficient_sgn(self, type_: DimType, pos: c_int) -> c_int {
    unsafe {
      let ret = isl_aff_coefficient_sgn(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_denominator_val(self) -> Option<Val> {
    unsafe {
      let ret = isl_aff_get_denominator_val(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_cst(self) -> Option<bool> {
    unsafe {
      let ret = isl_aff_is_cst(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: Option<CStr>) -> c_int {
    unsafe {
      let ret = isl_aff_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, aff2: AffRef) -> Option<bool> {
    unsafe {
      let ret = isl_aff_plain_is_equal(self.to(), aff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_zero(self) -> Option<bool> {
    unsafe {
      let ret = isl_aff_plain_is_zero(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_nan(self) -> Option<bool> {
    unsafe {
      let ret = isl_aff_is_nan(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_div(self, pos: c_int) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_get_div(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_aff_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_aff_dump(self.to());
      (ret).to()
    }
  }
}

impl CtxRef {
  #[inline(always)]
  pub fn aff_read_from_str(self, str: Option<CStr>) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_aff_read_from_str(self, str: Option<CStr>) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_aff_read_from_str(self, str: Option<CStr>) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_multi_aff_read_from_str(self, str: Option<CStr>) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_multi_aff_read_from_str(self, str: Option<CStr>) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_pw_aff_read_from_str(self, str: Option<CStr>) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_aff_read_from_str(self, str: Option<CStr>) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_union_pw_aff_read_from_str(self, str: Option<CStr>) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_aff_list_alloc(self, n: c_int) -> Option<UnionPwAffList> {
    unsafe {
      let ret = isl_union_pw_aff_list_alloc(self.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_multi_aff_list_alloc(self, n: c_int) -> Option<UnionPwMultiAffList> {
    unsafe {
      let ret = isl_union_pw_multi_aff_list_alloc(self.to(), n.to());
      (ret).to()
    }
  }
}

impl LocalSpace {
  #[inline(always)]
  pub fn aff_zero_on_domain(self) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_zero_on_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn aff_val_on_domain(self, val: Val) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_val_on_domain(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn aff_var_on_domain(self, type_: DimType, pos: c_uint) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_var_on_domain(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn aff_nan_on_domain(self) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_nan_on_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_aff_zero_on_domain(self) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_zero_on_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_aff_var_on_domain(self, type_: DimType, pos: c_uint) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_var_on_domain(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_aff_nan_on_domain(self) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_nan_on_domain(self.to());
      (ret).to()
    }
  }
}

impl Map {
  #[inline(always)]
  pub fn pw_multi_aff_from_map(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_from_map(self.to());
      (ret).to()
    }
  }
}

impl MultiAff {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_multi_aff_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_name(self, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_set_dim_name(self.to(), type_.to(), pos.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_id(self, type_: DimType, pos: c_uint, id: Id) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_set_dim_id(self.to(), type_.to(), pos.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_name(self, type_: DimType, s: Option<CStr>) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_set_tuple_name(self.to(), type_.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_id(self, type_: DimType, id: Id) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_set_tuple_id(self.to(), type_.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_tuple_id(self, type_: DimType) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_reset_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_drop_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_aff(self, pos: c_int, el: Aff) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_set_aff(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_splice(self, pos: c_uint, multi2: MultiAff) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_range_splice(self.to(), pos.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten_range(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_flatten_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flat_range_product(self, multi2: MultiAff) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_flat_range_product(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_product(self, multi2: MultiAff) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_range_product(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn factor_range(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_factor_domain(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_range_factor_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_factor_range(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_range_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_val(self, v: Val) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_scale_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_val(self, v: Val) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_scale_down_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_multi_val(self, mv: MultiVal) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_scale_multi_val(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_multi_val(self, mv: MultiVal) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_scale_down_multi_val(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mod_multi_val(self, mv: MultiVal) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_mod_multi_val(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, multi2: MultiAff) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_add(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sub(self, multi2: MultiAff) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_sub(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_align_params(self.to(), model.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn from_range(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_from_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_insert_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_dims(self, type_: DimType, n: c_uint) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_add_dims(self.to(), type_.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn project_domain_on_params(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_project_domain_on_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn product(self, multi2: MultiAff) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_product(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn splice(self, in_pos: c_uint, out_pos: c_uint, multi2: MultiAff) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_splice(self.to(), in_pos.to(), out_pos.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn floor(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_floor(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_params(self, context: Set) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_gist_params(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, context: Set) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_gist(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lift(self) -> Option<(MultiAff, LocalSpace)> {
    unsafe {
      let ref mut ls = 0 as *mut c_void;
      let ret = isl_multi_aff_lift(self.to(), ls as *mut _ as _);
      (ret, *ls).to()
    }
  }
  #[inline(always)]
  pub fn pullback_multi_aff(self, ma2: MultiAff) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_pullback_multi_aff(self.to(), ma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn move_dims(self, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_move_dims(self.to(), dst_type.to(), dst_pos.to(), src_type.to(), src_pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_lt_set(self, ma2: MultiAff) -> Option<Set> {
    unsafe {
      let ret = isl_multi_aff_lex_lt_set(self.to(), ma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_le_set(self, ma2: MultiAff) -> Option<Set> {
    unsafe {
      let ret = isl_multi_aff_lex_le_set(self.to(), ma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_gt_set(self, ma2: MultiAff) -> Option<Set> {
    unsafe {
      let ret = isl_multi_aff_lex_gt_set(self.to(), ma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_ge_set(self, ma2: MultiAff) -> Option<Set> {
    unsafe {
      let ret = isl_multi_aff_lex_ge_set(self.to(), ma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_multi_aff_from_multi_aff(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_from_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten_domain(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_flatten_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_pw_aff_from_multi_aff(self) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_from_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_union_pw_aff_from_multi_aff(self) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_from_multi_aff(self.to());
      (ret).to()
    }
  }
}

impl MultiAffRef {
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_uint {
    unsafe {
      let ret = isl_multi_aff_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_multi_aff_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_multi_aff_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_domain_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_multi_aff_get_domain_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: Option<CStr>) -> c_int {
    unsafe {
      let ret = isl_multi_aff_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, multi2: MultiAffRef) -> Option<bool> {
    unsafe {
      let ret = isl_multi_aff_plain_is_equal(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_nan(self) -> Option<bool> {
    unsafe {
      let ret = isl_multi_aff_involves_nan(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_id(self, type_: DimType, id: IdRef) -> c_int {
    unsafe {
      let ret = isl_multi_aff_find_dim_by_id(self.to(), type_.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_id(self, type_: DimType, pos: c_uint) -> Option<Id> {
    unsafe {
      let ret = isl_multi_aff_get_dim_id(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_name(self, type_: DimType) -> Option<CStr> {
    unsafe {
      let ret = isl_multi_aff_get_tuple_name(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_tuple_id(self, type_: DimType) -> Option<bool> {
    unsafe {
      let ret = isl_multi_aff_has_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_id(self, type_: DimType) -> Option<Id> {
    unsafe {
      let ret = isl_multi_aff_get_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_aff(self, pos: c_int) -> Option<Aff> {
    unsafe {
      let ret = isl_multi_aff_get_aff(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_is_wrapping(self) -> Option<bool> {
    unsafe {
      let ret = isl_multi_aff_range_is_wrapping(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_cmp(self, multi2: MultiAffRef) -> c_int {
    unsafe {
      let ret = isl_multi_aff_plain_cmp(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<bool> {
    unsafe {
      let ret = isl_multi_aff_involves_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_multi_aff_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_multi_aff_dump(self.to());
      (ret).to()
    }
  }
}

impl MultiPwAff {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_multi_pw_aff_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_name(self, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_set_dim_name(self.to(), type_.to(), pos.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_id(self, type_: DimType, pos: c_uint, id: Id) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_set_dim_id(self.to(), type_.to(), pos.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_name(self, type_: DimType, s: Option<CStr>) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_set_tuple_name(self.to(), type_.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_id(self, type_: DimType, id: Id) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_set_tuple_id(self.to(), type_.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_tuple_id(self, type_: DimType) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_reset_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_drop_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_pw_aff(self, pos: c_int, el: PwAff) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_set_pw_aff(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_splice(self, pos: c_uint, multi2: MultiPwAff) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_range_splice(self.to(), pos.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten_range(self) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_flatten_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flat_range_product(self, multi2: MultiPwAff) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_flat_range_product(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_product(self, multi2: MultiPwAff) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_range_product(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn factor_range(self) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_factor_domain(self) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_range_factor_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_factor_range(self) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_range_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_val(self, v: Val) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_scale_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_val(self, v: Val) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_scale_down_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_multi_val(self, mv: MultiVal) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_scale_multi_val(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_multi_val(self, mv: MultiVal) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_scale_down_multi_val(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mod_multi_val(self, mv: MultiVal) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_mod_multi_val(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, multi2: MultiPwAff) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_add(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sub(self, multi2: MultiPwAff) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_sub(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_align_params(self.to(), model.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn from_range(self) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_from_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg(self) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_insert_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_dims(self, type_: DimType, n: c_uint) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_add_dims(self.to(), type_.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn project_domain_on_params(self) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_project_domain_on_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn product(self, multi2: MultiPwAff) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_product(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn splice(self, in_pos: c_uint, out_pos: c_uint, multi2: MultiPwAff) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_splice(self.to(), in_pos.to(), out_pos.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain(self) -> Option<Set> {
    unsafe {
      let ret = isl_multi_pw_aff_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_params(self, set: Set) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_intersect_params(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain(self, domain: Set) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_intersect_domain(self.to(), domain.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coalesce(self) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_coalesce(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, set: Set) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_gist(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_params(self, set: Set) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_gist_params(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pullback_multi_aff(self, ma: MultiAff) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_pullback_multi_aff(self.to(), ma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pullback_pw_multi_aff(self, pma: PwMultiAff) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_pullback_pw_multi_aff(self.to(), pma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pullback_multi_pw_aff(self, mpa2: MultiPwAff) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_pullback_multi_pw_aff(self.to(), mpa2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn move_dims(self, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_move_dims(self.to(), dst_type.to(), dst_pos.to(), src_type.to(), src_pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_from_multi_pw_aff(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_from_multi_pw_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_from_multi_pw_aff(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_from_multi_pw_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_multi_aff_from_multi_pw_aff(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_from_multi_pw_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eq_map(self, mpa2: MultiPwAff) -> Option<Map> {
    unsafe {
      let ret = isl_multi_pw_aff_eq_map(self.to(), mpa2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_lt_map(self, mpa2: MultiPwAff) -> Option<Map> {
    unsafe {
      let ret = isl_multi_pw_aff_lex_lt_map(self.to(), mpa2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_gt_map(self, mpa2: MultiPwAff) -> Option<Map> {
    unsafe {
      let ret = isl_multi_pw_aff_lex_gt_map(self.to(), mpa2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_union_pw_aff_from_multi_pw_aff(self) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_from_multi_pw_aff(self.to());
      (ret).to()
    }
  }
}

impl MultiPwAffRef {
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_uint {
    unsafe {
      let ret = isl_multi_pw_aff_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_multi_pw_aff_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_multi_pw_aff_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_domain_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_multi_pw_aff_get_domain_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: Option<CStr>) -> c_int {
    unsafe {
      let ret = isl_multi_pw_aff_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, multi2: MultiPwAffRef) -> Option<bool> {
    unsafe {
      let ret = isl_multi_pw_aff_plain_is_equal(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_nan(self) -> Option<bool> {
    unsafe {
      let ret = isl_multi_pw_aff_involves_nan(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_id(self, type_: DimType, id: IdRef) -> c_int {
    unsafe {
      let ret = isl_multi_pw_aff_find_dim_by_id(self.to(), type_.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_id(self, type_: DimType, pos: c_uint) -> Option<Id> {
    unsafe {
      let ret = isl_multi_pw_aff_get_dim_id(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_name(self, type_: DimType) -> Option<CStr> {
    unsafe {
      let ret = isl_multi_pw_aff_get_tuple_name(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_tuple_id(self, type_: DimType) -> Option<bool> {
    unsafe {
      let ret = isl_multi_pw_aff_has_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_id(self, type_: DimType) -> Option<Id> {
    unsafe {
      let ret = isl_multi_pw_aff_get_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_pw_aff(self, pos: c_int) -> Option<PwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_get_pw_aff(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_is_wrapping(self) -> Option<bool> {
    unsafe {
      let ret = isl_multi_pw_aff_range_is_wrapping(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<bool> {
    unsafe {
      let ret = isl_multi_pw_aff_involves_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_hash(self) -> c_uint {
    unsafe {
      let ret = isl_multi_pw_aff_get_hash(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_cst(self) -> Option<bool> {
    unsafe {
      let ret = isl_multi_pw_aff_is_cst(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_equal(self, mpa2: MultiPwAffRef) -> Option<bool> {
    unsafe {
      let ret = isl_multi_pw_aff_is_equal(self.to(), mpa2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_multi_pw_aff_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_multi_pw_aff_dump(self.to());
      (ret).to()
    }
  }
}

impl MultiUnionPwAff {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_multi_union_pw_aff_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_name(self, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_set_dim_name(self.to(), type_.to(), pos.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_id(self, type_: DimType, pos: c_uint, id: Id) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_set_dim_id(self.to(), type_.to(), pos.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_name(self, type_: DimType, s: Option<CStr>) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_set_tuple_name(self.to(), type_.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_id(self, type_: DimType, id: Id) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_set_tuple_id(self.to(), type_.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_tuple_id(self, type_: DimType) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_reset_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_drop_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_union_pw_aff(self, pos: c_int, el: UnionPwAff) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_set_union_pw_aff(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_splice(self, pos: c_uint, multi2: MultiUnionPwAff) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_range_splice(self.to(), pos.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten_range(self) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_flatten_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flat_range_product(self, multi2: MultiUnionPwAff) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_flat_range_product(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_product(self, multi2: MultiUnionPwAff) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_range_product(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn factor_range(self) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_factor_domain(self) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_range_factor_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_factor_range(self) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_range_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_val(self, v: Val) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_scale_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_val(self, v: Val) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_scale_down_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_multi_val(self, mv: MultiVal) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_scale_multi_val(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_multi_val(self, mv: MultiVal) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_scale_down_multi_val(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mod_multi_val(self, mv: MultiVal) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_mod_multi_val(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, multi2: MultiUnionPwAff) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_add(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sub(self, multi2: MultiUnionPwAff) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_sub(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_align_params(self.to(), model.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn from_range(self) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_from_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg(self) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn floor(self) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_floor(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain(self, uset: UnionSet) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_intersect_domain(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_params(self, params: Set) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_intersect_params(self.to(), params.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_range(self, set: Set) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_intersect_range(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_multi_union_pw_aff_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coalesce(self) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_coalesce(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, context: UnionSet) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_gist(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_params(self, context: Set) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_gist_params(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn apply_aff(self, aff: Aff) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_apply_aff(self.to(), aff.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn apply_multi_aff(self, ma: MultiAff) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_apply_multi_aff(self.to(), ma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn apply_pw_aff(self, pa: PwAff) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_apply_pw_aff(self.to(), pa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn apply_pw_multi_aff(self, pma: PwMultiAff) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_apply_pw_multi_aff(self.to(), pma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pullback_union_pw_multi_aff(self, upma: UnionPwMultiAff) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_pullback_union_pw_multi_aff(self.to(), upma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_multi_aff_from_multi_union_pw_aff(self) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_from_multi_union_pw_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_add(self, mupa2: MultiUnionPwAff) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_union_add(self.to(), mupa2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_map_from_multi_union_pw_aff(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_from_multi_union_pw_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn zero_union_set(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_multi_union_pw_aff_zero_union_set(self.to());
      (ret).to()
    }
  }
}

impl MultiUnionPwAffRef {
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_uint {
    unsafe {
      let ret = isl_multi_union_pw_aff_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_multi_union_pw_aff_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_multi_union_pw_aff_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_domain_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_multi_union_pw_aff_get_domain_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: Option<CStr>) -> c_int {
    unsafe {
      let ret = isl_multi_union_pw_aff_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, multi2: MultiUnionPwAffRef) -> Option<bool> {
    unsafe {
      let ret = isl_multi_union_pw_aff_plain_is_equal(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_nan(self) -> Option<bool> {
    unsafe {
      let ret = isl_multi_union_pw_aff_involves_nan(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_id(self, type_: DimType, id: IdRef) -> c_int {
    unsafe {
      let ret = isl_multi_union_pw_aff_find_dim_by_id(self.to(), type_.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_id(self, type_: DimType, pos: c_uint) -> Option<Id> {
    unsafe {
      let ret = isl_multi_union_pw_aff_get_dim_id(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_name(self, type_: DimType) -> Option<CStr> {
    unsafe {
      let ret = isl_multi_union_pw_aff_get_tuple_name(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_tuple_id(self, type_: DimType) -> Option<bool> {
    unsafe {
      let ret = isl_multi_union_pw_aff_has_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_id(self, type_: DimType) -> Option<Id> {
    unsafe {
      let ret = isl_multi_union_pw_aff_get_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_union_pw_aff(self, pos: c_int) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_get_union_pw_aff(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_is_wrapping(self) -> Option<bool> {
    unsafe {
      let ret = isl_multi_union_pw_aff_range_is_wrapping(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn extract_multi_pw_aff(self, space: Space) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_extract_multi_pw_aff(self.to(), space.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_multi_union_pw_aff_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_multi_union_pw_aff_dump(self.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_aff(self, aff: AffRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_aff(self.to(), aff.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_pw_aff(self, pwaff: PwAffRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_pw_aff(self.to(), pwaff.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_multi_aff(self, maff: MultiAffRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_multi_aff(self.to(), maff.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_pw_multi_aff(self, pma: PwMultiAffRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_pw_multi_aff(self.to(), pma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_union_pw_multi_aff(self, upma: UnionPwMultiAffRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_union_pw_multi_aff(self.to(), upma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_multi_pw_aff(self, mpa: MultiPwAffRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_multi_pw_aff(self.to(), mpa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_union_pw_aff(self, upa: UnionPwAffRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_union_pw_aff(self.to(), upa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_multi_union_pw_aff(self, mupa: MultiUnionPwAffRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_multi_union_pw_aff(self.to(), mupa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_union_pw_aff_list(self, list: UnionPwAffListRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_union_pw_aff_list(self.to(), list.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_union_pw_multi_aff_list(self, list: UnionPwMultiAffListRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_union_pw_multi_aff_list(self.to(), list.to());
      (ret).to()
    }
  }
}

impl PwAff {
  #[inline(always)]
  pub fn set_dim_id(self, type_: DimType, pos: c_uint, id: Id) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_set_dim_id(self.to(), type_.to(), pos.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_min(self, pwaff2: PwAff) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_union_min(self.to(), pwaff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_max(self, pwaff2: PwAff) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_union_max(self.to(), pwaff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_add(self, pwaff2: PwAff) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_union_add(self.to(), pwaff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_pw_aff_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn project_domain_on_params(self) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_project_domain_on_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_align_params(self.to(), model.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_id(self, type_: DimType, id: Id) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_set_tuple_id(self.to(), type_.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_tuple_id(self, type_: DimType) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_reset_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn params(self) -> Option<Set> {
    unsafe {
      let ret = isl_pw_aff_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain(self) -> Option<Set> {
    unsafe {
      let ret = isl_pw_aff_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn from_range(self) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_from_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn min(self, pwaff2: PwAff) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_min(self.to(), pwaff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn max(self, pwaff2: PwAff) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_max(self.to(), pwaff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mul(self, pwaff2: PwAff) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_mul(self.to(), pwaff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn div(self, pa2: PwAff) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_div(self.to(), pa2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, pwaff2: PwAff) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_add(self.to(), pwaff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sub(self, pwaff2: PwAff) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_sub(self.to(), pwaff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg(self) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ceil(self) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_ceil(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn floor(self) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_floor(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mod_val(self, mod_: Val) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_mod_val(self.to(), mod_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn tdiv_q(self, pa2: PwAff) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_tdiv_q(self.to(), pa2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn tdiv_r(self, pa2: PwAff) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_tdiv_r(self.to(), pa2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_params(self, set: Set) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_intersect_params(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain(self, set: Set) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_intersect_domain(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract_domain(self, set: Set) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_subtract_domain(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn cond(self, pwaff_true: PwAff, pwaff_false: PwAff) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_cond(self.to(), pwaff_true.to(), pwaff_false.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_val(self, v: Val) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_scale_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_val(self, f: Val) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_scale_down_val(self.to(), f.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_insert_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_dims(self, type_: DimType, n: c_uint) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_add_dims(self.to(), type_.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn move_dims(self, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_move_dims(self.to(), dst_type.to(), dst_pos.to(), src_type.to(), src_pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_drop_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coalesce(self) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_coalesce(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, context: Set) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_gist(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_params(self, context: Set) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_gist_params(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pullback_multi_aff(self, ma: MultiAff) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_pullback_multi_aff(self.to(), ma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pullback_pw_multi_aff(self, pma: PwMultiAff) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_pullback_pw_multi_aff(self.to(), pma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pullback_multi_pw_aff(self, mpa: MultiPwAff) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_pullback_multi_pw_aff(self.to(), mpa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_from_pw_aff(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_from_pw_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_from_pw_aff(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_from_pw_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pos_set(self) -> Option<Set> {
    unsafe {
      let ret = isl_pw_aff_pos_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn nonneg_set(self) -> Option<Set> {
    unsafe {
      let ret = isl_pw_aff_nonneg_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn zero_set(self) -> Option<Set> {
    unsafe {
      let ret = isl_pw_aff_zero_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn non_zero_set(self) -> Option<Set> {
    unsafe {
      let ret = isl_pw_aff_non_zero_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eq_set(self, pwaff2: PwAff) -> Option<Set> {
    unsafe {
      let ret = isl_pw_aff_eq_set(self.to(), pwaff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ne_set(self, pwaff2: PwAff) -> Option<Set> {
    unsafe {
      let ret = isl_pw_aff_ne_set(self.to(), pwaff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn le_set(self, pwaff2: PwAff) -> Option<Set> {
    unsafe {
      let ret = isl_pw_aff_le_set(self.to(), pwaff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lt_set(self, pwaff2: PwAff) -> Option<Set> {
    unsafe {
      let ret = isl_pw_aff_lt_set(self.to(), pwaff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ge_set(self, pwaff2: PwAff) -> Option<Set> {
    unsafe {
      let ret = isl_pw_aff_ge_set(self.to(), pwaff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gt_set(self, pwaff2: PwAff) -> Option<Set> {
    unsafe {
      let ret = isl_pw_aff_gt_set(self.to(), pwaff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eq_map(self, pa2: PwAff) -> Option<Map> {
    unsafe {
      let ret = isl_pw_aff_eq_map(self.to(), pa2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lt_map(self, pa2: PwAff) -> Option<Map> {
    unsafe {
      let ret = isl_pw_aff_lt_map(self.to(), pa2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gt_map(self, pa2: PwAff) -> Option<Map> {
    unsafe {
      let ret = isl_pw_aff_gt_map(self.to(), pa2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_multi_aff_from_pw_aff(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_from_pw_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_pw_aff_from_pw_aff(self) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_from_pw_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_aff_from_pw_aff(self) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_from_pw_aff(self.to());
      (ret).to()
    }
  }
}

impl PwAffList {
  #[inline(always)]
  pub fn min(self) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_list_min(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn max(self) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_list_max(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eq_set(self, list2: PwAffList) -> Option<Set> {
    unsafe {
      let ret = isl_pw_aff_list_eq_set(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ne_set(self, list2: PwAffList) -> Option<Set> {
    unsafe {
      let ret = isl_pw_aff_list_ne_set(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn le_set(self, list2: PwAffList) -> Option<Set> {
    unsafe {
      let ret = isl_pw_aff_list_le_set(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lt_set(self, list2: PwAffList) -> Option<Set> {
    unsafe {
      let ret = isl_pw_aff_list_lt_set(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ge_set(self, list2: PwAffList) -> Option<Set> {
    unsafe {
      let ret = isl_pw_aff_list_ge_set(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gt_set(self, list2: PwAffList) -> Option<Set> {
    unsafe {
      let ret = isl_pw_aff_list_gt_set(self.to(), list2.to());
      (ret).to()
    }
  }
}

impl PwAffRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_pw_aff_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_hash(self) -> c_uint {
    unsafe {
      let ret = isl_pw_aff_get_hash(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_domain_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_pw_aff_get_domain_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_pw_aff_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_name(self, type_: DimType, pos: c_uint) -> Option<CStr> {
    unsafe {
      let ret = isl_pw_aff_get_dim_name(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_dim_id(self, type_: DimType, pos: c_uint) -> Option<bool> {
    unsafe {
      let ret = isl_pw_aff_has_dim_id(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_id(self, type_: DimType, pos: c_uint) -> Option<Id> {
    unsafe {
      let ret = isl_pw_aff_get_dim_id(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: Option<CStr>) -> c_int {
    unsafe {
      let ret = isl_pw_aff_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_empty(self) -> Option<bool> {
    unsafe {
      let ret = isl_pw_aff_is_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_nan(self) -> Option<bool> {
    unsafe {
      let ret = isl_pw_aff_involves_nan(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_cmp(self, pa2: PwAffRef) -> c_int {
    unsafe {
      let ret = isl_pw_aff_plain_cmp(self.to(), pa2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, pwaff2: PwAffRef) -> Option<bool> {
    unsafe {
      let ret = isl_pw_aff_plain_is_equal(self.to(), pwaff2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_equal(self, pa2: PwAffRef) -> Option<bool> {
    unsafe {
      let ret = isl_pw_aff_is_equal(self.to(), pa2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_uint {
    unsafe {
      let ret = isl_pw_aff_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<bool> {
    unsafe {
      let ret = isl_pw_aff_involves_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_cst(self) -> Option<bool> {
    unsafe {
      let ret = isl_pw_aff_is_cst(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_tuple_id(self, type_: DimType) -> Option<bool> {
    unsafe {
      let ret = isl_pw_aff_has_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_id(self, type_: DimType) -> Option<Id> {
    unsafe {
      let ret = isl_pw_aff_get_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_piece(self) -> c_int {
    unsafe {
      let ret = isl_pw_aff_n_piece(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_piece<F1: FnMut(Set, Aff) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(Set, Aff) -> Option<()>>(set: Set, aff: Aff, user: *mut c_void) -> Stat { (*(user as *mut F))(set.to(), aff.to()).to() }
    unsafe {
      let ret = isl_pw_aff_foreach_piece(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_pw_aff_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_pw_aff_dump(self.to());
      (ret).to()
    }
  }
}

impl PwMultiAff {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_pw_multi_aff_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_pw_aff(self, pos: c_uint, pa: PwAff) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_set_pw_aff(self.to(), pos.to(), pa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_id(self, type_: DimType, id: Id) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_set_tuple_id(self.to(), type_.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_tuple_id(self, type_: DimType) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_reset_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_drop_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain(self) -> Option<Set> {
    unsafe {
      let ret = isl_pw_multi_aff_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_id(self, type_: DimType, pos: c_uint, id: Id) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_set_dim_id(self.to(), type_.to(), pos.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn fix_si(self, type_: DimType, pos: c_uint, value: c_int) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_fix_si(self.to(), type_.to(), pos.to(), value.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_add(self, pma2: PwMultiAff) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_union_add(self.to(), pma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, pma2: PwMultiAff) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_add(self.to(), pma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sub(self, pma2: PwMultiAff) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_sub(self.to(), pma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_val(self, v: Val) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_scale_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_val(self, v: Val) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_scale_down_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_multi_val(self, mv: MultiVal) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_scale_multi_val(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_lexmin(self, pma2: PwMultiAff) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_union_lexmin(self.to(), pma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_lexmax(self, pma2: PwMultiAff) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_union_lexmax(self.to(), pma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_product(self, pma2: PwMultiAff) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_range_product(self.to(), pma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flat_range_product(self, pma2: PwMultiAff) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_flat_range_product(self.to(), pma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn product(self, pma2: PwMultiAff) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_product(self.to(), pma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_params(self, set: Set) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_intersect_params(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain(self, set: Set) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_intersect_domain(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract_domain(self, set: Set) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_subtract_domain(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn project_domain_on_params(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_project_domain_on_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_align_params(self.to(), model.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coalesce(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_coalesce(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_params(self, set: Set) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_gist_params(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, set: Set) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_gist(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pullback_multi_aff(self, ma: MultiAff) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_pullback_multi_aff(self.to(), ma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pullback_pw_multi_aff(self, pma2: PwMultiAff) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_pullback_pw_multi_aff(self.to(), pma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_from_pw_multi_aff(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_from_pw_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_from_pw_multi_aff(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_from_pw_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_multi_aff_from_pw_multi_aff(self) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_from_pw_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_pw_aff_from_pw_multi_aff(self) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_from_pw_multi_aff(self.to());
      (ret).to()
    }
  }
}

impl PwMultiAffRef {
  #[inline(always)]
  pub fn copy(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_uint {
    unsafe {
      let ret = isl_pw_multi_aff_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_pw_aff(self, pos: c_int) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_multi_aff_get_pw_aff(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_pw_multi_aff_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_domain_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_pw_multi_aff_get_domain_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_pw_multi_aff_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_tuple_name(self, type_: DimType) -> Option<bool> {
    unsafe {
      let ret = isl_pw_multi_aff_has_tuple_name(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_name(self, type_: DimType) -> Option<CStr> {
    unsafe {
      let ret = isl_pw_multi_aff_get_tuple_name(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_id(self, type_: DimType) -> Option<Id> {
    unsafe {
      let ret = isl_pw_multi_aff_get_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_tuple_id(self, type_: DimType) -> Option<bool> {
    unsafe {
      let ret = isl_pw_multi_aff_has_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: Option<CStr>) -> c_int {
    unsafe {
      let ret = isl_pw_multi_aff_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_name(self, type_: DimType, pos: c_uint) -> Option<CStr> {
    unsafe {
      let ret = isl_pw_multi_aff_get_dim_name(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_id(self, type_: DimType, pos: c_uint) -> Option<Id> {
    unsafe {
      let ret = isl_pw_multi_aff_get_dim_id(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_nan(self) -> Option<bool> {
    unsafe {
      let ret = isl_pw_multi_aff_involves_nan(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, pma2: PwMultiAffRef) -> Option<bool> {
    unsafe {
      let ret = isl_pw_multi_aff_plain_is_equal(self.to(), pma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_equal(self, pma2: PwMultiAffRef) -> Option<bool> {
    unsafe {
      let ret = isl_pw_multi_aff_is_equal(self.to(), pma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_piece(self) -> c_int {
    unsafe {
      let ret = isl_pw_multi_aff_n_piece(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_piece<F1: FnMut(Set, MultiAff) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(Set, MultiAff) -> Option<()>>(set: Set, maff: MultiAff, user: *mut c_void) -> Stat { (*(user as *mut F))(set.to(), maff.to()).to() }
    unsafe {
      let ret = isl_pw_multi_aff_foreach_piece(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_pw_multi_aff_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_pw_multi_aff_dump(self.to());
      (ret).to()
    }
  }
}

impl Set {
  #[inline(always)]
  pub fn pw_aff_alloc(self, aff: Aff) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_alloc(self.to(), aff.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_aff_val_on_domain(self, v: Val) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_val_on_domain(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn indicator_function(self) -> Option<PwAff> {
    unsafe {
      let ret = isl_set_indicator_function(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_multi_aff_alloc(self, maff: MultiAff) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_alloc(self.to(), maff.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_multi_aff_from_domain(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_from_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_multi_aff_multi_val_on_domain(self, mv: MultiVal) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_multi_val_on_domain(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_multi_aff_from_set(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_from_set(self.to());
      (ret).to()
    }
  }
}

impl Space {
  #[inline(always)]
  pub fn aff_param_on_domain_space_id(self, id: Id) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_param_on_domain_space_id(self.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_aff_empty(self) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_aff_from_aff_list(self, list: AffList) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_from_aff_list(self.to(), list.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_aff_zero(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_zero(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_aff_identity(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_identity(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_aff_domain_map(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_domain_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_aff_range_map(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_range_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_aff_project_out_map(self, type_: DimType, first: c_uint, n: c_uint) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_project_out_map(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_aff_multi_val_on_space(self, mv: MultiVal) -> Option<MultiAff> {
    unsafe {
      let ret = isl_multi_aff_multi_val_on_space(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_pw_aff_from_pw_aff_list(self, list: PwAffList) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_from_pw_aff_list(self.to(), list.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_pw_aff_zero(self) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_zero(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_multi_aff_zero(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_zero(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_multi_aff_identity(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_identity(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_multi_aff_range_map(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_range_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_multi_aff_project_out_map(self, type_: DimType, first: c_uint, n: c_uint) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_project_out_map(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_multi_aff_empty(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_pw_multi_aff_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_multi_aff_empty(self) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_pw_aff_identity(self) -> Option<MultiPwAff> {
    unsafe {
      let ret = isl_multi_pw_aff_identity(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_aff_empty(self) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_union_pw_aff_from_union_pw_aff_list(self, list: UnionPwAffList) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_from_union_pw_aff_list(self.to(), list.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_union_pw_aff_zero(self) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_zero(self.to());
      (ret).to()
    }
  }
}

impl UnionMap {
  #[inline(always)]
  pub fn union_pw_multi_aff_from_union_map(self) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_from_union_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_union_pw_aff_from_union_map(self) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_from_union_map(self.to());
      (ret).to()
    }
  }
}

impl UnionPwAff {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_union_pw_aff_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_name(self, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_set_dim_name(self.to(), type_.to(), pos.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_drop_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_pw_aff(self, pa: PwAff) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_add_pw_aff(self.to(), pa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_multi_aff_from_union_pw_aff(self) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_from_union_pw_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_pw_aff_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg(self) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, upa2: UnionPwAff) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_add(self.to(), upa2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_add(self, upa2: UnionPwAff) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_union_add(self.to(), upa2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sub(self, upa2: UnionPwAff) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_sub(self.to(), upa2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coalesce(self) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_coalesce(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, context: UnionSet) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_gist(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_params(self, context: Set) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_gist_params(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pullback_union_pw_multi_aff(self, upma: UnionPwMultiAff) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_pullback_union_pw_multi_aff(self.to(), upma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn floor(self) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_floor(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_val(self, v: Val) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_scale_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_val(self, v: Val) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_scale_down_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mod_val(self, f: Val) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_mod_val(self.to(), f.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_align_params(self.to(), model.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_params(self, set: Set) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_intersect_params(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain(self, uset: UnionSet) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_intersect_domain(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract_domain(self, uset: UnionSet) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_subtract_domain(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn zero_union_set(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_pw_aff_zero_union_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_map_from_union_pw_aff(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_from_union_pw_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_union_pw_aff_from_union_pw_aff(self) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_from_union_pw_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn list_from_union_pw_aff(self) -> Option<UnionPwAffList> {
    unsafe {
      let ret = isl_union_pw_aff_list_from_union_pw_aff(self.to());
      (ret).to()
    }
  }
}

impl UnionPwAffList {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_union_pw_aff_list_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, el: UnionPwAff) -> Option<UnionPwAffList> {
    unsafe {
      let ret = isl_union_pw_aff_list_add(self.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert(self, pos: c_uint, el: UnionPwAff) -> Option<UnionPwAffList> {
    unsafe {
      let ret = isl_union_pw_aff_list_insert(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, first: c_uint, n: c_uint) -> Option<UnionPwAffList> {
    unsafe {
      let ret = isl_union_pw_aff_list_drop(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, list2: UnionPwAffList) -> Option<UnionPwAffList> {
    unsafe {
      let ret = isl_union_pw_aff_list_concat(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_union_pw_aff(self, index: c_int, el: UnionPwAff) -> Option<UnionPwAffList> {
    unsafe {
      let ret = isl_union_pw_aff_list_set_union_pw_aff(self.to(), index.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map<F1: FnMut(UnionPwAff) -> Option<UnionPwAff>>(self, fn_: &mut F1) -> Option<UnionPwAffList> {
    unsafe extern "C" fn fn1<F: FnMut(UnionPwAff) -> Option<UnionPwAff>>(el: UnionPwAff, user: *mut c_void) -> Option<UnionPwAff> { (*(user as *mut F))(el.to()).to() }
    unsafe {
      let ret = isl_union_pw_aff_list_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort<F1: FnMut(UnionPwAffRef, UnionPwAffRef) -> c_int>(self, cmp: &mut F1) -> Option<UnionPwAffList> {
    unsafe extern "C" fn fn1<F: FnMut(UnionPwAffRef, UnionPwAffRef) -> c_int>(a: UnionPwAffRef, b: UnionPwAffRef, user: *mut c_void) -> c_int { (*(user as *mut F))(a.to(), b.to()).to() }
    unsafe {
      let ret = isl_union_pw_aff_list_sort(self.to(), fn1::<F1>, cmp as *mut _ as _);
      (ret).to()
    }
  }
}

impl UnionPwAffListRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_union_pw_aff_list_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<UnionPwAffList> {
    unsafe {
      let ret = isl_union_pw_aff_list_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_union_pw_aff(self) -> c_int {
    unsafe {
      let ret = isl_union_pw_aff_list_n_union_pw_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_union_pw_aff(self, index: c_int) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_list_get_union_pw_aff(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(UnionPwAff) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(UnionPwAff) -> Option<()>>(el: UnionPwAff, user: *mut c_void) -> Stat { (*(user as *mut F))(el.to()).to() }
    unsafe {
      let ret = isl_union_pw_aff_list_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_scc<F1: FnMut(UnionPwAffRef, UnionPwAffRef) -> Option<bool>, F2: FnMut(UnionPwAffList) -> Option<()>>(self, follows: &mut F1, fn_: &mut F2) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(UnionPwAffRef, UnionPwAffRef) -> Option<bool>>(a: UnionPwAffRef, b: UnionPwAffRef, user: *mut c_void) -> Bool { (*(user as *mut F))(a.to(), b.to()).to() }
    unsafe extern "C" fn fn2<F: FnMut(UnionPwAffList) -> Option<()>>(scc: UnionPwAffList, user: *mut c_void) -> Stat { (*(user as *mut F))(scc.to()).to() }
    unsafe {
      let ret = isl_union_pw_aff_list_foreach_scc(self.to(), fn1::<F1>, follows as *mut _ as _, fn2::<F2>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_union_pw_aff_list_dump(self.to());
      (ret).to()
    }
  }
}

impl UnionPwAffRef {
  #[inline(always)]
  pub fn copy(self) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_union_pw_aff_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_union_pw_aff_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_uint {
    unsafe {
      let ret = isl_union_pw_aff_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: Option<CStr>) -> c_int {
    unsafe {
      let ret = isl_union_pw_aff_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_pw_aff(self) -> c_int {
    unsafe {
      let ret = isl_union_pw_aff_n_pw_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_pw_aff<F1: FnMut(PwAff) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(PwAff) -> Option<()>>(pa: PwAff, user: *mut c_void) -> Stat { (*(user as *mut F))(pa.to()).to() }
    unsafe {
      let ret = isl_union_pw_aff_foreach_pw_aff(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn extract_pw_aff(self, space: Space) -> Option<PwAff> {
    unsafe {
      let ret = isl_union_pw_aff_extract_pw_aff(self.to(), space.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_nan(self) -> Option<bool> {
    unsafe {
      let ret = isl_union_pw_aff_involves_nan(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, upa2: UnionPwAffRef) -> Option<bool> {
    unsafe {
      let ret = isl_union_pw_aff_plain_is_equal(self.to(), upa2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_union_pw_aff_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_union_pw_aff_dump(self.to());
      (ret).to()
    }
  }
}

impl UnionPwMultiAff {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_union_pw_multi_aff_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_pw_multi_aff(self, pma: PwMultiAff) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_add_pw_multi_aff(self.to(), pma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_name(self, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_set_dim_name(self.to(), type_.to(), pos.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_drop_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coalesce(self) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_coalesce(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_params(self, context: Set) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_gist_params(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, context: UnionSet) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_gist(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pullback_union_pw_multi_aff(self, upma2: UnionPwMultiAff) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_pullback_union_pw_multi_aff(self.to(), upma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_align_params(self.to(), model.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_pw_multi_aff_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg(self) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, upma2: UnionPwMultiAff) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_add(self.to(), upma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_add(self, upma2: UnionPwMultiAff) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_union_add(self.to(), upma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sub(self, upma2: UnionPwMultiAff) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_sub(self.to(), upma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_val(self, val: Val) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_scale_val(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_val(self, val: Val) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_scale_down_val(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_multi_val(self, mv: MultiVal) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_scale_multi_val(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flat_range_product(self, upma2: UnionPwMultiAff) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_flat_range_product(self.to(), upma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_params(self, set: Set) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_intersect_params(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain(self, uset: UnionSet) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_intersect_domain(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract_domain(self, uset: UnionSet) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_subtract_domain(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_map_from_union_pw_multi_aff(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_from_union_pw_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_union_pw_aff_from_union_pw_multi_aff(self) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_from_union_pw_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn list_from_union_pw_multi_aff(self) -> Option<UnionPwMultiAffList> {
    unsafe {
      let ret = isl_union_pw_multi_aff_list_from_union_pw_multi_aff(self.to());
      (ret).to()
    }
  }
}

impl UnionPwMultiAffList {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_union_pw_multi_aff_list_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, el: UnionPwMultiAff) -> Option<UnionPwMultiAffList> {
    unsafe {
      let ret = isl_union_pw_multi_aff_list_add(self.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert(self, pos: c_uint, el: UnionPwMultiAff) -> Option<UnionPwMultiAffList> {
    unsafe {
      let ret = isl_union_pw_multi_aff_list_insert(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, first: c_uint, n: c_uint) -> Option<UnionPwMultiAffList> {
    unsafe {
      let ret = isl_union_pw_multi_aff_list_drop(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, list2: UnionPwMultiAffList) -> Option<UnionPwMultiAffList> {
    unsafe {
      let ret = isl_union_pw_multi_aff_list_concat(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_union_pw_multi_aff(self, index: c_int, el: UnionPwMultiAff) -> Option<UnionPwMultiAffList> {
    unsafe {
      let ret = isl_union_pw_multi_aff_list_set_union_pw_multi_aff(self.to(), index.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map<F1: FnMut(UnionPwMultiAff) -> Option<UnionPwMultiAff>>(self, fn_: &mut F1) -> Option<UnionPwMultiAffList> {
    unsafe extern "C" fn fn1<F: FnMut(UnionPwMultiAff) -> Option<UnionPwMultiAff>>(el: UnionPwMultiAff, user: *mut c_void) -> Option<UnionPwMultiAff> { (*(user as *mut F))(el.to()).to() }
    unsafe {
      let ret = isl_union_pw_multi_aff_list_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort<F1: FnMut(UnionPwMultiAffRef, UnionPwMultiAffRef) -> c_int>(self, cmp: &mut F1) -> Option<UnionPwMultiAffList> {
    unsafe extern "C" fn fn1<F: FnMut(UnionPwMultiAffRef, UnionPwMultiAffRef) -> c_int>(a: UnionPwMultiAffRef, b: UnionPwMultiAffRef, user: *mut c_void) -> c_int { (*(user as *mut F))(a.to(), b.to()).to() }
    unsafe {
      let ret = isl_union_pw_multi_aff_list_sort(self.to(), fn1::<F1>, cmp as *mut _ as _);
      (ret).to()
    }
  }
}

impl UnionPwMultiAffListRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_union_pw_multi_aff_list_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<UnionPwMultiAffList> {
    unsafe {
      let ret = isl_union_pw_multi_aff_list_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_union_pw_multi_aff(self) -> c_int {
    unsafe {
      let ret = isl_union_pw_multi_aff_list_n_union_pw_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_union_pw_multi_aff(self, index: c_int) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_list_get_union_pw_multi_aff(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(UnionPwMultiAff) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(UnionPwMultiAff) -> Option<()>>(el: UnionPwMultiAff, user: *mut c_void) -> Stat { (*(user as *mut F))(el.to()).to() }
    unsafe {
      let ret = isl_union_pw_multi_aff_list_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_scc<F1: FnMut(UnionPwMultiAffRef, UnionPwMultiAffRef) -> Option<bool>, F2: FnMut(UnionPwMultiAffList) -> Option<()>>(self, follows: &mut F1, fn_: &mut F2) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(UnionPwMultiAffRef, UnionPwMultiAffRef) -> Option<bool>>(a: UnionPwMultiAffRef, b: UnionPwMultiAffRef, user: *mut c_void) -> Bool { (*(user as *mut F))(a.to(), b.to()).to() }
    unsafe extern "C" fn fn2<F: FnMut(UnionPwMultiAffList) -> Option<()>>(scc: UnionPwMultiAffList, user: *mut c_void) -> Stat { (*(user as *mut F))(scc.to()).to() }
    unsafe {
      let ret = isl_union_pw_multi_aff_list_foreach_scc(self.to(), fn1::<F1>, follows as *mut _ as _, fn2::<F2>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_union_pw_multi_aff_list_dump(self.to());
      (ret).to()
    }
  }
}

impl UnionPwMultiAffRef {
  #[inline(always)]
  pub fn copy(self) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_union_pw_aff(self, pos: c_int) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_get_union_pw_aff(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_union_pw_multi_aff_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_union_pw_multi_aff_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_uint {
    unsafe {
      let ret = isl_union_pw_multi_aff_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: Option<CStr>) -> c_int {
    unsafe {
      let ret = isl_union_pw_multi_aff_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_pw_multi_aff(self) -> c_int {
    unsafe {
      let ret = isl_union_pw_multi_aff_n_pw_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_pw_multi_aff<F1: FnMut(PwMultiAff) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(PwMultiAff) -> Option<()>>(pma: PwMultiAff, user: *mut c_void) -> Stat { (*(user as *mut F))(pma.to()).to() }
    unsafe {
      let ret = isl_union_pw_multi_aff_foreach_pw_multi_aff(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn extract_pw_multi_aff(self, space: Space) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_extract_pw_multi_aff(self.to(), space.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_nan(self) -> Option<bool> {
    unsafe {
      let ret = isl_union_pw_multi_aff_involves_nan(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, upma2: UnionPwMultiAffRef) -> Option<bool> {
    unsafe {
      let ret = isl_union_pw_multi_aff_plain_is_equal(self.to(), upma2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_union_pw_multi_aff_dump(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_union_pw_multi_aff_to_str(self.to());
      (ret).to()
    }
  }
}

impl UnionSet {
  #[inline(always)]
  pub fn union_pw_multi_aff_from_domain(self) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_from_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_multi_aff_multi_val_on_domain(self, mv: MultiVal) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_multi_val_on_domain(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_aff_param_on_domain_id(self, id: Id) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_param_on_domain_id(self.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn identity_union_pw_multi_aff(self) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_set_identity_union_pw_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_multi_aff_from_union_set(self) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_pw_multi_aff_from_union_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_aff_val_on_domain(self, v: Val) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_val_on_domain(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_aff_aff_on_domain(self, aff: Aff) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_aff_on_domain(self.to(), aff.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_aff_pw_aff_on_domain(self, pa: PwAff) -> Option<UnionPwAff> {
    unsafe {
      let ret = isl_union_pw_aff_pw_aff_on_domain(self.to(), pa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_union_pw_aff_multi_val_on_domain(self, mv: MultiVal) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_multi_val_on_domain(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_union_pw_aff_multi_aff_on_domain(self, ma: MultiAff) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_multi_aff_on_domain(self.to(), ma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_union_pw_aff_pw_multi_aff_on_domain(self, pma: PwMultiAff) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_multi_union_pw_aff_pw_multi_aff_on_domain(self.to(), pma.to());
      (ret).to()
    }
  }
}

impl Drop for Aff {
  fn drop(&mut self) { Aff(self.0).free() }
}

impl fmt::Display for AffRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for Aff {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl Drop for MultiAff {
  fn drop(&mut self) { MultiAff(self.0).free() }
}

impl fmt::Display for MultiAffRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for MultiAff {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl Drop for MultiPwAff {
  fn drop(&mut self) { MultiPwAff(self.0).free() }
}

impl fmt::Display for MultiPwAffRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for MultiPwAff {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl Drop for MultiUnionPwAff {
  fn drop(&mut self) { MultiUnionPwAff(self.0).free() }
}

impl fmt::Display for MultiUnionPwAffRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for MultiUnionPwAff {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl Drop for PwAff {
  fn drop(&mut self) { PwAff(self.0).free() }
}

impl fmt::Display for PwAffRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for PwAff {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl Drop for PwMultiAff {
  fn drop(&mut self) { PwMultiAff(self.0).free() }
}

impl fmt::Display for PwMultiAffRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for PwMultiAff {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl Drop for UnionPwAff {
  fn drop(&mut self) { UnionPwAff(self.0).free() }
}

impl Drop for UnionPwAffList {
  fn drop(&mut self) { UnionPwAffList(self.0).free() }
}

impl fmt::Display for UnionPwAffRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for UnionPwAff {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl Drop for UnionPwMultiAff {
  fn drop(&mut self) { UnionPwMultiAff(self.0).free() }
}

impl Drop for UnionPwMultiAffList {
  fn drop(&mut self) { UnionPwMultiAffList(self.0).free() }
}

impl fmt::Display for UnionPwMultiAffRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for UnionPwMultiAff {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

