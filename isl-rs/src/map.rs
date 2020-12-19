use crate::*;

extern "C" {
  pub fn isl_basic_map_n_in(bmap: BasicMapRef) -> c_uint;
  pub fn isl_basic_map_n_out(bmap: BasicMapRef) -> c_uint;
  pub fn isl_basic_map_n_param(bmap: BasicMapRef) -> c_uint;
  pub fn isl_basic_map_n_div(bmap: BasicMapRef) -> c_uint;
  pub fn isl_basic_map_total_dim(bmap: BasicMapRef) -> c_uint;
  pub fn isl_basic_map_dim(bmap: BasicMapRef, type_: DimType) -> c_uint;
  pub fn isl_map_n_in(map: MapRef) -> c_uint;
  pub fn isl_map_n_out(map: MapRef) -> c_uint;
  pub fn isl_map_n_param(map: MapRef) -> c_uint;
  pub fn isl_map_dim(map: MapRef, type_: DimType) -> c_uint;
  pub fn isl_basic_map_get_ctx(bmap: BasicMapRef) -> Option<CtxRef>;
  pub fn isl_map_get_ctx(map: MapRef) -> Option<CtxRef>;
  pub fn isl_basic_map_get_space(bmap: BasicMapRef) -> Option<Space>;
  pub fn isl_map_get_space(map: MapRef) -> Option<Space>;
  pub fn isl_basic_map_get_div(bmap: BasicMapRef, pos: c_int) -> Option<Aff>;
  pub fn isl_basic_map_get_local_space(bmap: BasicMapRef) -> Option<LocalSpace>;
  pub fn isl_basic_map_set_tuple_name(bmap: BasicMap, type_: DimType, s: CStr) -> Option<BasicMap>;
  pub fn isl_basic_map_get_tuple_name(bmap: BasicMapRef, type_: DimType) -> Option<CStr>;
  pub fn isl_map_has_tuple_name(map: MapRef, type_: DimType) -> Bool;
  pub fn isl_map_get_tuple_name(map: MapRef, type_: DimType) -> Option<CStr>;
  pub fn isl_map_set_tuple_name(map: Map, type_: DimType, s: CStr) -> Option<Map>;
  pub fn isl_basic_map_get_dim_name(bmap: BasicMapRef, type_: DimType, pos: c_uint) -> Option<CStr>;
  pub fn isl_map_has_dim_name(map: MapRef, type_: DimType, pos: c_uint) -> Bool;
  pub fn isl_map_get_dim_name(map: MapRef, type_: DimType, pos: c_uint) -> Option<CStr>;
  pub fn isl_basic_map_set_dim_name(bmap: BasicMap, type_: DimType, pos: c_uint, s: CStr) -> Option<BasicMap>;
  pub fn isl_map_set_dim_name(map: Map, type_: DimType, pos: c_uint, s: CStr) -> Option<Map>;
  pub fn isl_basic_map_set_tuple_id(bmap: BasicMap, type_: DimType, id: Id) -> Option<BasicMap>;
  pub fn isl_map_set_dim_id(map: Map, type_: DimType, pos: c_uint, id: Id) -> Option<Map>;
  pub fn isl_basic_map_has_dim_id(bmap: BasicMapRef, type_: DimType, pos: c_uint) -> Bool;
  pub fn isl_map_has_dim_id(map: MapRef, type_: DimType, pos: c_uint) -> Bool;
  pub fn isl_map_get_dim_id(map: MapRef, type_: DimType, pos: c_uint) -> Option<Id>;
  pub fn isl_map_set_tuple_id(map: Map, type_: DimType, id: Id) -> Option<Map>;
  pub fn isl_map_reset_tuple_id(map: Map, type_: DimType) -> Option<Map>;
  pub fn isl_map_has_tuple_id(map: MapRef, type_: DimType) -> Bool;
  pub fn isl_map_get_tuple_id(map: MapRef, type_: DimType) -> Option<Id>;
  pub fn isl_map_reset_user(map: Map) -> Option<Map>;
  pub fn isl_basic_map_find_dim_by_name(bmap: BasicMapRef, type_: DimType, name: CStr) -> c_int;
  pub fn isl_map_find_dim_by_id(map: MapRef, type_: DimType, id: IdRef) -> c_int;
  pub fn isl_map_find_dim_by_name(map: MapRef, type_: DimType, name: CStr) -> c_int;
  pub fn isl_basic_map_is_rational(bmap: BasicMapRef) -> Bool;
  pub fn isl_basic_map_identity(dim: Space) -> Option<BasicMap>;
  pub fn isl_basic_map_free(bmap: BasicMap) -> *mut c_void;
  pub fn isl_basic_map_copy(bmap: BasicMapRef) -> Option<BasicMap>;
  pub fn isl_basic_map_equal(dim: Space, n_equal: c_uint) -> Option<BasicMap>;
  pub fn isl_basic_map_less_at(dim: Space, pos: c_uint) -> Option<BasicMap>;
  pub fn isl_basic_map_more_at(dim: Space, pos: c_uint) -> Option<BasicMap>;
  pub fn isl_basic_map_empty(space: Space) -> Option<BasicMap>;
  pub fn isl_basic_map_universe(space: Space) -> Option<BasicMap>;
  pub fn isl_basic_map_nat_universe(dim: Space) -> Option<BasicMap>;
  pub fn isl_basic_map_remove_redundancies(bmap: BasicMap) -> Option<BasicMap>;
  pub fn isl_map_remove_redundancies(map: Map) -> Option<Map>;
  pub fn isl_map_simple_hull(map: Map) -> Option<BasicMap>;
  pub fn isl_map_unshifted_simple_hull(map: Map) -> Option<BasicMap>;
  pub fn isl_map_plain_unshifted_simple_hull(map: Map) -> Option<BasicMap>;
  pub fn isl_map_unshifted_simple_hull_from_map_list(map: Map, list: MapList) -> Option<BasicMap>;
  pub fn isl_basic_map_intersect_domain(bmap: BasicMap, bset: BasicSet) -> Option<BasicMap>;
  pub fn isl_basic_map_intersect_range(bmap: BasicMap, bset: BasicSet) -> Option<BasicMap>;
  pub fn isl_basic_map_intersect(bmap1: BasicMap, bmap2: BasicMap) -> Option<BasicMap>;
  pub fn isl_basic_map_list_intersect(list: BasicMapList) -> Option<BasicMap>;
  pub fn isl_basic_map_union(bmap1: BasicMap, bmap2: BasicMap) -> Option<Map>;
  pub fn isl_basic_map_apply_domain(bmap1: BasicMap, bmap2: BasicMap) -> Option<BasicMap>;
  pub fn isl_basic_map_apply_range(bmap1: BasicMap, bmap2: BasicMap) -> Option<BasicMap>;
  pub fn isl_basic_map_affine_hull(bmap: BasicMap) -> Option<BasicMap>;
  pub fn isl_basic_map_preimage_domain_multi_aff(bmap: BasicMap, ma: MultiAff) -> Option<BasicMap>;
  pub fn isl_basic_map_preimage_range_multi_aff(bmap: BasicMap, ma: MultiAff) -> Option<BasicMap>;
  pub fn isl_basic_map_reverse(bmap: BasicMap) -> Option<BasicMap>;
  pub fn isl_basic_map_domain(bmap: BasicMap) -> Option<BasicSet>;
  pub fn isl_basic_map_range(bmap: BasicMap) -> Option<BasicSet>;
  pub fn isl_basic_map_domain_map(bmap: BasicMap) -> Option<BasicMap>;
  pub fn isl_basic_map_range_map(bmap: BasicMap) -> Option<BasicMap>;
  pub fn isl_basic_map_remove_dims(bmap: BasicMap, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicMap>;
  pub fn isl_basic_map_eliminate(bmap: BasicMap, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicMap>;
  pub fn isl_basic_map_sample(bmap: BasicMap) -> Option<BasicMap>;
  pub fn isl_basic_map_detect_equalities(bmap: BasicMap) -> Option<BasicMap>;
  pub fn isl_basic_map_read_from_file(ctx: CtxRef, input: *mut FILE) -> Option<BasicMap>;
  pub fn isl_basic_map_read_from_str(ctx: CtxRef, str: CStr) -> Option<BasicMap>;
  pub fn isl_map_read_from_file(ctx: CtxRef, input: *mut FILE) -> Option<Map>;
  pub fn isl_map_read_from_str(ctx: CtxRef, str: CStr) -> Option<Map>;
  pub fn isl_basic_map_dump(bmap: BasicMapRef) -> ();
  pub fn isl_map_dump(map: MapRef) -> ();
  pub fn isl_basic_map_to_str(bmap: BasicMapRef) -> Option<CString>;
  pub fn isl_printer_print_basic_map(printer: Printer, bmap: BasicMapRef) -> Option<Printer>;
  pub fn isl_map_to_str(map: MapRef) -> Option<CString>;
  pub fn isl_printer_print_map(printer: Printer, map: MapRef) -> Option<Printer>;
  pub fn isl_basic_map_fix_si(bmap: BasicMap, type_: DimType, pos: c_uint, value: c_int) -> Option<BasicMap>;
  pub fn isl_basic_map_fix_val(bmap: BasicMap, type_: DimType, pos: c_uint, v: Val) -> Option<BasicMap>;
  pub fn isl_basic_map_lower_bound_si(bmap: BasicMap, type_: DimType, pos: c_uint, value: c_int) -> Option<BasicMap>;
  pub fn isl_basic_map_upper_bound_si(bmap: BasicMap, type_: DimType, pos: c_uint, value: c_int) -> Option<BasicMap>;
  pub fn isl_basic_map_sum(bmap1: BasicMap, bmap2: BasicMap) -> Option<BasicMap>;
  pub fn isl_basic_map_neg(bmap: BasicMap) -> Option<BasicMap>;
  pub fn isl_map_sum(map1: Map, map2: Map) -> Option<Map>;
  pub fn isl_map_neg(map: Map) -> Option<Map>;
  pub fn isl_map_floordiv_val(map: Map, d: Val) -> Option<Map>;
  pub fn isl_basic_map_is_equal(bmap1: BasicMapRef, bmap2: BasicMapRef) -> Bool;
  pub fn isl_basic_map_is_disjoint(bmap1: BasicMapRef, bmap2: BasicMapRef) -> Bool;
  pub fn isl_basic_map_partial_lexmax(bmap: BasicMap, dom: BasicSet, empty: *mut Set) -> Option<Map>;
  pub fn isl_basic_map_partial_lexmin(bmap: BasicMap, dom: BasicSet, empty: *mut Set) -> Option<Map>;
  pub fn isl_map_partial_lexmax(map: Map, dom: Set, empty: *mut Set) -> Option<Map>;
  pub fn isl_map_partial_lexmin(map: Map, dom: Set, empty: *mut Set) -> Option<Map>;
  pub fn isl_basic_map_lexmin(bmap: BasicMap) -> Option<Map>;
  pub fn isl_basic_map_lexmax(bmap: BasicMap) -> Option<Map>;
  pub fn isl_map_lexmin(map: Map) -> Option<Map>;
  pub fn isl_map_lexmax(map: Map) -> Option<Map>;
  pub fn isl_basic_map_partial_lexmin_pw_multi_aff(bmap: BasicMap, dom: BasicSet, empty: *mut Set) -> Option<PwMultiAff>;
  pub fn isl_basic_map_partial_lexmax_pw_multi_aff(bmap: BasicMap, dom: BasicSet, empty: *mut Set) -> Option<PwMultiAff>;
  pub fn isl_basic_map_lexmin_pw_multi_aff(bmap: BasicMap) -> Option<PwMultiAff>;
  pub fn isl_map_lexmin_pw_multi_aff(map: Map) -> Option<PwMultiAff>;
  pub fn isl_map_lexmax_pw_multi_aff(map: Map) -> Option<PwMultiAff>;
  pub fn isl_basic_map_print_internal(bmap: BasicMapRef, out: *mut FILE, indent: c_int) -> ();
  pub fn isl_basic_map_plain_get_val_if_fixed(bmap: BasicMapRef, type_: DimType, pos: c_uint) -> Option<Val>;
  pub fn isl_basic_map_image_is_bounded(bmap: BasicMapRef) -> Bool;
  pub fn isl_basic_map_plain_is_universe(bmap: BasicMapRef) -> Bool;
  pub fn isl_basic_map_is_universe(bmap: BasicMapRef) -> Bool;
  pub fn isl_basic_map_plain_is_empty(bmap: BasicMapRef) -> Bool;
  pub fn isl_basic_map_is_empty(bmap: BasicMapRef) -> Bool;
  pub fn isl_basic_map_is_subset(bmap1: BasicMapRef, bmap2: BasicMapRef) -> Bool;
  pub fn isl_basic_map_is_strict_subset(bmap1: BasicMapRef, bmap2: BasicMapRef) -> Bool;
  pub fn isl_map_universe(space: Space) -> Option<Map>;
  pub fn isl_map_nat_universe(dim: Space) -> Option<Map>;
  pub fn isl_map_empty(space: Space) -> Option<Map>;
  pub fn isl_map_identity(dim: Space) -> Option<Map>;
  pub fn isl_map_lex_lt_first(dim: Space, n: c_uint) -> Option<Map>;
  pub fn isl_map_lex_le_first(dim: Space, n: c_uint) -> Option<Map>;
  pub fn isl_map_lex_lt(set_dim: Space) -> Option<Map>;
  pub fn isl_map_lex_le(set_dim: Space) -> Option<Map>;
  pub fn isl_map_lex_gt_first(dim: Space, n: c_uint) -> Option<Map>;
  pub fn isl_map_lex_ge_first(dim: Space, n: c_uint) -> Option<Map>;
  pub fn isl_map_lex_gt(set_dim: Space) -> Option<Map>;
  pub fn isl_map_lex_ge(set_dim: Space) -> Option<Map>;
  pub fn isl_map_free(map: Map) -> *mut c_void;
  pub fn isl_map_copy(map: MapRef) -> Option<Map>;
  pub fn isl_map_reverse(map: Map) -> Option<Map>;
  pub fn isl_map_union(map1: Map, map2: Map) -> Option<Map>;
  pub fn isl_map_union_disjoint(map1: Map, map2: Map) -> Option<Map>;
  pub fn isl_map_intersect_domain(map: Map, set: Set) -> Option<Map>;
  pub fn isl_map_intersect_range(map: Map, set: Set) -> Option<Map>;
  pub fn isl_map_intersect_domain_factor_range(map: Map, factor: Map) -> Option<Map>;
  pub fn isl_map_intersect_range_factor_range(map: Map, factor: Map) -> Option<Map>;
  pub fn isl_map_apply_domain(map1: Map, map2: Map) -> Option<Map>;
  pub fn isl_map_apply_range(map1: Map, map2: Map) -> Option<Map>;
  pub fn isl_map_preimage_domain_multi_aff(map: Map, ma: MultiAff) -> Option<Map>;
  pub fn isl_map_preimage_range_multi_aff(map: Map, ma: MultiAff) -> Option<Map>;
  pub fn isl_map_preimage_domain_pw_multi_aff(map: Map, pma: PwMultiAff) -> Option<Map>;
  pub fn isl_map_preimage_range_pw_multi_aff(map: Map, pma: PwMultiAff) -> Option<Map>;
  pub fn isl_map_preimage_domain_multi_pw_aff(map: Map, mpa: MultiPwAff) -> Option<Map>;
  pub fn isl_basic_map_product(bmap1: BasicMap, bmap2: BasicMap) -> Option<BasicMap>;
  pub fn isl_map_product(map1: Map, map2: Map) -> Option<Map>;
  pub fn isl_basic_map_domain_product(bmap1: BasicMap, bmap2: BasicMap) -> Option<BasicMap>;
  pub fn isl_basic_map_range_product(bmap1: BasicMap, bmap2: BasicMap) -> Option<BasicMap>;
  pub fn isl_map_domain_product(map1: Map, map2: Map) -> Option<Map>;
  pub fn isl_map_range_product(map1: Map, map2: Map) -> Option<Map>;
  pub fn isl_basic_map_flat_product(bmap1: BasicMap, bmap2: BasicMap) -> Option<BasicMap>;
  pub fn isl_map_flat_product(map1: Map, map2: Map) -> Option<Map>;
  pub fn isl_basic_map_flat_range_product(bmap1: BasicMap, bmap2: BasicMap) -> Option<BasicMap>;
  pub fn isl_map_flat_domain_product(map1: Map, map2: Map) -> Option<Map>;
  pub fn isl_map_flat_range_product(map1: Map, map2: Map) -> Option<Map>;
  pub fn isl_map_domain_is_wrapping(map: MapRef) -> Bool;
  pub fn isl_map_range_is_wrapping(map: MapRef) -> Bool;
  pub fn isl_map_is_product(map: MapRef) -> Bool;
  pub fn isl_map_factor_domain(map: Map) -> Option<Map>;
  pub fn isl_map_factor_range(map: Map) -> Option<Map>;
  pub fn isl_map_domain_factor_domain(map: Map) -> Option<Map>;
  pub fn isl_map_domain_factor_range(map: Map) -> Option<Map>;
  pub fn isl_map_range_factor_domain(map: Map) -> Option<Map>;
  pub fn isl_map_range_factor_range(map: Map) -> Option<Map>;
  pub fn isl_map_intersect(map1: Map, map2: Map) -> Option<Map>;
  pub fn isl_map_intersect_params(map: Map, params: Set) -> Option<Map>;
  pub fn isl_map_subtract(map1: Map, map2: Map) -> Option<Map>;
  pub fn isl_map_subtract_domain(map: Map, dom: Set) -> Option<Map>;
  pub fn isl_map_subtract_range(map: Map, dom: Set) -> Option<Map>;
  pub fn isl_map_complement(map: Map) -> Option<Map>;
  pub fn isl_map_fix_input_si(map: Map, input: c_uint, value: c_int) -> Option<Map>;
  pub fn isl_map_fix_si(map: Map, type_: DimType, pos: c_uint, value: c_int) -> Option<Map>;
  pub fn isl_map_fix_val(map: Map, type_: DimType, pos: c_uint, v: Val) -> Option<Map>;
  pub fn isl_map_lower_bound_si(map: Map, type_: DimType, pos: c_uint, value: c_int) -> Option<Map>;
  pub fn isl_map_upper_bound_si(map: Map, type_: DimType, pos: c_uint, value: c_int) -> Option<Map>;
  pub fn isl_basic_map_deltas(bmap: BasicMap) -> Option<BasicSet>;
  pub fn isl_map_deltas(map: Map) -> Option<Set>;
  pub fn isl_basic_map_deltas_map(bmap: BasicMap) -> Option<BasicMap>;
  pub fn isl_map_deltas_map(map: Map) -> Option<Map>;
  pub fn isl_map_detect_equalities(map: Map) -> Option<Map>;
  pub fn isl_map_affine_hull(map: Map) -> Option<BasicMap>;
  pub fn isl_map_convex_hull(map: Map) -> Option<BasicMap>;
  pub fn isl_map_polyhedral_hull(map: Map) -> Option<BasicMap>;
  pub fn isl_basic_map_add_dims(bmap: BasicMap, type_: DimType, n: c_uint) -> Option<BasicMap>;
  pub fn isl_map_add_dims(map: Map, type_: DimType, n: c_uint) -> Option<Map>;
  pub fn isl_basic_map_insert_dims(bmap: BasicMap, type_: DimType, pos: c_uint, n: c_uint) -> Option<BasicMap>;
  pub fn isl_map_insert_dims(map: Map, type_: DimType, pos: c_uint, n: c_uint) -> Option<Map>;
  pub fn isl_basic_map_move_dims(bmap: BasicMap, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<BasicMap>;
  pub fn isl_map_move_dims(map: Map, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<Map>;
  pub fn isl_basic_map_project_out(bmap: BasicMap, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicMap>;
  pub fn isl_map_project_out(map: Map, type_: DimType, first: c_uint, n: c_uint) -> Option<Map>;
  pub fn isl_basic_map_remove_divs(bmap: BasicMap) -> Option<BasicMap>;
  pub fn isl_map_remove_unknown_divs(map: Map) -> Option<Map>;
  pub fn isl_map_remove_divs(map: Map) -> Option<Map>;
  pub fn isl_map_eliminate(map: Map, type_: DimType, first: c_uint, n: c_uint) -> Option<Map>;
  pub fn isl_map_remove_dims(map: Map, type_: DimType, first: c_uint, n: c_uint) -> Option<Map>;
  pub fn isl_basic_map_remove_divs_involving_dims(bmap: BasicMap, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicMap>;
  pub fn isl_map_remove_divs_involving_dims(map: Map, type_: DimType, first: c_uint, n: c_uint) -> Option<Map>;
  pub fn isl_map_remove_inputs(map: Map, first: c_uint, n: c_uint) -> Option<Map>;
  pub fn isl_basic_map_equate(bmap: BasicMap, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<BasicMap>;
  pub fn isl_basic_map_order_ge(bmap: BasicMap, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<BasicMap>;
  pub fn isl_map_order_ge(map: Map, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<Map>;
  pub fn isl_map_order_le(map: Map, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<Map>;
  pub fn isl_map_equate(map: Map, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<Map>;
  pub fn isl_map_oppose(map: Map, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<Map>;
  pub fn isl_map_order_lt(map: Map, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<Map>;
  pub fn isl_basic_map_order_gt(bmap: BasicMap, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<BasicMap>;
  pub fn isl_map_order_gt(map: Map, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<Map>;
  pub fn isl_set_identity(set: Set) -> Option<Map>;
  pub fn isl_basic_set_is_wrapping(bset: BasicSetRef) -> Bool;
  pub fn isl_set_is_wrapping(set: SetRef) -> Bool;
  pub fn isl_basic_map_wrap(bmap: BasicMap) -> Option<BasicSet>;
  pub fn isl_map_wrap(map: Map) -> Option<Set>;
  pub fn isl_basic_set_unwrap(bset: BasicSet) -> Option<BasicMap>;
  pub fn isl_set_unwrap(set: Set) -> Option<Map>;
  pub fn isl_basic_map_flatten(bmap: BasicMap) -> Option<BasicMap>;
  pub fn isl_map_flatten(map: Map) -> Option<Map>;
  pub fn isl_basic_map_flatten_domain(bmap: BasicMap) -> Option<BasicMap>;
  pub fn isl_basic_map_flatten_range(bmap: BasicMap) -> Option<BasicMap>;
  pub fn isl_map_flatten_domain(map: Map) -> Option<Map>;
  pub fn isl_map_flatten_range(map: Map) -> Option<Map>;
  pub fn isl_basic_set_flatten(bset: BasicSet) -> Option<BasicSet>;
  pub fn isl_set_flatten(set: Set) -> Option<Set>;
  pub fn isl_set_flatten_map(set: Set) -> Option<Map>;
  pub fn isl_map_params(map: Map) -> Option<Set>;
  pub fn isl_map_domain(bmap: Map) -> Option<Set>;
  pub fn isl_map_range(map: Map) -> Option<Set>;
  pub fn isl_map_domain_map(map: Map) -> Option<Map>;
  pub fn isl_map_range_map(map: Map) -> Option<Map>;
  pub fn isl_set_wrapped_domain_map(set: Set) -> Option<Map>;
  pub fn isl_map_from_basic_map(bmap: BasicMap) -> Option<Map>;
  pub fn isl_map_from_domain(set: Set) -> Option<Map>;
  pub fn isl_basic_map_from_domain(bset: BasicSet) -> Option<BasicMap>;
  pub fn isl_basic_map_from_range(bset: BasicSet) -> Option<BasicMap>;
  pub fn isl_map_from_range(set: Set) -> Option<Map>;
  pub fn isl_basic_map_from_domain_and_range(domain: BasicSet, range: BasicSet) -> Option<BasicMap>;
  pub fn isl_map_from_domain_and_range(domain: Set, range: Set) -> Option<Map>;
  pub fn isl_map_sample(map: Map) -> Option<BasicMap>;
  pub fn isl_map_plain_is_empty(map: MapRef) -> Bool;
  pub fn isl_map_plain_is_universe(map: MapRef) -> Bool;
  pub fn isl_map_is_empty(map: MapRef) -> Bool;
  pub fn isl_map_is_subset(map1: MapRef, map2: MapRef) -> Bool;
  pub fn isl_map_is_strict_subset(map1: MapRef, map2: MapRef) -> Bool;
  pub fn isl_map_is_equal(map1: MapRef, map2: MapRef) -> Bool;
  pub fn isl_map_is_disjoint(map1: MapRef, map2: MapRef) -> Bool;
  pub fn isl_basic_map_is_single_valued(bmap: BasicMapRef) -> Bool;
  pub fn isl_map_plain_is_single_valued(map: MapRef) -> Bool;
  pub fn isl_map_is_single_valued(map: MapRef) -> Bool;
  pub fn isl_map_plain_is_injective(map: MapRef) -> Bool;
  pub fn isl_map_is_injective(map: MapRef) -> Bool;
  pub fn isl_map_is_bijective(map: MapRef) -> Bool;
  pub fn isl_map_is_identity(map: MapRef) -> Bool;
  pub fn isl_map_is_translation(map: MapRef) -> c_int;
  pub fn isl_map_has_equal_space(map1: MapRef, map2: MapRef) -> Bool;
  pub fn isl_basic_map_can_zip(bmap: BasicMapRef) -> Bool;
  pub fn isl_map_can_zip(map: MapRef) -> Bool;
  pub fn isl_basic_map_zip(bmap: BasicMap) -> Option<BasicMap>;
  pub fn isl_map_zip(map: Map) -> Option<Map>;
  pub fn isl_basic_map_can_curry(bmap: BasicMapRef) -> Bool;
  pub fn isl_map_can_curry(map: MapRef) -> Bool;
  pub fn isl_basic_map_curry(bmap: BasicMap) -> Option<BasicMap>;
  pub fn isl_map_curry(map: Map) -> Option<Map>;
  pub fn isl_map_can_range_curry(map: MapRef) -> Bool;
  pub fn isl_map_range_curry(map: Map) -> Option<Map>;
  pub fn isl_basic_map_can_uncurry(bmap: BasicMapRef) -> Bool;
  pub fn isl_map_can_uncurry(map: MapRef) -> Bool;
  pub fn isl_basic_map_uncurry(bmap: BasicMap) -> Option<BasicMap>;
  pub fn isl_map_uncurry(map: Map) -> Option<Map>;
  pub fn isl_map_make_disjoint(map: Map) -> Option<Map>;
  pub fn isl_basic_map_compute_divs(bmap: BasicMap) -> Option<Map>;
  pub fn isl_map_compute_divs(map: Map) -> Option<Map>;
  pub fn isl_map_align_divs(map: Map) -> Option<Map>;
  pub fn isl_basic_map_drop_constraints_involving_dims(bmap: BasicMap, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicMap>;
  pub fn isl_basic_map_drop_constraints_not_involving_dims(bmap: BasicMap, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicMap>;
  pub fn isl_map_drop_constraints_involving_dims(map: Map, type_: DimType, first: c_uint, n: c_uint) -> Option<Map>;
  pub fn isl_map_drop_constraints_not_involving_dims(map: Map, type_: DimType, first: c_uint, n: c_uint) -> Option<Map>;
  pub fn isl_basic_map_involves_dims(bmap: BasicMapRef, type_: DimType, first: c_uint, n: c_uint) -> Bool;
  pub fn isl_map_involves_dims(map: MapRef, type_: DimType, first: c_uint, n: c_uint) -> Bool;
  pub fn isl_map_print_internal(map: MapRef, out: *mut FILE, indent: c_int) -> ();
  pub fn isl_map_plain_get_val_if_fixed(map: MapRef, type_: DimType, pos: c_uint) -> Option<Val>;
  pub fn isl_basic_map_gist_domain(bmap: BasicMap, context: BasicSet) -> Option<BasicMap>;
  pub fn isl_basic_map_gist(bmap: BasicMap, context: BasicMap) -> Option<BasicMap>;
  pub fn isl_map_gist(map: Map, context: Map) -> Option<Map>;
  pub fn isl_map_gist_domain(map: Map, context: Set) -> Option<Map>;
  pub fn isl_map_gist_range(map: Map, context: Set) -> Option<Map>;
  pub fn isl_map_gist_params(map: Map, context: Set) -> Option<Map>;
  pub fn isl_map_gist_basic_map(map: Map, context: BasicMap) -> Option<Map>;
  pub fn isl_map_coalesce(map: Map) -> Option<Map>;
  pub fn isl_map_plain_is_equal(map1: MapRef, map2: MapRef) -> Bool;
  pub fn isl_map_get_hash(map: MapRef) -> c_uint;
  pub fn isl_map_n_basic_map(map: MapRef) -> c_int;
  pub fn isl_map_foreach_basic_map(map: MapRef, fn_: unsafe extern "C" fn(bmap: BasicMap, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_map_fixed_power_val(map: Map, exp: Val) -> Option<Map>;
  pub fn isl_map_power(map: Map, exact: *mut c_int) -> Option<Map>;
  pub fn isl_map_reaching_path_lengths(map: Map, exact: *mut c_int) -> Option<Map>;
  pub fn isl_map_transitive_closure(map: Map, exact: *mut c_int) -> Option<Map>;
  pub fn isl_map_lex_le_map(map1: Map, map2: Map) -> Option<Map>;
  pub fn isl_map_lex_lt_map(map1: Map, map2: Map) -> Option<Map>;
  pub fn isl_map_lex_ge_map(map1: Map, map2: Map) -> Option<Map>;
  pub fn isl_map_lex_gt_map(map1: Map, map2: Map) -> Option<Map>;
  pub fn isl_basic_map_align_params(bmap: BasicMap, model: Space) -> Option<BasicMap>;
  pub fn isl_map_align_params(map: Map, model: Space) -> Option<Map>;
  pub fn isl_basic_map_equalities_matrix(bmap: BasicMapRef, c1: DimType, c2: DimType, c3: DimType, c4: DimType, c5: DimType) -> Option<Mat>;
  pub fn isl_basic_map_inequalities_matrix(bmap: BasicMapRef, c1: DimType, c2: DimType, c3: DimType, c4: DimType, c5: DimType) -> Option<Mat>;
  pub fn isl_basic_map_from_constraint_matrices(dim: Space, eq: Mat, ineq: Mat, c1: DimType, c2: DimType, c3: DimType, c4: DimType, c5: DimType) -> Option<BasicMap>;
  pub fn isl_basic_map_from_aff(aff: Aff) -> Option<BasicMap>;
  pub fn isl_basic_map_from_multi_aff(maff: MultiAff) -> Option<BasicMap>;
  pub fn isl_basic_map_from_aff_list(domain_dim: Space, list: AffList) -> Option<BasicMap>;
  pub fn isl_map_from_aff(aff: Aff) -> Option<Map>;
  pub fn isl_map_from_multi_aff(maff: MultiAff) -> Option<Map>;
  pub fn isl_map_dim_min(map: Map, pos: c_int) -> Option<PwAff>;
  pub fn isl_map_dim_max(map: Map, pos: c_int) -> Option<PwAff>;
  pub fn isl_basic_map_list_get_ctx(list: BasicMapListRef) -> Option<CtxRef>;
  pub fn isl_basic_map_list_from_basic_map(el: BasicMap) -> Option<BasicMapList>;
  pub fn isl_basic_map_list_alloc(ctx: CtxRef, n: c_int) -> Option<BasicMapList>;
  pub fn isl_basic_map_list_copy(list: BasicMapListRef) -> Option<BasicMapList>;
  pub fn isl_basic_map_list_free(list: BasicMapList) -> *mut c_void;
  pub fn isl_basic_map_list_add(list: BasicMapList, el: BasicMap) -> Option<BasicMapList>;
  pub fn isl_basic_map_list_insert(list: BasicMapList, pos: c_uint, el: BasicMap) -> Option<BasicMapList>;
  pub fn isl_basic_map_list_drop(list: BasicMapList, first: c_uint, n: c_uint) -> Option<BasicMapList>;
  pub fn isl_basic_map_list_concat(list1: BasicMapList, list2: BasicMapList) -> Option<BasicMapList>;
  pub fn isl_basic_map_list_n_basic_map(list: BasicMapListRef) -> c_int;
  pub fn isl_basic_map_list_get_basic_map(list: BasicMapListRef, index: c_int) -> Option<BasicMap>;
  pub fn isl_basic_map_list_set_basic_map(list: BasicMapList, index: c_int, el: BasicMap) -> Option<BasicMapList>;
  pub fn isl_basic_map_list_foreach(list: BasicMapListRef, fn_: unsafe extern "C" fn(el: BasicMap, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_basic_map_list_map(list: BasicMapList, fn_: unsafe extern "C" fn(el: BasicMap, user: *mut c_void) -> Option<BasicMap>, user: *mut c_void) -> Option<BasicMapList>;
  pub fn isl_basic_map_list_sort(list: BasicMapList, cmp: unsafe extern "C" fn(a: BasicMapRef, b: BasicMapRef, user: *mut c_void) -> c_int, user: *mut c_void) -> Option<BasicMapList>;
  pub fn isl_basic_map_list_foreach_scc(list: BasicMapListRef, follows: unsafe extern "C" fn(a: BasicMapRef, b: BasicMapRef, user: *mut c_void) -> Bool, follows_user: *mut c_void, fn_: unsafe extern "C" fn(scc: BasicMapList, user: *mut c_void) -> Stat, fn_user: *mut c_void) -> Stat;
  pub fn isl_printer_print_basic_map_list(p: Printer, list: BasicMapListRef) -> Option<Printer>;
  pub fn isl_basic_map_list_dump(list: BasicMapListRef) -> ();
  pub fn isl_map_list_get_ctx(list: MapListRef) -> Option<CtxRef>;
  pub fn isl_map_list_from_map(el: Map) -> Option<MapList>;
  pub fn isl_map_list_alloc(ctx: CtxRef, n: c_int) -> Option<MapList>;
  pub fn isl_map_list_copy(list: MapListRef) -> Option<MapList>;
  pub fn isl_map_list_free(list: MapList) -> *mut c_void;
  pub fn isl_map_list_add(list: MapList, el: Map) -> Option<MapList>;
  pub fn isl_map_list_insert(list: MapList, pos: c_uint, el: Map) -> Option<MapList>;
  pub fn isl_map_list_drop(list: MapList, first: c_uint, n: c_uint) -> Option<MapList>;
  pub fn isl_map_list_concat(list1: MapList, list2: MapList) -> Option<MapList>;
  pub fn isl_map_list_n_map(list: MapListRef) -> c_int;
  pub fn isl_map_list_get_map(list: MapListRef, index: c_int) -> Option<Map>;
  pub fn isl_map_list_set_map(list: MapList, index: c_int, el: Map) -> Option<MapList>;
  pub fn isl_map_list_foreach(list: MapListRef, fn_: unsafe extern "C" fn(el: Map, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_map_list_map(list: MapList, fn_: unsafe extern "C" fn(el: Map, user: *mut c_void) -> Option<Map>, user: *mut c_void) -> Option<MapList>;
  pub fn isl_map_list_sort(list: MapList, cmp: unsafe extern "C" fn(a: MapRef, b: MapRef, user: *mut c_void) -> c_int, user: *mut c_void) -> Option<MapList>;
  pub fn isl_map_list_foreach_scc(list: MapListRef, follows: unsafe extern "C" fn(a: MapRef, b: MapRef, user: *mut c_void) -> Bool, follows_user: *mut c_void, fn_: unsafe extern "C" fn(scc: MapList, user: *mut c_void) -> Stat, fn_user: *mut c_void) -> Stat;
  pub fn isl_printer_print_map_list(p: Printer, list: MapListRef) -> Option<Printer>;
  pub fn isl_map_list_dump(list: MapListRef) -> ();
}

impl Aff {
  #[inline(always)]
  pub fn basic_map_from_aff(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_from_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_from_aff(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_from_aff(self.to());
      (ret).to()
    }
  }
}

impl BasicMap {
  #[inline(always)]
  pub fn set_tuple_name(self, type_: DimType, s: CStr) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_set_tuple_name(self.to(), type_.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_name(self, type_: DimType, pos: c_uint, s: CStr) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_set_dim_name(self.to(), type_.to(), pos.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_id(self, type_: DimType, id: Id) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_set_tuple_id(self.to(), type_.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_basic_map_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_redundancies(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_remove_redundancies(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain(self, bset: BasicSet) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_intersect_domain(self.to(), bset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_range(self, bset: BasicSet) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_intersect_range(self.to(), bset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect(self, bmap2: BasicMap) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_intersect(self.to(), bmap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union(self, bmap2: BasicMap) -> Option<Map> {
    unsafe {
      let ret = isl_basic_map_union(self.to(), bmap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn apply_domain(self, bmap2: BasicMap) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_apply_domain(self.to(), bmap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn apply_range(self, bmap2: BasicMap) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_apply_range(self.to(), bmap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn affine_hull(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_affine_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_domain_multi_aff(self, ma: MultiAff) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_preimage_domain_multi_aff(self.to(), ma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_range_multi_aff(self, ma: MultiAff) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_preimage_range_multi_aff(self.to(), ma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reverse(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_reverse(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_map_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_map_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain_map(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_domain_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_map(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_range_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_remove_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eliminate(self, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_eliminate(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sample(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_sample(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn detect_equalities(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_detect_equalities(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn fix_si(self, type_: DimType, pos: c_uint, value: c_int) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_fix_si(self.to(), type_.to(), pos.to(), value.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn fix_val(self, type_: DimType, pos: c_uint, v: Val) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_fix_val(self.to(), type_.to(), pos.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lower_bound_si(self, type_: DimType, pos: c_uint, value: c_int) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_lower_bound_si(self.to(), type_.to(), pos.to(), value.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn upper_bound_si(self, type_: DimType, pos: c_uint, value: c_int) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_upper_bound_si(self.to(), type_.to(), pos.to(), value.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sum(self, bmap2: BasicMap) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_sum(self.to(), bmap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn partial_lexmax(self, dom: BasicSet) -> Option<(Map, Set)> {
    unsafe {
      let ref mut empty = 0 as *mut c_void;
      let ret = isl_basic_map_partial_lexmax(self.to(), dom.to(), empty as *mut _ as _);
      (ret, *empty).to()
    }
  }
  #[inline(always)]
  pub fn partial_lexmin(self, dom: BasicSet) -> Option<(Map, Set)> {
    unsafe {
      let ref mut empty = 0 as *mut c_void;
      let ret = isl_basic_map_partial_lexmin(self.to(), dom.to(), empty as *mut _ as _);
      (ret, *empty).to()
    }
  }
  #[inline(always)]
  pub fn lexmin(self) -> Option<Map> {
    unsafe {
      let ret = isl_basic_map_lexmin(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lexmax(self) -> Option<Map> {
    unsafe {
      let ret = isl_basic_map_lexmax(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn partial_lexmin_pw_multi_aff(self, dom: BasicSet) -> Option<(PwMultiAff, Set)> {
    unsafe {
      let ref mut empty = 0 as *mut c_void;
      let ret = isl_basic_map_partial_lexmin_pw_multi_aff(self.to(), dom.to(), empty as *mut _ as _);
      (ret, *empty).to()
    }
  }
  #[inline(always)]
  pub fn partial_lexmax_pw_multi_aff(self, dom: BasicSet) -> Option<(PwMultiAff, Set)> {
    unsafe {
      let ref mut empty = 0 as *mut c_void;
      let ret = isl_basic_map_partial_lexmax_pw_multi_aff(self.to(), dom.to(), empty as *mut _ as _);
      (ret, *empty).to()
    }
  }
  #[inline(always)]
  pub fn lexmin_pw_multi_aff(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_basic_map_lexmin_pw_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn product(self, bmap2: BasicMap) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_product(self.to(), bmap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain_product(self, bmap2: BasicMap) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_domain_product(self.to(), bmap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_product(self, bmap2: BasicMap) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_range_product(self.to(), bmap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flat_product(self, bmap2: BasicMap) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_flat_product(self.to(), bmap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flat_range_product(self, bmap2: BasicMap) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_flat_range_product(self.to(), bmap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn deltas(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_map_deltas(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn deltas_map(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_deltas_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_dims(self, type_: DimType, n: c_uint) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_add_dims(self.to(), type_.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_dims(self, type_: DimType, pos: c_uint, n: c_uint) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_insert_dims(self.to(), type_.to(), pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn move_dims(self, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_move_dims(self.to(), dst_type.to(), dst_pos.to(), src_type.to(), src_pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn project_out(self, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_project_out(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_divs(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_remove_divs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_divs_involving_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_remove_divs_involving_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn equate(self, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_equate(self.to(), type1.to(), pos1.to(), type2.to(), pos2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn order_ge(self, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_order_ge(self.to(), type1.to(), pos1.to(), type2.to(), pos2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn order_gt(self, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_order_gt(self.to(), type1.to(), pos1.to(), type2.to(), pos2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn wrap(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_map_wrap(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_flatten(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten_domain(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_flatten_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten_range(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_flatten_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_from_basic_map(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_from_basic_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn zip(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_zip(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn curry(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_curry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn uncurry(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_uncurry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn compute_divs(self) -> Option<Map> {
    unsafe {
      let ret = isl_basic_map_compute_divs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_constraints_involving_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_drop_constraints_involving_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_constraints_not_involving_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_drop_constraints_not_involving_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_domain(self, context: BasicSet) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_gist_domain(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, context: BasicMap) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_gist(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_align_params(self.to(), model.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn list_from_basic_map(self) -> Option<BasicMapList> {
    unsafe {
      let ret = isl_basic_map_list_from_basic_map(self.to());
      (ret).to()
    }
  }
}

impl BasicMapList {
  #[inline(always)]
  pub fn intersect(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_list_intersect(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_basic_map_list_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, el: BasicMap) -> Option<BasicMapList> {
    unsafe {
      let ret = isl_basic_map_list_add(self.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert(self, pos: c_uint, el: BasicMap) -> Option<BasicMapList> {
    unsafe {
      let ret = isl_basic_map_list_insert(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, first: c_uint, n: c_uint) -> Option<BasicMapList> {
    unsafe {
      let ret = isl_basic_map_list_drop(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, list2: BasicMapList) -> Option<BasicMapList> {
    unsafe {
      let ret = isl_basic_map_list_concat(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_basic_map(self, index: c_int, el: BasicMap) -> Option<BasicMapList> {
    unsafe {
      let ret = isl_basic_map_list_set_basic_map(self.to(), index.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map<F1: FnMut(BasicMap) -> Option<BasicMap>>(self, fn_: &mut F1) -> Option<BasicMapList> {
    unsafe extern "C" fn fn1<F: FnMut(BasicMap) -> Option<BasicMap>>(el: BasicMap, user: *mut c_void) -> Option<BasicMap> { (*(user as *mut F))(el.to()).to() }
    unsafe {
      let ret = isl_basic_map_list_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort<F1: FnMut(BasicMapRef, BasicMapRef) -> c_int>(self, cmp: &mut F1) -> Option<BasicMapList> {
    unsafe extern "C" fn fn1<F: FnMut(BasicMapRef, BasicMapRef) -> c_int>(a: BasicMapRef, b: BasicMapRef, user: *mut c_void) -> c_int { (*(user as *mut F))(a.to(), b.to()).to() }
    unsafe {
      let ret = isl_basic_map_list_sort(self.to(), fn1::<F1>, cmp as *mut _ as _);
      (ret).to()
    }
  }
}

impl BasicMapListRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_basic_map_list_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<BasicMapList> {
    unsafe {
      let ret = isl_basic_map_list_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_basic_map(self) -> c_int {
    unsafe {
      let ret = isl_basic_map_list_n_basic_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_basic_map(self, index: c_int) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_list_get_basic_map(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(BasicMap) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(BasicMap) -> Option<()>>(el: BasicMap, user: *mut c_void) -> Stat { (*(user as *mut F))(el.to()).to() }
    unsafe {
      let ret = isl_basic_map_list_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_scc<F1: FnMut(BasicMapRef, BasicMapRef) -> Option<bool>, F2: FnMut(BasicMapList) -> Option<()>>(self, follows: &mut F1, fn_: &mut F2) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(BasicMapRef, BasicMapRef) -> Option<bool>>(a: BasicMapRef, b: BasicMapRef, user: *mut c_void) -> Bool { (*(user as *mut F))(a.to(), b.to()).to() }
    unsafe extern "C" fn fn2<F: FnMut(BasicMapList) -> Option<()>>(scc: BasicMapList, user: *mut c_void) -> Stat { (*(user as *mut F))(scc.to()).to() }
    unsafe {
      let ret = isl_basic_map_list_foreach_scc(self.to(), fn1::<F1>, follows as *mut _ as _, fn2::<F2>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_basic_map_list_dump(self.to());
      (ret).to()
    }
  }
}

impl BasicMapRef {
  #[inline(always)]
  pub fn n_in(self) -> c_uint {
    unsafe {
      let ret = isl_basic_map_n_in(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_out(self) -> c_uint {
    unsafe {
      let ret = isl_basic_map_n_out(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_param(self) -> c_uint {
    unsafe {
      let ret = isl_basic_map_n_param(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_div(self) -> c_uint {
    unsafe {
      let ret = isl_basic_map_n_div(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn total_dim(self) -> c_uint {
    unsafe {
      let ret = isl_basic_map_total_dim(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_uint {
    unsafe {
      let ret = isl_basic_map_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_basic_map_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_basic_map_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_div(self, pos: c_int) -> Option<Aff> {
    unsafe {
      let ret = isl_basic_map_get_div(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_local_space(self) -> Option<LocalSpace> {
    unsafe {
      let ret = isl_basic_map_get_local_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_name(self, type_: DimType) -> Option<CStr> {
    unsafe {
      let ret = isl_basic_map_get_tuple_name(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_name(self, type_: DimType, pos: c_uint) -> Option<CStr> {
    unsafe {
      let ret = isl_basic_map_get_dim_name(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_dim_id(self, type_: DimType, pos: c_uint) -> Option<bool> {
    unsafe {
      let ret = isl_basic_map_has_dim_id(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: CStr) -> c_int {
    unsafe {
      let ret = isl_basic_map_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_rational(self) -> Option<bool> {
    unsafe {
      let ret = isl_basic_map_is_rational(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_basic_map_dump(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_basic_map_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_equal(self, bmap2: BasicMapRef) -> Option<bool> {
    unsafe {
      let ret = isl_basic_map_is_equal(self.to(), bmap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_disjoint(self, bmap2: BasicMapRef) -> Option<bool> {
    unsafe {
      let ret = isl_basic_map_is_disjoint(self.to(), bmap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_internal(self, out: *mut FILE, indent: c_int) -> () {
    unsafe {
      let ret = isl_basic_map_print_internal(self.to(), out.to(), indent.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_get_val_if_fixed(self, type_: DimType, pos: c_uint) -> Option<Val> {
    unsafe {
      let ret = isl_basic_map_plain_get_val_if_fixed(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn image_is_bounded(self) -> Option<bool> {
    unsafe {
      let ret = isl_basic_map_image_is_bounded(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_universe(self) -> Option<bool> {
    unsafe {
      let ret = isl_basic_map_plain_is_universe(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_universe(self) -> Option<bool> {
    unsafe {
      let ret = isl_basic_map_is_universe(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_empty(self) -> Option<bool> {
    unsafe {
      let ret = isl_basic_map_plain_is_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_empty(self) -> Option<bool> {
    unsafe {
      let ret = isl_basic_map_is_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_subset(self, bmap2: BasicMapRef) -> Option<bool> {
    unsafe {
      let ret = isl_basic_map_is_subset(self.to(), bmap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_strict_subset(self, bmap2: BasicMapRef) -> Option<bool> {
    unsafe {
      let ret = isl_basic_map_is_strict_subset(self.to(), bmap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_single_valued(self) -> Option<bool> {
    unsafe {
      let ret = isl_basic_map_is_single_valued(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn can_zip(self) -> Option<bool> {
    unsafe {
      let ret = isl_basic_map_can_zip(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn can_curry(self) -> Option<bool> {
    unsafe {
      let ret = isl_basic_map_can_curry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn can_uncurry(self) -> Option<bool> {
    unsafe {
      let ret = isl_basic_map_can_uncurry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<bool> {
    unsafe {
      let ret = isl_basic_map_involves_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn equalities_matrix(self, c1: DimType, c2: DimType, c3: DimType, c4: DimType, c5: DimType) -> Option<Mat> {
    unsafe {
      let ret = isl_basic_map_equalities_matrix(self.to(), c1.to(), c2.to(), c3.to(), c4.to(), c5.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn inequalities_matrix(self, c1: DimType, c2: DimType, c3: DimType, c4: DimType, c5: DimType) -> Option<Mat> {
    unsafe {
      let ret = isl_basic_map_inequalities_matrix(self.to(), c1.to(), c2.to(), c3.to(), c4.to(), c5.to());
      (ret).to()
    }
  }
}

impl BasicSet {
  #[inline(always)]
  pub fn unwrap(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_set_unwrap(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_flatten(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_map_from_domain(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_from_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_map_from_range(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_from_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_map_from_domain_and_range(self, range: BasicSet) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_from_domain_and_range(self.to(), range.to());
      (ret).to()
    }
  }
}

impl BasicSetRef {
  #[inline(always)]
  pub fn is_wrapping(self) -> Option<bool> {
    unsafe {
      let ret = isl_basic_set_is_wrapping(self.to());
      (ret).to()
    }
  }
}

impl CtxRef {
  #[inline(always)]
  pub fn basic_map_read_from_file(self, input: *mut FILE) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_read_from_file(self.to(), input.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_map_read_from_str(self, str: CStr) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_read_from_file(self, input: *mut FILE) -> Option<Map> {
    unsafe {
      let ret = isl_map_read_from_file(self.to(), input.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_read_from_str(self, str: CStr) -> Option<Map> {
    unsafe {
      let ret = isl_map_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_map_list_alloc(self, n: c_int) -> Option<BasicMapList> {
    unsafe {
      let ret = isl_basic_map_list_alloc(self.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_list_alloc(self, n: c_int) -> Option<MapList> {
    unsafe {
      let ret = isl_map_list_alloc(self.to(), n.to());
      (ret).to()
    }
  }
}

impl Map {
  #[inline(always)]
  pub fn set_tuple_name(self, type_: DimType, s: CStr) -> Option<Map> {
    unsafe {
      let ret = isl_map_set_tuple_name(self.to(), type_.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_name(self, type_: DimType, pos: c_uint, s: CStr) -> Option<Map> {
    unsafe {
      let ret = isl_map_set_dim_name(self.to(), type_.to(), pos.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_id(self, type_: DimType, pos: c_uint, id: Id) -> Option<Map> {
    unsafe {
      let ret = isl_map_set_dim_id(self.to(), type_.to(), pos.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_id(self, type_: DimType, id: Id) -> Option<Map> {
    unsafe {
      let ret = isl_map_set_tuple_id(self.to(), type_.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_tuple_id(self, type_: DimType) -> Option<Map> {
    unsafe {
      let ret = isl_map_reset_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_redundancies(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_remove_redundancies(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn simple_hull(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_map_simple_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn unshifted_simple_hull(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_map_unshifted_simple_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_unshifted_simple_hull(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_map_plain_unshifted_simple_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn unshifted_simple_hull_from_map_list(self, list: MapList) -> Option<BasicMap> {
    unsafe {
      let ret = isl_map_unshifted_simple_hull_from_map_list(self.to(), list.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sum(self, map2: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_sum(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn floordiv_val(self, d: Val) -> Option<Map> {
    unsafe {
      let ret = isl_map_floordiv_val(self.to(), d.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn partial_lexmax(self, dom: Set) -> Option<(Map, Set)> {
    unsafe {
      let ref mut empty = 0 as *mut c_void;
      let ret = isl_map_partial_lexmax(self.to(), dom.to(), empty as *mut _ as _);
      (ret, *empty).to()
    }
  }
  #[inline(always)]
  pub fn partial_lexmin(self, dom: Set) -> Option<(Map, Set)> {
    unsafe {
      let ref mut empty = 0 as *mut c_void;
      let ret = isl_map_partial_lexmin(self.to(), dom.to(), empty as *mut _ as _);
      (ret, *empty).to()
    }
  }
  #[inline(always)]
  pub fn lexmin(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_lexmin(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lexmax(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_lexmax(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lexmin_pw_multi_aff(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_map_lexmin_pw_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lexmax_pw_multi_aff(self) -> Option<PwMultiAff> {
    unsafe {
      let ret = isl_map_lexmax_pw_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_map_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reverse(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_reverse(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union(self, map2: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_union(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_disjoint(self, map2: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_union_disjoint(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain(self, set: Set) -> Option<Map> {
    unsafe {
      let ret = isl_map_intersect_domain(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_range(self, set: Set) -> Option<Map> {
    unsafe {
      let ret = isl_map_intersect_range(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain_factor_range(self, factor: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_intersect_domain_factor_range(self.to(), factor.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_range_factor_range(self, factor: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_intersect_range_factor_range(self.to(), factor.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn apply_domain(self, map2: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_apply_domain(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn apply_range(self, map2: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_apply_range(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_domain_multi_aff(self, ma: MultiAff) -> Option<Map> {
    unsafe {
      let ret = isl_map_preimage_domain_multi_aff(self.to(), ma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_range_multi_aff(self, ma: MultiAff) -> Option<Map> {
    unsafe {
      let ret = isl_map_preimage_range_multi_aff(self.to(), ma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_domain_pw_multi_aff(self, pma: PwMultiAff) -> Option<Map> {
    unsafe {
      let ret = isl_map_preimage_domain_pw_multi_aff(self.to(), pma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_range_pw_multi_aff(self, pma: PwMultiAff) -> Option<Map> {
    unsafe {
      let ret = isl_map_preimage_range_pw_multi_aff(self.to(), pma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_domain_multi_pw_aff(self, mpa: MultiPwAff) -> Option<Map> {
    unsafe {
      let ret = isl_map_preimage_domain_multi_pw_aff(self.to(), mpa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn product(self, map2: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_product(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain_product(self, map2: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_domain_product(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_product(self, map2: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_range_product(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flat_product(self, map2: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_flat_product(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flat_domain_product(self, map2: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_flat_domain_product(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flat_range_product(self, map2: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_flat_range_product(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn factor_domain(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_factor_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn factor_range(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain_factor_domain(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_domain_factor_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain_factor_range(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_domain_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_factor_domain(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_range_factor_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_factor_range(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_range_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect(self, map2: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_intersect(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_params(self, params: Set) -> Option<Map> {
    unsafe {
      let ret = isl_map_intersect_params(self.to(), params.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract(self, map2: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_subtract(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract_domain(self, dom: Set) -> Option<Map> {
    unsafe {
      let ret = isl_map_subtract_domain(self.to(), dom.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract_range(self, dom: Set) -> Option<Map> {
    unsafe {
      let ret = isl_map_subtract_range(self.to(), dom.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn complement(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_complement(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn fix_input_si(self, input: c_uint, value: c_int) -> Option<Map> {
    unsafe {
      let ret = isl_map_fix_input_si(self.to(), input.to(), value.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn fix_si(self, type_: DimType, pos: c_uint, value: c_int) -> Option<Map> {
    unsafe {
      let ret = isl_map_fix_si(self.to(), type_.to(), pos.to(), value.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn fix_val(self, type_: DimType, pos: c_uint, v: Val) -> Option<Map> {
    unsafe {
      let ret = isl_map_fix_val(self.to(), type_.to(), pos.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lower_bound_si(self, type_: DimType, pos: c_uint, value: c_int) -> Option<Map> {
    unsafe {
      let ret = isl_map_lower_bound_si(self.to(), type_.to(), pos.to(), value.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn upper_bound_si(self, type_: DimType, pos: c_uint, value: c_int) -> Option<Map> {
    unsafe {
      let ret = isl_map_upper_bound_si(self.to(), type_.to(), pos.to(), value.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn deltas(self) -> Option<Set> {
    unsafe {
      let ret = isl_map_deltas(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn deltas_map(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_deltas_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn detect_equalities(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_detect_equalities(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn affine_hull(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_map_affine_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn convex_hull(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_map_convex_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn polyhedral_hull(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_map_polyhedral_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_dims(self, type_: DimType, n: c_uint) -> Option<Map> {
    unsafe {
      let ret = isl_map_add_dims(self.to(), type_.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_dims(self, type_: DimType, pos: c_uint, n: c_uint) -> Option<Map> {
    unsafe {
      let ret = isl_map_insert_dims(self.to(), type_.to(), pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn move_dims(self, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<Map> {
    unsafe {
      let ret = isl_map_move_dims(self.to(), dst_type.to(), dst_pos.to(), src_type.to(), src_pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn project_out(self, type_: DimType, first: c_uint, n: c_uint) -> Option<Map> {
    unsafe {
      let ret = isl_map_project_out(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_unknown_divs(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_remove_unknown_divs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_divs(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_remove_divs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eliminate(self, type_: DimType, first: c_uint, n: c_uint) -> Option<Map> {
    unsafe {
      let ret = isl_map_eliminate(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<Map> {
    unsafe {
      let ret = isl_map_remove_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_divs_involving_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<Map> {
    unsafe {
      let ret = isl_map_remove_divs_involving_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_inputs(self, first: c_uint, n: c_uint) -> Option<Map> {
    unsafe {
      let ret = isl_map_remove_inputs(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn order_ge(self, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<Map> {
    unsafe {
      let ret = isl_map_order_ge(self.to(), type1.to(), pos1.to(), type2.to(), pos2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn order_le(self, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<Map> {
    unsafe {
      let ret = isl_map_order_le(self.to(), type1.to(), pos1.to(), type2.to(), pos2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn equate(self, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<Map> {
    unsafe {
      let ret = isl_map_equate(self.to(), type1.to(), pos1.to(), type2.to(), pos2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn oppose(self, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<Map> {
    unsafe {
      let ret = isl_map_oppose(self.to(), type1.to(), pos1.to(), type2.to(), pos2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn order_lt(self, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<Map> {
    unsafe {
      let ret = isl_map_order_lt(self.to(), type1.to(), pos1.to(), type2.to(), pos2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn order_gt(self, type1: DimType, pos1: c_int, type2: DimType, pos2: c_int) -> Option<Map> {
    unsafe {
      let ret = isl_map_order_gt(self.to(), type1.to(), pos1.to(), type2.to(), pos2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn wrap(self) -> Option<Set> {
    unsafe {
      let ret = isl_map_wrap(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_flatten(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten_domain(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_flatten_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten_range(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_flatten_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn params(self) -> Option<Set> {
    unsafe {
      let ret = isl_map_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain(self) -> Option<Set> {
    unsafe {
      let ret = isl_map_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range(self) -> Option<Set> {
    unsafe {
      let ret = isl_map_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain_map(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_domain_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_map(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_range_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sample(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_map_sample(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn zip(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_zip(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn curry(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_curry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_curry(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_range_curry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn uncurry(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_uncurry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn make_disjoint(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_make_disjoint(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn compute_divs(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_compute_divs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_divs(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_align_divs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_constraints_involving_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<Map> {
    unsafe {
      let ret = isl_map_drop_constraints_involving_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_constraints_not_involving_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<Map> {
    unsafe {
      let ret = isl_map_drop_constraints_not_involving_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, context: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_gist(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_domain(self, context: Set) -> Option<Map> {
    unsafe {
      let ret = isl_map_gist_domain(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_range(self, context: Set) -> Option<Map> {
    unsafe {
      let ret = isl_map_gist_range(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_params(self, context: Set) -> Option<Map> {
    unsafe {
      let ret = isl_map_gist_params(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_basic_map(self, context: BasicMap) -> Option<Map> {
    unsafe {
      let ret = isl_map_gist_basic_map(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coalesce(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_coalesce(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn fixed_power_val(self, exp: Val) -> Option<Map> {
    unsafe {
      let ret = isl_map_fixed_power_val(self.to(), exp.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn power(self, exact: &mut c_int) -> Option<Map> {
    unsafe {
      let ret = isl_map_power(self.to(), exact.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reaching_path_lengths(self, exact: &mut c_int) -> Option<Map> {
    unsafe {
      let ret = isl_map_reaching_path_lengths(self.to(), exact.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn transitive_closure(self, exact: &mut c_int) -> Option<Map> {
    unsafe {
      let ret = isl_map_transitive_closure(self.to(), exact.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_le_map(self, map2: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_lex_le_map(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_lt_map(self, map2: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_lex_lt_map(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_ge_map(self, map2: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_lex_ge_map(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_gt_map(self, map2: Map) -> Option<Map> {
    unsafe {
      let ret = isl_map_lex_gt_map(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<Map> {
    unsafe {
      let ret = isl_map_align_params(self.to(), model.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim_min(self, pos: c_int) -> Option<PwAff> {
    unsafe {
      let ret = isl_map_dim_min(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim_max(self, pos: c_int) -> Option<PwAff> {
    unsafe {
      let ret = isl_map_dim_max(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn list_from_map(self) -> Option<MapList> {
    unsafe {
      let ret = isl_map_list_from_map(self.to());
      (ret).to()
    }
  }
}

impl MapList {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_map_list_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, el: Map) -> Option<MapList> {
    unsafe {
      let ret = isl_map_list_add(self.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert(self, pos: c_uint, el: Map) -> Option<MapList> {
    unsafe {
      let ret = isl_map_list_insert(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, first: c_uint, n: c_uint) -> Option<MapList> {
    unsafe {
      let ret = isl_map_list_drop(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, list2: MapList) -> Option<MapList> {
    unsafe {
      let ret = isl_map_list_concat(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_map(self, index: c_int, el: Map) -> Option<MapList> {
    unsafe {
      let ret = isl_map_list_set_map(self.to(), index.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map<F1: FnMut(Map) -> Option<Map>>(self, fn_: &mut F1) -> Option<MapList> {
    unsafe extern "C" fn fn1<F: FnMut(Map) -> Option<Map>>(el: Map, user: *mut c_void) -> Option<Map> { (*(user as *mut F))(el.to()).to() }
    unsafe {
      let ret = isl_map_list_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort<F1: FnMut(MapRef, MapRef) -> c_int>(self, cmp: &mut F1) -> Option<MapList> {
    unsafe extern "C" fn fn1<F: FnMut(MapRef, MapRef) -> c_int>(a: MapRef, b: MapRef, user: *mut c_void) -> c_int { (*(user as *mut F))(a.to(), b.to()).to() }
    unsafe {
      let ret = isl_map_list_sort(self.to(), fn1::<F1>, cmp as *mut _ as _);
      (ret).to()
    }
  }
}

impl MapListRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_map_list_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<MapList> {
    unsafe {
      let ret = isl_map_list_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_map(self) -> c_int {
    unsafe {
      let ret = isl_map_list_n_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_map(self, index: c_int) -> Option<Map> {
    unsafe {
      let ret = isl_map_list_get_map(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(Map) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(Map) -> Option<()>>(el: Map, user: *mut c_void) -> Stat { (*(user as *mut F))(el.to()).to() }
    unsafe {
      let ret = isl_map_list_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_scc<F1: FnMut(MapRef, MapRef) -> Option<bool>, F2: FnMut(MapList) -> Option<()>>(self, follows: &mut F1, fn_: &mut F2) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(MapRef, MapRef) -> Option<bool>>(a: MapRef, b: MapRef, user: *mut c_void) -> Bool { (*(user as *mut F))(a.to(), b.to()).to() }
    unsafe extern "C" fn fn2<F: FnMut(MapList) -> Option<()>>(scc: MapList, user: *mut c_void) -> Stat { (*(user as *mut F))(scc.to()).to() }
    unsafe {
      let ret = isl_map_list_foreach_scc(self.to(), fn1::<F1>, follows as *mut _ as _, fn2::<F2>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_map_list_dump(self.to());
      (ret).to()
    }
  }
}

impl MapRef {
  #[inline(always)]
  pub fn n_in(self) -> c_uint {
    unsafe {
      let ret = isl_map_n_in(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_out(self) -> c_uint {
    unsafe {
      let ret = isl_map_n_out(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_param(self) -> c_uint {
    unsafe {
      let ret = isl_map_n_param(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_uint {
    unsafe {
      let ret = isl_map_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_map_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_map_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_tuple_name(self, type_: DimType) -> Option<bool> {
    unsafe {
      let ret = isl_map_has_tuple_name(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_name(self, type_: DimType) -> Option<CStr> {
    unsafe {
      let ret = isl_map_get_tuple_name(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_dim_name(self, type_: DimType, pos: c_uint) -> Option<bool> {
    unsafe {
      let ret = isl_map_has_dim_name(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_name(self, type_: DimType, pos: c_uint) -> Option<CStr> {
    unsafe {
      let ret = isl_map_get_dim_name(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_dim_id(self, type_: DimType, pos: c_uint) -> Option<bool> {
    unsafe {
      let ret = isl_map_has_dim_id(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_id(self, type_: DimType, pos: c_uint) -> Option<Id> {
    unsafe {
      let ret = isl_map_get_dim_id(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_tuple_id(self, type_: DimType) -> Option<bool> {
    unsafe {
      let ret = isl_map_has_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_id(self, type_: DimType) -> Option<Id> {
    unsafe {
      let ret = isl_map_get_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_id(self, type_: DimType, id: IdRef) -> c_int {
    unsafe {
      let ret = isl_map_find_dim_by_id(self.to(), type_.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: CStr) -> c_int {
    unsafe {
      let ret = isl_map_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_map_dump(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_map_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain_is_wrapping(self) -> Option<bool> {
    unsafe {
      let ret = isl_map_domain_is_wrapping(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_is_wrapping(self) -> Option<bool> {
    unsafe {
      let ret = isl_map_range_is_wrapping(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_product(self) -> Option<bool> {
    unsafe {
      let ret = isl_map_is_product(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_empty(self) -> Option<bool> {
    unsafe {
      let ret = isl_map_plain_is_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_universe(self) -> Option<bool> {
    unsafe {
      let ret = isl_map_plain_is_universe(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_empty(self) -> Option<bool> {
    unsafe {
      let ret = isl_map_is_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_subset(self, map2: MapRef) -> Option<bool> {
    unsafe {
      let ret = isl_map_is_subset(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_strict_subset(self, map2: MapRef) -> Option<bool> {
    unsafe {
      let ret = isl_map_is_strict_subset(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_equal(self, map2: MapRef) -> Option<bool> {
    unsafe {
      let ret = isl_map_is_equal(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_disjoint(self, map2: MapRef) -> Option<bool> {
    unsafe {
      let ret = isl_map_is_disjoint(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_single_valued(self) -> Option<bool> {
    unsafe {
      let ret = isl_map_plain_is_single_valued(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_single_valued(self) -> Option<bool> {
    unsafe {
      let ret = isl_map_is_single_valued(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_injective(self) -> Option<bool> {
    unsafe {
      let ret = isl_map_plain_is_injective(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_injective(self) -> Option<bool> {
    unsafe {
      let ret = isl_map_is_injective(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_bijective(self) -> Option<bool> {
    unsafe {
      let ret = isl_map_is_bijective(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_identity(self) -> Option<bool> {
    unsafe {
      let ret = isl_map_is_identity(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_translation(self) -> c_int {
    unsafe {
      let ret = isl_map_is_translation(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_equal_space(self, map2: MapRef) -> Option<bool> {
    unsafe {
      let ret = isl_map_has_equal_space(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn can_zip(self) -> Option<bool> {
    unsafe {
      let ret = isl_map_can_zip(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn can_curry(self) -> Option<bool> {
    unsafe {
      let ret = isl_map_can_curry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn can_range_curry(self) -> Option<bool> {
    unsafe {
      let ret = isl_map_can_range_curry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn can_uncurry(self) -> Option<bool> {
    unsafe {
      let ret = isl_map_can_uncurry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<bool> {
    unsafe {
      let ret = isl_map_involves_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_internal(self, out: *mut FILE, indent: c_int) -> () {
    unsafe {
      let ret = isl_map_print_internal(self.to(), out.to(), indent.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_get_val_if_fixed(self, type_: DimType, pos: c_uint) -> Option<Val> {
    unsafe {
      let ret = isl_map_plain_get_val_if_fixed(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, map2: MapRef) -> Option<bool> {
    unsafe {
      let ret = isl_map_plain_is_equal(self.to(), map2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_hash(self) -> c_uint {
    unsafe {
      let ret = isl_map_get_hash(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_basic_map(self) -> c_int {
    unsafe {
      let ret = isl_map_n_basic_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_basic_map<F1: FnMut(BasicMap) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(BasicMap) -> Option<()>>(bmap: BasicMap, user: *mut c_void) -> Stat { (*(user as *mut F))(bmap.to()).to() }
    unsafe {
      let ret = isl_map_foreach_basic_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
}

impl MultiAff {
  #[inline(always)]
  pub fn basic_map_from_multi_aff(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_from_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_from_multi_aff(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_from_multi_aff(self.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_basic_map(self, bmap: BasicMapRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_basic_map(self.to(), bmap.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_map(self, map: MapRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_map(self.to(), map.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_basic_map_list(self, list: BasicMapListRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_basic_map_list(self.to(), list.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_map_list(self, list: MapListRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_map_list(self.to(), list.to());
      (ret).to()
    }
  }
}

impl Set {
  #[inline(always)]
  pub fn identity(self) -> Option<Map> {
    unsafe {
      let ret = isl_set_identity(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn unwrap(self) -> Option<Map> {
    unsafe {
      let ret = isl_set_unwrap(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten(self) -> Option<Set> {
    unsafe {
      let ret = isl_set_flatten(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten_map(self) -> Option<Map> {
    unsafe {
      let ret = isl_set_flatten_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn wrapped_domain_map(self) -> Option<Map> {
    unsafe {
      let ret = isl_set_wrapped_domain_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_from_domain(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_from_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_from_range(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_from_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_from_domain_and_range(self, range: Set) -> Option<Map> {
    unsafe {
      let ret = isl_map_from_domain_and_range(self.to(), range.to());
      (ret).to()
    }
  }
}

impl SetRef {
  #[inline(always)]
  pub fn is_wrapping(self) -> Option<bool> {
    unsafe {
      let ret = isl_set_is_wrapping(self.to());
      (ret).to()
    }
  }
}

impl Space {
  #[inline(always)]
  pub fn basic_map_identity(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_identity(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_map_equal(self, n_equal: c_uint) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_equal(self.to(), n_equal.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_map_less_at(self, pos: c_uint) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_less_at(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_map_more_at(self, pos: c_uint) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_more_at(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_map_empty(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_map_universe(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_universe(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_map_nat_universe(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_nat_universe(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_universe(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_universe(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_nat_universe(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_nat_universe(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_empty(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_identity(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_identity(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_lex_lt_first(self, n: c_uint) -> Option<Map> {
    unsafe {
      let ret = isl_map_lex_lt_first(self.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_lex_le_first(self, n: c_uint) -> Option<Map> {
    unsafe {
      let ret = isl_map_lex_le_first(self.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_lex_lt(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_lex_lt(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_lex_le(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_lex_le(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_lex_gt_first(self, n: c_uint) -> Option<Map> {
    unsafe {
      let ret = isl_map_lex_gt_first(self.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_lex_ge_first(self, n: c_uint) -> Option<Map> {
    unsafe {
      let ret = isl_map_lex_ge_first(self.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_lex_gt(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_lex_gt(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_lex_ge(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_lex_ge(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_map_from_constraint_matrices(self, eq: Mat, ineq: Mat, c1: DimType, c2: DimType, c3: DimType, c4: DimType, c5: DimType) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_from_constraint_matrices(self.to(), eq.to(), ineq.to(), c1.to(), c2.to(), c3.to(), c4.to(), c5.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_map_from_aff_list(self, list: AffList) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_from_aff_list(self.to(), list.to());
      (ret).to()
    }
  }
}

impl Drop for BasicMap {
  fn drop(&mut self) { BasicMap(self.0).free() }
}

impl Drop for BasicMapList {
  fn drop(&mut self) { BasicMapList(self.0).free() }
}

impl fmt::Display for BasicMapRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for BasicMap {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl Drop for Map {
  fn drop(&mut self) { Map(self.0).free() }
}

impl Drop for MapList {
  fn drop(&mut self) { MapList(self.0).free() }
}

impl fmt::Display for MapRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for Map {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

