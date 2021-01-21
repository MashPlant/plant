use crate::*;

extern "C" {
  pub fn isl_union_map_dim(umap: UnionMapRef, type_: DimType) -> c_int;
  pub fn isl_union_map_involves_dims(umap: UnionMapRef, type_: DimType, first: c_uint, n: c_uint) -> Bool;
  pub fn isl_union_map_get_dim_id(umap: UnionMapRef, type_: DimType, pos: c_uint) -> Option<Id>;
  pub fn isl_union_map_from_basic_map(bmap: BasicMap) -> Option<UnionMap>;
  pub fn isl_union_map_from_map(map: Map) -> Option<UnionMap>;
  pub fn isl_union_map_empty_ctx(ctx: CtxRef) -> Option<UnionMap>;
  pub fn isl_union_map_empty_space(space: Space) -> Option<UnionMap>;
  pub fn isl_union_map_empty(space: Space) -> Option<UnionMap>;
  pub fn isl_union_map_copy(umap: UnionMapRef) -> Option<UnionMap>;
  pub fn isl_union_map_free(umap: UnionMap) -> *mut c_void;
  pub fn isl_union_map_get_ctx(umap: UnionMapRef) -> Option<CtxRef>;
  pub fn isl_union_map_get_space(umap: UnionMapRef) -> Option<Space>;
  pub fn isl_union_map_reset_user(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_find_dim_by_name(umap: UnionMapRef, type_: DimType, name: Option<CStr>) -> c_int;
  pub fn isl_union_map_universe(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_params(umap: UnionMap) -> Option<Set>;
  pub fn isl_union_map_domain(umap: UnionMap) -> Option<UnionSet>;
  pub fn isl_union_map_range(umap: UnionMap) -> Option<UnionSet>;
  pub fn isl_union_map_domain_map(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_domain_map_union_pw_multi_aff(umap: UnionMap) -> Option<UnionPwMultiAff>;
  pub fn isl_union_map_range_map(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_set_wrapped_domain_map(uset: UnionSet) -> Option<UnionMap>;
  pub fn isl_union_map_from_domain(uset: UnionSet) -> Option<UnionMap>;
  pub fn isl_union_map_from_range(uset: UnionSet) -> Option<UnionMap>;
  pub fn isl_union_map_affine_hull(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_polyhedral_hull(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_remove_redundancies(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_simple_hull(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_coalesce(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_compute_divs(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_lexmin(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_lexmax(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_add_map(umap: UnionMap, map: Map) -> Option<UnionMap>;
  pub fn isl_union_map_union(umap1: UnionMap, umap2: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_subtract(umap1: UnionMap, umap2: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_intersect(umap1: UnionMap, umap2: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_intersect_params(umap: UnionMap, set: Set) -> Option<UnionMap>;
  pub fn isl_union_map_product(umap1: UnionMap, umap2: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_domain_product(umap1: UnionMap, umap2: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_flat_domain_product(umap1: UnionMap, umap2: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_range_product(umap1: UnionMap, umap2: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_flat_range_product(umap1: UnionMap, umap2: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_domain_factor_domain(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_domain_factor_range(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_range_factor_domain(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_range_factor_range(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_factor_domain(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_factor_range(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_gist(umap: UnionMap, context: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_gist_params(umap: UnionMap, set: Set) -> Option<UnionMap>;
  pub fn isl_union_map_gist_domain(umap: UnionMap, uset: UnionSet) -> Option<UnionMap>;
  pub fn isl_union_map_gist_range(umap: UnionMap, uset: UnionSet) -> Option<UnionMap>;
  pub fn isl_union_map_intersect_domain_union_set(umap: UnionMap, uset: UnionSet) -> Option<UnionMap>;
  pub fn isl_union_map_intersect_domain_space(umap: UnionMap, space: Space) -> Option<UnionMap>;
  pub fn isl_union_map_intersect_domain(umap: UnionMap, uset: UnionSet) -> Option<UnionMap>;
  pub fn isl_union_map_intersect_range_union_set(umap: UnionMap, uset: UnionSet) -> Option<UnionMap>;
  pub fn isl_union_map_intersect_range_space(umap: UnionMap, space: Space) -> Option<UnionMap>;
  pub fn isl_union_map_intersect_range(umap: UnionMap, uset: UnionSet) -> Option<UnionMap>;
  pub fn isl_union_map_intersect_domain_factor_domain(umap: UnionMap, factor: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_intersect_domain_factor_range(umap: UnionMap, factor: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_intersect_range_factor_domain(umap: UnionMap, factor: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_intersect_range_factor_range(umap: UnionMap, factor: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_subtract_domain(umap: UnionMap, dom: UnionSet) -> Option<UnionMap>;
  pub fn isl_union_map_subtract_range(umap: UnionMap, dom: UnionSet) -> Option<UnionMap>;
  pub fn isl_union_map_apply_domain(umap1: UnionMap, umap2: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_apply_range(umap1: UnionMap, umap2: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_preimage_domain_multi_aff(umap: UnionMap, ma: MultiAff) -> Option<UnionMap>;
  pub fn isl_union_map_preimage_range_multi_aff(umap: UnionMap, ma: MultiAff) -> Option<UnionMap>;
  pub fn isl_union_map_preimage_domain_pw_multi_aff(umap: UnionMap, pma: PwMultiAff) -> Option<UnionMap>;
  pub fn isl_union_map_preimage_range_pw_multi_aff(umap: UnionMap, pma: PwMultiAff) -> Option<UnionMap>;
  pub fn isl_union_map_preimage_domain_multi_pw_aff(umap: UnionMap, mpa: MultiPwAff) -> Option<UnionMap>;
  pub fn isl_union_map_preimage_domain_union_pw_multi_aff(umap: UnionMap, upma: UnionPwMultiAff) -> Option<UnionMap>;
  pub fn isl_union_map_preimage_range_union_pw_multi_aff(umap: UnionMap, upma: UnionPwMultiAff) -> Option<UnionMap>;
  pub fn isl_union_map_reverse(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_range_reverse(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_from_domain_and_range(domain: UnionSet, range: UnionSet) -> Option<UnionMap>;
  pub fn isl_union_map_detect_equalities(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_deltas(umap: UnionMap) -> Option<UnionSet>;
  pub fn isl_union_map_deltas_map(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_set_identity(uset: UnionSet) -> Option<UnionMap>;
  pub fn isl_union_map_project_out(umap: UnionMap, type_: DimType, first: c_uint, n: c_uint) -> Option<UnionMap>;
  pub fn isl_union_map_project_out_all_params(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_remove_divs(bmap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_bind_range(umap: UnionMap, tuple: MultiId) -> Option<UnionSet>;
  pub fn isl_union_map_plain_is_empty(umap: UnionMapRef) -> Bool;
  pub fn isl_union_map_is_empty(umap: UnionMapRef) -> Bool;
  pub fn isl_union_map_is_single_valued(umap: UnionMapRef) -> Bool;
  pub fn isl_union_map_plain_is_injective(umap: UnionMapRef) -> Bool;
  pub fn isl_union_map_is_injective(umap: UnionMapRef) -> Bool;
  pub fn isl_union_map_is_bijective(umap: UnionMapRef) -> Bool;
  pub fn isl_union_map_is_identity(umap: UnionMapRef) -> Bool;
  pub fn isl_union_map_is_subset(umap1: UnionMapRef, umap2: UnionMapRef) -> Bool;
  pub fn isl_union_map_is_equal(umap1: UnionMapRef, umap2: UnionMapRef) -> Bool;
  pub fn isl_union_map_is_disjoint(umap1: UnionMapRef, umap2: UnionMapRef) -> Bool;
  pub fn isl_union_map_is_strict_subset(umap1: UnionMapRef, umap2: UnionMapRef) -> Bool;
  pub fn isl_union_map_get_hash(umap: UnionMapRef) -> c_uint;
  pub fn isl_union_map_n_map(umap: UnionMapRef) -> c_int;
  pub fn isl_union_map_foreach_map(umap: UnionMapRef, fn_: unsafe extern "C" fn(map: Map, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_union_map_get_map_list(umap: UnionMapRef) -> Option<MapList>;
  pub fn isl_union_map_every_map(umap: UnionMapRef, test: unsafe extern "C" fn(map: MapRef, user: *mut c_void) -> Bool, user: *mut c_void) -> Bool;
  pub fn isl_union_map_remove_map_if(umap: UnionMap, fn_: unsafe extern "C" fn(map: MapRef, user: *mut c_void) -> Bool, user: *mut c_void) -> Option<UnionMap>;
  pub fn isl_union_map_contains(umap: UnionMapRef, space: SpaceRef) -> Bool;
  pub fn isl_union_map_extract_map(umap: UnionMapRef, space: Space) -> Option<Map>;
  pub fn isl_union_map_isa_map(umap: UnionMapRef) -> Bool;
  pub fn isl_map_from_union_map(umap: UnionMap) -> Option<Map>;
  pub fn isl_union_map_sample(umap: UnionMap) -> Option<BasicMap>;
  pub fn isl_union_map_fixed_power_val(umap: UnionMap, exp: Val) -> Option<UnionMap>;
  pub fn isl_union_map_power(umap: UnionMap, exact: *mut Bool) -> Option<UnionMap>;
  pub fn isl_union_map_transitive_closure(umap: UnionMap, exact: *mut Bool) -> Option<UnionMap>;
  pub fn isl_union_map_lex_lt_union_map(umap1: UnionMap, umap2: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_lex_le_union_map(umap1: UnionMap, umap2: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_lex_gt_union_map(umap1: UnionMap, umap2: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_lex_ge_union_map(umap1: UnionMap, umap2: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_eq_at_multi_union_pw_aff(umap: UnionMap, mupa: MultiUnionPwAff) -> Option<UnionMap>;
  pub fn isl_union_map_lex_le_at_multi_union_pw_aff(umap: UnionMap, mupa: MultiUnionPwAff) -> Option<UnionMap>;
  pub fn isl_union_map_lex_lt_at_multi_union_pw_aff(umap: UnionMap, mupa: MultiUnionPwAff) -> Option<UnionMap>;
  pub fn isl_union_map_lex_ge_at_multi_union_pw_aff(umap: UnionMap, mupa: MultiUnionPwAff) -> Option<UnionMap>;
  pub fn isl_union_map_lex_gt_at_multi_union_pw_aff(umap: UnionMap, mupa: MultiUnionPwAff) -> Option<UnionMap>;
  pub fn isl_union_map_read_from_file(ctx: CtxRef, input: *mut FILE) -> Option<UnionMap>;
  pub fn isl_union_map_read_from_str(ctx: CtxRef, str: Option<CStr>) -> Option<UnionMap>;
  pub fn isl_union_map_to_str(umap: UnionMapRef) -> Option<CString>;
  pub fn isl_printer_print_union_map(p: Printer, umap: UnionMapRef) -> Option<Printer>;
  pub fn isl_union_map_dump(umap: UnionMapRef) -> ();
  pub fn isl_union_map_wrap(umap: UnionMap) -> Option<UnionSet>;
  pub fn isl_union_set_unwrap(uset: UnionSet) -> Option<UnionMap>;
  pub fn isl_union_map_zip(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_curry(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_range_curry(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_uncurry(umap: UnionMap) -> Option<UnionMap>;
  pub fn isl_union_map_align_params(umap: UnionMap, model: Space) -> Option<UnionMap>;
  pub fn isl_union_set_align_params(uset: UnionSet, model: Space) -> Option<UnionSet>;
  pub fn isl_union_map_list_get_ctx(list: UnionMapListRef) -> Option<CtxRef>;
  pub fn isl_union_map_list_from_union_map(el: UnionMap) -> Option<UnionMapList>;
  pub fn isl_union_map_list_alloc(ctx: CtxRef, n: c_int) -> Option<UnionMapList>;
  pub fn isl_union_map_list_copy(list: UnionMapListRef) -> Option<UnionMapList>;
  pub fn isl_union_map_list_free(list: UnionMapList) -> *mut c_void;
  pub fn isl_union_map_list_add(list: UnionMapList, el: UnionMap) -> Option<UnionMapList>;
  pub fn isl_union_map_list_insert(list: UnionMapList, pos: c_uint, el: UnionMap) -> Option<UnionMapList>;
  pub fn isl_union_map_list_drop(list: UnionMapList, first: c_uint, n: c_uint) -> Option<UnionMapList>;
  pub fn isl_union_map_list_clear(list: UnionMapList) -> Option<UnionMapList>;
  pub fn isl_union_map_list_swap(list: UnionMapList, pos1: c_uint, pos2: c_uint) -> Option<UnionMapList>;
  pub fn isl_union_map_list_reverse(list: UnionMapList) -> Option<UnionMapList>;
  pub fn isl_union_map_list_concat(list1: UnionMapList, list2: UnionMapList) -> Option<UnionMapList>;
  pub fn isl_union_map_list_size(list: UnionMapListRef) -> c_int;
  pub fn isl_union_map_list_n_union_map(list: UnionMapListRef) -> c_int;
  pub fn isl_union_map_list_get_at(list: UnionMapListRef, index: c_int) -> Option<UnionMap>;
  pub fn isl_union_map_list_get_union_map(list: UnionMapListRef, index: c_int) -> Option<UnionMap>;
  pub fn isl_union_map_list_set_union_map(list: UnionMapList, index: c_int, el: UnionMap) -> Option<UnionMapList>;
  pub fn isl_union_map_list_foreach(list: UnionMapListRef, fn_: unsafe extern "C" fn(el: UnionMap, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_union_map_list_every(list: UnionMapListRef, test: unsafe extern "C" fn(el: UnionMapRef, user: *mut c_void) -> Bool, user: *mut c_void) -> Bool;
  pub fn isl_union_map_list_map(list: UnionMapList, fn_: unsafe extern "C" fn(el: UnionMap, user: *mut c_void) -> Option<UnionMap>, user: *mut c_void) -> Option<UnionMapList>;
  pub fn isl_union_map_list_sort(list: UnionMapList, cmp: unsafe extern "C" fn(a: UnionMapRef, b: UnionMapRef, user: *mut c_void) -> c_int, user: *mut c_void) -> Option<UnionMapList>;
  pub fn isl_union_map_list_foreach_scc(list: UnionMapListRef, follows: unsafe extern "C" fn(a: UnionMapRef, b: UnionMapRef, user: *mut c_void) -> Bool, follows_user: *mut c_void, fn_: unsafe extern "C" fn(scc: UnionMapList, user: *mut c_void) -> Stat, fn_user: *mut c_void) -> Stat;
  pub fn isl_union_map_list_to_str(list: UnionMapListRef) -> Option<CString>;
  pub fn isl_printer_print_union_map_list(p: Printer, list: UnionMapListRef) -> Option<Printer>;
  pub fn isl_union_map_list_dump(list: UnionMapListRef) -> ();
}

impl BasicMap {
  #[inline(always)]
  pub fn union_map_from_basic_map(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_from_basic_map(self.to());
      (ret).to()
    }
  }
}

impl CtxRef {
  #[inline(always)]
  pub fn union_map_empty_ctx(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_empty_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_map_read_from_file(self, input: *mut FILE) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_read_from_file(self.to(), input.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_map_read_from_str(self, str: Option<CStr>) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_map_list_alloc(self, n: c_int) -> Option<UnionMapList> {
    unsafe {
      let ret = isl_union_map_list_alloc(self.to(), n.to());
      (ret).to()
    }
  }
}

impl Map {
  #[inline(always)]
  pub fn union_map_from_map(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_from_map(self.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_union_map(self, umap: UnionMapRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_union_map(self.to(), umap.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_union_map_list(self, list: UnionMapListRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_union_map_list(self.to(), list.to());
      (ret).to()
    }
  }
}

impl Space {
  #[inline(always)]
  pub fn union_map_empty_space(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_empty_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_map_empty(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_empty(self.to());
      (ret).to()
    }
  }
}

impl UnionMap {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_union_map_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn universe(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_universe(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn params(self) -> Option<Set> {
    unsafe {
      let ret = isl_union_map_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_map_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_map_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain_map(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_domain_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain_map_union_pw_multi_aff(self) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_union_map_domain_map_union_pw_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_map(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_range_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn affine_hull(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_affine_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn polyhedral_hull(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_polyhedral_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_redundancies(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_remove_redundancies(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn simple_hull(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_simple_hull(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coalesce(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_coalesce(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn compute_divs(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_compute_divs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lexmin(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_lexmin(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lexmax(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_lexmax(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_map(self, map: Map) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_add_map(self.to(), map.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union(self, umap2: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_union(self.to(), umap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract(self, umap2: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_subtract(self.to(), umap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect(self, umap2: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_intersect(self.to(), umap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_params(self, set: Set) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_intersect_params(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn product(self, umap2: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_product(self.to(), umap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain_product(self, umap2: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_domain_product(self.to(), umap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flat_domain_product(self, umap2: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_flat_domain_product(self.to(), umap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_product(self, umap2: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_range_product(self.to(), umap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flat_range_product(self, umap2: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_flat_range_product(self.to(), umap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain_factor_domain(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_domain_factor_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain_factor_range(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_domain_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_factor_domain(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_range_factor_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_factor_range(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_range_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn factor_domain(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_factor_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn factor_range(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, context: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_gist(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_params(self, set: Set) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_gist_params(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_domain(self, uset: UnionSet) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_gist_domain(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_range(self, uset: UnionSet) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_gist_range(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain_union_set(self, uset: UnionSet) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_intersect_domain_union_set(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain_space(self, space: Space) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_intersect_domain_space(self.to(), space.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain(self, uset: UnionSet) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_intersect_domain(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_range_union_set(self, uset: UnionSet) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_intersect_range_union_set(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_range_space(self, space: Space) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_intersect_range_space(self.to(), space.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_range(self, uset: UnionSet) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_intersect_range(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain_factor_domain(self, factor: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_intersect_domain_factor_domain(self.to(), factor.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain_factor_range(self, factor: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_intersect_domain_factor_range(self.to(), factor.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_range_factor_domain(self, factor: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_intersect_range_factor_domain(self.to(), factor.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_range_factor_range(self, factor: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_intersect_range_factor_range(self.to(), factor.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract_domain(self, dom: UnionSet) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_subtract_domain(self.to(), dom.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract_range(self, dom: UnionSet) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_subtract_range(self.to(), dom.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn apply_domain(self, umap2: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_apply_domain(self.to(), umap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn apply_range(self, umap2: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_apply_range(self.to(), umap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_domain_multi_aff(self, ma: MultiAff) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_preimage_domain_multi_aff(self.to(), ma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_range_multi_aff(self, ma: MultiAff) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_preimage_range_multi_aff(self.to(), ma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_domain_pw_multi_aff(self, pma: PwMultiAff) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_preimage_domain_pw_multi_aff(self.to(), pma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_range_pw_multi_aff(self, pma: PwMultiAff) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_preimage_range_pw_multi_aff(self.to(), pma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_domain_multi_pw_aff(self, mpa: MultiPwAff) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_preimage_domain_multi_pw_aff(self.to(), mpa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_domain_union_pw_multi_aff(self, upma: UnionPwMultiAff) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_preimage_domain_union_pw_multi_aff(self.to(), upma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn preimage_range_union_pw_multi_aff(self, upma: UnionPwMultiAff) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_preimage_range_union_pw_multi_aff(self.to(), upma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reverse(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_reverse(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_reverse(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_range_reverse(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn detect_equalities(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_detect_equalities(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn deltas(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_map_deltas(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn deltas_map(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_deltas_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn project_out(self, type_: DimType, first: c_uint, n: c_uint) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_project_out(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn project_out_all_params(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_project_out_all_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_divs(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_remove_divs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn bind_range(self, tuple: MultiId) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_map_bind_range(self.to(), tuple.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn remove_map_if<F1: FnMut(MapRef) -> Bool>(self, fn_: &mut F1) -> Option<UnionMap> {
    unsafe extern "C" fn fn1<F: FnMut(MapRef) -> Bool>(map: MapRef, user: *mut c_void) -> Bool { (*(user as *mut F))(map.to()) }
    unsafe {
      let ret = isl_union_map_remove_map_if(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_from_union_map(self) -> Option<Map> {
    unsafe {
      let ret = isl_map_from_union_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sample(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_union_map_sample(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn fixed_power_val(self, exp: Val) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_fixed_power_val(self.to(), exp.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn power(self, exact: &mut Bool) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_power(self.to(), exact.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn transitive_closure(self, exact: &mut Bool) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_transitive_closure(self.to(), exact.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_lt_union_map(self, umap2: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_lex_lt_union_map(self.to(), umap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_le_union_map(self, umap2: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_lex_le_union_map(self.to(), umap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_gt_union_map(self, umap2: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_lex_gt_union_map(self.to(), umap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_ge_union_map(self, umap2: UnionMap) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_lex_ge_union_map(self.to(), umap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eq_at_multi_union_pw_aff(self, mupa: MultiUnionPwAff) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_eq_at_multi_union_pw_aff(self.to(), mupa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_le_at_multi_union_pw_aff(self, mupa: MultiUnionPwAff) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_lex_le_at_multi_union_pw_aff(self.to(), mupa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_lt_at_multi_union_pw_aff(self, mupa: MultiUnionPwAff) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_lex_lt_at_multi_union_pw_aff(self.to(), mupa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_ge_at_multi_union_pw_aff(self, mupa: MultiUnionPwAff) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_lex_ge_at_multi_union_pw_aff(self.to(), mupa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lex_gt_at_multi_union_pw_aff(self, mupa: MultiUnionPwAff) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_lex_gt_at_multi_union_pw_aff(self.to(), mupa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn wrap(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_map_wrap(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn zip(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_zip(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn curry(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_curry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_curry(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_range_curry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn uncurry(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_uncurry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_align_params(self.to(), model.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn list_from_union_map(self) -> Option<UnionMapList> {
    unsafe {
      let ret = isl_union_map_list_from_union_map(self.to());
      (ret).to()
    }
  }
}

impl UnionMapList {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_union_map_list_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, el: UnionMap) -> Option<UnionMapList> {
    unsafe {
      let ret = isl_union_map_list_add(self.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert(self, pos: c_uint, el: UnionMap) -> Option<UnionMapList> {
    unsafe {
      let ret = isl_union_map_list_insert(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, first: c_uint, n: c_uint) -> Option<UnionMapList> {
    unsafe {
      let ret = isl_union_map_list_drop(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn clear(self) -> Option<UnionMapList> {
    unsafe {
      let ret = isl_union_map_list_clear(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn swap(self, pos1: c_uint, pos2: c_uint) -> Option<UnionMapList> {
    unsafe {
      let ret = isl_union_map_list_swap(self.to(), pos1.to(), pos2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reverse(self) -> Option<UnionMapList> {
    unsafe {
      let ret = isl_union_map_list_reverse(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, list2: UnionMapList) -> Option<UnionMapList> {
    unsafe {
      let ret = isl_union_map_list_concat(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_union_map(self, index: c_int, el: UnionMap) -> Option<UnionMapList> {
    unsafe {
      let ret = isl_union_map_list_set_union_map(self.to(), index.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map<F1: FnMut(UnionMap) -> Option<UnionMap>>(self, fn_: &mut F1) -> Option<UnionMapList> {
    unsafe extern "C" fn fn1<F: FnMut(UnionMap) -> Option<UnionMap>>(el: UnionMap, user: *mut c_void) -> Option<UnionMap> { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_union_map_list_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort<F1: FnMut(UnionMapRef, UnionMapRef) -> c_int>(self, cmp: &mut F1) -> Option<UnionMapList> {
    unsafe extern "C" fn fn1<F: FnMut(UnionMapRef, UnionMapRef) -> c_int>(a: UnionMapRef, b: UnionMapRef, user: *mut c_void) -> c_int { (*(user as *mut F))(a.to(), b.to()) }
    unsafe {
      let ret = isl_union_map_list_sort(self.to(), fn1::<F1>, cmp as *mut _ as _);
      (ret).to()
    }
  }
}

impl UnionMapListRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_union_map_list_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<UnionMapList> {
    unsafe {
      let ret = isl_union_map_list_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn size(self) -> c_int {
    unsafe {
      let ret = isl_union_map_list_size(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_union_map(self) -> c_int {
    unsafe {
      let ret = isl_union_map_list_n_union_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_at(self, index: c_int) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_list_get_at(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_union_map(self, index: c_int) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_list_get_union_map(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(UnionMap) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(UnionMap) -> Stat>(el: UnionMap, user: *mut c_void) -> Stat { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_union_map_list_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn every<F1: FnMut(UnionMapRef) -> Bool>(self, test: &mut F1) -> Bool {
    unsafe extern "C" fn fn1<F: FnMut(UnionMapRef) -> Bool>(el: UnionMapRef, user: *mut c_void) -> Bool { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_union_map_list_every(self.to(), fn1::<F1>, test as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_scc<F1: FnMut(UnionMapRef, UnionMapRef) -> Bool, F2: FnMut(UnionMapList) -> Stat>(self, follows: &mut F1, fn_: &mut F2) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(UnionMapRef, UnionMapRef) -> Bool>(a: UnionMapRef, b: UnionMapRef, user: *mut c_void) -> Bool { (*(user as *mut F))(a.to(), b.to()) }
    unsafe extern "C" fn fn2<F: FnMut(UnionMapList) -> Stat>(scc: UnionMapList, user: *mut c_void) -> Stat { (*(user as *mut F))(scc.to()) }
    unsafe {
      let ret = isl_union_map_list_foreach_scc(self.to(), fn1::<F1>, follows as *mut _ as _, fn2::<F2>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_union_map_list_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_union_map_list_dump(self.to());
      (ret).to()
    }
  }
}

impl UnionMapRef {
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_int {
    unsafe {
      let ret = isl_union_map_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Bool {
    unsafe {
      let ret = isl_union_map_involves_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_id(self, type_: DimType, pos: c_uint) -> Option<Id> {
    unsafe {
      let ret = isl_union_map_get_dim_id(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_union_map_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_union_map_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: Option<CStr>) -> c_int {
    unsafe {
      let ret = isl_union_map_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_empty(self) -> Bool {
    unsafe {
      let ret = isl_union_map_plain_is_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_empty(self) -> Bool {
    unsafe {
      let ret = isl_union_map_is_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_single_valued(self) -> Bool {
    unsafe {
      let ret = isl_union_map_is_single_valued(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_injective(self) -> Bool {
    unsafe {
      let ret = isl_union_map_plain_is_injective(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_injective(self) -> Bool {
    unsafe {
      let ret = isl_union_map_is_injective(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_bijective(self) -> Bool {
    unsafe {
      let ret = isl_union_map_is_bijective(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_identity(self) -> Bool {
    unsafe {
      let ret = isl_union_map_is_identity(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_subset(self, umap2: UnionMapRef) -> Bool {
    unsafe {
      let ret = isl_union_map_is_subset(self.to(), umap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_equal(self, umap2: UnionMapRef) -> Bool {
    unsafe {
      let ret = isl_union_map_is_equal(self.to(), umap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_disjoint(self, umap2: UnionMapRef) -> Bool {
    unsafe {
      let ret = isl_union_map_is_disjoint(self.to(), umap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_strict_subset(self, umap2: UnionMapRef) -> Bool {
    unsafe {
      let ret = isl_union_map_is_strict_subset(self.to(), umap2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_hash(self) -> c_uint {
    unsafe {
      let ret = isl_union_map_get_hash(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_map(self) -> c_int {
    unsafe {
      let ret = isl_union_map_n_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_map<F1: FnMut(Map) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(Map) -> Stat>(map: Map, user: *mut c_void) -> Stat { (*(user as *mut F))(map.to()) }
    unsafe {
      let ret = isl_union_map_foreach_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_map_list(self) -> Option<MapList> {
    unsafe {
      let ret = isl_union_map_get_map_list(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn every_map<F1: FnMut(MapRef) -> Bool>(self, test: &mut F1) -> Bool {
    unsafe extern "C" fn fn1<F: FnMut(MapRef) -> Bool>(map: MapRef, user: *mut c_void) -> Bool { (*(user as *mut F))(map.to()) }
    unsafe {
      let ret = isl_union_map_every_map(self.to(), fn1::<F1>, test as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn contains(self, space: SpaceRef) -> Bool {
    unsafe {
      let ret = isl_union_map_contains(self.to(), space.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn extract_map(self, space: Space) -> Option<Map> {
    unsafe {
      let ret = isl_union_map_extract_map(self.to(), space.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn isa_map(self) -> Bool {
    unsafe {
      let ret = isl_union_map_isa_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_union_map_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_union_map_dump(self.to());
      (ret).to()
    }
  }
}

impl UnionSet {
  #[inline(always)]
  pub fn wrapped_domain_map(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_set_wrapped_domain_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_map_from_domain(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_from_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_map_from_range(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_from_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_map_from_domain_and_range(self, range: UnionSet) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_map_from_domain_and_range(self.to(), range.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn identity(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_set_identity(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn unwrap(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_set_unwrap(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_set_align_params(self.to(), model.to());
      (ret).to()
    }
  }
}

impl Drop for UnionMap {
  fn drop(&mut self) { UnionMap(self.0).free() }
}

impl Drop for UnionMapList {
  fn drop(&mut self) { UnionMapList(self.0).free() }
}

impl fmt::Display for UnionMapListRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for UnionMapList {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl fmt::Display for UnionMapRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for UnionMap {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

