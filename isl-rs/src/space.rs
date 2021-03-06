use crate::*;

extern "C" {
  pub fn isl_space_get_ctx(space: SpaceRef) -> Option<CtxRef>;
  pub fn isl_space_unit(ctx: CtxRef) -> Option<Space>;
  pub fn isl_space_alloc(ctx: CtxRef, nparam: c_uint, n_in: c_uint, n_out: c_uint) -> Option<Space>;
  pub fn isl_space_set_alloc(ctx: CtxRef, nparam: c_uint, dim: c_uint) -> Option<Space>;
  pub fn isl_space_params_alloc(ctx: CtxRef, nparam: c_uint) -> Option<Space>;
  pub fn isl_space_copy(space: SpaceRef) -> Option<Space>;
  pub fn isl_space_free(space: Space) -> *mut c_void;
  pub fn isl_space_is_params(space: SpaceRef) -> Bool;
  pub fn isl_space_is_set(space: SpaceRef) -> Bool;
  pub fn isl_space_is_map(space: SpaceRef) -> Bool;
  pub fn isl_space_add_param_id(space: Space, id: Id) -> Option<Space>;
  pub fn isl_space_set_tuple_name(space: Space, type_: DimType, s: Option<CStr>) -> Option<Space>;
  pub fn isl_space_has_tuple_name(space: SpaceRef, type_: DimType) -> Bool;
  pub fn isl_space_get_tuple_name(space: SpaceRef, type_: DimType) -> Option<CStr>;
  pub fn isl_space_set_tuple_id(space: Space, type_: DimType, id: Id) -> Option<Space>;
  pub fn isl_space_reset_tuple_id(space: Space, type_: DimType) -> Option<Space>;
  pub fn isl_space_has_tuple_id(space: SpaceRef, type_: DimType) -> Bool;
  pub fn isl_space_get_tuple_id(space: SpaceRef, type_: DimType) -> Option<Id>;
  pub fn isl_space_reset_user(space: Space) -> Option<Space>;
  pub fn isl_space_set_dim_id(space: Space, type_: DimType, pos: c_uint, id: Id) -> Option<Space>;
  pub fn isl_space_has_dim_id(space: SpaceRef, type_: DimType, pos: c_uint) -> Bool;
  pub fn isl_space_get_dim_id(space: SpaceRef, type_: DimType, pos: c_uint) -> Option<Id>;
  pub fn isl_space_find_dim_by_id(space: SpaceRef, type_: DimType, id: IdRef) -> c_int;
  pub fn isl_space_find_dim_by_name(space: SpaceRef, type_: DimType, name: Option<CStr>) -> c_int;
  pub fn isl_space_has_dim_name(space: SpaceRef, type_: DimType, pos: c_uint) -> Bool;
  pub fn isl_space_set_dim_name(space: Space, type_: DimType, pos: c_uint, name: Option<CStr>) -> Option<Space>;
  pub fn isl_space_get_dim_name(space: SpaceRef, type_: DimType, pos: c_uint) -> Option<CStr>;
  pub fn isl_space_extend(space: Space, nparam: c_uint, n_in: c_uint, n_out: c_uint) -> Option<Space>;
  pub fn isl_space_add_dims(space: Space, type_: DimType, n: c_uint) -> Option<Space>;
  pub fn isl_space_move_dims(space: Space, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<Space>;
  pub fn isl_space_insert_dims(space: Space, type_: DimType, pos: c_uint, n: c_uint) -> Option<Space>;
  pub fn isl_space_join(left: Space, right: Space) -> Option<Space>;
  pub fn isl_space_product(left: Space, right: Space) -> Option<Space>;
  pub fn isl_space_domain_product(left: Space, right: Space) -> Option<Space>;
  pub fn isl_space_range_product(left: Space, right: Space) -> Option<Space>;
  pub fn isl_space_factor_domain(space: Space) -> Option<Space>;
  pub fn isl_space_factor_range(space: Space) -> Option<Space>;
  pub fn isl_space_domain_factor_domain(space: Space) -> Option<Space>;
  pub fn isl_space_domain_factor_range(space: Space) -> Option<Space>;
  pub fn isl_space_range_factor_domain(space: Space) -> Option<Space>;
  pub fn isl_space_range_factor_range(space: Space) -> Option<Space>;
  pub fn isl_space_map_from_set(space: Space) -> Option<Space>;
  pub fn isl_space_map_from_domain_and_range(domain: Space, range: Space) -> Option<Space>;
  pub fn isl_space_reverse(space: Space) -> Option<Space>;
  pub fn isl_space_range_reverse(space: Space) -> Option<Space>;
  pub fn isl_space_drop_dims(space: Space, type_: DimType, first: c_uint, num: c_uint) -> Option<Space>;
  pub fn isl_space_drop_inputs(space: Space, first: c_uint, n: c_uint) -> Option<Space>;
  pub fn isl_space_drop_outputs(space: Space, first: c_uint, n: c_uint) -> Option<Space>;
  pub fn isl_space_drop_all_params(space: Space) -> Option<Space>;
  pub fn isl_space_domain(space: Space) -> Option<Space>;
  pub fn isl_space_from_domain(space: Space) -> Option<Space>;
  pub fn isl_space_range(space: Space) -> Option<Space>;
  pub fn isl_space_from_range(space: Space) -> Option<Space>;
  pub fn isl_space_domain_map(space: Space) -> Option<Space>;
  pub fn isl_space_range_map(space: Space) -> Option<Space>;
  pub fn isl_space_params(space: Space) -> Option<Space>;
  pub fn isl_space_add_unnamed_tuple_ui(space: Space, dim: c_uint) -> Option<Space>;
  pub fn isl_space_add_named_tuple_id_ui(space: Space, tuple_id: Id, dim: c_uint) -> Option<Space>;
  pub fn isl_space_set_from_params(space: Space) -> Option<Space>;
  pub fn isl_space_align_params(space1: Space, space2: Space) -> Option<Space>;
  pub fn isl_space_is_wrapping(space: SpaceRef) -> Bool;
  pub fn isl_space_domain_is_wrapping(space: SpaceRef) -> Bool;
  pub fn isl_space_range_is_wrapping(space: SpaceRef) -> Bool;
  pub fn isl_space_is_product(space: SpaceRef) -> Bool;
  pub fn isl_space_wrap(space: Space) -> Option<Space>;
  pub fn isl_space_unwrap(space: Space) -> Option<Space>;
  pub fn isl_space_can_zip(space: SpaceRef) -> Bool;
  pub fn isl_space_zip(space: Space) -> Option<Space>;
  pub fn isl_space_can_curry(space: SpaceRef) -> Bool;
  pub fn isl_space_curry(space: Space) -> Option<Space>;
  pub fn isl_space_can_range_curry(space: SpaceRef) -> Bool;
  pub fn isl_space_range_curry(space: Space) -> Option<Space>;
  pub fn isl_space_can_uncurry(space: SpaceRef) -> Bool;
  pub fn isl_space_uncurry(space: Space) -> Option<Space>;
  pub fn isl_space_is_domain(space1: SpaceRef, space2: SpaceRef) -> Bool;
  pub fn isl_space_is_range(space1: SpaceRef, space2: SpaceRef) -> Bool;
  pub fn isl_space_is_equal(space1: SpaceRef, space2: SpaceRef) -> Bool;
  pub fn isl_space_has_equal_params(space1: SpaceRef, space2: SpaceRef) -> Bool;
  pub fn isl_space_has_equal_tuples(space1: SpaceRef, space2: SpaceRef) -> Bool;
  pub fn isl_space_tuple_is_equal(space1: SpaceRef, type1: DimType, space2: SpaceRef, type2: DimType) -> Bool;
  pub fn isl_space_match(space1: SpaceRef, type1: DimType, space2: SpaceRef, type2: DimType) -> Bool;
  pub fn isl_space_dim(space: SpaceRef, type_: DimType) -> c_int;
  pub fn isl_space_flatten_domain(space: Space) -> Option<Space>;
  pub fn isl_space_flatten_range(space: Space) -> Option<Space>;
  pub fn isl_space_to_str(space: SpaceRef) -> Option<CString>;
  pub fn isl_printer_print_space(p: Printer, space: SpaceRef) -> Option<Printer>;
  pub fn isl_space_dump(space: SpaceRef) -> ();
}

impl CtxRef {
  #[inline(always)]
  pub fn space_unit(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_unit(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn space_alloc(self, nparam: c_uint, n_in: c_uint, n_out: c_uint) -> Option<Space> {
    unsafe {
      let ret = isl_space_alloc(self.to(), nparam.to(), n_in.to(), n_out.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn space_set_alloc(self, nparam: c_uint, dim: c_uint) -> Option<Space> {
    unsafe {
      let ret = isl_space_set_alloc(self.to(), nparam.to(), dim.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn space_params_alloc(self, nparam: c_uint) -> Option<Space> {
    unsafe {
      let ret = isl_space_params_alloc(self.to(), nparam.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_space(self, space: SpaceRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_space(self.to(), space.to());
      (ret).to()
    }
  }
}

impl Space {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_space_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_param_id(self, id: Id) -> Option<Space> {
    unsafe {
      let ret = isl_space_add_param_id(self.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_name(self, type_: DimType, s: Option<CStr>) -> Option<Space> {
    unsafe {
      let ret = isl_space_set_tuple_name(self.to(), type_.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_tuple_id(self, type_: DimType, id: Id) -> Option<Space> {
    unsafe {
      let ret = isl_space_set_tuple_id(self.to(), type_.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_tuple_id(self, type_: DimType) -> Option<Space> {
    unsafe {
      let ret = isl_space_reset_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_id(self, type_: DimType, pos: c_uint, id: Id) -> Option<Space> {
    unsafe {
      let ret = isl_space_set_dim_id(self.to(), type_.to(), pos.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_name(self, type_: DimType, pos: c_uint, name: Option<CStr>) -> Option<Space> {
    unsafe {
      let ret = isl_space_set_dim_name(self.to(), type_.to(), pos.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn extend(self, nparam: c_uint, n_in: c_uint, n_out: c_uint) -> Option<Space> {
    unsafe {
      let ret = isl_space_extend(self.to(), nparam.to(), n_in.to(), n_out.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_dims(self, type_: DimType, n: c_uint) -> Option<Space> {
    unsafe {
      let ret = isl_space_add_dims(self.to(), type_.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn move_dims(self, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<Space> {
    unsafe {
      let ret = isl_space_move_dims(self.to(), dst_type.to(), dst_pos.to(), src_type.to(), src_pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_dims(self, type_: DimType, pos: c_uint, n: c_uint) -> Option<Space> {
    unsafe {
      let ret = isl_space_insert_dims(self.to(), type_.to(), pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn join(self, right: Space) -> Option<Space> {
    unsafe {
      let ret = isl_space_join(self.to(), right.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn product(self, right: Space) -> Option<Space> {
    unsafe {
      let ret = isl_space_product(self.to(), right.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain_product(self, right: Space) -> Option<Space> {
    unsafe {
      let ret = isl_space_domain_product(self.to(), right.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_product(self, right: Space) -> Option<Space> {
    unsafe {
      let ret = isl_space_range_product(self.to(), right.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn factor_domain(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_factor_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn factor_range(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain_factor_domain(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_domain_factor_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain_factor_range(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_domain_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_factor_domain(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_range_factor_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_factor_range(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_range_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_from_set(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_map_from_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_from_domain_and_range(self, range: Space) -> Option<Space> {
    unsafe {
      let ret = isl_space_map_from_domain_and_range(self.to(), range.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reverse(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_reverse(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_reverse(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_range_reverse(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_dims(self, type_: DimType, first: c_uint, num: c_uint) -> Option<Space> {
    unsafe {
      let ret = isl_space_drop_dims(self.to(), type_.to(), first.to(), num.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_inputs(self, first: c_uint, n: c_uint) -> Option<Space> {
    unsafe {
      let ret = isl_space_drop_inputs(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_outputs(self, first: c_uint, n: c_uint) -> Option<Space> {
    unsafe {
      let ret = isl_space_drop_outputs(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_all_params(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_drop_all_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn from_domain(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_from_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn from_range(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_from_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain_map(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_domain_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_map(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_range_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn params(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_unnamed_tuple_ui(self, dim: c_uint) -> Option<Space> {
    unsafe {
      let ret = isl_space_add_unnamed_tuple_ui(self.to(), dim.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_named_tuple_id_ui(self, tuple_id: Id, dim: c_uint) -> Option<Space> {
    unsafe {
      let ret = isl_space_add_named_tuple_id_ui(self.to(), tuple_id.to(), dim.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_from_params(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_set_from_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, space2: Space) -> Option<Space> {
    unsafe {
      let ret = isl_space_align_params(self.to(), space2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn wrap(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_wrap(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn unwrap(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_unwrap(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn zip(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_zip(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn curry(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_curry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_curry(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_range_curry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn uncurry(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_uncurry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten_domain(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_flatten_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten_range(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_flatten_range(self.to());
      (ret).to()
    }
  }
}

impl SpaceRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_space_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<Space> {
    unsafe {
      let ret = isl_space_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_params(self) -> Bool {
    unsafe {
      let ret = isl_space_is_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_set(self) -> Bool {
    unsafe {
      let ret = isl_space_is_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_map(self) -> Bool {
    unsafe {
      let ret = isl_space_is_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_tuple_name(self, type_: DimType) -> Bool {
    unsafe {
      let ret = isl_space_has_tuple_name(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_name(self, type_: DimType) -> Option<CStr> {
    unsafe {
      let ret = isl_space_get_tuple_name(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_tuple_id(self, type_: DimType) -> Bool {
    unsafe {
      let ret = isl_space_has_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tuple_id(self, type_: DimType) -> Option<Id> {
    unsafe {
      let ret = isl_space_get_tuple_id(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_dim_id(self, type_: DimType, pos: c_uint) -> Bool {
    unsafe {
      let ret = isl_space_has_dim_id(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_id(self, type_: DimType, pos: c_uint) -> Option<Id> {
    unsafe {
      let ret = isl_space_get_dim_id(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_id(self, type_: DimType, id: IdRef) -> c_int {
    unsafe {
      let ret = isl_space_find_dim_by_id(self.to(), type_.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: Option<CStr>) -> c_int {
    unsafe {
      let ret = isl_space_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_dim_name(self, type_: DimType, pos: c_uint) -> Bool {
    unsafe {
      let ret = isl_space_has_dim_name(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_dim_name(self, type_: DimType, pos: c_uint) -> Option<CStr> {
    unsafe {
      let ret = isl_space_get_dim_name(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_wrapping(self) -> Bool {
    unsafe {
      let ret = isl_space_is_wrapping(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain_is_wrapping(self) -> Bool {
    unsafe {
      let ret = isl_space_domain_is_wrapping(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_is_wrapping(self) -> Bool {
    unsafe {
      let ret = isl_space_range_is_wrapping(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_product(self) -> Bool {
    unsafe {
      let ret = isl_space_is_product(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn can_zip(self) -> Bool {
    unsafe {
      let ret = isl_space_can_zip(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn can_curry(self) -> Bool {
    unsafe {
      let ret = isl_space_can_curry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn can_range_curry(self) -> Bool {
    unsafe {
      let ret = isl_space_can_range_curry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn can_uncurry(self) -> Bool {
    unsafe {
      let ret = isl_space_can_uncurry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_domain(self, space2: SpaceRef) -> Bool {
    unsafe {
      let ret = isl_space_is_domain(self.to(), space2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_range(self, space2: SpaceRef) -> Bool {
    unsafe {
      let ret = isl_space_is_range(self.to(), space2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_equal(self, space2: SpaceRef) -> Bool {
    unsafe {
      let ret = isl_space_is_equal(self.to(), space2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_equal_params(self, space2: SpaceRef) -> Bool {
    unsafe {
      let ret = isl_space_has_equal_params(self.to(), space2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_equal_tuples(self, space2: SpaceRef) -> Bool {
    unsafe {
      let ret = isl_space_has_equal_tuples(self.to(), space2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn tuple_is_equal(self, type1: DimType, space2: SpaceRef, type2: DimType) -> Bool {
    unsafe {
      let ret = isl_space_tuple_is_equal(self.to(), type1.to(), space2.to(), type2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn match_(self, type1: DimType, space2: SpaceRef, type2: DimType) -> Bool {
    unsafe {
      let ret = isl_space_match(self.to(), type1.to(), space2.to(), type2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_int {
    unsafe {
      let ret = isl_space_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_space_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_space_dump(self.to());
      (ret).to()
    }
  }
}

impl Drop for Space {
  fn drop(&mut self) { Space(self.0).free() }
}

impl fmt::Display for SpaceRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for Space {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

