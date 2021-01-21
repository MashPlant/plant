use crate::*;

extern "C" {
  pub fn isl_qpolynomial_get_ctx(qp: QpolynomialRef) -> Option<CtxRef>;
  pub fn isl_qpolynomial_get_domain_space(qp: QpolynomialRef) -> Option<Space>;
  pub fn isl_qpolynomial_get_space(qp: QpolynomialRef) -> Option<Space>;
  pub fn isl_qpolynomial_dim(qp: QpolynomialRef, type_: DimType) -> c_int;
  pub fn isl_qpolynomial_involves_dims(qp: QpolynomialRef, type_: DimType, first: c_uint, n: c_uint) -> Bool;
  pub fn isl_qpolynomial_get_constant_val(qp: QpolynomialRef) -> Option<Val>;
  pub fn isl_qpolynomial_set_dim_name(qp: Qpolynomial, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_zero_on_domain(domain: Space) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_one_on_domain(domain: Space) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_infty_on_domain(domain: Space) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_neginfty_on_domain(domain: Space) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_nan_on_domain(domain: Space) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_val_on_domain(space: Space, val: Val) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_var_on_domain(domain: Space, type_: DimType, pos: c_uint) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_copy(qp: QpolynomialRef) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_free(qp: Qpolynomial) -> *mut c_void;
  pub fn isl_qpolynomial_plain_is_equal(qp1: QpolynomialRef, qp2: QpolynomialRef) -> Bool;
  pub fn isl_qpolynomial_is_zero(qp: QpolynomialRef) -> Bool;
  pub fn isl_qpolynomial_is_nan(qp: QpolynomialRef) -> Bool;
  pub fn isl_qpolynomial_is_infty(qp: QpolynomialRef) -> Bool;
  pub fn isl_qpolynomial_is_neginfty(qp: QpolynomialRef) -> Bool;
  pub fn isl_qpolynomial_sgn(qp: QpolynomialRef) -> c_int;
  pub fn isl_qpolynomial_neg(qp: Qpolynomial) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_add(qp1: Qpolynomial, qp2: Qpolynomial) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_sub(qp1: Qpolynomial, qp2: Qpolynomial) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_mul(qp1: Qpolynomial, qp2: Qpolynomial) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_pow(qp: Qpolynomial, power: c_uint) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_scale_val(qp: Qpolynomial, v: Val) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_scale_down_val(qp: Qpolynomial, v: Val) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_insert_dims(qp: Qpolynomial, type_: DimType, first: c_uint, n: c_uint) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_add_dims(qp: Qpolynomial, type_: DimType, n: c_uint) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_move_dims(qp: Qpolynomial, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_project_domain_on_params(qp: Qpolynomial) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_drop_dims(qp: Qpolynomial, type_: DimType, first: c_uint, n: c_uint) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_substitute(qp: Qpolynomial, type_: DimType, first: c_uint, n: c_uint, subs: *mut Qpolynomial) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_as_polynomial_on_domain(qp: QpolynomialRef, bset: BasicSetRef, fn_: unsafe extern "C" fn(bset: BasicSet, poly: Qpolynomial, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_qpolynomial_homogenize(poly: Qpolynomial) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_align_params(qp: Qpolynomial, model: Space) -> Option<Qpolynomial>;
  pub fn isl_term_get_ctx(term: TermRef) -> Option<CtxRef>;
  pub fn isl_term_copy(term: TermRef) -> Option<Term>;
  pub fn isl_term_free(term: Term) -> *mut c_void;
  pub fn isl_term_dim(term: TermRef, type_: DimType) -> c_int;
  pub fn isl_term_get_coefficient_val(term: TermRef) -> Option<Val>;
  pub fn isl_term_get_exp(term: TermRef, type_: DimType, pos: c_uint) -> c_int;
  pub fn isl_term_get_div(term: TermRef, pos: c_uint) -> Option<Aff>;
  pub fn isl_qpolynomial_foreach_term(qp: QpolynomialRef, fn_: unsafe extern "C" fn(term: Term, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_qpolynomial_eval(qp: Qpolynomial, pnt: Point) -> Option<Val>;
  pub fn isl_qpolynomial_gist_params(qp: Qpolynomial, context: Set) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_gist(qp: Qpolynomial, context: Set) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_from_constraint(c: Constraint, type_: DimType, pos: c_uint) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_from_term(term: Term) -> Option<Qpolynomial>;
  pub fn isl_qpolynomial_from_aff(aff: Aff) -> Option<Qpolynomial>;
  pub fn isl_basic_map_from_qpolynomial(qp: Qpolynomial) -> Option<BasicMap>;
  pub fn isl_printer_print_qpolynomial(p: Printer, qp: QpolynomialRef) -> Option<Printer>;
  pub fn isl_qpolynomial_print(qp: QpolynomialRef, out: *mut FILE, output_format: c_uint) -> ();
  pub fn isl_qpolynomial_dump(qp: QpolynomialRef) -> ();
  pub fn isl_pw_qpolynomial_get_ctx(pwqp: PwQpolynomialRef) -> Option<CtxRef>;
  pub fn isl_pw_qpolynomial_involves_nan(pwqp: PwQpolynomialRef) -> Bool;
  pub fn isl_pw_qpolynomial_plain_is_equal(pwqp1: PwQpolynomialRef, pwqp2: PwQpolynomialRef) -> Bool;
  pub fn isl_pw_qpolynomial_zero(space: Space) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_alloc(set: Set, qp: Qpolynomial) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_from_qpolynomial(qp: Qpolynomial) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_copy(pwqp: PwQpolynomialRef) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_free(pwqp: PwQpolynomial) -> *mut c_void;
  pub fn isl_pw_qpolynomial_is_zero(pwqp: PwQpolynomialRef) -> Bool;
  pub fn isl_pw_qpolynomial_get_domain_space(pwqp: PwQpolynomialRef) -> Option<Space>;
  pub fn isl_pw_qpolynomial_get_space(pwqp: PwQpolynomialRef) -> Option<Space>;
  pub fn isl_pw_qpolynomial_reset_domain_space(pwqp: PwQpolynomial, space: Space) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_dim(pwqp: PwQpolynomialRef, type_: DimType) -> c_int;
  pub fn isl_pw_qpolynomial_involves_param_id(pwqp: PwQpolynomialRef, id: IdRef) -> Bool;
  pub fn isl_pw_qpolynomial_involves_dims(pwqp: PwQpolynomialRef, type_: DimType, first: c_uint, n: c_uint) -> Bool;
  pub fn isl_pw_qpolynomial_has_equal_space(pwqp1: PwQpolynomialRef, pwqp2: PwQpolynomialRef) -> Bool;
  pub fn isl_pw_qpolynomial_set_dim_name(pwqp: PwQpolynomial, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_find_dim_by_name(pwqp: PwQpolynomialRef, type_: DimType, name: Option<CStr>) -> c_int;
  pub fn isl_pw_qpolynomial_reset_user(pwqp: PwQpolynomial) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_domain(pwqp: PwQpolynomial) -> Option<Set>;
  pub fn isl_pw_qpolynomial_intersect_domain(pwpq: PwQpolynomial, set: Set) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_intersect_domain_wrapped_domain(pwpq: PwQpolynomial, set: Set) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_intersect_domain_wrapped_range(pwpq: PwQpolynomial, set: Set) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_intersect_params(pwpq: PwQpolynomial, set: Set) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_subtract_domain(pwpq: PwQpolynomial, set: Set) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_project_domain_on_params(pwqp: PwQpolynomial) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_from_range(pwqp: PwQpolynomial) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_drop_dims(pwqp: PwQpolynomial, type_: DimType, first: c_uint, n: c_uint) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_split_dims(pwqp: PwQpolynomial, type_: DimType, first: c_uint, n: c_uint) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_drop_unused_params(pwqp: PwQpolynomial) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_add(pwqp1: PwQpolynomial, pwqp2: PwQpolynomial) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_sub(pwqp1: PwQpolynomial, pwqp2: PwQpolynomial) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_add_disjoint(pwqp1: PwQpolynomial, pwqp2: PwQpolynomial) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_neg(pwqp: PwQpolynomial) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_mul(pwqp1: PwQpolynomial, pwqp2: PwQpolynomial) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_scale_val(pwqp: PwQpolynomial, v: Val) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_scale_down_val(pwqp: PwQpolynomial, v: Val) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_pow(pwqp: PwQpolynomial, exponent: c_uint) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_insert_dims(pwqp: PwQpolynomial, type_: DimType, first: c_uint, n: c_uint) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_add_dims(pwqp: PwQpolynomial, type_: DimType, n: c_uint) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_move_dims(pwqp: PwQpolynomial, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_fix_val(pwqp: PwQpolynomial, type_: DimType, n: c_uint, v: Val) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_eval(pwqp: PwQpolynomial, pnt: Point) -> Option<Val>;
  pub fn isl_pw_qpolynomial_max(pwqp: PwQpolynomial) -> Option<Val>;
  pub fn isl_pw_qpolynomial_min(pwqp: PwQpolynomial) -> Option<Val>;
  pub fn isl_pw_qpolynomial_n_piece(pwqp: PwQpolynomialRef) -> c_int;
  pub fn isl_pw_qpolynomial_foreach_piece(pwqp: PwQpolynomialRef, fn_: unsafe extern "C" fn(set: Set, qp: Qpolynomial, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_pw_qpolynomial_every_piece(pwqp: PwQpolynomialRef, test: unsafe extern "C" fn(set: SetRef, qp: QpolynomialRef, user: *mut c_void) -> Bool, user: *mut c_void) -> Bool;
  pub fn isl_pw_qpolynomial_foreach_lifted_piece(pwqp: PwQpolynomialRef, fn_: unsafe extern "C" fn(set: Set, qp: Qpolynomial, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_pw_qpolynomial_isa_qpolynomial(pwqp: PwQpolynomialRef) -> Bool;
  pub fn isl_pw_qpolynomial_as_qpolynomial(pwqp: PwQpolynomial) -> Option<Qpolynomial>;
  pub fn isl_pw_qpolynomial_from_pw_aff(pwaff: PwAff) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_read_from_str(ctx: CtxRef, str: Option<CStr>) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_read_from_file(ctx: CtxRef, input: *mut FILE) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_to_str(pwqp: PwQpolynomialRef) -> Option<CString>;
  pub fn isl_printer_print_pw_qpolynomial(p: Printer, pwqp: PwQpolynomialRef) -> Option<Printer>;
  pub fn isl_pw_qpolynomial_print(pwqp: PwQpolynomialRef, out: *mut FILE, output_format: c_uint) -> ();
  pub fn isl_pw_qpolynomial_dump(pwqp: PwQpolynomialRef) -> ();
  pub fn isl_pw_qpolynomial_coalesce(pwqp: PwQpolynomial) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_gist(pwqp: PwQpolynomial, context: Set) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_gist_params(pwqp: PwQpolynomial, context: Set) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_split_periods(pwqp: PwQpolynomial, max_periods: c_int) -> Option<PwQpolynomial>;
  pub fn isl_basic_set_multiplicative_call(bset: BasicSet, fn_: unsafe extern "C" fn(bset: BasicSet) -> Option<PwQpolynomial>) -> Option<PwQpolynomial>;
  pub fn isl_qpolynomial_fold_get_ctx(fold: QpolynomialFoldRef) -> Option<CtxRef>;
  pub fn isl_qpolynomial_fold_get_type(fold: QpolynomialFoldRef) -> Fold;
  pub fn isl_qpolynomial_fold_empty(type_: Fold, space: Space) -> Option<QpolynomialFold>;
  pub fn isl_qpolynomial_fold_alloc(type_: Fold, qp: Qpolynomial) -> Option<QpolynomialFold>;
  pub fn isl_qpolynomial_fold_copy(fold: QpolynomialFoldRef) -> Option<QpolynomialFold>;
  pub fn isl_qpolynomial_fold_free(fold: QpolynomialFold) -> *mut c_void;
  pub fn isl_qpolynomial_fold_is_empty(fold: QpolynomialFoldRef) -> Bool;
  pub fn isl_qpolynomial_fold_is_nan(fold: QpolynomialFoldRef) -> Bool;
  pub fn isl_qpolynomial_fold_plain_is_equal(fold1: QpolynomialFoldRef, fold2: QpolynomialFoldRef) -> c_int;
  pub fn isl_qpolynomial_fold_get_domain_space(fold: QpolynomialFoldRef) -> Option<Space>;
  pub fn isl_qpolynomial_fold_get_space(fold: QpolynomialFoldRef) -> Option<Space>;
  pub fn isl_qpolynomial_fold_fold(fold1: QpolynomialFold, fold2: QpolynomialFold) -> Option<QpolynomialFold>;
  pub fn isl_qpolynomial_fold_scale_val(fold: QpolynomialFold, v: Val) -> Option<QpolynomialFold>;
  pub fn isl_qpolynomial_fold_scale_down_val(fold: QpolynomialFold, v: Val) -> Option<QpolynomialFold>;
  pub fn isl_qpolynomial_fold_move_dims(fold: QpolynomialFold, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<QpolynomialFold>;
  pub fn isl_qpolynomial_fold_substitute(fold: QpolynomialFold, type_: DimType, first: c_uint, n: c_uint, subs: *mut Qpolynomial) -> Option<QpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_fix_val(pwf: PwQpolynomialFold, type_: DimType, n: c_uint, v: Val) -> Option<PwQpolynomialFold>;
  pub fn isl_qpolynomial_fold_eval(fold: QpolynomialFold, pnt: Point) -> Option<Val>;
  pub fn isl_qpolynomial_fold_gist_params(fold: QpolynomialFold, context: Set) -> Option<QpolynomialFold>;
  pub fn isl_qpolynomial_fold_gist(fold: QpolynomialFold, context: Set) -> Option<QpolynomialFold>;
  pub fn isl_qpolynomial_fold_foreach_qpolynomial(fold: QpolynomialFoldRef, fn_: unsafe extern "C" fn(qp: Qpolynomial, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_printer_print_qpolynomial_fold(p: Printer, fold: QpolynomialFoldRef) -> Option<Printer>;
  pub fn isl_qpolynomial_fold_print(fold: QpolynomialFoldRef, out: *mut FILE, output_format: c_uint) -> ();
  pub fn isl_qpolynomial_fold_dump(fold: QpolynomialFoldRef) -> ();
  pub fn isl_pw_qpolynomial_fold_get_ctx(pwf: PwQpolynomialFoldRef) -> Option<CtxRef>;
  pub fn isl_pw_qpolynomial_fold_get_type(pwf: PwQpolynomialFoldRef) -> Fold;
  pub fn isl_pw_qpolynomial_fold_involves_nan(pwf: PwQpolynomialFoldRef) -> Bool;
  pub fn isl_pw_qpolynomial_fold_plain_is_equal(pwf1: PwQpolynomialFoldRef, pwf2: PwQpolynomialFoldRef) -> Bool;
  pub fn isl_pw_qpolynomial_fold_from_pw_qpolynomial(type_: Fold, pwqp: PwQpolynomial) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_alloc(type_: Fold, set: Set, fold: QpolynomialFold) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_from_qpolynomial_fold(fold: QpolynomialFold) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_copy(pwf: PwQpolynomialFoldRef) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_free(pwf: PwQpolynomialFold) -> *mut c_void;
  pub fn isl_pw_qpolynomial_fold_is_zero(pwf: PwQpolynomialFoldRef) -> Bool;
  pub fn isl_pw_qpolynomial_fold_get_domain_space(pwf: PwQpolynomialFoldRef) -> Option<Space>;
  pub fn isl_pw_qpolynomial_fold_get_space(pwf: PwQpolynomialFoldRef) -> Option<Space>;
  pub fn isl_pw_qpolynomial_fold_reset_space(pwf: PwQpolynomialFold, space: Space) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_dim(pwf: PwQpolynomialFoldRef, type_: DimType) -> c_int;
  pub fn isl_pw_qpolynomial_fold_involves_param_id(pwf: PwQpolynomialFoldRef, id: IdRef) -> Bool;
  pub fn isl_pw_qpolynomial_fold_has_equal_space(pwf1: PwQpolynomialFoldRef, pwf2: PwQpolynomialFoldRef) -> Bool;
  pub fn isl_pw_qpolynomial_fold_size(pwf: PwQpolynomialFoldRef) -> c_int;
  pub fn isl_pw_qpolynomial_fold_zero(space: Space, type_: Fold) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_set_dim_name(pwf: PwQpolynomialFold, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_find_dim_by_name(pwf: PwQpolynomialFoldRef, type_: DimType, name: Option<CStr>) -> c_int;
  pub fn isl_pw_qpolynomial_fold_reset_user(pwf: PwQpolynomialFold) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_domain(pwf: PwQpolynomialFold) -> Option<Set>;
  pub fn isl_pw_qpolynomial_fold_intersect_domain(pwf: PwQpolynomialFold, set: Set) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_intersect_domain_wrapped_domain(pwf: PwQpolynomialFold, set: Set) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_intersect_domain_wrapped_range(pwf: PwQpolynomialFold, set: Set) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_intersect_params(pwf: PwQpolynomialFold, set: Set) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_subtract_domain(pwf: PwQpolynomialFold, set: Set) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_add(pwf1: PwQpolynomialFold, pwf2: PwQpolynomialFold) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_fold(pwf1: PwQpolynomialFold, pwf2: PwQpolynomialFold) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_add_disjoint(pwf1: PwQpolynomialFold, pwf2: PwQpolynomialFold) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_scale_val(pwf: PwQpolynomialFold, v: Val) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_scale_down_val(pwf: PwQpolynomialFold, v: Val) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_project_domain_on_params(pwf: PwQpolynomialFold) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_from_range(pwf: PwQpolynomialFold) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_drop_dims(pwf: PwQpolynomialFold, type_: DimType, first: c_uint, n: c_uint) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_move_dims(pwf: PwQpolynomialFold, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_drop_unused_params(pwf: PwQpolynomialFold) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_eval(pwf: PwQpolynomialFold, pnt: Point) -> Option<Val>;
  pub fn isl_pw_qpolynomial_fold_n_piece(pwf: PwQpolynomialFoldRef) -> c_int;
  pub fn isl_pw_qpolynomial_fold_foreach_piece(pwf: PwQpolynomialFoldRef, fn_: unsafe extern "C" fn(set: Set, fold: QpolynomialFold, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_pw_qpolynomial_fold_every_piece(pwf: PwQpolynomialFoldRef, test: unsafe extern "C" fn(set: SetRef, fold: QpolynomialFoldRef, user: *mut c_void) -> Bool, user: *mut c_void) -> Bool;
  pub fn isl_pw_qpolynomial_fold_foreach_lifted_piece(pwf: PwQpolynomialFoldRef, fn_: unsafe extern "C" fn(set: Set, fold: QpolynomialFold, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_pw_qpolynomial_fold_isa_qpolynomial_fold(pwf: PwQpolynomialFoldRef) -> Bool;
  pub fn isl_pw_qpolynomial_fold_as_qpolynomial_fold(pwf: PwQpolynomialFold) -> Option<QpolynomialFold>;
  pub fn isl_printer_print_pw_qpolynomial_fold(p: Printer, pwf: PwQpolynomialFoldRef) -> Option<Printer>;
  pub fn isl_pw_qpolynomial_fold_print(pwf: PwQpolynomialFoldRef, out: *mut FILE, output_format: c_uint) -> ();
  pub fn isl_pw_qpolynomial_fold_dump(pwf: PwQpolynomialFoldRef) -> ();
  pub fn isl_pw_qpolynomial_fold_coalesce(pwf: PwQpolynomialFold) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_gist(pwf: PwQpolynomialFold, context: Set) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_gist_params(pwf: PwQpolynomialFold, context: Set) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_max(pwf: PwQpolynomialFold) -> Option<Val>;
  pub fn isl_pw_qpolynomial_fold_min(pwf: PwQpolynomialFold) -> Option<Val>;
  pub fn isl_pw_qpolynomial_bound(pwqp: PwQpolynomial, type_: Fold, tight: *mut Bool) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_bound(pwf: PwQpolynomialFold, tight: *mut Bool) -> Option<PwQpolynomialFold>;
  pub fn isl_set_apply_pw_qpolynomial_fold(set: Set, pwf: PwQpolynomialFold, tight: *mut Bool) -> Option<PwQpolynomialFold>;
  pub fn isl_map_apply_pw_qpolynomial_fold(map: Map, pwf: PwQpolynomialFold, tight: *mut Bool) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_to_polynomial(pwqp: PwQpolynomial, sign: c_int) -> Option<PwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_get_ctx(upwqp: UnionPwQpolynomialRef) -> Option<CtxRef>;
  pub fn isl_union_pw_qpolynomial_dim(upwqp: UnionPwQpolynomialRef, type_: DimType) -> c_int;
  pub fn isl_union_pw_qpolynomial_involves_nan(upwqp: UnionPwQpolynomialRef) -> Bool;
  pub fn isl_union_pw_qpolynomial_plain_is_equal(upwqp1: UnionPwQpolynomialRef, upwqp2: UnionPwQpolynomialRef) -> Bool;
  pub fn isl_union_pw_qpolynomial_from_pw_qpolynomial(pwqp: PwQpolynomial) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_zero_ctx(ctx: CtxRef) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_zero_space(space: Space) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_zero(space: Space) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_add_pw_qpolynomial(upwqp: UnionPwQpolynomial, pwqp: PwQpolynomial) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_copy(upwqp: UnionPwQpolynomialRef) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_free(upwqp: UnionPwQpolynomial) -> *mut c_void;
  pub fn isl_union_pw_qpolynomial_read_from_str(ctx: CtxRef, str: Option<CStr>) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_to_str(upwqp: UnionPwQpolynomialRef) -> Option<CString>;
  pub fn isl_union_pw_qpolynomial_neg(upwqp: UnionPwQpolynomial) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_add(upwqp1: UnionPwQpolynomial, upwqp2: UnionPwQpolynomial) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_sub(upwqp1: UnionPwQpolynomial, upwqp2: UnionPwQpolynomial) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_mul(upwqp1: UnionPwQpolynomial, upwqp2: UnionPwQpolynomial) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_scale_val(upwqp: UnionPwQpolynomial, v: Val) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_scale_down_val(upwqp: UnionPwQpolynomial, v: Val) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_domain(upwqp: UnionPwQpolynomial) -> Option<UnionSet>;
  pub fn isl_union_pw_qpolynomial_intersect_domain_space(upwpq: UnionPwQpolynomial, space: Space) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_intersect_domain_union_set(upwpq: UnionPwQpolynomial, uset: UnionSet) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_intersect_domain(upwpq: UnionPwQpolynomial, uset: UnionSet) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_intersect_domain_wrapped_domain(upwpq: UnionPwQpolynomial, uset: UnionSet) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_intersect_domain_wrapped_range(upwpq: UnionPwQpolynomial, uset: UnionSet) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_intersect_params(upwpq: UnionPwQpolynomial, set: Set) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_subtract_domain_union_set(upwpq: UnionPwQpolynomial, uset: UnionSet) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_subtract_domain_space(upwpq: UnionPwQpolynomial, space: Space) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_subtract_domain(upwpq: UnionPwQpolynomial, uset: UnionSet) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_get_space(upwqp: UnionPwQpolynomialRef) -> Option<Space>;
  pub fn isl_union_pw_qpolynomial_get_pw_qpolynomial_list(upwqp: UnionPwQpolynomialRef) -> Option<PwQpolynomialList>;
  pub fn isl_union_pw_qpolynomial_set_dim_name(upwqp: UnionPwQpolynomial, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_find_dim_by_name(upwqp: UnionPwQpolynomialRef, type_: DimType, name: Option<CStr>) -> c_int;
  pub fn isl_union_pw_qpolynomial_drop_dims(upwqp: UnionPwQpolynomial, type_: DimType, first: c_uint, n: c_uint) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_reset_user(upwqp: UnionPwQpolynomial) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_eval(upwqp: UnionPwQpolynomial, pnt: Point) -> Option<Val>;
  pub fn isl_union_pw_qpolynomial_coalesce(upwqp: UnionPwQpolynomial) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_gist(upwqp: UnionPwQpolynomial, context: UnionSet) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_gist_params(upwqp: UnionPwQpolynomial, context: Set) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_align_params(upwqp: UnionPwQpolynomial, model: Space) -> Option<UnionPwQpolynomial>;
  pub fn isl_union_pw_qpolynomial_n_pw_qpolynomial(upwqp: UnionPwQpolynomialRef) -> c_int;
  pub fn isl_union_pw_qpolynomial_foreach_pw_qpolynomial(upwqp: UnionPwQpolynomialRef, fn_: unsafe extern "C" fn(pwqp: PwQpolynomial, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_union_pw_qpolynomial_every_pw_qpolynomial(upwqp: UnionPwQpolynomialRef, test: unsafe extern "C" fn(pwqp: PwQpolynomialRef, user: *mut c_void) -> Bool, user: *mut c_void) -> Bool;
  pub fn isl_union_pw_qpolynomial_extract_pw_qpolynomial(upwqp: UnionPwQpolynomialRef, space: Space) -> Option<PwQpolynomial>;
  pub fn isl_printer_print_union_pw_qpolynomial(p: Printer, upwqp: UnionPwQpolynomialRef) -> Option<Printer>;
  pub fn isl_union_pw_qpolynomial_fold_get_ctx(upwf: UnionPwQpolynomialFoldRef) -> Option<CtxRef>;
  pub fn isl_union_pw_qpolynomial_fold_dim(upwf: UnionPwQpolynomialFoldRef, type_: DimType) -> c_int;
  pub fn isl_union_pw_qpolynomial_fold_involves_nan(upwf: UnionPwQpolynomialFoldRef) -> Bool;
  pub fn isl_union_pw_qpolynomial_fold_plain_is_equal(upwf1: UnionPwQpolynomialFoldRef, upwf2: UnionPwQpolynomialFoldRef) -> Bool;
  pub fn isl_union_pw_qpolynomial_fold_from_pw_qpolynomial_fold(pwf: PwQpolynomialFold) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_zero_ctx(ctx: CtxRef, type_: Fold) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_zero_space(space: Space, type_: Fold) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_zero(space: Space, type_: Fold) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_fold_pw_qpolynomial_fold(upwqp: UnionPwQpolynomialFold, pwqp: PwQpolynomialFold) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_free(upwf: UnionPwQpolynomialFold) -> *mut c_void;
  pub fn isl_union_pw_qpolynomial_fold_copy(upwf: UnionPwQpolynomialFoldRef) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_fold(upwf1: UnionPwQpolynomialFold, upwf2: UnionPwQpolynomialFold) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_add_union_pw_qpolynomial(upwf: UnionPwQpolynomialFold, upwqp: UnionPwQpolynomial) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_scale_val(upwf: UnionPwQpolynomialFold, v: Val) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_scale_down_val(upwf: UnionPwQpolynomialFold, v: Val) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_domain(upwf: UnionPwQpolynomialFold) -> Option<UnionSet>;
  pub fn isl_union_pw_qpolynomial_fold_intersect_domain_space(upwf: UnionPwQpolynomialFold, space: Space) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_intersect_domain_union_set(upwf: UnionPwQpolynomialFold, uset: UnionSet) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_intersect_domain(upwf: UnionPwQpolynomialFold, uset: UnionSet) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_intersect_domain_wrapped_domain(upwf: UnionPwQpolynomialFold, uset: UnionSet) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_intersect_domain_wrapped_range(upwf: UnionPwQpolynomialFold, uset: UnionSet) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_intersect_params(upwf: UnionPwQpolynomialFold, set: Set) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_subtract_domain_union_set(upwf: UnionPwQpolynomialFold, uset: UnionSet) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_subtract_domain_space(upwf: UnionPwQpolynomialFold, space: Space) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_subtract_domain(upwf: UnionPwQpolynomialFold, uset: UnionSet) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_get_type(upwf: UnionPwQpolynomialFoldRef) -> Fold;
  pub fn isl_union_pw_qpolynomial_fold_get_space(upwf: UnionPwQpolynomialFoldRef) -> Option<Space>;
  pub fn isl_union_pw_qpolynomial_fold_get_pw_qpolynomial_fold_list(upwf: UnionPwQpolynomialFoldRef) -> Option<PwQpolynomialFoldList>;
  pub fn isl_union_pw_qpolynomial_fold_set_dim_name(upwf: UnionPwQpolynomialFold, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_find_dim_by_name(upwf: UnionPwQpolynomialFoldRef, type_: DimType, name: Option<CStr>) -> c_int;
  pub fn isl_union_pw_qpolynomial_fold_drop_dims(upwf: UnionPwQpolynomialFold, type_: DimType, first: c_uint, n: c_uint) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_reset_user(upwf: UnionPwQpolynomialFold) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_eval(upwf: UnionPwQpolynomialFold, pnt: Point) -> Option<Val>;
  pub fn isl_union_pw_qpolynomial_fold_coalesce(upwf: UnionPwQpolynomialFold) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_gist(upwf: UnionPwQpolynomialFold, context: UnionSet) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_gist_params(upwf: UnionPwQpolynomialFold, context: Set) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_align_params(upwf: UnionPwQpolynomialFold, model: Space) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_fold_n_pw_qpolynomial_fold(upwf: UnionPwQpolynomialFoldRef) -> c_int;
  pub fn isl_union_pw_qpolynomial_fold_foreach_pw_qpolynomial_fold(upwf: UnionPwQpolynomialFoldRef, fn_: unsafe extern "C" fn(pwf: PwQpolynomialFold, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_union_pw_qpolynomial_fold_every_pw_qpolynomial_fold(upwf: UnionPwQpolynomialFoldRef, test: unsafe extern "C" fn(pwf: PwQpolynomialFoldRef, user: *mut c_void) -> Bool, user: *mut c_void) -> Bool;
  pub fn isl_union_pw_qpolynomial_fold_extract_pw_qpolynomial_fold(upwf: UnionPwQpolynomialFoldRef, space: Space) -> Option<PwQpolynomialFold>;
  pub fn isl_printer_print_union_pw_qpolynomial_fold(p: Printer, upwf: UnionPwQpolynomialFoldRef) -> Option<Printer>;
  pub fn isl_union_pw_qpolynomial_bound(upwqp: UnionPwQpolynomial, type_: Fold, tight: *mut Bool) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_set_apply_union_pw_qpolynomial_fold(uset: UnionSet, upwf: UnionPwQpolynomialFold, tight: *mut Bool) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_map_apply_union_pw_qpolynomial_fold(umap: UnionMap, upwf: UnionPwQpolynomialFold, tight: *mut Bool) -> Option<UnionPwQpolynomialFold>;
  pub fn isl_union_pw_qpolynomial_to_polynomial(upwqp: UnionPwQpolynomial, sign: c_int) -> Option<UnionPwQpolynomial>;
  pub fn isl_pw_qpolynomial_list_get_ctx(list: PwQpolynomialListRef) -> Option<CtxRef>;
  pub fn isl_pw_qpolynomial_list_from_pw_qpolynomial(el: PwQpolynomial) -> Option<PwQpolynomialList>;
  pub fn isl_pw_qpolynomial_list_alloc(ctx: CtxRef, n: c_int) -> Option<PwQpolynomialList>;
  pub fn isl_pw_qpolynomial_list_copy(list: PwQpolynomialListRef) -> Option<PwQpolynomialList>;
  pub fn isl_pw_qpolynomial_list_free(list: PwQpolynomialList) -> *mut c_void;
  pub fn isl_pw_qpolynomial_list_add(list: PwQpolynomialList, el: PwQpolynomial) -> Option<PwQpolynomialList>;
  pub fn isl_pw_qpolynomial_list_insert(list: PwQpolynomialList, pos: c_uint, el: PwQpolynomial) -> Option<PwQpolynomialList>;
  pub fn isl_pw_qpolynomial_list_drop(list: PwQpolynomialList, first: c_uint, n: c_uint) -> Option<PwQpolynomialList>;
  pub fn isl_pw_qpolynomial_list_clear(list: PwQpolynomialList) -> Option<PwQpolynomialList>;
  pub fn isl_pw_qpolynomial_list_swap(list: PwQpolynomialList, pos1: c_uint, pos2: c_uint) -> Option<PwQpolynomialList>;
  pub fn isl_pw_qpolynomial_list_reverse(list: PwQpolynomialList) -> Option<PwQpolynomialList>;
  pub fn isl_pw_qpolynomial_list_concat(list1: PwQpolynomialList, list2: PwQpolynomialList) -> Option<PwQpolynomialList>;
  pub fn isl_pw_qpolynomial_list_size(list: PwQpolynomialListRef) -> c_int;
  pub fn isl_pw_qpolynomial_list_n_pw_qpolynomial(list: PwQpolynomialListRef) -> c_int;
  pub fn isl_pw_qpolynomial_list_get_at(list: PwQpolynomialListRef, index: c_int) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_list_get_pw_qpolynomial(list: PwQpolynomialListRef, index: c_int) -> Option<PwQpolynomial>;
  pub fn isl_pw_qpolynomial_list_set_pw_qpolynomial(list: PwQpolynomialList, index: c_int, el: PwQpolynomial) -> Option<PwQpolynomialList>;
  pub fn isl_pw_qpolynomial_list_foreach(list: PwQpolynomialListRef, fn_: unsafe extern "C" fn(el: PwQpolynomial, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_pw_qpolynomial_list_every(list: PwQpolynomialListRef, test: unsafe extern "C" fn(el: PwQpolynomialRef, user: *mut c_void) -> Bool, user: *mut c_void) -> Bool;
  pub fn isl_pw_qpolynomial_list_map(list: PwQpolynomialList, fn_: unsafe extern "C" fn(el: PwQpolynomial, user: *mut c_void) -> Option<PwQpolynomial>, user: *mut c_void) -> Option<PwQpolynomialList>;
  pub fn isl_pw_qpolynomial_list_sort(list: PwQpolynomialList, cmp: unsafe extern "C" fn(a: PwQpolynomialRef, b: PwQpolynomialRef, user: *mut c_void) -> c_int, user: *mut c_void) -> Option<PwQpolynomialList>;
  pub fn isl_pw_qpolynomial_list_foreach_scc(list: PwQpolynomialListRef, follows: unsafe extern "C" fn(a: PwQpolynomialRef, b: PwQpolynomialRef, user: *mut c_void) -> Bool, follows_user: *mut c_void, fn_: unsafe extern "C" fn(scc: PwQpolynomialList, user: *mut c_void) -> Stat, fn_user: *mut c_void) -> Stat;
  pub fn isl_pw_qpolynomial_list_to_str(list: PwQpolynomialListRef) -> Option<CString>;
  pub fn isl_printer_print_pw_qpolynomial_list(p: Printer, list: PwQpolynomialListRef) -> Option<Printer>;
  pub fn isl_pw_qpolynomial_list_dump(list: PwQpolynomialListRef) -> ();
  pub fn isl_pw_qpolynomial_fold_list_get_ctx(list: PwQpolynomialFoldListRef) -> Option<CtxRef>;
  pub fn isl_pw_qpolynomial_fold_list_from_pw_qpolynomial_fold(el: PwQpolynomialFold) -> Option<PwQpolynomialFoldList>;
  pub fn isl_pw_qpolynomial_fold_list_alloc(ctx: CtxRef, n: c_int) -> Option<PwQpolynomialFoldList>;
  pub fn isl_pw_qpolynomial_fold_list_copy(list: PwQpolynomialFoldListRef) -> Option<PwQpolynomialFoldList>;
  pub fn isl_pw_qpolynomial_fold_list_free(list: PwQpolynomialFoldList) -> *mut c_void;
  pub fn isl_pw_qpolynomial_fold_list_add(list: PwQpolynomialFoldList, el: PwQpolynomialFold) -> Option<PwQpolynomialFoldList>;
  pub fn isl_pw_qpolynomial_fold_list_insert(list: PwQpolynomialFoldList, pos: c_uint, el: PwQpolynomialFold) -> Option<PwQpolynomialFoldList>;
  pub fn isl_pw_qpolynomial_fold_list_drop(list: PwQpolynomialFoldList, first: c_uint, n: c_uint) -> Option<PwQpolynomialFoldList>;
  pub fn isl_pw_qpolynomial_fold_list_clear(list: PwQpolynomialFoldList) -> Option<PwQpolynomialFoldList>;
  pub fn isl_pw_qpolynomial_fold_list_swap(list: PwQpolynomialFoldList, pos1: c_uint, pos2: c_uint) -> Option<PwQpolynomialFoldList>;
  pub fn isl_pw_qpolynomial_fold_list_reverse(list: PwQpolynomialFoldList) -> Option<PwQpolynomialFoldList>;
  pub fn isl_pw_qpolynomial_fold_list_concat(list1: PwQpolynomialFoldList, list2: PwQpolynomialFoldList) -> Option<PwQpolynomialFoldList>;
  pub fn isl_pw_qpolynomial_fold_list_size(list: PwQpolynomialFoldListRef) -> c_int;
  pub fn isl_pw_qpolynomial_fold_list_n_pw_qpolynomial_fold(list: PwQpolynomialFoldListRef) -> c_int;
  pub fn isl_pw_qpolynomial_fold_list_get_at(list: PwQpolynomialFoldListRef, index: c_int) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_list_get_pw_qpolynomial_fold(list: PwQpolynomialFoldListRef, index: c_int) -> Option<PwQpolynomialFold>;
  pub fn isl_pw_qpolynomial_fold_list_set_pw_qpolynomial_fold(list: PwQpolynomialFoldList, index: c_int, el: PwQpolynomialFold) -> Option<PwQpolynomialFoldList>;
  pub fn isl_pw_qpolynomial_fold_list_foreach(list: PwQpolynomialFoldListRef, fn_: unsafe extern "C" fn(el: PwQpolynomialFold, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_pw_qpolynomial_fold_list_every(list: PwQpolynomialFoldListRef, test: unsafe extern "C" fn(el: PwQpolynomialFoldRef, user: *mut c_void) -> Bool, user: *mut c_void) -> Bool;
  pub fn isl_pw_qpolynomial_fold_list_map(list: PwQpolynomialFoldList, fn_: unsafe extern "C" fn(el: PwQpolynomialFold, user: *mut c_void) -> Option<PwQpolynomialFold>, user: *mut c_void) -> Option<PwQpolynomialFoldList>;
  pub fn isl_pw_qpolynomial_fold_list_sort(list: PwQpolynomialFoldList, cmp: unsafe extern "C" fn(a: PwQpolynomialFoldRef, b: PwQpolynomialFoldRef, user: *mut c_void) -> c_int, user: *mut c_void) -> Option<PwQpolynomialFoldList>;
  pub fn isl_pw_qpolynomial_fold_list_foreach_scc(list: PwQpolynomialFoldListRef, follows: unsafe extern "C" fn(a: PwQpolynomialFoldRef, b: PwQpolynomialFoldRef, user: *mut c_void) -> Bool, follows_user: *mut c_void, fn_: unsafe extern "C" fn(scc: PwQpolynomialFoldList, user: *mut c_void) -> Stat, fn_user: *mut c_void) -> Stat;
  pub fn isl_pw_qpolynomial_fold_list_to_str(list: PwQpolynomialFoldListRef) -> Option<CString>;
  pub fn isl_printer_print_pw_qpolynomial_fold_list(p: Printer, list: PwQpolynomialFoldListRef) -> Option<Printer>;
  pub fn isl_pw_qpolynomial_fold_list_dump(list: PwQpolynomialFoldListRef) -> ();
}

impl Aff {
  #[inline(always)]
  pub fn qpolynomial_from_aff(self) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_from_aff(self.to());
      (ret).to()
    }
  }
}

impl BasicSet {
  #[inline(always)]
  pub fn multiplicative_call(self, fn_: unsafe extern "C" fn(bset: BasicSet) -> Option<PwQpolynomial>) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_basic_set_multiplicative_call(self.to(), fn_.to());
      (ret).to()
    }
  }
}

impl Constraint {
  #[inline(always)]
  pub fn qpolynomial_from_constraint(self, type_: DimType, pos: c_uint) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_from_constraint(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
}

impl CtxRef {
  #[inline(always)]
  pub fn pw_qpolynomial_read_from_str(self, str: Option<CStr>) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_qpolynomial_read_from_file(self, input: *mut FILE) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_read_from_file(self.to(), input.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_qpolynomial_zero_ctx(self) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_zero_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_qpolynomial_read_from_str(self, str: Option<CStr>) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_qpolynomial_fold_zero_ctx(self, type_: Fold) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_zero_ctx(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_qpolynomial_list_alloc(self, n: c_int) -> Option<PwQpolynomialList> {
    unsafe {
      let ret = isl_pw_qpolynomial_list_alloc(self.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_qpolynomial_fold_list_alloc(self, n: c_int) -> Option<PwQpolynomialFoldList> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_alloc(self.to(), n.to());
      (ret).to()
    }
  }
}

impl Map {
  #[inline(always)]
  pub fn apply_pw_qpolynomial_fold(self, pwf: PwQpolynomialFold, tight: &mut Bool) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_map_apply_pw_qpolynomial_fold(self.to(), pwf.to(), tight.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_qpolynomial(self, qp: QpolynomialRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_qpolynomial(self.to(), qp.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_pw_qpolynomial(self, pwqp: PwQpolynomialRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_pw_qpolynomial(self.to(), pwqp.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_qpolynomial_fold(self, fold: QpolynomialFoldRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_qpolynomial_fold(self.to(), fold.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_pw_qpolynomial_fold(self, pwf: PwQpolynomialFoldRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_pw_qpolynomial_fold(self.to(), pwf.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_union_pw_qpolynomial(self, upwqp: UnionPwQpolynomialRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_union_pw_qpolynomial(self.to(), upwqp.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_union_pw_qpolynomial_fold(self, upwf: UnionPwQpolynomialFoldRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_union_pw_qpolynomial_fold(self.to(), upwf.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_pw_qpolynomial_list(self, list: PwQpolynomialListRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_pw_qpolynomial_list(self.to(), list.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_pw_qpolynomial_fold_list(self, list: PwQpolynomialFoldListRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_pw_qpolynomial_fold_list(self.to(), list.to());
      (ret).to()
    }
  }
}

impl PwAff {
  #[inline(always)]
  pub fn pw_qpolynomial_from_pw_aff(self) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_from_pw_aff(self.to());
      (ret).to()
    }
  }
}

impl PwQpolynomial {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_pw_qpolynomial_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_domain_space(self, space: Space) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_reset_domain_space(self.to(), space.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_name(self, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_set_dim_name(self.to(), type_.to(), pos.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain(self) -> Option<Set> {
    unsafe {
      let ret = isl_pw_qpolynomial_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain(self, set: Set) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_intersect_domain(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain_wrapped_domain(self, set: Set) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_intersect_domain_wrapped_domain(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain_wrapped_range(self, set: Set) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_intersect_domain_wrapped_range(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_params(self, set: Set) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_intersect_params(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract_domain(self, set: Set) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_subtract_domain(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn project_domain_on_params(self) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_project_domain_on_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn from_range(self) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_from_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_drop_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn split_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_split_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_unused_params(self) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_drop_unused_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, pwqp2: PwQpolynomial) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_add(self.to(), pwqp2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sub(self, pwqp2: PwQpolynomial) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_sub(self.to(), pwqp2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_disjoint(self, pwqp2: PwQpolynomial) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_add_disjoint(self.to(), pwqp2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg(self) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mul(self, pwqp2: PwQpolynomial) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_mul(self.to(), pwqp2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_val(self, v: Val) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_scale_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_val(self, v: Val) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_scale_down_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pow(self, exponent: c_uint) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_pow(self.to(), exponent.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_insert_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_dims(self, type_: DimType, n: c_uint) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_add_dims(self.to(), type_.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn move_dims(self, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_move_dims(self.to(), dst_type.to(), dst_pos.to(), src_type.to(), src_pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn fix_val(self, type_: DimType, n: c_uint, v: Val) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_fix_val(self.to(), type_.to(), n.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eval(self, pnt: Point) -> Option<Val> {
    unsafe {
      let ret = isl_pw_qpolynomial_eval(self.to(), pnt.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn max(self) -> Option<Val> {
    unsafe {
      let ret = isl_pw_qpolynomial_max(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn min(self) -> Option<Val> {
    unsafe {
      let ret = isl_pw_qpolynomial_min(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn as_qpolynomial(self) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_as_qpolynomial(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coalesce(self) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_coalesce(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, context: Set) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_gist(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_params(self, context: Set) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_gist_params(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn split_periods(self, max_periods: c_int) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_split_periods(self.to(), max_periods.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn bound(self, type_: Fold, tight: &mut Bool) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_bound(self.to(), type_.to(), tight.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_polynomial(self, sign: c_int) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_to_polynomial(self.to(), sign.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_qpolynomial_from_pw_qpolynomial(self) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_from_pw_qpolynomial(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn list_from_pw_qpolynomial(self) -> Option<PwQpolynomialList> {
    unsafe {
      let ret = isl_pw_qpolynomial_list_from_pw_qpolynomial(self.to());
      (ret).to()
    }
  }
}

impl PwQpolynomialFold {
  #[inline(always)]
  pub fn fix_val(self, type_: DimType, n: c_uint, v: Val) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_fix_val(self.to(), type_.to(), n.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_space(self, space: Space) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_reset_space(self.to(), space.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_name(self, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_set_dim_name(self.to(), type_.to(), pos.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain(self) -> Option<Set> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain(self, set: Set) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_intersect_domain(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain_wrapped_domain(self, set: Set) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_intersect_domain_wrapped_domain(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain_wrapped_range(self, set: Set) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_intersect_domain_wrapped_range(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_params(self, set: Set) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_intersect_params(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract_domain(self, set: Set) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_subtract_domain(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, pwf2: PwQpolynomialFold) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_add(self.to(), pwf2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn fold(self, pwf2: PwQpolynomialFold) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_fold(self.to(), pwf2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_disjoint(self, pwf2: PwQpolynomialFold) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_add_disjoint(self.to(), pwf2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_val(self, v: Val) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_scale_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_val(self, v: Val) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_scale_down_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn project_domain_on_params(self) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_project_domain_on_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn from_range(self) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_from_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_drop_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn move_dims(self, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_move_dims(self.to(), dst_type.to(), dst_pos.to(), src_type.to(), src_pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_unused_params(self) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_drop_unused_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eval(self, pnt: Point) -> Option<Val> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_eval(self.to(), pnt.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn as_qpolynomial_fold(self) -> Option<QpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_as_qpolynomial_fold(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coalesce(self) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_coalesce(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, context: Set) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_gist(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_params(self, context: Set) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_gist_params(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn max(self) -> Option<Val> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_max(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn min(self) -> Option<Val> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_min(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn bound(self, tight: &mut Bool) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_bound(self.to(), tight.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_qpolynomial_fold_from_pw_qpolynomial_fold(self) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_from_pw_qpolynomial_fold(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn list_from_pw_qpolynomial_fold(self) -> Option<PwQpolynomialFoldList> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_from_pw_qpolynomial_fold(self.to());
      (ret).to()
    }
  }
}

impl PwQpolynomialFoldList {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, el: PwQpolynomialFold) -> Option<PwQpolynomialFoldList> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_add(self.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert(self, pos: c_uint, el: PwQpolynomialFold) -> Option<PwQpolynomialFoldList> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_insert(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, first: c_uint, n: c_uint) -> Option<PwQpolynomialFoldList> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_drop(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn clear(self) -> Option<PwQpolynomialFoldList> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_clear(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn swap(self, pos1: c_uint, pos2: c_uint) -> Option<PwQpolynomialFoldList> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_swap(self.to(), pos1.to(), pos2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reverse(self) -> Option<PwQpolynomialFoldList> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_reverse(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, list2: PwQpolynomialFoldList) -> Option<PwQpolynomialFoldList> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_concat(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_pw_qpolynomial_fold(self, index: c_int, el: PwQpolynomialFold) -> Option<PwQpolynomialFoldList> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_set_pw_qpolynomial_fold(self.to(), index.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map<F1: FnMut(PwQpolynomialFold) -> Option<PwQpolynomialFold>>(self, fn_: &mut F1) -> Option<PwQpolynomialFoldList> {
    unsafe extern "C" fn fn1<F: FnMut(PwQpolynomialFold) -> Option<PwQpolynomialFold>>(el: PwQpolynomialFold, user: *mut c_void) -> Option<PwQpolynomialFold> { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort<F1: FnMut(PwQpolynomialFoldRef, PwQpolynomialFoldRef) -> c_int>(self, cmp: &mut F1) -> Option<PwQpolynomialFoldList> {
    unsafe extern "C" fn fn1<F: FnMut(PwQpolynomialFoldRef, PwQpolynomialFoldRef) -> c_int>(a: PwQpolynomialFoldRef, b: PwQpolynomialFoldRef, user: *mut c_void) -> c_int { (*(user as *mut F))(a.to(), b.to()) }
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_sort(self.to(), fn1::<F1>, cmp as *mut _ as _);
      (ret).to()
    }
  }
}

impl PwQpolynomialFoldListRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<PwQpolynomialFoldList> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn size(self) -> c_int {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_size(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_pw_qpolynomial_fold(self) -> c_int {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_n_pw_qpolynomial_fold(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_at(self, index: c_int) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_get_at(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_pw_qpolynomial_fold(self, index: c_int) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_get_pw_qpolynomial_fold(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(PwQpolynomialFold) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(PwQpolynomialFold) -> Stat>(el: PwQpolynomialFold, user: *mut c_void) -> Stat { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn every<F1: FnMut(PwQpolynomialFoldRef) -> Bool>(self, test: &mut F1) -> Bool {
    unsafe extern "C" fn fn1<F: FnMut(PwQpolynomialFoldRef) -> Bool>(el: PwQpolynomialFoldRef, user: *mut c_void) -> Bool { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_every(self.to(), fn1::<F1>, test as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_scc<F1: FnMut(PwQpolynomialFoldRef, PwQpolynomialFoldRef) -> Bool, F2: FnMut(PwQpolynomialFoldList) -> Stat>(self, follows: &mut F1, fn_: &mut F2) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(PwQpolynomialFoldRef, PwQpolynomialFoldRef) -> Bool>(a: PwQpolynomialFoldRef, b: PwQpolynomialFoldRef, user: *mut c_void) -> Bool { (*(user as *mut F))(a.to(), b.to()) }
    unsafe extern "C" fn fn2<F: FnMut(PwQpolynomialFoldList) -> Stat>(scc: PwQpolynomialFoldList, user: *mut c_void) -> Stat { (*(user as *mut F))(scc.to()) }
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_foreach_scc(self.to(), fn1::<F1>, follows as *mut _ as _, fn2::<F2>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_list_dump(self.to());
      (ret).to()
    }
  }
}

impl PwQpolynomialFoldRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_type(self) -> Fold {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_get_type(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_nan(self) -> Bool {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_involves_nan(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, pwf2: PwQpolynomialFoldRef) -> Bool {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_plain_is_equal(self.to(), pwf2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_zero(self) -> Bool {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_is_zero(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_domain_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_get_domain_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_int {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_param_id(self, id: IdRef) -> Bool {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_involves_param_id(self.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_equal_space(self, pwf2: PwQpolynomialFoldRef) -> Bool {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_has_equal_space(self.to(), pwf2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn size(self) -> c_int {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_size(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: Option<CStr>) -> c_int {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_piece(self) -> c_int {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_n_piece(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_piece<F1: FnMut(Set, QpolynomialFold) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(Set, QpolynomialFold) -> Stat>(set: Set, fold: QpolynomialFold, user: *mut c_void) -> Stat { (*(user as *mut F))(set.to(), fold.to()) }
    unsafe {
      let ret = isl_pw_qpolynomial_fold_foreach_piece(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn every_piece<F1: FnMut(SetRef, QpolynomialFoldRef) -> Bool>(self, test: &mut F1) -> Bool {
    unsafe extern "C" fn fn1<F: FnMut(SetRef, QpolynomialFoldRef) -> Bool>(set: SetRef, fold: QpolynomialFoldRef, user: *mut c_void) -> Bool { (*(user as *mut F))(set.to(), fold.to()) }
    unsafe {
      let ret = isl_pw_qpolynomial_fold_every_piece(self.to(), fn1::<F1>, test as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_lifted_piece<F1: FnMut(Set, QpolynomialFold) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(Set, QpolynomialFold) -> Stat>(set: Set, fold: QpolynomialFold, user: *mut c_void) -> Stat { (*(user as *mut F))(set.to(), fold.to()) }
    unsafe {
      let ret = isl_pw_qpolynomial_fold_foreach_lifted_piece(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn isa_qpolynomial_fold(self) -> Bool {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_isa_qpolynomial_fold(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print(self, out: *mut FILE, output_format: c_uint) -> () {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_print(self.to(), out.to(), output_format.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_dump(self.to());
      (ret).to()
    }
  }
}

impl PwQpolynomialList {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_pw_qpolynomial_list_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, el: PwQpolynomial) -> Option<PwQpolynomialList> {
    unsafe {
      let ret = isl_pw_qpolynomial_list_add(self.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert(self, pos: c_uint, el: PwQpolynomial) -> Option<PwQpolynomialList> {
    unsafe {
      let ret = isl_pw_qpolynomial_list_insert(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, first: c_uint, n: c_uint) -> Option<PwQpolynomialList> {
    unsafe {
      let ret = isl_pw_qpolynomial_list_drop(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn clear(self) -> Option<PwQpolynomialList> {
    unsafe {
      let ret = isl_pw_qpolynomial_list_clear(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn swap(self, pos1: c_uint, pos2: c_uint) -> Option<PwQpolynomialList> {
    unsafe {
      let ret = isl_pw_qpolynomial_list_swap(self.to(), pos1.to(), pos2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reverse(self) -> Option<PwQpolynomialList> {
    unsafe {
      let ret = isl_pw_qpolynomial_list_reverse(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, list2: PwQpolynomialList) -> Option<PwQpolynomialList> {
    unsafe {
      let ret = isl_pw_qpolynomial_list_concat(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_pw_qpolynomial(self, index: c_int, el: PwQpolynomial) -> Option<PwQpolynomialList> {
    unsafe {
      let ret = isl_pw_qpolynomial_list_set_pw_qpolynomial(self.to(), index.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map<F1: FnMut(PwQpolynomial) -> Option<PwQpolynomial>>(self, fn_: &mut F1) -> Option<PwQpolynomialList> {
    unsafe extern "C" fn fn1<F: FnMut(PwQpolynomial) -> Option<PwQpolynomial>>(el: PwQpolynomial, user: *mut c_void) -> Option<PwQpolynomial> { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_pw_qpolynomial_list_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort<F1: FnMut(PwQpolynomialRef, PwQpolynomialRef) -> c_int>(self, cmp: &mut F1) -> Option<PwQpolynomialList> {
    unsafe extern "C" fn fn1<F: FnMut(PwQpolynomialRef, PwQpolynomialRef) -> c_int>(a: PwQpolynomialRef, b: PwQpolynomialRef, user: *mut c_void) -> c_int { (*(user as *mut F))(a.to(), b.to()) }
    unsafe {
      let ret = isl_pw_qpolynomial_list_sort(self.to(), fn1::<F1>, cmp as *mut _ as _);
      (ret).to()
    }
  }
}

impl PwQpolynomialListRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_pw_qpolynomial_list_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<PwQpolynomialList> {
    unsafe {
      let ret = isl_pw_qpolynomial_list_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn size(self) -> c_int {
    unsafe {
      let ret = isl_pw_qpolynomial_list_size(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_pw_qpolynomial(self) -> c_int {
    unsafe {
      let ret = isl_pw_qpolynomial_list_n_pw_qpolynomial(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_at(self, index: c_int) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_list_get_at(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_pw_qpolynomial(self, index: c_int) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_list_get_pw_qpolynomial(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(PwQpolynomial) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(PwQpolynomial) -> Stat>(el: PwQpolynomial, user: *mut c_void) -> Stat { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_pw_qpolynomial_list_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn every<F1: FnMut(PwQpolynomialRef) -> Bool>(self, test: &mut F1) -> Bool {
    unsafe extern "C" fn fn1<F: FnMut(PwQpolynomialRef) -> Bool>(el: PwQpolynomialRef, user: *mut c_void) -> Bool { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_pw_qpolynomial_list_every(self.to(), fn1::<F1>, test as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_scc<F1: FnMut(PwQpolynomialRef, PwQpolynomialRef) -> Bool, F2: FnMut(PwQpolynomialList) -> Stat>(self, follows: &mut F1, fn_: &mut F2) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(PwQpolynomialRef, PwQpolynomialRef) -> Bool>(a: PwQpolynomialRef, b: PwQpolynomialRef, user: *mut c_void) -> Bool { (*(user as *mut F))(a.to(), b.to()) }
    unsafe extern "C" fn fn2<F: FnMut(PwQpolynomialList) -> Stat>(scc: PwQpolynomialList, user: *mut c_void) -> Stat { (*(user as *mut F))(scc.to()) }
    unsafe {
      let ret = isl_pw_qpolynomial_list_foreach_scc(self.to(), fn1::<F1>, follows as *mut _ as _, fn2::<F2>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_pw_qpolynomial_list_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_pw_qpolynomial_list_dump(self.to());
      (ret).to()
    }
  }
}

impl PwQpolynomialRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_pw_qpolynomial_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_nan(self) -> Bool {
    unsafe {
      let ret = isl_pw_qpolynomial_involves_nan(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, pwqp2: PwQpolynomialRef) -> Bool {
    unsafe {
      let ret = isl_pw_qpolynomial_plain_is_equal(self.to(), pwqp2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_zero(self) -> Bool {
    unsafe {
      let ret = isl_pw_qpolynomial_is_zero(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_domain_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_pw_qpolynomial_get_domain_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_pw_qpolynomial_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_int {
    unsafe {
      let ret = isl_pw_qpolynomial_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_param_id(self, id: IdRef) -> Bool {
    unsafe {
      let ret = isl_pw_qpolynomial_involves_param_id(self.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Bool {
    unsafe {
      let ret = isl_pw_qpolynomial_involves_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_equal_space(self, pwqp2: PwQpolynomialRef) -> Bool {
    unsafe {
      let ret = isl_pw_qpolynomial_has_equal_space(self.to(), pwqp2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: Option<CStr>) -> c_int {
    unsafe {
      let ret = isl_pw_qpolynomial_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_piece(self) -> c_int {
    unsafe {
      let ret = isl_pw_qpolynomial_n_piece(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_piece<F1: FnMut(Set, Qpolynomial) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(Set, Qpolynomial) -> Stat>(set: Set, qp: Qpolynomial, user: *mut c_void) -> Stat { (*(user as *mut F))(set.to(), qp.to()) }
    unsafe {
      let ret = isl_pw_qpolynomial_foreach_piece(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn every_piece<F1: FnMut(SetRef, QpolynomialRef) -> Bool>(self, test: &mut F1) -> Bool {
    unsafe extern "C" fn fn1<F: FnMut(SetRef, QpolynomialRef) -> Bool>(set: SetRef, qp: QpolynomialRef, user: *mut c_void) -> Bool { (*(user as *mut F))(set.to(), qp.to()) }
    unsafe {
      let ret = isl_pw_qpolynomial_every_piece(self.to(), fn1::<F1>, test as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_lifted_piece<F1: FnMut(Set, Qpolynomial) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(Set, Qpolynomial) -> Stat>(set: Set, qp: Qpolynomial, user: *mut c_void) -> Stat { (*(user as *mut F))(set.to(), qp.to()) }
    unsafe {
      let ret = isl_pw_qpolynomial_foreach_lifted_piece(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn isa_qpolynomial(self) -> Bool {
    unsafe {
      let ret = isl_pw_qpolynomial_isa_qpolynomial(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_pw_qpolynomial_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print(self, out: *mut FILE, output_format: c_uint) -> () {
    unsafe {
      let ret = isl_pw_qpolynomial_print(self.to(), out.to(), output_format.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_pw_qpolynomial_dump(self.to());
      (ret).to()
    }
  }
}

impl Qpolynomial {
  #[inline(always)]
  pub fn set_dim_name(self, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_set_dim_name(self.to(), type_.to(), pos.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_qpolynomial_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg(self) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, qp2: Qpolynomial) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_add(self.to(), qp2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sub(self, qp2: Qpolynomial) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_sub(self.to(), qp2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mul(self, qp2: Qpolynomial) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_mul(self.to(), qp2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pow(self, power: c_uint) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_pow(self.to(), power.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_val(self, v: Val) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_scale_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_val(self, v: Val) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_scale_down_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_insert_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_dims(self, type_: DimType, n: c_uint) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_add_dims(self.to(), type_.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn move_dims(self, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_move_dims(self.to(), dst_type.to(), dst_pos.to(), src_type.to(), src_pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn project_domain_on_params(self) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_project_domain_on_params(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_drop_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn substitute(self, type_: DimType, first: c_uint, subs: &mut [Qpolynomial]) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_substitute(self.to(), type_.to(), first.to(), subs.to(), subs.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn homogenize(self) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_homogenize(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_align_params(self.to(), model.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eval(self, pnt: Point) -> Option<Val> {
    unsafe {
      let ret = isl_qpolynomial_eval(self.to(), pnt.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_params(self, context: Set) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_gist_params(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, context: Set) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_gist(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn basic_map_from_qpolynomial(self) -> Option<BasicMap> {
    unsafe {
      let ret = isl_basic_map_from_qpolynomial(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_qpolynomial_from_qpolynomial(self) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_from_qpolynomial(self.to());
      (ret).to()
    }
  }
}

impl QpolynomialFold {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_qpolynomial_fold_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn fold(self, fold2: QpolynomialFold) -> Option<QpolynomialFold> {
    unsafe {
      let ret = isl_qpolynomial_fold_fold(self.to(), fold2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_val(self, v: Val) -> Option<QpolynomialFold> {
    unsafe {
      let ret = isl_qpolynomial_fold_scale_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_val(self, v: Val) -> Option<QpolynomialFold> {
    unsafe {
      let ret = isl_qpolynomial_fold_scale_down_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn move_dims(self, dst_type: DimType, dst_pos: c_uint, src_type: DimType, src_pos: c_uint, n: c_uint) -> Option<QpolynomialFold> {
    unsafe {
      let ret = isl_qpolynomial_fold_move_dims(self.to(), dst_type.to(), dst_pos.to(), src_type.to(), src_pos.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn substitute(self, type_: DimType, first: c_uint, subs: &mut [Qpolynomial]) -> Option<QpolynomialFold> {
    unsafe {
      let ret = isl_qpolynomial_fold_substitute(self.to(), type_.to(), first.to(), subs.to(), subs.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eval(self, pnt: Point) -> Option<Val> {
    unsafe {
      let ret = isl_qpolynomial_fold_eval(self.to(), pnt.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_params(self, context: Set) -> Option<QpolynomialFold> {
    unsafe {
      let ret = isl_qpolynomial_fold_gist_params(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, context: Set) -> Option<QpolynomialFold> {
    unsafe {
      let ret = isl_qpolynomial_fold_gist(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_qpolynomial_fold_from_qpolynomial_fold(self) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_from_qpolynomial_fold(self.to());
      (ret).to()
    }
  }
}

impl QpolynomialFoldRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_qpolynomial_fold_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_type(self) -> Fold {
    unsafe {
      let ret = isl_qpolynomial_fold_get_type(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<QpolynomialFold> {
    unsafe {
      let ret = isl_qpolynomial_fold_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_empty(self) -> Bool {
    unsafe {
      let ret = isl_qpolynomial_fold_is_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_nan(self) -> Bool {
    unsafe {
      let ret = isl_qpolynomial_fold_is_nan(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, fold2: QpolynomialFoldRef) -> c_int {
    unsafe {
      let ret = isl_qpolynomial_fold_plain_is_equal(self.to(), fold2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_domain_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_qpolynomial_fold_get_domain_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_qpolynomial_fold_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_qpolynomial<F1: FnMut(Qpolynomial) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(Qpolynomial) -> Stat>(qp: Qpolynomial, user: *mut c_void) -> Stat { (*(user as *mut F))(qp.to()) }
    unsafe {
      let ret = isl_qpolynomial_fold_foreach_qpolynomial(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print(self, out: *mut FILE, output_format: c_uint) -> () {
    unsafe {
      let ret = isl_qpolynomial_fold_print(self.to(), out.to(), output_format.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_qpolynomial_fold_dump(self.to());
      (ret).to()
    }
  }
}

impl QpolynomialRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_qpolynomial_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_domain_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_qpolynomial_get_domain_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_qpolynomial_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_int {
    unsafe {
      let ret = isl_qpolynomial_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Bool {
    unsafe {
      let ret = isl_qpolynomial_involves_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_constant_val(self) -> Option<Val> {
    unsafe {
      let ret = isl_qpolynomial_get_constant_val(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, qp2: QpolynomialRef) -> Bool {
    unsafe {
      let ret = isl_qpolynomial_plain_is_equal(self.to(), qp2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_zero(self) -> Bool {
    unsafe {
      let ret = isl_qpolynomial_is_zero(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_nan(self) -> Bool {
    unsafe {
      let ret = isl_qpolynomial_is_nan(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_infty(self) -> Bool {
    unsafe {
      let ret = isl_qpolynomial_is_infty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_neginfty(self) -> Bool {
    unsafe {
      let ret = isl_qpolynomial_is_neginfty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sgn(self) -> c_int {
    unsafe {
      let ret = isl_qpolynomial_sgn(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn as_polynomial_on_domain<F1: FnMut(BasicSet, Qpolynomial) -> Stat>(self, bset: BasicSetRef, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(BasicSet, Qpolynomial) -> Stat>(bset: BasicSet, poly: Qpolynomial, user: *mut c_void) -> Stat { (*(user as *mut F))(bset.to(), poly.to()) }
    unsafe {
      let ret = isl_qpolynomial_as_polynomial_on_domain(self.to(), bset.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_term<F1: FnMut(Term) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(Term) -> Stat>(term: Term, user: *mut c_void) -> Stat { (*(user as *mut F))(term.to()) }
    unsafe {
      let ret = isl_qpolynomial_foreach_term(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print(self, out: *mut FILE, output_format: c_uint) -> () {
    unsafe {
      let ret = isl_qpolynomial_print(self.to(), out.to(), output_format.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_qpolynomial_dump(self.to());
      (ret).to()
    }
  }
}

impl Set {
  #[inline(always)]
  pub fn pw_qpolynomial_alloc(self, qp: Qpolynomial) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_alloc(self.to(), qp.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn apply_pw_qpolynomial_fold(self, pwf: PwQpolynomialFold, tight: &mut Bool) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_set_apply_pw_qpolynomial_fold(self.to(), pwf.to(), tight.to());
      (ret).to()
    }
  }
}

impl Space {
  #[inline(always)]
  pub fn qpolynomial_zero_on_domain(self) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_zero_on_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn qpolynomial_one_on_domain(self) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_one_on_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn qpolynomial_infty_on_domain(self) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_infty_on_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn qpolynomial_neginfty_on_domain(self) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_neginfty_on_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn qpolynomial_nan_on_domain(self) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_nan_on_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn qpolynomial_val_on_domain(self, val: Val) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_val_on_domain(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn qpolynomial_var_on_domain(self, type_: DimType, pos: c_uint) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_var_on_domain(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_qpolynomial_zero(self) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_pw_qpolynomial_zero(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_qpolynomial_fold_zero(self, type_: Fold) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_pw_qpolynomial_fold_zero(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_qpolynomial_zero_space(self) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_zero_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_qpolynomial_zero(self) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_zero(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_qpolynomial_fold_zero_space(self, type_: Fold) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_zero_space(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn union_pw_qpolynomial_fold_zero(self, type_: Fold) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_zero(self.to(), type_.to());
      (ret).to()
    }
  }
}

impl Term {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_term_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn qpolynomial_from_term(self) -> Option<Qpolynomial> {
    unsafe {
      let ret = isl_qpolynomial_from_term(self.to());
      (ret).to()
    }
  }
}

impl TermRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_term_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<Term> {
    unsafe {
      let ret = isl_term_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_int {
    unsafe {
      let ret = isl_term_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_coefficient_val(self) -> Option<Val> {
    unsafe {
      let ret = isl_term_get_coefficient_val(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_exp(self, type_: DimType, pos: c_uint) -> c_int {
    unsafe {
      let ret = isl_term_get_exp(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_div(self, pos: c_uint) -> Option<Aff> {
    unsafe {
      let ret = isl_term_get_div(self.to(), pos.to());
      (ret).to()
    }
  }
}

impl UnionMap {
  #[inline(always)]
  pub fn apply_union_pw_qpolynomial_fold(self, upwf: UnionPwQpolynomialFold, tight: &mut Bool) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_map_apply_union_pw_qpolynomial_fold(self.to(), upwf.to(), tight.to());
      (ret).to()
    }
  }
}

impl UnionPwQpolynomial {
  #[inline(always)]
  pub fn add_pw_qpolynomial(self, pwqp: PwQpolynomial) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_add_pw_qpolynomial(self.to(), pwqp.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_union_pw_qpolynomial_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn neg(self) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, upwqp2: UnionPwQpolynomial) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_add(self.to(), upwqp2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sub(self, upwqp2: UnionPwQpolynomial) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_sub(self.to(), upwqp2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mul(self, upwqp2: UnionPwQpolynomial) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_mul(self.to(), upwqp2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_val(self, v: Val) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_scale_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_val(self, v: Val) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_scale_down_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain_space(self, space: Space) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_intersect_domain_space(self.to(), space.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain_union_set(self, uset: UnionSet) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_intersect_domain_union_set(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain(self, uset: UnionSet) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_intersect_domain(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain_wrapped_domain(self, uset: UnionSet) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_intersect_domain_wrapped_domain(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain_wrapped_range(self, uset: UnionSet) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_intersect_domain_wrapped_range(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_params(self, set: Set) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_intersect_params(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract_domain_union_set(self, uset: UnionSet) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_subtract_domain_union_set(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract_domain_space(self, space: Space) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_subtract_domain_space(self.to(), space.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract_domain(self, uset: UnionSet) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_subtract_domain(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_name(self, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_set_dim_name(self.to(), type_.to(), pos.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_drop_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eval(self, pnt: Point) -> Option<Val> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_eval(self.to(), pnt.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coalesce(self) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_coalesce(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, context: UnionSet) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_gist(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_params(self, context: Set) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_gist_params(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_align_params(self.to(), model.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn bound(self, type_: Fold, tight: &mut Bool) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_bound(self.to(), type_.to(), tight.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_polynomial(self, sign: c_int) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_to_polynomial(self.to(), sign.to());
      (ret).to()
    }
  }
}

impl UnionPwQpolynomialFold {
  #[inline(always)]
  pub fn fold_pw_qpolynomial_fold(self, pwqp: PwQpolynomialFold) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_fold_pw_qpolynomial_fold(self.to(), pwqp.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn fold(self, upwf2: UnionPwQpolynomialFold) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_fold(self.to(), upwf2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_union_pw_qpolynomial(self, upwqp: UnionPwQpolynomial) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_add_union_pw_qpolynomial(self.to(), upwqp.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_val(self, v: Val) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_scale_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn scale_down_val(self, v: Val) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_scale_down_val(self.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain_space(self, space: Space) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_intersect_domain_space(self.to(), space.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain_union_set(self, uset: UnionSet) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_intersect_domain_union_set(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain(self, uset: UnionSet) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_intersect_domain(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain_wrapped_domain(self, uset: UnionSet) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_intersect_domain_wrapped_domain(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain_wrapped_range(self, uset: UnionSet) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_intersect_domain_wrapped_range(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_params(self, set: Set) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_intersect_params(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract_domain_union_set(self, uset: UnionSet) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_subtract_domain_union_set(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract_domain_space(self, space: Space) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_subtract_domain_space(self.to(), space.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn subtract_domain(self, uset: UnionSet) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_subtract_domain(self.to(), uset.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_dim_name(self, type_: DimType, pos: c_uint, s: Option<CStr>) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_set_dim_name(self.to(), type_.to(), pos.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_dims(self, type_: DimType, first: c_uint, n: c_uint) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_drop_dims(self.to(), type_.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eval(self, pnt: Point) -> Option<Val> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_eval(self.to(), pnt.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn coalesce(self) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_coalesce(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist(self, context: UnionSet) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_gist(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_params(self, context: Set) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_gist_params(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_align_params(self.to(), model.to());
      (ret).to()
    }
  }
}

impl UnionPwQpolynomialFoldRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_int {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_nan(self) -> Bool {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_involves_nan(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, upwf2: UnionPwQpolynomialFoldRef) -> Bool {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_plain_is_equal(self.to(), upwf2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_type(self) -> Fold {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_get_type(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_pw_qpolynomial_fold_list(self) -> Option<PwQpolynomialFoldList> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_get_pw_qpolynomial_fold_list(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: Option<CStr>) -> c_int {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_pw_qpolynomial_fold(self) -> c_int {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_n_pw_qpolynomial_fold(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_pw_qpolynomial_fold<F1: FnMut(PwQpolynomialFold) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(PwQpolynomialFold) -> Stat>(pwf: PwQpolynomialFold, user: *mut c_void) -> Stat { (*(user as *mut F))(pwf.to()) }
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_foreach_pw_qpolynomial_fold(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn every_pw_qpolynomial_fold<F1: FnMut(PwQpolynomialFoldRef) -> Bool>(self, test: &mut F1) -> Bool {
    unsafe extern "C" fn fn1<F: FnMut(PwQpolynomialFoldRef) -> Bool>(pwf: PwQpolynomialFoldRef, user: *mut c_void) -> Bool { (*(user as *mut F))(pwf.to()) }
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_every_pw_qpolynomial_fold(self.to(), fn1::<F1>, test as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn extract_pw_qpolynomial_fold(self, space: Space) -> Option<PwQpolynomialFold> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_fold_extract_pw_qpolynomial_fold(self.to(), space.to());
      (ret).to()
    }
  }
}

impl UnionPwQpolynomialRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim(self, type_: DimType) -> c_int {
    unsafe {
      let ret = isl_union_pw_qpolynomial_dim(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn involves_nan(self) -> Bool {
    unsafe {
      let ret = isl_union_pw_qpolynomial_involves_nan(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, upwqp2: UnionPwQpolynomialRef) -> Bool {
    unsafe {
      let ret = isl_union_pw_qpolynomial_plain_is_equal(self.to(), upwqp2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<UnionPwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_pw_qpolynomial_list(self) -> Option<PwQpolynomialList> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_get_pw_qpolynomial_list(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn find_dim_by_name(self, type_: DimType, name: Option<CStr>) -> c_int {
    unsafe {
      let ret = isl_union_pw_qpolynomial_find_dim_by_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_pw_qpolynomial(self) -> c_int {
    unsafe {
      let ret = isl_union_pw_qpolynomial_n_pw_qpolynomial(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_pw_qpolynomial<F1: FnMut(PwQpolynomial) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(PwQpolynomial) -> Stat>(pwqp: PwQpolynomial, user: *mut c_void) -> Stat { (*(user as *mut F))(pwqp.to()) }
    unsafe {
      let ret = isl_union_pw_qpolynomial_foreach_pw_qpolynomial(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn every_pw_qpolynomial<F1: FnMut(PwQpolynomialRef) -> Bool>(self, test: &mut F1) -> Bool {
    unsafe extern "C" fn fn1<F: FnMut(PwQpolynomialRef) -> Bool>(pwqp: PwQpolynomialRef, user: *mut c_void) -> Bool { (*(user as *mut F))(pwqp.to()) }
    unsafe {
      let ret = isl_union_pw_qpolynomial_every_pw_qpolynomial(self.to(), fn1::<F1>, test as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn extract_pw_qpolynomial(self, space: Space) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_union_pw_qpolynomial_extract_pw_qpolynomial(self.to(), space.to());
      (ret).to()
    }
  }
}

impl UnionSet {
  #[inline(always)]
  pub fn apply_union_pw_qpolynomial_fold(self, upwf: UnionPwQpolynomialFold, tight: &mut Bool) -> Option<UnionPwQpolynomialFold> {
    unsafe {
      let ret = isl_union_set_apply_union_pw_qpolynomial_fold(self.to(), upwf.to(), tight.to());
      (ret).to()
    }
  }
}

impl Drop for PwQpolynomial {
  fn drop(&mut self) { PwQpolynomial(self.0).free() }
}

impl Drop for PwQpolynomialFold {
  fn drop(&mut self) { PwQpolynomialFold(self.0).free() }
}

impl Drop for PwQpolynomialFoldList {
  fn drop(&mut self) { PwQpolynomialFoldList(self.0).free() }
}

impl fmt::Display for PwQpolynomialFoldListRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for PwQpolynomialFoldList {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl Drop for PwQpolynomialList {
  fn drop(&mut self) { PwQpolynomialList(self.0).free() }
}

impl fmt::Display for PwQpolynomialListRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for PwQpolynomialList {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl fmt::Display for PwQpolynomialRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for PwQpolynomial {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl Drop for Qpolynomial {
  fn drop(&mut self) { Qpolynomial(self.0).free() }
}

impl Drop for QpolynomialFold {
  fn drop(&mut self) { QpolynomialFold(self.0).free() }
}

impl Drop for Term {
  fn drop(&mut self) { Term(self.0).free() }
}

impl Drop for UnionPwQpolynomial {
  fn drop(&mut self) { UnionPwQpolynomial(self.0).free() }
}

impl Drop for UnionPwQpolynomialFold {
  fn drop(&mut self) { UnionPwQpolynomialFold(self.0).free() }
}

impl fmt::Display for UnionPwQpolynomialRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for UnionPwQpolynomial {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

