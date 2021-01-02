use crate::*;

extern "C" {
  pub fn isl_options_set_schedule_max_coefficient(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_schedule_max_coefficient(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_schedule_max_constant_term(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_schedule_max_constant_term(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_schedule_maximize_band_depth(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_schedule_maximize_band_depth(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_schedule_maximize_coincidence(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_schedule_maximize_coincidence(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_schedule_outer_coincidence(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_schedule_outer_coincidence(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_schedule_split_scaled(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_schedule_split_scaled(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_schedule_treat_coalescing(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_schedule_treat_coalescing(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_schedule_separate_components(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_schedule_separate_components(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_schedule_serialize_sccs(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_schedule_serialize_sccs(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_schedule_whole_component(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_schedule_whole_component(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_schedule_carry_self_first(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_schedule_carry_self_first(ctx: CtxRef) -> c_int;
  pub fn isl_schedule_constraints_copy(sc: ScheduleConstraintsRef) -> Option<ScheduleConstraints>;
  pub fn isl_schedule_constraints_on_domain(domain: UnionSet) -> Option<ScheduleConstraints>;
  pub fn isl_schedule_constraints_set_context(sc: ScheduleConstraints, context: Set) -> Option<ScheduleConstraints>;
  pub fn isl_schedule_constraints_set_validity(sc: ScheduleConstraints, validity: UnionMap) -> Option<ScheduleConstraints>;
  pub fn isl_schedule_constraints_set_coincidence(sc: ScheduleConstraints, coincidence: UnionMap) -> Option<ScheduleConstraints>;
  pub fn isl_schedule_constraints_set_proximity(sc: ScheduleConstraints, proximity: UnionMap) -> Option<ScheduleConstraints>;
  pub fn isl_schedule_constraints_set_conditional_validity(sc: ScheduleConstraints, condition: UnionMap, validity: UnionMap) -> Option<ScheduleConstraints>;
  pub fn isl_schedule_constraints_free(sc: ScheduleConstraints) -> *mut c_void;
  pub fn isl_schedule_constraints_get_ctx(sc: ScheduleConstraintsRef) -> Option<CtxRef>;
  pub fn isl_schedule_constraints_get_domain(sc: ScheduleConstraintsRef) -> Option<UnionSet>;
  pub fn isl_schedule_constraints_get_context(sc: ScheduleConstraintsRef) -> Option<Set>;
  pub fn isl_schedule_constraints_get_validity(sc: ScheduleConstraintsRef) -> Option<UnionMap>;
  pub fn isl_schedule_constraints_get_coincidence(sc: ScheduleConstraintsRef) -> Option<UnionMap>;
  pub fn isl_schedule_constraints_get_proximity(sc: ScheduleConstraintsRef) -> Option<UnionMap>;
  pub fn isl_schedule_constraints_get_conditional_validity(sc: ScheduleConstraintsRef) -> Option<UnionMap>;
  pub fn isl_schedule_constraints_get_conditional_validity_condition(sc: ScheduleConstraintsRef) -> Option<UnionMap>;
  pub fn isl_schedule_constraints_apply(sc: ScheduleConstraints, umap: UnionMap) -> Option<ScheduleConstraints>;
  pub fn isl_schedule_constraints_read_from_str(ctx: CtxRef, str: CStr) -> Option<ScheduleConstraints>;
  pub fn isl_schedule_constraints_read_from_file(ctx: CtxRef, input: *mut FILE) -> Option<ScheduleConstraints>;
  pub fn isl_printer_print_schedule_constraints(p: Printer, sc: ScheduleConstraintsRef) -> Option<Printer>;
  pub fn isl_schedule_constraints_dump(sc: ScheduleConstraintsRef) -> ();
  pub fn isl_schedule_constraints_to_str(sc: ScheduleConstraintsRef) -> Option<CString>;
  pub fn isl_schedule_constraints_compute_schedule(sc: ScheduleConstraints) -> Option<Schedule>;
  pub fn isl_union_set_compute_schedule(domain: UnionSet, validity: UnionMap, proximity: UnionMap) -> Option<Schedule>;
  pub fn isl_schedule_empty(space: Space) -> Option<Schedule>;
  pub fn isl_schedule_from_domain(domain: UnionSet) -> Option<Schedule>;
  pub fn isl_schedule_copy(sched: ScheduleRef) -> Option<Schedule>;
  pub fn isl_schedule_free(sched: Schedule) -> *mut c_void;
  pub fn isl_schedule_get_map(sched: ScheduleRef) -> Option<UnionMap>;
  pub fn isl_schedule_get_ctx(sched: ScheduleRef) -> Option<CtxRef>;
  pub fn isl_schedule_plain_is_equal(schedule1: ScheduleRef, schedule2: ScheduleRef) -> Bool;
  pub fn isl_schedule_get_root(schedule: ScheduleRef) -> Option<ScheduleNode>;
  pub fn isl_schedule_get_domain(schedule: ScheduleRef) -> Option<UnionSet>;
  pub fn isl_schedule_foreach_schedule_node_top_down(sched: ScheduleRef, fn_: unsafe extern "C" fn(node: ScheduleNodeRef, user: *mut c_void) -> Bool, user: *mut c_void) -> Stat;
  pub fn isl_schedule_map_schedule_node_bottom_up(schedule: Schedule, fn_: unsafe extern "C" fn(node: ScheduleNode, user: *mut c_void) -> Option<ScheduleNode>, user: *mut c_void) -> Option<Schedule>;
  pub fn isl_schedule_insert_context(schedule: Schedule, context: Set) -> Option<Schedule>;
  pub fn isl_schedule_insert_partial_schedule(schedule: Schedule, partial: MultiUnionPwAff) -> Option<Schedule>;
  pub fn isl_schedule_insert_guard(schedule: Schedule, guard: Set) -> Option<Schedule>;
  pub fn isl_schedule_sequence(schedule1: Schedule, schedule2: Schedule) -> Option<Schedule>;
  pub fn isl_schedule_set(schedule1: Schedule, schedule2: Schedule) -> Option<Schedule>;
  pub fn isl_schedule_intersect_domain(schedule: Schedule, domain: UnionSet) -> Option<Schedule>;
  pub fn isl_schedule_gist_domain_params(schedule: Schedule, context: Set) -> Option<Schedule>;
  pub fn isl_schedule_reset_user(schedule: Schedule) -> Option<Schedule>;
  pub fn isl_schedule_align_params(schedule: Schedule, space: Space) -> Option<Schedule>;
  pub fn isl_schedule_pullback_union_pw_multi_aff(schedule: Schedule, upma: UnionPwMultiAff) -> Option<Schedule>;
  pub fn isl_schedule_expand(schedule: Schedule, contraction: UnionPwMultiAff, expansion: Schedule) -> Option<Schedule>;
  pub fn isl_schedule_read_from_file(ctx: CtxRef, input: *mut FILE) -> Option<Schedule>;
  pub fn isl_schedule_read_from_str(ctx: CtxRef, str: CStr) -> Option<Schedule>;
  pub fn isl_printer_print_schedule(p: Printer, schedule: ScheduleRef) -> Option<Printer>;
  pub fn isl_schedule_dump(schedule: ScheduleRef) -> ();
  pub fn isl_schedule_to_str(schedule: ScheduleRef) -> Option<CString>;
}

#[repr(transparent)]
#[derive(Debug)]
pub struct ScheduleConstraints(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct ScheduleConstraintsRef(pub NonNull<c_void>);

impl ScheduleConstraints {
  #[inline(always)]
  pub fn read(&self) -> ScheduleConstraints { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: ScheduleConstraints) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<ScheduleConstraintsRef> for ScheduleConstraints {
  #[inline(always)]
  fn as_ref(&self) -> &ScheduleConstraintsRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for ScheduleConstraints {
  type Target = ScheduleConstraintsRef;
  #[inline(always)]
  fn deref(&self) -> &ScheduleConstraintsRef { self.as_ref() }
}

impl To<Option<ScheduleConstraints>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<ScheduleConstraints> { NonNull::new(self).map(ScheduleConstraints) }
}

impl CtxRef {
  #[inline(always)]
  pub fn options_set_schedule_max_coefficient(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_schedule_max_coefficient(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_schedule_max_coefficient(self) -> c_int {
    unsafe {
      let ret = isl_options_get_schedule_max_coefficient(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_schedule_max_constant_term(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_schedule_max_constant_term(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_schedule_max_constant_term(self) -> c_int {
    unsafe {
      let ret = isl_options_get_schedule_max_constant_term(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_schedule_maximize_band_depth(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_schedule_maximize_band_depth(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_schedule_maximize_band_depth(self) -> c_int {
    unsafe {
      let ret = isl_options_get_schedule_maximize_band_depth(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_schedule_maximize_coincidence(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_schedule_maximize_coincidence(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_schedule_maximize_coincidence(self) -> c_int {
    unsafe {
      let ret = isl_options_get_schedule_maximize_coincidence(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_schedule_outer_coincidence(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_schedule_outer_coincidence(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_schedule_outer_coincidence(self) -> c_int {
    unsafe {
      let ret = isl_options_get_schedule_outer_coincidence(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_schedule_split_scaled(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_schedule_split_scaled(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_schedule_split_scaled(self) -> c_int {
    unsafe {
      let ret = isl_options_get_schedule_split_scaled(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_schedule_treat_coalescing(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_schedule_treat_coalescing(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_schedule_treat_coalescing(self) -> c_int {
    unsafe {
      let ret = isl_options_get_schedule_treat_coalescing(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_schedule_separate_components(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_schedule_separate_components(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_schedule_separate_components(self) -> c_int {
    unsafe {
      let ret = isl_options_get_schedule_separate_components(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_schedule_serialize_sccs(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_schedule_serialize_sccs(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_schedule_serialize_sccs(self) -> c_int {
    unsafe {
      let ret = isl_options_get_schedule_serialize_sccs(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_schedule_whole_component(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_schedule_whole_component(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_schedule_whole_component(self) -> c_int {
    unsafe {
      let ret = isl_options_get_schedule_whole_component(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_schedule_carry_self_first(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_schedule_carry_self_first(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_schedule_carry_self_first(self) -> c_int {
    unsafe {
      let ret = isl_options_get_schedule_carry_self_first(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn schedule_constraints_read_from_str(self, str: CStr) -> Option<ScheduleConstraints> {
    unsafe {
      let ret = isl_schedule_constraints_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn schedule_constraints_read_from_file(self, input: *mut FILE) -> Option<ScheduleConstraints> {
    unsafe {
      let ret = isl_schedule_constraints_read_from_file(self.to(), input.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn schedule_read_from_file(self, input: *mut FILE) -> Option<Schedule> {
    unsafe {
      let ret = isl_schedule_read_from_file(self.to(), input.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn schedule_read_from_str(self, str: CStr) -> Option<Schedule> {
    unsafe {
      let ret = isl_schedule_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_schedule_constraints(self, sc: ScheduleConstraintsRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_schedule_constraints(self.to(), sc.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_schedule(self, schedule: ScheduleRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_schedule(self.to(), schedule.to());
      (ret).to()
    }
  }
}

impl Schedule {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_schedule_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_schedule_node_bottom_up<F1: FnMut(ScheduleNode) -> Option<ScheduleNode>>(self, fn_: &mut F1) -> Option<Schedule> {
    unsafe extern "C" fn fn1<F: FnMut(ScheduleNode) -> Option<ScheduleNode>>(node: ScheduleNode, user: *mut c_void) -> Option<ScheduleNode> { (*(user as *mut F))(node.to()).to() }
    unsafe {
      let ret = isl_schedule_map_schedule_node_bottom_up(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_context(self, context: Set) -> Option<Schedule> {
    unsafe {
      let ret = isl_schedule_insert_context(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_partial_schedule(self, partial: MultiUnionPwAff) -> Option<Schedule> {
    unsafe {
      let ret = isl_schedule_insert_partial_schedule(self.to(), partial.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_guard(self, guard: Set) -> Option<Schedule> {
    unsafe {
      let ret = isl_schedule_insert_guard(self.to(), guard.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sequence(self, schedule2: Schedule) -> Option<Schedule> {
    unsafe {
      let ret = isl_schedule_sequence(self.to(), schedule2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set(self, schedule2: Schedule) -> Option<Schedule> {
    unsafe {
      let ret = isl_schedule_set(self.to(), schedule2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn intersect_domain(self, domain: UnionSet) -> Option<Schedule> {
    unsafe {
      let ret = isl_schedule_intersect_domain(self.to(), domain.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gist_domain_params(self, context: Set) -> Option<Schedule> {
    unsafe {
      let ret = isl_schedule_gist_domain_params(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<Schedule> {
    unsafe {
      let ret = isl_schedule_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, space: Space) -> Option<Schedule> {
    unsafe {
      let ret = isl_schedule_align_params(self.to(), space.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pullback_union_pw_multi_aff(self, upma: UnionPwMultiAff) -> Option<Schedule> {
    unsafe {
      let ret = isl_schedule_pullback_union_pw_multi_aff(self.to(), upma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn expand(self, contraction: UnionPwMultiAff, expansion: Schedule) -> Option<Schedule> {
    unsafe {
      let ret = isl_schedule_expand(self.to(), contraction.to(), expansion.to());
      (ret).to()
    }
  }
}

impl ScheduleConstraints {
  #[inline(always)]
  pub fn set_context(self, context: Set) -> Option<ScheduleConstraints> {
    unsafe {
      let ret = isl_schedule_constraints_set_context(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_validity(self, validity: UnionMap) -> Option<ScheduleConstraints> {
    unsafe {
      let ret = isl_schedule_constraints_set_validity(self.to(), validity.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_coincidence(self, coincidence: UnionMap) -> Option<ScheduleConstraints> {
    unsafe {
      let ret = isl_schedule_constraints_set_coincidence(self.to(), coincidence.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_proximity(self, proximity: UnionMap) -> Option<ScheduleConstraints> {
    unsafe {
      let ret = isl_schedule_constraints_set_proximity(self.to(), proximity.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_conditional_validity(self, condition: UnionMap, validity: UnionMap) -> Option<ScheduleConstraints> {
    unsafe {
      let ret = isl_schedule_constraints_set_conditional_validity(self.to(), condition.to(), validity.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_schedule_constraints_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn apply(self, umap: UnionMap) -> Option<ScheduleConstraints> {
    unsafe {
      let ret = isl_schedule_constraints_apply(self.to(), umap.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn compute_schedule(self) -> Option<Schedule> {
    unsafe {
      let ret = isl_schedule_constraints_compute_schedule(self.to());
      (ret).to()
    }
  }
}

impl ScheduleConstraintsRef {
  #[inline(always)]
  pub fn copy(self) -> Option<ScheduleConstraints> {
    unsafe {
      let ret = isl_schedule_constraints_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_schedule_constraints_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_domain(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_schedule_constraints_get_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_context(self) -> Option<Set> {
    unsafe {
      let ret = isl_schedule_constraints_get_context(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_validity(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_schedule_constraints_get_validity(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_coincidence(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_schedule_constraints_get_coincidence(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_proximity(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_schedule_constraints_get_proximity(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_conditional_validity(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_schedule_constraints_get_conditional_validity(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_conditional_validity_condition(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_schedule_constraints_get_conditional_validity_condition(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_schedule_constraints_dump(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_schedule_constraints_to_str(self.to());
      (ret).to()
    }
  }
}

impl ScheduleRef {
  #[inline(always)]
  pub fn copy(self) -> Option<Schedule> {
    unsafe {
      let ret = isl_schedule_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_map(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_schedule_get_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_schedule_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, schedule2: ScheduleRef) -> Option<bool> {
    unsafe {
      let ret = isl_schedule_plain_is_equal(self.to(), schedule2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_root(self) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_get_root(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_domain(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_schedule_get_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_schedule_node_top_down<F1: FnMut(ScheduleNodeRef) -> Option<bool>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(ScheduleNodeRef) -> Option<bool>>(node: ScheduleNodeRef, user: *mut c_void) -> Bool { (*(user as *mut F))(node.to()).to() }
    unsafe {
      let ret = isl_schedule_foreach_schedule_node_top_down(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_schedule_dump(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_schedule_to_str(self.to());
      (ret).to()
    }
  }
}

impl Space {
  #[inline(always)]
  pub fn schedule_empty(self) -> Option<Schedule> {
    unsafe {
      let ret = isl_schedule_empty(self.to());
      (ret).to()
    }
  }
}

impl UnionSet {
  #[inline(always)]
  pub fn schedule_constraints_on_domain(self) -> Option<ScheduleConstraints> {
    unsafe {
      let ret = isl_schedule_constraints_on_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn compute_schedule(self, validity: UnionMap, proximity: UnionMap) -> Option<Schedule> {
    unsafe {
      let ret = isl_union_set_compute_schedule(self.to(), validity.to(), proximity.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn schedule_from_domain(self) -> Option<Schedule> {
    unsafe {
      let ret = isl_schedule_from_domain(self.to());
      (ret).to()
    }
  }
}

impl Drop for Schedule {
  fn drop(&mut self) { Schedule(self.0).free() }
}

impl Drop for ScheduleConstraints {
  fn drop(&mut self) { ScheduleConstraints(self.0).free() }
}

impl fmt::Display for ScheduleConstraintsRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for ScheduleConstraints {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl fmt::Display for ScheduleRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for Schedule {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

