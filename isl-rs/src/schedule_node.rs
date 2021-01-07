use crate::*;

extern "C" {
  pub fn isl_schedule_node_from_domain(domain: UnionSet) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_from_extension(extension: UnionMap) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_copy(node: ScheduleNodeRef) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_free(node: ScheduleNode) -> *mut c_void;
  pub fn isl_schedule_node_is_equal(node1: ScheduleNodeRef, node2: ScheduleNodeRef) -> Bool;
  pub fn isl_schedule_node_get_ctx(node: ScheduleNodeRef) -> Option<CtxRef>;
  pub fn isl_schedule_node_get_type(node: ScheduleNodeRef) -> ScheduleNodeType;
  pub fn isl_schedule_node_get_parent_type(node: ScheduleNodeRef) -> ScheduleNodeType;
  pub fn isl_schedule_node_get_schedule(node: ScheduleNodeRef) -> Option<Schedule>;
  pub fn isl_schedule_node_foreach_descendant_top_down(node: ScheduleNodeRef, fn_: unsafe extern "C" fn(node: ScheduleNodeRef, user: *mut c_void) -> Bool, user: *mut c_void) -> Stat;
  pub fn isl_schedule_node_every_descendant(node: ScheduleNodeRef, test: unsafe extern "C" fn(node: ScheduleNodeRef, user: *mut c_void) -> Bool, user: *mut c_void) -> Bool;
  pub fn isl_schedule_node_foreach_ancestor_top_down(node: ScheduleNodeRef, fn_: unsafe extern "C" fn(node: ScheduleNodeRef, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_schedule_node_map_descendant_bottom_up(node: ScheduleNode, fn_: unsafe extern "C" fn(node: ScheduleNode, user: *mut c_void) -> Option<ScheduleNode>, user: *mut c_void) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_get_tree_depth(node: ScheduleNodeRef) -> c_int;
  pub fn isl_schedule_node_has_parent(node: ScheduleNodeRef) -> Bool;
  pub fn isl_schedule_node_has_children(node: ScheduleNodeRef) -> Bool;
  pub fn isl_schedule_node_has_previous_sibling(node: ScheduleNodeRef) -> Bool;
  pub fn isl_schedule_node_has_next_sibling(node: ScheduleNodeRef) -> Bool;
  pub fn isl_schedule_node_n_children(node: ScheduleNodeRef) -> c_int;
  pub fn isl_schedule_node_get_child_position(node: ScheduleNodeRef) -> c_int;
  pub fn isl_schedule_node_get_ancestor_child_position(node: ScheduleNodeRef, ancestor: ScheduleNodeRef) -> c_int;
  pub fn isl_schedule_node_get_child(node: ScheduleNodeRef, pos: c_int) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_get_shared_ancestor(node1: ScheduleNodeRef, node2: ScheduleNodeRef) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_root(node: ScheduleNode) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_parent(node: ScheduleNode) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_ancestor(node: ScheduleNode, generation: c_int) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_child(node: ScheduleNode, pos: c_int) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_first_child(node: ScheduleNode) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_previous_sibling(node: ScheduleNode) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_next_sibling(node: ScheduleNode) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_is_subtree_anchored(node: ScheduleNodeRef) -> Bool;
  pub fn isl_schedule_node_group(node: ScheduleNode, group_id: Id) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_sequence_splice_child(node: ScheduleNode, pos: c_int) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_band_get_space(node: ScheduleNodeRef) -> Option<Space>;
  pub fn isl_schedule_node_band_get_partial_schedule(node: ScheduleNodeRef) -> Option<MultiUnionPwAff>;
  pub fn isl_schedule_node_band_get_partial_schedule_union_map(node: ScheduleNodeRef) -> Option<UnionMap>;
  pub fn isl_schedule_node_band_member_get_ast_loop_type(node: ScheduleNodeRef, pos: c_int) -> AstLoopType;
  pub fn isl_schedule_node_band_member_set_ast_loop_type(node: ScheduleNode, pos: c_int, type_: AstLoopType) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_band_member_get_isolate_ast_loop_type(node: ScheduleNodeRef, pos: c_int) -> AstLoopType;
  pub fn isl_schedule_node_band_member_set_isolate_ast_loop_type(node: ScheduleNode, pos: c_int, type_: AstLoopType) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_band_get_ast_build_options(node: ScheduleNodeRef) -> Option<UnionSet>;
  pub fn isl_schedule_node_band_set_ast_build_options(node: ScheduleNode, options: UnionSet) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_band_get_ast_isolate_option(node: ScheduleNodeRef) -> Option<Set>;
  pub fn isl_schedule_node_band_n_member(node: ScheduleNodeRef) -> c_uint;
  pub fn isl_schedule_node_band_member_get_coincident(node: ScheduleNodeRef, pos: c_int) -> Bool;
  pub fn isl_schedule_node_band_member_set_coincident(node: ScheduleNode, pos: c_int, coincident: c_int) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_band_get_permutable(node: ScheduleNodeRef) -> Bool;
  pub fn isl_schedule_node_band_set_permutable(node: ScheduleNode, permutable: c_int) -> Option<ScheduleNode>;
  pub fn isl_options_set_tile_scale_tile_loops(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_tile_scale_tile_loops(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_tile_shift_point_loops(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_tile_shift_point_loops(ctx: CtxRef) -> c_int;
  pub fn isl_schedule_node_band_scale(node: ScheduleNode, mv: MultiVal) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_band_scale_down(node: ScheduleNode, mv: MultiVal) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_band_mod(node: ScheduleNode, mv: MultiVal) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_band_shift(node: ScheduleNode, shift: MultiUnionPwAff) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_band_tile(node: ScheduleNode, sizes: MultiVal) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_band_sink(node: ScheduleNode) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_band_split(node: ScheduleNode, pos: c_int) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_context_get_context(node: ScheduleNodeRef) -> Option<Set>;
  pub fn isl_schedule_node_domain_get_domain(node: ScheduleNodeRef) -> Option<UnionSet>;
  pub fn isl_schedule_node_expansion_get_expansion(node: ScheduleNodeRef) -> Option<UnionMap>;
  pub fn isl_schedule_node_expansion_get_contraction(node: ScheduleNodeRef) -> Option<UnionPwMultiAff>;
  pub fn isl_schedule_node_extension_get_extension(node: ScheduleNodeRef) -> Option<UnionMap>;
  pub fn isl_schedule_node_filter_get_filter(node: ScheduleNodeRef) -> Option<UnionSet>;
  pub fn isl_schedule_node_guard_get_guard(node: ScheduleNodeRef) -> Option<Set>;
  pub fn isl_schedule_node_mark_get_id(node: ScheduleNodeRef) -> Option<Id>;
  pub fn isl_schedule_node_get_schedule_depth(node: ScheduleNodeRef) -> c_int;
  pub fn isl_schedule_node_get_domain(node: ScheduleNodeRef) -> Option<UnionSet>;
  pub fn isl_schedule_node_get_universe_domain(node: ScheduleNodeRef) -> Option<UnionSet>;
  pub fn isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(node: ScheduleNodeRef) -> Option<MultiUnionPwAff>;
  pub fn isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(node: ScheduleNodeRef) -> Option<UnionPwMultiAff>;
  pub fn isl_schedule_node_get_prefix_schedule_union_map(node: ScheduleNodeRef) -> Option<UnionMap>;
  pub fn isl_schedule_node_get_prefix_schedule_relation(node: ScheduleNodeRef) -> Option<UnionMap>;
  pub fn isl_schedule_node_get_subtree_schedule_union_map(node: ScheduleNodeRef) -> Option<UnionMap>;
  pub fn isl_schedule_node_get_subtree_expansion(node: ScheduleNodeRef) -> Option<UnionMap>;
  pub fn isl_schedule_node_get_subtree_contraction(node: ScheduleNodeRef) -> Option<UnionPwMultiAff>;
  pub fn isl_schedule_node_insert_context(node: ScheduleNode, context: Set) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_insert_partial_schedule(node: ScheduleNode, schedule: MultiUnionPwAff) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_insert_filter(node: ScheduleNode, filter: UnionSet) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_insert_guard(node: ScheduleNode, context: Set) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_insert_mark(node: ScheduleNode, mark: Id) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_insert_sequence(node: ScheduleNode, filters: UnionSetList) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_insert_set(node: ScheduleNode, filters: UnionSetList) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_cut(node: ScheduleNode) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_delete(node: ScheduleNode) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_order_before(node: ScheduleNode, filter: UnionSet) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_order_after(node: ScheduleNode, filter: UnionSet) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_graft_before(node: ScheduleNode, graft: ScheduleNode) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_graft_after(node: ScheduleNode, graft: ScheduleNode) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_reset_user(node: ScheduleNode) -> Option<ScheduleNode>;
  pub fn isl_schedule_node_align_params(node: ScheduleNode, space: Space) -> Option<ScheduleNode>;
  pub fn isl_printer_print_schedule_node(p: Printer, node: ScheduleNodeRef) -> Option<Printer>;
  pub fn isl_schedule_node_dump(node: ScheduleNodeRef) -> ();
  pub fn isl_schedule_node_to_str(node: ScheduleNodeRef) -> Option<CString>;
}

impl CtxRef {
  #[inline(always)]
  pub fn options_set_tile_scale_tile_loops(self, val: c_int) -> Stat {
    unsafe {
      let ret = isl_options_set_tile_scale_tile_loops(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_tile_scale_tile_loops(self) -> c_int {
    unsafe {
      let ret = isl_options_get_tile_scale_tile_loops(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_tile_shift_point_loops(self, val: c_int) -> Stat {
    unsafe {
      let ret = isl_options_set_tile_shift_point_loops(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_tile_shift_point_loops(self) -> c_int {
    unsafe {
      let ret = isl_options_get_tile_shift_point_loops(self.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_schedule_node(self, node: ScheduleNodeRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_schedule_node(self.to(), node.to());
      (ret).to()
    }
  }
}

impl ScheduleNode {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_schedule_node_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map_descendant_bottom_up<F1: FnMut(ScheduleNode) -> Option<ScheduleNode>>(self, fn_: &mut F1) -> Option<ScheduleNode> {
    unsafe extern "C" fn fn1<F: FnMut(ScheduleNode) -> Option<ScheduleNode>>(node: ScheduleNode, user: *mut c_void) -> Option<ScheduleNode> { (*(user as *mut F))(node.to()) }
    unsafe {
      let ret = isl_schedule_node_map_descendant_bottom_up(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn root(self) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_root(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn parent(self) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_parent(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ancestor(self, generation: c_int) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_ancestor(self.to(), generation.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn child(self, pos: c_int) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_child(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn first_child(self) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_first_child(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn previous_sibling(self) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_previous_sibling(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn next_sibling(self) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_next_sibling(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn group(self, group_id: Id) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_group(self.to(), group_id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sequence_splice_child(self, pos: c_int) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_sequence_splice_child(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_member_set_ast_loop_type(self, pos: c_int, type_: AstLoopType) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_band_member_set_ast_loop_type(self.to(), pos.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_member_set_isolate_ast_loop_type(self, pos: c_int, type_: AstLoopType) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_band_member_set_isolate_ast_loop_type(self.to(), pos.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_set_ast_build_options(self, options: UnionSet) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_band_set_ast_build_options(self.to(), options.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_member_set_coincident(self, pos: c_int, coincident: c_int) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_band_member_set_coincident(self.to(), pos.to(), coincident.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_set_permutable(self, permutable: c_int) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_band_set_permutable(self.to(), permutable.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_scale(self, mv: MultiVal) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_band_scale(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_scale_down(self, mv: MultiVal) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_band_scale_down(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_mod(self, mv: MultiVal) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_band_mod(self.to(), mv.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_shift(self, shift: MultiUnionPwAff) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_band_shift(self.to(), shift.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_tile(self, sizes: MultiVal) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_band_tile(self.to(), sizes.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_sink(self) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_band_sink(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_split(self, pos: c_int) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_band_split(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_context(self, context: Set) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_insert_context(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_partial_schedule(self, schedule: MultiUnionPwAff) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_insert_partial_schedule(self.to(), schedule.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_filter(self, filter: UnionSet) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_insert_filter(self.to(), filter.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_guard(self, context: Set) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_insert_guard(self.to(), context.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_mark(self, mark: Id) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_insert_mark(self.to(), mark.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_sequence(self, filters: UnionSetList) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_insert_sequence(self.to(), filters.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_set(self, filters: UnionSetList) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_insert_set(self.to(), filters.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn cut(self) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_cut(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn delete(self) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_delete(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn order_before(self, filter: UnionSet) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_order_before(self.to(), filter.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn order_after(self, filter: UnionSet) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_order_after(self.to(), filter.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn graft_before(self, graft: ScheduleNode) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_graft_before(self.to(), graft.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn graft_after(self, graft: ScheduleNode) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_graft_after(self.to(), graft.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, space: Space) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_align_params(self.to(), space.to());
      (ret).to()
    }
  }
}

impl ScheduleNodeRef {
  #[inline(always)]
  pub fn copy(self) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_equal(self, node2: ScheduleNodeRef) -> Bool {
    unsafe {
      let ret = isl_schedule_node_is_equal(self.to(), node2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_schedule_node_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_type(self) -> ScheduleNodeType {
    unsafe {
      let ret = isl_schedule_node_get_type(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_parent_type(self) -> ScheduleNodeType {
    unsafe {
      let ret = isl_schedule_node_get_parent_type(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_schedule(self) -> Option<Schedule> {
    unsafe {
      let ret = isl_schedule_node_get_schedule(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_descendant_top_down<F1: FnMut(ScheduleNodeRef) -> Bool>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(ScheduleNodeRef) -> Bool>(node: ScheduleNodeRef, user: *mut c_void) -> Bool { (*(user as *mut F))(node.to()) }
    unsafe {
      let ret = isl_schedule_node_foreach_descendant_top_down(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn every_descendant<F1: FnMut(ScheduleNodeRef) -> Bool>(self, test: &mut F1) -> Bool {
    unsafe extern "C" fn fn1<F: FnMut(ScheduleNodeRef) -> Bool>(node: ScheduleNodeRef, user: *mut c_void) -> Bool { (*(user as *mut F))(node.to()) }
    unsafe {
      let ret = isl_schedule_node_every_descendant(self.to(), fn1::<F1>, test as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_ancestor_top_down<F1: FnMut(ScheduleNodeRef) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(ScheduleNodeRef) -> Stat>(node: ScheduleNodeRef, user: *mut c_void) -> Stat { (*(user as *mut F))(node.to()) }
    unsafe {
      let ret = isl_schedule_node_foreach_ancestor_top_down(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_tree_depth(self) -> c_int {
    unsafe {
      let ret = isl_schedule_node_get_tree_depth(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_parent(self) -> Bool {
    unsafe {
      let ret = isl_schedule_node_has_parent(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_children(self) -> Bool {
    unsafe {
      let ret = isl_schedule_node_has_children(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_previous_sibling(self) -> Bool {
    unsafe {
      let ret = isl_schedule_node_has_previous_sibling(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_next_sibling(self) -> Bool {
    unsafe {
      let ret = isl_schedule_node_has_next_sibling(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_children(self) -> c_int {
    unsafe {
      let ret = isl_schedule_node_n_children(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_child_position(self) -> c_int {
    unsafe {
      let ret = isl_schedule_node_get_child_position(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ancestor_child_position(self, ancestor: ScheduleNodeRef) -> c_int {
    unsafe {
      let ret = isl_schedule_node_get_ancestor_child_position(self.to(), ancestor.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_child(self, pos: c_int) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_get_child(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_shared_ancestor(self, node2: ScheduleNodeRef) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_get_shared_ancestor(self.to(), node2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_subtree_anchored(self) -> Bool {
    unsafe {
      let ret = isl_schedule_node_is_subtree_anchored(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_schedule_node_band_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_get_partial_schedule(self) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_schedule_node_band_get_partial_schedule(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_get_partial_schedule_union_map(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_schedule_node_band_get_partial_schedule_union_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_member_get_ast_loop_type(self, pos: c_int) -> AstLoopType {
    unsafe {
      let ret = isl_schedule_node_band_member_get_ast_loop_type(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_member_get_isolate_ast_loop_type(self, pos: c_int) -> AstLoopType {
    unsafe {
      let ret = isl_schedule_node_band_member_get_isolate_ast_loop_type(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_get_ast_build_options(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_schedule_node_band_get_ast_build_options(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_get_ast_isolate_option(self) -> Option<Set> {
    unsafe {
      let ret = isl_schedule_node_band_get_ast_isolate_option(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_n_member(self) -> c_uint {
    unsafe {
      let ret = isl_schedule_node_band_n_member(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_member_get_coincident(self, pos: c_int) -> Bool {
    unsafe {
      let ret = isl_schedule_node_band_member_get_coincident(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn band_get_permutable(self) -> Bool {
    unsafe {
      let ret = isl_schedule_node_band_get_permutable(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn context_get_context(self) -> Option<Set> {
    unsafe {
      let ret = isl_schedule_node_context_get_context(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn domain_get_domain(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_schedule_node_domain_get_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn expansion_get_expansion(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_schedule_node_expansion_get_expansion(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn expansion_get_contraction(self) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_schedule_node_expansion_get_contraction(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn extension_get_extension(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_schedule_node_extension_get_extension(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn filter_get_filter(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_schedule_node_filter_get_filter(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn guard_get_guard(self) -> Option<Set> {
    unsafe {
      let ret = isl_schedule_node_guard_get_guard(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mark_get_id(self) -> Option<Id> {
    unsafe {
      let ret = isl_schedule_node_mark_get_id(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_schedule_depth(self) -> c_int {
    unsafe {
      let ret = isl_schedule_node_get_schedule_depth(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_domain(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_schedule_node_get_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_universe_domain(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_schedule_node_get_universe_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_prefix_schedule_multi_union_pw_aff(self) -> Option<MultiUnionPwAff> {
    unsafe {
      let ret = isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_prefix_schedule_union_pw_multi_aff(self) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_prefix_schedule_union_map(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_schedule_node_get_prefix_schedule_union_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_prefix_schedule_relation(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_schedule_node_get_prefix_schedule_relation(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_subtree_schedule_union_map(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_schedule_node_get_subtree_schedule_union_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_subtree_expansion(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_schedule_node_get_subtree_expansion(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_subtree_contraction(self) -> Option<UnionPwMultiAff> {
    unsafe {
      let ret = isl_schedule_node_get_subtree_contraction(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_schedule_node_dump(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_schedule_node_to_str(self.to());
      (ret).to()
    }
  }
}

impl UnionMap {
  #[inline(always)]
  pub fn schedule_node_from_extension(self) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_from_extension(self.to());
      (ret).to()
    }
  }
}

impl UnionSet {
  #[inline(always)]
  pub fn schedule_node_from_domain(self) -> Option<ScheduleNode> {
    unsafe {
      let ret = isl_schedule_node_from_domain(self.to());
      (ret).to()
    }
  }
}

impl Drop for ScheduleNode {
  fn drop(&mut self) { ScheduleNode(self.0).free() }
}

impl fmt::Display for ScheduleNodeRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for ScheduleNode {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

