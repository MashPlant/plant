use crate::*;

extern "C" {
  pub fn isl_options_set_ast_build_atomic_upper_bound(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_ast_build_atomic_upper_bound(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_ast_build_prefer_pdiv(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_ast_build_prefer_pdiv(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_ast_build_detect_min_max(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_ast_build_detect_min_max(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_ast_build_exploit_nested_bounds(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_ast_build_exploit_nested_bounds(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_ast_build_group_coscheduled(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_ast_build_group_coscheduled(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_ast_build_separation_bounds(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_ast_build_separation_bounds(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_ast_build_scale_strides(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_ast_build_scale_strides(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_ast_build_allow_else(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_ast_build_allow_else(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_ast_build_allow_or(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_ast_build_allow_or(ctx: CtxRef) -> c_int;
  pub fn isl_ast_build_get_ctx(build: AstBuildRef) -> Option<CtxRef>;
  pub fn isl_ast_build_alloc(ctx: CtxRef) -> Option<AstBuild>;
  pub fn isl_ast_build_from_context(set: Set) -> Option<AstBuild>;
  pub fn isl_ast_build_get_schedule_space(build: AstBuildRef) -> Option<Space>;
  pub fn isl_ast_build_get_schedule(build: AstBuildRef) -> Option<UnionMap>;
  pub fn isl_ast_build_restrict(build: AstBuild, set: Set) -> Option<AstBuild>;
  pub fn isl_ast_build_copy(build: AstBuildRef) -> Option<AstBuild>;
  pub fn isl_ast_build_free(build: AstBuild) -> *mut c_void;
  pub fn isl_ast_build_set_options(build: AstBuild, options: UnionMap) -> Option<AstBuild>;
  pub fn isl_ast_build_set_iterators(build: AstBuild, iterators: IdList) -> Option<AstBuild>;
  pub fn isl_ast_build_set_at_each_domain(build: AstBuild, fn_: unsafe extern "C" fn(node: AstNode, build: AstBuildRef, user: *mut c_void) -> Option<AstNode>, user: *mut c_void) -> Option<AstBuild>;
  pub fn isl_ast_build_set_before_each_for(build: AstBuild, fn_: unsafe extern "C" fn(build: AstBuildRef, user: *mut c_void) -> Option<Id>, user: *mut c_void) -> Option<AstBuild>;
  pub fn isl_ast_build_set_after_each_for(build: AstBuild, fn_: unsafe extern "C" fn(node: AstNode, build: AstBuildRef, user: *mut c_void) -> Option<AstNode>, user: *mut c_void) -> Option<AstBuild>;
  pub fn isl_ast_build_set_before_each_mark(build: AstBuild, fn_: unsafe extern "C" fn(mark: IdRef, build: AstBuildRef, user: *mut c_void) -> Stat, user: *mut c_void) -> Option<AstBuild>;
  pub fn isl_ast_build_set_after_each_mark(build: AstBuild, fn_: unsafe extern "C" fn(node: AstNode, build: AstBuildRef, user: *mut c_void) -> Option<AstNode>, user: *mut c_void) -> Option<AstBuild>;
  pub fn isl_ast_build_set_create_leaf(build: AstBuild, fn_: unsafe extern "C" fn(build: AstBuild, user: *mut c_void) -> Option<AstNode>, user: *mut c_void) -> Option<AstBuild>;
  pub fn isl_ast_build_expr_from_set(build: AstBuildRef, set: Set) -> Option<AstExpr>;
  pub fn isl_ast_build_expr_from_pw_aff(build: AstBuildRef, pa: PwAff) -> Option<AstExpr>;
  pub fn isl_ast_build_access_from_pw_multi_aff(build: AstBuildRef, pma: PwMultiAff) -> Option<AstExpr>;
  pub fn isl_ast_build_access_from_multi_pw_aff(build: AstBuildRef, mpa: MultiPwAff) -> Option<AstExpr>;
  pub fn isl_ast_build_call_from_pw_multi_aff(build: AstBuildRef, pma: PwMultiAff) -> Option<AstExpr>;
  pub fn isl_ast_build_call_from_multi_pw_aff(build: AstBuildRef, mpa: MultiPwAff) -> Option<AstExpr>;
  pub fn isl_ast_build_node_from_schedule(build: AstBuildRef, schedule: Schedule) -> Option<AstNode>;
  pub fn isl_ast_build_node_from_schedule_map(build: AstBuildRef, schedule: UnionMap) -> Option<AstNode>;
  pub fn isl_ast_build_ast_from_schedule(build: AstBuildRef, schedule: UnionMap) -> Option<AstNode>;
}

#[repr(transparent)]
#[derive(Debug)]
pub struct AstBuild(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct AstBuildRef(pub NonNull<c_void>);

impl AstBuild {
  #[inline(always)]
  pub fn read(&self) -> AstBuild { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: AstBuild) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<AstBuildRef> for AstBuild {
  #[inline(always)]
  fn as_ref(&self) -> &AstBuildRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for AstBuild {
  type Target = AstBuildRef;
  #[inline(always)]
  fn deref(&self) -> &AstBuildRef { self.as_ref() }
}

impl To<Option<AstBuild>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<AstBuild> { NonNull::new(self).map(AstBuild) }
}

impl AstBuild {
  #[inline(always)]
  pub fn restrict(self, set: Set) -> Option<AstBuild> {
    unsafe {
      let ret = isl_ast_build_restrict(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_ast_build_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_options(self, options: UnionMap) -> Option<AstBuild> {
    unsafe {
      let ret = isl_ast_build_set_options(self.to(), options.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_iterators(self, iterators: IdList) -> Option<AstBuild> {
    unsafe {
      let ret = isl_ast_build_set_iterators(self.to(), iterators.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_at_each_domain<F1: FnMut(AstNode, AstBuildRef) -> Option<AstNode>>(self, fn_: &mut F1) -> Option<AstBuild> {
    unsafe extern "C" fn fn1<F: FnMut(AstNode, AstBuildRef) -> Option<AstNode>>(node: AstNode, build: AstBuildRef, user: *mut c_void) -> Option<AstNode> { (*(user as *mut F))(node.to(), build.to()).to() }
    unsafe {
      let ret = isl_ast_build_set_at_each_domain(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_before_each_for<F1: FnMut(AstBuildRef) -> Option<Id>>(self, fn_: &mut F1) -> Option<AstBuild> {
    unsafe extern "C" fn fn1<F: FnMut(AstBuildRef) -> Option<Id>>(build: AstBuildRef, user: *mut c_void) -> Option<Id> { (*(user as *mut F))(build.to()).to() }
    unsafe {
      let ret = isl_ast_build_set_before_each_for(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_after_each_for<F1: FnMut(AstNode, AstBuildRef) -> Option<AstNode>>(self, fn_: &mut F1) -> Option<AstBuild> {
    unsafe extern "C" fn fn1<F: FnMut(AstNode, AstBuildRef) -> Option<AstNode>>(node: AstNode, build: AstBuildRef, user: *mut c_void) -> Option<AstNode> { (*(user as *mut F))(node.to(), build.to()).to() }
    unsafe {
      let ret = isl_ast_build_set_after_each_for(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_before_each_mark<F1: FnMut(IdRef, AstBuildRef) -> Option<()>>(self, fn_: &mut F1) -> Option<AstBuild> {
    unsafe extern "C" fn fn1<F: FnMut(IdRef, AstBuildRef) -> Option<()>>(mark: IdRef, build: AstBuildRef, user: *mut c_void) -> Stat { (*(user as *mut F))(mark.to(), build.to()).to() }
    unsafe {
      let ret = isl_ast_build_set_before_each_mark(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_after_each_mark<F1: FnMut(AstNode, AstBuildRef) -> Option<AstNode>>(self, fn_: &mut F1) -> Option<AstBuild> {
    unsafe extern "C" fn fn1<F: FnMut(AstNode, AstBuildRef) -> Option<AstNode>>(node: AstNode, build: AstBuildRef, user: *mut c_void) -> Option<AstNode> { (*(user as *mut F))(node.to(), build.to()).to() }
    unsafe {
      let ret = isl_ast_build_set_after_each_mark(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_create_leaf<F1: FnMut(AstBuild) -> Option<AstNode>>(self, fn_: &mut F1) -> Option<AstBuild> {
    unsafe extern "C" fn fn1<F: FnMut(AstBuild) -> Option<AstNode>>(build: AstBuild, user: *mut c_void) -> Option<AstNode> { (*(user as *mut F))(build.to()).to() }
    unsafe {
      let ret = isl_ast_build_set_create_leaf(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
}

impl AstBuildRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_ast_build_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_schedule_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_ast_build_get_schedule_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_schedule(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_ast_build_get_schedule(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<AstBuild> {
    unsafe {
      let ret = isl_ast_build_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn expr_from_set(self, set: Set) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_build_expr_from_set(self.to(), set.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn expr_from_pw_aff(self, pa: PwAff) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_build_expr_from_pw_aff(self.to(), pa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn access_from_pw_multi_aff(self, pma: PwMultiAff) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_build_access_from_pw_multi_aff(self.to(), pma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn access_from_multi_pw_aff(self, mpa: MultiPwAff) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_build_access_from_multi_pw_aff(self.to(), mpa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn call_from_pw_multi_aff(self, pma: PwMultiAff) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_build_call_from_pw_multi_aff(self.to(), pma.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn call_from_multi_pw_aff(self, mpa: MultiPwAff) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_build_call_from_multi_pw_aff(self.to(), mpa.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn node_from_schedule(self, schedule: Schedule) -> Option<AstNode> {
    unsafe {
      let ret = isl_ast_build_node_from_schedule(self.to(), schedule.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn node_from_schedule_map(self, schedule: UnionMap) -> Option<AstNode> {
    unsafe {
      let ret = isl_ast_build_node_from_schedule_map(self.to(), schedule.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ast_from_schedule(self, schedule: UnionMap) -> Option<AstNode> {
    unsafe {
      let ret = isl_ast_build_ast_from_schedule(self.to(), schedule.to());
      (ret).to()
    }
  }
}

impl CtxRef {
  #[inline(always)]
  pub fn options_set_ast_build_atomic_upper_bound(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_ast_build_atomic_upper_bound(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_ast_build_atomic_upper_bound(self) -> c_int {
    unsafe {
      let ret = isl_options_get_ast_build_atomic_upper_bound(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_ast_build_prefer_pdiv(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_ast_build_prefer_pdiv(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_ast_build_prefer_pdiv(self) -> c_int {
    unsafe {
      let ret = isl_options_get_ast_build_prefer_pdiv(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_ast_build_detect_min_max(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_ast_build_detect_min_max(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_ast_build_detect_min_max(self) -> c_int {
    unsafe {
      let ret = isl_options_get_ast_build_detect_min_max(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_ast_build_exploit_nested_bounds(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_ast_build_exploit_nested_bounds(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_ast_build_exploit_nested_bounds(self) -> c_int {
    unsafe {
      let ret = isl_options_get_ast_build_exploit_nested_bounds(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_ast_build_group_coscheduled(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_ast_build_group_coscheduled(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_ast_build_group_coscheduled(self) -> c_int {
    unsafe {
      let ret = isl_options_get_ast_build_group_coscheduled(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_ast_build_separation_bounds(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_ast_build_separation_bounds(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_ast_build_separation_bounds(self) -> c_int {
    unsafe {
      let ret = isl_options_get_ast_build_separation_bounds(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_ast_build_scale_strides(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_ast_build_scale_strides(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_ast_build_scale_strides(self) -> c_int {
    unsafe {
      let ret = isl_options_get_ast_build_scale_strides(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_ast_build_allow_else(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_ast_build_allow_else(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_ast_build_allow_else(self) -> c_int {
    unsafe {
      let ret = isl_options_get_ast_build_allow_else(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_ast_build_allow_or(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_ast_build_allow_or(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_ast_build_allow_or(self) -> c_int {
    unsafe {
      let ret = isl_options_get_ast_build_allow_or(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ast_build_alloc(self) -> Option<AstBuild> {
    unsafe {
      let ret = isl_ast_build_alloc(self.to());
      (ret).to()
    }
  }
}

impl Set {
  #[inline(always)]
  pub fn ast_build_from_context(self) -> Option<AstBuild> {
    unsafe {
      let ret = isl_ast_build_from_context(self.to());
      (ret).to()
    }
  }
}

impl Drop for AstBuild {
  fn drop(&mut self) { AstBuild(self.0).free() }
}

