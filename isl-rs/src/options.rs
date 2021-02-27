use crate::*;

extern "C" {
  pub fn isl_options_set_bound(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_bound(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_on_error(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_on_error(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_gbr_only_first(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_gbr_only_first(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_schedule_algorithm(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_schedule_algorithm(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_pip_symmetry(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_pip_symmetry(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_coalesce_bounded_wrapping(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_coalesce_bounded_wrapping(ctx: CtxRef) -> c_int;
  pub fn isl_options_set_coalesce_preserve_locals(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_coalesce_preserve_locals(ctx: CtxRef) -> c_int;
}

impl CtxRef {
  #[inline(always)]
  pub fn options_set_bound(self, val: c_int) -> Stat {
    unsafe {
      let ret = isl_options_set_bound(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_bound(self) -> c_int {
    unsafe {
      let ret = isl_options_get_bound(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_on_error(self, val: c_int) -> Stat {
    unsafe {
      let ret = isl_options_set_on_error(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_on_error(self) -> c_int {
    unsafe {
      let ret = isl_options_get_on_error(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_gbr_only_first(self, val: c_int) -> Stat {
    unsafe {
      let ret = isl_options_set_gbr_only_first(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_gbr_only_first(self) -> c_int {
    unsafe {
      let ret = isl_options_get_gbr_only_first(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_schedule_algorithm(self, val: c_int) -> Stat {
    unsafe {
      let ret = isl_options_set_schedule_algorithm(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_schedule_algorithm(self) -> c_int {
    unsafe {
      let ret = isl_options_get_schedule_algorithm(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_pip_symmetry(self, val: c_int) -> Stat {
    unsafe {
      let ret = isl_options_set_pip_symmetry(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_pip_symmetry(self) -> c_int {
    unsafe {
      let ret = isl_options_get_pip_symmetry(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_coalesce_bounded_wrapping(self, val: c_int) -> Stat {
    unsafe {
      let ret = isl_options_set_coalesce_bounded_wrapping(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_coalesce_bounded_wrapping(self) -> c_int {
    unsafe {
      let ret = isl_options_get_coalesce_bounded_wrapping(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_coalesce_preserve_locals(self, val: c_int) -> Stat {
    unsafe {
      let ret = isl_options_set_coalesce_preserve_locals(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_coalesce_preserve_locals(self) -> c_int {
    unsafe {
      let ret = isl_options_get_coalesce_preserve_locals(self.to());
      (ret).to()
    }
  }
}

