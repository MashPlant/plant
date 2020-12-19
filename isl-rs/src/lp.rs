use crate::*;

extern "C" {
  pub fn isl_basic_set_min_lp_val(bset: BasicSetRef, obj: AffRef) -> Option<Val>;
  pub fn isl_basic_set_max_lp_val(bset: BasicSetRef, obj: AffRef) -> Option<Val>;
}

impl BasicSetRef {
  #[inline(always)]
  pub fn min_lp_val(self, obj: AffRef) -> Option<Val> {
    unsafe {
      let ret = isl_basic_set_min_lp_val(self.to(), obj.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn max_lp_val(self, obj: AffRef) -> Option<Val> {
    unsafe {
      let ret = isl_basic_set_max_lp_val(self.to(), obj.to());
      (ret).to()
    }
  }
}

