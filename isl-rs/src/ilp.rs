use crate::*;

extern "C" {
  pub fn isl_basic_set_max_val(bset: BasicSetRef, obj: AffRef) -> Option<Val>;
  pub fn isl_set_min_val(set: SetRef, obj: AffRef) -> Option<Val>;
  pub fn isl_set_max_val(set: SetRef, obj: AffRef) -> Option<Val>;
  pub fn isl_union_set_min_multi_union_pw_aff(set: UnionSetRef, obj: MultiUnionPwAffRef) -> Option<MultiVal>;
  pub fn isl_basic_set_dim_max_val(bset: BasicSet, pos: c_int) -> Option<Val>;
}

impl BasicSet {
  #[inline(always)]
  pub fn dim_max_val(self, pos: c_int) -> Option<Val> {
    unsafe {
      let ret = isl_basic_set_dim_max_val(self.to(), pos.to());
      (ret).to()
    }
  }
}

impl BasicSetRef {
  #[inline(always)]
  pub fn max_val(self, obj: AffRef) -> Option<Val> {
    unsafe {
      let ret = isl_basic_set_max_val(self.to(), obj.to());
      (ret).to()
    }
  }
}

impl SetRef {
  #[inline(always)]
  pub fn min_val(self, obj: AffRef) -> Option<Val> {
    unsafe {
      let ret = isl_set_min_val(self.to(), obj.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn max_val(self, obj: AffRef) -> Option<Val> {
    unsafe {
      let ret = isl_set_max_val(self.to(), obj.to());
      (ret).to()
    }
  }
}

impl UnionSetRef {
  #[inline(always)]
  pub fn min_multi_union_pw_aff(self, obj: MultiUnionPwAffRef) -> Option<MultiVal> {
    unsafe {
      let ret = isl_union_set_min_multi_union_pw_aff(self.to(), obj.to());
      (ret).to()
    }
  }
}

