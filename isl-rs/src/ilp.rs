use crate::*;

extern "C" {
  pub fn isl_basic_set_max_val(bset: BasicSetRef, obj: AffRef) -> Option<Val>;
  pub fn isl_set_min_val(set: SetRef, obj: AffRef) -> Option<Val>;
  pub fn isl_set_max_val(set: SetRef, obj: AffRef) -> Option<Val>;
  pub fn isl_union_set_min_multi_union_pw_aff(uset: UnionSetRef, obj: MultiUnionPwAffRef) -> Option<MultiVal>;
  pub fn isl_pw_multi_aff_min_multi_val(pma: PwMultiAff) -> Option<MultiVal>;
  pub fn isl_pw_multi_aff_max_multi_val(pma: PwMultiAff) -> Option<MultiVal>;
  pub fn isl_multi_pw_aff_min_multi_val(mpa: MultiPwAff) -> Option<MultiVal>;
  pub fn isl_multi_pw_aff_max_multi_val(mpa: MultiPwAff) -> Option<MultiVal>;
  pub fn isl_union_pw_aff_min_val(upa: UnionPwAff) -> Option<Val>;
  pub fn isl_union_pw_aff_max_val(upa: UnionPwAff) -> Option<Val>;
  pub fn isl_multi_union_pw_aff_min_multi_val(mupa: MultiUnionPwAff) -> Option<MultiVal>;
  pub fn isl_multi_union_pw_aff_max_multi_val(mupa: MultiUnionPwAff) -> Option<MultiVal>;
  pub fn isl_basic_set_dim_max_val(bset: BasicSet, pos: c_int) -> Option<Val>;
  pub fn isl_set_dim_min_val(set: Set, pos: c_int) -> Option<Val>;
  pub fn isl_set_dim_max_val(set: Set, pos: c_int) -> Option<Val>;
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

impl MultiPwAff {
  #[inline(always)]
  pub fn min_multi_val(self) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_pw_aff_min_multi_val(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn max_multi_val(self) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_pw_aff_max_multi_val(self.to());
      (ret).to()
    }
  }
}

impl MultiUnionPwAff {
  #[inline(always)]
  pub fn min_multi_val(self) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_union_pw_aff_min_multi_val(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn max_multi_val(self) -> Option<MultiVal> {
    unsafe {
      let ret = isl_multi_union_pw_aff_max_multi_val(self.to());
      (ret).to()
    }
  }
}

impl PwMultiAff {
  #[inline(always)]
  pub fn min_multi_val(self) -> Option<MultiVal> {
    unsafe {
      let ret = isl_pw_multi_aff_min_multi_val(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn max_multi_val(self) -> Option<MultiVal> {
    unsafe {
      let ret = isl_pw_multi_aff_max_multi_val(self.to());
      (ret).to()
    }
  }
}

impl Set {
  #[inline(always)]
  pub fn dim_min_val(self, pos: c_int) -> Option<Val> {
    unsafe {
      let ret = isl_set_dim_min_val(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dim_max_val(self, pos: c_int) -> Option<Val> {
    unsafe {
      let ret = isl_set_dim_max_val(self.to(), pos.to());
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

impl UnionPwAff {
  #[inline(always)]
  pub fn min_val(self) -> Option<Val> {
    unsafe {
      let ret = isl_union_pw_aff_min_val(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn max_val(self) -> Option<Val> {
    unsafe {
      let ret = isl_union_pw_aff_max_val(self.to());
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

