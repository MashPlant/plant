use crate::*;

extern "C" {
  pub fn isl_stride_info_get_ctx(si: StrideInfoRef) -> Option<CtxRef>;
  pub fn isl_stride_info_get_stride(si: StrideInfoRef) -> Option<Val>;
  pub fn isl_stride_info_get_offset(si: StrideInfoRef) -> Option<Aff>;
  pub fn isl_stride_info_free(si: StrideInfo) -> *mut c_void;
  pub fn isl_stride_info_copy(si: StrideInfoRef) -> Option<StrideInfo>;
}

#[repr(transparent)]
#[derive(Debug)]
pub struct StrideInfo(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct StrideInfoRef(pub NonNull<c_void>);

impl_try!(StrideInfo);
impl_try!(StrideInfoRef);

impl StrideInfo {
  #[inline(always)]
  pub fn read(&self) -> StrideInfo { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: StrideInfo) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<StrideInfoRef> for StrideInfo {
  #[inline(always)]
  fn as_ref(&self) -> &StrideInfoRef { unsafe { mem::transmute(self) } }
}

impl Deref for StrideInfo {
  type Target = StrideInfoRef;
  #[inline(always)]
  fn deref(&self) -> &StrideInfoRef { self.as_ref() }
}

impl To<Option<StrideInfo>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<StrideInfo> { NonNull::new(self).map(StrideInfo) }
}

impl StrideInfo {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_stride_info_free(self.to());
      (ret).to()
    }
  }
}

impl StrideInfoRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_stride_info_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_stride(self) -> Option<Val> {
    unsafe {
      let ret = isl_stride_info_get_stride(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_offset(self) -> Option<Aff> {
    unsafe {
      let ret = isl_stride_info_get_offset(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<StrideInfo> {
    unsafe {
      let ret = isl_stride_info_copy(self.to());
      (ret).to()
    }
  }
}

impl Drop for StrideInfo {
  fn drop(&mut self) { StrideInfo(self.0).free() }
}

