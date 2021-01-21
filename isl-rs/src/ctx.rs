use crate::*;

extern "C" {
  pub fn isl_stat_non_null(obj: *mut c_void) -> Stat;
  pub fn isl_bool_not(b: Bool) -> Bool;
  pub fn isl_bool_ok(b: c_int) -> Bool;
  pub fn isl_handle_error(ctx: CtxRef, error: Error, msg: Option<CStr>, file: Option<CStr>, line: c_int) -> ();
  pub fn isl_ctx_alloc() -> Option<Ctx>;
  pub fn isl_ctx_ref(ctx: CtxRef) -> ();
  pub fn isl_ctx_deref(ctx: CtxRef) -> ();
  pub fn isl_ctx_free(ctx: Ctx) -> ();
  pub fn isl_ctx_abort(ctx: CtxRef) -> ();
  pub fn isl_ctx_resume(ctx: CtxRef) -> ();
  pub fn isl_ctx_aborted(ctx: CtxRef) -> c_int;
  pub fn isl_ctx_set_max_operations(ctx: CtxRef, max_operations: c_ulong) -> ();
  pub fn isl_ctx_get_max_operations(ctx: CtxRef) -> c_ulong;
  pub fn isl_ctx_reset_operations(ctx: CtxRef) -> ();
  pub fn isl_ctx_last_error(ctx: CtxRef) -> Error;
  pub fn isl_ctx_last_error_msg(ctx: CtxRef) -> Option<CStr>;
  pub fn isl_ctx_last_error_file(ctx: CtxRef) -> Option<CStr>;
  pub fn isl_ctx_last_error_line(ctx: CtxRef) -> c_int;
  pub fn isl_ctx_reset_error(ctx: CtxRef) -> ();
  pub fn isl_ctx_set_error(ctx: CtxRef, error: Error) -> ();
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Size(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct SizeRef(pub NonNull<c_void>);

impl_try!(Size);
impl_try!(SizeRef);

impl Size {
  #[inline(always)]
  pub fn read(&self) -> Size { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Size) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<SizeRef> for Size {
  #[inline(always)]
  fn as_ref(&self) -> &SizeRef { unsafe { mem::transmute(self) } }
}

impl Deref for Size {
  type Target = SizeRef;
  #[inline(always)]
  fn deref(&self) -> &SizeRef { self.as_ref() }
}

impl To<Option<Size>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Size> { NonNull::new(self).map(Size) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Ctx(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct CtxRef(pub NonNull<c_void>);

impl_try!(Ctx);
impl_try!(CtxRef);

impl Ctx {
  #[inline(always)]
  pub fn read(&self) -> Ctx { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Ctx) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<CtxRef> for Ctx {
  #[inline(always)]
  fn as_ref(&self) -> &CtxRef { unsafe { mem::transmute(self) } }
}

impl Deref for Ctx {
  type Target = CtxRef;
  #[inline(always)]
  fn deref(&self) -> &CtxRef { self.as_ref() }
}

impl To<Option<Ctx>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Ctx> { NonNull::new(self).map(Ctx) }
}

impl Ctx {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_ctx_free(self.to());
      (ret).to()
    }
  }
}

impl CtxRef {
  #[inline(always)]
  pub fn handle_error(self, error: Error, msg: Option<CStr>, file: Option<CStr>, line: c_int) -> () {
    unsafe {
      let ret = isl_handle_error(self.to(), error.to(), msg.to(), file.to(), line.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ref_(self) -> () {
    unsafe {
      let ret = isl_ctx_ref(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn deref(self) -> () {
    unsafe {
      let ret = isl_ctx_deref(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn abort(self) -> () {
    unsafe {
      let ret = isl_ctx_abort(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn resume(self) -> () {
    unsafe {
      let ret = isl_ctx_resume(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn aborted(self) -> c_int {
    unsafe {
      let ret = isl_ctx_aborted(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_max_operations(self, max_operations: c_ulong) -> () {
    unsafe {
      let ret = isl_ctx_set_max_operations(self.to(), max_operations.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_max_operations(self) -> c_ulong {
    unsafe {
      let ret = isl_ctx_get_max_operations(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_operations(self) -> () {
    unsafe {
      let ret = isl_ctx_reset_operations(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn last_error(self) -> Error {
    unsafe {
      let ret = isl_ctx_last_error(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn last_error_msg(self) -> Option<CStr> {
    unsafe {
      let ret = isl_ctx_last_error_msg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn last_error_file(self) -> Option<CStr> {
    unsafe {
      let ret = isl_ctx_last_error_file(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn last_error_line(self) -> c_int {
    unsafe {
      let ret = isl_ctx_last_error_line(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_error(self) -> () {
    unsafe {
      let ret = isl_ctx_reset_error(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_error(self, error: Error) -> () {
    unsafe {
      let ret = isl_ctx_set_error(self.to(), error.to());
      (ret).to()
    }
  }
}

impl Drop for Ctx {
  fn drop(&mut self) { Ctx(self.0).free() }
}

