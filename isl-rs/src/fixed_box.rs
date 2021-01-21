use crate::*;

extern "C" {
  pub fn isl_fixed_box_get_ctx(box_: FixedBoxRef) -> Option<CtxRef>;
  pub fn isl_fixed_box_get_space(box_: FixedBoxRef) -> Option<Space>;
  pub fn isl_fixed_box_is_valid(box_: FixedBoxRef) -> Bool;
  pub fn isl_fixed_box_get_offset(box_: FixedBoxRef) -> Option<MultiAff>;
  pub fn isl_fixed_box_get_size(box_: FixedBoxRef) -> Option<MultiVal>;
  pub fn isl_fixed_box_copy(box_: FixedBoxRef) -> Option<FixedBox>;
  pub fn isl_fixed_box_free(box_: FixedBox) -> *mut c_void;
  pub fn isl_printer_print_fixed_box(p: Printer, box_: FixedBoxRef) -> Option<Printer>;
  pub fn isl_fixed_box_to_str(box_: FixedBoxRef) -> Option<CString>;
  pub fn isl_fixed_box_dump(box_: FixedBoxRef) -> ();
}

#[repr(transparent)]
#[derive(Debug)]
pub struct FixedBox(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct FixedBoxRef(pub NonNull<c_void>);

impl_try!(FixedBox);
impl_try!(FixedBoxRef);

impl FixedBox {
  #[inline(always)]
  pub fn read(&self) -> FixedBox { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: FixedBox) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<FixedBoxRef> for FixedBox {
  #[inline(always)]
  fn as_ref(&self) -> &FixedBoxRef { unsafe { mem::transmute(self) } }
}

impl Deref for FixedBox {
  type Target = FixedBoxRef;
  #[inline(always)]
  fn deref(&self) -> &FixedBoxRef { self.as_ref() }
}

impl To<Option<FixedBox>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<FixedBox> { NonNull::new(self).map(FixedBox) }
}

impl FixedBox {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_fixed_box_free(self.to());
      (ret).to()
    }
  }
}

impl FixedBoxRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_fixed_box_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_fixed_box_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_valid(self) -> Bool {
    unsafe {
      let ret = isl_fixed_box_is_valid(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_offset(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_fixed_box_get_offset(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_size(self) -> Option<MultiVal> {
    unsafe {
      let ret = isl_fixed_box_get_size(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<FixedBox> {
    unsafe {
      let ret = isl_fixed_box_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_fixed_box_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_fixed_box_dump(self.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_fixed_box(self, box_: FixedBoxRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_fixed_box(self.to(), box_.to());
      (ret).to()
    }
  }
}

impl Drop for FixedBox {
  fn drop(&mut self) { FixedBox(self.0).free() }
}

impl fmt::Display for FixedBoxRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for FixedBox {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

