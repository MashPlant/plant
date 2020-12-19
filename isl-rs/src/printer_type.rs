use crate::*;

extern "C" {
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Printer(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct PrinterRef(pub NonNull<c_void>);

impl Printer {
  #[inline(always)]
  pub fn read(&self) -> Printer { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&mut self, x: Printer) { unsafe { ptr::write(self, x) } }
}

impl AsRef<PrinterRef> for Printer {
  #[inline(always)]
  fn as_ref(&self) -> &PrinterRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for Printer {
  type Target = PrinterRef;
  #[inline(always)]
  fn deref(&self) -> &PrinterRef { self.as_ref() }
}

impl To<Option<Printer>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Printer> { NonNull::new(self).map(Printer) }
}

