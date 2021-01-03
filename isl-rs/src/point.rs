use crate::*;

extern "C" {
  pub fn isl_point_get_ctx(pnt: PointRef) -> Option<CtxRef>;
  pub fn isl_point_get_space(pnt: PointRef) -> Option<Space>;
  pub fn isl_point_zero(dim: Space) -> Option<Point>;
  pub fn isl_point_copy(pnt: PointRef) -> Option<Point>;
  pub fn isl_point_free(pnt: Point) -> *mut c_void;
  pub fn isl_point_get_coordinate_val(pnt: PointRef, type_: DimType, pos: c_int) -> Option<Val>;
  pub fn isl_point_set_coordinate_val(pnt: Point, type_: DimType, pos: c_int, v: Val) -> Option<Point>;
  pub fn isl_point_add_ui(pnt: Point, type_: DimType, pos: c_int, val: c_uint) -> Option<Point>;
  pub fn isl_point_sub_ui(pnt: Point, type_: DimType, pos: c_int, val: c_uint) -> Option<Point>;
  pub fn isl_point_void(dim: Space) -> Option<Point>;
  pub fn isl_point_is_void(pnt: PointRef) -> Bool;
  pub fn isl_printer_print_point(printer: Printer, pnt: PointRef) -> Option<Printer>;
  pub fn isl_point_to_str(pnt: PointRef) -> Option<CString>;
  pub fn isl_point_dump(pnt: PointRef) -> ();
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Point(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct PointRef(pub NonNull<c_void>);

impl Point {
  #[inline(always)]
  pub fn read(&self) -> Point { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Point) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<PointRef> for Point {
  #[inline(always)]
  fn as_ref(&self) -> &PointRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for Point {
  type Target = PointRef;
  #[inline(always)]
  fn deref(&self) -> &PointRef { self.as_ref() }
}

impl To<Option<Point>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Point> { NonNull::new(self).map(Point) }
}

impl Point {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_point_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_coordinate_val(self, type_: DimType, pos: c_int, v: Val) -> Option<Point> {
    unsafe {
      let ret = isl_point_set_coordinate_val(self.to(), type_.to(), pos.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_ui(self, type_: DimType, pos: c_int, val: c_uint) -> Option<Point> {
    unsafe {
      let ret = isl_point_add_ui(self.to(), type_.to(), pos.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sub_ui(self, type_: DimType, pos: c_int, val: c_uint) -> Option<Point> {
    unsafe {
      let ret = isl_point_sub_ui(self.to(), type_.to(), pos.to(), val.to());
      (ret).to()
    }
  }
}

impl PointRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_point_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_point_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<Point> {
    unsafe {
      let ret = isl_point_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_coordinate_val(self, type_: DimType, pos: c_int) -> Option<Val> {
    unsafe {
      let ret = isl_point_get_coordinate_val(self.to(), type_.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_void(self) -> Option<bool> {
    unsafe {
      let ret = isl_point_is_void(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_point_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_point_dump(self.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_point(self, pnt: PointRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_point(self.to(), pnt.to());
      (ret).to()
    }
  }
}

impl Space {
  #[inline(always)]
  pub fn point_zero(self) -> Option<Point> {
    unsafe {
      let ret = isl_point_zero(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn point_void(self) -> Option<Point> {
    unsafe {
      let ret = isl_point_void(self.to());
      (ret).to()
    }
  }
}

impl Drop for Point {
  fn drop(&mut self) { Point(self.0).free() }
}

impl fmt::Display for PointRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for Point {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

