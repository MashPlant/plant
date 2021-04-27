use pyo3::{prelude::*, wrap_pyfunction};
use smallvec::SmallVec;
use std::{mem::*, num::NonZeroUsize, fmt};
use plant_runtime::{*, Array as A};

#[pyfunction]
fn parallel_init(th: u32) { plant_runtime::parallel_init(th) }

type D = SmallVec<[u32; 4]>;

#[pyclass]
struct Array {
  ptr: NonZeroUsize,
  ty: Type,
  loc: Backend,
  owned: bool,
  dims: D,
}

#[pyfunction]
unsafe fn array_alloc(dims: Vec<u32>, ty: u32, loc: u32) -> Array {
  assert!(ty < Void as u32 && loc <= GPU as u32);
  let ty = transmute::<_, Type>(ty as u8);
  let loc = if loc == 0 { CPU } else { GPU };
  let bytes = dims.iter().map(|x| *x as usize).product::<usize>() * ty.size();
  let arr = if loc == CPU { A::<u8, _>::new(bytes) } else {
    #[cfg(feature = "gpu-runtime")] { A::<u8, _>::new_gpu(bytes) }
    #[cfg(not(feature = "gpu-runtime"))] panic!("gpu-runtime not enabled")
  };
  Array::from_array(arr, dims.into(), ty)
}

#[pyfunction]
unsafe fn array_borrow(dims: Vec<u32>, ptr: usize, ty: u32) -> Array {
  assert!(ptr != 0 && ty < Void as u32);
  Array { ptr: transmute(ptr), ty: transmute(ty as u8), loc: CPU, owned: false, dims: dims.into() }
}

impl Array {
  unsafe fn from_array(arr: A<u8, usize>, dims: D, ty: Type) -> Array {
    let (ptr, loc) = (transmute(arr.ptr), arr.loc);
    forget(arr);
    Array { ptr, ty, loc, owned: true, dims }
  }

  unsafe fn slice(&self) -> Slice<u8, usize> {
    Slice { ptr: transmute(self.ptr), dim: self.elems() * self.ty.size(), loc: self.loc }
  }
}

macro_rules! handle_all {
  ($ty: expr, $handle: ident) => {
    match $ty {
      I8 => $handle!(i8),
      U8 => $handle!(u8),
      I16 => $handle!(i16),
      U16 => $handle!(u16),
      I32 => $handle!(i32),
      U32 => $handle!(u32),
      I64 => $handle!(i64),
      U64 => $handle!(u64),
      F32 => $handle!(f32),
      F64 => $handle!(f64),
      _ => std::hint::unreachable_unchecked(),
    }
  };
}

#[pymethods]
impl Array {
  #[getter]
  fn ptr(&self) -> usize { self.ptr.get() }
  #[getter]
  fn ty(&self) -> u32 { self.ty as _ }
  #[getter]
  fn loc(&self) -> u32 { self.loc as _ }
  #[getter]
  fn dims(&self) -> Vec<u32> { self.dims.to_vec() }
  #[getter]
  fn elems(&self) -> usize { self.dims.iter().map(|x| *x as usize).product::<usize>() }

  unsafe fn assert_close(&self, rhs: &Self, threshold: f64) {
    assert!(self.ty == rhs.ty && self.dims == rhs.dims);
    let (p1, d1, l1) = (self.ptr, self.elems(), self.loc);
    let (p2, d2, l2) = (rhs.ptr, rhs.elems(), rhs.loc);
    macro_rules! handle {
      ($ty: ident) => { Slice::<$ty, _> { ptr: transmute(p1), dim: d1, loc: l1 }.assert_close(&Slice { ptr: transmute(p2), dim: d2, loc: l2 }, threshold as $ty) };
    }
    handle_all!(self.ty, handle);
  }

  fn __str__(&self) -> String { self.to_string() }
}

#[pymethods]
#[cfg(feature = "gpu-runtime")]
impl Array {
  unsafe fn to_cpu(&self) -> Array { Array::from_array(self.slice().to_cpu(), self.dims.clone(), self.ty) }
  unsafe fn to_gpu(&self) -> Array { Array::from_array(self.slice().to_gpu(), self.dims.clone(), self.ty) }
}

impl fmt::Display for Array {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    let mut d = f.debug_struct("Array");
    d.field("ptr", &(self.ptr.get() as *const u8))
      .field("ty", &self.ty)
      .field("loc", &self.loc)
      .field("dims", &self.dims);
    #[cfg(feature = "gpu-runtime")] let tmp;
    let data = if self.loc == CPU { self } else {
      #[cfg(feature = "gpu-runtime")] { (tmp = unsafe { self.to_cpu() }, &tmp).1 }
      #[cfg(not(feature = "gpu-runtime"))] panic!("gpu-runtime not enabled")
    };
    let (p, dim, loc) = (data.ptr, data.elems(), data.loc);
    macro_rules! handle {($ty: ident) => { d.field("data", &Slice::<$ty, _> { ptr: transmute(p), dim, loc }) }; }
    unsafe { handle_all!(data.ty, handle); }
    d.finish()
  }
}

impl Drop for Array {
  fn drop(&mut self) { if self.owned { unsafe { transmute::<_, A::<u8, usize>>(self.slice()); } } }
}

#[pyclass]
struct Func { lib: Lib }

#[pymethods]
impl Func {
  #[new]
  unsafe fn __new__(path: &str, name: &str) -> Self { Self { lib: Lib::new(path, name).expect("failed to load lib") } }
  #[call]
  fn __call__(&self, args: Vec<usize>) { (self.lib.f)(args.as_ptr() as _) }
  fn eval(&self, args: Vec<usize>, n_discard: u32, n_repeat: u32, timeout: u32) -> (f32, bool) {
    eval(self.lib.f, n_discard, n_repeat, timeout, args.as_ptr() as _)
  }
}

#[pymodule]
fn plant(_py: Python, m: &PyModule) -> PyResult<()> {
  m.add_function(wrap_pyfunction!(parallel_init, m)?)?;
  m.add_function(wrap_pyfunction!(array_alloc, m)?)?;
  m.add_function(wrap_pyfunction!(array_borrow, m)?)?;
  m.add_class::<Array>()?;
  m.add_class::<Func>()
}
