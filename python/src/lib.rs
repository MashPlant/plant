use pyo3::{prelude::*, wrap_pyfunction};
use plant_runtime::*;

#[pyfunction]
fn parallel_init(th: u32) { plant_runtime::parallel_init(th) }

#[pyclass]
struct FuncBase { lib: Lib }

#[pymethods]
impl FuncBase {
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
  m.add_class::<FuncBase>()
}
