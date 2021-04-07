use plant::*;

type Mat = Array<i32, (usize, usize)>;

fn triangle_gen(n: u32, backend: Backend) -> Lib {
  let f = Func::new("triangle");
  let a = f.buf("a", I32, Out, x![n, n]);
  let c = c!(for i in 0..n { for j in 0..i { i + j } });
  c.store(a);
  match backend {
    CPU => c.tags(0..=1, Parallel),
    GPU => c.tag(0, GPUBlockX).tag(1, GPUThreadX),
  };
  f.set_tmp(true).set_backend(backend).codegen(&[a]).unwrap()
}

fn triangle_test(a: &Mat) {
  let n = a.dim.0;
  for i in 0..n {
    for j in 0..n {
      assert_eq!(a[(i, j)], if j < i { (i + j) as _ } else { 0 });
    }
  }
}

#[test]
fn triangle() {
  parallel_init_default();
  let lib = triangle_gen(100, CPU);
  let a = Mat::zeroed((100, 100));
  (lib.f)([a.wrapper()].as_ptr());
  triangle_test(&a);
}

#[test]
#[cfg(feature = "gpu-runtime")]
fn triangle_gpu() {
  let lib = triangle_gen(100, GPU);
  let a = Mat::zeroed((100, 100)).to_gpu();
  (lib.f)([a.wrapper()].as_ptr());
  triangle_test(&a.to_cpu());
}
