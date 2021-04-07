use test_case::test_case;
use plant::*;

type Mat = Array<i32, (usize, usize)>;

pub fn matmul_ikj(a: &Mat, b: &Mat) -> Mat {
  let (n, s, m) = (a.dim.0, a.dim.1, b.dim.1);
  let mut c = Mat::zeroed((n, m));
  for i in 0..n {
    for k in 0..s {
      for j in 0..m {
        c[(i, j)] = c[(i, j)].wrapping_add(a[(i, k)].wrapping_mul(b[(k, j)]));
      }
    }
  }
  c
}

#[test_case(8, 9, 10)]
#[test_case(345, 567, 789)]
#[test_case(1024, 1024, 1024)]
fn matmul(n: u32, m: u32, s: u32) {
  parallel_init_default();
  let (tile_i, tile_j, tile_k) = (8, 32, 2);

  let f = Func::new("matmul");
  let a = f.buf("a", I32, In, x![n, s]);
  let b = f.buf("b", I32, In, x![s, m]);
  let c_init = f.comp("C_init", x![n, m], x!(0));
  let c = f.comp("C", x![n, m, s], x!(0));
  c.set_expr(x!(a(i0, i2) * b(i2, i1) + c(i0, i1, i2 - 1)));
  c_init.tile(0, 1, tile_i, tile_j);
  c.tile_3(0, 1, 2, tile_i, tile_j, tile_k);
  c.after(c_init, 0);
  c.tag(0, Parallel);
  let buf_c = f.buf("c", I32, Out, x![n, m]);
  c_init.store(buf_c);
  c.store_at(buf_c, x![i0, i1]);
  f.set_tmp(true); // 避免测试留下文件

  let lib = f.codegen(&[a.into(), b.into(), buf_c.into()]).unwrap();
  let rng = XorShiftRng(19260817);
  let a = Mat::rand((n as usize, s as usize), &rng);
  let b = Mat::rand((s as usize, m as usize), &rng);
  let c = Mat::new((n as usize, m as usize));
  (lib.f)([a.wrapper(), b.wrapper(), c.wrapper()].as_ptr());
  let c1 = matmul_ikj(&a, &b);
  c.assert_eq(&c1);
}
