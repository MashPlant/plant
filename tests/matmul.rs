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
  let (tile_i, tile_j, tile_k) = (8, 32, 2);

  let f = Func::new("matmul");
  let (ref i, ref j, ref k) = (f.iter(0), f.iter(1), f.iter(2));
  let a = f.buf("a", I32, In, &[n, s]);
  let b = f.buf("b", I32, In, &[s, m]);
  let c_init = f.comp("C_init", &[(0, n), (0, m)], 0i32);
  let c = f.comp("C", &[(0, n), (0, m), (0, s)], 0i32);
  c.set_expr(a.at(&[i, k]) * b.at(&[k, j]) + c.at(&[i, j, &(k - 1)]));
  c_init.tile(0, 1, tile_i, tile_j);
  c.tile_3(0, 1, 2, tile_i, tile_j, tile_k);
  c.after(c_init, 0);
  c.tag_dim(0, Parallel);
  let buf_c = f.buf("c", I32, Out, &[n, m]);
  c_init.store(buf_c);
  c.store_at(buf_c, &[i, j]);
  f.set_tmp(true); // 避免测试留下文件

  let lib = f.codegen(&[a.into(), b.into(), buf_c.into()]).unwrap();
  let f = unsafe { lib.get::<fn(*const i32, *const i32, *mut i32)>(b"matmul\0").unwrap() };
  let rng = XorShiftRng(19260817);
  let a = Mat::rand((n as usize, s as usize), &rng);
  let b = Mat::rand((s as usize, m as usize), &rng);
  let c = Mat::new((n as usize, m as usize));
  f(a.ptr(), b.ptr(), c.ptr());
  let c1 = matmul_ikj(&a, &b);
  c.assert_eq(&c1);
}
