use plant::*;

fn gemm(n: u32, m: u32, s: u32) -> (Vec<P<Buf>>, Box<Func>) {
  let (bi1, bi2, bi3) = (1, 2, 1024);
  let (bj1, bj2, bj3) = (64, 1, 1);
  let bk = 2;

  let f = Func::new("gemm");
  let a = f.buf("a", F32, In, x![n, s]).set_align(128);
  let b = f.buf("b", F32, In, x![s, m]).set_align(128);
  let bt = f.buf("bt", F32, Temp, x![((s / bj1)), ((m * bj1))]).set_align(128);
  let bt_load = bt.load();
  let c_init = f.comp("C_init", x![n, m], x!(0f32));
  let c = f.comp("C", x![n, m, s], x!(0f32));
  let c_final = f.comp("C_final", x![n, m], x!(c(i0, i1, 0)));
  c.set_expr(x!(a(i0, i2) * bt_load(i1 / bj1, i2 * bj1 + i1 % bj1) + c(i0, i1, i2 - 1)));

  let b_trans = f.comp("B_trans", x![((m / bj1)), s, bj1], x!(b(i1, i0 * bj1 + i2)));

  for c in &[c, c_init, c_final] {
    c.split(0, bi1).split(0, bi2).split(0, bi3)
      .split(4, bj1).split(4, bj2).split(4, bj3);
  }

  for c in &[c_init, c_final] {
    // i_o_o_o, i_o_o_i, i_o_i, i_i, j_o_o_o, j_o_o_i, j_o_i, j_i
    c.reorder_n(&[(0, 0), (1, 4), (2, 1), (3, 5), (4, 2), (5, 6), (6, 3), (7, 7)]);
    // i_o_o_o, j_o_o_o, i_o_o_i, j_o_o_i, i_o_i, j_o_i, i_i, j_i
  }

  c.split(8, bk);
  // i_o_o_o, i_o_o_i, i_o_i, i_i, j_o_o_o, j_o_o_i, j_o_i, j_i, k_o, k_i
  c.reorder_n(&[(0, 0), (1, 4), (2, 1), (3, 5), (4, 8), (5, 2), (6, 6), (7, 9), (8, 3), (9, 7)]);
  // i_o_o_o, j_o_o_o, i_o_o_i, j_o_o_i, k_o, i_o_i, j_o_i, k_i, i_i, j_i

  for c in &[c, c_init, c_final] { c.fuse(0); }

  b_trans.before(c_init, 0).before(c, 3).before(c_final, 3);

  b_trans.tag(0, Parallel).tag(2, Vectorize);
  c_init.tag(6, Vectorize);
  c_final.tag(6, Vectorize);
  c.tag(0, Parallel).tag(8, Vectorize);

  let buf_c = f.buf("c", F32, Out, x![n, m]).set_align(128);
  let local_c = f.buf("local_c", F32, Temp, x![bi2, bi1, bj1]).set_align(128).set_loc(Local);
  local_c.alloc_at(c_init, 0);
  c_init.store_at(local_c, x![i0 / bi1 % bi2, i0 % bi1, i1 % bj1]);
  c.store_at(local_c, x![i0 / bi1 % bi2, i0 % bi1, i1 % bj1]);
  c_final.store(buf_c);

  b_trans.store_at(bt, x![i0, i1 * bj1 + i2]);
  bt.alloc_at_func();

  f.compile_arg("-mprefer-vector-width=512");
  (vec![a.p(), b.p(), buf_c.p()], f)
}

fn main() {
  init_log(FUNC_FILTER);
  parallel_init(0);

  let (bufs, f) = gemm(2048, 2048, 2048);
  let e = TimeEvaluator::new(500, 500, std::time::Duration::from_secs(1));
  e.init(&bufs);
  let lib = f.codegen(&bufs).unwrap();
  let (t, _) = e.eval(unsafe { *lib.get(b"gemm_wrapper\0").unwrap() });
  println!("{}s", t);
}
