use plant::*;

fn gemm(n: u32, m: u32, s: u32) -> (Vec<P<Buf>>, Box<Func>) {
  let f = Func::new("gemm");
  let a = f.buf("a", F32, In, x![n, s]).set_loc(Global);
  let b = f.buf("b", F32, In, x![s, m]).set_loc(Global);
  let buf_c = f.buf("c", F32, Out, x![n, m]).set_loc(Global);
  // [2(i_c_o_o_o_i), 8(i_c_o_o_i), 2(i_c_o_i), 2(i_c_i), 2(k_o_i), 8(k_i)]
  let a_shared = f.buf("a_shared", F32, Temp, x![2, 8, 2, 2, 2, 8]).set_loc(Shared);
  let al = a_shared.load();
  // [2(k_o_i), 8(k_i), 2(j_c_o_o_o_i), 16(j_c_o_o_i), 2(j_c_o_i), 1(j_c_i)]
  let b_shared = f.buf("b_shared", F32, Temp, x![2, 8, 2, 16, 2, 1]).set_loc(Shared);
  let bl = b_shared.load();
  // [2(i_c_o_o_o_i), 2(j_c_o_o_o_i), 2(i_c_o_i), 2(i_c_i), 1(j_c_i), 2(j_c_o_i)]
  let c_local = f.buf("c_local", F32, Temp, x![2, 2, 2, 2, 1, 2])
    .set_loc(Local).set_zero_init(true);

  let c = f.comp("C", x![n, m, s], x!(0f32));
  let c_final = f.comp("C_final", x![n, m], x!(c(i0, i1, 0)));
  c.set_expr(x!(al(i0/32%2, i0/4%8, i0/2%2, i0%2, i2/8%2, i2%8) * bl(i2/8%2, i2%8, i1/32%2, i1/2%16, i1/1%2, i1%1) + c(i0, i1, i2 - 1)));

  for x in &[c_final] {
    x.split(0, 4).split(0, 8).split(0, 2)
      .split(4, 2).split(4, 16).split(4, 2);
    // 32:i0/64 2:i0/32%2 8:i0/4%8 4:i0%4 32:i1/64 2:i1/32%2 16:i1/2%16 2:i1%2
    // i_o_o_o, i_o_o_i,  i_o_i,   i_i,   j_o_o_o, j_o_o_i,  j_o_i,     j_i
    x.reorder_n(&[(0, 0), (1, 4), (2, 2), (3, 6), (4, 3), (5, 7), (6, 1), (7, 5), ]);
    // i_o_o_o, j_o_o_o, i_o_i, j_o_i, i_i, j_i, i_o_o_i, j_o_o_i
    x.fuse(0).fuse(1).fuse(4);
  }

  c.split(0, 2).split(0, 2).split(0, 8).split(0, 2)
    .split(5, 1).split(5, 2).split(5, 16).split(5, 2)
    .split(10, 8).split(10, 2);
  // 32:i0/64     2:i0/32%2    8:i0/4%8   2:i0/2%2 2:i0%2 32:i0/64     2:i1/32%2   16:i1/2%16  2:i1/1%2 1:i1%1 128:i2/16 2:i2/8%2 8:i2%8
  // i_c_o_o_o_o, i_c_o_o_o_i, i_c_o_o_i, i_c_o_i, i_c_i, j_c_o_o_o_o, j_c_o_o_o_i, j_c_o_o_i, j_c_o_i, j_c_i, k_o_o, k_o_i, k_i
  c.reorder_n(&[(0, 0), (1, 5), (2, 2), (3, 7), (4, 10), (5, 11), (6, 3), (7, 8), (8, 12), (9, 4), (10, 9), (11, 1), (12, 6), ]);
  // i_c_o_o_o_o, j_c_o_o_o_o, i_c_o_o_i, j_c_o_o_i, k_o_o, k_o_i, i_c_o_i, j_c_o_i, k_i, i_c_i, j_c_i, i_c_o_o_o_i, j_c_o_o_o_i
  c.fuse(0).fuse(1).fuse(9);

  c.store_at(c_local, x![i0/32%2, i1/32%2, i0/2%2, i0%2, i1%1, i1/1%2]);
  c_final.store_at(buf_c, x![i0, i1]);

  c.tag(0, GPUBlockX).tag(1, GPUThreadX);

  let a_cache = f.comp("a_cache", x![1024, 128, 128, 2, 4], x!(a(i0/32 * 64 + i3 * 32 + i1/4, i2*16 + i1%4 * 4 + i4)));
  a_cache.store_at(a_shared, x![0, 0, 0, 0, 0, i3 * 512 + i1 * 4 + i4])
    .tag(3, UnrollExplicit).tag(4, Vectorize);
  let b_cache = f.comp("b_cache", x![1024, 128, 128, 4, 2], x!(b(i2 * 16 + i3 * 4 + i1/32, i0%32 * 64 + i1%32 * 2 + i4)));
  b_cache.store_at(b_shared, x![0, 0, 0, 0, 0, i3 * 256 + i1 * 2 + i4])
    .tag(3, UnrollExplicit).tag(4, Vectorize);
  c.tag(7, UnrollExplicit).tag(9, UnrollExplicit);
  c_final.tag(4, UnrollExplicit);
  let sync1 = f.comp("sync1", x![1024, 128, 128], syncthreads());
  let sync2 = f.comp("sync2", x![1024, 128, 128], syncthreads());
  let sync_stream = f.comp("sync_stream", x![], x!(cudaStreamSynchronize::<()>(0)));
  sync1.before(a_cache, 3).before(b_cache, 3).before(sync2, 3)
    .before(c, 3).before(c_final, 2).before(sync_stream, 0);
  c_local.alloc_at(sync1, 1);
  a_shared.alloc_at(sync1, 2);
  b_shared.alloc_at(sync1, 2);

  f.set_backend(GPU);
  (vec![a, b, buf_c], f)
}

fn main() {
  init_log(FUNC_FILTER);

  let (args, f) = gemm(2048, 2048, 2048);
  let e = TimeEvaluator::new(500, 500, 1000);
  e.init(&args);
  let lib = f.codegen(&args).unwrap();
  let (t, _) = e.eval(lib.f);
  println!("{}s", t);
}
