use plant::*;

fn conv(batch: u32, in_channel: u32, out_channel: u32, in_size: u32, kernel: u32, pad: u32) -> (Vec<P<Buf>>, Box<Func>) {
  let f = Func::new("conv");
  let a = f.buf("A", F32, In, x![batch, in_channel, in_size, in_size]).set_align(128); // NCHW
  let w = f.buf("W", F32, In, x![out_channel, in_channel, kernel, kernel]).set_align(128); // OIHW
  let out_size = in_size + 2 * pad - kernel + 1;
  let buf_b = f.buf("B", F32, Out, x![batch, out_channel, out_size, out_size]).set_align(128); // NCHW
  let a_pad = f.comp("A_pad", x![batch, in_channel, in_size + 2 * pad, in_size + 2 * pad],
    x!(if i2 >= pad && i2 - pad < in_size && i3 >= pad && i3 - pad < in_size { a(i0, i1, i2 - pad, i3 - pad) } else { 0f32 }));
  a_pad.set_inline(true);
  let pad_buf = f.buf("pad", F32, Temp, x![in_channel, ((in_size + 2 * pad)), ((in_size + 2 * pad))])
    .set_loc(Local).set_align(128);
  let cache_pad = f.comp("cache_pad", x![((batch * 4)), in_channel, ((in_size + 2 * pad)), ((in_size + 2 * pad))], // 4 == ff_o_i
    x!(a_pad(i0 % 256, i1, i2, i3)));

  let b_init = f.comp("B_init", x![batch, out_channel, out_size, out_size], x!(0f32));
  let b = f.comp("B", x![batch, out_channel, out_size, out_size, in_channel, kernel, kernel], x!(0f32));
  b.set_expr(x!(pad_buf(i4, i2 + i5, i3 + i6) * w(i1, i4, i5, i6) + b(i0, i1, i2, i3, i4, i5, i6)));

  for b in &[b_init, b] {
    b.split(0, 1).split(0, 1).split(0, 256)
      .split(4, 1).split(4, 4).split(4, 32)
      .split(8, 1).split(8, 2).split(8, 7)
      .split(12, 14).split(12, 1).split(12, 1);
  }
  // nn_o_o_o, nn_o_o_i, nn_o_i, nn_i, ff_o_o_o, ff_o_o_i, ff_o_i, ff_i, yy_o_o_o, yy_o_o_i, yy_o_i, yy_i, xx_o_o_o, xx_o_o_i, xx_o_i, xx_i
  b_init.reorder_n(&[(0, 0), (1, 4), (2, 8), (3, 12), (4, 1), (5, 5), (6, 9), (7, 13), (8, 2), (9, 6), (10, 10), (11, 14), (12, 3), (13, 7), (14, 11), (15, 15), ]);
  // nn_o_o_o, ff_o_o_o, yy_o_o_o, xx_o_o_o, nn_o_o_i, ff_o_o_i, yy_o_o_i, xx_o_o_i, nn_o_i, ff_o_i, yy_o_i, xx_o_i, nn_i, ff_i, yy_i, xx_i

  b.split(16, 1);
  b.split(18, 3);
  b.split(20, 1);
  //        1       256       1     1         4        32       4     1         1         7       2     1         1         1       1    14   256     1     1     3     3     1
  // nn_o_o_o, nn_o_o_i, nn_o_i, nn_i, ff_o_o_o, ff_o_o_i, ff_o_i, ff_i, yy_o_o_o, yy_o_o_i, yy_o_i, yy_i, xx_o_o_o, xx_o_o_i, xx_o_i, xx_i, rc_o, rc_i, ry_o, ry_i, rx_o, rx_i
  b.reorder_n(&[(0, 0), (1, 4), (2, 8), (3, 12), (4, 1), (5, 5), (6, 9), (7, 13), (8, 16), (9, 18), (10, 20), (11, 2), (12, 6), (13, 10), (14, 14), (15, 17), (16, 19), (17, 21), (18, 3), (19, 7), (20, 11), (21, 15), ]);
  // nn_o_o_o, ff_o_o_o, yy_o_o_o, xx_o_o_o, nn_o_o_i, ff_o_o_i, yy_o_o_i, xx_o_o_i, rc_o, ry_o, rx_o, nn_o_i, ff_o_i, yy_o_i, xx_o_i, rc_i, ry_i, rx_i, nn_i, ff_i, yy_i, xx_i

  for b in &[b_init, b] {
    for _ in 0..4 { b.fuse(0); }
  }

  cache_pad.before(b_init, 1).before(b, 3);
  cache_pad.tag(2, Unroll).tag(3, Unroll);
  b_init.tag(11, Vectorize).tag(5, Unroll).tag(6, Unroll);
  b.tag(0, Parallel);
  b.tag(17, Vectorize).tag(6, Unroll).tag(8, Unroll).tag(9, Unroll).tag(12, Unroll);

  pad_buf.alloc_at(cache_pad, 0);
  cache_pad.store_at(pad_buf, x![i1, i2, i3]);
  b_init.store(buf_b);
  b.store_at(buf_b, x![i0, i1, i2, i3]);

  f.compile_arg("-mprefer-vector-width=512");
  (vec![a, w, buf_b], f)
}

fn main() {
  init_log(FUNC_FILTER);
  parallel_init(0);

  let (bufs, f) = conv(256, 256, 512, 14, 3, 1);
  let e = TimeEvaluator::new(200, 200, std::time::Duration::from_secs(1));
  e.init(&bufs);
  let lib = f.codegen(&bufs).unwrap();
  let (t, _) = e.eval(lib.f);
  println!("{}s", t);
}
