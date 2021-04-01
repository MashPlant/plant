use plant::*;

fn conv(batch: u32, in_channel: u32, out_channel: u32, in_size: u32, kernel: u32, pad: u32) -> (Vec<P<Buf>>, Box<Func>) {
  let f = Func::new("conv");
  let a = f.buf("A", F32, In, x![batch, in_channel, in_size, in_size]).set_loc(Global); // NCHW
  let w = f.buf("W", F32, In, x![out_channel, in_channel, kernel, kernel]).set_loc(Global); // OIHW
  let out_size = in_size + 2 * pad - kernel + 1;
  let buf_b = f.buf("B", F32, Out, x![batch, out_channel, out_size, out_size]).set_loc(Global); // NCHW
  let a_pad = f.comp("A_pad", x![batch, in_channel, in_size + 2 * pad, in_size + 2 * pad],
    x!(if i2 >= pad && i2 - pad < in_size && i3 >= pad && i3 - pad < in_size { a(i0, i1, i2 - pad, i3 - pad) } else { 0f32 }));
  a_pad.set_inline(true);
  // [2(nn_c_o_i), 4(nn_c_i), 2(ff_c_o_i), 2(xx_c_i)]
  let b_local = f.buf("b_local", F32, Temp, x![2, 4, 2, 2])
    .set_loc(Local).set_zero_init(true);
  // [2(nn_c_o_i), 4(nn_c_i), 4(rc_i), 9(yy_c_o_o_i + ry_i), 4(xx_c_i + rx_i)]
  let pad_shared = f.buf("pad_shared", F32, Temp, x![2, 4, 4, 9, 4]).set_loc(Shared);
  let pad_load = pad_shared.load();
  // [32(ff_c_o_o_i), 2(ff_c_o_i), 4(rc_i), 3(ry_i), 3(rx_i)]
  let w_shared = f.buf("w_shared", F32, Temp, x![32, 2, 4, 3, 3]).set_loc(Shared);
  let w_load = w_shared.load();
  let b = f.comp("B", x![batch, out_channel, out_size, out_size, in_channel, kernel, kernel], x!(0f32));
  b.set_expr(x!(pad_load(i0/4%2, i0%4, i4%4, i2%7 + i5, i3%2 + i6) * w_load(i1/2%32, i1%2, i4%4, i5, i6) + b(i0, i1, i2, i3, i4, i5, i6)));
  let b_final = f.comp("B_final", x![batch, out_channel, out_size, out_size], x!(b(i0, i1, i2, i3, 0, 0, 0)));

  for b in &[b, b_final] {
    b.split(0, 4).split(0, 2) // nn
      .split(3, 2).split(3, 32)// ff
      .split(6, 7). // xx
      split(8, 2); // yy
  }
  b.split(10, 4); // rc
  //           32         2       4             8          32         2             2           7             7       2      64     4     3     3
  // nn_c_o_o_o_o, nn_c_o_i, nn_c_i, ff_c_o_o_o_o, ff_c_o_o_i, ff_c_o_i, yy_c_o_o_o_o, yy_c_o_o_i, xx_c_o_o_o_o, xx_c_i, rc_o_o, rc_i, ry_i, rx_i
  b.reorder_n(&[(0, 0), (1, 3), (2, 6), (3, 8), (4, 4), (5, 7), (6, 10), (7, 12), (8, 1), (9, 5), (10, 11), (11, 13), (12, 2), (13, 9), ]);
  // nn_c_o_o_o_o, ff_c_o_o_o_o, yy_c_o_o_o_o, xx_c_o_o_o_o, ff_c_o_o_i, yy_c_o_o_i, rc_o_o, ry_i, nn_c_o_i, ff_c_o_i, rc_i, rx_i, nn_c_i, xx_c_i

  // nn_c_o_o_o_o, nn_c_o_i, nn_c_i, ff_c_o_o_o_o, ff_c_o_o_i, ff_c_o_i, yy_c_o_o_o_o, yy_c_o_o_i, xx_c_o_o_o_o, xx_c_i
  b_final.reorder_n(&[(0, 0), (1, 3), (2, 6), (3, 8), (4, 4), (5, 7), (6, 1), (7, 5), (8, 2), (9, 9), ]);
  // nn_c_o_o_o_o, ff_c_o_o_o_o, yy_c_o_o_o_o, xx_c_o_o_o_o, ff_c_o_o_i, yy_c_o_o_i, nn_c_o_i, ff_c_o_i, nn_c_i, xx_c_i

  b.tags(0..=3, GPUBlockX).tags(4..=5, GPUThreadX).tags(8..=13, UnrollExplicit);

  // threadIdx.x = i4*7+i5, copy_iter = i7 * 224 + threadIdx.x = i7*224+i4*7+i5
  let pad_cache = f.comp("pad_cache", x![32, 8, 2, 7, 32, 7, 64, 2, 4],
    x!(a_pad(i0*8 + (i7*224+i4*7+i5)/36, i6*4 + (i7*224+i4*7+i5)%36/9, i2*7 + (i7*224+i4*7+i5)%9, i3*2 + i8)))
    .set_cond(Some(x!(i7 < 1 || i4*7+i5 < 64))) // 等价于i7 * 896 + (i4*7+i5) * 4 + i8 < 1152，这样写利于优化
    .store_at(pad_shared, x![0, 0, 0, 0, i7 * 896 + (i4*7+i5) * 4 + i8])
    .tags(7..=8, UnrollExplicit);
  let w_cache = f.comp("w_cache", x![32, 8, 2, 7, 32, 7, 64, 4, 3],
    x!(w(i1*64 + (i7*224+i4*7+i5)/12, i6*4, (i7*224+i4*7+i5)%12, i8)))
    .set_cond(Some(x!(i7 < 3 || i4*7+i5 < 96))) // 等价于i7 * 672 + (i4*7+i5) * 3 + i8 < 2304，这样写利于优化
    .store_at(w_shared, x![0, 0, 0, 0, i7 * 672 + (i4*7+i5) * 3 + i8])
    .tags(7..=8, UnrollExplicit);

  let sync1 = f.comp("sync1", x![32, 8, 2, 7, 32, 7, 64], Sync);
  let sync2 = f.comp("sync2", x![32, 8, 2, 7, 32, 7, 64], Sync);
  let sync_stream = f.comp("sync_stream", x![], x!(cudaStreamSynchronize::<()>(0)));
  sync1.before(pad_cache, 7).before(w_cache, 7).before(sync2, 7)
    .before(b, 7).before(b_final, 6).before(sync_stream, 0);
  b_local.alloc_at(sync1, 5);
  pad_shared.alloc_at(sync1, 6);
  w_shared.alloc_at(sync1, 6);

  b.store_at(b_local, x![i0/4%2, i0%4, i1%2, i3%2]);
  b_final.store(buf_b);

  f.set_backend(GPU);
  (vec![a, w, buf_b], f)
}

fn main() {
  init_log(FUNC_FILTER);

  let (bufs, f) = conv(256, 256, 512, 14, 3, 1);
  let e = TimeEvaluator::new(200, 200, std::time::Duration::from_secs(1));
  e.init(&bufs);
  let lib = f.codegen(&bufs).unwrap();
  let (t, _) = e.eval(lib.f);
  println!("{}s", t);
}
