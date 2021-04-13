use plant::*;

fn conv(ic: u32, oc: u32, size: u32, kern: u32, stride: u32, pad: u32, cfg: &ConfigEntity)
  -> (Vec<P<Buf>>, Box<Func>) {
  let f = Func::new("conv");
  let a = f.buf("A", F32, In, x![ic, size, size]); // NCHW
  let w = f.buf("W", F32, In, x![oc, ic, kern, kern]); // OIHW
  let bias = f.buf("BIAS", F32, In, x![oc,]);
  let osize = (size - kern + 2 * pad) / stride + 1;
  let buf_b = f.buf("B", F32, Out, x![oc, osize, osize]); // NCHW

  let (ff_sp, xx_sp, yy_sp, rc_sp, rx_sp, ry_sp) =
    (cfg.get("ff_sp"), cfg.get("xx_sp"), cfg.get("yy_sp"), cfg.get("rc_sp"), cfg.get("rx_sp"), cfg.get("ry_sp"));
  let [ff0, ff1, ff2, xx0, xx1, xx2, yy0, yy1, yy2, rc0, rx0, ry0] =
    [ff_sp[0], ff_sp[1], ff_sp[2], xx_sp[0], xx_sp[1], xx_sp[2], yy_sp[0], yy_sp[1], yy_sp[2], rc_sp[0], rx_sp[0], ry_sp[0]];

  let pad_buf = if pad == 0 { a } else {
    let pad_size = (osize - 1) * stride + kern; // <= size + 2 * pad，因为osize中/ stride不一定是整除
    let pad_buf = f.buf("pad_buf", F32, Temp, x![ic, pad_size, pad_size]).set_loc(Local);
    f.comp("cache_pad", x![ic, pad_size, pad_size],
      x!(if i1 >= pad && i1 - pad < size && i2 >= pad && i2 - pad < size { a(i0, i1 - pad, i2 - pad) } else { 0f32 }))
      .tags(0..=(if ic < 32 { 1 } else { 0 }), Parallel).store(pad_buf);
    pad_buf
  };

  let b = f.comp("B", x![oc, osize, osize, ic, kern, kern], x!(0f32));
  b.set_expr(x!(pad_buf(i3, i1 * stride + i4, i2 * stride + i5) * w(i0, i3, i4, i5) + b(i0, i1, i2, i3, i4, i5)));
  let b_final = f.comp("B_final", x![oc, osize, osize], x!(b(i0, i1, i2, 0, 0, 0) + bias(i0)));

  for b in &[b, b_final] {
    b.split(0, ff0).split(0, ff1).split(0, ff2)
      .split(4, xx0).split(4, xx1).split(4, xx2)
      .split(8, yy0).split(8, yy1).split(8, yy2);
  }

  b.split(12, rc0).split(14, rx0).split(16, ry0);
  // ff_o_o_o, ff_o_o_i, ff_o_i, ff_i, yy_o_o_o, yy_o_o_i, yy_o_i, yy_i, xx_o_o_o, xx_o_o_i, xx_o_i, xx_i, rc_o, rc_i, rx_o, rx_i, ry_o, ry_i
  b.reorder_n(&[(0, 0), (1, 4), (2, 8), (3, 1), (4, 5), (5, 9), (6, 12), (7, 14), (8, 16), (9, 2), (10, 6), (11, 10), (12, 13), (13, 15), (14, 17), (15, 3), (16, 7), (17, 11), ]);
  // ff_o_o_o, yy_o_o_o, xx_o_o_o, ff_o_o_i, yy_o_o_i, xx_o_o_i, rc_o, rx_o, ry_o, ff_o_i, yy_o_i, xx_o_i, rc_i, rx_i, ry_i, ff_i, yy_i, xx_i

  // ff_o_o_o, ff_o_o_i, ff_o_i, ff_i, yy_o_o_o, yy_o_o_i, yy_o_i, yy_i, xx_o_o_o, xx_o_o_i, xx_o_i, xx_i
  b_final.reorder_n(&[(0, 0), (1, 4), (2, 8), (3, 1), (4, 5), (5, 9), (6, 2), (7, 6), (8, 10), (9, 3), (10, 7), (11, 11), ]);
  // ff_o_o_o, yy_o_o_o, xx_o_o_o, ff_o_o_i, yy_o_o_i, xx_o_o_i, ff_o_i, yy_o_i, xx_o_i, ff_i, yy_i, xx_i

  b.tags(0..=(if oc / ff0 / ff1 / ff2 < 32 { 5 } else { 0 }), Parallel);
  if yy0 > 1 && yy0 < 32 { b.tag(17, Vectorize); }

  let (ff_local, xx_local, yy_local) = (ff0 * ff1, xx0 * xx1, yy0 * yy1);
  let b_local = f.buf("b_local", F32, Temp, x![ff_local, xx_local, yy_local])
    .set_loc(Local).set_zero_init(true);
  b_local.alloc_at(b, 5);
  b.before(b_final, 6);
  b.store_at(b_local, x![i0 % ff_local, i1 % xx_local, i2 % yy_local]);
  b_final.store(buf_b);
  if pad_buf != a { pad_buf.alloc_at_func(); }

  f.compile_arg("-mprefer-vector-width=512");
  (vec![a, w, bias, buf_b], f)
}

fn main() {
  init_log(TUNER_FILTER);
  parallel_init(0);

  let shapes = [
    (3, 64, 224, 7, 2, 3),
    (64, 64, 56, 3, 1, 1),
    (64, 128, 56, 1, 2, 0),
    (64, 128, 56, 3, 2, 1),
    (128, 128, 28, 3, 1, 1),
    (128, 256, 28, 1, 2, 0),
    (128, 256, 28, 3, 2, 1),
    (256, 256, 14, 3, 1, 1),
    (256, 512, 14, 1, 2, 0),
    (256, 512, 14, 3, 2, 1),
    (512, 512, 7, 3, 1, 1),

    // (3, 64, 224, 7, 2, 3),
    (64, 64, 56, 1, 1, 0),
    // (64, 64, 56, 3, 1, 1),
    (64, 256, 56, 1, 1, 0),
    (256, 64, 56, 1, 1, 0),
    (256, 128, 56, 1, 2, 0),
    // (128, 128, 28, 3, 1, 1),
    (128, 512, 28, 1, 1, 0),
    (256, 512, 56, 1, 2, 0),
    (512, 128, 28, 1, 1, 0),
    (512, 256, 28, 1, 2, 0),
    // (256, 256, 14, 3, 1, 1),
    (256, 1024, 14, 1, 1, 0),
    (512, 1024, 28, 1, 2, 0),
    (1024, 256, 14, 1, 1, 0),
    (1024, 512, 14, 1, 2, 0),
    // (512, 512, 7, 3, 1, 1),
    (512, 2048, 7, 1, 1, 0),
    (1024, 2048, 14, 1, 2, 0),
    (2048, 512, 7, 1, 1, 0),
  ];

  for &(ic, oc, size, kern, stride, pad) in &shapes {
    let space = ConfigSpace::new(move |cfg| conv(ic, oc, size, kern, stride, pad, cfg));
    space.define_split("ff_sp", SplitPolicy::new(oc).set_n_output(4))
      .define_split("xx_sp", SplitPolicy::new(size).set_n_output(4))
      .define_split("yy_sp", SplitPolicy::new(size).set_n_output(4))
      .define_split("rc_sp", SplitPolicy::new(ic).set_n_output(2))
      .define_split("rx_sp", SplitPolicy::new(kern).set_n_output(2))
      .define_split("ry_sp", SplitPolicy::new(kern).set_n_output(2));
    let xgb = XGBModel::new(&space, Loss::Rank, Knob);
    xgb.set_plan_size(64).set_sa_iter(100).set_batch_size(64);
    let tuner = Tuner::new(space, XGB(xgb));
    tuner.evaluator.set_timeout(20).set_n_discard(100).set_n_repeat(100);
    tuner.set_early_stopping(2000).tune(5000);
  }
}
