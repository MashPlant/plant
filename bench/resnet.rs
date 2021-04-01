use plant::*;

macro_rules! read { ($s: expr, $($arg:tt)*) => { ArrayInit::Data(&std::fs::read(&format!(concat!("resnet_data/", $s), $($arg)*)).unwrap()) }; }

type M = Slice<u8, usize>;

fn conv(id: u32, ichan: u32, ochan: u32, size: u32, kern: u32, stride: u32, pad: u32, add: bool, relu: bool)
  -> (impl Fn(M, Option<M>, M), P<Buf>, Option<P<Buf>>, P<Buf>, Box<Func>) {
  println!("conv: ichan: {}, ochan: {}, size: {}, kern: {}, stride: {}, pad: {}", ichan, ochan, size, kern, stride, pad);
  static TILE_MAP: [([u32; 6], [u32; 12]); 11] = [
    ([3, 64, 224, 7, 2, 3], [32, 1, 1, 1, 1, 1, 16, 1, 7, 1, 1, 1]),
    ([64, 64, 56, 3, 1, 1], [2, 32, 1, 1, 1, 2, 56, 1, 1, 32, 3, 1]),
    ([64, 128, 56, 1, 2, 0], [8, 2, 1, 1, 1, 1, 14, 1, 1, 4, 1, 1]),
    ([64, 128, 56, 3, 2, 1], [4, 32, 1, 1, 1, 4, 14, 1, 1, 2, 3, 3]),
    ([128, 128, 28, 3, 1, 1], [2, 8, 4, 2, 1, 14, 14, 1, 2, 32, 3, 3]),
    ([128, 256, 28, 1, 2, 0], [4, 4, 1, 1, 1, 1, 14, 1, 1, 4, 1, 1]),
    ([128, 256, 28, 3, 2, 1], [4, 4, 1, 1, 1, 7, 14, 1, 1, 1, 1, 3]),
    ([256, 256, 14, 3, 1, 1], [4, 1, 2, 2, 1, 7, 14, 1, 1, 2, 1, 3]),
    ([256, 512, 14, 1, 2, 0], [16, 1, 1, 1, 1, 1, 7, 1, 1, 64, 1, 1]),
    ([256, 512, 14, 3, 2, 1], [16, 1, 1, 1, 1, 1, 7, 1, 1, 2, 1, 1]),
    ([512, 512, 7, 3, 1, 1], [4, 1, 4, 7, 1, 1, 7, 1, 1, 1, 3, 1]),
  ];
  let [ff0, ff1, ff2, xx0, xx1, xx2, yy0, yy1, yy2, rc0, rx0, ry0] =
    TILE_MAP.iter().find(|(k, _)| k == &[ichan, ochan, size, kern, stride, pad]).unwrap().1;
  let f = Func::new(&format!("conv{}", id));
  let a = f.buf("A", F32, In, x![ichan, size, size]); // NCHW
  let w = f.buf("W", F32, In, x![ochan, ichan, kern, kern]); // OIHW
  let bias = f.buf("BIAS", F32, In, x![ochan,]);
  let osize = (size - kern + 2 * pad) / stride + 1;
  let buf_b = f.buf("B", F32, Out, x![ochan, osize, osize]); // NCHW
  let pad_buf = if pad == 0 { a } else {
    let pad_size = (osize - 1) * stride + kern; // <= size + 2 * pad，因为osize中/ stride不一定是整除
    let pad_buf = f.buf("pad_buf", F32, Temp, x![ichan, pad_size, pad_size]).set_loc(Local);
    f.comp("cache_pad", x![ichan, pad_size, pad_size],
      x!(if i1 >= pad && i1 - pad < size && i2 >= pad && i2 - pad < size { a(i0, i1 - pad, i2 - pad) } else { 0f32 }))
      .tags(0..=(if ichan < 32 { 1 } else { 0 }), Parallel).store(pad_buf);
    pad_buf
  };

  let b = f.comp("B", x![ochan, osize, osize, ichan, kern, kern], x!(0f32));
  b.set_expr(x!(pad_buf(i3, i1 * stride + i4, i2 * stride + i5) * w(i0, i3, i4, i5) + b(i0, i1, i2, i3, i4, i5)));
  let mut b_final = x!(b(i0, i1, i2, 0, 0, 0) + bias(i0));
  let add = if add { // add-relu
    let add = f.buf("ADD", F32, In, x![ochan, osize, osize]);
    b_final = x!(max::<f32>(0, b_final + add(i0, i1, i2)));
    Some(add)
  } else {
    if relu { b_final = x!(max::<f32>(0, b_final)); }
    None
  };
  let b_final = f.comp("B_final", x![ochan, osize, osize], b_final);

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

  b.tags(0..=(if ochan / ff0 / ff1 / ff2 < 32 { 5 } else { 0 }), Parallel);
  if yy0 < 32 { b.tag(17, Vectorize); }

  let (ff_local, xx_local, yy_local) = (ff0 * ff1, xx0 * xx1, yy0 * yy1);
  let b_local = f.buf("b_local", F32, Temp, x![ff_local, xx_local, yy_local])
    .set_loc(Local).set_zero_init(true);
  b_local.alloc_at(b, 5);
  b.before(b_final, 6);
  b.store_at(b_local, x![i0 % ff_local, i1 % xx_local, i2 % yy_local]);
  b_final.store(buf_b);
  if pad_buf != a { pad_buf.alloc_at_func(); }

  f.compile_arg("-mprefer-vector-width=512");
  let lib = if let Some(x) = add { f.codegen(&[a, w, bias, x, buf_b]) } else { f.codegen(&[a, w, bias, buf_b]) }.unwrap();
  let (w, bias) = (w.array(read!("conv{}_w", id)), bias.array(read!("conv{}_bias", id)));
  (move |i, add, o| {
    if let Some(x) = add { (lib.f)([i, *w, *bias, x, o].as_ptr()); } else { (lib.f)([i, *w, *bias, o].as_ptr()); }
  }, a, add, buf_b, f)
}

// naive版本，能跑但很慢
// fn conv(id: u32, ichan: u32, ochan: u32, size: u32, kern: u32, stride: u32, pad: u32, add: bool, relu: bool)
//   -> (impl Fn(M, Option<M>, M), P<Buf>, Option<P<Buf>>, P<Buf>, Box<Func>) {
//   let f = Func::new(&format!("conv{}", id));
//   let a = f.buf("A", F32, In, x![ichan, size, size]); // NCHW
//   let w = f.buf("W", F32, In, x![ochan, ichan, kern, kern]); // OIHW
//   let bias = f.buf("BIAS", F32, In, x![ochan,]);
//   let osize = (size - kern + 2 * pad) / stride + 1;
//   let buf_b = f.buf("B", F32, Out, x![ochan, osize, osize]); // NCHW
//   let a_pad = f.comp("A_pad", x![ichan, size + 2 * pad, size + 2 * pad],
//     x!(if i1 >= pad && i1 - pad < size && i2 >= pad && i2 - pad < size { a(i0, i1 - pad, i2 - pad) } else { 0f32 }));
//   a_pad.set_inline(true);
//
//   let b_init = f.comp("B_init", x![ochan, osize, osize], x!(bias(i0)));
//   let b = f.comp("B", x![ochan, osize, osize, ichan, kern, kern], x!(0f32));
//   b.set_expr(x!(a_pad(i3, i1 * stride + i4, i2 * stride + i5) * w(i0, i3, i4, i5) + b(i0, i1, i2, i3, i4, i5)));
//   let (b_final, add) = if add { // add-relu
//     let add = f.buf("ADD", F32, In, x![ochan, osize, osize]);
//     (x!(max::<f32>(0, add(i0, i1, i2) + buf_b(i0, i1, i2))), Some(add))
//   } else {
//     (if relu { x!(max::<f32>(0, buf_b(i0, i1, i2))) } else { x!(buf_b(i0, i1, i2)) }, None)
//   };
//   let b_final = f.comp("B_final", x![ochan, osize, osize], b_final);
//   b_init.before(b, 3).before(b_final, 3);
//   b_init.store(buf_b);
//   b.store_at(buf_b, x![i0, i1, i2]);
//   b_final.store(buf_b);
//
//   let lib = if let Some(x) = add { f.codegen(&[a, w, bias, x, buf_b]) } else { f.codegen(&[a, w, bias, buf_b]) }.unwrap();
//   let (w, bias) = (w.array(read!("conv{}_w", id)), bias.array(read!("conv{}_bias", id)));
//   (move |i, add, o| {
//     if let Some(x) = add { (lib.f)([i, *w, *bias, x, o].as_ptr()); } else { (lib.f)([i, *w, *bias, o].as_ptr()); }
//   }, a, add, buf_b, f)
// }

fn maxpool(chan: u32, size: u32, kern: u32, stride: u32, pad: u32) -> (impl Fn(M, M), P<Buf>, P<Buf>, Box<Func>) {
  let f = Func::new("maxpool");
  let a = f.buf("A", F32, In, x![chan, size, size]);
  let a_pad = f.comp("A_pad", x![chan, size + 2 * pad, size + 2 * pad],
    x!(if i1 >= pad && i1 - pad < size && i2 >= pad && i2 - pad < size { a(i0, i1 - pad, i2 - pad) } else { 0f32 }));
  a_pad.set_inline(true);
  let osize = (size - kern + 2 * pad) / stride + 1;
  let buf_b = f.buf("B", F32, Out, x![chan, osize, osize]);
  let b_init = f.comp("B_init", x![chan, osize, osize], x!(0)); // 初值取0是可行的，因为在relu后，输入都是>=0的
  let b = f.comp("B", x![chan, osize, osize, kern, kern],
    x!(max::<f32>(a_pad(i0, i1 * stride + i3, i2 * stride + i4), buf_b(i0, i1, i2))));
  b_init.before(b, 3);
  b_init.store(buf_b);
  b.store_at(buf_b, x![i0, i1, i2]);
  b.tag(0, Parallel);

  let lib = f.codegen(&[a, buf_b]).unwrap();
  (move |i, o| { (lib.f)([i, o].as_ptr()) }, a, buf_b, f)
}

fn avgpool(chan: u32, size: u32) -> (impl Fn(M, M), P<Buf>, P<Buf>, Box<Func>) {
  let f = Func::new("avgpool");
  let a = f.buf("A", F32, In, x![chan, size, size]);
  let buf_b = f.buf("B", F32, Out, x![chan,]);
  let b_init = f.comp("B_init", x![chan,], x!(0));
  let b = f.comp("B", x![chan, size, size], x!(a(i0, i1, i2) + buf_b(i0)));
  let b_final = f.comp("B_final", x![chan,], x!(buf_b(i0) / ((size * size))));
  b_init.before(b, 1).before(b_final, 1);
  b_init.store(buf_b);
  b.store_at(buf_b, x![i0,]);
  b_final.store(buf_b);

  let lib = f.codegen(&[a, buf_b]).unwrap();
  (move |i, o| { (lib.f)([i, o].as_ptr()) }, a, buf_b, f)
}

fn gemv(m: u32, n: u32) -> (impl Fn(M, M), P<Buf>, P<Buf>, Box<Func>) {
  let f = Func::new("gemv");
  let a = f.buf("A", F32, In, x![n,]);
  let w = f.buf("W", F32, In, x![m, n]);
  let c = f.buf("C", F32, In, x![m,]);
  let buf_b = f.buf("B", F32, Out, x![m,]);
  let b_init = f.comp("B_init", x![m,], x!(c(i0)));
  let b = f.comp("B", x![m, n], x!(a(i1) * w(i0, i1) + buf_b(i0)));
  b_init.store(buf_b);
  b.store_at(buf_b, x![i0,]);
  b_init.before(b, 1);

  let lib = f.codegen(&[a, w, c, buf_b]).unwrap();
  let (w, c) = (w.array(read!("gemv_w",)), c.array(read!("gemv_c",)));
  (move |i, o| { (lib.f)([i, *w, *c, o].as_ptr()) }, a, buf_b, f)
}

// chan，size不变
fn block1(id: u32, chan: u32, size: u32) -> (impl Fn(M, M), P<Buf>, P<Buf>, (Box<Func>, Box<Func>)) {
  let (f1, a1, _, b1, _f1) = conv(id, chan, chan, size, 3, 1, 1, false, true);
  let (f2, a2, add, b2, _f2) = conv(id + 1, chan, chan, size, 3, 1, 1, true, true);
  let size = a1.bytes_val();
  debug_assert!(b1.bytes_val() == size && a2.bytes_val() == size && b2.bytes_val() == size && add.unwrap().bytes_val() == size);
  let b1 = b1.array(ArrayInit::None);
  (move |i, o| {
    debug_assert!(i.dim == size && o.dim == size);
    f1(i, None, *b1);
    f2(*b1, Some(i), o);
  }, a1, b2, (_f1, _f2))
}

// chan*2，size/2
fn block2(id: u32, chan: u32, size: u32) -> (impl Fn(M, M), P<Buf>, P<Buf>, (Box<Func>, Box<Func>)) {
  debug_assert_eq!(size % 2, 0);
  let (f1, a1, _, b1, _f1) = conv(id, chan, chan * 2, size, 1, 2, 0, false, false);
  let (f2, a2, _, b2, _f2) = conv(id + 1, chan, chan * 2, size, 3, 2, 1, false, true);
  let (f3, a3, add, b3, _f3) = conv(id + 2, chan * 2, chan * 2, size / 2, 3, 1, 1, true, true);
  let (f4, a4, b4, _f4) = block1(id + 3, chan * 2, size / 2);
  let (size1, size2) = (a1.bytes_val(), a3.bytes_val());
  debug_assert!(a2.bytes_val() == size1 && b1.bytes_val() == size2 && b2.bytes_val() == size2 && b3.bytes_val() == size2 && add.unwrap().bytes_val() == size2 && a4.bytes_val() == size2 && b4.bytes_val() == size2);
  let (b1, b2, b3) = (b1.array(ArrayInit::None), b2.array(ArrayInit::None), b3.array(ArrayInit::None));
  (move |i, o| {
    debug_assert!(i.dim == size1 && o.dim == size2);
    f1(i, None, *b1);
    f2(i, None, *b2);
    f3(*b2, Some(*b1), *b3);
    f4(*b3, o)
  }, a1, b4, (_f1, _f4.1))
}

fn main() {
  parallel_init(0);

  let (f1, a1, _, b1, _f) = conv(1, 3, 64, 224, 7, 2, 3, false, true);
  let input = a1.array(read!("input",));
  let (f2, a2, b2, _f) = maxpool(64, 112, 3, 2, 1);
  let (f3, a3, b3, _f) = block1(11, 64, 56);
  let (f4, a4, b4, _f) = block1(21, 64, 56);
  let (f5, a5, b5, _f) = block2(31, 64, 56);
  let (f6, a6, b6, _f) = block2(41, 128, 28);
  let (f7, a7, b7, _f) = block2(51, 256, 14);
  let (f8, a8, b8, _f) = avgpool(512, 7);
  let (f9, a9, b9, _f) = gemv(1000, 512);
  debug_assert!(a2.bytes_val() == b1.bytes_val() && a3.bytes_val() == b2.bytes_val() && a4.bytes_val() == b3.bytes_val() && a5.bytes_val() == b4.bytes_val()
    && a6.bytes_val() == b5.bytes_val() && a7.bytes_val() == b6.bytes_val() && a8.bytes_val() == b7.bytes_val() && a9.bytes_val() == b8.bytes_val());
  let (b1, b2, b3, b4, b5, b6, b7, b8, b9) =
    (b1.array(ArrayInit::None), b2.array(ArrayInit::None), b3.array(ArrayInit::None), b4.array(ArrayInit::None), b5.array(ArrayInit::None), b6.array(ArrayInit::None), b7.array(ArrayInit::None), b8.array(ArrayInit::None), b9.array(ArrayInit::None));

  for _ in 0..4 {
    let beg = std::time::Instant::now();
    for _ in 0..2000 {
      f1(*input, None, *b1);
      f2(*b1, *b2);
      f3(*b2, *b3);
      f4(*b3, *b4);
      f5(*b4, *b5);
      f6(*b5, *b6);
      f7(*b6, *b7);
      f8(*b7, *b8);
      f9(*b8, *b9);
    }
    println!("{}s", std::time::Instant::now().duration_since(beg).as_secs_f32() / 2000.0);
  }

  fn softmax(x: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for x in x.iter() { m = m.max(*x); }
    let mut s = 0.0;
    for x in x.iter_mut() { s += (*x = (*x - m).exp(), *x).1; }
    for x in x.iter_mut() { *x /= s; }
  }
  let result = b9.transmute::<f32, _>(1000);
  softmax(result.flat());
  let mut result = result.flat().iter().copied().enumerate().collect::<Vec<_>>();
  result.sort_unstable_by(|&(_, x1), &(_, x2)| {
    use std::cmp::Ordering::*;
    if x1 > x2 { Less } else if x1 < x2 { Greater } else { Equal }
  });
  for (i, x) in &result[0..5] { println!("class = {}, prob = {}", i, x) }
}
