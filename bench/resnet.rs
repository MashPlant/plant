use plant::*;
use std::{rc::Rc, time::Instant, env, cmp::Ordering::*};

macro_rules! read { ($s: expr, $($arg:tt)*) => { ArrayInit::Data(&std::fs::read(&format!(concat!("resnet_data/", $s), $($arg)*)).unwrap()) }; }

type M = Slice<u8, usize>;

static TILE_MAP: [([u32; 6], [u32; 12]); 26] = [
  // resnet18, 34
  ([3, 64, 224, 7, 2, 3], [16, 1, 4, 1, 1, 1, 16, 1, 7, 3, 1, 7]),
  ([64, 64, 56, 3, 1, 1], [4, 2, 8, 1, 1, 28, 14, 2, 2, 2, 1, 1]),
  ([64, 128, 56, 1, 2, 0], [4, 8, 2, 1, 1, 1, 14, 1, 2, 1, 1, 1]),
  ([64, 128, 56, 3, 2, 1], [16, 1, 8, 1, 1, 14, 14, 2, 2, 16, 1, 3]),
  ([128, 128, 28, 3, 1, 1], [4, 2, 16, 1, 1, 14, 14, 2, 1, 4, 1, 1]),
  ([128, 256, 28, 1, 2, 0], [8, 4, 8, 1, 1, 14, 14, 2, 1, 1, 1, 1]),
  ([128, 256, 28, 3, 2, 1], [8, 2, 16, 1, 1, 1, 14, 1, 1, 2, 1, 1]),
  ([256, 256, 14, 3, 1, 1], [8, 1, 16, 2, 1, 1, 14, 1, 1, 16, 1, 1]),
  ([256, 512, 14, 1, 2, 0], [16, 1, 8, 1, 7, 1, 7, 1, 1, 128, 1, 1]),
  ([256, 512, 14, 3, 2, 1], [8, 2, 32, 1, 1, 14, 7, 2, 1, 4, 1, 1]),
  ([512, 512, 7, 3, 1, 1], [2, 4, 64, 1, 7, 1, 7, 1, 1, 1, 3, 1]),
  // resent50, 101, 152，有5个shape前面出现过了
  // ([3, 64, 224, 7, 2, 3], [16, 1, 4, 1, 1, 1, 16, 1, 7, 3, 1, 7]),
  ([64, 64, 56, 1, 1, 0], [4, 2, 1, 2, 1, 2, 8, 1, 1, 1, 1, 1]),
  // ([64, 64, 56, 3, 1, 1], [4, 2, 8, 1, 1, 28, 14, 2, 2, 2, 1, 1]),
  ([64, 256, 56, 1, 1, 0], [8, 1, 2, 1, 2, 2, 8, 1, 1, 1, 1, 1]),
  ([256, 64, 56, 1, 1, 0], [8, 1, 2, 1, 2, 1, 8, 1, 1, 1, 1, 1]),
  ([256, 128, 56, 1, 2, 0], [16, 2, 2, 1, 1, 1, 14, 1, 4, 1, 1, 1]),
  // ([128, 128, 28, 3, 1, 1], [4, 2, 16, 1, 1, 14, 14, 2, 1, 4, 1, 1]),
  ([128, 512, 28, 1, 1, 0], [4, 1, 8, 1, 1, 1, 14, 2, 1, 8, 1, 1]),
  ([256, 512, 56, 1, 2, 0], [16, 2, 8, 1, 1, 2, 14, 1, 2, 1, 1, 1]),
  ([512, 128, 28, 1, 1, 0], [1, 8, 8, 1, 1, 2, 14, 2, 1, 2, 1, 1]),
  ([512, 256, 28, 1, 2, 0], [8, 2, 2, 1, 1, 1, 14, 1, 2, 2, 1, 1]),
  // ([256, 256, 14, 3, 1, 1], [8, 1, 16, 2, 1, 1, 14, 1, 1, 16, 1, 1]),
  ([256, 1024, 14, 1, 1, 0], [8, 1, 64, 2, 1, 7, 14, 1, 1, 128, 1, 1]),
  ([512, 1024, 28, 1, 2, 0], [16, 1, 32, 2, 1, 1, 14, 2, 1, 2, 1, 1]),
  ([1024, 256, 14, 1, 1, 0], [8, 1, 2, 2, 1, 1, 14, 1, 1, 1024, 1, 1]),
  ([1024, 512, 14, 1, 2, 0], [8, 2, 1, 1, 1, 2, 7, 2, 1, 128, 1, 1]),
  // ([512, 512, 7, 3, 1, 1], [2, 4, 64, 1, 7, 1, 7, 1, 1, 1, 3, 1]),
  ([512, 2048, 7, 1, 1, 0], [4, 1, 4, 7, 1, 1, 7, 1, 1, 1, 1, 1]),
  ([1024, 2048, 14, 1, 2, 0], [4, 16, 1, 1, 1, 7, 7, 2, 1, 8, 1, 1]),
  ([2048, 512, 7, 1, 1, 0], [4, 1, 4, 7, 1, 1, 7, 1, 1, 2048, 1, 1]),
];

fn conv(ic: u32, oc: u32, size: u32, kern: u32, stride: u32, pad: u32, add: u32, relu: u32) -> (impl Fn(M, Option<M>), M) {
  let name = format!("ic{}_oc{}_size{}_kern{}_stride{}_pad{}_add{}_relu{}", ic, oc, size, kern, stride, pad, add, relu);
  let f = Func::new(&name);
  let a = f.buf("A", F32, In, x![ic, size, size]); // NCHW
  let w = f.buf("W", F32, In, x![oc, ic, kern, kern]); // OIHW
  let bias = f.buf("BIAS", F32, In, x![oc,]);
  let osize = (size - kern + 2 * pad) / stride + 1;
  let buf_add = if add != 0 { Some(f.buf("ADD", F32, In, x![oc, osize, osize])) } else { None };
  let buf_b = f.buf("B", F32, Out, x![oc, osize, osize]); // NCHW

  static mut LIB_CACHE: Vec<([u32; 8], Rc<Lib>)> = Vec::new();
  let lib_cache = unsafe { &mut LIB_CACHE };
  let lib = if let Some((_, x)) = lib_cache.iter().find(|(k, _)|
    k == &[ic, oc, size, kern, stride, pad, add, relu]) {
    println!("{} reused", name);
    x.clone()
  } else {
    println!("{} compiling", name);
    let [ff0, ff1, ff2, xx0, xx1, xx2, yy0, yy1, yy2, rc0, rx0, ry0] =
      TILE_MAP.iter().find(|(k, _)| k == &[ic, oc, size, kern, stride, pad]).unwrap().1;

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
    let mut b_final = x!(b(i0, i1, i2, 0, 0, 0) + bias(i0));
    if let Some(x) = buf_add { // add-relu
      b_final = x!(max::<f32>(0, b_final + x(i0, i1, i2)))
    } else if relu != 0 { b_final = x!(max::<f32>(0, b_final)); }
    let b_final = f.comp("B_final", x![oc, osize, osize], b_final);

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
    let lib = Rc::new(if let Some(x) = buf_add { f.codegen(&[a, w, bias, x, buf_b]) } else { f.codegen(&[a, w, bias, buf_b]) }.unwrap());
    lib_cache.push(([ic, oc, size, kern, stride, pad, add, relu], lib.clone()));
    lib
  };

  static mut ID: u32 = 0;
  let id = unsafe { (ID, ID += 1).0 };
  let (w, bias, b) = (w.array(read!("conv{}_w", id)), bias.array(read!("conv{}_b", id)), buf_b.array(ArrayInit::None));
  let b1 = *b;
  (move |i, add| {
    if let Some(x) = add { (lib.f)([i, *w, *bias, x, *b].as_ptr()); } else { (lib.f)([i, *w, *bias, *b].as_ptr()); }
  }, b1)
}

// naive版本，能跑但很慢
// fn conv(ic: u32, oc: u32, size: u32, kern: u32, stride: u32, pad: u32, add: u32, relu: u32)
//   -> (impl Fn(M, Option<M>), M) {
//   println!("ic: {}, oc: {}, size: {}, kern: {}, stride: {}, pad: {}", ic, oc, size, kern, stride, pad);
//
//   let name = format!("ic{}_oc{}_size{}_kern{}_stride{}_pad{}_add{}_relu{}", ic, oc, size, kern, stride, pad, add, relu);
//   let f = Func::new(&name);
//   let a = f.buf("A", F32, In, x![ic, size, size]); // NCHW
//   let w = f.buf("W", F32, In, x![oc, ic, kern, kern]); // OIHW
//   let bias = f.buf("BIAS", F32, In, x![oc,]);
//   let osize = (size - kern + 2 * pad) / stride + 1;
//   let buf_b = f.buf("B", F32, Out, x![oc, osize, osize]); // NCHW
//   let a_pad = f.comp("A_pad", x![ic, size + 2 * pad, size + 2 * pad],
//     x!(if i1 >= pad && i1 - pad < size && i2 >= pad && i2 - pad < size { a(i0, i1 - pad, i2 - pad) } else { 0f32 }));
//   a_pad.set_inline(true);
//
//   let b_init = f.comp("B_init", x![oc, osize, osize], x!(bias(i0)));
//   let b = f.comp("B", x![oc, osize, osize, ic, kern, kern], x!(0f32));
//   b.set_expr(x!(a_pad(i3, i1 * stride + i4, i2 * stride + i5) * w(i0, i3, i4, i5) + b(i0, i1, i2, i3, i4, i5)));
//   let (b_final, add) = if add != 0 { // add-relu
//     let add = f.buf("ADD", F32, In, x![oc, osize, osize]);
//     (x!(max::<f32>(0, add(i0, i1, i2) + buf_b(i0, i1, i2))), Some(add))
//   } else {
//     (if relu != 0 { x!(max::<f32>(0, buf_b(i0, i1, i2))) } else { x!(buf_b(i0, i1, i2)) }, None)
//   };
//   let b_final = f.comp("B_final", x![oc, osize, osize], b_final);
//   b_init.before(b, 3).before(b_final, 3);
//   b_init.store(buf_b);
//   b.store_at(buf_b, x![i0, i1, i2]);
//   b_final.store(buf_b);
//
//   let lib = if let Some(x) = add { f.codegen(&[a, w, bias, x, buf_b]) } else { f.codegen(&[a, w, bias, buf_b]) }.unwrap();
//
//   static mut ID: u32 = 0;
//   let id = unsafe { (ID, ID += 1).0 };
//   let (w, bias, b) = (w.array(read!("conv{}_w", id)), bias.array(read!("conv{}_b", id)), buf_b.array(ArrayInit::None));
//   let b1 = *b;
//   (move |i, add| {
//     if let Some(x) = add { (lib.f)([i, *w, *bias, x, *b].as_ptr()); } else { (lib.f)([i, *w, *bias, *b].as_ptr()); }
//   }, b1)
// }

fn maxpool(chan: u32, size: u32, kern: u32, stride: u32, pad: u32) -> (impl Fn(M), M) {
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
  let b = buf_b.array(ArrayInit::None);
  let b1 = *b;
  (move |i| { (lib.f)([i, *b].as_ptr()) }, b1)
}

fn avgpool(chan: u32, size: u32) -> (impl Fn(M), M) {
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
  let b = buf_b.array(ArrayInit::None);
  let b1 = *b;
  (move |i| { (lib.f)([i, *b].as_ptr()) }, b1)
}

fn gemv(m: u32, n: u32) -> (impl Fn(M), M) {
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
  b.tag(0, Parallel);

  let lib = f.codegen(&[a, w, c, buf_b]).unwrap();
  let (w, c, b) = (w.array(read!("gemv_w",)), c.array(read!("gemv_b",)), buf_b.array(ArrayInit::None));
  let b1 = *b;
  (move |i| { (lib.f)([i, *w, *c, *b].as_ptr()) }, b1)
}

fn block(inplanes: u32, planes: u32, size: u32, stride: u32, bottleneck: bool) -> (Box<dyn Fn(M)>, M) {
  let expansion = if bottleneck { 4 } else { 1 };
  let downsample = stride != 1 || inplanes != planes * expansion;
  if bottleneck {
    let (f1, b1) = conv(inplanes, planes, size, 1, stride, 0, 0, 1);
    let (f2, b2) = conv(planes, planes, size / stride, 3, 1, 1, 0, 1);
    let (f3, b3) = conv(planes, planes * expansion, size / stride, 1, 1, 0, 1, 1);
    let f4 = if downsample { Some(conv(inplanes, planes * expansion, size, 1, stride, 0, 0, 0)) } else { None };
    (Box::new(move |i| {
      if let Some((f4, _)) = &f4 { f4(i, None); }
      f1(i, None);
      f2(b1, None);
      f3(b2, Some(if let Some((_, b4)) = f4 { b4 } else { i }));
    }), b3)
  } else {
    let (f1, b1) = conv(inplanes, planes, size, 3, stride, 1, 0, 1);
    let (f2, b2) = conv(planes, planes, size / stride, 3, 1, 1, 1, 1);
    let f3 = if downsample { Some(conv(inplanes, planes * expansion, size, 1, stride, 0, 0, 0)) } else { None };
    (Box::new(move |i| {
      if let Some((f3, _)) = &f3 { f3(i, None); }
      f1(i, None);
      f2(b1, Some(if let Some((_, b3)) = f3 { b3 } else { i }));
    }), b2)
  }
}

fn layer(inplanes: u32, planes: u32, blocks: u32, size: u32, stride: u32, bottleneck: bool) -> (impl Fn(M), M) {
  let expansion = if bottleneck { 4 } else { 1 };
  let mut layers = Vec::with_capacity(blocks as _);
  layers.push(block(inplanes, planes, size, stride, bottleneck));
  for _ in 1..blocks {
    layers.push(block(planes * expansion, planes, size / stride, 1, bottleneck));
  }
  let b = layers.last().unwrap().1;
  (move |mut i| for (f, b) in &layers { f((i, i = *b).0); }, b)
}

fn main() {
  parallel_init(0);

  let args = env::args().collect::<Vec<_>>();
  assert_eq!(args.len(), 3, "usage: cargo run --bin resnet <layer> <repeat>");
  let repeat = args[2].parse::<u32>().unwrap();
  let (blocks, bottleneck) = match args[1].as_str() {
    "18" => (&[2, 2, 2, 2], false),
    "34" => (&[3, 4, 6, 3], false),
    "50" => (&[3, 4, 6, 3], true),
    "101" => (&[3, 4, 23, 3], true),
    "152" => (&[3, 8, 36, 3], true),
    x => panic!("expect 1st argument to be [18, 34, 50, 101, 152], found {}", x),
  };
  let expansion = if bottleneck { 4 } else { 1 };

  let input = Func::new("_").buf("input", F32, In, x![3, 224, 224]).array(read!("input",));
  let (f1, b1) = conv(3, 64, 224, 7, 2, 3, 0, 1);
  let (f2, b2) = maxpool(64, 112, 3, 2, 1);
  let (f3, b3) = layer(64, 64, blocks[0], 56, 1, bottleneck);
  let (f4, b4) = layer(64 * expansion, 128, blocks[1], 56, 2, bottleneck);
  let (f5, b5) = layer(128 * expansion, 256, blocks[2], 28, 2, bottleneck);
  let (f6, b6) = layer(256 * expansion, 512, blocks[3], 14, 2, bottleneck);
  let (f7, b7) = avgpool(512 * expansion, 7);
  let (f8, b8) = gemv(1000, 512 * expansion);

  for _ in 0..4 {
    let beg = Instant::now();
    for _ in 0..repeat {
      f1(*input, None);
      f2(b1);
      f3(b2);
      f4(b3);
      f5(b4);
      f6(b5);
      f7(b6);
      f8(b7);
    }
    println!("{}s", Instant::now().duration_since(beg).as_secs_f32() / repeat as f32);
  }

  fn softmax(x: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for x in x.iter() { m = m.max(*x); }
    let mut s = 0.0;
    for x in x.iter_mut() { s += (*x = (*x - m).exp(), *x).1; }
    for x in x.iter_mut() { *x /= s; }
  }
  let result = b8.transmute::<f32, _>(1000);
  softmax(result.flat());
  let mut result = result.flat().iter().copied().enumerate().collect::<Vec<_>>();
  result.sort_unstable_by(|&(_, x1), &(_, x2)| if x1 > x2 { Less } else if x1 < x2 { Greater } else { Equal });
  for (i, x) in &result[0..5] { println!("class = {}, prob = {}", i, x) }
}
