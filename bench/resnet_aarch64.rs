use byteorder::{LE, ReadBytesExt, WriteBytesExt};
use std::{fs, io::{self, Write, Read}, env, str, net::*};
use plant::*;

const TARGET: &str = "aarch64-linux-gnu";

static mut FILES: Vec<(Box<str>, Vec<u8>)> = Vec::new();

static TILE_MAP: [([u32; 6], [u32; 12]); 26] = [
  ([3, 64, 224, 7, 2, 3], [8, 2, 4, 2, 1, 1, 4, 1, 14, 3, 7, 7]),
  ([64, 64, 56, 3, 1, 1], [8, 2, 4, 1, 8, 1, 8, 1, 1, 8, 3, 3]),
  ([64, 128, 56, 1, 2, 0], [8, 1, 16, 2, 1, 1, 4, 1, 2, 64, 1, 1]),
  ([64, 128, 56, 3, 2, 1], [8, 2, 8, 2, 1, 1, 4, 7, 2, 8, 3, 3]),
  ([128, 128, 28, 3, 1, 1], [2, 4, 4, 7, 1, 4, 4, 7, 1, 8, 3, 1]),
  ([128, 256, 28, 1, 2, 0], [16, 1, 16, 1, 1, 1, 4, 7, 1, 64, 1, 1]),
  ([128, 256, 28, 3, 2, 1], [8, 2, 16, 2, 7, 2, 4, 7, 1, 16, 3, 3]),
  ([256, 256, 14, 3, 1, 1], [2, 2, 64, 7, 2, 1, 2, 7, 1, 2, 3, 1]),
  ([256, 512, 14, 1, 2, 0], [16, 2, 8, 1, 2, 7, 2, 7, 1, 16, 1, 1]),
  ([256, 512, 14, 3, 2, 1], [4, 1, 128, 1, 7, 2, 7, 1, 2, 8, 3, 3]),
  ([512, 512, 7, 3, 1, 1], [4, 8, 16, 1, 7, 1, 7, 1, 1, 32, 3, 3]),
  ([64, 64, 56, 1, 1, 0], [2, 4, 8, 1, 1, 1, 8, 1, 7, 4, 1, 1]),
  ([64, 256, 56, 1, 1, 0], [2, 4, 32, 1, 1, 1, 8, 1, 7, 4, 1, 1]),
  ([256, 64, 56, 1, 1, 0], [8, 8, 1, 1, 1, 8, 8, 7, 1, 32, 1, 1]),
  ([256, 128, 56, 1, 2, 0], [8, 16, 1, 2, 1, 14, 4, 2, 7, 64, 1, 1]),
  ([128, 512, 28, 1, 1, 0], [8, 2, 16, 1, 1, 2, 4, 1, 7, 2, 1, 1]),
  ([256, 512, 56, 1, 2, 0], [8, 32, 2, 2, 1, 1, 4, 2, 7, 64, 1, 1]),
  ([512, 128, 28, 1, 1, 0], [1, 64, 2, 7, 1, 2, 1, 14, 1, 8, 1, 1]),
  ([512, 256, 28, 1, 2, 0], [2, 32, 4, 1, 1, 1, 4, 7, 1, 4, 1, 1]),
  ([256, 1024, 14, 1, 1, 0], [4, 2, 32, 2, 7, 1, 2, 7, 1, 4, 1, 1]),
  ([512, 1024, 28, 1, 2, 0], [8, 32, 2, 2, 1, 14, 4, 7, 1, 32, 1, 1]),
  ([1024, 256, 14, 1, 1, 0], [4, 2, 32, 2, 7, 1, 2, 7, 1, 4, 1, 1]),
  ([1024, 512, 14, 1, 2, 0], [4, 8, 16, 1, 7, 1, 2, 7, 1, 4, 1, 1]),
  ([512, 2048, 7, 1, 1, 0], [2, 2, 512, 1, 1, 7, 7, 1, 1, 8, 1, 1]),
  ([1024, 2048, 14, 1, 2, 0], [2, 64, 4, 7, 1, 1, 2, 7, 1, 16, 1, 1]),
  ([2048, 512, 7, 1, 1, 0], [1, 1, 512, 1, 7, 1, 1, 7, 1, 1, 1, 1]),
];

fn conv(ic: u32, oc: u32, size: u32, kern: u32, stride: u32, pad: u32, add: u32, relu: u32) {
  let name = format!("ic{}_oc{}_size{}_kern{}_stride{}_pad{}_add{}_relu{}", ic, oc, size, kern, stride, pad, add, relu);
  if unsafe { FILES.iter().find(|(n, _)| n.as_ref() == name).is_some() } {
    println!("{} reused", name);
    return;
  }
  println!("{} compiling", name);

  let f = Func::new(&name);
  let a = f.buf("A", F32, In, x![ic, size, size]); // NCHW
  let w = f.buf("W", F32, In, x![oc, ic, kern, kern]); // OIHW
  let bias = f.buf("BIAS", F32, In, x![oc,]);
  let osize = (size - kern + 2 * pad) / stride + 1;
  let buf_add = if add != 0 { Some(f.buf("ADD", F32, In, x![oc, osize, osize])) } else { None };
  let buf_b = f.buf("B", F32, Out, x![oc, osize, osize]); // NCHW

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

  let lib = if let Some(x) = buf_add {
    f.codegen_remote(&[a, w, bias, x, buf_b], TARGET)
  } else { f.codegen_remote(&[a, w, bias, buf_b], TARGET) }.unwrap();
  unsafe { FILES.push((f.name, lib)); }
}

fn maxpool(chan: u32, size: u32, kern: u32, stride: u32, pad: u32) {
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

  let lib = f.codegen_remote(&[a, buf_b], TARGET).unwrap();
  unsafe { FILES.push((f.name, lib)); }
}

fn avgpool(chan: u32, size: u32) {
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

  let lib = f.codegen_remote(&[a, buf_b], TARGET).unwrap();
  unsafe { FILES.push((f.name, lib)); }
}

fn gemv(m: u32, n: u32) {
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

  let lib = f.codegen_remote(&[a, w, c, buf_b], TARGET).unwrap();
  unsafe { FILES.push((f.name, lib)); }
}

fn block(inplanes: u32, planes: u32, size: u32, stride: u32, bottleneck: bool) {
  let expansion = if bottleneck { 4 } else { 1 };
  let downsample = stride != 1 || inplanes != planes * expansion;
  if bottleneck {
    conv(inplanes, planes, size, 1, stride, 0, 0, 1);
    conv(planes, planes, size / stride, 3, 1, 1, 0, 1);
    conv(planes, planes * expansion, size / stride, 1, 1, 0, 1, 1);
    if downsample { conv(inplanes, planes * expansion, size, 1, stride, 0, 0, 0); }
  } else {
    conv(inplanes, planes, size, 3, stride, 1, 0, 1);
    conv(planes, planes, size / stride, 3, 1, 1, 1, 1);
    if downsample { conv(inplanes, planes * expansion, size, 1, stride, 0, 0, 0); }
  }
}

fn layer(inplanes: u32, planes: u32, blocks: u32, size: u32, stride: u32, bottleneck: bool) {
  let expansion = if bottleneck { 4 } else { 1 };
  block(inplanes, planes, size, stride, bottleneck);
  for _ in 1..blocks { block(planes * expansion, planes, size / stride, 1, bottleneck) }
}

fn main() -> io::Result<()> {
  unsafe { num_thread = 6; }

  let args = env::args().collect::<Vec<_>>();
  assert_eq!(args.len(), 3, "usage: cargo run --bin resnet <layer> <repeat>");
  let _repeat = args[2].parse::<u32>().unwrap();
  let (blocks, bottleneck) = match args[1].as_str() {
    "18" => (&[2, 2, 2, 2], false),
    "34" => (&[3, 4, 6, 3], false),
    "50" => (&[3, 4, 6, 3], true),
    "101" => (&[3, 4, 23, 3], true),
    "152" => (&[3, 8, 36, 3], true),
    x => panic!("expect 1st argument to be [18, 34, 50, 101, 152], found {}", x),
  };
  let expansion = if bottleneck { 4 } else { 1 };

  conv(3, 64, 224, 7, 2, 3, 0, 1);
  maxpool(64, 112, 3, 2, 1);
  layer(64, 64, blocks[0], 56, 1, bottleneck);
  layer(64 * expansion, 128, blocks[1], 56, 2, bottleneck);
  layer(128 * expansion, 256, blocks[2], 28, 2, bottleneck);
  layer(256 * expansion, 512, blocks[3], 14, 2, bottleneck);
  avgpool(512 * expansion, 7);
  gemv(1000, 512 * expansion);

  let files = unsafe { &mut FILES };
  let libs_len = files.len();

  for x in fs::read_dir("resnet_data")? {
    let x = x?;
    files.push((x.file_name().to_string_lossy().into(), fs::read(x.path())?));
  }
  files.push(("common.h".into(), include_str!("reference/common.h").into()));
  files.push(("parallel.c".into(), include_str!("../runtime/src/parallel.c").into()));
  files.push(("resnet_deploy.cpp".into(), include_str!("reference/resnet_deploy.cpp").into()));

  let mut s = TcpStream::connect("192.168.0.2:8888")?;
  s.write_u32::<LE>(((files.len() as u32) << 2) | RemoteOpc::File as u32)?;
  for (i, (name, data)) in files.iter().enumerate() {
    s.write_u32::<LE>(((name.len() as u32) << 1) | (i < libs_len) as u32)?;
    s.write_all(name.as_bytes())?;
    s.write_u32::<LE>(data.len() as _)?;
    s.write_all(&data)?;
  }
  s.write_u32::<LE>((args.len() - 1) as _)?;
  for arg in &args[1..] {
    s.write_u32::<LE>(arg.len() as _)?;
    s.write_all(arg.as_bytes())?;
  }
  s.flush()?;

  let ret_len = s.read_u32::<LE>()?;
  let mut ret = vec![0; ret_len as _];
  s.read_exact(&mut ret)?;
  println!("{}", str::from_utf8(&ret).unwrap());

  s.write_u32::<LE>(RemoteOpc::Close as _)
}
