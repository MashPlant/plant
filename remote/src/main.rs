use byteorder::{LE, ReadBytesExt, WriteBytesExt};
use tempfile::NamedTempFile;
use std::{io::*, net::*, mem::*, path::*, str::from_utf8_unchecked as utf8, fs::File, process::Command};
use plant_runtime::*;

fn copy(mut f: File, rd: &mut BufReader<TcpStream>, mut copy_len: usize) -> Result<()> {
  while copy_len != 0 {
    let buf = rd.fill_buf()?;
    let len = buf.len().min(copy_len);
    let buf = &buf[..len];
    f.write_all(buf)?;
    rd.consume(len);
    copy_len -= len;
  }
  f.flush()
}

fn read_bytes(rd: &mut BufReader<TcpStream>, dst: &mut Vec<u8>, len: usize) -> Result<()> {
  dst.clear();
  dst.reserve(len);
  unsafe { dst.set_len(len); }
  rd.read_exact(dst)
}

fn compile_so(obj_path: &Path) -> Result<PathBuf> {
  let so_path = obj_path.with_extension("so");
  let status = Command::new("gcc").arg(&obj_path).arg("-fPIC").arg("-shared").arg("-o").arg(&so_path).status()?;
  debug_assert!(status.success());
  Ok(so_path)
}

fn main() -> Result<()> {
  parallel_init(0);
  let listener = TcpListener::bind((Ipv4Addr::UNSPECIFIED, 8888))?;
  println!("STARTED: listening on {}", listener.local_addr()?);
  for stream in listener.incoming() {
    let stream = stream?;
    stream.set_nodelay(true)?;
    println!("CONNECTED: connection established with {}", stream.peer_addr()?);
    let ref mut rd = BufReader::new(stream);
    let (mut n_discard, mut n_repeat, mut timeout) = (0, 0, 0);
    let mut data = Vec::new();
    let mut name = Vec::new();
    loop {
      let op = rd.read_u32::<LE>()?;
      let (arg, opc) = (op >> 2, op & 3);
      match () {
        _ if opc == RemoteOpc::Eval as u32 => {
          debug_assert!(!name.is_empty());
          let (f, path) = NamedTempFile::new()?.into_parts();
          copy(f, rd, arg as _)?;
          let so_path = compile_so(&path)?;
          let lib = unsafe { Lib::new(&so_path, utf8(&name)) }.expect("failed to load lib");
          let ret = eval(lib.f, n_discard, n_repeat, timeout, data.as_ptr() as _);
          rd.get_ref().write_u64::<LE>(((ret.0.to_bits() as u64) << 32) | ret.1 as u64)?;
          println!("EVAL: elapsed = {}s, timeout = {}", ret.0, !ret.1);
        }
        _ if opc == RemoteOpc::Init as u32 => {
          data.clear();
          data.reserve(arg as _);
          let rng = XorShiftRng(19260817);
          for _ in 0..arg {
            let (len, ty) = (rd.read_u32::<LE>()? as usize, rd.read_u32::<LE>()?);
            debug_assert!(ty <= Void as _);
            let ty = unsafe { transmute::<_, Type>(ty as u8) };
            let elem = ty.size();
            let arr = Array::<u8, _>::new(len * elem);
            let p = arr.ptr();
            for i in 0..len { unsafe { rng.fill(ty, p.add(i * elem)); } }
            data.push(arr);
            println!("INIT: len = {}, ty = {}", len, ty);
          }
          n_discard = rd.read_u32::<LE>()?;
          n_repeat = rd.read_u32::<LE>()?;
          timeout = rd.read_u32::<LE>()?;
          let name_len = rd.read_u32::<LE>()?;
          read_bytes(rd, &mut name, name_len as _)?;
          println!("INIT: n_discard = {}, n_repeat = {}, timeout = {}, name = {:?}",
            n_discard, n_repeat, timeout, unsafe { utf8(&name) });
        }
        _ if opc == RemoteOpc::File as u32 => {
          println!("FILE: arg = {}", arg);
          let ref mut buf = Vec::new();
          for _ in 0..arg {
            let len = rd.read_u32::<LE>()?;
            let (len, compile) = (len >> 1, len & 1);
            read_bytes(rd, buf, len as _)?;
            let name = unsafe { utf8(buf) };
            let len = rd.read_u32::<LE>()?;
            copy(File::create(name)?, rd, len as _)?;
            if compile != 0 { compile_so(name.as_ref())?; }
            println!("FILE: name = {:?}, len = {}, compile = {}", name, len, compile != 0);
          }
          debug_assert!(!buf.is_empty());
          // 编译最后一个文件
          let status = Command::new("g++").arg(unsafe { utf8(buf) })
            .arg("-std=c++17").arg("-ldl").arg("-pthread")
            .arg("-Ofast").arg("-march=native").arg("-o").arg("./a.out").status()?;
          debug_assert!(status.success());
          let mut cmd = Command::new("./a.out");
          let args_len = rd.read_u32::<LE>()?;
          for _ in 0..args_len {
            let arg_len = rd.read_u32::<LE>()?;
            read_bytes(rd, buf, arg_len as _)?;
            cmd.arg(unsafe { utf8(buf) });
          }
          println!("FILE: command = {:?}", cmd);
          let output = cmd.output()?;
          println!("FILE: output = {:?}", output);
          rd.get_ref().write_u32::<LE>(output.stdout.len() as _)?;
          rd.get_ref().write_all(&output.stdout)?;
        }
        _ if opc == RemoteOpc::Close as u32 => {
          println!("CLOSED");
          break;
        }
        _ => panic!("invalid opc {} from {:b}", opc, op),
      }
    }
  }
  Ok(())
}
