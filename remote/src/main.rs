use byteorder::{LE, ReadBytesExt, WriteBytesExt};
use tempfile::NamedTempFile;
use std::{io::*, net::*, mem, str, process::Command};
use plant_runtime::*;

fn main() -> Result<()> {
  parallel_init(0);
  let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 8080))?;
  for stream in listener.incoming() {
    let stream = stream?;
    stream.set_nodelay(true)?;
    println!("CONNECTED: connection established with {}", stream.peer_addr()?);
    let mut rd = BufReader::new(stream);
    let (mut n_discard, mut n_repeat, mut timeout) = (0, 0, 0);
    let mut data = Vec::new();
    let mut name = Vec::new();
    loop {
      let op = rd.read_u32::<LE>()?;
      let (arg, opc) = (op >> 2, op & 3);
      match () {
        _ if opc == RemoteOpc::Eval as u32 => {
          debug_assert!(!name.is_empty());
          let (mut f, path) = NamedTempFile::new()?.into_parts();
          let so_path = path.with_extension("so");
          let mut lib_size = arg as usize;
          while lib_size != 0 {
            let buf = rd.fill_buf()?;
            let size = buf.len().min(lib_size);
            let buf = &buf[..size];
            f.write_all(buf)?;
            rd.consume(size);
            lib_size -= size;
          }
          f.flush()?;
          let status = Command::new("gcc").arg(&path).arg("-fPIC").arg("-shared").arg("-o").arg(&so_path).status()?;
          debug_assert!(status.success());
          let lib = unsafe { Lib::new(&so_path, str::from_utf8_unchecked(&name)) }.expect("failed to load lib");
          let ret = eval(lib.f, n_discard, n_repeat, timeout, data.as_ptr() as _);
          rd.get_ref().write_u64::<LE>(((ret.0.to_bits() as u64) << 32) | ret.1 as u64)?;
          println!("EVAL: elapsed = {}, timeout = {}", ret.0, !ret.1);
        }
        _ if opc == RemoteOpc::Init as u32 => {
          data.clear();
          data.reserve(arg as _);
          let rng = XorShiftRng(19260817);
          for _ in 0..arg {
            let (size, ty) = (rd.read_u32::<LE>()? as usize, rd.read_u32::<LE>()?);
            debug_assert!(ty <= Void as _);
            let ty = unsafe { mem::transmute::<_, Type>(ty as u8) };
            let elem = ty.size();
            let arr = Array::<u8, _>::new(size * elem);
            let p = arr.ptr();
            for i in 0..size { unsafe { rng.fill(ty, p.add(i * elem)); } }
            data.push(arr);
            println!("INIT: size = {}, ty = {}", size, ty);
          }
          n_discard = rd.read_u32::<LE>()?;
          n_repeat = rd.read_u32::<LE>()?;
          timeout = rd.read_u32::<LE>()?;
          let name_len = rd.read_u32::<LE>()? as _;
          name.clear();
          name.reserve(name_len);
          unsafe { name.set_len(name_len); }
          rd.read_exact(&mut name)?;
          println!("INIT: n_discard = {}, n_repeat = {}, timeout = {}, name = {:?}",
            n_discard, n_repeat, timeout, unsafe { str::from_utf8_unchecked(&name) });
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
