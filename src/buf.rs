use std::{hash::{Hash, Hasher}, num::NonZeroU32};
use crate::*;

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum BufKind { In, Out, Temp }

// Host是CPU上malloc的内存，Global是GPU上cudaMalloc的内存
// Local是栈上的内存，可以是CPU或GPU上的，Shared是用GPU上的共享内存，形式上类似栈上的内存，只是有__shared__标记
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum BufLoc { Host, Global, Local, Shared }

#[derive(Debug)]
pub struct Buf {
  pub func: P<Func>,
  pub name: Box<str>,
  pub ty: Type,
  pub kind: BufKind,
  pub loc: BufLoc,
  pub align: Option<NonZeroU32>,
  pub sizes: Box<[Expr]>,
}

impl_try!(&Buf);

// 用Buf::name做hash，仍然用Buf地址判等；作用是在HashSet/Map<Buf>中保证稳定的顺序
// 这一点之所以成立，还因为我使用了AHasher的默认初值作为HashSet/Map的初始状态，这是固定的值，没有任何随机性
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct NameHashBuf(pub P<Buf>);

impl Hash for NameHashBuf {
  fn hash<H: Hasher>(&self, state: &mut H) { self.0.name.hash(state) }
}

impl Func {
  // 默认loc为Host
  pub fn buf(&self, name: &str, ty: Type, kind: BufKind, sizes: Box<[Expr]>) -> &Buf {
    debug_assert!(self.find_buf(name).is_none() && !sizes.is_empty() && ty != Void);
    let buf = box Buf { func: self.into(), name: name.into(), ty, kind, loc: Host, align: None, sizes };
    debug!("buf: create buf {}, sizes = [{}]", name, comma_sep(buf.sizes.iter()));
    let ret = buf.as_ref().p();
    self.p().bufs.push(buf);
    ret.get()
  }
}

#[derive(Debug)]
pub struct AllocInfo {
  pub alloc: P<Comp>,
  pub free: Option<P<Comp>>,
}

impl_try!(AllocInfo);

impl Buf {
  pub fn at(&self, idx: Box<[Expr]>) -> Expr {
    debug_assert_eq!(idx.len(), self.sizes.len());
    let mut idx = idx.into_vec().into_iter();
    let mut x = idx.next()?;
    // 构造(i0 * size1) + i1 ...
    for (i, s) in idx.zip(self.sizes.iter().skip(1)) { x = x * s + i; }
    Load(self.into(), box x)
  }

  impl_setter!(set_loc loc BufLoc);

  pub fn set_align(&self, align: u32) -> &Self {
    self.p().align = NonZeroU32::new(align);
    self
  }

  pub fn clone(&self) -> &Buf {
    self.func.buf(&format!("_clone{}_{}", self.func.new_buf_id(), self.name), self.ty, self.kind, self.sizes.clone())
  }

  // 创建一个identity访问自身的Comp，可用于Comp::cache
  pub fn load(&self) -> &Comp {
    let f = self.func;
    f.comp(&format!("_load{}_{}", f.new_buf_id(), self.name), self.sizes.clone(),
      self.at((0..self.sizes.len() as u32).map(iter).collect()))
      .set_inline(true).store(self).p().get()
  }

  // 在comp的循环层次at的前/后放置Alloc/Free
  // 前/后是当前的，不保证之后添加新的计算后这对Alloc/Free仍然在前/后，alloc_at_func同理
  pub fn alloc_at(&self, comp: &Comp, i: u32) -> AllocInfo {
    debug_assert!(i < comp.loop_dim());
    let mut dom = project_static_dim(comp.schedule());
    let (n, i) = (dom.n_dim() as u32, i + 1);
    dom = dom.project_out(DimType::Set, i, n - i)?;
    let info = self.mk_alloc(dom);
    comp.after_between_pred(&info.alloc, i);
    if let Some(x) = info.free { x.after_between_pred(&comp, i); }
    info
  }

  // 在函数的开头/结尾放置Alloc/Free
  pub fn alloc_at_func(&self) -> AllocInfo {
    let mut f = self.func;
    let info = self.mk_alloc(f.ctx.space_set_alloc(0, 0)?.set_universe()?);
    let idx = f.comps.len() - 1 - info.free.is_some() as usize;
    let c = f.comps.remove(idx);
    f.comps.insert(0, c);
    info
  }

  fn mk_alloc(&self, dom: Set) -> AllocInfo {
    debug!("mk_alloc: dom = {}", dom);
    let f = self.func;
    let alloc = f.comp_raw(dom.copy()?.set_tuple_name(
      cstr(&format!("_alloc{}_{}\0", f.new_comp_id(), self.name)))?, Alloc(self.into())).p();
    let free = if self.loc == Host || self.loc == BufLoc::Global {
      Some(f.comp_raw(dom.set_tuple_name(
        cstr(&format!("_free{}_{}\0", f.new_comp_id(), self.name)))?, Free(self.into())).p())
    } else { None };
    AllocInfo { alloc, free }
  }

  // 返回host Buf
  pub fn auto_transfer(&self) -> &Buf {
    debug_assert_ne!(self.kind, Temp);
    let host = self.clone().p();
    self.set_loc(BufLoc::Global);
    let mut f = self.func;
    let dom = f.ctx.space_set_alloc(0, 0)?.set_universe()?.set_tuple_name(
      cstr(&format!("_memcpy{}_{}\0", f.new_comp_id(), self.name)))?;
    f.comp_raw(dom, if self.kind == In { Memcpy(self.into(), host) } else { Memcpy(host, self.into()) });
    if self.kind == In {
      let idx = f.comps.len() - 1;
      let c = f.comps.remove(idx);
      f.comps.insert(0, c);
    }
    self.alloc_at_func();
    host.get()
  }

  // 检查本Buf是否可以作为函数参数
  pub fn check_arg(&self) {
    debug_assert!(self.kind == In || self.kind == Out);
    debug_assert!(self.loc == Host || self.loc == BufLoc::Global);
  }

  // 输出本Buf作为函数参数的形式
  // 这里不检查check_arg，因为lambda捕获的Buf也会调用arg，它们不一定满足要求
  pub fn arg<'a>(&'a self) -> impl Display + 'a {
    fn2display(move |f| write!(f, "{}{}{}*__restrict__ {}", if self.kind == In { "const " } else { "" }, self.ty,
      fn2display(move |f| if let Some(a) = self.align { write!(f, " __attribute__((aligned({})))", a) } else { Ok(()) }),
      self.name))
  }

  // 输出元素数
  pub fn elems<'a>(&'a self) -> impl Display + 'a {
    fn2display(move |f| write!(f, "{}", sep(self.sizes.iter(), "*")))
  }

  // 输出字节数
  pub fn bytes<'a>(&'a self) -> impl Display + 'a {
    fn2display(move |f| write!(f, "{}*sizeof({})", self.elems(), self.ty))
  }
}
