#[cfg(feature = "gpu-runtime")]
use cuda_runtime_sys::*;

use num::Signed;
use std::{ptr::*, alloc::*, mem::*, ops::*, slice, hash::{Hash, Hasher}};
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
    let buf = box Buf { func: self.into(), name: name.into(), ty, kind, loc: Host, sizes };
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
  // 这里不检查check_arg，因为GPU kern捕获的Buf也会调用arg，它们不一定满足要求
  pub fn arg<'a>(&'a self) -> impl Display + 'a {
    fn2display(move |f| write!(f, "{}{}*__restrict__ {}",
      if self.kind == In { "const " } else { "" }, self.ty, self.name))
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

pub trait Primitive { const TYPE: Type; }

macro_rules! impl_primitive {
  ($($val: ident $ty: ident),*) => {
    $(impl Primitive for $ty { const TYPE: Type = $val; })*
  };
}

impl_primitive!(I8 i8, U8 u8, I16 i16, U16 u16, I32 i32, U32 u32, I64 i64, U64 u64, F32 f32, F64 f64);

// 即runtime buffer，用于测试，T必须是基本类型
#[repr(C)]
pub struct Array<T, D: Dims> {
  pub ptr: NonNull<T>,
  pub dim: D,
  pub loc: Backend,
}

// 和Array的唯一区别在于没有实现Drop
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Slice<T, D: Dims> {
  pub ptr: NonNull<T>,
  pub dim: D,
  pub loc: Backend,
}

impl<T: Primitive, D: Dims> Array<T, D> {
  // 不初始化元素
  pub fn new(dim: D) -> Self {
    let p = unsafe { alloc(Layout::from_size_align_unchecked(dim.total() * size_of::<T>(), 32)) };
    Array { ptr: NonNull::new(p as _).expect("failed to alloc"), dim, loc: CPU }
  }

  // 所有元素初始化为逐字节全0
  pub fn zeroed(dim: D) -> Self {
    let p = unsafe { alloc_zeroed(Layout::from_size_align_unchecked(dim.total() * size_of::<T>(), 32)) };
    Array { ptr: NonNull::new(p as _).expect("failed to alloc"), dim, loc: CPU }
  }

  // 用rng随机初始化每个元素
  pub fn rand(dim: D, rng: &XorShiftRng) -> Self {
    let ret = Self::new(dim);
    let p = ret.ptr() as *mut u8;
    for i in 0..dim.total() { unsafe { rng.fill(T::TYPE, p.add(i * size_of::<T>())); } }
    ret
  }
}

#[cfg(feature = "gpu-runtime")]
impl<T: Primitive, D: Dims> Array<T, D> {
  pub fn new_gpu(dim: D) -> Self {
    let mut p = null_mut();
    unsafe { cudaMalloc(&mut p as _, dim.total() * size_of::<T>()); }
    Array { ptr: NonNull::new(p as _).expect("failed to alloc"), dim, loc: GPU }
  }

  pub fn to_gpu(&self) -> Self {
    debug_assert_eq!(self.loc, CPU);
    let ret = Self::new_gpu(self.dim);
    self.cuda_memcpy(&ret, cudaMemcpyKind::cudaMemcpyHostToDevice);
    ret
  }

  pub fn to_cpu(&self) -> Self {
    debug_assert_eq!(self.loc, GPU);
    let ret = Self::new(self.dim);
    self.cuda_memcpy(&ret, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    ret
  }

  fn cuda_memcpy(&self, dst: &Self, kind: cudaMemcpyKind) {
    unsafe { cudaMemcpy(dst.ptr() as _, self.ptr() as _, self.dim.total() * size_of::<T>(), kind); }
  }
}

impl<T, D: Dims> Drop for Array<T, D> {
  fn drop(&mut self) {
    match self.loc {
      CPU => unsafe { dealloc(self.ptr() as _, Layout::from_size_align_unchecked(self.dim.total() * size_of::<T>(), 32)) },
      GPU => {
        #[cfg(feature = "gpu-runtime")] unsafe { cudaFree(self.ptr() as _); }
        #[cfg(not(feature = "gpu-runtime"))] panic!("gpu-runtime not enabled");
      }
    }
  }
}

impl<T: Primitive, D: Dims> Clone for Array<T, D> {
  fn clone(&self) -> Self {
    let ret;
    match self.loc {
      CPU => {
        ret = Self::new(self.dim);
        unsafe { ret.ptr().copy_from_nonoverlapping(self.ptr(), self.dim.total()); }
      }
      GPU => {
        #[cfg(feature = "gpu-runtime")] {
          ret = Self::new_gpu(self.dim);
          self.cuda_memcpy(&ret, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        }
        #[cfg(not(feature = "gpu-runtime"))] panic!("gpu-runtime not enabled");
      }
    }
    ret
  }
}

impl<T, D: Dims> Deref for Array<T, D> {
  type Target = Slice<T, D>;
  fn deref(&self) -> &Self::Target { unsafe { transmute(self) } }
}

impl<T, D: Dims> DerefMut for Array<T, D> {
  fn deref_mut(&mut self) -> &mut Self::Target { unsafe { transmute(self) } }
}

impl<T, D: Dims> Slice<T, D> {
  pub fn ptr(&self) -> *mut T { self.ptr.as_ptr() }

  pub fn flat(&self) -> &mut [T] {
    debug_assert_eq!(self.loc, CPU);
    unsafe { slice::from_raw_parts_mut(self.ptr(), self.dim.total()) }
  }

  // 第一层维度的长度
  pub fn len(&self) -> usize { self.dim.sub().0 }

  pub fn sub(&self, idx: usize) -> Slice<T, D::Sub> {
    let (n, sub) = self.dim.sub();
    debug_assert!(idx < n);
    unsafe { Slice { ptr: NonNull::new_unchecked(self.ptr().add(idx * sub.total())), dim: sub, loc: self.loc } }
  }

  pub fn assert_eq(&self, rhs: &Self) where D: Debug + PartialEq, T: Debug + PartialEq {
    assert_eq!(self.dim, rhs.dim);
    for (x, y) in self.flat().iter().zip(rhs.flat().iter()) { assert_eq!(x, y); }
  }

  pub fn assert_close(&self, rhs: &Self, threshold: T) where D: Debug + PartialEq, T: Signed + PartialOrd + Copy {
    assert_eq!(self.dim, rhs.dim);
    for (&x, &y) in self.flat().iter().zip(rhs.flat().iter()) { assert!((x - y).abs() < threshold); }
  }
}

// 仅在debug模式下有下标范围检查
impl<T, D: Dims> Index<D> for Slice<T, D> {
  type Output = T;
  fn index(&self, idx: D) -> &Self::Output {
    debug_assert_eq!(self.loc, CPU);
    unsafe { &*self.ptr().add(self.dim.offset(idx)) }
  }
}

impl<T, D: Dims> IndexMut<D> for Slice<T, D> {
  fn index_mut(&mut self, idx: D) -> &mut Self::Output {
    debug_assert_eq!(self.loc, CPU);
    unsafe { &mut *(self.index(idx) as *const _ as *mut _) }
  }
}

impl<T: Debug, D: Dims> Display for Array<T, D> {
  fn fmt(&self, f: &mut Formatter) -> FmtResult { write!(f, "{:?}", &**self) }
}

impl<T: Debug, D: Dims> Debug for Array<T, D> {
  fn fmt(&self, f: &mut Formatter) -> FmtResult { write!(f, "{:?}", &**self) }
}

impl<T: Debug, D: Dims> Display for Slice<T, D> {
  fn fmt(&self, f: &mut Formatter) -> FmtResult { write!(f, "{:?}", self) }
}

impl<T: Debug, D: Dims> Debug for Slice<T, D> {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    // 一维Dims的Sub大小和自身相等
    if size_of::<D::Sub>() == size_of::<D>() {
      write!(f, "{:?}", self.flat())
    } else {
      let mut l = f.debug_list();
      for i in 0..self.len() { l.entry(&self.sub(i)); }
      l.finish()
    }
  }
}

pub trait Dims: Copy {
  type Sub: Dims;
  fn total(self) -> usize;
  fn offset(self, idx: Self) -> usize;
  // 返回第0维的大小和表示剩余维度的Dims，对于usize和(usize,)的实现是直接panic，因为不存在剩余维度
  fn sub(self) -> (usize, Self::Sub) { unimplemented!() }
}

// 一般我不加#[inline(always)]，相信编译器能够优化好
// 但经测试最常见的(usize, usize)，即矩阵情形中，Dims::offset真的没有inline，导致一些程序变慢数十倍
// 而且神奇的是把代码提取出去放到专门的测试项目中，就会inline
impl Dims for usize {
  // 设置Sub和自身相同，Debug实现中利用了这个性质
  type Sub = Self;
  #[inline(always)]
  fn total(self) -> usize { self }
  #[inline(always)]
  fn offset(self, idx: Self) -> usize { (debug_assert!(idx < self), idx).1 }
}

impl Dims for (usize, ) {
  type Sub = Self;
  #[inline(always)]
  fn total(self) -> usize { self.0 }
  #[inline(always)]
  fn offset(self, idx: Self) -> usize { (debug_assert!(idx.0 < self.0), idx.0).1 }
}

macro_rules! impl_dims {
  ($($t: ident)*, $($i: ident)*, $($j: ident)*) => {
    impl Dims for (usize, $($t),*) {
      type Sub = ($($t,)*);
      #[inline(always)]
      fn total(self) -> usize {
        let (i0, $($i),*) = self;
        i0 $(*$i)*
      }
      #[inline(always)]
      fn offset(self, idx: Self) -> usize {
        let (i0, $($i),*) = self;
        let (mut j0, $($j),*) = idx;
        debug_assert!(j0 < i0 $(&&$j < $i)*);
        $(j0 = j0 * $i + $j;)*
        j0
      }
      #[inline(always)]
      fn sub(self) -> (usize, Self::Sub) {
        let (i0, $($i),*) = self;
        (i0, ($($i,)*))
      }
    }
  };
}

impl_dims!(usize, i1, j1);
impl_dims!(usize usize, i1 i2, j1 j2);
impl_dims!(usize usize usize, i1 i2 i3, j1 j2 j3);
impl_dims!(usize usize usize usize, i1 i2 i3 i4, j1 j2 j3 j4);
impl_dims!(usize usize usize usize usize, i1 i2 i3 i4 i5, j1 j2 j3 j4 j5);
