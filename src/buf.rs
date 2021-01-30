use std::{ptr::*, alloc::*, mem::*, ops::*, slice};
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
  pub sizes: Vec<Expr>,
}

impl_try!(&Buf);

impl Func {
  // 默认loc为Host
  pub fn buf<E: IntoExpr>(&self, name: &str, ty: Type, kind: BufKind, sizes: impl IntoIterator<Item=E>) -> &Buf {
    let sizes = sizes.into_iter().map(|e| e.clone_expr()).collect::<Vec<_>>();
    debug_assert!(self.find_buf(name).is_none() && !sizes.is_empty());
    let buf = box Buf { func: self.into(), name: name.into(), ty, kind, loc: Host, sizes };
    debug!("buf: create buf {}, sizes = [{}]", name, comma_sep(buf.sizes.iter()));
    let ret = buf.as_ref().p();
    self.p().bufs.push(buf);
    ret.get()
  }
}

impl Buf {
  pub fn at<E: IntoExpr>(&self, idx: impl IntoIterator<Item=E>) -> Expr {
    let idx = idx.into_iter().map(|e| e.clone_expr()).collect::<Box<[_]>>();
    debug_assert_eq!(idx.len(), self.sizes.len());
    Load(self.into(), idx)
  }

  pub fn set_loc(&self, loc: BufLoc) -> &Buf {
    self.p().loc = loc;
    self
  }

  pub fn dup(&self) -> &Buf {
    let name = format!("_{}_dup{}", self.name, self.func.new_buf_id());
    self.func.buf(&name, self.ty, self.kind, &self.sizes)
  }

  // 在comp的循环层次at的开头/结尾放置Alloc/Free；若at == -1，则是在函数的开头/结尾放置Alloc/Free
  // 开头/结尾是当前的，不保证之后添加新的计算后这对Alloc/Free仍然在开头/结尾
  pub fn alloc_at(&self, comp: &Comp, at: i32) -> &Buf {
    debug_assert!(at >= -1);
    let mut f = comp.func;
    let mut dom = project_static_dim(comp.schedule());
    let (n, at) = (dom.n_dim() as u32, (at + 1) as u32);
    dom = dom.project_out(DimType::Set, at, n - at)?;
    debug!("alloc_at: dom = {}", dom);
    let alloc = f.comp_raw(dom.copy()?.set_tuple_name(
      cstr(&format!("_alloc{}_{}\0", f.new_comp_id(), self.name)))?, Alloc(self.into()));
    let free = f.comp_raw(dom.set_tuple_name(
      cstr(&format!("_free{}_{}\0", f.new_comp_id(), self.name)))?, Free(self.into()));
    if at > 0 {
      comp.root_comp(at).after_between_pred(alloc, at);
      free.after_between_pred(comp.leaf_comp(at), at);
    } else {
      let alloc_idx = f.comps.len() - 2;
      let alloc = f.comps.remove(alloc_idx);
      f.comps.insert(0, alloc);
    }
    self
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

// 即runtime buffer，用于测试，T必须是基本类型
pub struct Array<T, D: Dims> {
  pub ptr: NonNull<T>,
  pub dim: D,
}

// 和Array的唯一区别在于没有实现Drop
#[derive(Copy, Clone)]
pub struct Slice<T, D: Dims> {
  pub ptr: NonNull<T>,
  pub dim: D,
}

impl<T, D: Dims> Array<T, D> {
  // 不初始化元素
  pub fn new(dim: D) -> Array<T, D> {
    unsafe {
      let ptr = alloc(Layout::from_size_align_unchecked(dim.total() * size_of::<T>(), 32));
      Array { ptr: NonNull::new(ptr as _).expect("failed to alloc"), dim }
    }
  }

  // 所有元素初始化为逐字节全0
  pub fn zeroed(dim: D) -> Array<T, D> {
    unsafe {
      let ptr = alloc_zeroed(Layout::from_size_align_unchecked(dim.total() * size_of::<T>(), 32));
      debug_assert!(!ptr.is_null());
      Array { ptr: NonNull::new(ptr as _).expect("failed to alloc"), dim }
    }
  }
}

impl<T, D: Dims> Drop for Array<T, D> {
  fn drop(&mut self) {
    unsafe { dealloc(self.ptr() as _, Layout::from_size_align_unchecked(self.dim.total() * size_of::<T>(), 32)); }
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
    unsafe { slice::from_raw_parts_mut(self.ptr(), self.dim.total()) }
  }

  // 第一层维度的长度
  pub fn len(&self) -> usize { self.dim.sub().0 }

  pub fn sub(&self, idx: usize) -> Slice<T, D::Sub> {
    let (n, sub) = self.dim.sub();
    debug_assert!(idx < n);
    unsafe { Slice { ptr: NonNull::new_unchecked(self.ptr().add(idx * sub.total())), dim: sub } }
  }
}

// 仅在debug模式下有下标范围检查
impl<T, D: Dims> Index<D> for Slice<T, D> {
  type Output = T;
  fn index(&self, idx: D) -> &Self::Output {
    unsafe { &*self.ptr().add(self.dim.offset(idx)) }
  }
}

impl<T, D: Dims> IndexMut<D> for Slice<T, D> {
  fn index_mut(&mut self, idx: D) -> &mut Self::Output {
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

impl Dims for usize {
  // 设置Sub和自身相同，Debug实现中利用了这个性质
  type Sub = Self;
  fn total(self) -> usize { self }
  fn offset(self, idx: Self) -> usize { (debug_assert!(idx < self), idx).1 }
}

impl Dims for (usize, ) {
  type Sub = Self;
  fn total(self) -> usize { self.0 }
  fn offset(self, idx: Self) -> usize { (debug_assert!(idx.0 < self.0), idx.0).1 }
}

macro_rules! impl_dims {
  ($($t: ident)*, $($i: ident)*, $($j: ident)*) => {
    impl Dims for (usize, $($t),*) {
      type Sub = ($($t,)*);
      fn total(self) -> usize {
        let (i0, $($i),*) = self;
        i0 $(*$i)*
      }
      fn offset(self, idx: Self) -> usize {
        let (i0, $($i),*) = self;
        let (mut j0, $($j),*) = idx;
        debug_assert!(j0 < i0 $(&&$j < $i)*);
        $(j0 = j0 * $i + $j;)*
        j0
      }
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
