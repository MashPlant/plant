use std::{ptr::NonNull, alloc::*, mem::*, ops::*, slice, fmt::*};
use crate::*;

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum BufKind { In, Out, Temp }

#[derive(Debug)]
pub struct Buf {
  pub name: Box<str>,
  pub ty: Type,
  pub kind: BufKind,
  pub sizes: Vec<Expr>,
}

impl Func {
  pub fn buf(&self, name: &str, ty: Type, kind: BufKind, sizes: &[impl IntoExpr]) -> &Buf {
    assert!(self.find_buf(name).is_none() && !sizes.is_empty());
    let buf = box Buf { name: name.into(), ty, kind, sizes: sizes.iter().map(|e| e.clone_expr()).collect() };
    let ret = R::new(&*buf);
    P::new(self).bufs.push(buf);
    ret.get()
  }
}

impl Buf {
  pub fn at(&self, idx: &[impl IntoExpr]) -> Expr {
    assert_eq!(idx.len(), self.sizes.len());
    Load(self.into(), idx.iter().map(|e| e.clone_expr()).collect())
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
      Array { ptr: NonNull::new(ptr as _).unwrap(), dim }
    }
  }

  // 所有元素初始化为逐字节全0
  pub fn zeroed(dim: D) -> Array<T, D> {
    unsafe {
      let ptr = alloc_zeroed(Layout::from_size_align_unchecked(dim.total() * size_of::<T>(), 32));
      Array { ptr: NonNull::new(ptr as _).unwrap(), dim }
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
  fn fmt(&self, f: &mut Formatter) -> Result { write!(f, "{:?}", &**self) }
}

impl<T: Debug, D: Dims> Debug for Array<T, D> {
  fn fmt(&self, f: &mut Formatter) -> Result { write!(f, "{:?}", &**self) }
}

impl<T: Debug, D: Dims> Display for Slice<T, D> {
  fn fmt(&self, f: &mut Formatter) -> Result { write!(f, "{:?}", self) }
}

impl<T: Debug, D: Dims> Debug for Slice<T, D> {
  fn fmt(&self, f: &mut Formatter) -> Result {
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
