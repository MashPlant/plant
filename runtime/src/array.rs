#[cfg(feature = "gpu-runtime")]
use cuda_runtime_sys::*;

use num::Signed;
use std::{ptr::*, alloc::*, mem::*, ops::*, slice};
use crate::*;

pub trait Primitive { const TYPE: Type; }

macro_rules! impl_primitive {
  ($($val: ident $ty: ident),*) => {
    $(impl Primitive for $ty { const TYPE: Type = $val; })*
  };
}

impl_primitive!(I8 i8, U8 u8, I16 i16, U16 u16, I32 i32, U32 u32, I64 i64, U64 u64, F32 f32, F64 f64);

// 即runtime buffer
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
    let p = unsafe { alloc(Layout::from_size_align_unchecked(dim.total() * size_of::<T>(), 128)) };
    Array { ptr: NonNull::new(p as _).expect("failed to alloc"), dim, loc: CPU }
  }

  // 所有元素初始化为逐字节全0
  pub fn zeroed(dim: D) -> Self {
    let p = unsafe { alloc_zeroed(Layout::from_size_align_unchecked(dim.total() * size_of::<T>(), 128)) };
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
      CPU => unsafe { dealloc(self.ptr() as _, Layout::from_size_align_unchecked(self.dim.total() * size_of::<T>(), 128)) },
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
