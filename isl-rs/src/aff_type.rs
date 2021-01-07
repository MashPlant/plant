use crate::*;

extern "C" {
  pub fn isl_aff_list_get_ctx(list: AffListRef) -> Option<CtxRef>;
  pub fn isl_aff_list_from_aff(el: Aff) -> Option<AffList>;
  pub fn isl_aff_list_alloc(ctx: CtxRef, n: c_int) -> Option<AffList>;
  pub fn isl_aff_list_copy(list: AffListRef) -> Option<AffList>;
  pub fn isl_aff_list_free(list: AffList) -> *mut c_void;
  pub fn isl_aff_list_add(list: AffList, el: Aff) -> Option<AffList>;
  pub fn isl_aff_list_insert(list: AffList, pos: c_uint, el: Aff) -> Option<AffList>;
  pub fn isl_aff_list_drop(list: AffList, first: c_uint, n: c_uint) -> Option<AffList>;
  pub fn isl_aff_list_concat(list1: AffList, list2: AffList) -> Option<AffList>;
  pub fn isl_aff_list_n_aff(list: AffListRef) -> c_int;
  pub fn isl_aff_list_get_aff(list: AffListRef, index: c_int) -> Option<Aff>;
  pub fn isl_aff_list_set_aff(list: AffList, index: c_int, el: Aff) -> Option<AffList>;
  pub fn isl_aff_list_foreach(list: AffListRef, fn_: unsafe extern "C" fn(el: Aff, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_aff_list_map(list: AffList, fn_: unsafe extern "C" fn(el: Aff, user: *mut c_void) -> Option<Aff>, user: *mut c_void) -> Option<AffList>;
  pub fn isl_aff_list_sort(list: AffList, cmp: unsafe extern "C" fn(a: AffRef, b: AffRef, user: *mut c_void) -> c_int, user: *mut c_void) -> Option<AffList>;
  pub fn isl_aff_list_foreach_scc(list: AffListRef, follows: unsafe extern "C" fn(a: AffRef, b: AffRef, user: *mut c_void) -> Bool, follows_user: *mut c_void, fn_: unsafe extern "C" fn(scc: AffList, user: *mut c_void) -> Stat, fn_user: *mut c_void) -> Stat;
  pub fn isl_printer_print_aff_list(p: Printer, list: AffListRef) -> Option<Printer>;
  pub fn isl_aff_list_dump(list: AffListRef) -> ();
  pub fn isl_pw_aff_list_get_ctx(list: PwAffListRef) -> Option<CtxRef>;
  pub fn isl_pw_aff_list_from_pw_aff(el: PwAff) -> Option<PwAffList>;
  pub fn isl_pw_aff_list_alloc(ctx: CtxRef, n: c_int) -> Option<PwAffList>;
  pub fn isl_pw_aff_list_copy(list: PwAffListRef) -> Option<PwAffList>;
  pub fn isl_pw_aff_list_free(list: PwAffList) -> *mut c_void;
  pub fn isl_pw_aff_list_add(list: PwAffList, el: PwAff) -> Option<PwAffList>;
  pub fn isl_pw_aff_list_insert(list: PwAffList, pos: c_uint, el: PwAff) -> Option<PwAffList>;
  pub fn isl_pw_aff_list_drop(list: PwAffList, first: c_uint, n: c_uint) -> Option<PwAffList>;
  pub fn isl_pw_aff_list_concat(list1: PwAffList, list2: PwAffList) -> Option<PwAffList>;
  pub fn isl_pw_aff_list_n_pw_aff(list: PwAffListRef) -> c_int;
  pub fn isl_pw_aff_list_get_pw_aff(list: PwAffListRef, index: c_int) -> Option<PwAff>;
  pub fn isl_pw_aff_list_set_pw_aff(list: PwAffList, index: c_int, el: PwAff) -> Option<PwAffList>;
  pub fn isl_pw_aff_list_foreach(list: PwAffListRef, fn_: unsafe extern "C" fn(el: PwAff, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_pw_aff_list_map(list: PwAffList, fn_: unsafe extern "C" fn(el: PwAff, user: *mut c_void) -> Option<PwAff>, user: *mut c_void) -> Option<PwAffList>;
  pub fn isl_pw_aff_list_sort(list: PwAffList, cmp: unsafe extern "C" fn(a: PwAffRef, b: PwAffRef, user: *mut c_void) -> c_int, user: *mut c_void) -> Option<PwAffList>;
  pub fn isl_pw_aff_list_foreach_scc(list: PwAffListRef, follows: unsafe extern "C" fn(a: PwAffRef, b: PwAffRef, user: *mut c_void) -> Bool, follows_user: *mut c_void, fn_: unsafe extern "C" fn(scc: PwAffList, user: *mut c_void) -> Stat, fn_user: *mut c_void) -> Stat;
  pub fn isl_printer_print_pw_aff_list(p: Printer, list: PwAffListRef) -> Option<Printer>;
  pub fn isl_pw_aff_list_dump(list: PwAffListRef) -> ();
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Aff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct AffRef(pub NonNull<c_void>);

impl_try!(Aff);
impl_try!(AffRef);

impl Aff {
  #[inline(always)]
  pub fn read(&self) -> Aff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Aff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<AffRef> for Aff {
  #[inline(always)]
  fn as_ref(&self) -> &AffRef { unsafe { mem::transmute(self) } }
}

impl Deref for Aff {
  type Target = AffRef;
  #[inline(always)]
  fn deref(&self) -> &AffRef { self.as_ref() }
}

impl To<Option<Aff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Aff> { NonNull::new(self).map(Aff) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct AffList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct AffListRef(pub NonNull<c_void>);

impl_try!(AffList);
impl_try!(AffListRef);

impl AffList {
  #[inline(always)]
  pub fn read(&self) -> AffList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: AffList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<AffListRef> for AffList {
  #[inline(always)]
  fn as_ref(&self) -> &AffListRef { unsafe { mem::transmute(self) } }
}

impl Deref for AffList {
  type Target = AffListRef;
  #[inline(always)]
  fn deref(&self) -> &AffListRef { self.as_ref() }
}

impl To<Option<AffList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<AffList> { NonNull::new(self).map(AffList) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct PwAff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct PwAffRef(pub NonNull<c_void>);

impl_try!(PwAff);
impl_try!(PwAffRef);

impl PwAff {
  #[inline(always)]
  pub fn read(&self) -> PwAff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: PwAff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<PwAffRef> for PwAff {
  #[inline(always)]
  fn as_ref(&self) -> &PwAffRef { unsafe { mem::transmute(self) } }
}

impl Deref for PwAff {
  type Target = PwAffRef;
  #[inline(always)]
  fn deref(&self) -> &PwAffRef { self.as_ref() }
}

impl To<Option<PwAff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<PwAff> { NonNull::new(self).map(PwAff) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct PwAffList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct PwAffListRef(pub NonNull<c_void>);

impl_try!(PwAffList);
impl_try!(PwAffListRef);

impl PwAffList {
  #[inline(always)]
  pub fn read(&self) -> PwAffList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: PwAffList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<PwAffListRef> for PwAffList {
  #[inline(always)]
  fn as_ref(&self) -> &PwAffListRef { unsafe { mem::transmute(self) } }
}

impl Deref for PwAffList {
  type Target = PwAffListRef;
  #[inline(always)]
  fn deref(&self) -> &PwAffListRef { self.as_ref() }
}

impl To<Option<PwAffList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<PwAffList> { NonNull::new(self).map(PwAffList) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct UnionPwAff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct UnionPwAffRef(pub NonNull<c_void>);

impl_try!(UnionPwAff);
impl_try!(UnionPwAffRef);

impl UnionPwAff {
  #[inline(always)]
  pub fn read(&self) -> UnionPwAff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: UnionPwAff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<UnionPwAffRef> for UnionPwAff {
  #[inline(always)]
  fn as_ref(&self) -> &UnionPwAffRef { unsafe { mem::transmute(self) } }
}

impl Deref for UnionPwAff {
  type Target = UnionPwAffRef;
  #[inline(always)]
  fn deref(&self) -> &UnionPwAffRef { self.as_ref() }
}

impl To<Option<UnionPwAff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<UnionPwAff> { NonNull::new(self).map(UnionPwAff) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct UnionPwAffList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct UnionPwAffListRef(pub NonNull<c_void>);

impl_try!(UnionPwAffList);
impl_try!(UnionPwAffListRef);

impl UnionPwAffList {
  #[inline(always)]
  pub fn read(&self) -> UnionPwAffList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: UnionPwAffList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<UnionPwAffListRef> for UnionPwAffList {
  #[inline(always)]
  fn as_ref(&self) -> &UnionPwAffListRef { unsafe { mem::transmute(self) } }
}

impl Deref for UnionPwAffList {
  type Target = UnionPwAffListRef;
  #[inline(always)]
  fn deref(&self) -> &UnionPwAffListRef { self.as_ref() }
}

impl To<Option<UnionPwAffList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<UnionPwAffList> { NonNull::new(self).map(UnionPwAffList) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MultiAff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct MultiAffRef(pub NonNull<c_void>);

impl_try!(MultiAff);
impl_try!(MultiAffRef);

impl MultiAff {
  #[inline(always)]
  pub fn read(&self) -> MultiAff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: MultiAff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MultiAffRef> for MultiAff {
  #[inline(always)]
  fn as_ref(&self) -> &MultiAffRef { unsafe { mem::transmute(self) } }
}

impl Deref for MultiAff {
  type Target = MultiAffRef;
  #[inline(always)]
  fn deref(&self) -> &MultiAffRef { self.as_ref() }
}

impl To<Option<MultiAff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MultiAff> { NonNull::new(self).map(MultiAff) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct PwMultiAff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct PwMultiAffRef(pub NonNull<c_void>);

impl_try!(PwMultiAff);
impl_try!(PwMultiAffRef);

impl PwMultiAff {
  #[inline(always)]
  pub fn read(&self) -> PwMultiAff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: PwMultiAff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<PwMultiAffRef> for PwMultiAff {
  #[inline(always)]
  fn as_ref(&self) -> &PwMultiAffRef { unsafe { mem::transmute(self) } }
}

impl Deref for PwMultiAff {
  type Target = PwMultiAffRef;
  #[inline(always)]
  fn deref(&self) -> &PwMultiAffRef { self.as_ref() }
}

impl To<Option<PwMultiAff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<PwMultiAff> { NonNull::new(self).map(PwMultiAff) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct UnionPwMultiAff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct UnionPwMultiAffRef(pub NonNull<c_void>);

impl_try!(UnionPwMultiAff);
impl_try!(UnionPwMultiAffRef);

impl UnionPwMultiAff {
  #[inline(always)]
  pub fn read(&self) -> UnionPwMultiAff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: UnionPwMultiAff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<UnionPwMultiAffRef> for UnionPwMultiAff {
  #[inline(always)]
  fn as_ref(&self) -> &UnionPwMultiAffRef { unsafe { mem::transmute(self) } }
}

impl Deref for UnionPwMultiAff {
  type Target = UnionPwMultiAffRef;
  #[inline(always)]
  fn deref(&self) -> &UnionPwMultiAffRef { self.as_ref() }
}

impl To<Option<UnionPwMultiAff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<UnionPwMultiAff> { NonNull::new(self).map(UnionPwMultiAff) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct UnionPwMultiAffList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct UnionPwMultiAffListRef(pub NonNull<c_void>);

impl_try!(UnionPwMultiAffList);
impl_try!(UnionPwMultiAffListRef);

impl UnionPwMultiAffList {
  #[inline(always)]
  pub fn read(&self) -> UnionPwMultiAffList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: UnionPwMultiAffList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<UnionPwMultiAffListRef> for UnionPwMultiAffList {
  #[inline(always)]
  fn as_ref(&self) -> &UnionPwMultiAffListRef { unsafe { mem::transmute(self) } }
}

impl Deref for UnionPwMultiAffList {
  type Target = UnionPwMultiAffListRef;
  #[inline(always)]
  fn deref(&self) -> &UnionPwMultiAffListRef { self.as_ref() }
}

impl To<Option<UnionPwMultiAffList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<UnionPwMultiAffList> { NonNull::new(self).map(UnionPwMultiAffList) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MultiPwAff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct MultiPwAffRef(pub NonNull<c_void>);

impl_try!(MultiPwAff);
impl_try!(MultiPwAffRef);

impl MultiPwAff {
  #[inline(always)]
  pub fn read(&self) -> MultiPwAff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: MultiPwAff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MultiPwAffRef> for MultiPwAff {
  #[inline(always)]
  fn as_ref(&self) -> &MultiPwAffRef { unsafe { mem::transmute(self) } }
}

impl Deref for MultiPwAff {
  type Target = MultiPwAffRef;
  #[inline(always)]
  fn deref(&self) -> &MultiPwAffRef { self.as_ref() }
}

impl To<Option<MultiPwAff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MultiPwAff> { NonNull::new(self).map(MultiPwAff) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MultiUnionPwAff(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct MultiUnionPwAffRef(pub NonNull<c_void>);

impl_try!(MultiUnionPwAff);
impl_try!(MultiUnionPwAffRef);

impl MultiUnionPwAff {
  #[inline(always)]
  pub fn read(&self) -> MultiUnionPwAff { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: MultiUnionPwAff) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MultiUnionPwAffRef> for MultiUnionPwAff {
  #[inline(always)]
  fn as_ref(&self) -> &MultiUnionPwAffRef { unsafe { mem::transmute(self) } }
}

impl Deref for MultiUnionPwAff {
  type Target = MultiUnionPwAffRef;
  #[inline(always)]
  fn deref(&self) -> &MultiUnionPwAffRef { self.as_ref() }
}

impl To<Option<MultiUnionPwAff>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MultiUnionPwAff> { NonNull::new(self).map(MultiUnionPwAff) }
}

impl Aff {
  #[inline(always)]
  pub fn list_from_aff(self) -> Option<AffList> {
    unsafe {
      let ret = isl_aff_list_from_aff(self.to());
      (ret).to()
    }
  }
}

impl AffList {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_aff_list_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, el: Aff) -> Option<AffList> {
    unsafe {
      let ret = isl_aff_list_add(self.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert(self, pos: c_uint, el: Aff) -> Option<AffList> {
    unsafe {
      let ret = isl_aff_list_insert(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, first: c_uint, n: c_uint) -> Option<AffList> {
    unsafe {
      let ret = isl_aff_list_drop(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, list2: AffList) -> Option<AffList> {
    unsafe {
      let ret = isl_aff_list_concat(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_aff(self, index: c_int, el: Aff) -> Option<AffList> {
    unsafe {
      let ret = isl_aff_list_set_aff(self.to(), index.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map<F1: FnMut(Aff) -> Option<Aff>>(self, fn_: &mut F1) -> Option<AffList> {
    unsafe extern "C" fn fn1<F: FnMut(Aff) -> Option<Aff>>(el: Aff, user: *mut c_void) -> Option<Aff> { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_aff_list_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort<F1: FnMut(AffRef, AffRef) -> c_int>(self, cmp: &mut F1) -> Option<AffList> {
    unsafe extern "C" fn fn1<F: FnMut(AffRef, AffRef) -> c_int>(a: AffRef, b: AffRef, user: *mut c_void) -> c_int { (*(user as *mut F))(a.to(), b.to()) }
    unsafe {
      let ret = isl_aff_list_sort(self.to(), fn1::<F1>, cmp as *mut _ as _);
      (ret).to()
    }
  }
}

impl AffListRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_aff_list_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<AffList> {
    unsafe {
      let ret = isl_aff_list_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_aff(self) -> c_int {
    unsafe {
      let ret = isl_aff_list_n_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_aff(self, index: c_int) -> Option<Aff> {
    unsafe {
      let ret = isl_aff_list_get_aff(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(Aff) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(Aff) -> Stat>(el: Aff, user: *mut c_void) -> Stat { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_aff_list_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_scc<F1: FnMut(AffRef, AffRef) -> Bool, F2: FnMut(AffList) -> Stat>(self, follows: &mut F1, fn_: &mut F2) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(AffRef, AffRef) -> Bool>(a: AffRef, b: AffRef, user: *mut c_void) -> Bool { (*(user as *mut F))(a.to(), b.to()) }
    unsafe extern "C" fn fn2<F: FnMut(AffList) -> Stat>(scc: AffList, user: *mut c_void) -> Stat { (*(user as *mut F))(scc.to()) }
    unsafe {
      let ret = isl_aff_list_foreach_scc(self.to(), fn1::<F1>, follows as *mut _ as _, fn2::<F2>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_aff_list_dump(self.to());
      (ret).to()
    }
  }
}

impl CtxRef {
  #[inline(always)]
  pub fn aff_list_alloc(self, n: c_int) -> Option<AffList> {
    unsafe {
      let ret = isl_aff_list_alloc(self.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pw_aff_list_alloc(self, n: c_int) -> Option<PwAffList> {
    unsafe {
      let ret = isl_pw_aff_list_alloc(self.to(), n.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_aff_list(self, list: AffListRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_aff_list(self.to(), list.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_pw_aff_list(self, list: PwAffListRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_pw_aff_list(self.to(), list.to());
      (ret).to()
    }
  }
}

impl PwAff {
  #[inline(always)]
  pub fn list_from_pw_aff(self) -> Option<PwAffList> {
    unsafe {
      let ret = isl_pw_aff_list_from_pw_aff(self.to());
      (ret).to()
    }
  }
}

impl PwAffList {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_pw_aff_list_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, el: PwAff) -> Option<PwAffList> {
    unsafe {
      let ret = isl_pw_aff_list_add(self.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert(self, pos: c_uint, el: PwAff) -> Option<PwAffList> {
    unsafe {
      let ret = isl_pw_aff_list_insert(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, first: c_uint, n: c_uint) -> Option<PwAffList> {
    unsafe {
      let ret = isl_pw_aff_list_drop(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, list2: PwAffList) -> Option<PwAffList> {
    unsafe {
      let ret = isl_pw_aff_list_concat(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_pw_aff(self, index: c_int, el: PwAff) -> Option<PwAffList> {
    unsafe {
      let ret = isl_pw_aff_list_set_pw_aff(self.to(), index.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map<F1: FnMut(PwAff) -> Option<PwAff>>(self, fn_: &mut F1) -> Option<PwAffList> {
    unsafe extern "C" fn fn1<F: FnMut(PwAff) -> Option<PwAff>>(el: PwAff, user: *mut c_void) -> Option<PwAff> { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_pw_aff_list_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort<F1: FnMut(PwAffRef, PwAffRef) -> c_int>(self, cmp: &mut F1) -> Option<PwAffList> {
    unsafe extern "C" fn fn1<F: FnMut(PwAffRef, PwAffRef) -> c_int>(a: PwAffRef, b: PwAffRef, user: *mut c_void) -> c_int { (*(user as *mut F))(a.to(), b.to()) }
    unsafe {
      let ret = isl_pw_aff_list_sort(self.to(), fn1::<F1>, cmp as *mut _ as _);
      (ret).to()
    }
  }
}

impl PwAffListRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_pw_aff_list_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<PwAffList> {
    unsafe {
      let ret = isl_pw_aff_list_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_pw_aff(self) -> c_int {
    unsafe {
      let ret = isl_pw_aff_list_n_pw_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_pw_aff(self, index: c_int) -> Option<PwAff> {
    unsafe {
      let ret = isl_pw_aff_list_get_pw_aff(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(PwAff) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(PwAff) -> Stat>(el: PwAff, user: *mut c_void) -> Stat { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_pw_aff_list_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_scc<F1: FnMut(PwAffRef, PwAffRef) -> Bool, F2: FnMut(PwAffList) -> Stat>(self, follows: &mut F1, fn_: &mut F2) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(PwAffRef, PwAffRef) -> Bool>(a: PwAffRef, b: PwAffRef, user: *mut c_void) -> Bool { (*(user as *mut F))(a.to(), b.to()) }
    unsafe extern "C" fn fn2<F: FnMut(PwAffList) -> Stat>(scc: PwAffList, user: *mut c_void) -> Stat { (*(user as *mut F))(scc.to()) }
    unsafe {
      let ret = isl_pw_aff_list_foreach_scc(self.to(), fn1::<F1>, follows as *mut _ as _, fn2::<F2>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_pw_aff_list_dump(self.to());
      (ret).to()
    }
  }
}

impl Drop for AffList {
  fn drop(&mut self) { AffList(self.0).free() }
}

impl Drop for PwAffList {
  fn drop(&mut self) { PwAffList(self.0).free() }
}

