use crate::*;

extern "C" {
  pub fn isl_id_list_get_ctx(list: IdListRef) -> Option<CtxRef>;
  pub fn isl_id_list_from_id(el: Id) -> Option<IdList>;
  pub fn isl_id_list_alloc(ctx: CtxRef, n: c_int) -> Option<IdList>;
  pub fn isl_id_list_copy(list: IdListRef) -> Option<IdList>;
  pub fn isl_id_list_free(list: IdList) -> *mut c_void;
  pub fn isl_id_list_add(list: IdList, el: Id) -> Option<IdList>;
  pub fn isl_id_list_insert(list: IdList, pos: c_uint, el: Id) -> Option<IdList>;
  pub fn isl_id_list_drop(list: IdList, first: c_uint, n: c_uint) -> Option<IdList>;
  pub fn isl_id_list_concat(list1: IdList, list2: IdList) -> Option<IdList>;
  pub fn isl_id_list_n_id(list: IdListRef) -> c_int;
  pub fn isl_id_list_get_id(list: IdListRef, index: c_int) -> Option<Id>;
  pub fn isl_id_list_set_id(list: IdList, index: c_int, el: Id) -> Option<IdList>;
  pub fn isl_id_list_foreach(list: IdListRef, fn_: unsafe extern "C" fn(el: Id, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_id_list_map(list: IdList, fn_: unsafe extern "C" fn(el: Id, user: *mut c_void) -> Option<Id>, user: *mut c_void) -> Option<IdList>;
  pub fn isl_id_list_sort(list: IdList, cmp: unsafe extern "C" fn(a: IdRef, b: IdRef, user: *mut c_void) -> c_int, user: *mut c_void) -> Option<IdList>;
  pub fn isl_id_list_foreach_scc(list: IdListRef, follows: unsafe extern "C" fn(a: IdRef, b: IdRef, user: *mut c_void) -> Bool, follows_user: *mut c_void, fn_: unsafe extern "C" fn(scc: IdList, user: *mut c_void) -> Stat, fn_user: *mut c_void) -> Stat;
  pub fn isl_printer_print_id_list(p: Printer, list: IdListRef) -> Option<Printer>;
  pub fn isl_id_list_dump(list: IdListRef) -> ();
  pub fn isl_id_get_ctx(id: IdRef) -> Option<CtxRef>;
  pub fn isl_id_get_hash(id: IdRef) -> c_uint;
  pub fn isl_id_alloc(ctx: CtxRef, name: Option<CStr>, user: *mut c_void) -> Option<Id>;
  pub fn isl_id_copy(id: IdRef) -> Option<Id>;
  pub fn isl_id_free(id: Id) -> *mut c_void;
  pub fn isl_id_get_user(id: IdRef) -> *mut c_void;
  pub fn isl_id_get_name(id: IdRef) -> Option<CStr>;
  pub fn isl_id_set_free_user(id: Id, free_user: unsafe extern "C" fn(user: *mut c_void) -> ()) -> Option<Id>;
  pub fn isl_id_to_str(id: IdRef) -> Option<CString>;
  pub fn isl_printer_print_id(p: Printer, id: IdRef) -> Option<Printer>;
  pub fn isl_id_dump(id: IdRef) -> ();
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Id(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct IdRef(pub NonNull<c_void>);

impl Id {
  #[inline(always)]
  pub fn read(&self) -> Id { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Id) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<IdRef> for Id {
  #[inline(always)]
  fn as_ref(&self) -> &IdRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for Id {
  type Target = IdRef;
  #[inline(always)]
  fn deref(&self) -> &IdRef { self.as_ref() }
}

impl To<Option<Id>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Id> { NonNull::new(self).map(Id) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct IdList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct IdListRef(pub NonNull<c_void>);

impl IdList {
  #[inline(always)]
  pub fn read(&self) -> IdList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: IdList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<IdListRef> for IdList {
  #[inline(always)]
  fn as_ref(&self) -> &IdListRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for IdList {
  type Target = IdListRef;
  #[inline(always)]
  fn deref(&self) -> &IdListRef { self.as_ref() }
}

impl To<Option<IdList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<IdList> { NonNull::new(self).map(IdList) }
}

impl CtxRef {
  #[inline(always)]
  pub fn id_list_alloc(self, n: c_int) -> Option<IdList> {
    unsafe {
      let ret = isl_id_list_alloc(self.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn id_alloc(self, name: Option<CStr>, user: *mut c_void) -> Option<Id> {
    unsafe {
      let ret = isl_id_alloc(self.to(), name.to(), user.to());
      (ret).to()
    }
  }
}

impl Id {
  #[inline(always)]
  pub fn list_from_id(self) -> Option<IdList> {
    unsafe {
      let ret = isl_id_list_from_id(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_id_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_free_user(self, free_user: unsafe extern "C" fn(user: *mut c_void) -> ()) -> Option<Id> {
    unsafe {
      let ret = isl_id_set_free_user(self.to(), free_user.to());
      (ret).to()
    }
  }
}

impl IdList {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_id_list_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, el: Id) -> Option<IdList> {
    unsafe {
      let ret = isl_id_list_add(self.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert(self, pos: c_uint, el: Id) -> Option<IdList> {
    unsafe {
      let ret = isl_id_list_insert(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, first: c_uint, n: c_uint) -> Option<IdList> {
    unsafe {
      let ret = isl_id_list_drop(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, list2: IdList) -> Option<IdList> {
    unsafe {
      let ret = isl_id_list_concat(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_id(self, index: c_int, el: Id) -> Option<IdList> {
    unsafe {
      let ret = isl_id_list_set_id(self.to(), index.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map<F1: FnMut(Id) -> Option<Id>>(self, fn_: &mut F1) -> Option<IdList> {
    unsafe extern "C" fn fn1<F: FnMut(Id) -> Option<Id>>(el: Id, user: *mut c_void) -> Option<Id> { (*(user as *mut F))(el.to()).to() }
    unsafe {
      let ret = isl_id_list_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort<F1: FnMut(IdRef, IdRef) -> c_int>(self, cmp: &mut F1) -> Option<IdList> {
    unsafe extern "C" fn fn1<F: FnMut(IdRef, IdRef) -> c_int>(a: IdRef, b: IdRef, user: *mut c_void) -> c_int { (*(user as *mut F))(a.to(), b.to()).to() }
    unsafe {
      let ret = isl_id_list_sort(self.to(), fn1::<F1>, cmp as *mut _ as _);
      (ret).to()
    }
  }
}

impl IdListRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_id_list_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<IdList> {
    unsafe {
      let ret = isl_id_list_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_id(self) -> c_int {
    unsafe {
      let ret = isl_id_list_n_id(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_id(self, index: c_int) -> Option<Id> {
    unsafe {
      let ret = isl_id_list_get_id(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(Id) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(Id) -> Option<()>>(el: Id, user: *mut c_void) -> Stat { (*(user as *mut F))(el.to()).to() }
    unsafe {
      let ret = isl_id_list_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_scc<F1: FnMut(IdRef, IdRef) -> Option<bool>, F2: FnMut(IdList) -> Option<()>>(self, follows: &mut F1, fn_: &mut F2) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(IdRef, IdRef) -> Option<bool>>(a: IdRef, b: IdRef, user: *mut c_void) -> Bool { (*(user as *mut F))(a.to(), b.to()).to() }
    unsafe extern "C" fn fn2<F: FnMut(IdList) -> Option<()>>(scc: IdList, user: *mut c_void) -> Stat { (*(user as *mut F))(scc.to()).to() }
    unsafe {
      let ret = isl_id_list_foreach_scc(self.to(), fn1::<F1>, follows as *mut _ as _, fn2::<F2>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_id_list_dump(self.to());
      (ret).to()
    }
  }
}

impl IdRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_id_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_hash(self) -> c_uint {
    unsafe {
      let ret = isl_id_get_hash(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<Id> {
    unsafe {
      let ret = isl_id_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_user(self) -> *mut c_void {
    unsafe {
      let ret = isl_id_get_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_name(self) -> Option<CStr> {
    unsafe {
      let ret = isl_id_get_name(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_id_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_id_dump(self.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_id_list(self, list: IdListRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_id_list(self.to(), list.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_id(self, id: IdRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_id(self.to(), id.to());
      (ret).to()
    }
  }
}

impl Drop for Id {
  fn drop(&mut self) { Id(self.0).free() }
}

impl Drop for IdList {
  fn drop(&mut self) { IdList(self.0).free() }
}

impl fmt::Display for IdRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for Id {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

