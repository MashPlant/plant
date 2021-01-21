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
  pub fn isl_id_list_clear(list: IdList) -> Option<IdList>;
  pub fn isl_id_list_swap(list: IdList, pos1: c_uint, pos2: c_uint) -> Option<IdList>;
  pub fn isl_id_list_reverse(list: IdList) -> Option<IdList>;
  pub fn isl_id_list_concat(list1: IdList, list2: IdList) -> Option<IdList>;
  pub fn isl_id_list_size(list: IdListRef) -> c_int;
  pub fn isl_id_list_n_id(list: IdListRef) -> c_int;
  pub fn isl_id_list_get_at(list: IdListRef, index: c_int) -> Option<Id>;
  pub fn isl_id_list_get_id(list: IdListRef, index: c_int) -> Option<Id>;
  pub fn isl_id_list_set_id(list: IdList, index: c_int, el: Id) -> Option<IdList>;
  pub fn isl_id_list_foreach(list: IdListRef, fn_: unsafe extern "C" fn(el: Id, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_id_list_every(list: IdListRef, test: unsafe extern "C" fn(el: IdRef, user: *mut c_void) -> Bool, user: *mut c_void) -> Bool;
  pub fn isl_id_list_map(list: IdList, fn_: unsafe extern "C" fn(el: Id, user: *mut c_void) -> Option<Id>, user: *mut c_void) -> Option<IdList>;
  pub fn isl_id_list_sort(list: IdList, cmp: unsafe extern "C" fn(a: IdRef, b: IdRef, user: *mut c_void) -> c_int, user: *mut c_void) -> Option<IdList>;
  pub fn isl_id_list_foreach_scc(list: IdListRef, follows: unsafe extern "C" fn(a: IdRef, b: IdRef, user: *mut c_void) -> Bool, follows_user: *mut c_void, fn_: unsafe extern "C" fn(scc: IdList, user: *mut c_void) -> Stat, fn_user: *mut c_void) -> Stat;
  pub fn isl_id_list_to_str(list: IdListRef) -> Option<CString>;
  pub fn isl_printer_print_id_list(p: Printer, list: IdListRef) -> Option<Printer>;
  pub fn isl_id_list_dump(list: IdListRef) -> ();
  pub fn isl_multi_id_get_ctx(multi: MultiIdRef) -> Option<CtxRef>;
  pub fn isl_multi_id_get_space(multi: MultiIdRef) -> Option<Space>;
  pub fn isl_multi_id_get_domain_space(multi: MultiIdRef) -> Option<Space>;
  pub fn isl_multi_id_get_list(multi: MultiIdRef) -> Option<IdList>;
  pub fn isl_multi_id_from_id_list(space: Space, list: IdList) -> Option<MultiId>;
  pub fn isl_multi_id_copy(multi: MultiIdRef) -> Option<MultiId>;
  pub fn isl_multi_id_free(multi: MultiId) -> *mut c_void;
  pub fn isl_multi_id_plain_is_equal(multi1: MultiIdRef, multi2: MultiIdRef) -> Bool;
  pub fn isl_multi_id_reset_user(multi: MultiId) -> Option<MultiId>;
  pub fn isl_multi_id_size(multi: MultiIdRef) -> c_int;
  pub fn isl_multi_id_get_at(multi: MultiIdRef, pos: c_int) -> Option<Id>;
  pub fn isl_multi_id_get_id(multi: MultiIdRef, pos: c_int) -> Option<Id>;
  pub fn isl_multi_id_set_at(multi: MultiId, pos: c_int, el: Id) -> Option<MultiId>;
  pub fn isl_multi_id_set_id(multi: MultiId, pos: c_int, el: Id) -> Option<MultiId>;
  pub fn isl_multi_id_range_splice(multi1: MultiId, pos: c_uint, multi2: MultiId) -> Option<MultiId>;
  pub fn isl_multi_id_flatten_range(multi: MultiId) -> Option<MultiId>;
  pub fn isl_multi_id_flat_range_product(multi1: MultiId, multi2: MultiId) -> Option<MultiId>;
  pub fn isl_multi_id_range_product(multi1: MultiId, multi2: MultiId) -> Option<MultiId>;
  pub fn isl_multi_id_factor_range(multi: MultiId) -> Option<MultiId>;
  pub fn isl_multi_id_range_is_wrapping(multi: MultiIdRef) -> Bool;
  pub fn isl_multi_id_range_factor_domain(multi: MultiId) -> Option<MultiId>;
  pub fn isl_multi_id_range_factor_range(multi: MultiId) -> Option<MultiId>;
  pub fn isl_multi_id_align_params(multi: MultiId, model: Space) -> Option<MultiId>;
  pub fn isl_multi_id_from_range(multi: MultiId) -> Option<MultiId>;
  pub fn isl_id_get_ctx(id: IdRef) -> Option<CtxRef>;
  pub fn isl_id_get_hash(id: IdRef) -> c_uint;
  pub fn isl_id_alloc(ctx: CtxRef, name: Option<CStr>, user: *mut c_void) -> Option<Id>;
  pub fn isl_id_copy(id: IdRef) -> Option<Id>;
  pub fn isl_id_free(id: Id) -> *mut c_void;
  pub fn isl_id_get_user(id: IdRef) -> *mut c_void;
  pub fn isl_id_get_name(id: IdRef) -> Option<CStr>;
  pub fn isl_id_set_free_user(id: Id, free_user: unsafe extern "C" fn(user: *mut c_void) -> ()) -> Option<Id>;
  pub fn isl_id_read_from_str(ctx: CtxRef, str: Option<CStr>) -> Option<Id>;
  pub fn isl_id_to_str(id: IdRef) -> Option<CString>;
  pub fn isl_printer_print_id(p: Printer, id: IdRef) -> Option<Printer>;
  pub fn isl_id_dump(id: IdRef) -> ();
  pub fn isl_multi_id_read_from_str(ctx: CtxRef, str: Option<CStr>) -> Option<MultiId>;
  pub fn isl_printer_print_multi_id(p: Printer, mi: MultiIdRef) -> Option<Printer>;
  pub fn isl_multi_id_dump(mi: MultiIdRef) -> ();
  pub fn isl_multi_id_to_str(mi: MultiIdRef) -> Option<CString>;
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
  #[inline(always)]
  pub fn id_read_from_str(self, str: Option<CStr>) -> Option<Id> {
    unsafe {
      let ret = isl_id_read_from_str(self.to(), str.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn multi_id_read_from_str(self, str: Option<CStr>) -> Option<MultiId> {
    unsafe {
      let ret = isl_multi_id_read_from_str(self.to(), str.to());
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
  pub fn clear(self) -> Option<IdList> {
    unsafe {
      let ret = isl_id_list_clear(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn swap(self, pos1: c_uint, pos2: c_uint) -> Option<IdList> {
    unsafe {
      let ret = isl_id_list_swap(self.to(), pos1.to(), pos2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reverse(self) -> Option<IdList> {
    unsafe {
      let ret = isl_id_list_reverse(self.to());
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
    unsafe extern "C" fn fn1<F: FnMut(Id) -> Option<Id>>(el: Id, user: *mut c_void) -> Option<Id> { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_id_list_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort<F1: FnMut(IdRef, IdRef) -> c_int>(self, cmp: &mut F1) -> Option<IdList> {
    unsafe extern "C" fn fn1<F: FnMut(IdRef, IdRef) -> c_int>(a: IdRef, b: IdRef, user: *mut c_void) -> c_int { (*(user as *mut F))(a.to(), b.to()) }
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
  pub fn size(self) -> c_int {
    unsafe {
      let ret = isl_id_list_size(self.to());
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
  pub fn get_at(self, index: c_int) -> Option<Id> {
    unsafe {
      let ret = isl_id_list_get_at(self.to(), index.to());
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
  pub fn foreach<F1: FnMut(Id) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(Id) -> Stat>(el: Id, user: *mut c_void) -> Stat { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_id_list_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn every<F1: FnMut(IdRef) -> Bool>(self, test: &mut F1) -> Bool {
    unsafe extern "C" fn fn1<F: FnMut(IdRef) -> Bool>(el: IdRef, user: *mut c_void) -> Bool { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_id_list_every(self.to(), fn1::<F1>, test as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_scc<F1: FnMut(IdRef, IdRef) -> Bool, F2: FnMut(IdList) -> Stat>(self, follows: &mut F1, fn_: &mut F2) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(IdRef, IdRef) -> Bool>(a: IdRef, b: IdRef, user: *mut c_void) -> Bool { (*(user as *mut F))(a.to(), b.to()) }
    unsafe extern "C" fn fn2<F: FnMut(IdList) -> Stat>(scc: IdList, user: *mut c_void) -> Stat { (*(user as *mut F))(scc.to()) }
    unsafe {
      let ret = isl_id_list_foreach_scc(self.to(), fn1::<F1>, follows as *mut _ as _, fn2::<F2>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_id_list_to_str(self.to());
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

impl MultiId {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_multi_id_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reset_user(self) -> Option<MultiId> {
    unsafe {
      let ret = isl_multi_id_reset_user(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_at(self, pos: c_int, el: Id) -> Option<MultiId> {
    unsafe {
      let ret = isl_multi_id_set_at(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_id(self, pos: c_int, el: Id) -> Option<MultiId> {
    unsafe {
      let ret = isl_multi_id_set_id(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_splice(self, pos: c_uint, multi2: MultiId) -> Option<MultiId> {
    unsafe {
      let ret = isl_multi_id_range_splice(self.to(), pos.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flatten_range(self) -> Option<MultiId> {
    unsafe {
      let ret = isl_multi_id_flatten_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flat_range_product(self, multi2: MultiId) -> Option<MultiId> {
    unsafe {
      let ret = isl_multi_id_flat_range_product(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_product(self, multi2: MultiId) -> Option<MultiId> {
    unsafe {
      let ret = isl_multi_id_range_product(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn factor_range(self) -> Option<MultiId> {
    unsafe {
      let ret = isl_multi_id_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_factor_domain(self) -> Option<MultiId> {
    unsafe {
      let ret = isl_multi_id_range_factor_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_factor_range(self) -> Option<MultiId> {
    unsafe {
      let ret = isl_multi_id_range_factor_range(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn align_params(self, model: Space) -> Option<MultiId> {
    unsafe {
      let ret = isl_multi_id_align_params(self.to(), model.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn from_range(self) -> Option<MultiId> {
    unsafe {
      let ret = isl_multi_id_from_range(self.to());
      (ret).to()
    }
  }
}

impl MultiIdRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_multi_id_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_multi_id_get_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_domain_space(self) -> Option<Space> {
    unsafe {
      let ret = isl_multi_id_get_domain_space(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_list(self) -> Option<IdList> {
    unsafe {
      let ret = isl_multi_id_get_list(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<MultiId> {
    unsafe {
      let ret = isl_multi_id_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn plain_is_equal(self, multi2: MultiIdRef) -> Bool {
    unsafe {
      let ret = isl_multi_id_plain_is_equal(self.to(), multi2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn size(self) -> c_int {
    unsafe {
      let ret = isl_multi_id_size(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_at(self, pos: c_int) -> Option<Id> {
    unsafe {
      let ret = isl_multi_id_get_at(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_id(self, pos: c_int) -> Option<Id> {
    unsafe {
      let ret = isl_multi_id_get_id(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn range_is_wrapping(self) -> Bool {
    unsafe {
      let ret = isl_multi_id_range_is_wrapping(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_multi_id_dump(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_multi_id_to_str(self.to());
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
  #[inline(always)]
  pub fn print_multi_id(self, mi: MultiIdRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_multi_id(self.to(), mi.to());
      (ret).to()
    }
  }
}

impl Space {
  #[inline(always)]
  pub fn multi_id_from_id_list(self, list: IdList) -> Option<MultiId> {
    unsafe {
      let ret = isl_multi_id_from_id_list(self.to(), list.to());
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

impl fmt::Display for IdListRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for IdList {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl fmt::Display for IdRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for Id {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl Drop for MultiId {
  fn drop(&mut self) { MultiId(self.0).free() }
}

impl fmt::Display for MultiIdRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for MultiId {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

