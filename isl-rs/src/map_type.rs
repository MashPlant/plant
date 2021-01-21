use crate::*;

extern "C" {
  pub fn isl_basic_set_list_get_ctx(list: BasicSetListRef) -> Option<CtxRef>;
  pub fn isl_basic_set_list_from_basic_set(el: BasicSet) -> Option<BasicSetList>;
  pub fn isl_basic_set_list_alloc(ctx: CtxRef, n: c_int) -> Option<BasicSetList>;
  pub fn isl_basic_set_list_copy(list: BasicSetListRef) -> Option<BasicSetList>;
  pub fn isl_basic_set_list_free(list: BasicSetList) -> *mut c_void;
  pub fn isl_basic_set_list_add(list: BasicSetList, el: BasicSet) -> Option<BasicSetList>;
  pub fn isl_basic_set_list_insert(list: BasicSetList, pos: c_uint, el: BasicSet) -> Option<BasicSetList>;
  pub fn isl_basic_set_list_drop(list: BasicSetList, first: c_uint, n: c_uint) -> Option<BasicSetList>;
  pub fn isl_basic_set_list_clear(list: BasicSetList) -> Option<BasicSetList>;
  pub fn isl_basic_set_list_swap(list: BasicSetList, pos1: c_uint, pos2: c_uint) -> Option<BasicSetList>;
  pub fn isl_basic_set_list_reverse(list: BasicSetList) -> Option<BasicSetList>;
  pub fn isl_basic_set_list_concat(list1: BasicSetList, list2: BasicSetList) -> Option<BasicSetList>;
  pub fn isl_basic_set_list_size(list: BasicSetListRef) -> c_int;
  pub fn isl_basic_set_list_n_basic_set(list: BasicSetListRef) -> c_int;
  pub fn isl_basic_set_list_get_at(list: BasicSetListRef, index: c_int) -> Option<BasicSet>;
  pub fn isl_basic_set_list_get_basic_set(list: BasicSetListRef, index: c_int) -> Option<BasicSet>;
  pub fn isl_basic_set_list_set_basic_set(list: BasicSetList, index: c_int, el: BasicSet) -> Option<BasicSetList>;
  pub fn isl_basic_set_list_foreach(list: BasicSetListRef, fn_: unsafe extern "C" fn(el: BasicSet, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_basic_set_list_every(list: BasicSetListRef, test: unsafe extern "C" fn(el: BasicSetRef, user: *mut c_void) -> Bool, user: *mut c_void) -> Bool;
  pub fn isl_basic_set_list_map(list: BasicSetList, fn_: unsafe extern "C" fn(el: BasicSet, user: *mut c_void) -> Option<BasicSet>, user: *mut c_void) -> Option<BasicSetList>;
  pub fn isl_basic_set_list_sort(list: BasicSetList, cmp: unsafe extern "C" fn(a: BasicSetRef, b: BasicSetRef, user: *mut c_void) -> c_int, user: *mut c_void) -> Option<BasicSetList>;
  pub fn isl_basic_set_list_foreach_scc(list: BasicSetListRef, follows: unsafe extern "C" fn(a: BasicSetRef, b: BasicSetRef, user: *mut c_void) -> Bool, follows_user: *mut c_void, fn_: unsafe extern "C" fn(scc: BasicSetList, user: *mut c_void) -> Stat, fn_user: *mut c_void) -> Stat;
  pub fn isl_basic_set_list_to_str(list: BasicSetListRef) -> Option<CString>;
  pub fn isl_printer_print_basic_set_list(p: Printer, list: BasicSetListRef) -> Option<Printer>;
  pub fn isl_basic_set_list_dump(list: BasicSetListRef) -> ();
  pub fn isl_set_list_get_ctx(list: SetListRef) -> Option<CtxRef>;
  pub fn isl_set_list_from_set(el: Set) -> Option<SetList>;
  pub fn isl_set_list_alloc(ctx: CtxRef, n: c_int) -> Option<SetList>;
  pub fn isl_set_list_copy(list: SetListRef) -> Option<SetList>;
  pub fn isl_set_list_free(list: SetList) -> *mut c_void;
  pub fn isl_set_list_add(list: SetList, el: Set) -> Option<SetList>;
  pub fn isl_set_list_insert(list: SetList, pos: c_uint, el: Set) -> Option<SetList>;
  pub fn isl_set_list_drop(list: SetList, first: c_uint, n: c_uint) -> Option<SetList>;
  pub fn isl_set_list_clear(list: SetList) -> Option<SetList>;
  pub fn isl_set_list_swap(list: SetList, pos1: c_uint, pos2: c_uint) -> Option<SetList>;
  pub fn isl_set_list_reverse(list: SetList) -> Option<SetList>;
  pub fn isl_set_list_concat(list1: SetList, list2: SetList) -> Option<SetList>;
  pub fn isl_set_list_size(list: SetListRef) -> c_int;
  pub fn isl_set_list_n_set(list: SetListRef) -> c_int;
  pub fn isl_set_list_get_at(list: SetListRef, index: c_int) -> Option<Set>;
  pub fn isl_set_list_get_set(list: SetListRef, index: c_int) -> Option<Set>;
  pub fn isl_set_list_set_set(list: SetList, index: c_int, el: Set) -> Option<SetList>;
  pub fn isl_set_list_foreach(list: SetListRef, fn_: unsafe extern "C" fn(el: Set, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_set_list_every(list: SetListRef, test: unsafe extern "C" fn(el: SetRef, user: *mut c_void) -> Bool, user: *mut c_void) -> Bool;
  pub fn isl_set_list_map(list: SetList, fn_: unsafe extern "C" fn(el: Set, user: *mut c_void) -> Option<Set>, user: *mut c_void) -> Option<SetList>;
  pub fn isl_set_list_sort(list: SetList, cmp: unsafe extern "C" fn(a: SetRef, b: SetRef, user: *mut c_void) -> c_int, user: *mut c_void) -> Option<SetList>;
  pub fn isl_set_list_foreach_scc(list: SetListRef, follows: unsafe extern "C" fn(a: SetRef, b: SetRef, user: *mut c_void) -> Bool, follows_user: *mut c_void, fn_: unsafe extern "C" fn(scc: SetList, user: *mut c_void) -> Stat, fn_user: *mut c_void) -> Stat;
  pub fn isl_set_list_to_str(list: SetListRef) -> Option<CString>;
  pub fn isl_printer_print_set_list(p: Printer, list: SetListRef) -> Option<Printer>;
  pub fn isl_set_list_dump(list: SetListRef) -> ();
}

#[repr(transparent)]
#[derive(Debug)]
pub struct BasicMap(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct BasicMapRef(pub NonNull<c_void>);

impl_try!(BasicMap);
impl_try!(BasicMapRef);

impl BasicMap {
  #[inline(always)]
  pub fn read(&self) -> BasicMap { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: BasicMap) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<BasicMapRef> for BasicMap {
  #[inline(always)]
  fn as_ref(&self) -> &BasicMapRef { unsafe { mem::transmute(self) } }
}

impl Deref for BasicMap {
  type Target = BasicMapRef;
  #[inline(always)]
  fn deref(&self) -> &BasicMapRef { self.as_ref() }
}

impl To<Option<BasicMap>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<BasicMap> { NonNull::new(self).map(BasicMap) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct BasicMapList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct BasicMapListRef(pub NonNull<c_void>);

impl_try!(BasicMapList);
impl_try!(BasicMapListRef);

impl BasicMapList {
  #[inline(always)]
  pub fn read(&self) -> BasicMapList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: BasicMapList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<BasicMapListRef> for BasicMapList {
  #[inline(always)]
  fn as_ref(&self) -> &BasicMapListRef { unsafe { mem::transmute(self) } }
}

impl Deref for BasicMapList {
  type Target = BasicMapListRef;
  #[inline(always)]
  fn deref(&self) -> &BasicMapListRef { self.as_ref() }
}

impl To<Option<BasicMapList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<BasicMapList> { NonNull::new(self).map(BasicMapList) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Map(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct MapRef(pub NonNull<c_void>);

impl_try!(Map);
impl_try!(MapRef);

impl Map {
  #[inline(always)]
  pub fn read(&self) -> Map { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Map) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MapRef> for Map {
  #[inline(always)]
  fn as_ref(&self) -> &MapRef { unsafe { mem::transmute(self) } }
}

impl Deref for Map {
  type Target = MapRef;
  #[inline(always)]
  fn deref(&self) -> &MapRef { self.as_ref() }
}

impl To<Option<Map>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Map> { NonNull::new(self).map(Map) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct MapList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct MapListRef(pub NonNull<c_void>);

impl_try!(MapList);
impl_try!(MapListRef);

impl MapList {
  #[inline(always)]
  pub fn read(&self) -> MapList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: MapList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MapListRef> for MapList {
  #[inline(always)]
  fn as_ref(&self) -> &MapListRef { unsafe { mem::transmute(self) } }
}

impl Deref for MapList {
  type Target = MapListRef;
  #[inline(always)]
  fn deref(&self) -> &MapListRef { self.as_ref() }
}

impl To<Option<MapList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<MapList> { NonNull::new(self).map(MapList) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct BasicSet(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct BasicSetRef(pub NonNull<c_void>);

impl_try!(BasicSet);
impl_try!(BasicSetRef);

impl BasicSet {
  #[inline(always)]
  pub fn read(&self) -> BasicSet { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: BasicSet) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<BasicSetRef> for BasicSet {
  #[inline(always)]
  fn as_ref(&self) -> &BasicSetRef { unsafe { mem::transmute(self) } }
}

impl Deref for BasicSet {
  type Target = BasicSetRef;
  #[inline(always)]
  fn deref(&self) -> &BasicSetRef { self.as_ref() }
}

impl To<Option<BasicSet>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<BasicSet> { NonNull::new(self).map(BasicSet) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct BasicSetList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct BasicSetListRef(pub NonNull<c_void>);

impl_try!(BasicSetList);
impl_try!(BasicSetListRef);

impl BasicSetList {
  #[inline(always)]
  pub fn read(&self) -> BasicSetList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: BasicSetList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<BasicSetListRef> for BasicSetList {
  #[inline(always)]
  fn as_ref(&self) -> &BasicSetListRef { unsafe { mem::transmute(self) } }
}

impl Deref for BasicSetList {
  type Target = BasicSetListRef;
  #[inline(always)]
  fn deref(&self) -> &BasicSetListRef { self.as_ref() }
}

impl To<Option<BasicSetList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<BasicSetList> { NonNull::new(self).map(BasicSetList) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Set(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct SetRef(pub NonNull<c_void>);

impl_try!(Set);
impl_try!(SetRef);

impl Set {
  #[inline(always)]
  pub fn read(&self) -> Set { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Set) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<SetRef> for Set {
  #[inline(always)]
  fn as_ref(&self) -> &SetRef { unsafe { mem::transmute(self) } }
}

impl Deref for Set {
  type Target = SetRef;
  #[inline(always)]
  fn deref(&self) -> &SetRef { self.as_ref() }
}

impl To<Option<Set>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Set> { NonNull::new(self).map(Set) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct SetList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct SetListRef(pub NonNull<c_void>);

impl_try!(SetList);
impl_try!(SetListRef);

impl SetList {
  #[inline(always)]
  pub fn read(&self) -> SetList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: SetList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<SetListRef> for SetList {
  #[inline(always)]
  fn as_ref(&self) -> &SetListRef { unsafe { mem::transmute(self) } }
}

impl Deref for SetList {
  type Target = SetListRef;
  #[inline(always)]
  fn deref(&self) -> &SetListRef { self.as_ref() }
}

impl To<Option<SetList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<SetList> { NonNull::new(self).map(SetList) }
}

impl BasicSet {
  #[inline(always)]
  pub fn list_from_basic_set(self) -> Option<BasicSetList> {
    unsafe {
      let ret = isl_basic_set_list_from_basic_set(self.to());
      (ret).to()
    }
  }
}

impl BasicSetList {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_basic_set_list_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, el: BasicSet) -> Option<BasicSetList> {
    unsafe {
      let ret = isl_basic_set_list_add(self.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert(self, pos: c_uint, el: BasicSet) -> Option<BasicSetList> {
    unsafe {
      let ret = isl_basic_set_list_insert(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, first: c_uint, n: c_uint) -> Option<BasicSetList> {
    unsafe {
      let ret = isl_basic_set_list_drop(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn clear(self) -> Option<BasicSetList> {
    unsafe {
      let ret = isl_basic_set_list_clear(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn swap(self, pos1: c_uint, pos2: c_uint) -> Option<BasicSetList> {
    unsafe {
      let ret = isl_basic_set_list_swap(self.to(), pos1.to(), pos2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reverse(self) -> Option<BasicSetList> {
    unsafe {
      let ret = isl_basic_set_list_reverse(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, list2: BasicSetList) -> Option<BasicSetList> {
    unsafe {
      let ret = isl_basic_set_list_concat(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_basic_set(self, index: c_int, el: BasicSet) -> Option<BasicSetList> {
    unsafe {
      let ret = isl_basic_set_list_set_basic_set(self.to(), index.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map<F1: FnMut(BasicSet) -> Option<BasicSet>>(self, fn_: &mut F1) -> Option<BasicSetList> {
    unsafe extern "C" fn fn1<F: FnMut(BasicSet) -> Option<BasicSet>>(el: BasicSet, user: *mut c_void) -> Option<BasicSet> { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_basic_set_list_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort<F1: FnMut(BasicSetRef, BasicSetRef) -> c_int>(self, cmp: &mut F1) -> Option<BasicSetList> {
    unsafe extern "C" fn fn1<F: FnMut(BasicSetRef, BasicSetRef) -> c_int>(a: BasicSetRef, b: BasicSetRef, user: *mut c_void) -> c_int { (*(user as *mut F))(a.to(), b.to()) }
    unsafe {
      let ret = isl_basic_set_list_sort(self.to(), fn1::<F1>, cmp as *mut _ as _);
      (ret).to()
    }
  }
}

impl BasicSetListRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_basic_set_list_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<BasicSetList> {
    unsafe {
      let ret = isl_basic_set_list_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn size(self) -> c_int {
    unsafe {
      let ret = isl_basic_set_list_size(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_basic_set(self) -> c_int {
    unsafe {
      let ret = isl_basic_set_list_n_basic_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_at(self, index: c_int) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_list_get_at(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_basic_set(self, index: c_int) -> Option<BasicSet> {
    unsafe {
      let ret = isl_basic_set_list_get_basic_set(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(BasicSet) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(BasicSet) -> Stat>(el: BasicSet, user: *mut c_void) -> Stat { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_basic_set_list_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn every<F1: FnMut(BasicSetRef) -> Bool>(self, test: &mut F1) -> Bool {
    unsafe extern "C" fn fn1<F: FnMut(BasicSetRef) -> Bool>(el: BasicSetRef, user: *mut c_void) -> Bool { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_basic_set_list_every(self.to(), fn1::<F1>, test as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_scc<F1: FnMut(BasicSetRef, BasicSetRef) -> Bool, F2: FnMut(BasicSetList) -> Stat>(self, follows: &mut F1, fn_: &mut F2) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(BasicSetRef, BasicSetRef) -> Bool>(a: BasicSetRef, b: BasicSetRef, user: *mut c_void) -> Bool { (*(user as *mut F))(a.to(), b.to()) }
    unsafe extern "C" fn fn2<F: FnMut(BasicSetList) -> Stat>(scc: BasicSetList, user: *mut c_void) -> Stat { (*(user as *mut F))(scc.to()) }
    unsafe {
      let ret = isl_basic_set_list_foreach_scc(self.to(), fn1::<F1>, follows as *mut _ as _, fn2::<F2>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_basic_set_list_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_basic_set_list_dump(self.to());
      (ret).to()
    }
  }
}

impl CtxRef {
  #[inline(always)]
  pub fn basic_set_list_alloc(self, n: c_int) -> Option<BasicSetList> {
    unsafe {
      let ret = isl_basic_set_list_alloc(self.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_list_alloc(self, n: c_int) -> Option<SetList> {
    unsafe {
      let ret = isl_set_list_alloc(self.to(), n.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_basic_set_list(self, list: BasicSetListRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_basic_set_list(self.to(), list.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_set_list(self, list: SetListRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_set_list(self.to(), list.to());
      (ret).to()
    }
  }
}

impl Set {
  #[inline(always)]
  pub fn list_from_set(self) -> Option<SetList> {
    unsafe {
      let ret = isl_set_list_from_set(self.to());
      (ret).to()
    }
  }
}

impl SetList {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_set_list_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, el: Set) -> Option<SetList> {
    unsafe {
      let ret = isl_set_list_add(self.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert(self, pos: c_uint, el: Set) -> Option<SetList> {
    unsafe {
      let ret = isl_set_list_insert(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, first: c_uint, n: c_uint) -> Option<SetList> {
    unsafe {
      let ret = isl_set_list_drop(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn clear(self) -> Option<SetList> {
    unsafe {
      let ret = isl_set_list_clear(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn swap(self, pos1: c_uint, pos2: c_uint) -> Option<SetList> {
    unsafe {
      let ret = isl_set_list_swap(self.to(), pos1.to(), pos2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn reverse(self) -> Option<SetList> {
    unsafe {
      let ret = isl_set_list_reverse(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, list2: SetList) -> Option<SetList> {
    unsafe {
      let ret = isl_set_list_concat(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_set(self, index: c_int, el: Set) -> Option<SetList> {
    unsafe {
      let ret = isl_set_list_set_set(self.to(), index.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map<F1: FnMut(Set) -> Option<Set>>(self, fn_: &mut F1) -> Option<SetList> {
    unsafe extern "C" fn fn1<F: FnMut(Set) -> Option<Set>>(el: Set, user: *mut c_void) -> Option<Set> { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_set_list_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort<F1: FnMut(SetRef, SetRef) -> c_int>(self, cmp: &mut F1) -> Option<SetList> {
    unsafe extern "C" fn fn1<F: FnMut(SetRef, SetRef) -> c_int>(a: SetRef, b: SetRef, user: *mut c_void) -> c_int { (*(user as *mut F))(a.to(), b.to()) }
    unsafe {
      let ret = isl_set_list_sort(self.to(), fn1::<F1>, cmp as *mut _ as _);
      (ret).to()
    }
  }
}

impl SetListRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_set_list_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<SetList> {
    unsafe {
      let ret = isl_set_list_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn size(self) -> c_int {
    unsafe {
      let ret = isl_set_list_size(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_set(self) -> c_int {
    unsafe {
      let ret = isl_set_list_n_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_at(self, index: c_int) -> Option<Set> {
    unsafe {
      let ret = isl_set_list_get_at(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_set(self, index: c_int) -> Option<Set> {
    unsafe {
      let ret = isl_set_list_get_set(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(Set) -> Stat>(self, fn_: &mut F1) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(Set) -> Stat>(el: Set, user: *mut c_void) -> Stat { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_set_list_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn every<F1: FnMut(SetRef) -> Bool>(self, test: &mut F1) -> Bool {
    unsafe extern "C" fn fn1<F: FnMut(SetRef) -> Bool>(el: SetRef, user: *mut c_void) -> Bool { (*(user as *mut F))(el.to()) }
    unsafe {
      let ret = isl_set_list_every(self.to(), fn1::<F1>, test as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_scc<F1: FnMut(SetRef, SetRef) -> Bool, F2: FnMut(SetList) -> Stat>(self, follows: &mut F1, fn_: &mut F2) -> Stat {
    unsafe extern "C" fn fn1<F: FnMut(SetRef, SetRef) -> Bool>(a: SetRef, b: SetRef, user: *mut c_void) -> Bool { (*(user as *mut F))(a.to(), b.to()) }
    unsafe extern "C" fn fn2<F: FnMut(SetList) -> Stat>(scc: SetList, user: *mut c_void) -> Stat { (*(user as *mut F))(scc.to()) }
    unsafe {
      let ret = isl_set_list_foreach_scc(self.to(), fn1::<F1>, follows as *mut _ as _, fn2::<F2>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_set_list_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_set_list_dump(self.to());
      (ret).to()
    }
  }
}

impl Drop for BasicSetList {
  fn drop(&mut self) { BasicSetList(self.0).free() }
}

impl fmt::Display for BasicSetListRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for BasicSetList {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl Drop for SetList {
  fn drop(&mut self) { SetList(self.0).free() }
}

impl fmt::Display for SetListRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().ok_or(fmt::Error)?) }
}

impl fmt::Display for SetList {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

