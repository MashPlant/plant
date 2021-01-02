use crate::*;

extern "C" {
  pub fn isl_restriction_free(restr: Restriction) -> *mut c_void;
  pub fn isl_restriction_empty(source_map: Map) -> Option<Restriction>;
  pub fn isl_restriction_none(source_map: Map) -> Option<Restriction>;
  pub fn isl_restriction_input(source_restr: Set, sink_restr: Set) -> Option<Restriction>;
  pub fn isl_restriction_output(source_restr: Set) -> Option<Restriction>;
  pub fn isl_restriction_get_ctx(restr: RestrictionRef) -> Option<CtxRef>;
  pub fn isl_access_info_alloc(sink: Map, sink_user: *mut c_void, fn_: unsafe extern "C" fn(first: *mut c_void, second: *mut c_void) -> c_int, max_source: c_int) -> Option<AccessInfo>;
  pub fn isl_access_info_set_restrict(acc: AccessInfo, fn_: unsafe extern "C" fn(source_map: MapRef, sink: SetRef, source_user: *mut c_void, user: *mut c_void) -> Option<Restriction>, user: *mut c_void) -> Option<AccessInfo>;
  pub fn isl_access_info_add_source(acc: AccessInfo, source: Map, must: c_int, source_user: *mut c_void) -> Option<AccessInfo>;
  pub fn isl_access_info_free(acc: AccessInfo) -> *mut c_void;
  pub fn isl_access_info_get_ctx(acc: AccessInfoRef) -> Option<CtxRef>;
  pub fn isl_access_info_compute_flow(acc: AccessInfo) -> Option<Flow>;
  pub fn isl_flow_foreach(deps: FlowRef, fn_: unsafe extern "C" fn(dep: Map, must: c_int, dep_user: *mut c_void, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_flow_get_no_source(deps: FlowRef, must: c_int) -> Option<Map>;
  pub fn isl_flow_free(deps: Flow) -> ();
  pub fn isl_flow_get_ctx(deps: FlowRef) -> Option<CtxRef>;
  pub fn isl_union_access_info_from_sink(sink: UnionMap) -> Option<UnionAccessInfo>;
  pub fn isl_union_access_info_set_must_source(access: UnionAccessInfo, must_source: UnionMap) -> Option<UnionAccessInfo>;
  pub fn isl_union_access_info_set_may_source(access: UnionAccessInfo, may_source: UnionMap) -> Option<UnionAccessInfo>;
  pub fn isl_union_access_info_set_kill(access: UnionAccessInfo, kill: UnionMap) -> Option<UnionAccessInfo>;
  pub fn isl_union_access_info_set_schedule(access: UnionAccessInfo, schedule: Schedule) -> Option<UnionAccessInfo>;
  pub fn isl_union_access_info_set_schedule_map(access: UnionAccessInfo, schedule_map: UnionMap) -> Option<UnionAccessInfo>;
  pub fn isl_union_access_info_copy(access: UnionAccessInfoRef) -> Option<UnionAccessInfo>;
  pub fn isl_union_access_info_free(access: UnionAccessInfo) -> *mut c_void;
  pub fn isl_union_access_info_get_ctx(access: UnionAccessInfoRef) -> Option<CtxRef>;
  pub fn isl_union_access_info_read_from_file(ctx: CtxRef, input: *mut FILE) -> Option<UnionAccessInfo>;
  pub fn isl_printer_print_union_access_info(p: Printer, access: UnionAccessInfoRef) -> Option<Printer>;
  pub fn isl_union_access_info_to_str(access: UnionAccessInfoRef) -> Option<CString>;
  pub fn isl_union_access_info_compute_flow(access: UnionAccessInfo) -> Option<UnionFlow>;
  pub fn isl_union_flow_get_ctx(flow: UnionFlowRef) -> Option<CtxRef>;
  pub fn isl_union_flow_copy(flow: UnionFlowRef) -> Option<UnionFlow>;
  pub fn isl_union_flow_get_must_dependence(flow: UnionFlowRef) -> Option<UnionMap>;
  pub fn isl_union_flow_get_may_dependence(flow: UnionFlowRef) -> Option<UnionMap>;
  pub fn isl_union_flow_get_full_must_dependence(flow: UnionFlowRef) -> Option<UnionMap>;
  pub fn isl_union_flow_get_full_may_dependence(flow: UnionFlowRef) -> Option<UnionMap>;
  pub fn isl_union_flow_get_must_no_source(flow: UnionFlowRef) -> Option<UnionMap>;
  pub fn isl_union_flow_get_may_no_source(flow: UnionFlowRef) -> Option<UnionMap>;
  pub fn isl_union_flow_free(flow: UnionFlow) -> *mut c_void;
  pub fn isl_printer_print_union_flow(p: Printer, flow: UnionFlowRef) -> Option<Printer>;
  pub fn isl_union_flow_to_str(flow: UnionFlowRef) -> Option<CString>;
  pub fn isl_union_map_compute_flow(sink: UnionMap, must_source: UnionMap, may_source: UnionMap, schedule: UnionMap, must_dep: *mut UnionMap, may_dep: *mut UnionMap, must_no_source: *mut UnionMap, may_no_source: *mut UnionMap) -> c_int;
}

#[repr(transparent)]
#[derive(Debug)]
pub struct AccessLevelBefore(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct AccessLevelBeforeRef(pub NonNull<c_void>);

impl AccessLevelBefore {
  #[inline(always)]
  pub fn read(&self) -> AccessLevelBefore { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: AccessLevelBefore) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<AccessLevelBeforeRef> for AccessLevelBefore {
  #[inline(always)]
  fn as_ref(&self) -> &AccessLevelBeforeRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for AccessLevelBefore {
  type Target = AccessLevelBeforeRef;
  #[inline(always)]
  fn deref(&self) -> &AccessLevelBeforeRef { self.as_ref() }
}

impl To<Option<AccessLevelBefore>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<AccessLevelBefore> { NonNull::new(self).map(AccessLevelBefore) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Restriction(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct RestrictionRef(pub NonNull<c_void>);

impl Restriction {
  #[inline(always)]
  pub fn read(&self) -> Restriction { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Restriction) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<RestrictionRef> for Restriction {
  #[inline(always)]
  fn as_ref(&self) -> &RestrictionRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for Restriction {
  type Target = RestrictionRef;
  #[inline(always)]
  fn deref(&self) -> &RestrictionRef { self.as_ref() }
}

impl To<Option<Restriction>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Restriction> { NonNull::new(self).map(Restriction) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct AccessRestrict(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct AccessRestrictRef(pub NonNull<c_void>);

impl AccessRestrict {
  #[inline(always)]
  pub fn read(&self) -> AccessRestrict { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: AccessRestrict) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<AccessRestrictRef> for AccessRestrict {
  #[inline(always)]
  fn as_ref(&self) -> &AccessRestrictRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for AccessRestrict {
  type Target = AccessRestrictRef;
  #[inline(always)]
  fn deref(&self) -> &AccessRestrictRef { self.as_ref() }
}

impl To<Option<AccessRestrict>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<AccessRestrict> { NonNull::new(self).map(AccessRestrict) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct AccessInfo(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct AccessInfoRef(pub NonNull<c_void>);

impl AccessInfo {
  #[inline(always)]
  pub fn read(&self) -> AccessInfo { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: AccessInfo) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<AccessInfoRef> for AccessInfo {
  #[inline(always)]
  fn as_ref(&self) -> &AccessInfoRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for AccessInfo {
  type Target = AccessInfoRef;
  #[inline(always)]
  fn deref(&self) -> &AccessInfoRef { self.as_ref() }
}

impl To<Option<AccessInfo>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<AccessInfo> { NonNull::new(self).map(AccessInfo) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Flow(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct FlowRef(pub NonNull<c_void>);

impl Flow {
  #[inline(always)]
  pub fn read(&self) -> Flow { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Flow) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<FlowRef> for Flow {
  #[inline(always)]
  fn as_ref(&self) -> &FlowRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for Flow {
  type Target = FlowRef;
  #[inline(always)]
  fn deref(&self) -> &FlowRef { self.as_ref() }
}

impl To<Option<Flow>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Flow> { NonNull::new(self).map(Flow) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct UnionAccessInfo(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct UnionAccessInfoRef(pub NonNull<c_void>);

impl UnionAccessInfo {
  #[inline(always)]
  pub fn read(&self) -> UnionAccessInfo { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: UnionAccessInfo) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<UnionAccessInfoRef> for UnionAccessInfo {
  #[inline(always)]
  fn as_ref(&self) -> &UnionAccessInfoRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for UnionAccessInfo {
  type Target = UnionAccessInfoRef;
  #[inline(always)]
  fn deref(&self) -> &UnionAccessInfoRef { self.as_ref() }
}

impl To<Option<UnionAccessInfo>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<UnionAccessInfo> { NonNull::new(self).map(UnionAccessInfo) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct UnionFlow(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct UnionFlowRef(pub NonNull<c_void>);

impl UnionFlow {
  #[inline(always)]
  pub fn read(&self) -> UnionFlow { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: UnionFlow) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<UnionFlowRef> for UnionFlow {
  #[inline(always)]
  fn as_ref(&self) -> &UnionFlowRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for UnionFlow {
  type Target = UnionFlowRef;
  #[inline(always)]
  fn deref(&self) -> &UnionFlowRef { self.as_ref() }
}

impl To<Option<UnionFlow>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<UnionFlow> { NonNull::new(self).map(UnionFlow) }
}

impl AccessInfo {
  #[inline(always)]
  pub fn set_restrict<F1: FnMut(MapRef, SetRef, *mut c_void) -> Option<Restriction>>(self, fn_: &mut F1) -> Option<AccessInfo> {
    unsafe extern "C" fn fn1<F: FnMut(MapRef, SetRef, *mut c_void) -> Option<Restriction>>(source_map: MapRef, sink: SetRef, source_user: *mut c_void, user: *mut c_void) -> Option<Restriction> { (*(user as *mut F))(source_map.to(), sink.to(), source_user.to()).to() }
    unsafe {
      let ret = isl_access_info_set_restrict(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_source(self, source: Map, must: c_int, source_user: *mut c_void) -> Option<AccessInfo> {
    unsafe {
      let ret = isl_access_info_add_source(self.to(), source.to(), must.to(), source_user.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_access_info_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn compute_flow(self) -> Option<Flow> {
    unsafe {
      let ret = isl_access_info_compute_flow(self.to());
      (ret).to()
    }
  }
}

impl AccessInfoRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_access_info_get_ctx(self.to());
      (ret).to()
    }
  }
}

impl CtxRef {
  #[inline(always)]
  pub fn union_access_info_read_from_file(self, input: *mut FILE) -> Option<UnionAccessInfo> {
    unsafe {
      let ret = isl_union_access_info_read_from_file(self.to(), input.to());
      (ret).to()
    }
  }
}

impl Flow {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_flow_free(self.to());
      (ret).to()
    }
  }
}

impl FlowRef {
  #[inline(always)]
  pub fn foreach<F1: FnMut(Map, c_int, *mut c_void) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(Map, c_int, *mut c_void) -> Option<()>>(dep: Map, must: c_int, dep_user: *mut c_void, user: *mut c_void) -> Stat { (*(user as *mut F))(dep.to(), must.to(), dep_user.to()).to() }
    unsafe {
      let ret = isl_flow_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_no_source(self, must: c_int) -> Option<Map> {
    unsafe {
      let ret = isl_flow_get_no_source(self.to(), must.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_flow_get_ctx(self.to());
      (ret).to()
    }
  }
}

impl Map {
  #[inline(always)]
  pub fn restriction_empty(self) -> Option<Restriction> {
    unsafe {
      let ret = isl_restriction_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn restriction_none(self) -> Option<Restriction> {
    unsafe {
      let ret = isl_restriction_none(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn access_info_alloc(self, sink_user: *mut c_void, fn_: unsafe extern "C" fn(first: *mut c_void, second: *mut c_void) -> c_int, max_source: c_int) -> Option<AccessInfo> {
    unsafe {
      let ret = isl_access_info_alloc(self.to(), sink_user.to(), fn_.to(), max_source.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_union_access_info(self, access: UnionAccessInfoRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_union_access_info(self.to(), access.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_union_flow(self, flow: UnionFlowRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_union_flow(self.to(), flow.to());
      (ret).to()
    }
  }
}

impl Restriction {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_restriction_free(self.to());
      (ret).to()
    }
  }
}

impl RestrictionRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_restriction_get_ctx(self.to());
      (ret).to()
    }
  }
}

impl Set {
  #[inline(always)]
  pub fn restriction_input(self, sink_restr: Set) -> Option<Restriction> {
    unsafe {
      let ret = isl_restriction_input(self.to(), sink_restr.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn restriction_output(self) -> Option<Restriction> {
    unsafe {
      let ret = isl_restriction_output(self.to());
      (ret).to()
    }
  }
}

impl UnionAccessInfo {
  #[inline(always)]
  pub fn set_must_source(self, must_source: UnionMap) -> Option<UnionAccessInfo> {
    unsafe {
      let ret = isl_union_access_info_set_must_source(self.to(), must_source.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_may_source(self, may_source: UnionMap) -> Option<UnionAccessInfo> {
    unsafe {
      let ret = isl_union_access_info_set_may_source(self.to(), may_source.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_kill(self, kill: UnionMap) -> Option<UnionAccessInfo> {
    unsafe {
      let ret = isl_union_access_info_set_kill(self.to(), kill.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_schedule(self, schedule: Schedule) -> Option<UnionAccessInfo> {
    unsafe {
      let ret = isl_union_access_info_set_schedule(self.to(), schedule.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_schedule_map(self, schedule_map: UnionMap) -> Option<UnionAccessInfo> {
    unsafe {
      let ret = isl_union_access_info_set_schedule_map(self.to(), schedule_map.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_union_access_info_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn compute_flow(self) -> Option<UnionFlow> {
    unsafe {
      let ret = isl_union_access_info_compute_flow(self.to());
      (ret).to()
    }
  }
}

impl UnionAccessInfoRef {
  #[inline(always)]
  pub fn copy(self) -> Option<UnionAccessInfo> {
    unsafe {
      let ret = isl_union_access_info_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_union_access_info_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_union_access_info_to_str(self.to());
      (ret).to()
    }
  }
}

impl UnionFlow {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_union_flow_free(self.to());
      (ret).to()
    }
  }
}

impl UnionFlowRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_union_flow_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<UnionFlow> {
    unsafe {
      let ret = isl_union_flow_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_must_dependence(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_flow_get_must_dependence(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_may_dependence(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_flow_get_may_dependence(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_full_must_dependence(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_flow_get_full_must_dependence(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_full_may_dependence(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_flow_get_full_may_dependence(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_must_no_source(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_flow_get_must_no_source(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_may_no_source(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_union_flow_get_may_no_source(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_union_flow_to_str(self.to());
      (ret).to()
    }
  }
}

impl UnionMap {
  #[inline(always)]
  pub fn union_access_info_from_sink(self) -> Option<UnionAccessInfo> {
    unsafe {
      let ret = isl_union_access_info_from_sink(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn compute_flow(self, must_source: UnionMap, may_source: UnionMap, schedule: UnionMap) -> Option<(c_int, UnionMap, UnionMap, UnionMap, UnionMap)> {
    unsafe {
      let ref mut must_dep = 0 as *mut c_void;
      let ref mut may_dep = 0 as *mut c_void;
      let ref mut must_no_source = 0 as *mut c_void;
      let ref mut may_no_source = 0 as *mut c_void;
      let ret = isl_union_map_compute_flow(self.to(), must_source.to(), may_source.to(), schedule.to(), must_dep as *mut _ as _, may_dep as *mut _ as _, must_no_source as *mut _ as _, may_no_source as *mut _ as _);
      (ret, *must_dep, *may_dep, *must_no_source, *may_no_source).to()
    }
  }
}

impl Drop for AccessInfo {
  fn drop(&mut self) { AccessInfo(self.0).free() }
}

impl Drop for Flow {
  fn drop(&mut self) { Flow(self.0).free() }
}

impl Drop for Restriction {
  fn drop(&mut self) { Restriction(self.0).free() }
}

impl Drop for UnionAccessInfo {
  fn drop(&mut self) { UnionAccessInfo(self.0).free() }
}

impl fmt::Display for UnionAccessInfoRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for UnionAccessInfo {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl Drop for UnionFlow {
  fn drop(&mut self) { UnionFlow(self.0).free() }
}

impl fmt::Display for UnionFlowRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for UnionFlow {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

