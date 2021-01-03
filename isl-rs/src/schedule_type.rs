use crate::*;

extern "C" {
}

#[repr(transparent)]
#[derive(Debug)]
pub struct ScheduleNode(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct ScheduleNodeRef(pub NonNull<c_void>);

impl ScheduleNode {
  #[inline(always)]
  pub fn read(&self) -> ScheduleNode { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: ScheduleNode) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<ScheduleNodeRef> for ScheduleNode {
  #[inline(always)]
  fn as_ref(&self) -> &ScheduleNodeRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for ScheduleNode {
  type Target = ScheduleNodeRef;
  #[inline(always)]
  fn deref(&self) -> &ScheduleNodeRef { self.as_ref() }
}

impl To<Option<ScheduleNode>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<ScheduleNode> { NonNull::new(self).map(ScheduleNode) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Schedule(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct ScheduleRef(pub NonNull<c_void>);

impl Schedule {
  #[inline(always)]
  pub fn read(&self) -> Schedule { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Schedule) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<ScheduleRef> for Schedule {
  #[inline(always)]
  fn as_ref(&self) -> &ScheduleRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for Schedule {
  type Target = ScheduleRef;
  #[inline(always)]
  fn deref(&self) -> &ScheduleRef { self.as_ref() }
}

impl To<Option<Schedule>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Schedule> { NonNull::new(self).map(Schedule) }
}

