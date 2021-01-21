use crate::*;

extern "C" {
}

#[repr(transparent)]
#[derive(Debug)]
pub struct AstExpr(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct AstExprRef(pub NonNull<c_void>);

impl_try!(AstExpr);
impl_try!(AstExprRef);

impl AstExpr {
  #[inline(always)]
  pub fn read(&self) -> AstExpr { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: AstExpr) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<AstExprRef> for AstExpr {
  #[inline(always)]
  fn as_ref(&self) -> &AstExprRef { unsafe { mem::transmute(self) } }
}

impl Deref for AstExpr {
  type Target = AstExprRef;
  #[inline(always)]
  fn deref(&self) -> &AstExprRef { self.as_ref() }
}

impl To<Option<AstExpr>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<AstExpr> { NonNull::new(self).map(AstExpr) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct AstNode(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct AstNodeRef(pub NonNull<c_void>);

impl_try!(AstNode);
impl_try!(AstNodeRef);

impl AstNode {
  #[inline(always)]
  pub fn read(&self) -> AstNode { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: AstNode) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<AstNodeRef> for AstNode {
  #[inline(always)]
  fn as_ref(&self) -> &AstNodeRef { unsafe { mem::transmute(self) } }
}

impl Deref for AstNode {
  type Target = AstNodeRef;
  #[inline(always)]
  fn deref(&self) -> &AstNodeRef { self.as_ref() }
}

impl To<Option<AstNode>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<AstNode> { NonNull::new(self).map(AstNode) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct AstPrintOptions(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct AstPrintOptionsRef(pub NonNull<c_void>);

impl_try!(AstPrintOptions);
impl_try!(AstPrintOptionsRef);

impl AstPrintOptions {
  #[inline(always)]
  pub fn read(&self) -> AstPrintOptions { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: AstPrintOptions) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<AstPrintOptionsRef> for AstPrintOptions {
  #[inline(always)]
  fn as_ref(&self) -> &AstPrintOptionsRef { unsafe { mem::transmute(self) } }
}

impl Deref for AstPrintOptions {
  type Target = AstPrintOptionsRef;
  #[inline(always)]
  fn deref(&self) -> &AstPrintOptionsRef { self.as_ref() }
}

impl To<Option<AstPrintOptions>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<AstPrintOptions> { NonNull::new(self).map(AstPrintOptions) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct AstExprList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct AstExprListRef(pub NonNull<c_void>);

impl_try!(AstExprList);
impl_try!(AstExprListRef);

impl AstExprList {
  #[inline(always)]
  pub fn read(&self) -> AstExprList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: AstExprList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<AstExprListRef> for AstExprList {
  #[inline(always)]
  fn as_ref(&self) -> &AstExprListRef { unsafe { mem::transmute(self) } }
}

impl Deref for AstExprList {
  type Target = AstExprListRef;
  #[inline(always)]
  fn deref(&self) -> &AstExprListRef { self.as_ref() }
}

impl To<Option<AstExprList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<AstExprList> { NonNull::new(self).map(AstExprList) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct AstNodeList(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct AstNodeListRef(pub NonNull<c_void>);

impl_try!(AstNodeList);
impl_try!(AstNodeListRef);

impl AstNodeList {
  #[inline(always)]
  pub fn read(&self) -> AstNodeList { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: AstNodeList) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<AstNodeListRef> for AstNodeList {
  #[inline(always)]
  fn as_ref(&self) -> &AstNodeListRef { unsafe { mem::transmute(self) } }
}

impl Deref for AstNodeList {
  type Target = AstNodeListRef;
  #[inline(always)]
  fn deref(&self) -> &AstNodeListRef { self.as_ref() }
}

impl To<Option<AstNodeList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<AstNodeList> { NonNull::new(self).map(AstNodeList) }
}

