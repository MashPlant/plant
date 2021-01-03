use crate::*;

extern "C" {
  pub fn isl_ast_expr_list_get_ctx(list: AstExprListRef) -> Option<CtxRef>;
  pub fn isl_ast_expr_list_from_ast_expr(el: AstExpr) -> Option<AstExprList>;
  pub fn isl_ast_expr_list_alloc(ctx: CtxRef, n: c_int) -> Option<AstExprList>;
  pub fn isl_ast_expr_list_copy(list: AstExprListRef) -> Option<AstExprList>;
  pub fn isl_ast_expr_list_free(list: AstExprList) -> *mut c_void;
  pub fn isl_ast_expr_list_add(list: AstExprList, el: AstExpr) -> Option<AstExprList>;
  pub fn isl_ast_expr_list_insert(list: AstExprList, pos: c_uint, el: AstExpr) -> Option<AstExprList>;
  pub fn isl_ast_expr_list_drop(list: AstExprList, first: c_uint, n: c_uint) -> Option<AstExprList>;
  pub fn isl_ast_expr_list_concat(list1: AstExprList, list2: AstExprList) -> Option<AstExprList>;
  pub fn isl_ast_expr_list_n_ast_expr(list: AstExprListRef) -> c_int;
  pub fn isl_ast_expr_list_get_ast_expr(list: AstExprListRef, index: c_int) -> Option<AstExpr>;
  pub fn isl_ast_expr_list_set_ast_expr(list: AstExprList, index: c_int, el: AstExpr) -> Option<AstExprList>;
  pub fn isl_ast_expr_list_foreach(list: AstExprListRef, fn_: unsafe extern "C" fn(el: AstExpr, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_ast_expr_list_map(list: AstExprList, fn_: unsafe extern "C" fn(el: AstExpr, user: *mut c_void) -> Option<AstExpr>, user: *mut c_void) -> Option<AstExprList>;
  pub fn isl_ast_expr_list_sort(list: AstExprList, cmp: unsafe extern "C" fn(a: AstExprRef, b: AstExprRef, user: *mut c_void) -> c_int, user: *mut c_void) -> Option<AstExprList>;
  pub fn isl_ast_expr_list_foreach_scc(list: AstExprListRef, follows: unsafe extern "C" fn(a: AstExprRef, b: AstExprRef, user: *mut c_void) -> Bool, follows_user: *mut c_void, fn_: unsafe extern "C" fn(scc: AstExprList, user: *mut c_void) -> Stat, fn_user: *mut c_void) -> Stat;
  pub fn isl_printer_print_ast_expr_list(p: Printer, list: AstExprListRef) -> Option<Printer>;
  pub fn isl_ast_expr_list_dump(list: AstExprListRef) -> ();
  pub fn isl_ast_node_list_get_ctx(list: AstNodeListRef) -> Option<CtxRef>;
  pub fn isl_ast_node_list_from_ast_node(el: AstNode) -> Option<AstNodeList>;
  pub fn isl_ast_node_list_alloc(ctx: CtxRef, n: c_int) -> Option<AstNodeList>;
  pub fn isl_ast_node_list_copy(list: AstNodeListRef) -> Option<AstNodeList>;
  pub fn isl_ast_node_list_free(list: AstNodeList) -> *mut c_void;
  pub fn isl_ast_node_list_add(list: AstNodeList, el: AstNode) -> Option<AstNodeList>;
  pub fn isl_ast_node_list_insert(list: AstNodeList, pos: c_uint, el: AstNode) -> Option<AstNodeList>;
  pub fn isl_ast_node_list_drop(list: AstNodeList, first: c_uint, n: c_uint) -> Option<AstNodeList>;
  pub fn isl_ast_node_list_concat(list1: AstNodeList, list2: AstNodeList) -> Option<AstNodeList>;
  pub fn isl_ast_node_list_n_ast_node(list: AstNodeListRef) -> c_int;
  pub fn isl_ast_node_list_get_ast_node(list: AstNodeListRef, index: c_int) -> Option<AstNode>;
  pub fn isl_ast_node_list_set_ast_node(list: AstNodeList, index: c_int, el: AstNode) -> Option<AstNodeList>;
  pub fn isl_ast_node_list_foreach(list: AstNodeListRef, fn_: unsafe extern "C" fn(el: AstNode, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_ast_node_list_map(list: AstNodeList, fn_: unsafe extern "C" fn(el: AstNode, user: *mut c_void) -> Option<AstNode>, user: *mut c_void) -> Option<AstNodeList>;
  pub fn isl_ast_node_list_sort(list: AstNodeList, cmp: unsafe extern "C" fn(a: AstNodeRef, b: AstNodeRef, user: *mut c_void) -> c_int, user: *mut c_void) -> Option<AstNodeList>;
  pub fn isl_ast_node_list_foreach_scc(list: AstNodeListRef, follows: unsafe extern "C" fn(a: AstNodeRef, b: AstNodeRef, user: *mut c_void) -> Bool, follows_user: *mut c_void, fn_: unsafe extern "C" fn(scc: AstNodeList, user: *mut c_void) -> Stat, fn_user: *mut c_void) -> Stat;
  pub fn isl_printer_print_ast_node_list(p: Printer, list: AstNodeListRef) -> Option<Printer>;
  pub fn isl_ast_node_list_dump(list: AstNodeListRef) -> ();
}

#[repr(transparent)]
#[derive(Debug)]
pub struct AstExpr(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct AstExprRef(pub NonNull<c_void>);

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

impl std::ops::Deref for AstExpr {
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

impl std::ops::Deref for AstNode {
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

impl std::ops::Deref for AstPrintOptions {
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

impl std::ops::Deref for AstExprList {
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

impl std::ops::Deref for AstNodeList {
  type Target = AstNodeListRef;
  #[inline(always)]
  fn deref(&self) -> &AstNodeListRef { self.as_ref() }
}

impl To<Option<AstNodeList>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<AstNodeList> { NonNull::new(self).map(AstNodeList) }
}

impl AstExpr {
  #[inline(always)]
  pub fn list_from_ast_expr(self) -> Option<AstExprList> {
    unsafe {
      let ret = isl_ast_expr_list_from_ast_expr(self.to());
      (ret).to()
    }
  }
}

impl AstExprList {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_ast_expr_list_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, el: AstExpr) -> Option<AstExprList> {
    unsafe {
      let ret = isl_ast_expr_list_add(self.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert(self, pos: c_uint, el: AstExpr) -> Option<AstExprList> {
    unsafe {
      let ret = isl_ast_expr_list_insert(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, first: c_uint, n: c_uint) -> Option<AstExprList> {
    unsafe {
      let ret = isl_ast_expr_list_drop(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, list2: AstExprList) -> Option<AstExprList> {
    unsafe {
      let ret = isl_ast_expr_list_concat(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_ast_expr(self, index: c_int, el: AstExpr) -> Option<AstExprList> {
    unsafe {
      let ret = isl_ast_expr_list_set_ast_expr(self.to(), index.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map<F1: FnMut(AstExpr) -> Option<AstExpr>>(self, fn_: &mut F1) -> Option<AstExprList> {
    unsafe extern "C" fn fn1<F: FnMut(AstExpr) -> Option<AstExpr>>(el: AstExpr, user: *mut c_void) -> Option<AstExpr> { (*(user as *mut F))(el.to()).to() }
    unsafe {
      let ret = isl_ast_expr_list_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort<F1: FnMut(AstExprRef, AstExprRef) -> c_int>(self, cmp: &mut F1) -> Option<AstExprList> {
    unsafe extern "C" fn fn1<F: FnMut(AstExprRef, AstExprRef) -> c_int>(a: AstExprRef, b: AstExprRef, user: *mut c_void) -> c_int { (*(user as *mut F))(a.to(), b.to()).to() }
    unsafe {
      let ret = isl_ast_expr_list_sort(self.to(), fn1::<F1>, cmp as *mut _ as _);
      (ret).to()
    }
  }
}

impl AstExprListRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_ast_expr_list_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<AstExprList> {
    unsafe {
      let ret = isl_ast_expr_list_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_ast_expr(self) -> c_int {
    unsafe {
      let ret = isl_ast_expr_list_n_ast_expr(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ast_expr(self, index: c_int) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_list_get_ast_expr(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(AstExpr) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(AstExpr) -> Option<()>>(el: AstExpr, user: *mut c_void) -> Stat { (*(user as *mut F))(el.to()).to() }
    unsafe {
      let ret = isl_ast_expr_list_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_scc<F1: FnMut(AstExprRef, AstExprRef) -> Option<bool>, F2: FnMut(AstExprList) -> Option<()>>(self, follows: &mut F1, fn_: &mut F2) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(AstExprRef, AstExprRef) -> Option<bool>>(a: AstExprRef, b: AstExprRef, user: *mut c_void) -> Bool { (*(user as *mut F))(a.to(), b.to()).to() }
    unsafe extern "C" fn fn2<F: FnMut(AstExprList) -> Option<()>>(scc: AstExprList, user: *mut c_void) -> Stat { (*(user as *mut F))(scc.to()).to() }
    unsafe {
      let ret = isl_ast_expr_list_foreach_scc(self.to(), fn1::<F1>, follows as *mut _ as _, fn2::<F2>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_ast_expr_list_dump(self.to());
      (ret).to()
    }
  }
}

impl AstNode {
  #[inline(always)]
  pub fn list_from_ast_node(self) -> Option<AstNodeList> {
    unsafe {
      let ret = isl_ast_node_list_from_ast_node(self.to());
      (ret).to()
    }
  }
}

impl AstNodeList {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_ast_node_list_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, el: AstNode) -> Option<AstNodeList> {
    unsafe {
      let ret = isl_ast_node_list_add(self.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert(self, pos: c_uint, el: AstNode) -> Option<AstNodeList> {
    unsafe {
      let ret = isl_ast_node_list_insert(self.to(), pos.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop(self, first: c_uint, n: c_uint) -> Option<AstNodeList> {
    unsafe {
      let ret = isl_ast_node_list_drop(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, list2: AstNodeList) -> Option<AstNodeList> {
    unsafe {
      let ret = isl_ast_node_list_concat(self.to(), list2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_ast_node(self, index: c_int, el: AstNode) -> Option<AstNodeList> {
    unsafe {
      let ret = isl_ast_node_list_set_ast_node(self.to(), index.to(), el.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn map<F1: FnMut(AstNode) -> Option<AstNode>>(self, fn_: &mut F1) -> Option<AstNodeList> {
    unsafe extern "C" fn fn1<F: FnMut(AstNode) -> Option<AstNode>>(el: AstNode, user: *mut c_void) -> Option<AstNode> { (*(user as *mut F))(el.to()).to() }
    unsafe {
      let ret = isl_ast_node_list_map(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sort<F1: FnMut(AstNodeRef, AstNodeRef) -> c_int>(self, cmp: &mut F1) -> Option<AstNodeList> {
    unsafe extern "C" fn fn1<F: FnMut(AstNodeRef, AstNodeRef) -> c_int>(a: AstNodeRef, b: AstNodeRef, user: *mut c_void) -> c_int { (*(user as *mut F))(a.to(), b.to()).to() }
    unsafe {
      let ret = isl_ast_node_list_sort(self.to(), fn1::<F1>, cmp as *mut _ as _);
      (ret).to()
    }
  }
}

impl AstNodeListRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_ast_node_list_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<AstNodeList> {
    unsafe {
      let ret = isl_ast_node_list_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn n_ast_node(self) -> c_int {
    unsafe {
      let ret = isl_ast_node_list_n_ast_node(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ast_node(self, index: c_int) -> Option<AstNode> {
    unsafe {
      let ret = isl_ast_node_list_get_ast_node(self.to(), index.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach<F1: FnMut(AstNode) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(AstNode) -> Option<()>>(el: AstNode, user: *mut c_void) -> Stat { (*(user as *mut F))(el.to()).to() }
    unsafe {
      let ret = isl_ast_node_list_foreach(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_scc<F1: FnMut(AstNodeRef, AstNodeRef) -> Option<bool>, F2: FnMut(AstNodeList) -> Option<()>>(self, follows: &mut F1, fn_: &mut F2) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(AstNodeRef, AstNodeRef) -> Option<bool>>(a: AstNodeRef, b: AstNodeRef, user: *mut c_void) -> Bool { (*(user as *mut F))(a.to(), b.to()).to() }
    unsafe extern "C" fn fn2<F: FnMut(AstNodeList) -> Option<()>>(scc: AstNodeList, user: *mut c_void) -> Stat { (*(user as *mut F))(scc.to()).to() }
    unsafe {
      let ret = isl_ast_node_list_foreach_scc(self.to(), fn1::<F1>, follows as *mut _ as _, fn2::<F2>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_ast_node_list_dump(self.to());
      (ret).to()
    }
  }
}

impl CtxRef {
  #[inline(always)]
  pub fn ast_expr_list_alloc(self, n: c_int) -> Option<AstExprList> {
    unsafe {
      let ret = isl_ast_expr_list_alloc(self.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ast_node_list_alloc(self, n: c_int) -> Option<AstNodeList> {
    unsafe {
      let ret = isl_ast_node_list_alloc(self.to(), n.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_ast_expr_list(self, list: AstExprListRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_ast_expr_list(self.to(), list.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_ast_node_list(self, list: AstNodeListRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_ast_node_list(self.to(), list.to());
      (ret).to()
    }
  }
}

impl Drop for AstExprList {
  fn drop(&mut self) { AstExprList(self.0).free() }
}

impl Drop for AstNodeList {
  fn drop(&mut self) { AstNodeList(self.0).free() }
}

