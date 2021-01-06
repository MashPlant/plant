use crate::*;

extern "C" {
  pub fn isl_options_set_ast_iterator_type(ctx: CtxRef, val: Option<CStr>) -> Stat;
  pub fn isl_options_get_ast_iterator_type(ctx: CtxRef) -> Option<CStr>;
  pub fn isl_options_set_ast_always_print_block(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_ast_always_print_block(ctx: CtxRef) -> c_int;
  pub fn isl_ast_expr_from_val(v: Val) -> Option<AstExpr>;
  pub fn isl_ast_expr_from_id(id: Id) -> Option<AstExpr>;
  pub fn isl_ast_expr_neg(expr: AstExpr) -> Option<AstExpr>;
  pub fn isl_ast_expr_add(expr1: AstExpr, expr2: AstExpr) -> Option<AstExpr>;
  pub fn isl_ast_expr_sub(expr1: AstExpr, expr2: AstExpr) -> Option<AstExpr>;
  pub fn isl_ast_expr_mul(expr1: AstExpr, expr2: AstExpr) -> Option<AstExpr>;
  pub fn isl_ast_expr_div(expr1: AstExpr, expr2: AstExpr) -> Option<AstExpr>;
  pub fn isl_ast_expr_pdiv_q(expr1: AstExpr, expr2: AstExpr) -> Option<AstExpr>;
  pub fn isl_ast_expr_pdiv_r(expr1: AstExpr, expr2: AstExpr) -> Option<AstExpr>;
  pub fn isl_ast_expr_and(expr1: AstExpr, expr2: AstExpr) -> Option<AstExpr>;
  pub fn isl_ast_expr_and_then(expr1: AstExpr, expr2: AstExpr) -> Option<AstExpr>;
  pub fn isl_ast_expr_or(expr1: AstExpr, expr2: AstExpr) -> Option<AstExpr>;
  pub fn isl_ast_expr_or_else(expr1: AstExpr, expr2: AstExpr) -> Option<AstExpr>;
  pub fn isl_ast_expr_le(expr1: AstExpr, expr2: AstExpr) -> Option<AstExpr>;
  pub fn isl_ast_expr_lt(expr1: AstExpr, expr2: AstExpr) -> Option<AstExpr>;
  pub fn isl_ast_expr_ge(expr1: AstExpr, expr2: AstExpr) -> Option<AstExpr>;
  pub fn isl_ast_expr_gt(expr1: AstExpr, expr2: AstExpr) -> Option<AstExpr>;
  pub fn isl_ast_expr_eq(expr1: AstExpr, expr2: AstExpr) -> Option<AstExpr>;
  pub fn isl_ast_expr_access(array: AstExpr, indices: AstExprList) -> Option<AstExpr>;
  pub fn isl_ast_expr_call(function: AstExpr, arguments: AstExprList) -> Option<AstExpr>;
  pub fn isl_ast_expr_address_of(expr: AstExpr) -> Option<AstExpr>;
  pub fn isl_ast_expr_copy(expr: AstExprRef) -> Option<AstExpr>;
  pub fn isl_ast_expr_free(expr: AstExpr) -> *mut c_void;
  pub fn isl_ast_expr_get_ctx(expr: AstExprRef) -> Option<CtxRef>;
  pub fn isl_ast_expr_get_type(expr: AstExprRef) -> AstExprType;
  pub fn isl_ast_expr_get_val(expr: AstExprRef) -> Option<Val>;
  pub fn isl_ast_expr_get_id(expr: AstExprRef) -> Option<Id>;
  pub fn isl_ast_expr_get_op_type(expr: AstExprRef) -> AstOpType;
  pub fn isl_ast_expr_get_op_n_arg(expr: AstExprRef) -> c_int;
  pub fn isl_ast_expr_get_op_arg(expr: AstExprRef, pos: c_int) -> Option<AstExpr>;
  pub fn isl_ast_expr_set_op_arg(expr: AstExpr, pos: c_int, arg: AstExpr) -> Option<AstExpr>;
  pub fn isl_ast_expr_is_equal(expr1: AstExprRef, expr2: AstExprRef) -> Bool;
  pub fn isl_ast_expr_substitute_ids(expr: AstExpr, id2expr: IdToAstExpr) -> Option<AstExpr>;
  pub fn isl_printer_print_ast_expr(p: Printer, expr: AstExprRef) -> Option<Printer>;
  pub fn isl_ast_expr_dump(expr: AstExprRef) -> ();
  pub fn isl_ast_expr_to_str(expr: AstExprRef) -> Option<CString>;
  pub fn isl_ast_expr_to_C_str(expr: AstExprRef) -> Option<CString>;
  pub fn isl_ast_node_alloc_user(expr: AstExpr) -> Option<AstNode>;
  pub fn isl_ast_node_copy(node: AstNodeRef) -> Option<AstNode>;
  pub fn isl_ast_node_free(node: AstNode) -> *mut c_void;
  pub fn isl_ast_node_get_ctx(node: AstNodeRef) -> Option<CtxRef>;
  pub fn isl_ast_node_get_type(node: AstNodeRef) -> AstNodeType;
  pub fn isl_ast_node_set_annotation(node: AstNode, annotation: Id) -> Option<AstNode>;
  pub fn isl_ast_node_get_annotation(node: AstNodeRef) -> Option<Id>;
  pub fn isl_ast_node_for_get_iterator(node: AstNodeRef) -> Option<AstExpr>;
  pub fn isl_ast_node_for_get_init(node: AstNodeRef) -> Option<AstExpr>;
  pub fn isl_ast_node_for_get_cond(node: AstNodeRef) -> Option<AstExpr>;
  pub fn isl_ast_node_for_get_inc(node: AstNodeRef) -> Option<AstExpr>;
  pub fn isl_ast_node_for_get_body(node: AstNodeRef) -> Option<AstNode>;
  pub fn isl_ast_node_for_is_degenerate(node: AstNodeRef) -> Bool;
  pub fn isl_ast_node_if_get_cond(node: AstNodeRef) -> Option<AstExpr>;
  pub fn isl_ast_node_if_get_then(node: AstNodeRef) -> Option<AstNode>;
  pub fn isl_ast_node_if_has_else(node: AstNodeRef) -> Bool;
  pub fn isl_ast_node_if_get_else(node: AstNodeRef) -> Option<AstNode>;
  pub fn isl_ast_node_block_get_children(node: AstNodeRef) -> Option<AstNodeList>;
  pub fn isl_ast_node_mark_get_id(node: AstNodeRef) -> Option<Id>;
  pub fn isl_ast_node_mark_get_node(node: AstNodeRef) -> Option<AstNode>;
  pub fn isl_ast_node_user_get_expr(node: AstNodeRef) -> Option<AstExpr>;
  pub fn isl_ast_node_foreach_descendant_top_down(node: AstNodeRef, fn_: unsafe extern "C" fn(node: AstNodeRef, user: *mut c_void) -> Bool, user: *mut c_void) -> Stat;
  pub fn isl_printer_print_ast_node(p: Printer, node: AstNodeRef) -> Option<Printer>;
  pub fn isl_ast_node_dump(node: AstNodeRef) -> ();
  pub fn isl_ast_node_to_str(node: AstNodeRef) -> Option<CString>;
  pub fn isl_ast_print_options_alloc(ctx: CtxRef) -> Option<AstPrintOptions>;
  pub fn isl_ast_print_options_copy(options: AstPrintOptionsRef) -> Option<AstPrintOptions>;
  pub fn isl_ast_print_options_free(options: AstPrintOptions) -> *mut c_void;
  pub fn isl_ast_print_options_get_ctx(options: AstPrintOptionsRef) -> Option<CtxRef>;
  pub fn isl_ast_print_options_set_print_user(options: AstPrintOptions, print_user: unsafe extern "C" fn(p: Printer, options: AstPrintOptions, node: AstNodeRef, user: *mut c_void) -> Option<Printer>, user: *mut c_void) -> Option<AstPrintOptions>;
  pub fn isl_ast_print_options_set_print_for(options: AstPrintOptions, print_for: unsafe extern "C" fn(p: Printer, options: AstPrintOptions, node: AstNodeRef, user: *mut c_void) -> Option<Printer>, user: *mut c_void) -> Option<AstPrintOptions>;
  pub fn isl_options_set_ast_print_macro_once(ctx: CtxRef, val: c_int) -> Stat;
  pub fn isl_options_get_ast_print_macro_once(ctx: CtxRef) -> c_int;
  pub fn isl_ast_expr_foreach_ast_op_type(expr: AstExprRef, fn_: unsafe extern "C" fn(type_: AstOpType, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_ast_node_foreach_ast_op_type(node: AstNodeRef, fn_: unsafe extern "C" fn(type_: AstOpType, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_ast_op_type_set_print_name(p: Printer, type_: AstOpType, name: Option<CStr>) -> Option<Printer>;
  pub fn isl_ast_op_type_print_macro(type_: AstOpType, p: Printer) -> Option<Printer>;
  pub fn isl_ast_expr_print_macros(expr: AstExprRef, p: Printer) -> Option<Printer>;
  pub fn isl_ast_node_print_macros(node: AstNodeRef, p: Printer) -> Option<Printer>;
  pub fn isl_ast_node_print(node: AstNodeRef, p: Printer, options: AstPrintOptions) -> Option<Printer>;
  pub fn isl_ast_node_for_print(node: AstNodeRef, p: Printer, options: AstPrintOptions) -> Option<Printer>;
  pub fn isl_ast_node_if_print(node: AstNodeRef, p: Printer, options: AstPrintOptions) -> Option<Printer>;
  pub fn isl_ast_node_to_C_str(node: AstNodeRef) -> Option<CString>;
}

impl AstExpr {
  #[inline(always)]
  pub fn neg(self) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_neg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add(self, expr2: AstExpr) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_add(self.to(), expr2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn sub(self, expr2: AstExpr) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_sub(self.to(), expr2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mul(self, expr2: AstExpr) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_mul(self.to(), expr2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn div(self, expr2: AstExpr) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_div(self.to(), expr2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pdiv_q(self, expr2: AstExpr) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_pdiv_q(self.to(), expr2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn pdiv_r(self, expr2: AstExpr) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_pdiv_r(self.to(), expr2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn and(self, expr2: AstExpr) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_and(self.to(), expr2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn and_then(self, expr2: AstExpr) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_and_then(self.to(), expr2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn or(self, expr2: AstExpr) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_or(self.to(), expr2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn or_else(self, expr2: AstExpr) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_or_else(self.to(), expr2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn le(self, expr2: AstExpr) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_le(self.to(), expr2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn lt(self, expr2: AstExpr) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_lt(self.to(), expr2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ge(self, expr2: AstExpr) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_ge(self.to(), expr2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn gt(self, expr2: AstExpr) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_gt(self.to(), expr2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eq(self, expr2: AstExpr) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_eq(self.to(), expr2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn access(self, indices: AstExprList) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_access(self.to(), indices.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn call(self, arguments: AstExprList) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_call(self.to(), arguments.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn address_of(self) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_address_of(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_ast_expr_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_op_arg(self, pos: c_int, arg: AstExpr) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_set_op_arg(self.to(), pos.to(), arg.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn substitute_ids(self, id2expr: IdToAstExpr) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_substitute_ids(self.to(), id2expr.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ast_node_alloc_user(self) -> Option<AstNode> {
    unsafe {
      let ret = isl_ast_node_alloc_user(self.to());
      (ret).to()
    }
  }
}

impl AstExprRef {
  #[inline(always)]
  pub fn copy(self) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_ast_expr_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_type(self) -> AstExprType {
    unsafe {
      let ret = isl_ast_expr_get_type(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_val(self) -> Option<Val> {
    unsafe {
      let ret = isl_ast_expr_get_val(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_id(self) -> Option<Id> {
    unsafe {
      let ret = isl_ast_expr_get_id(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_op_type(self) -> AstOpType {
    unsafe {
      let ret = isl_ast_expr_get_op_type(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_op_n_arg(self) -> c_int {
    unsafe {
      let ret = isl_ast_expr_get_op_n_arg(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_op_arg(self, pos: c_int) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_get_op_arg(self.to(), pos.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_equal(self, expr2: AstExprRef) -> Option<bool> {
    unsafe {
      let ret = isl_ast_expr_is_equal(self.to(), expr2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_ast_expr_dump(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_ast_expr_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_C_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_ast_expr_to_C_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_ast_op_type<F1: FnMut(AstOpType) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(AstOpType) -> Option<()>>(type_: AstOpType, user: *mut c_void) -> Stat { (*(user as *mut F))(type_.to()).to() }
    unsafe {
      let ret = isl_ast_expr_foreach_ast_op_type(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_macros(self, p: Printer) -> Option<Printer> {
    unsafe {
      let ret = isl_ast_expr_print_macros(self.to(), p.to());
      (ret).to()
    }
  }
}

impl AstNode {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_ast_node_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_annotation(self, annotation: Id) -> Option<AstNode> {
    unsafe {
      let ret = isl_ast_node_set_annotation(self.to(), annotation.to());
      (ret).to()
    }
  }
}

impl AstNodeRef {
  #[inline(always)]
  pub fn copy(self) -> Option<AstNode> {
    unsafe {
      let ret = isl_ast_node_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_ast_node_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_type(self) -> AstNodeType {
    unsafe {
      let ret = isl_ast_node_get_type(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_annotation(self) -> Option<Id> {
    unsafe {
      let ret = isl_ast_node_get_annotation(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn for_get_iterator(self) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_node_for_get_iterator(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn for_get_init(self) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_node_for_get_init(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn for_get_cond(self) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_node_for_get_cond(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn for_get_inc(self) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_node_for_get_inc(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn for_get_body(self) -> Option<AstNode> {
    unsafe {
      let ret = isl_ast_node_for_get_body(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn for_is_degenerate(self) -> Option<bool> {
    unsafe {
      let ret = isl_ast_node_for_is_degenerate(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn if_get_cond(self) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_node_if_get_cond(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn if_get_then(self) -> Option<AstNode> {
    unsafe {
      let ret = isl_ast_node_if_get_then(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn if_has_else(self) -> Option<bool> {
    unsafe {
      let ret = isl_ast_node_if_has_else(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn if_get_else(self) -> Option<AstNode> {
    unsafe {
      let ret = isl_ast_node_if_get_else(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn block_get_children(self) -> Option<AstNodeList> {
    unsafe {
      let ret = isl_ast_node_block_get_children(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mark_get_id(self) -> Option<Id> {
    unsafe {
      let ret = isl_ast_node_mark_get_id(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mark_get_node(self) -> Option<AstNode> {
    unsafe {
      let ret = isl_ast_node_mark_get_node(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn user_get_expr(self) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_node_user_get_expr(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_descendant_top_down<F1: FnMut(AstNodeRef) -> Option<bool>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(AstNodeRef) -> Option<bool>>(node: AstNodeRef, user: *mut c_void) -> Bool { (*(user as *mut F))(node.to()).to() }
    unsafe {
      let ret = isl_ast_node_foreach_descendant_top_down(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_ast_node_dump(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_ast_node_to_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_ast_op_type<F1: FnMut(AstOpType) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(AstOpType) -> Option<()>>(type_: AstOpType, user: *mut c_void) -> Stat { (*(user as *mut F))(type_.to()).to() }
    unsafe {
      let ret = isl_ast_node_foreach_ast_op_type(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_macros(self, p: Printer) -> Option<Printer> {
    unsafe {
      let ret = isl_ast_node_print_macros(self.to(), p.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print(self, p: Printer, options: AstPrintOptions) -> Option<Printer> {
    unsafe {
      let ret = isl_ast_node_print(self.to(), p.to(), options.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn for_print(self, p: Printer, options: AstPrintOptions) -> Option<Printer> {
    unsafe {
      let ret = isl_ast_node_for_print(self.to(), p.to(), options.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn if_print(self, p: Printer, options: AstPrintOptions) -> Option<Printer> {
    unsafe {
      let ret = isl_ast_node_if_print(self.to(), p.to(), options.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn to_C_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_ast_node_to_C_str(self.to());
      (ret).to()
    }
  }
}

impl AstPrintOptions {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_ast_print_options_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_print_user<F1: FnMut(Printer, AstPrintOptions, AstNodeRef) -> Option<Printer>>(self, print_user: &mut F1) -> Option<AstPrintOptions> {
    unsafe extern "C" fn fn1<F: FnMut(Printer, AstPrintOptions, AstNodeRef) -> Option<Printer>>(p: Printer, options: AstPrintOptions, node: AstNodeRef, user: *mut c_void) -> Option<Printer> { (*(user as *mut F))(p.to(), options.to(), node.to()).to() }
    unsafe {
      let ret = isl_ast_print_options_set_print_user(self.to(), fn1::<F1>, print_user as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_print_for<F1: FnMut(Printer, AstPrintOptions, AstNodeRef) -> Option<Printer>>(self, print_for: &mut F1) -> Option<AstPrintOptions> {
    unsafe extern "C" fn fn1<F: FnMut(Printer, AstPrintOptions, AstNodeRef) -> Option<Printer>>(p: Printer, options: AstPrintOptions, node: AstNodeRef, user: *mut c_void) -> Option<Printer> { (*(user as *mut F))(p.to(), options.to(), node.to()).to() }
    unsafe {
      let ret = isl_ast_print_options_set_print_for(self.to(), fn1::<F1>, print_for as *mut _ as _);
      (ret).to()
    }
  }
}

impl AstPrintOptionsRef {
  #[inline(always)]
  pub fn copy(self) -> Option<AstPrintOptions> {
    unsafe {
      let ret = isl_ast_print_options_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_ast_print_options_get_ctx(self.to());
      (ret).to()
    }
  }
}

impl CtxRef {
  #[inline(always)]
  pub fn options_set_ast_iterator_type(self, val: Option<CStr>) -> Option<()> {
    unsafe {
      let ret = isl_options_set_ast_iterator_type(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_ast_iterator_type(self) -> Option<CStr> {
    unsafe {
      let ret = isl_options_get_ast_iterator_type(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_ast_always_print_block(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_ast_always_print_block(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_ast_always_print_block(self) -> c_int {
    unsafe {
      let ret = isl_options_get_ast_always_print_block(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ast_print_options_alloc(self) -> Option<AstPrintOptions> {
    unsafe {
      let ret = isl_ast_print_options_alloc(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_set_ast_print_macro_once(self, val: c_int) -> Option<()> {
    unsafe {
      let ret = isl_options_set_ast_print_macro_once(self.to(), val.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn options_get_ast_print_macro_once(self) -> c_int {
    unsafe {
      let ret = isl_options_get_ast_print_macro_once(self.to());
      (ret).to()
    }
  }
}

impl Id {
  #[inline(always)]
  pub fn ast_expr_from_id(self) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_from_id(self.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn print_ast_expr(self, expr: AstExprRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_ast_expr(self.to(), expr.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_ast_node(self, node: AstNodeRef) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_ast_node(self.to(), node.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn ast_op_type_set_print_name(self, type_: AstOpType, name: Option<CStr>) -> Option<Printer> {
    unsafe {
      let ret = isl_ast_op_type_set_print_name(self.to(), type_.to(), name.to());
      (ret).to()
    }
  }
}

impl Val {
  #[inline(always)]
  pub fn ast_expr_from_val(self) -> Option<AstExpr> {
    unsafe {
      let ret = isl_ast_expr_from_val(self.to());
      (ret).to()
    }
  }
}

impl Drop for AstExpr {
  fn drop(&mut self) { AstExpr(self.0).free() }
}

impl fmt::Display for AstExprRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for AstExpr {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl Drop for AstNode {
  fn drop(&mut self) { AstNode(self.0).free() }
}

impl fmt::Display for AstNodeRef {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { f.pad(&*self.to_str().unwrap()) }
}

impl fmt::Display for AstNode {
  #[inline(always)]
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", &**self) }
}

impl Drop for AstPrintOptions {
  fn drop(&mut self) { AstPrintOptions(self.0).free() }
}

