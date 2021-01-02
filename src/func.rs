use isl::{Ctx, BasicSet, BasicMap, DimType, AstNode, UnionMap};
use std::io;
use crate::*;

#[derive(Debug)]
pub struct Func {
  // 限制符号常量/参数取值范围
  pub func_ctx: Option<BasicSet>,
  pub name: Box<str>,
  pub comps: Vec<Box<Comp>>,
  pub bufs: Vec<Box<Buf>>,
  pub iter_ty: Type,
  // Ctx必须在所有引用Ctx的成员析构后析构
  pub ctx: Ctx,
}

impl Func {
  pub fn new(name: &str) -> Option<Box<Func>> {
    Some(box Func { func_ctx: None, name: name.into(), comps: Vec::new(), bufs: Vec::new(), iter_ty: I32, ctx: Ctx::new()? })
  }

  pub fn find_comp(&self, name: &str) -> Option<P<Comp>> {
    self.comps.iter().find(|c| c.name() == name).map(|c| P::new(&**c))
  }

  pub fn find_buf(&self, name: &str) -> Option<P<Buf>> {
    self.bufs.iter().find(|c| &*c.name == name).map(|c| P::new(&**c))
  }

  pub fn iter(&self, level: u32) -> Expr { Expr(self.iter_ty, Iter(level)) }

  pub fn add_ctx_constraint(&self, ctx: BasicSet) {
    if let Some(x) = &self.func_ctx {
      x.write(x.read().intersect(ctx).unwrap());
    } else { P::new(self).func_ctx = Some(ctx); }
  }

  pub fn align_schedule(&self) {
    let max_dim = self.comps.iter().map(|c| c.sch_dim()).max().unwrap_or(0);
    for c in &self.comps {
      c.schedule.write(align_dim(c.schedule.read(), max_dim).unwrap());
      debug!("aligned schedule: {}", c.schedule);
    }
  }

  pub fn codegen(&self, args: &[&Buf], path: &str) -> io::Result<()> {
    for b in args { assert_ne!(b.kind, BufKind::Temp); }
    self.align_schedule();
    let mut vis = HashSet::default();
    for c in self.sch_comps() {
      if c.pred.is_none() { self.sch_graph_dfs(P::new(c), &mut vis); }
    }
    let ast = self.build_isl_ast().unwrap();
    debug!("codegen: ast = {}", ast);
    Ok(())
  }
}

impl Func {
  // 返回参与调度的所有Comp。Comp::expr为None表示一个输入，不参与调度
  fn sch_comps(&self) -> impl IntoIterator<Item=&Comp> {
    self.comps.iter().map(|c| c.as_ref()).filter(
      |c| match c.expr { OptionExpr::Expr(_) => true, _ => false })
  }

  fn sch_graph_dfs(&self, c: P<Comp>, vis: &mut HashSet<P<Comp>>) {
    if !vis.insert(c) { panic!("schedule graph should be acyclic"); }
    for (&pred, &at) in &c.succ {
      pred.get().after_raw(c.get(), at);
      self.sch_graph_dfs(pred, vis);
    }
  }

  fn build_isl_ast(&self) -> Option<AstNode> {
    let mut union_sch = None::<UnionMap>;
    for c in self.sch_comps() {
      let sch_domain = c.domain.copy()?.apply(c.schedule.copy()?)?.set_tuple_name(c.name().into())?;
      // out dim名字必须是空的，ISL才会生成不完美嵌套的循环，否则只能生成多个完美嵌套的循环，`identity_map`保证这一点
      let sch = identity_map(&sch_domain)?.union_map_from_basic_map()?;
      union_sch = Some(if let Some(x) = union_sch { x.union(sch)? } else { sch });
    }
    let union_sch = union_sch?;
    debug!("build_isl_ast: union_sch = {}", union_sch);
    let mut build = if let Some(ctx) = self.func_ctx.as_ref() {
      ctx.copy()?.set_from_basic_set()?.ast_build_from_context()
    } else { self.ctx.ast_build_alloc() }?;
    build = build.set_at_each_domain(&mut move |node, _build| {
      Some(node)
    })?;
    self.ctx.options_set_ast_build_atomic_upper_bound(1)?;
    self.ctx.options_set_ast_build_exploit_nested_bounds(1)?;
    self.ctx.options_set_ast_build_group_coscheduled(1)?;
    build.ast_from_schedule(union_sch)
  }
}

fn align_dim(mut map: BasicMap, max_dim: u32) -> Option<BasicMap> {
  let orig_dim = map.dim(DimType::Out);
  map = map.add_dims(DimType::Out, max_dim - orig_dim)?;
  for i in orig_dim..max_dim { map = map_add_constraint(map, i, 0, 0)?; }
  Some(map)
}
