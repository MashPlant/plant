use isl::{Ctx, BasicSet, DimType, AstNode, UnionMap, AstBuildRef, Map, AstExpr, AstNodeType};
use std::{io::{self, Write, BufWriter}, fs::File, fmt::Display};
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

  // 将所有Comp的schedule的range维度统一成最大的，不足的维度补0
  // 并将domain和schedule的params都统一成全部params
  pub fn align_schedule(&self) { self.align_schedule_impl().unwrap() }

  // 这是一种我常用的模式，虽然返回Option类型，但发生的错误不会被处理，而是直接unwrap()
  // 只是为了避免写太多unwrap()，用一个返回Option的函数包一下，里面可以用?来处理
  fn align_schedule_impl(&self) -> Option<()> {
    let mut max_dim = 0;
    let mut all_params = self.comps.get(0)?.domain.get_space()?;
    for c in &self.comps {
      max_dim = max_dim.max(c.sch_dim());
      all_params = all_params.align_params(c.schedule.get_space()?)?;
    }
    for c in &self.comps {
      let mut sch = c.schedule.read();
      let orig_dim = sch.dim(DimType::Out);
      sch = sch.add_dims(DimType::Out, max_dim - orig_dim)?;
      for i in orig_dim..max_dim { sch = map_add_constraint(sch, i, 0, 0)?; }
      sch = sch.align_params(all_params.copy()?)?;
      c.schedule.write(sch);
      c.domain.write(c.domain.read().align_params(all_params.copy()?)?);
      debug!("aligned schedule: {}; domain: {}", c.schedule, c.domain);
    }
    Some(())
  }

  pub fn codegen(&self, args: &[&Buf], path: &str) -> io::Result<()> {
    for b in args { assert_ne!(b.kind, BufKind::Temp); }
    self.align_schedule();
    let mut vis = HashSet::default();
    for c in self.sch_comps() {
      if c.pred.is_none() { self.sch_graph_dfs(P::new(c), &mut vis); }
    }
    let ast = self.build_isl_ast().unwrap(); // todo: 可以从这个ast中提取特征，无需自己维护ast了
    let mut w = BufWriter::new(File::create(path)?);
    debug!("codegen: ast = {}", ast);
    write!(w, "{}", self.gen(ast))?;
    Ok(())
  }
}

impl Func {
  // 返回参与调度的所有Comp。Comp::expr为None表示一个输入，不参与调度
  fn sch_comps(&self) -> impl IntoIterator<Item=&Comp> {
    self.comps.iter().map(|c| c.as_ref())
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
    let n_dim = self.sch_comps().into_iter().next()?.sch_dim();
    debug_assert_eq!(n_dim % 2, 1); // 一定是 static, dynamic, ..., static的模式
    let mut iters = self.ctx.id_list_alloc(n_dim as _)?;
    for i in 0..n_dim / 2 {
      // static dim名字是_i{i}，生成的代码中不会用到，dynamic dim名字是i{i}
      iters = iters.add(self.ctx.id_alloc(format!("_i{}\0", i).as_str().into(), 0 as _)?)?
        .add(self.ctx.id_alloc(format!("i{}\0", i).as_str().into(), 0 as _)?)?;
    }
    // 最后一个static dim没有设置名字，这没有影响，因为所有static dim的名字都没用
    build = build.set_iterators(iters)?
      .set_at_each_domain(&mut move |node, build| self.modify_comp(node, build))?;
    self.ctx.options_set_ast_build_atomic_upper_bound(1)?;
    self.ctx.options_set_ast_build_exploit_nested_bounds(1)?;
    self.ctx.options_set_ast_build_group_coscheduled(1)?;
    build.ast_from_schedule(union_sch)
  }

  // 将node表示的comp的expr中的原下标替换成新下标，对comp的access替换成对buf的load
  // 将store的位置也用load保存在store_expr中(虽然实际不是load，但下标表示是一样的)
  fn modify_comp(&self, mut node: AstNode, build: AstBuildRef) -> Option<AstNode> {
    let expr = node.user_get_expr()?;
    let name = expr.get_op_arg(0)?.get_id()?.get_name()?;
    let mut comp = self.find_comp(&name)?;
    node = node.set_annotation(self.ctx.id_alloc("\0".into(), comp.0.as_ptr() as _)?)?;
    if let Some(store) = comp.store.as_ref() {
      let access = store.copy()?.apply_domain(comp.schedule.copy()?)?
        .set_tuple_name(DimType::In, comp.name().into())?.map_from_basic_map()?;
      let idx = comp_access(build, access)?;
      comp.store_expr = Some(Expr::from_isl(self, idx)?);
    }
    // 创建一个在新domain中访问原domain的下标的expr，从而得到每个原下标用新下标的表示形式
    let access_self = identity_map(&comp.domain)?
      .apply_domain(comp.schedule.copy()?)?
      .set_tuple_name(DimType::In, comp.name().into())?.map_from_basic_map()?;
    let access_self = comp_access(build, access_self)?;
    let n = access_self.get_op_n_arg();
    let mut iter_map = Vec::with_capacity(n as usize - 1);
    for i in 1..n {
      iter_map.push(Expr::from_isl(self, access_self.get_op_arg(i)?)?);
    }
    debug!("modify_comp: iter_map = [{}]", comma_sep(iter_map.iter()));
    // comp.clone()只是指针拷贝，用于规避借用检查，lambda中也使用comp
    comp.clone().expr.visit_mut(&mut move |e| match &e.1 {
      // access_to_load已经将原下标替换成了新下标，不能再访问它的孩子再替换一次了
      Access(arg, idx) => {
        *e = access_to_load(build, &comp, arg, idx).unwrap();
        debug!("modify_comp: replaced access = {}", e);
        false
      }
      &Iter(x) => {
        *e = iter_map[x as usize].clone();
        false
      }
      _ => true
    });
    Some(node)
  }

  fn gen<'a>(&'a self, node: AstNode) -> impl Display + 'a {
    use std::fmt::Error as E;
    fn2display(move |f| {
      match node.get_type() {
        AstNodeType::For => {
          let it = node.for_get_iterator().ok_or(E)?.get_id().ok_or(E)?.get_name().ok_or(E)?.as_str();
          let init = node.for_get_init().ok_or(E)?.to_C_str().ok_or(E)?;
          let cond = node.for_get_cond().ok_or(E)?.to_C_str().ok_or(E)?;
          let inc = node.for_get_inc().ok_or(E)?.to_C_str().ok_or(E)?;
          let body = node.for_get_body().ok_or(E)?;
          write!(f, "for({} {it}={};{};{it}+={}){{{}}}", self.iter_ty, init, cond, inc, self.gen(body), it = it)?;
        }
        AstNodeType::If => {
          let cond = node.if_get_cond().ok_or(E)?.to_C_str().ok_or(E)?;
          let t = node.if_get_then().ok_or(E)?;
          write!(f, "if({}){{{}}}", cond, self.gen(t))?;
          if let Some(e) = node.if_get_else() {
            write!(f, "else{{{}}}", self.gen(e))?;
          }
        }
        AstNodeType::Block => {
          let ch = node.block_get_children().ok_or(E)?;
          let n = ch.n_ast_node();
          f.write_str("{")?;
          for i in 0..n {
            let ch = ch.get_ast_node(i).ok_or(E)?;
            write!(f, "{}", self.gen(ch))?;
          }
          f.write_str("}")?;
        }
        AstNodeType::User => {
          let comp = P::new(node.get_annotation().ok_or(E)?.get_user() as *const Comp);
          write!(f, "{}", comp.expr)?;
        }
        _ => panic!("invalid ast node type"),
      }
      Ok(())
    })
  }
}

fn access_to_load(build: AstBuildRef, comp: &Comp, arg: &Comp, idx: &[Expr]) -> Option<Expr> {
  let params = comma_sep((0..comp.domain.dim(DimType::Param)).map(|i| comp.domain.get_dim_name(DimType::Param, i).unwrap()));
  let s = format!("[{}] -> {{ {}{} -> {}[{}] }}\0", params,
    comp.name(), i0_in(comp.n_dim()), arg.name(), comma_sep(idx.iter()));
  debug!("access_to_load: {}", s);
  let store = arg.store.as_ref()?.copy()?; // todo: 处理arg没有store的情形
  let access = comp.ctx.basic_map_read_from_str(s.as_str().into())?
    .apply_range(store)?
    .apply_domain(comp.schedule.copy()?)?
    .set_tuple_name(DimType::In, comp.name().into())?.map_from_basic_map()?;
  debug!("access_to_load: access = {}", access);
  let expr = comp_access(build, access)?;
  Expr::from_isl(&comp.func, expr)
}

fn comp_access(build: AstBuildRef, access: Map) -> Option<AstExpr> {
  let sch = build.get_schedule()?.map_from_union_map()?;
  let map = sch.reverse()?;
  let mut iter_map = map.pw_multi_aff_from_map()?;
  let index_aff = access.pw_multi_aff_from_map()?;
  iter_map = index_aff.pullback_pw_multi_aff(iter_map)?;
  build.access_from_pw_multi_aff(iter_map)
}
