use std::{io::{self, Write, BufWriter}, fs::File};
use crate::*;

#[derive(Debug)]
pub struct Func {
  // 限制符号常量/参数取值范围
  pub func_ctx: Option<BasicSet>,
  pub name: Box<str>,
  pub comps: Vec<Box<Comp>>,
  pub bufs: Vec<Box<Buf>>,
  pub iter_ty: Type,
  // 用于命名自动生成的Comp
  pub comp_cnt: u32,
  // Ctx必须在所有引用Ctx的成员析构后析构
  pub ctx: Ctx,
}

impl Func {
  pub fn new(name: &str) -> Box<Func> {
    box Func { func_ctx: None, name: name.into(), comps: Vec::new(), bufs: Vec::new(), iter_ty: I32, comp_cnt: 0, ctx: Ctx::new() }
  }

  pub fn find_comp(&self, name: &str) -> Option<P<Comp>> {
    self.comps.iter().find(|c| c.name() == name).map(|c| P::new(&**c))
  }

  pub fn find_buf(&self, name: &str) -> Option<P<Buf>> {
    self.bufs.iter().find(|c| &*c.name == name).map(|c| P::new(&**c))
  }

  pub fn iter(&self, level: u32) -> Expr { Iter(self.iter_ty, level) }

  pub(crate) fn new_comp_name(&self) -> String {
    format!("_C{}\0", (self.comp_cnt, P::new(self).comp_cnt += 1).0)
  }

  // 设置domain/schedule中的params的取值范围
  pub fn set_constraint(&self, csts: &[Expr]) -> Unit {
    self.align_schedule();
    let s = format!("{} -> {{: {}}}\0", self.comps.first().expect("no comp").params(), sep(csts.iter(), " and "));
    debug!("set_constraint: {}", s);
    P::new(self).func_ctx = Some(self.ctx.basic_set_read_from_str(cstr(&s))?);
    Unit
  }

  // 将所有Comp的schedule的range维度统一成最大的，不足的维度补0
  // 并将domain和schedule的params都统一成全部params
  pub fn align_schedule(&self) -> Unit {
    let mut max_dim = 0;
    let mut all_params = self.comps.get(0).expect("no comp").domain.get_space()?;
    for c in &self.comps {
      max_dim = max_dim.max(c.sch_dim());
      all_params = all_params.align_params(c.schedule.get_space()?)?;
    }
    for c in &self.comps {
      let mut sch = c.schedule.read();
      let orig_dim = sch.dim(DimType::Out);
      sch = sch.add_dims(DimType::Out, max_dim - orig_dim)?;
      for i in orig_dim..max_dim { sch = map_add_constraint(sch, i, 0, 0); }
      sch = sch.align_params(all_params.copy()?)?;
      c.schedule.write(sch);
      c.domain.write(c.domain.read().align_params(all_params.copy()?)?);
      debug!("aligned schedule: {}; domain: {}", c.schedule, c.domain);
      if let Some(store) = &c.store {
        store.write(store.read().align_params(all_params.copy()?)?);
        debug!("aligned store: {}", store);
      }
    }
    Unit
  }

  pub fn codegen(&self, args: &[&Buf], path: &str) -> io::Result<()> {
    for b in args { assert_ne!(b.kind, BufKind::Temp); }
    self.align_schedule();
    let mut vis = HashSet::default();
    let mut prev = None;
    for c in &self.comps {
      if c.pred.is_none() {
        // 所有没有前驱的节点按定义顺序排序
        // after_raw要求按照从前往后顺序调用，例如B after_raw C, A after_raw B，不能是A after_raw B, B after_raw C
        // sch_graph_dfs保证这一点(pre-order dfs)，并且需要在sch_graph_dfs之前确定c after_raw 谁
        if let Some(x) = prev { c.after_raw(x, 0); }
        prev = Some(c);
        self.sch_graph_dfs(P::new(c.as_ref()), &mut vis);
      }
    }
    let mut info = Vec::new();
    let ast = self.build_isl_ast(&mut info); // todo: 可以从这个ast中提取特征，无需自己维护ast了
    let mut tags = HashMap::default();
    extract_tags(*ast, &mut Vec::new(), &mut tags);
    let mut w = BufWriter::new(File::create(path)?);
    debug!("codegen: ast = {}", ast);
    w.write_all(include_bytes!("inc.h"))?;
    write!(w, "void {}({}){{{}}}", self.name,
      comma_sep(args.iter().map(|&x| fn2display(move |f| write!(f, "{} *restrict {}", x.ty, x.name)))),
      self.gen(ast, &tags))?;
    Ok(())
  }
}

// codegen期间生成的Comp信息
// set_at_each_domain的函数可能多次接受表示同一个Comp的节点，这些节点会在生成的代码中重复多次
// 同一个Comp在代码的不同位置中可能会有不同的iter_map(见visit_comp)，所以不能直接修改Comp，而是保存到CompInfo中
struct CompInfo {
  // 将Comp::schedule中的每个循环层次(sch_dims() / 2个)映射到ISL AST中的for层次
  // 正常情况下dim_map为[Some(0), Some(1), ...]，但也可能不是，有些循环层次不会映射到for
  dim_map: Vec<Option<u32>>,
  // store的目标位置，以Load表示
  store: Option<Expr>,
  expr: Expr,
  comp: P<Comp>,
}

impl Func {
  fn sch_graph_dfs(&self, c: P<Comp>, vis: &mut HashSet<P<Comp>>) {
    if !vis.insert(c) { panic!("cyclic schedule graph"); }
    for (&s, &at) in &c.succ {
      s.get().after_raw(c.get(), at);
      self.sch_graph_dfs(s, vis);
    }
  }

  // info不能是本函数的局部变量，否则会在本函数中析构，而codegen中仍需使用其中的数据
  // 若真的是局部变量，set_at_each_domain(&mut move ...)和&mut ...的区别在于前者info在这句结束时析构，后者本函数结束时析构
  // 前一种可能不会引发错误，因为那时info中还没有元素，析构是空操作，之后还可以向里面push，后push的元素不会析构
  // 但这也是错误的，首先会内存泄露，其次本函数结束后访问info是访问无效的栈内存
  fn build_isl_ast(&self, info: &mut Vec<Box<CompInfo>>) -> AstNode {
    let mut union_sch = None;
    for c in &self.comps {
      let sch_domain = c.domain.copy()?.apply(c.schedule.copy()?)?.set_tuple_name(cstr(c.name()))?;
      // out dim名字必须是空的，ISL才会生成不完美嵌套的循环，否则只能生成多个完美嵌套的循环，`identity_map`保证这一点
      let sch = identity_map(&sch_domain).union_map_from_map()?;
      union_sch = Some(if let Some(x) = union_sch { sch.union(x)? } else { sch });
    }
    let union_sch = union_sch?;
    debug!("build_isl_ast: union_sch = {}", union_sch);
    let mut build = if let Some(ctx) = self.func_ctx.as_ref() {
      ctx.copy()?.set_from_basic_set()?.ast_build_from_context()
    } else { self.ctx.ast_build_alloc() }?;
    let n_dim = self.comps.first().expect("no comp").sch_dim();
    debug_assert_eq!(n_dim % 2, 1); // 一定是 static, dynamic, ..., static的模式
    let mut iters = self.ctx.id_list_alloc(n_dim as _)?;
    for i in 0..n_dim / 2 {
      // static dim名字是_i{i}，生成的代码中不会用到，dynamic dim名字是i{i}
      iters = iters.add(self.ctx.id_alloc(cstr(&format!("_i{}\0", i)), 0 as _)?)?
        .add(self.ctx.id_alloc(cstr(&format!("i{}\0", i)), 0 as _)?)?;
    }
    // 最后一个static dim没有设置名字，这没有影响，因为所有static dim的名字都没用
    build = build.set_iterators(iters)?
      .set_at_each_domain(&mut |node, build| self.visit_comp(node, build, info).into())?;
    self.ctx.options_set_ast_build_atomic_upper_bound(1)?;
    self.ctx.options_set_ast_build_exploit_nested_bounds(1)?;
    self.ctx.options_set_ast_build_group_coscheduled(1)?;
    build.ast_from_schedule(union_sch)?
  }

  // 将node表示的comp的expr中的原下标替换成新下标，对comp的access替换成对buf的load
  // 将store的位置也用load保存在store_expr中(虽然实际不是load，但下标表示是一样的)
  fn visit_comp(&self, node: AstNode, build: AstBuildRef, info: &mut Vec<Box<CompInfo>>) -> AstNode {
    let expr = node.user_get_expr()?;
    let name = expr.get_op_arg(0)?.get_id()?.get_name()?;
    let comp = self.find_comp(&name)?;
    let mut dim_map = vec![None; comp.sch_dim() as usize / 2];
    // get_schedule_space返回的space形如{ [i0, ix, ...] }，每维的名字来源于build.set_iterators
    // 每维对应AST中的一个for层次，例如[i0, i2]是for (i0) { for(i2) {} }，得到dim_map是[Some(0), None, Some(1)]
    let sp = build.get_schedule_space()?;
    for i in 0..sp.dim(DimType::Out) {
      let it = sp.get_dim_name(DimType::Out, i)?.as_str();
      assert!(it.starts_with("i"));
      dim_map[it.get(1..)?.parse::<usize>().ok()?] = Some(i);
    }
    let store = if let Some(store) = comp.store.as_ref() {
      let access = store.copy()?.apply_domain(comp.schedule.copy()?)?
        .set_tuple_name(DimType::In, comp.name_cstr())?;
      Some(Expr::from_isl(self, comp_access(build, access)))
    } else { None };
    // 创建一个在新domain中访问原domain的下标的expr，从而得到每个原下标用新下标的表示形式
    let access_self = comp_access(build, identity_map(&comp.domain)
      .apply_domain(comp.schedule.copy()?)?
      .set_tuple_name(DimType::In, comp.name_cstr())?);
    let n = access_self.get_op_n_arg();
    let mut iter_map = Vec::with_capacity(n as usize - 1);
    for i in 1..n {
      iter_map.push(Expr::from_isl(self, access_self.get_op_arg(i)?)?);
    }
    debug!("visit_comp: comp = {}, dim_map = {:?}, iter_map = [{}]", comp.name(), dim_map, comma_sep(iter_map.iter()));
    let mut expr = comp.expr.clone();
    expr.visit_mut(&mut move |e| match e {
      // access_to_load已经将原下标替换成了新下标，不能再访问它的孩子再替换一次了
      Access(arg, idx) => {
        *e = access_to_load(build, &comp, arg, idx);
        debug!("modify_comp: replaced access = {}", e);
        false
      }
      Iter(_, x) => {
        *e = iter_map[*x as usize].clone();
        false
      }
      _ => true
    });
    let ci = box CompInfo { dim_map, store, expr, comp };
    let node = node.set_annotation(self.ctx.id_alloc(None, ci.as_ref() as *const _ as _)?)?;
    info.push(ci);
    node
  }

  fn gen<'a>(&'a self, node: AstNode, tags: &'a HashMap<AstNodeRef, DimTag>) -> impl Display + 'a {
    use std::fmt::Error as E;
    fn2display(move |f| {
      match node.get_type() {
        AstNodeType::For => {
          let it = node.for_get_iterator().ok_or(E)?.get_id().ok_or(E)?.get_name().ok_or(E)?.as_str();
          let init = node.for_get_init().ok_or(E)?.to_C_str().ok_or(E)?;
          let cond = node.for_get_cond().ok_or(E)?.to_C_str().ok_or(E)?;
          let inc = node.for_get_inc().ok_or(E)?.to_C_str().ok_or(E)?;
          let body = node.for_get_body().ok_or(E)?;
          match tags.get(&*node) {
            // todo: 支持更复杂的parallel模式，比如#pragma omp parallel，每个线程自己同步
            Some(Parallel) => f.write_str("\n#pragma omp parallel for\n")?,
            _ => {}
          }
          write!(f, "for({} {it}={};{};{it}+={}){{{}}}", self.iter_ty, init, cond, inc, self.gen(body, tags), it = it)?;
        }
        AstNodeType::If => {
          let cond = node.if_get_cond().ok_or(E)?.to_C_str().ok_or(E)?;
          let t = node.if_get_then().ok_or(E)?;
          write!(f, "if({}){{{}}}", cond, self.gen(t, tags))?;
          if let Some(e) = node.if_get_else() {
            write!(f, "else{{{}}}", self.gen(e, tags))?;
          }
        }
        AstNodeType::Block => {
          // block node不需要{}包裹，因为if，for已经有{}了，而且用{}包裹会让一些局部变量无法访问
          let ch = node.block_get_children().ok_or(E)?;
          for i in 0..ch.n_ast_node() {
            let ch = ch.get_ast_node(i).ok_or(E)?;
            write!(f, "{}", self.gen(ch, tags))?;
          }
        }
        AstNodeType::User => {
          let comp = P::new(node.get_annotation().ok_or(E)?.get_user() as *const CompInfo);
          if let Some(store) = &comp.store {
            write!(f, "{}={};", store, comp.expr)?;
          } else {
            // 没有store的comp表示成一个标量定义
            write!(f, "{} {}={};", comp.expr.ty(), comp.comp.name(), comp.expr)?;
          }
        }
        _ => panic!("invalid ast node type"),
      }
      Ok(())
    })
  }
}

// 从user node中提取Comp::dim_tags中循环的tag，用HashMap保存for node对应的循环的tag
fn extract_tags(node: AstNodeRef, levels: &mut Vec<AstNodeRef>, tags: &mut HashMap<AstNodeRef, DimTag>) -> Unit {
  match node.get_type() {
    AstNodeType::For => {
      levels.push(node);
      extract_tags(*node.for_get_body()?, levels, tags)?;
      levels.pop();
    }
    AstNodeType::If => {
      extract_tags(*node.if_get_then()?, levels, tags)?;
      if let Some(e) = node.if_get_else() { extract_tags(*e, levels, tags)?; }
    }
    AstNodeType::Block => {
      let ch = node.block_get_children()?;
      for i in 0..ch.n_ast_node() { extract_tags(*ch.get_ast_node(i)?, levels, tags)?; }
    }
    AstNodeType::User => {
      let comp = P::new(node.get_annotation()?.get_user() as *const CompInfo);
      for (i, &tag) in comp.comp.dim_tags.iter().enumerate() {
        if let (Some(i), Some(tag)) = (comp.dim_map[i], tag) {
          tags.insert(levels[i as usize], tag);
        }
      }
    }
    _ => panic!("invalid ast node type"),
  }
  Unit
}

fn access_to_load(build: AstBuildRef, comp: &Comp, arg: &Comp, idx: &[Expr]) -> Expr {
  let s = format!("{} -> {{ {}{} -> {}[{}] }}\0", comp.params(),
    comp.name(), i0_in(comp.n_dim()), arg.name(), comma_sep(idx.iter()));
  debug!("access_to_load: {}", s);
  // 对于没有store的arg返回Param表达式，这不会影响到domain/schedule的参数，这个表达式之后只会用于输出
  let store = if let Some(x) = arg.store.as_ref() { x.copy()? } else { return arg.as_param(); };
  let access = comp.ctx.map_read_from_str(cstr(&s))
    .expect("failed to read access map, comp may have non-affine access")
    .apply_range(store)?.apply_domain(comp.schedule.copy()?)?
    .set_tuple_name(DimType::In, comp.name_cstr())?;
  debug!("access_to_load: access = {}", access);
  Expr::from_isl(&comp.func, comp_access(build, access))
}

fn comp_access(build: AstBuildRef, access: Map) -> AstExpr {
  let sch = build.get_schedule()?.map_from_union_map()?;
  let map = sch.reverse()?;
  let mut iter_map = map.pw_multi_aff_from_map()?;
  let index_aff = access.pw_multi_aff_from_map()?;
  iter_map = index_aff.pullback_pw_multi_aff(iter_map)?;
  build.access_from_pw_multi_aff(iter_map)?
}
