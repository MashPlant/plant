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
  // 用于命名自动生成的Buf
  pub buf_cnt: u32,
  // Ctx必须在所有引用Ctx的成员析构后析构
  pub ctx: Ctx,
}

impl Func {
  pub fn new(name: &str) -> Box<Func> {
    box Func { func_ctx: None, name: name.into(), comps: Vec::new(), bufs: Vec::new(), iter_ty: I32, comp_cnt: 0, buf_cnt: 0, ctx: Ctx::new() }
  }

  pub fn find_comp(&self, name: &str) -> Option<&Comp> {
    self.comps.iter().find(|c| c.name() == name).map(|c| c.as_ref())
  }

  pub fn find_buf(&self, name: &str) -> Option<&Buf> {
    self.bufs.iter().find(|c| &*c.name == name).map(|c| c.as_ref())
  }

  pub fn iter(&self, level: u32) -> Expr { Iter(self.iter_ty, level) }

  pub fn new_comp_id(&self) -> u32 { (self.comp_cnt, self.p().comp_cnt += 1).0 }

  pub fn auto_comp_name(&self, e: &Expr) -> String {
    // 这个名字没什么意义，只是为了人阅读方便
    let desc = match e {
      Val(..) => "val", Iter(..) => "iter", Param(..) => "param", Cast(..) => "cast", Unary(..) => "unary",
      Binary(..) => "binary", Call(..) => "call", Access(..) => "access", Load(..) => "load",
      Memcpy(..) => "memcpy", Alloc(..) => "alloc", Free(..) => "free"
    };
    format!("_{}{}", desc, self.new_comp_id())
  }

  pub fn new_buf_id(&self) -> u32 { (self.buf_cnt, self.p().buf_cnt += 1).0 }

  // 设置domain/schedule中的params的取值范围
  pub fn set_constraint(&self, csts: &[Expr]) -> Unit {
    self.align_schedule();
    let s = format!("{} -> {{: {}}}\0", self.comps.first().expect("no comp").params(), sep(csts.iter(), " and "));
    debug!("set_constraint: {}", s);
    self.p().func_ctx = Some(self.ctx.basic_set_read_from_str(cstr(&s))?);
    Unit
  }

  // 将所有Comp的schedule的range维度统一成最大的，不足的维度补0
  // 并将domain和schedule的params都统一成全部params
  pub fn align_schedule(&self) -> Unit {
    let mut max_dim = 0;
    let mut all_params = self.comps.get(0).expect("no comp").schedule.get_space()?;
    for c in &self.comps {
      max_dim = max_dim.max(c.sch_dim());
      all_params = all_params.align_params(c.schedule.get_space()?)?;
    }
    for c in &self.comps {
      let mut sch = c.schedule.read();
      let orig_dim = sch.dim(DimType::Out) as u32;
      sch = sch.add_dims(DimType::Out, max_dim - orig_dim)?;
      for i in orig_dim..max_dim { sch = map_add_constraint(sch, i, 0, 0); }
      sch = sch.align_params(all_params.copy()?)?;
      c.schedule.write(sch);
      debug!("aligned schedule: {}", c.schedule);
      if let Some(store) = &c.store {
        store.write(store.read().align_params(all_params.copy()?)?);
        debug!("aligned store: {}", store);
      }
    }
    Unit
  }

  pub fn codegen(&self, args: &[&Buf], path: &str) -> io::Result<()> {
    for b in args { debug_assert_ne!(b.kind, BufKind::Temp); }
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
        self.sch_graph_dfs(c.as_ref().p(), &mut vis);
      }
    }
    let mut info = Vec::new();
    let ast = self.build_isl_ast(&mut info); // todo: 可以从这个ast中提取特征，无需自己维护ast了
    let mut s = CodegenState::default();
    // 实现上必须访问两次才能提取used_buf和local_buf信息，extract_tags只给for加tag
    // 有了tag后，extract_buf才能提取信息并保存到第一个进入GPU的ForInfo中
    extract_tags(*ast, &mut Vec::new(), &mut s.info);
    extract_buf(self, *ast, &mut s);
    let mut w = BufWriter::new(File::create(path)?);
    debug!("codegen: ast = {}", ast);
    w.write_all(include_bytes!("inc.h"))?;
    write!(w, "void {}({}){{{}}}\n", self.name,
      comma_sep(args.iter().map(|&x| fn2display(move |f| write!(f, "{}*__restrict__ {}", x.ty, x.name)))),
      self.gen(ast, &s))?;
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

struct ForInfo {
  // 给这个for加上tag的Comp，有且仅有一个
  comp: P<Comp>,
  // 这个for对应schedule中的哪个dynamic dim
  // 一个for可能出现在多个Comp中，但在不同的Comp中它必须拥有相同的dynamic dim
  level: u32,
  tag: DimTag,
  // used_buf和local_buf用于GPU代码生成，分别表示kern中使用的所有Buf和kern中自己分配的Buf
  // 如果一个Buf被使用，但不是自己分配的，就作为参数传给kern
  // 代码生成中使用了extended lambda来生成kern，可以自动捕获使用的标量，所以不收集它们
  // 其实也可以自动捕获使用的指针，但经实验这样会丢失restrict信息，导致kern效率降低，所以还是手动收集它们
  used_buf: HashSet<P<Buf>>,
  local_buf: HashSet<P<Buf>>,
}

#[derive(Default)]
struct CodegenState {
  // tags保存AST中的for循环的信息，AstNodeRef的具体类型一定是for节点，有且仅有一个Comp在这个for上添加了tag
  info: HashMap<AstNodeRef, ForInfo>,
  in_kern: bool,
  // used_buf和local_buf与ForInfo中的意义相同，是借用这里保存一下，填好后放进第一个进入GPU的for的ForInfo中
  used_buf: HashSet<P<Buf>>,
  local_buf: HashSet<P<Buf>>,
  // kern的启动参数，6对应<<<dim3(...), dim3(...)>>>中的6个参数，如果是None就填1
  kern_cfg: [Option<Box<str>>; 6],
}

impl Func {
  fn sch_graph_dfs(&self, c: P<Comp>, vis: &mut HashSet<P<Comp>>) {
    if !vis.insert(c) { debug_panic!("cyclic schedule graph"); }
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
      // out dim名字必须是空的，ISL才会生成不完美嵌套的循环，否则只能生成多个完美嵌套的循环，`identity_map`保证这一点
      let sch = identity_map(c.schedule()).union_map_from_map()?;
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
    for i in 0..sp.dim(DimType::Out) as u32 {
      let it = sp.get_dim_name(DimType::Out, i)?.as_str();
      debug_assert!(it.starts_with("i"));
      dim_map[it.get(1..)?.parse::<usize>().ok()?] = Some(i);
    }
    let store = if let Some(store) = comp.store.as_ref() {
      let access = store.copy()?.apply_domain(comp.schedule.copy()?)?
        .set_tuple_name(DimType::In, comp.name_cstr())?;
      Some(Expr::from_isl(self, comp_access(build, access)))
    } else { None };
    // 创建一个在新domain中访问原domain的下标的expr，从而得到每个原下标用新下标的表示形式
    let access_self = comp_access(build, identity_map(comp.domain())
      .apply_domain(comp.schedule.copy()?)?
      .set_tuple_name(DimType::In, comp.name_cstr())?);
    let n = access_self.get_op_n_arg();
    let mut iter_map = Vec::with_capacity(n as usize - 1);
    for i in 1..n {
      iter_map.push(Expr::from_isl(self, access_self.get_op_arg(i)?));
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
    let ci = box CompInfo { dim_map, store, expr, comp: comp.into() };
    let node = node.set_annotation(self.ctx.id_alloc(None, ci.as_ref() as *const _ as _)?)?;
    info.push(ci);
    node
  }

  // 实际上会修改s中的内容，为了规避借用检查，使用不可变引用
  // 访问/修改s.in_kern来实现在第一个进入GPU的for处生成代码，访问/修改s.kern_cfg来获取kern的启动参数
  fn gen<'a>(&'a self, node: AstNode, s: &'a CodegenState) -> impl Display + 'a {
    let work = move |f: &mut Formatter| {
      match node.get_type() {
        AstNodeType::For => {
          let it = node.for_get_iterator()?.get_id()?.get_name()?;
          let init = node.for_get_init()?.to_C_str()?;
          let cond = node.for_get_cond()?.to_C_str()?;
          let inc = node.for_get_inc()?.to_C_str()?;
          let body = node.for_get_body()?;
          match s.info.get(&*node) {
            // todo: 支持更复杂的parallel模式，比如#pragma omp parallel，每个线程自己同步
            Some(ForInfo { tag: Parallel, .. }) => f.write_str("\n#pragma omp parallel for\n").ok()?,
            // 生成GPU代码的逻辑是：遇到第一个GPU tag的循环维度，在这里生成kern和kern调用
            // kern将包括这个循环下的所有内容，如果某内层循环有GPU tag，不会生成for，而是使用GPU index，否则按照正常代码一样生成
            // 这导致了Comp::tag_dim的注释中描述的问题
            Some(ForInfo { comp, level, tag, used_buf, local_buf }) => {
              let mut s = s.p();
              let old_in_kern = std::mem::replace(&mut s.in_kern, true);
              if !old_in_kern { // 第一个进入GPU的for，在这里生成代码
                let param_buf = comma_sep(used_buf.difference(local_buf).map(|x|
                  fn2display(move |f| write!(f, "{}*__restrict__ {}", x.ty, x.name))));
                write!(f, "{{auto _kern=[=]__device__({}){{", param_buf).ok()?;
              }
              let (sch, pos) = (comp.schedule(), (level * 2 + 1) as _);
              let (min, max) = (sch.copy()?.dim_min_val(pos)?, sch.dim_max_val(pos)?);
              debug!("gen: GPU idx for Comp {} loop {}: {} <= {} < {}", comp.name(), level, min, tag.gpu_idx(), max);
              let range = max.copy()?.sub(min.copy()?)?.add(self.ctx.val_one()?)?;
              let old_cfg = s.kern_cfg[(*tag as usize - GPUBlockX as usize)]
                .replace(range.to_str()?.as_str().into());
              debug_assert!(old_cfg.is_none(), "duplicate gpu tag"); // 嵌套中的多个循环标记了同一个gpu idx，不合法
              write!(f, "{ty} i{i}={idx}+{min};\
                assume({min}<=i{i}&&i{i}<={max});\
                if({init}<=i{i}&&{cond}){{{body}}}", ty = self.iter_ty, i = level, idx = tag.gpu_idx(),
                min = min, max = max, init = init, cond = cond, body = self.gen(body, &*s)).ok()?;
              s.in_kern = old_in_kern;
              if !old_in_kern {
                fn fmt<'a>(c: &'a [Option<Box<str>>]) -> impl Display + 'a {
                  // 用lambda表达式会报声明周期错误，所以用函数定义
                  comma_sep(c.iter().map(|s| s.as_deref().unwrap_or("1")))
                }
                let arg_buf = comma_sep(used_buf.difference(local_buf).map(|x| &x.name));
                write!(f, "}};exec_kern<<<dim3({}),dim3({})>>>(_kern,{});}}",
                  fmt(&s.kern_cfg[..3]), fmt(&s.kern_cfg[3..]), arg_buf).ok()?;
                s.kern_cfg = Default::default();
              }
              return Unit; // 跳过下面的for生成
            }
            _ => {}
          }
          write!(f, "for({} {it}={};{};{it}+={}){{{}}}", self.iter_ty, init, cond, inc, self.gen(body, s), it = it).ok()?;
        }
        AstNodeType::If => {
          let cond = node.if_get_cond()?.to_C_str()?;
          let t = node.if_get_then()?;
          write!(f, "if({}){{{}}}", cond, self.gen(t, s)).ok()?;
          if let Some(e) = node.if_get_else() {
            write!(f, "else{{{}}}", self.gen(e, s)).ok()?;
          }
        }
        AstNodeType::Block => {
          // block node不需要{}包裹，因为if，for已经有{}了，而且用{}包裹会让一些局部变量无法访问
          let ch = node.block_get_children()?;
          for i in 0..ch.n_ast_node() {
            write!(f, "{}", self.gen(ch.get_ast_node(i)?, s)).ok()?;
          }
        }
        AstNodeType::User => {
          let comp = P::new(node.get_annotation()?.get_user() as *const CompInfo);
          if let Some(store) = &comp.store {
            write!(f, "{}={};", store, comp.expr).ok()?;
          } else {
            // 没有store的comp表示成一个标量定义，如果类型是void就只写右手项
            if comp.expr.ty() == Void {
              write!(f, "{};", comp.expr).ok()?;
            } else {
              write!(f, "{} {}={};", comp.expr.ty(), comp.comp.name(), comp.expr).ok()?;
            }
          }
        }
        ty => debug_panic!("invalid ast node type: {:?}", ty),
      }
      Unit
    };
    fn2display(move |f| {
      work(f);
      Ok(())
    })
  }
}

// 从user node中提取Comp::dim_tags中循环的tag
fn extract_tags(node: AstNodeRef, levels: &mut Vec<AstNodeRef>, info: &mut HashMap<AstNodeRef, ForInfo>) -> Unit {
  match node.get_type() {
    AstNodeType::For => {
      levels.push(node);
      extract_tags(*node.for_get_body()?, levels, info)?;
      levels.pop();
    }
    AstNodeType::If => {
      extract_tags(*node.if_get_then()?, levels, info)?;
      if let Some(e) = node.if_get_else() { extract_tags(*e, levels, info)?; }
    }
    AstNodeType::Block => {
      let ch = node.block_get_children()?;
      for i in 0..ch.n_ast_node() { extract_tags(*ch.get_ast_node(i)?, levels, info)?; }
    }
    AstNodeType::User => {
      let comp = P::new(node.get_annotation()?.get_user() as *const CompInfo);
      for (i, &tag) in comp.comp.dim_tags.iter().enumerate() {
        if let (Some(i), Some(tag)) = (comp.dim_map[i], tag) {
          let old = info.insert(levels[i as usize],
            ForInfo { comp: comp.comp, level: i, tag, used_buf: <_>::default(), local_buf: <_>::default() });
          debug_assert!(old.is_none(), "duplicate tag");
        }
      }
    }
    ty => debug_panic!("invalid ast node type: {:?}", ty),
  }
  Unit
}

// 提取used_buf和local_buf信息
fn extract_buf(f: &Func, node: AstNodeRef, s: &mut CodegenState) -> Unit {
  fn extract_buf_expr(e: &Expr, s: &mut CodegenState) {
    e.visit(&mut move |e| match *e {
      Load(x, _) | Free(x) => { s.used_buf.insert(x); }
      Memcpy(x, y) => {
        s.used_buf.insert(x);
        s.used_buf.insert(y);
      }
      Alloc(x) => { s.local_buf.insert(x); }
      _ => {}
    });
  }
  let extract_buf_isl = move |e: AstExpr, s: &mut CodegenState| extract_buf_expr(&Expr::from_isl(f, e), s);
  match node.get_type() {
    AstNodeType::For => {
      let old_in_kern = s.in_kern;
      let info = s.info.get_mut(&node).filter(|x|
        GPUBlockX <= x.tag && x.tag <= GPUThreadZ).map(|x| x.p());
      if info.is_some() { s.in_kern = true; }
      if s.in_kern {
        extract_buf_isl(node.for_get_init()?, s);
        extract_buf_isl(node.for_get_cond()?, s);
        extract_buf_isl(node.for_get_inc()?, s);
      }
      extract_buf(f, *node.for_get_body()?, s)?;
      if let (Some(mut info), false) = (info, old_in_kern) {
        info.used_buf = std::mem::replace(&mut s.used_buf, <_>::default());
        info.local_buf = std::mem::replace(&mut s.local_buf, <_>::default());
      }
      s.in_kern = old_in_kern;
    }
    AstNodeType::If => {
      if s.in_kern { extract_buf_isl(node.if_get_cond()?, s); }
      extract_buf(f, *node.if_get_then()?, s)?;
      if let Some(e) = node.if_get_else() { extract_buf(f, *e, s)?; }
    }
    AstNodeType::Block => {
      let ch = node.block_get_children()?;
      for i in 0..ch.n_ast_node() { extract_buf(f, *ch.get_ast_node(i)?, s)?; }
    }
    AstNodeType::User => if s.in_kern {
      let comp = P::new(node.get_annotation()?.get_user() as *const CompInfo);
      if let Some(store) = &comp.store { extract_buf_expr(store, s); }
      extract_buf_expr(&comp.expr, s);
    }
    ty => debug_panic!("invalid ast node type: {:?}", ty),
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
