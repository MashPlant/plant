use libloading::Library;
use tempfile::NamedTempFile;
use std::{mem, io::{self, Write, BufWriter}, fs::{self, File}, path::Path, process::Command};
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
  // 默认为false，tmp为true时，codegen使用tempfile为文件名；否则以函数名为文件名
  // 理论上tmp和backend都可以作为codegen的参数，但放在这里codegen调用更方便一点
  pub tmp: bool,
  // 默认为false，keep_degenerate_for为true时，不删除退化的循环(循环变量只有一个取值)，一个dynamic dim必然对应一个循环
  // 在调用了Comp::tag_dim和提取feature这两种情况下，如果程序结构可能因为循环退化而改变，需要设置为true
  pub keep_degenerate_for: bool,
  // 默认为C
  pub backend: Backend,
  // Ctx必须在所有引用Ctx的成员析构后析构
  pub ctx: Ctx,
}

impl Func {
  pub fn new(name: &str) -> Box<Func> {
    box Func { func_ctx: None, name: name.into(), comps: Vec::new(), bufs: Vec::new(), iter_ty: I32, comp_cnt: 0, buf_cnt: 0, tmp: false, keep_degenerate_for: false, backend: C, ctx: Ctx::new() }
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
      Memcpy(..) => "memcpy", Alloc(..) => "alloc", Free(..) => "free", Sync => "sync", Opaque(..) => "opaque",
    };
    format!("_{}{}", desc, self.new_comp_id())
  }

  pub fn new_buf_id(&self) -> u32 { (self.buf_cnt, self.p().buf_cnt += 1).0 }

  impl_setter!(set_tmp tmp bool);
  impl_setter!(set_keep_degenerate_for keep_degenerate_for bool);
  impl_setter!(set_backend backend Backend);

  // 设置domain/schedule中的params的取值范围
  pub fn set_constraint(&self, csts: &[Expr]) -> Unit {
    self.align_schedule();
    let s = format!("{} -> {{: {}}}\0", self.comps[0].params(), sep(csts.iter(), " and "));
    debug!("set_constraint: {}", s);
    self.p().func_ctx = Some(self.ctx.basic_set_read_from_str(cstr(&s))?);
    Unit
  }

  // 将所有Comp的schedule的range维度统一成最大的，不足的维度补0
  // 并将domain和schedule的params都统一成全部params
  pub fn align_schedule(&self) -> Unit {
    let mut max_dim = 0;
    let mut all_params = self.comps[0].schedule.get_space()?;
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
}

// build_ast期间生成的Comp信息
// set_at_each_domain的函数可能多次接受表示同一个Comp的节点，这些节点会在生成的代码中重复多次
// 一个Comp可能出现多次，有不同的iter_map(见visit_comp)，所以不能修改Comp的字段，而是保存到CompInfo中
#[derive(Debug)]
pub struct CompInfo {
  // store的目标位置，以Load表示
  pub store: Option<Expr>,
  pub expr: Expr,
  pub comp: P<Comp>,
}

#[derive(Debug)]
pub struct ForInfo {
  // 包含这个for的任意一个Comp
  pub comp: P<Comp>,
  // 这个for对应schedule中的哪个dynamic dim
  // 一个for可能出现在多个Comp中，但在不同的Comp中它必须拥有相同的dynamic dim
  pub level: u32,
  pub tag: Option<DimTag>,
  // used_buf和local_buf用于GPU代码生成，分别表示kern中使用的所有Buf和kern中自己分配的Buf
  // 只有非kern循环和kern循环交界处的kern循环的ForInfo中才有内容
  // 如果一个Buf被使用，但不是自己分配的，就作为参数传给kern
  // 代码生成中使用extended lambda来生成kern，可以自动捕获使用的标量，所以不收集它们
  // 其实也可以自动捕获使用的指针，但经实验这样会丢失restrict信息，导致kern效率降低，所以还是手动收集它们
  pub used_buf: HashSet<P<Buf>>,
  pub local_buf: HashSet<P<Buf>>,
}

// (min, max, max - min + 1)
pub struct ForExtent(pub isl::val_type::Val, pub isl::val_type::Val, pub isl::val_type::Val);
impl_try!(ForExtent);

impl ForInfo {
  // todo: 虽然for在不同的Comp中有相同的dynamic dim，但上下界不一定相同，目前kern的启动参数和提取feature用到上下界
  pub fn extent(&self) -> ForExtent {
    let (sch, pos) = (self.comp.schedule(), (self.level * 2 + 1) as _);
    let (min, max) = (sch.copy()?.dim_min_val(pos)?, sch.dim_max_val(pos)?);
    let extent = max.copy()?.sub(min.copy()?)?.add(self.comp.ctx.val_one()?)?;
    ForExtent(min, max, extent)
  }
}

#[derive(Default)]
pub struct CodegenState {
  // tags保存AST中的for循环的信息，AstNodeRef的具体类型一定是for节点，有且仅有一个Comp在这个for上添加了tag
  pub info: HashMap<AstNodeRef, ForInfo>,
  // 以下都是codegen过程中的临时值，in_kern表示当前是否在GPU内核中
  in_kern: bool,
  // used_buf和local_buf与ForInfo中的意义相同，是借用这里保存一下，填好后放进第一个进入GPU的for的ForInfo中
  used_buf: HashSet<P<Buf>>,
  local_buf: HashSet<P<Buf>>,
  // kern的启动参数，6对应<<<dim3(...), dim3(...)>>>中的6个参数，如果是None就填1
  kern_cfg: [Option<Box<str>>; 6],
}

// AstNode树中的user node的annotation中的指针指向Vec<Box<CompInfo>>中的内容，所以把它放在这里
// 调用者应该不会使用它，但不能用_绑定它，例如let AstInfo(a, s, _) = f.build_ast();，否则它会立即析构
// 需改成let AstInfo(a, s, _i) = f.build_ast();
pub struct AstInfo(pub AstNode, pub CodegenState, pub Vec<Box<CompInfo>>);
impl_try!(AstInfo);

impl Func {
  // feature模块从这个AST中提取特征，无需自己定义AST
  pub fn build_ast(&self) -> AstInfo {
    self.align_schedule();
    // 依据Comp::succ表示的调度图，设置schedule的static dim，从而真正实现after的关系
    let mut vis = HashSet::default();
    let mut prev = None::<P<Comp>>;
    for c in &self.comps {
      if c.pred.is_none() {
        // 所有没有前驱的节点按定义顺序排序，后续节点排在前面节点的所有叶子节点后
        // after_raw要求按照从前往后顺序调用，例如B after_raw C, A after_raw B，不能是A after_raw B, B after_raw C
        // sch_graph_dfs保证这一点(pre-order dfs)，并且需要在sch_graph_dfs之前确定c after_raw 谁
        if let Some(x) = prev { c.after_raw(&*x, 0); }
        prev = Some(self.sch_graph_dfs(c.as_ref().p(), &mut vis));
      }
    }
    let mut union_sch = None;
    for c in &self.comps {
      // out dim名字必须是空的，ISL才会生成不完美嵌套的循环，否则只能生成多个完美嵌套的循环，identity_map保证这一点
      let sch = identity_map(c.schedule()).union_map_from_map()?;
      union_sch = Some(if let Some(x) = union_sch { sch.union(x)? } else { sch });
    }
    let union_sch = union_sch?;
    debug!("build_ast: union_sch = {}", union_sch);
    let mut build = if let Some(ctx) = self.func_ctx.as_ref() {
      ctx.copy()?.set_from_basic_set()?.ast_build_from_context()
    } else { self.ctx.ast_build_alloc() }?;
    let n_dim = self.comps[0].sch_dim();
    debug_assert_eq!(n_dim % 2, 1); // 一定是 static, dynamic, ..., static的模式
    let mut iters = self.ctx.id_list_alloc(n_dim as _)?;
    for i in 0..n_dim / 2 {
      // static dim名字是_i{i}，生成的代码中不会用到，dynamic dim名字是i{i}
      // 最后一个static dim没有设置名字，这没有影响，因为所有static dim的名字都没用
      iters = iters.add(self.ctx.id_alloc(cstr(&format!("_i{}\0", i)), 0 as _)?)?
        .add(self.ctx.id_alloc(cstr(&format!("i{}\0", i)), 0 as _)?)?;
    }
    // info在返回式被移动，这不影响其中的元素的地址
    let mut info = Vec::new();
    build = build.set_iterators(iters)?
      .set_at_each_domain(&mut |n, build| self.visit_comp(n, build, &mut info).into())?;
    if self.keep_degenerate_for {
      self.ctx.options_set_ast_build_keep_degenerate_for(1)?;
    }
    // 这几个ISL构建AST的选项影响不大，去掉也可以
    self.ctx.options_set_ast_build_atomic_upper_bound(1)?;
    self.ctx.options_set_ast_build_exploit_nested_bounds(1)?;
    self.ctx.options_set_ast_build_group_coscheduled(1)?;
    let ast = build.ast_from_schedule(union_sch)?;
    debug!("build_ast: ast = {}", ast);
    let mut s = CodegenState::default();
    // 实现上必须访问两次才能提取used_buf和local_buf信息，extract_tags只给for加tag
    // 有了tag后，extract_buf才能提取信息并保存到第一个进入GPU的ForInfo中
    extract_tags(*ast, &mut Vec::new(), &mut s.info);
    extract_buf(self, *ast, &mut s);
    debug!("build_ast: for info = {:?}", s.info);
    AstInfo(ast, s, info)
  }

  pub fn codegen(&self, args: &[P<Buf>]) -> io::Result<Library> {
    for b in args { debug_assert_ne!(b.kind, BufKind::Temp); }
    let AstInfo(ast, s, _info) = self.build_ast();
    let b = self.backend;
    let (f, path) = if self.tmp {
      let (f, path) = NamedTempFile::new()?.into_parts();
      (f, path.keep()?)
    } else {
      let path = Path::new(".").with_file_name(self.name.as_ref()).with_extension(if b == C { "c" } else { "cu" });
      (File::create(&path)?, path)
    };
    let mut w = BufWriter::new(f);
    w.write_all(include_bytes!("inc.h"))?;
    // 生成名为{self.name}的实际函数和名为{self.name}_wrapper的wrapper函数
    // wrapper函数接受void **p，从p[0], p[2], p[4], ...位置处读出实际函数的参数，以此调用实际函数
    write!(w, "void {f}({}){{{}}}\
      void {f}_wrapper(void**p){{{f}({});}}\n", comma_sep(args.iter().map(|x| x.arg())), self.gen(ast, &s),
      comma_sep(args.iter().enumerate().map(|(i, &x)| fn2display(move |f| write!(f, "({}*)p[{}]", x.ty, 2 * i)))),
      f = self.name)?;
    w.flush()?; // 如果没有这句，下面编译时内容可能尚未写入文件中
    let so_path = path.with_extension("so");
    let mut cmd = Command::new(if b == C { CC } else { NVCC });
    match b {
      C => cmd.arg("-x").arg("c").arg(&path).arg("-Ofast").arg("-march=native").arg("-fopenmp"),
      CUDA => cmd.arg("-x").arg("cu").arg(&path).arg("-O3"),
    };
    cmd.arg("-fPIC").arg("-shared").arg("-o").arg(&so_path);
    debug!("codegen: cmd = {:?}", cmd);
    let status = cmd.status()?;
    debug_assert!(status.success());
    let lib = Library::new(&so_path).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    if self.tmp { fs::remove_file(path)?; }
    fs::remove_file(so_path)?;
    Ok(lib)
  }
}

impl Func {
  // 返回c的排在最后的，即static dim字典序最大的孩子
  fn sch_graph_dfs(&self, c: P<Comp>, vis: &mut HashSet<P<Comp>>) -> P<Comp> {
    if !vis.insert(c) { debug_panic!("cyclic schedule graph"); }
    let mut ret = c;
    for (at, &s) in c.succ.iter().enumerate().rev() {
      if let Some(s) = s {
        s.after_raw(c.get(), at as _);
        ret = self.sch_graph_dfs(s, vis);
      }
    }
    ret
  }

  // 将node表示的comp的expr中的原下标替换成新下标，对comp的access替换成对buf的load
  // 将store的位置也用load保存在store_expr中(虽然实际不是load，但下标表示是一样的)
  fn visit_comp(&self, n: AstNode, build: AstBuildRef, info: &mut Vec<Box<CompInfo>>) -> AstNode {
    let expr = n.user_get_expr()?;
    let name = expr.get_op_arg(0)?.get_id()?.get_name()?;
    let comp = self.find_comp(&name)?;
    let store = if let Some(store) = comp.store.as_ref() {
      let access = store.copy()?.apply_domain(comp.schedule.copy()?)?
        .set_tuple_name(DimType::In, comp.name_cstr())?;
      Some(Expr::from_isl(self, comp_access(build, access)))
    } else { None };
    // 创建一个在新domain中访问原domain的下标的expr，从而得到每个原下标用新下标的表示形式
    let access_self = comp_access(build, identity_map(comp.domain())
      .apply_domain(comp.schedule.copy()?)?
      .set_tuple_name(DimType::In, comp.name_cstr())?);
    let op_n = access_self.get_op_n_arg();
    let mut iter_map = Vec::with_capacity(op_n as usize - 1);
    for i in 1..op_n {
      iter_map.push(Expr::from_isl(self, access_self.get_op_arg(i)?));
    }
    debug!("visit_comp: comp = {}, iter_map = [{}]", comp.name(), comma_sep(iter_map.iter()));
    let mut expr = comp.expr.clone();
    expr.visit_mut(&mut move |e| match e {
      // access_to_load已经将原下标替换成了新下标，不能再访问它的孩子再替换一次了
      Access(arg, idx) => {
        *e = access_to_load(build, &comp, arg, idx);
        debug!("visit_comp: replaced access = {}", e);
        false
      }
      Iter(_, x) => {
        *e = iter_map[*x as usize].clone();
        false
      }
      _ => true
    });
    let ci = box CompInfo { store, expr, comp: comp.into() };
    let n = n.set_annotation(self.ctx.id_alloc(None, ci.as_ref() as *const _ as _)?)?;
    info.push(ci);
    n
  }

  // 实际上会修改s中的内容，为了规避借用检查，使用不可变引用
  // 访问/修改s.in_kern来实现在第一个进入GPU的for处生成代码，访问/修改s.kern_cfg来获取kern的启动参数
  fn gen<'a>(&'a self, n: AstNode, s: &'a CodegenState) -> impl Display + 'a {
    let work = move |f: &mut Formatter| {
      match n.get_type() {
        AstNodeType::For => {
          let it = n.for_get_iterator()?.get_id()?.get_name()?;
          let init = n.for_get_init()?.to_C_str()?;
          let cond = n.for_get_cond()?.to_C_str()?;
          let inc = n.for_get_inc()?.to_C_str()?;
          let body = n.for_get_body()?;
          match s.info.get(&*n) {
            // todo: 支持更复杂的parallel模式，比如#pragma omp parallel，每个线程自己同步
            Some(ForInfo { tag: Some(Parallel), .. }) => f.write_str("\n#pragma omp parallel for\n").ok()?,
            // 生成GPU代码的逻辑是：遇到第一个GPU tag的循环维度，在这里生成kern和kern调用
            // kern将包括这个循环下的所有内容，如果某内层循环有GPU tag，不会生成for，而是使用GPU index，否则按照正常代码一样生成
            // 这导致了Comp::tag_dim的注释中描述的问题
            Some(info @ ForInfo { tag: Some(tag), .. }) => {
              let old_in_kern = mem::replace(&mut s.p().in_kern, true);
              if !old_in_kern { // 第一个进入GPU的for，在这里生成代码
                let param_buf = comma_sep(info.used_buf.difference(&info.local_buf).map(|x| x.arg()));
                write!(f, "{{auto _kern=[=]__device__({}){{", param_buf).ok()?;
              }
              let ForExtent(min, max, extent) = info.extent();
              debug!("gen: GPU idx for Comp {} loop {}: {} <= {} < {}", info.comp.name(), info.level, min, tag.gpu_idx(), max);
              let old_cfg = s.p().kern_cfg[(*tag as usize - GPUBlockX as usize)].replace(extent.to_str()?.as_str().into());
              debug_assert!(old_cfg.is_none(), "duplicate gpu tag"); // 嵌套中的多个循环标记了同一个gpu idx，不合法
              write!(f, "{ty} i{i}={idx}+{min};\
                assume({min}<=i{i}&&i{i}<={max});\
                if({init}<=i{i}&&{cond}){{{body}}}", ty = self.iter_ty, i = info.level, idx = tag.gpu_idx(),
                min = min, max = max, init = init, cond = cond, body = self.gen(body, &*s)).ok()?;
              s.p().in_kern = old_in_kern;
              if !old_in_kern {
                fn fmt<'a>(c: &'a [Option<Box<str>>]) -> impl Display + 'a {
                  // 用lambda表达式会报声明周期错误，所以用函数定义
                  comma_sep(c.iter().map(|s| s.as_deref().unwrap_or("1")))
                }
                let arg_buf = comma_sep(info.used_buf.difference(&info.local_buf).map(|x| &x.name));
                write!(f, "}};exec_kern<<<dim3({}),dim3({})>>>(_kern,{});}}",
                  fmt(&s.kern_cfg[..3]), fmt(&s.kern_cfg[3..]), arg_buf).ok()?;
                s.p().kern_cfg = Default::default();
              }
              return Unit; // 跳过下面的for生成
            }
            _ => {}
          }
          write!(f, "for({} {it}={};{};{it}+={}){{{}}}", self.iter_ty, init, cond, inc, self.gen(body, s), it = it).ok()?;
        }
        AstNodeType::If => {
          let cond = n.if_get_cond()?.to_C_str()?;
          let t = n.if_get_then()?;
          write!(f, "if({}){{{}}}", cond, self.gen(t, s)).ok()?;
          if let Some(e) = n.if_get_else() { write!(f, "else{{{}}}", self.gen(e, s)).ok()?; }
        }
        AstNodeType::Block => {
          // block node不需要{}包裹，因为if，for已经有{}了，而且用{}包裹会让一些局部变量无法访问
          let ch = n.block_get_children()?;
          for i in 0..ch.n_ast_node() { write!(f, "{}", self.gen(ch.get_ast_node(i)?, s)).ok()?; }
        }
        AstNodeType::User => {
          let comp = P::<CompInfo>::new(n.get_annotation()?.get_user() as _);
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
fn extract_tags(n: AstNodeRef, loops: &mut Vec<AstNodeRef>, info: &mut HashMap<AstNodeRef, ForInfo>) -> Unit {
  match n.get_type() {
    AstNodeType::For => {
      loops.push(n);
      extract_tags(*n.for_get_body()?, loops, info)?;
      loops.pop();
    }
    AstNodeType::If => {
      extract_tags(*n.if_get_then()?, loops, info)?;
      if let Some(e) = n.if_get_else() { extract_tags(*e, loops, info)?; }
    }
    AstNodeType::Block => {
      let ch = n.block_get_children()?;
      for i in 0..ch.n_ast_node() { extract_tags(*ch.get_ast_node(i)?, loops, info)?; }
    }
    AstNodeType::User => {
      let comp = P::<CompInfo>::new(n.get_annotation()?.get_user() as _);
      for i in 0..comp.comp.loop_dim() as usize {
        if let Some(&l) = loops.get(i) {
          let tag = comp.comp.dim_tags.get(i).copied().flatten();
          info.entry(l).and_modify(|old| {
            if old.tag.is_none() { old.tag = tag; } else { debug_assert!(tag.is_none(), "duplicate tag"); }
            debug_assert_eq!(old.level, i as _);
          }).or_insert(ForInfo { comp: comp.comp, level: i as _, tag, used_buf: <_>::default(), local_buf: <_>::default() });
        }
      }
    }
    ty => debug_panic!("invalid ast node type: {:?}", ty),
  }
  Unit
}

// 提取used_buf和local_buf信息
fn extract_buf(f: &Func, n: AstNodeRef, s: &mut CodegenState) -> Unit {
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
  match n.get_type() {
    AstNodeType::For => {
      s.info.get_mut(&n).unwrap();
      let old_in_kern = s.in_kern;
      let info = s.info.get_mut(&n).filter(|x|
        Some(GPUBlockX) <= x.tag && x.tag <= Some(GPUThreadZ)).map(|x| x.p());
      if info.is_some() { s.in_kern = true; }
      if s.in_kern {
        extract_buf_isl(n.for_get_init()?, s);
        extract_buf_isl(n.for_get_cond()?, s);
        extract_buf_isl(n.for_get_inc()?, s);
      }
      extract_buf(f, *n.for_get_body()?, s)?;
      if let (Some(mut info), false) = (info, old_in_kern) {
        info.used_buf = mem::replace(&mut s.used_buf, <_>::default());
        info.local_buf = mem::replace(&mut s.local_buf, <_>::default());
      }
      s.in_kern = old_in_kern;
    }
    AstNodeType::If => {
      if s.in_kern { extract_buf_isl(n.if_get_cond()?, s); }
      extract_buf(f, *n.if_get_then()?, s)?;
      if let Some(e) = n.if_get_else() { extract_buf(f, *e, s)?; }
    }
    AstNodeType::Block => {
      let ch = n.block_get_children()?;
      for i in 0..ch.n_ast_node() { extract_buf(f, *ch.get_ast_node(i)?, s)?; }
    }
    AstNodeType::User => if s.in_kern {
      let comp = P::<CompInfo>::new(n.get_annotation()?.get_user() as _);
      if let Some(store) = &comp.store { extract_buf_expr(store, s); }
      extract_buf_expr(&comp.expr, s);
    }
    ty => debug_panic!("invalid ast node type: {:?}", ty),
  }
  Unit
}

fn access_to_load(build: AstBuildRef, comp: &Comp, arg: &Comp, idx: &[Expr]) -> Expr {
  let s = format!("{} -> {{ {}{} -> {}[{}] }}\0", comp.params(),
    comp.name(), i0_in(comp.orig_dim()), arg.name(), comma_sep(idx.iter()));
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
