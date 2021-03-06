use tempfile::NamedTempFile;
use std::{mem, io::{self, Write, BufWriter}, fs::{self, File}, path::{Path, PathBuf}, process::Command};
use crate::*;

#[derive(Debug, Default)]
pub struct Func {
  // 限制符号常量/参数取值范围
  pub func_ctx: Option<BasicSet>,
  pub name: Box<str>,
  pub comps: Vec<Box<Comp>>,
  pub bufs: Vec<Box<Buf>>,
  // 用于命名自动生成的Comp
  pub comp_cnt: u32,
  // 用于命名自动生成的Buf
  pub buf_cnt: u32,
  // 默认为false，tmp为true时，codegen使用tempfile为文件名；否则以函数名为文件名
  // 理论上tmp和backend都可以作为codegen的参数，但放在这里codegen调用更方便一点
  pub tmp: bool,
  // 是否在生成的wrapper函数中为每个参数调用flush_cache
  pub flush_cache: bool,
  // 默认为CPU
  pub backend: Backend,
  // 用户自定义的额外编译参数
  pub compile_args: Vec<R<str>>,
  // Ctx必须在所有引用Ctx的成员析构后析构
  pub ctx: Ctx,
}

impl Func {
  pub fn new(name: &str) -> Box<Func> {
    let mut ret = box Func::default();
    ret.name = name.into();
    ret
  }

  pub fn find_comp(&self, name: &str) -> Option<&Comp> {
    self.comps.iter().find(|c| c.name() == name).map(|c| c.as_ref())
  }

  pub fn find_buf(&self, name: &str) -> Option<&Buf> {
    self.bufs.iter().find(|c| &*c.name == name).map(|c| c.as_ref())
  }

  pub fn new_comp_id(&self) -> u32 { (self.comp_cnt, self.p().comp_cnt += 1).0 }

  pub fn new_buf_id(&self) -> u32 { (self.buf_cnt, self.p().buf_cnt += 1).0 }

  impl_setter!(set_tmp tmp bool);
  impl_setter!(set_flush_cache flush_cache bool);
  impl_setter!(set_backend backend Backend);

  pub fn compile_arg(&self, args: &str) -> &Self {
    self.p().compile_args.push(args.r());
    self
  }

  // 设置domain/schedule中的params的取值范围
  pub fn set_constraint(&self, csts: Box<[Expr]>) -> Unit {
    self.align_schedule();
    let s = format!("{}{{:{}}}\0", self.comps[0].params(), sep(csts.iter(), "&&"));
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
      if let Some(x) = &c.store {
        x.write(x.read().align_params(all_params.copy()?)?);
      }
    }
    Unit
  }

  // 生成的代码中循环变量的名字是i{start}, i{start + 1}, ...，Expr::from_isl依赖这一点
  // has_static为true时生成的IdList包含static dim的名字，但只是用来占位置，这些名字没用
  pub(crate) fn iter_list(&self, loop_dim: u32, has_static: bool) -> IdList {
    let mut iters = self.ctx.id_list_alloc(if has_static { loop_dim * 2 } else { loop_dim } as _)?;
    for i in 0..loop_dim {
      // static dim名字是_i{i}，生成的代码中不会用到，dynamic dim名字是i{i}
      // 最后一个static dim没有设置名字，这没有影响，因为所有static dim的名字都没用
      if has_static { iters = iters.add(self.ctx.id_alloc(cstr(&format!("_i{}\0", i)), 0 as _)?)?; }
      iters = iters.add(self.ctx.id_alloc(cstr(&format!("i{}\0", i)), 0 as _)?)?;
    }
    iters
  }

  pub(crate) fn build_access_idx(&self, sch: Map, iters: IdList, access: MapRef) -> Vec<Expr> {
    let mut access_idx = Vec::new();
    self.build_impl(sch, iters, |_, b| {
      debug_assert!(access_idx.is_empty()); // 应该恰好访问一次
      access_idx = self.iter_map(b, access.copy()?);
      None // 这里不是为了构建AST，返回None即可
    });
    debug_assert_eq!(access_idx.len(), access.dim(DimType::Out) as _);
    access_idx
  }

  pub(crate) fn build_cond(&self, sch: Map, iters: IdList, cond: SetRef) -> Expr {
    let mut ret = None::<Expr>;
    self.build_impl(sch, iters, |n, b| {
      // sch可能是不连续的，这个函数会被调用多次，每次在不同限制下简化cond，结果需要与起来
      let cond = Expr::from_isl(self, b.expr_from_set(cond.copy()?)?);
      ret = Some(if let Some(x) = &ret { x.land(cond) } else { cond });
      Some(n) // 这里需要返回Some，否则调用一次就失败了
    });
    ret?
  }

  fn build_impl(&self, sch: Map, iters: IdList, mut f: impl FnMut(AstNode, AstBuildRef) -> Option<AstNode>) -> Option<AstNode> {
    let sch = sch.reset_tuple_id(DimType::In)?.reset_tuple_id(DimType::Out)?.union_map_from_map()?;
    self.ctx.ast_build_alloc()?.set_iterators(iters)?.set_at_each_domain(&mut |n, b| f(n, b))?.ast_from_schedule(sch)
  }

  fn iter_map(&self, build: AstBuildRef, access: Map) -> Vec<Expr> {
    debug!("iter_map: access = {}", access);
    let e = access_to_expr(build, access);
    let n = e.get_op_n_arg();
    let mut ret = Vec::with_capacity(n as usize - 1);
    for i in 1..n { ret.push(Expr::from_isl(self, e.get_op_arg(i).unwrap())); }
    ret
  }
}

// build_ast期间生成的Comp信息
// set_at_each_domain的函数可能多次接受表示同一个Comp的节点，这些节点会在生成的代码中重复多次
// 一个Comp可能出现多次，有不同的iter_map(见visit_comp)，所以不能修改Comp的字段，而是保存到CompInfo中
#[derive(Debug)]
pub struct CompInfo {
  // store的目标位置，以Load表示
  pub store: Option<Expr>,
  pub cond: Option<Expr>,
  pub expr: Expr,
  pub comp: P<Comp>,
}

#[derive(Debug)]
pub struct ForInfo {
  // 包含这个for的任意一个Comp
  pub comp: P<CompInfo>,
  // 这个for对应schedule中的哪个dynamic dim
  // 一个for可能出现在多个Comp中，但在不同的Comp中它必须拥有相同的dynamic dim
  pub level: u32,
  pub tag: Option<DimTag>,
  // Comp::fuse往往导致生成的代码非常糟糕，尽量避免使用
  // 作为替代，允许连续多个循环标记同一个tag，例如Parallel或GPUBlockX，这也是fuse唯一的实际用途
  // fuse_extent记录内层同一个tag的循环extent的累乘，借助它从"fuse"的变量中恢复出本层循环变量
  pub fuse_extent: i64,
  // used_buf和local_buf用于Parallel和GPU代码生成，分别表示kern中使用的所有Buf和kern中自己分配的Buf
  // 只有非kern循环和kern循环交界处的kern循环的ForInfo中才有内容
  // 如果一个Buf被使用，但不是自己分配的，就作为参数传给kern
  // 代码生成中使用lambda来生成kern，可以自动捕获使用的标量，所以不收集它们
  // 经实验，lambda捕获的指针会丢失restrict信息，导致kern效率降低，所以需要手动收集它们
  pub used_buf: HashSet<NameHashBuf>,
  pub local_buf: HashSet<NameHashBuf>,
}

impl ForInfo {
  // todo: 虽然for在不同的Comp中有相同的dynamic dim，但上下界不一定相同，目前kern的启动参数和提取feature用到上下界
  pub fn extent(&self) -> Extent { self.comp.comp.extent(self.level) }

  fn capture_scalar<'a>(&'a self) -> impl Display + 'a {
    fn2display(move |f| {
      f.write_str("=")?; // 拷贝捕获其他标量
      for x in self.used_buf.difference(&self.local_buf) {
        // =捕获数组是逐元素拷贝，为了捕获指针需要手动写&数组
        if x.0.loc == Local { write!(f, ",&{}", x.0.name)?; }
      }
      Ok(())
    })
  }

  fn capture_buf<'a>(&'a self) -> impl Display + 'a {
    comma_sep(self.used_buf.difference(&self.local_buf).map(|x| x.0.arg()))
  }

  fn capture_buf_arg<'a>(&'a self) -> impl Display + 'a {
    comma_sep(self.used_buf.difference(&self.local_buf).map(|x| &x.0.name))
  }

  fn capture_iter<'a>(&'a self, loop_dim: u32) -> impl Display + 'a {
    fn2display(move |f| {
      // 先把捕获的变量赋值给临时变量，再把临时变量赋值给同名的变量，这是因为C/C++中不能写int x = x;
      write!(f, "{} {};", iter_ty(), comma_sep((0..loop_dim).map(|i|
        fn2display(move |f| if i < self.level {
          write!(f, "__i{0}=i{0},i{0}=__i{0}", i)
        } else { write!(f, "i{}", i) }))))
    })
  }
}

#[derive(Default)]
pub struct CodegenState {
  // tags保存AST中的for循环的信息，AstNodeRef的具体类型一定是for节点，有且仅有一个Comp在这个for上添加了tag
  pub info: HashMap<AstNodeRef, ForInfo>,
  // 经过align_schedule，所有计算循环层次相同，都是loop_dim
  pub loop_dim: u32,
  // 当前在Parallel或GPU kern内时cur_kern为对应的tag，否则为None
  cur_kern: Option<DimTag>,
  // used_buf和local_buf与ForInfo中的意义相同，是借用这里保存一下，填好后放进第一个进入kern的ForInfo中
  used_buf: HashSet<NameHashBuf>,
  local_buf: HashSet<NameHashBuf>,
  // kern的启动参数，6对应<<<dim3(...), dim3(...)>>>中的6个参数，如果不存在就填1
  kern_cfg: [i64; 6],
}

// AstNode树中的user node的annotation中的指针指向Vec<Box<CompInfo>>中的内容，所以把它放在这里
// 调用者应该不会使用它，但不能用_绑定它，例如let AstInfo(a, s, _) = f.build_ast();，否则它会立即析构
// 需改成let AstInfo(a, s, _i) = f.build_ast();
pub struct AstInfo(pub AstNode, pub CodegenState, pub Vec<Box<CompInfo>>);
impl_try!(AstInfo);

impl Func {
  // 迭代self.comps中参与调度，即inline为false的Comp
  pub fn sch_comps(&self) -> impl IntoIterator<Item=&Comp> {
    self.comps.iter().map(|c| c.as_ref()).filter(|c| !c.inline)
  }

  // feature模块从这个AST中提取特征，无需自己定义AST
  pub fn build_ast(&self) -> AstInfo {
    self.align_schedule();
    // 依据Comp::succ表示的调度图，设置schedule的static dim，从而真正实现after的关系
    let mut vis = HashSet::default();
    let mut prev = None::<P<Comp>>;
    for c in self.sch_comps() {
      if c.pred.is_none() {
        // 所有没有前驱的节点按定义顺序排序，后续节点排在前面节点的所有叶子节点后
        // after_raw要求按照从前往后顺序调用，例如B after_raw C, A after_raw B，不能是A after_raw B, B after_raw C
        // sch_graph_dfs保证这一点(pre-order dfs)，并且需要在sch_graph_dfs之前确定c after_raw 谁
        if let Some(x) = prev { c.after_raw(x, 0); }
        prev = Some(self.sch_graph_dfs(c.p(), &mut vis));
      }
    }
    let mut union_sch = None;
    for c in self.sch_comps() {
      // out dim名字必须是空的，ISL才会生成不完美嵌套的循环，否则只能生成多个完美嵌套的循环
      let sch = c.schedule().identity()?.set_tuple_name(DimType::In, c.name_cstr())?.union_map_from_map()?;
      union_sch = Some(if let Some(x) = union_sch { sch.union(x)? } else { sch });
    }
    let union_sch = union_sch?;
    debug!("build_ast: union_sch = {}", union_sch);
    let mut build = if let Some(ctx) = &self.func_ctx {
      ctx.copy()?.set_from_basic_set()?.ast_build_from_context()
    } else { self.ctx.ast_build_alloc() }?;
    // info在返回式被移动，这不影响其中的元素的地址
    let mut info = Vec::new();
    // 如果写成set_at_each_domain(&mut ...)，理论上这句结束后闭包就析构，尽管实际上它的析构是no-op，严谨起见还是把它保存为变量
    let mut f = |n, build| self.visit_comp(n, build, &mut info).into();
    build = build.set_iterators(self.iter_list(self.comps[0].loop_dim(), true))?.set_at_each_domain(&mut f)?;
    // 这几个ISL构建AST的选项影响不大，去掉也可以
    self.ctx.options_set_ast_build_atomic_upper_bound(1)?;
    self.ctx.options_set_ast_build_exploit_nested_bounds(1)?;
    self.ctx.options_set_ast_build_group_coscheduled(1)?;
    let ast = build.ast_from_schedule(union_sch)?;
    debug!("build_ast: ast = {}", ast);
    let mut s = CodegenState { loop_dim: self.comps[0].loop_dim(), kern_cfg: [1; 6], ..<_>::default() };
    // 实现上必须访问两次才能提取used_buf和local_buf信息，extract_tags只给for加tag
    // 有了tag后，extract_buf才能提取信息并保存到第一个进入GPU的ForInfo中
    extract_tags(*ast, &mut Vec::new(), &mut s.info);
    extract_buf(self, *ast, &mut s);
    debug!("build_ast: for info = {:?}", s.info);
    AstInfo(ast, s, info)
  }

  pub fn codegen_source(&self, args: &[P<Buf>]) -> io::Result<PathBuf> {
    for b in args { b.check_arg(); }
    let AstInfo(ast, s, _info) = self.build_ast();
    let b = self.backend;
    let (f, path) = if self.tmp {
      let (f, path) = NamedTempFile::new()?.into_parts();
      (f, path.keep()?)
    } else {
      let path = Path::new(".").with_file_name(self.name.as_ref()).with_extension(if b == CPU { "cpp" } else { "cu" });
      (File::create(&path)?, path)
    };
    let mut w = BufWriter::new(f);
    w.write_all(include_bytes!("../runtime/src/inc.h"))?;
    // 生成名为{self.name}的实际函数和名为{self.name}_wrapper的wrapper函数
    // wrapper函数接受void **p，从p[0], p[3], p[6], ...位置处读出实际函数的参数，以此调用实际函数
    write!(w, "extern \"C\" void {f}({}){{{} {};{}}}\
      extern \"C\" void {f}_wrapper(void**p){{{}{f}({});}}\n",
      comma_sep(args.iter().map(|x| x.arg())),
      iter_ty(), i0_in(s.loop_dim), self.gen(*ast, &s),
      fn2display(move |f| {
        if self.flush_cache {
          for (i, &x) in args.iter().enumerate() {
            write!(f, "flush_cache(p[{}],{});", 3 * i, x.bytes())?;
          }
        }
        Ok(())
      }),
      comma_sep(args.iter().enumerate().map(|(i, &x)| fn2display(move |f| write!(f, "({}*)p[{}]", x.ty, 3 * i)))),
      f = self.name)?;
    w.flush()?; // 如果没有这句，编译时内容可能尚未写入文件中
    Ok(path)
  }

  fn exec_cmd(&self, mut cmd: Command, bin_path: &Path) {
    cmd.arg("-o").arg(&bin_path);
    for x in &self.compile_args { cmd.arg(&**x); }
    debug!("exec_cmd: cmd = {:?}", cmd);
    let status = cmd.status().expect("failed to exec cmd");
    debug_assert!(status.success());
  }

  fn remove_file(&self, path: &Path, bin_path: &Path) {
    if self.tmp { fs::remove_file(path).and_then(|_| fs::remove_file(bin_path)).expect("failed to remove file") }
  }

  pub fn codegen(&self, args: &[P<Buf>]) -> io::Result<Lib> {
    let path = self.codegen_source(args)?;
    let so_path = path.with_extension("so");
    let b = self.backend;
    let mut cmd = Command::new(if b == CPU { CC } else { NVCC });
    if b == CPU {
      cmd.arg("-x").arg("c++").arg(&path).arg("-Ofast").arg("-march=native").arg("-fPIC").arg("-shared");
    } else {
      cmd.arg("-x").arg("cu").arg(&path).arg("-O3").arg("-use_fast_math").arg("-extended-lambda")
        .arg("-Xcudafe").arg("\"--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used --diag_suppress=noreturn_function_does_return\"")
        .arg("-Xcompiler").arg("-fPIC").arg("-Xcompiler").arg("-shared");
    }
    self.exec_cmd(cmd, &so_path);
    let lib = unsafe { Lib::new(&so_path, &self.name) }.map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    self.remove_file(&path, &so_path);
    Ok(lib)
  }

  pub fn codegen_remote(&self, args: &[P<Buf>], target_triple: &str) -> io::Result<Vec<u8>> {
    let path = self.codegen_source(args)?;
    let obj_path = path.with_extension("o");
    debug_assert_eq!(self.backend, CPU); // 目前不支持remote GPU代码生成
    let mut cmd = Command::new(CC);
    cmd.arg("-x").arg("c++").arg(&path).arg("-Ofast").arg("-target").arg(target_triple).arg("-fPIC").arg("-c");
    self.exec_cmd(cmd, &obj_path);
    let obj = fs::read(&obj_path)?;
    self.remove_file(&path, &obj_path);
    Ok(obj)
  }
}

impl Func {
  // 返回c的排在最后的，即static dim字典序最大的孩子
  fn sch_graph_dfs(&self, c: P<Comp>, vis: &mut HashSet<P<Comp>>) -> P<Comp> {
    if !vis.insert(c) { debug_panic!("cyclic schedule graph"); }
    let mut ret = c;
    for (i, &s) in c.succ.iter().enumerate().rev() {
      if let Some(s) = s {
        s.after_raw(c, i as _);
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
    let store = if let Some(x) = &comp.store {
      let access = x.copy()?.apply_domain(comp.schedule.copy()?.reset_tuple_id(DimType::In)?)?;
      Some(Expr::from_isl(self, access_to_expr(build, access)))
    } else { None };
    // 创建一个在新domain中访问原domain的下标的expr，从而得到每个原下标用新下标的表示形式
    let iter_map = self.iter_map(build, comp.schedule.copy()?.reverse()?);
    debug!("visit_comp: comp = {}, iter_map = [{}]", comp.name(), comma_sep(iter_map.iter()));
    let ref replace_idx = move |e: &mut Expr| match e {
      Access(arg, idx) => {
        *e = access_comp(build, comp, arg, idx);
        debug!("visit_comp: replaced access = {}", e);
        false
      }
      Iter(_, x) => {
        *e = iter_map[*x as usize].clone();
        false
      }
      _ => true
    };
    // 将多次常数乘法合并。本来我也以为编译器可以自动完成，但经实验是否合并会影响到编译器的alias判断，某些情况下对性能有很大影响
    let optimize_mul = |mut e| {
      fn optimize_mul(e: &mut Expr, stride: i64) {
        match e {
          Binary(BinOp::Mul, box [Val(ty, x), y]) | Binary(BinOp::Mul, box [y, Val(ty, x)]) => {
            optimize_mul(y, stride * ty.val_i64(*x));
            *e = y.replace0();
          }
          Binary(BinOp::Add, box [l, r]) | Binary(BinOp::Sub, box [l, r]) => {
            optimize_mul(l, stride);
            optimize_mul(r, stride);
          }
          _ => {
            for x in e.args_mut() { optimize_mul(x, 1); }
            if stride != 1 { *e = e.replace0().mul(stride); }
          }
        }
      }
      (optimize_mul(&mut e, 1), e).1
    };
    let store = store.map(optimize_mul);
    let cond = comp.cond.as_ref().map(|e| optimize_mul(e.modify(replace_idx)));
    let expr = optimize_mul(comp.expr.modify(replace_idx));
    let ci = box CompInfo { store, cond, expr, comp: comp.into() };
    let n = n.set_annotation(self.ctx.id_alloc(None, ci.as_ref() as *const _ as _)?)?;
    info.push(ci);
    n
  }

  // 实际上会修改s中的内容，为了规避借用检查，使用不可变引用
  // 访问/修改s.in_kern来实现在第一个进入GPU的for处生成代码，访问/修改s.kern_cfg来获取kern的启动参数
  fn gen<'a>(&'a self, n: AstNodeRef, s: &'a CodegenState) -> impl Display + 'a {
    let work = move |f: &mut Formatter| {
      match n.get_type() {
        AstNodeType::For => {
          let it = n.for_get_iterator()?.get_id()?.get_name()?;
          let init = n.for_get_init()?.to_C_str()?;
          let cond = n.for_get_cond()?.to_C_str()?;
          let inc = n.for_get_inc()?.to_C_str()?;
          let body = *n.for_get_body()?;
          let info = s.info.get(&n)?;
          if let Some(tag) = info.tag {
            let Extent(min, max, ex) = info.extent();
            let ex = ex.get_num_si();
            let gen_it = |f: &mut Formatter, idx, last_kern|
              write!(f, "{it}={idx}{div}{rem}+{min};\
                assume({min}<={it}&&{it}<={max});\
                if({init}<={it}&&{cond}){{{body}}}",
                it = it, idx = idx, min = min, max = max, init = init, cond = cond, body = self.gen(body, s),
                div = fn2display(|f| if info.fuse_extent != 1 { write!(f, "/{}", info.fuse_extent) } else { Ok(()) }),
                // 本tag的最外层循环不需要取模，只有内层的需要
                rem = fn2display(|f| if last_kern == Some(tag) { write!(f, "%{}", ex) } else { Ok(()) })).ok();
            match tag {
              // 这里读num_thread，所以生成带Parallel的程序前就需要调用parallel_init或者手动设置它
              Parallel => {
                let last_kern = mem::replace(&mut s.p().cur_kern, Some(tag));
                if last_kern.is_none() {
                  write!(f, "{{auto _kern=[{}](u32 _i){{\
                    [=]({}){{{}assume(_i<{th});\
                    {} _s=min({ex},({ex}-1+{th})/{th}*_i),_e=min({ex},({ex}-1+{th})/{th}*(_i+1)),_par;\
                    for(_par=_s;_par<_e;++_par){{",
                    info.capture_scalar(), info.capture_buf(), info.capture_iter(s.loop_dim), iter_ty(),
                    th = unsafe { num_thread }, ex = info.fuse_extent * ex).ok()?;
                }
                gen_it(f, "_par", last_kern)?;
                s.p().cur_kern = last_kern;
                if last_kern.is_none() {
                  write!(f, "}}}}({});}};parallel_launch([](void*_p,u32 _i){{(*(decltype(_kern)*)_p)(_i);}},&_kern);}}", info.capture_buf_arg()).ok()?;
                }
              }
              Unroll => {
                // 编译器不总是会自动unroll，有时候还是需要人给一些hint
                // 循环和下面生成for一样，但现在有tag就会跳过for生成，而且我懒得改代码了，所以复制过来
                write!(f, "\n#pragma unroll\n\
                  for({it}={};{};{it}+={}){{{}}}", init, cond, inc, self.gen(body, s), it = it).ok()?;
              }
              UnrollExplicit => {
                let min = min.get_num_si();
                for i in min..min + ex {
                  write!(f, "{}={};{}", it, i, self.gen(body, s)).ok()?;
                }
              }
              // 我本来无意实现vectorize，相信编译器可以自动完成这件事，而且更加可靠和有效
              // 但经实验，至少在clang 11.0.0上#pragma vectorize与手动vectorize还是有差距，具体例子是kij的gemm micro kernel
              Vectorize => {
                let (comp, i, ex) = (info.comp, info.level, ex as _);
                // 应该确实只有这个计算在这个维度有Vectorize标记，只能处理这种简单情形
                debug_assert_eq!(comp.comp.tags.get(i as usize).copied().flatten(), Some(Vectorize));
                debug_assert!(comp.cond.is_none() && comp.store.is_some());
                // 只处理Load中的两种情形：下标中i出现一次，系数为常熟；下标中没有i。前者用Vector表达式包起来，后者不变，会自动broadcast
                // 其余情景也不保证报错，可能默默放过去产生错误结果
                let ref replace_idx = move |e: &mut Expr| if let Load(_, idx) = e {
                  fn vis_idx(e: &Expr, i: u32, stride: i64) -> i64 {
                    match e {
                      &Iter(_, x) if x == i => stride,
                      Binary(BinOp::Mul, box [Val(ty, x), y]) | Binary(BinOp::Mul, box [y, Val(ty, x)]) =>
                        vis_idx(y, i, stride * ty.val_i64(*x)),
                      _ => e.args().iter().map(|x| vis_idx(x, i, stride)).sum(),
                    }
                  }
                  let k = vis_idx(idx, i, 1);
                  if k != 0 { *e = Ramp(e.ty(), k as _, ex, box e.replace0()); }
                  false
                } else { true };
                let store = comp.store.as_ref()?.modify(replace_idx);
                let expr = comp.expr.modify(replace_idx);
                let ty = comp.expr.ty();
                // CUDA device code中无法使用自定义的向量，只能用预先定义好的，已经在inc.h中using了，不在这个范围内会编译失败
                if self.backend == CPU {
                  write!(f, "using {ty}x{} __attribute__((vector_size({})))={ty};", ex, ex * ty.size() as u32, ty = ty).ok()?;
                }
                // 加上向量类型的0值，如果右端是标量则可以转化成向量
                write!(f, "{}={};{}=({})+{}x{}{{}};", it, min, store, expr, ty, ex).ok()?;
              }
              // 生成GPU代码的逻辑是：遇到第一个GPU tag的循环维度，在这里生成kern和kern调用
              // kern将包括这个循环下的所有内容，如果某内层循环有GPU tag，不会生成for，而是使用GPU index，否则按照正常代码一样生成
              // 这导致了Comp::tag的注释中描述的问题
              _ => { // 必然是GPU相关的tag
                let last_kern = mem::replace(&mut s.p().cur_kern, Some(tag));
                if last_kern.is_none() {
                  write!(f, "{{auto _kern=[=]__device__({}){{{}", info.capture_buf(), info.capture_iter(s.loop_dim)).ok()?;
                }
                s.p().kern_cfg[(tag as usize - GPUBlockX as usize)] *= ex;
                gen_it(f, tag.gpu_idx(), last_kern)?;
                s.p().cur_kern = last_kern;
                if last_kern.is_none() {
                  // 这里假定_kern,后一定有内容，即至少用到了一个Buf，否则这程序也没什么意义
                  write!(f, "}};exec_kern<<<dim3({}),dim3({})>>>(_kern,{});}}", comma_sep(s.kern_cfg[..3].iter()),
                    comma_sep(s.kern_cfg[3..].iter()), info.capture_buf_arg()).ok()?;
                  s.p().kern_cfg = [1; 6];
                }
              }
            }
            return Unit; // 跳过下面的for生成
          }
          // 我修改了ISL的代码，使它不消除任何循环，而经过align_schedule，所有计算都有相同的循环层数，保存在CodegenState::loop_dim中
          // 如果直接生成代码，变量定义也会包裹在循环中，其他位置无法访问，所以对于degenerate的循环，不能用作用域包裹
          // 但这样又会导致循环变量重复定义，解决方案是在函数开头定义所有循环变量(见codegen函数)，这里只是赋值
          if n.for_is_degenerate()? {
            write!(f, "{}={};{}", it, init, self.gen(body, s)).ok()?;
          } else {
            write!(f, "for({it}={};{};{it}+={}){{{}}}", init, cond, inc, self.gen(body, s), it = it).ok()?;
          }
        }
        AstNodeType::If => {
          let cond = n.if_get_cond()?.to_C_str()?;
          let t = n.if_get_then()?;
          write!(f, "if({}){{{}}}", cond, self.gen(*t, s)).ok()?;
          if let Some(e) = n.if_get_else() { write!(f, "else{{{}}}", self.gen(*e, s)).ok()?; }
        }
        AstNodeType::Block => {
          // block node不需要{}包裹，因为if，for已经有{}了，而且用{}包裹会让一些局部变量无法访问
          let ch = n.block_get_children()?;
          for i in 0..ch.n_ast_node() { write!(f, "{}", self.gen(*ch.get_ast_node(i)?, s)).ok()?; }
        }
        AstNodeType::User => {
          let comp = P::<CompInfo>::new(n.get_annotation()?.get_user() as _);
          if let Some(x) = &comp.cond { write!(f, "if({}){{", x).ok()?; }
          if let Some(x) = &comp.store {
            write!(f, "{}={};", x, comp.expr).ok()?;
          } else {
            // 没有store的comp表示成一个标量定义，如果类型是void就只写右手项
            if comp.expr.ty() == Void {
              write!(f, "{};", comp.expr).ok()?;
            } else {
              write!(f, "{} {}={};", comp.expr.ty(), comp.comp.orig_name, comp.expr).ok()?;
            }
          }
          if comp.cond.is_some() { f.write_str("}").ok()?; }
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

// 从user node中提取Comp::tags中循环的tag
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
      debug_assert!(comp.comp.loop_dim() as usize <= loops.len());
      let (mut last_tag, mut fuse_extent) = (None, 1);
      for (i, &l) in loops.iter().enumerate().take(comp.comp.loop_dim() as usize).rev() {
        let tag = comp.comp.tags.get(i).copied().flatten();
        let fuse_extent1 = if tag == last_tag { fuse_extent } else { 1 };
        info.entry(l).and_modify(|old| {
          if old.tag.is_none() {
            old.tag = tag;
            old.fuse_extent = fuse_extent1;
          } else { debug_assert!(tag.is_none(), "duplicate tag"); }
          debug_assert_eq!(old.level, i as _);
        }).or_insert(ForInfo { comp, level: i as _, tag, fuse_extent: fuse_extent1, used_buf: <_>::default(), local_buf: <_>::default() });
        let ex = comp.comp.extent(i as _).2.get_num_si(); // 可能输出错误信息，得到的ex是错误的，但也可以继续
        if tag == last_tag { fuse_extent *= ex; } else { fuse_extent = ex; }
        last_tag = tag;
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
      Load(x, _) | Free(x) => { s.used_buf.insert(NameHashBuf(x)); }
      Memcpy(x, y) => {
        s.used_buf.insert(NameHashBuf(x));
        s.used_buf.insert(NameHashBuf(y));
      }
      Alloc(x) => { s.local_buf.insert(NameHashBuf(x)); }
      _ => {}
    });
  }
  let extract_buf_isl = move |e: AstExpr, s: &mut CodegenState| extract_buf_expr(&Expr::from_isl(f, e), s);
  match n.get_type() {
    AstNodeType::For => {
      let last_kern = s.cur_kern;
      let mut info = s.info.get_mut(&n)?.p();
      let cur_kern = info.tag.filter(|&x| x == Parallel || GPUBlockX <= x && x <= GPUThreadZ);
      // extract_buf()只用到cur_kern是否为Some，不关心其具体值，gen()会关心
      if cur_kern.is_some() { s.cur_kern = cur_kern; }
      if cur_kern.is_some() {
        extract_buf_isl(n.for_get_init()?, s);
        extract_buf_isl(n.for_get_cond()?, s);
        extract_buf_isl(n.for_get_inc()?, s);
      }
      extract_buf(f, *n.for_get_body()?, s)?;
      if cur_kern.is_some() && last_kern.is_none() {
        info.used_buf = mem::replace(&mut s.used_buf, <_>::default());
        info.local_buf = mem::replace(&mut s.local_buf, <_>::default());
      }
      s.cur_kern = last_kern;
    }
    AstNodeType::If => {
      if s.cur_kern.is_some() { extract_buf_isl(n.if_get_cond()?, s); }
      extract_buf(f, *n.if_get_then()?, s)?;
      if let Some(e) = n.if_get_else() { extract_buf(f, *e, s)?; }
    }
    AstNodeType::Block => {
      let ch = n.block_get_children()?;
      for i in 0..ch.n_ast_node() { extract_buf(f, *ch.get_ast_node(i)?, s)?; }
    }
    AstNodeType::User => if s.cur_kern.is_some() {
      let comp = P::<CompInfo>::new(n.get_annotation()?.get_user() as _);
      if let Some(x) = &comp.store { extract_buf_expr(x, s); }
      if let Some(x) = &comp.cond { extract_buf_expr(x, s); }
      extract_buf_expr(&comp.expr, s);
    }
    ty => debug_panic!("invalid ast node type: {:?}", ty),
  }
  Unit
}

fn access_to_expr(build: AstBuildRef, access: Map) -> AstExpr {
  let sch = build.get_schedule()?.map_from_union_map()?;
  let access = access.apply_domain(sch.reset_tuple_id(DimType::In)?)?;
  build.access_from_pw_multi_aff(access.pw_multi_aff_from_map()?)?
}

fn access_comp(build: AstBuildRef, comp: &Comp, arg: &Comp, idx: &[Expr]) -> Expr {
  // 访问有store的计算，转化为访问对应内存；没有store且inline，计算出访问的下标后把下标代入表达式中
  // 没有store且不inline，返回Param表达式，这个表达式之后只会用于输出计算的名字
  let store = match (&arg.store, arg.inline) {
    (Some(x), _) => Some(x.copy()?), (None, true) => None,
    (None, false) => return arg.as_param(),
  };
  let mut access = comp.access(arg.name(), idx);
  if let Some(x) = store {
    access = access.reset_tuple_id(DimType::Out)?.apply_range(x)?;
  }
  debug!("access_comp: access = {}", access);
  if arg.inline {
    // inline的Comp的expr中可能还有对别的Comp的访问，但这没法处理了，因为需要arg下的build才能构造表达式，现在只有comp下的build
    let mut e = arg.expr.clone();
    e.replace_iter(&comp.func.iter_map(build, access));
    e
  } else {
    Expr::from_isl(&comp.func, access_to_expr(build, access))
  }
}
