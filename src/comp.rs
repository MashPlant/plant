use crate::*;

#[derive(Debug)]
pub struct Comp {
  pub ctx: CtxRef,
  pub func: P<Func>,
  // 用于输出，在separate复制计算时，schedule中的名字会变，orig_name不变
  pub orig_name: Box<str>,
  pub expr: Expr,
  pub cond: Option<Expr>,
  // schedule将表示原始循环迭代范围的set映射到调度后的循环迭代范围的set，同时还包含循环间的顺序关系
  // in dim名字是Comp的名字，out dim名字是空的，#out = #in * 2 + 1
  // out dim分为static和dynamic dim，从循环层次i到包围它的static dim：i * 2；从循环层次i到它的dynamic dim：i * 2 + 1
  // isl_basic_set表示可以用一组仿射约束的交集定义的集合，isl_set表示一组isl_basic_set的并集
  // 这里需要用到isl_set，例如i <= max(x, y)这样的约束无法用交集表示，必须是i <= x or i <= y
  pub schedule: Map,
  pub store: Option<Map>,
  pub pred: Option<P<Comp>>,
  pub succ: Vec<Option<P<Comp>>>,
  pub tags: Vec<Option<DimTag>>,
  // 默认为false，c.inline为true时，c不出现在生成的代码中，而是在Access(c, idx)的地方替换成expr中的Iter替换成idx元素的结果
  pub inline: bool,
}

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum DimTag {
  Parallel,
  Unroll,
  UnrollExplicit,
  Vectorize,
  GPUBlockX,
  GPUBlockY,
  GPUBlockZ,
  GPUThreadX,
  GPUThreadY,
  GPUThreadZ,
}

// 将Comp传给接受impl IntoExpr的地方时，是将它作为Param表达式，而非Access表达式
// 用法区别请看Comp::as_param的注释
impl IntoExpr for P<Comp> {
  fn expr(self) -> Expr { self.as_param() }
}

impl Func {
  // ubs = 每个循环变量的upper bound
  // 包括此处在内，很多接受Box<[Expr]>的函数其实接受&[Expr]就够了，但为了配合expr-macro，让用户方便一点，这点浪费可以接受
  pub fn comp(&self, name: &str, ubs: Box<[Expr]>, e: Expr) -> P<Comp> {
    let e = e.expr();
    let mut params = HashSet::default();
    for ub in ubs.iter() { ub.collect_params(&mut params); }
    e.collect_params(&mut params);
    let s = format!("[{}]->{{{}[{}]:{}}}\0", comma_sep(params.iter().map(|c| c.name())),
      name, i0_in(ubs.len() as _), sep(ubs.iter().enumerate().map(|(i, ub)| fn2display(move |f|
        write!(f, "0<=i{}<{}", i, ub))), "&&"));
    debug!("comp: domain = {}", s);
    self.comp_raw(self.ctx.set_read_from_str(cstr(&s))?, e)
  }

  pub fn comp_raw(&self, domain: Set, expr: Expr) -> P<Comp> {
    // set_read_from_str生成的set可能有冗余，例如为i <= min(x, y)生成两个BasicSet，其实一个就可以表示，coalesce就是试图合并BasicSet
    let schedule = identity_schedule(domain.coalesce()?);
    debug!("comp_raw: initial identity schedule = {}", schedule);
    let orig_name = schedule.get_tuple_name(DimType::In).unwrap().as_str().into();
    let comp = box Comp { ctx: *self.ctx, func: self.into(), orig_name, expr, cond: None, schedule, store: None, pred: None, succ: Vec::new(), tags: Vec::new(), inline: false };
    debug_assert!(self.find_comp(comp.name()).is_none()); // 不允许相同名字的Comp
    if cfg!(debug_assertions) { comp.check_iter(&comp.expr); }
    let ret = comp.as_ref().p();
    self.p().comps.push(comp);
    ret
  }
}

impl Comp {
  // 返回的字符串来源于cstr，[len()]位置是\0，可以传入ISL的接口，更直接的是使用name_cstr
  pub fn name(&self) -> &str { self.name_cstr().unwrap().as_str() }

  pub fn name_cstr(&self) -> Option<CStr> { self.schedule.get_tuple_name(DimType::In) }

  pub fn orig_dim(&self) -> u32 { self.schedule.dim(DimType::In) as _ }

  pub fn sch_dim(&self) -> u32 { self.schedule.dim(DimType::Out) as _ }

  // 调度后的循环层数
  pub fn loop_dim(&self) -> u32 { self.sch_dim() / 2 }

  // 表示原始循环迭代范围的Set
  pub fn domain(&self) -> Set { self.schedule.copy()?.domain()? }

  // 表示调度后循环迭代范围的Set，名字为空，因为schedule的out dim名字为空
  pub fn schedule(&self) -> Set { self.schedule.copy()?.range()? }

  // 输出[逗号分隔的params列表]->
  pub fn params<'a>(&'a self) -> impl Display + 'a {
    fn2display(move |f| write!(f, "[{}]->", comma_sep((0..self.schedule.dim(DimType::Param) as u32)
      .map(|i| self.schedule.get_dim_name(DimType::Param, i).unwrap()))))
  }

  impl_setter!(set_expr expr Expr);
  impl_setter!(set_cond cond Option<Expr>);
  impl_setter!(set_inline inline bool);

  // 将自身作为一个Param表达式，Expr中直接通过它的名字使用它
  // 当且仅当它没有store的时候，生成的代码会在计算发生的地方定义一个对应名字的局部变量，所以它必须没有store
  // 如果Comp的计算结果用于其他Comp中的循环范围，Access下标，则必须用as_param
  // 如果只是用于普通运算，可以用as_param或at，as_param会往domain/schedule中引入一个参数，应该没有什么好处
  pub fn as_param(&self) -> Expr { Param(self.into()) }

  pub fn at(&self, idx: Box<[Expr]>) -> Expr {
    debug_assert_eq!(self.orig_dim(), idx.len() as _);
    Access(self.into(), idx)
  }

  // 长度为orig_dim的列表，每个元素用调度后的下标表示调度前的下标
  pub fn iter_map(&self) -> Vec<Expr> {
    let mut ret = Vec::new();
    (|| { // 在lambda表达式里才能用?操作符
      // project_static_dim_map省去也可以，写了后懒得去掉了，而且去掉无用的维度速度应该会快一点
      let sch = project_static_dim_map(self.schedule.copy()?.reset_tuple_id(DimType::In)?);
      // 与Func::visit_comp中一样，创建一个在新domain中访问原domain的下标的expr，从而得到每个原下标用新下标的表示形式
      ret = self.func.build_access_idx(sch.copy()?.range()?.identity()?,
        self.func.iter_list(self.loop_dim(), false), *sch.reverse()?);
      Unit
    })();
    ret
  }

  // 长度为loop_dim的列表，每个元素用调度前的下标表示调度后的下标
  pub fn iter_map_rev(&self) -> Vec<Expr> {
    let mut ret = Vec::new();
    (|| {
      let sch = project_static_dim_map(self.schedule.copy()?.reset_tuple_id(DimType::In)?);
      ret = self.func.build_access_idx(sch.copy()?.domain()?.identity()?,
        self.func.iter_list(self.loop_dim(), false), *sch);
      Unit
    })();
    ret
  }

  pub(crate) fn access(&self, arg: &str, idx: &[Expr]) -> Map {
    let s = format!("{}{{[{}]->{}[{}]}}\0", self.params(), i0_in(self.orig_dim()), arg, comma_sep(idx.iter()));
    debug!("access: {}", s);
    self.ctx.map_read_from_str(cstr(&s))
      .expect("failed to parse access map, comp may have non-affine access")
      .apply_domain(self.schedule.copy()?.reset_tuple_id(DimType::In)?)?
      .detect_equalities()?
  }
}

// Bound(zero, min, max)，意义是循环i及外层循环的空间上的0，i的最小值，最大值 + 1 (min <= i < max)
// 最小值和最大值用包围i的循环变量表示
pub struct Bound(Aff, PwAff, PwAff);
impl_try!(Bound);

// (min, max, max - min + 1)
// min和max相当于Bound中的min，max表达式再取min，max
pub struct Extent(pub isl::val_type::Val, pub isl::val_type::Val, pub isl::val_type::Val);
impl_try!(Extent);

impl Comp {
  pub fn tile(&self, i: u32, j: u32, tile_i: u32, tile_j: u32) -> P<Comp> {
    debug_assert!(i < j);
    self.split(i, tile_i).split(j + 1, tile_j).reorder(i + 1, j + 1)
  }

  pub fn tile_3(&self, i: u32, j: u32, k: u32, tile_i: u32, tile_j: u32, tile_k: u32) -> P<Comp> {
    debug_assert!(i < j && j < k);
    self.split(i, tile_i).split(j + 1, tile_j).split(k + 2, tile_k)
      // i0 i1 j0 j1 k0 k1 -> i0 j0 k0 i1 j1 k1
      .reorder_n(&[(i + 1, j + 1), (j + 1, k + 2), (j + 2, i + 1), (k + 2, j + 2)])
  }

  // split后外层循环extent为原extent/factor，内层为factor
  pub fn split(&self, i: u32, factor: u32) -> P<Comp> {
    debug_assert!(i < self.loop_dim());
    let (n, i) = (self.sch_dim(), i * 2 + 1);
    let s = format!("{{[{}]->[{}]:i{i0}=floor(i{i}/{f})&&i{i1}=i{i}%{f}}}\0", i0_in(n),
      comma_sep((0..n + 2).map(|x| fn2display(move |f|
        if x == i + 1 { f.write_str("0") } else {
          write!(f, "i{}", if x < i { x } else if x == i { n } else if x == i + 2 { n + 1 } else { x - 2 })
        }))),
      i0 = n, i1 = n + 1, i = i, f = factor);
    debug!("split: {}", s);
    self.apply_sch_raw(&s)
  }

  // 相当于split(extent / n_parts)
  // 且和split一样，生成的代码中都是用 外 * factor + 内 来表示原循环变量，而非 内 * n_parts + 外
  pub fn split_n_parts(&self, i: u32, n_parts: u32) -> P<Comp> {
    let extent = self.extent(i).2.get_num_si() as u32;
    debug!("split_n_parts: extent = {}, factor = {}", extent, extent / n_parts);
    self.split(i, extent / n_parts)
  }

  // 将循环层次i和i + 1合并成一个循环
  // 循环i + 1的static dim被丢弃；after，tag之类的都不会特殊处理，调用fuse前应保证它们不存在
  pub fn fuse(&self, i: u32) -> P<Comp> {
    debug_assert!(i + 1 < self.loop_dim());
    // min1 <= i1 < max1, min2 <= i2 < max2, 令j = (i1 - min1) * (max2 - min2) + (i2 - min2)
    // 则i1 = floor(j / (max2 - min2)) + min1, i2 = j % (max2 - min2) + min2
    let Extent(min1, _, _) = self.extent(i);
    let Extent(min2, _, extent2) = self.extent(i + 1);
    let (n, i) = (self.sch_dim(), i * 2 + 1);
    let s = format!("{{[{}]->[{}]:i{n}=(i{i1}-{min1})*{extent2}+(i{i2}-{min2})&&i{i1}=floor(i{n}/{extent2})+{min1}&&i{i2}=i{n}%{extent2}+{min2}}}\0", i0_in(n),
      comma_sep((0..n - 2).map(|x| fn2display(move |f|
        write!(f, "i{}", if x < i || x == i + 1 { x } else if x == i { n } else { x + 2 })
      ))),
      n = n, i1 = i, i2 = i + 2, min1 = min1, min2 = min2, extent2 = extent2);
    debug!("fuse: {}", s);
    self.apply_sch_raw(&s)
  }

  // 交换循环层次i和j
  // reorder和reorder_n都不会处理它们的tag，调用前应保证tag不存在
  pub fn reorder(&self, i: u32, j: u32) -> P<Comp> {
    self.reorder_n(&[(i, j), (j, i)])
  }

  // 对于map中每个元素(old, new)，将循环层次old用new代替
  // 用户须保证map是一一映射，即old不重复，new不重复，且old构成的集合与new构成的集合相等
  pub fn reorder_n(&self, map: &[(u32, u32)]) -> P<Comp> {
    if cfg!(debug_assertions) { // 非debug模式下整个块不执行，只用debug_assert编译器可能没法优化掉下面的东西
      let (mut old, mut new) = (HashSet::default(), HashSet::default());
      for &(o, n) in map {
        assert!(o < self.loop_dim() && n < self.loop_dim());
        assert!(old.insert(o) && new.insert(n)); // 不重复
      }
      assert_eq!(old, new);
    }
    let n = self.sch_dim();
    let s = format!("{{[{}]->[{}]}}\0", i0_in(n),
      comma_sep((0..n).map(|x| fn2display(move |f|
        write!(f, "i{}", map.iter().find(|&&(old, _)| x == old * 2 + 1)
          .map(|&(_, new)| new * 2 + 1).unwrap_or(x))
      ))));
    debug!("reorder_n: {}", s);
    self.apply_sch_raw(&s)
  }

  pub fn skew(&self, i: u32, j: u32, factor: u32) -> P<Comp> {
    debug_assert!(i < j);
    debug_assert!(i < self.loop_dim() && j < self.loop_dim());
    let (n, i, j) = (self.sch_dim(), i * 2 + 1, j * 2 + 1);
    let s = format!("{{[{}]->[{}]:i{j1}={f}*i{i}+i{j}}}\0", i0_in(n),
      comma_sep((0..n).map(|x| fn2display(move |f|
        write!(f, "i{}", if x < j { x } else if x == j { n } else { x - 1 })))),
      j1 = n, f = factor, i = i, j = j);
    debug!("skew: {}", s);
    self.apply_sch_raw(&s)
  }

  pub fn shift(&self, i: u32, n: i32) -> P<Comp> {
    debug_assert!(i < self.loop_dim());
    // 设置i_out = i_in + n
    self.schedule.write(map_set_eq(self.schedule.read(), i * 2 + 1, -1, n));
    debug!("shift: {}", self.schedule);
    self.p()
  }

  pub fn separate(&self, i: u32, factor: u32) -> Option<P<Comp>> {
    debug_assert!(i < self.loop_dim());
    let Bound(zero, min, max) = self.extract_bound(i);
    let (n, pos) = (self.sch_dim(), i * 2 + 1);
    let factor = self.ctx.val_int_from_ui(factor as _)?;
    let i_aff = zero.set_coefficient_si(DimType::In, pos as _, 1)?.pw_aff_from_aff()?;
    // sep = min + floor((max - min) / factor) * factor，划分成i < sep和sep <= i两个区间
    // ISL中pw_aff_scale_down_val是精确除法，结果保存成类似3 / 2的形式
    // pw_aff_div也是精确除法，它最终会调用scale_down_val
    // pw_aff_tdiv_q是r >= 0 ? pw_aff_floor(pw_aff_div(l, r)) : pw_aff_ceil(pw_aff_div(l, r))
    // 这里假定max - min >= 0成立，直接调用pw_aff_floor和pw_aff_scale_down_val即可
    let sep = min.copy()?.add(max.sub(min)?.scale_down_val(factor.copy()?)?.floor()?.scale_val(factor)?)?;
    debug!("separate: sep expr = {}", sep);
    let ge = self.schedule.copy()?.intersect_range(i_aff.copy()?.ge_set(sep.copy()?)?
      .add_dims(DimType::Out, n - pos - 1)?)?;
    // 被separate出去的计算迭代域是空集则不生成Comp
    if ge.is_empty()? {
      debug!("separate: sep comp has empty domain");
      return None;
    }
    let name = format!("_sep{}_{}\0", self.func.new_comp_id(), self.name());
    let c = box Comp {
      ctx: self.ctx,
      func: self.func,
      orig_name: self.name().into(),
      expr: self.expr.clone(),
      cond: self.cond.clone(),
      schedule: ge.set_tuple_name(DimType::In, cstr(&name))?,
      store: if let Some(x) = &self.store { Some(x.copy()?) } else { None },
      pred: None,
      succ: Vec::new(),
      tags: self.tags.clone(),
      inline: self.inline,
    };
    // 上面的所有操作都只修改局部变量(修改Func::comp_cnt可以接受)，因此错误可以恢复，从这里开始的错误不能恢复
    c.after(self.p(), i);
    self.schedule.write(self.schedule.read().intersect_range(i_aff.lt_set(sep).unwrap()
      .add_dims(DimType::Out, n - pos - 1).unwrap()).unwrap());
    debug!("separate: created sep comp {}, sch = {}", c.name(), c.schedule);
    let ret = c.as_ref().p();
    self.p().func.comps.push(c);
    Some(ret)
  }

  // 提取循环层次i的迭代范围，Bound的意义见Bound自身的注释
  pub fn extract_bound(&self, i: u32) -> Bound {
    debug_assert!(i < self.loop_dim());
    let (n, pos) = (self.sch_dim(), i * 2 + 1);
    // 通过投影去掉i之后的维度，用i前面的维度表示i的上下界
    let dom = self.schedule().project_out(DimType::Set, pos + 1, n - pos - 1)?;
    debug!("extract_bound: projected dom = {}", dom);
    // Div维度用来实现类似floor(x / 2) + floor(y / 3) + 4 <= 0的约束，后面提取上下界的时候处理不了它，所以要求它不存在
    debug_assert_eq!(dom.dim(DimType::Div), 0);
    // 只支持在一个BasicSet组成的Set上提取bound，否则情况太复杂
    debug_assert_eq!(dom.n_basic_set(), 1);
    let csts = dom.get_basic_set_list()?.get_basic_set(0)?.get_constraint_list()?;
    let (mut min, mut max) = (None, None);
    for i in 0..csts.n_constraint() {
      let cst = csts.get_constraint(i)?;
      let k = cst.get_coefficient_val(DimType::Out, pos as _)?.get_num_si();
      if k != 0 {
        let aff = cst.get_bound(DimType::Out, pos as _)?.pw_aff_from_aff()?;
        // 如果这个维度可以用前面的维度用等式表示，那它就只有这一个取值，即min/max都是它
        if cst.is_equality()? {
          debug!("extract_bound: dim has equality constraint: {}", aff);
          min = Some(aff.copy()?);
          max = Some(aff);
          break;
        }
        // ISL中inequality约束是... >= 0
        if k > 0 {
          min = Some(if let Some(x) = min { aff.max(x)? } else { aff });
        } else {
          max = Some(if let Some(x) = max { aff.min(x)? } else { aff });
        }
      }
    }
    let (min, max) = (min?.intersect_domain(dom.copy()?)?.coalesce()?, max?.intersect_domain(dom.copy()?)?.coalesce()?);
    let zero = dom.get_space()?.aff_zero_on_domain_space()?;
    let one = zero.copy()?.set_constant_si(1)?.pw_aff_from_aff()?;
    let max = max.add(one)?; // max += 1，循环范围是min <= i < max
    debug!("extract_bound: min = {}, max = {}", min, max);
    Bound(zero, min, max)
  }

  pub fn extent(&self, i: u32) -> Extent {
    let (sch, pos) = (self.schedule(), (i * 2 + 1) as _);
    let (min, max) = (sch.copy()?.dim_min_val(pos)?, sch.dim_max_val(pos)?);
    let extent = max.copy()?.sub(min.copy()?)?.add(self.ctx.val_one()?)?;
    Extent(min, max, extent)
  }

  pub fn apply_sch_raw(&self, s: &str) -> P<Comp> {
    debug_assert!(s.ends_with('\0'));
    let t = self.ctx.map_read_from_str(cstr(&s))?.align_params(self.schedule.get_space()?)?;
    self.schedule.write(self.schedule.read().apply_range(t)?.coalesce()?.detect_equalities()?);
    self.p()
  }
}

// 这个trait只是为了给Vec<P<Comp>>定义separate函数
pub trait Separate {
  fn separate(&self, i: u32, factor: u32);
}

impl Separate for Vec<P<Comp>> {
  // 对自身已有元素都执行Comp::separate(i, factor)，把生成的新Comp加到自身中(虽然参数是&self，但会修改自身)
  fn separate(&self, i: u32, factor: u32) {
    let mut s = self.p();
    for idx in 0..s.len() { // 不能用迭代器
      if let Some(x) = s[idx].separate(i, factor) { s.push(x); }
    }
  }
}

#[derive(Debug)]
pub struct CacheCfg {
  pub size: u32,
  // cache的目标下标
  pub dst: Expr,
  // cache的源下标
  pub src: Expr,
  // user的新下标，必须用user的domain表示
  pub access: Expr,
}

#[derive(Debug)]
pub struct CacheInfo {
  pub sync: Option<(P<Comp>, P<Comp>)>,
  pub copy: P<Comp>,
  pub buf: P<Buf>,
  pub buf_load: P<Comp>,
}

impl_try!(CacheInfo);

impl Comp {
  // 如果用来添加GPU相关的tag，需要保证tag的几个维度是完美嵌套的
  // 例如for (i) { for (j) { C1 }  C2 }，如果tag(i, GPUBlockX)，tag(j, GPUThreadX)
  // 生成的kernel会执行C1和C2，等价于for (i) { for (j) { C1 C2 } }，语义是错误的
  pub fn tag(&self, i: u32, tag: DimTag) -> P<Comp> {
    debug_assert!(i < self.loop_dim());
    let i = i as usize;
    let tags = &mut self.p().tags;
    if tags.len() <= i { tags.resize(i + 1, None); }
    tags[i] = Some(tag);
    self.p()
  }

  pub fn tags(&self, i: impl IntoIterator<Item=u32>, tag: DimTag) -> P<Comp> {
    for i in i { self.tag(i, tag); }
    self.p()
  }

  // identity store，即C[i, j, k, ...]保存在buf[i, j, k, ...]
  pub fn store(&self, buf: P<Buf>) -> P<Comp> {
    debug_assert_eq!(self.orig_dim(), buf.sizes.len() as _);
    let store = self.domain().identity()?.reset_tuple_id(DimType::In)?
      .set_tuple_name(DimType::Out, cstr(&format!("{}\0", buf.name)))?;
    debug!("store: {}", store);
    self.p().store = Some(store);
    self.p()
  }

  // 检查e的孩子中Iter(x)的x都小于loop_dim
  pub fn check_iter(&self, e: &Expr) {
    e.visit(&mut move |e| if let &Iter(_, x) = e { assert!(x < self.loop_dim()); })
  }

  pub fn store_at(&self, buf: P<Buf>, idx: Box<[Expr]>) -> P<Comp> {
    if cfg!(debug_assertions) {
      assert_eq!(idx.len(), buf.sizes.len());
      for e in idx.iter() { self.check_iter(e); }
    }
    let mut params = HashSet::default();
    for idx in idx.iter() { idx.collect_params(&mut params); }
    let s = format!("[{}]->{{[{}]->{}[{}]}}\0", comma_sep(params.iter().map(|c| c.name())), i0_in(self.orig_dim()), buf.name, comma_sep(idx.iter()));
    debug!("store_at: {}", s);
    self.p().store = Some(self.ctx.map_read_from_str(cstr(&s))?);
    self.p()
  }

  // 会将self.expr中所有对src的访问下标都替换成cfg.access的列表，不管原来的下标是什么；若原来有src[i][j]和src[j][i]，会导致错误的结果
  pub fn cache(&self, src: P<Comp>, i: u32, loc: BufLoc, extent: u32, cond: Option<Expr>, cfg: &mut [CacheCfg]) -> CacheInfo {
    // 不使用CacheCfg的Debug来输出，因为希望用Display输出Expr
    debug!("cache: extent = {}, cond = {}, cfg = [{}]", extent, opt(&cond),
      comma_sep(cfg.iter().map(|CacheCfg { size, dst, src, access }|
        fn2display(move |f| write!(f, "CacheCfg {{ size: {}, dst: {}, src: {}, access: {} }}", size, dst, src, access)))));
    if cfg!(debug_assertions) {
      assert_eq!(src.orig_dim(), cfg.len() as _);
      assert!(i < self.loop_dim());
      // dst和src不必满足self的迭代域，copy的comp_raw和store_at中会检查
      for c in cfg.iter() { self.check_iter(&c.access); }
    }
    let (f, src) = (self.func, src.p());
    let name = format!("_cache{}_{}\0", f.new_buf_id(), src.name());
    let buf = f.buf(&name[..name.len() - 1], src.expr.ty(), Temp,
      cfg.iter().map(|c| c.size.expr()).collect()).set_loc(loc);
    buf.alloc_at(self.p(), i);
    let i = i + 1;
    let mut dom = project_static_dim(self.schedule());
    let n = dom.n_dim() as u32;
    dom = dom.project_out(DimType::Set, i, n - i)?;
    let sync_dom = dom.copy()?; // 执行Sync的循环范围，与下面的buf.alloc_at的Alloc/Free的范围一样
    dom = dom.add_dims(DimType::Set, 1)?; // 添加一个新循环copy_iter，下面设置循环范围
    let cst = dom.get_space()?.local_space_from_space()?.constraint_alloc_inequality()?
      .set_coefficient_si(DimType::Set, i as _, 1)?;  // copy_iter >= 0
    dom = dom.add_constraint(cst)?;
    let cst = dom.get_space()?.local_space_from_space()?.constraint_alloc_inequality()?
      .set_coefficient_si(DimType::Set, i as _, -1)?
      .set_constant_si((extent - 1) as _)?; // copy_iter < extent
    dom = dom.add_constraint(cst)?;
    dom = dom.set_tuple_name(cstr(&name))?;
    debug!("cache: copy dom = {}", dom);
    let copy = Access(src, cfg.iter_mut().map(|c| c.src.replace0()).collect());
    debug!("cache: copy expr = {}", copy);
    let copy = f.comp_raw(dom, copy)
      .store_at(buf, cfg.iter_mut().map(|c| c.dst.replace0()).collect()).set_cond(cond);
    let sync = if loc == Shared {
      let sync1 = f.comp_raw(sync_dom.copy()?.set_tuple_name(cstr(&format!("_sync1{}", name)))?, syncthreads());
      let sync2 = f.comp_raw(sync_dom.set_tuple_name(cstr(&format!("_sync2{}", name)))?, syncthreads());
      self.after_between_pred(sync2, i).after_between_pred(copy, i).after_between_pred(sync1, i);
      Some((sync1.p(), sync2.p()))
    } else {
      self.after_between_pred(copy, i);
      None
    };
    let buf_load = buf.load().p();
    self.p().expr.visit_mut(&mut move |e| match e {
      Access(c, idx) => if *c == src {
        debug_assert_eq!(idx.len(), cfg.len());
        *c = buf_load;
        *idx = cfg.iter().map(|c| c.access.clone()).collect();
      }
      _ => {}
    });
    CacheInfo { sync, copy: copy.p(), buf: buf.p(), buf_load }
  }

  pub fn cache_identity(&self, src: P<Comp>, i: u32, loc: BufLoc) -> CacheInfo {
    debug_assert!(i < self.loop_dim());
    // 记录绑定到GPU thread的循环变量的位置和extent
    let mut threads = Vec::new();
    if loc == Shared {
      for (i, &tag) in self.tags.iter().enumerate() {
        if Some(GPUThreadX) <= tag && tag <= Some(GPUThreadZ) {
          threads.push((i as u32, self.extent(i as _).2.get_num_si() as u32));
        }
      }
    }
    debug!("cache_identity: threads = {:?}", threads);
    let in_threads = |i| threads.iter().find(|x| x.0 == i).is_some();
    let mut idx: &[_] = &[]; // 如果没有类型标注，默认是&[_; 0]，和下面不一致
    self.expr.visit(&mut |e| match e { Access(c, x) if *c == src.p() => { idx = x; } _ => {} });
    debug_assert!(!idx.is_empty(), "src not in self.expr");
    let n_idx = idx.len();
    let access = self.access("", idx);
    debug!("cache_identity: access = {}", access);
    let mut access_idx = self.func.build_access_idx(self.schedule().identity()?,
      self.func.iter_list(access.dim(DimType::In) as _, true), *access);
    debug!("cache_identity: access_idx = [{}]", comma_sep(access_idx.iter()));
    // access_mut将除了thread变量外，所有包裹i的变量置0，只留下thread变量+内层变量，这样下标的范围就是需要cache的范围
    let mut access_mut = access.set_tuple_name(DimType::In, None)?.set_tuple_name(DimType::Out, None)?;
    let mut affs = access_mut.coalesce()?.pw_multi_aff_from_map()?;
    for j in 0..affs.dim(DimType::Out) {
      let aff = affs.get_pw_aff(j)?;
      // 这两个函数用于从空开始逐个piece构建PwAff，但它们在ISL里没有公开，我不理解为什么，而且也没有别的方法了
      extern "C" {
        fn isl_pw_aff_alloc_size(space: Space, n: i32) -> Option<PwAff>;
        fn isl_pw_aff_add_piece(pw: PwAff, set: Set, aff: Aff) -> Option<PwAff>;
      }
      let aff1 = unsafe { isl_pw_aff_alloc_size(aff.get_space()?, aff.n_piece())? };
      aff.foreach_piece(&mut |set, mut aff| {
        for i in 0..=i {
          if !in_threads(i) { aff = aff.set_coefficient_si(DimType::In, (2 * i + 1) as _, 0)?; }
        }
        for &dim in &[DimType::Div, DimType::Param] {
          for i in 0..aff.dim(dim) { aff = aff.set_coefficient_si(dim, i, 0)?; }
        }
        unsafe { aff1.write(isl_pw_aff_add_piece(aff1.read(), set, aff)?); }
        Stat::Ok
      })?;
      affs = affs.set_pw_aff(j as _, aff1)?;
    }
    access_mut = affs.map_from_pw_multi_aff()?;
    debug!("cache_identity: access_mut = {}", access_mut);
    let mut copy_map = format!("{}{{", self.params());
    let mut idx_map = vec![HashMap::default(); n_idx];
    access_mut.copy()?.range()?
      .remove_dims(DimType::Param, 0, access_mut.dim(DimType::Param) as _)?
      .foreach_point(&mut |p| {
        let _ = copy_map.write_str("[");
        for (i, idx_map) in idx_map.iter_mut().enumerate() {
          let idx = p.get_coordinate_val(DimType::Set, i as _)?.get_num_si();
          let id = idx_map.len();
          let id = *idx_map.entry(idx).or_insert(id);
          let _ = write!(copy_map, "{}{}", if i == 0 { "" } else { "," }, id);
        }
        let _ = copy_map.write_str("]->[");
        for i in 0..idx_map.len() {
          let idx = p.get_coordinate_val(DimType::Set, i as _)?.get_num_si();
          let _ = write!(copy_map, "{}{}", if i == 0 { "" } else { "," }, idx);
        }
        let _ = copy_map.write_str("];");
        Stat::Ok
      })?;
    if copy_map.ends_with(';') { copy_map.pop(); }
    copy_map.push_str("}\0");
    let copy_map = self.ctx.map_read_from_str(cstr(&copy_map))?.coalesce()?;
    debug!("cache_identity: copy_map = {}", copy_map);
    let copy_dom = copy_map.copy()?.domain()?;
    let mut copy_idx = self.func.build_access_idx(copy_dom.copy()?.identity()?,
      self.func.iter_list(n_idx as _, false), *copy_map);
    debug!("cache_identity: copy_idx = [{}]", comma_sep(copy_idx.iter()));
    let mut access_mut_idx = self.func.build_access_idx(self.schedule.copy()?.reverse()?,
      self.func.iter_list(self.orig_dim(), false), *access_mut.copy()?.apply_range(copy_map.copy()?.reverse()?)?);
    debug!("cache_identity: access_mut_idx = [{}]", comma_sep(access_mut_idx.iter()));
    debug_assert!(access_idx.len() == n_idx && copy_idx.len() == n_idx && access_mut_idx.len() == n_idx);
    // 这里只是做identity拷贝，是否用/怎样用thread变量对后续访问没有影响
    let mut copy_iter = iter(i + 1);
    for &(i, extent) in &threads { copy_iter = copy_iter.mul(extent).add(iter(i)); }
    let mut extent = 1;
    let mut copy_iters = Vec::with_capacity(n_idx);
    for i in (0..n_idx).rev() {
      let (min, max) = (copy_dom.copy()?.dim_min_val(i as _)?, copy_dom.copy()?.dim_max_val(i as _)?);
      debug_assert!(min.is_zero()?);
      let size = max.add(self.ctx.val_one()?)?.get_num_si() as u32;
      copy_iters.push((copy_iter.clone().div(extent).rem(size), size));
      extent *= size;
    }
    copy_iters.reverse();
    debug!("cache_identity: copy_iters = [{}]", comma_sep(copy_iters.iter().map(|x| &x.0)));
    extent /= threads.iter().map(|x| x.1).product::<u32>();
    let mut cond = self.func.build_cond(identity_schedule(copy_map.range()?),
      self.func.iter_list(n_idx as _, true), *access_mut.range()?);
    debug!("cache_identity: cond = {}", cond);
    let mut cfg = Vec::with_capacity(n_idx);
    for ((((copy_iter, size), mut access_idx), copy_idx), access_mut_idx) in copy_iters.iter()
      .zip(access_idx.drain(..)).zip(copy_idx.iter_mut()).zip(access_mut_idx.drain(..)) {
      // 一般不对Expr做任何简化，但这里要把0 % x简化成0，因为0 % x不符合ISL的语法(而(0) % x符合，这太离谱了)
      // 这里如果做后序遍历会更方便，但其他地方都是先序，没有什么必要专门写个后序遍历的函数
      access_idx.visit_mut(&mut move |e| match e {
        Iter(ty, x) | Binary(BinOp::Rem, box [Iter(ty, x), _]) if *x > i || in_threads(*x) => { *e = Val(*ty, 0); } _ => {}
      });
      copy_idx.visit_mut(&mut |e| if let Iter(_, x) = e {
        *e = copy_iters[*x as usize].0.clone();
        false
      } else { true });
      let src = copy_idx.clone().add(access_idx);
      cfg.push(CacheCfg { size: *size, dst: copy_iter.clone(), src, access: access_mut_idx });
    }
    cond.replace_iter(&copy_idx);
    self.cache(src, i, loc, extent, Some(cond), &mut cfg)
  }
}

impl Comp {
  // at的意义是在包围at层循环的static dim上，self在after之后，at取值范围是0..=循环层数
  // A.after(B, i).after(C, j)的链式调用，语义是A在B后，B在C后
  // 要求self没有前驱；如果other在at处已经有后继x，则把self插在other和x之间
  pub fn after(&self, mut other: P<Comp>, i: u32) -> P<Comp> {
    debug!("after: setting {} after {}", self.name(), other.name());
    debug_assert!(self.pred.is_none());
    let i = i as usize;
    self.p().pred = Some(other);
    if other.succ.len() <= i { other.succ.resize(i + 1, None); }
    if let Some(mut x) = other.succ[i] {
      x.pred = Some(self.p());
      if self.succ.len() <= i { self.p().succ.resize(i + 1, None); }
      self.p().succ[i] = Some(x);
    }
    other.succ[i] = Some(self.p());
    other
  }

  // 与after区别在于：如果self有前驱p，则要求other没有前驱，并将other插在self和p间
  pub fn after_between_pred(&self, mut other: P<Comp>, i: u32) -> P<Comp> {
    if let Some(mut p) = self.pred {
      debug_assert!(other.pred.is_none());
      other.pred = Some(p);
      *p.succ.iter_mut().find(|x| **x == Some(self.p())).unwrap() = Some(other);
      self.p().pred = None;
    }
    self.after(other, i)
  }

  // A.before(B, i).before(C, j)的链式调用，语义是A在B前，B在C前
  pub fn before(&self, other: P<Comp>, i: u32) -> P<Comp> {
    other.after(self.p(), i);
    other
  }

  pub fn before_between_pred(&self, other: P<Comp>, i: u32) -> P<Comp> {
    other.after_between_pred(self.p(), i);
    other
  }

  // 用schedule中的static dim来实现after的逻辑。可以直接使用它，但多个Comp间的关系不一定可以保留
  // 例如A.after(B, i); B.after(C, i); 最终会正确生成A在B后，B在C后
  // 但A.after_raw(B, i); B.after_raw(C, i); 假设一开始static dim都是0，则最终A和B的都是1，分不出先后
  // 此外还须保证事先调用`Func::align_schedule`
  pub fn after_raw(&self, other: P<Comp>, i: u32) -> Unit {
    debug!("after_raw: i = {}, self = {}, other = {}", i, self.schedule, other.schedule);
    // debug_assert_eq!(self.sch_dim(), other.sch_dim());
    debug_assert!(i <= self.loop_dim() && i <= other.loop_dim());
    // 理论上只需要将other.schedule中pos处的constraint + 1即可，但是ISL不提供这样的操作，必须重新构建
    for pos in (0..self.sch_dim().min(other.sch_dim())).step_by(2) {
      // 获取map中对应位置的static dim
      let mut order = None;
      // Map由多个BasicMap的并集组成，理论上不同的BasicMap的同一个out dim处可以有不同的constraint
      // 但在这里不会发生，这里获取的是static dim，应该保证所有BasicMap的static dim都是相同的
      other.schedule.foreach_basic_map(&mut |m| {
        let csts = m.get_constraint_list()?;
        for i in 0..csts.n_constraint() {
          let cst = csts.get_constraint(i)?;
          let k = cst.get_coefficient_val(DimType::Out, pos as _)?;
          if k.is_one()? {
            let val = -cst.get_constant_val()?.get_num_si() as i32;
            debug_assert!(order == None || order == Some(val));
            order = Some(val);
          }
        }
        Stat::Ok
      })?;
      // 在other的对应位置上static dim上 + 1，其余不变
      let order = order? + (pos == i * 2) as i32;
      self.schedule.write(map_set_eq(self.schedule.read(), pos, 0, order));
    }
    debug!("after_raw: {}", self.schedule);
    Unit
  }
}

// 输出i0, ..., i{n-1}
pub(crate) fn i0_in(n: u32) -> impl Display {
  comma_sep((0..n).map(|i| fn2display(move |f| write!(f, "i{}", i))))
}

// 在pos处添加约束: k_in * i_in + i_out + val == 0; 如果k_in传0，就是设置out维度中pos处值为val
// 如果已经存在对这个位置约束则不能使用它
pub(crate) fn map_add_constraint(map: Map, pos: u32, k_in: i32, val: i32) -> Map {
  let pos = pos as _;
  let mut cst = map.get_space()?.local_space_from_space()?.constraint_alloc_equality()?;
  // 我的应用中out dim总是多于in dim，所以只需要检查pos是否在in dim范围内
  if pos < cst.dim(DimType::In) {
    cst = cst.set_coefficient_si(DimType::In, pos, k_in)?;
  }
  cst = cst.set_coefficient_si(DimType::Out, pos, 1)?.set_constant_si(-val)?;
  map.add_constraint(cst)?
}

// k_in，val语义和map_add_constraint中相同
// 可以处理已经存在对这个位置约束的情形，但比`map_add_constraint`开销更大
pub(crate) fn map_set_eq(map: Map, pos: u32, k_in: i32, val: i32) -> Map {
  let mut sp = map.get_space()?;
  let (n_in, n_out) = (sp.dim(DimType::In) as u32, sp.dim(DimType::Out) as u32);
  sp = sp.add_dims(DimType::In, n_out - n_in)?;
  let mut trans = sp.map_universe()?;
  for i in 0..n_out {
    // 除pos外其他维度，包括static和dynamic dim，都是恒等映射
    let (k_in, val) = if i == pos { (k_in, val) } else { (-1, 0) };
    trans = map_add_constraint(trans, i, k_in, val);
  }
  map.apply_range(trans)?
}

// 返回map的in_dim名字与dom名字相同，out_dim名字为空
// out_dim = in_dim * 2 + 1，在0，2，4...，2 * in_dim位置为0，在1，3，5...，2 * in_dim - 1位置与in_dim相应位置相等
pub(crate) fn identity_schedule(domain: Set) -> Map {
  let in_dim = domain.n_dim() as u32;
  // set.identity()返回的out dim名字和in dim名字都是set名字
  // 注意ISL中名字是空串和空指针被认为是不相等的，用reset_tuple_id或set_tuple_name传None将名字赋成空指针
  let mut sch = domain.identity()?.reset_tuple_id(DimType::Out)?;
  for i in 0..=in_dim { // 在0，2，4...，2 * in_dim下标处插入0
    sch = sch.insert_dims(DimType::Out, 2 * i, 1)?;
    sch = map_add_constraint(sch, 2 * i, 0, 0);
  }
  sch
}

// set表示是一个调度后的schedule，偶数下标处都是static dim，这个函数去掉这些static dim，让它可以用作未经调度的schedule
pub(crate) fn project_static_dim(mut set: Set) -> Set {
  let n = set.n_dim() as u32;
  debug_assert_eq!(n % 2, 1);
  // static dim是0, 2, ..., 2 * (n / 2)，但去掉0后，下一个就变成1，去掉1后，下一个就变成2...故循环删除0, 1, ..., n / 2
  for i in 0..n / 2 { set = set.project_out(DimType::Set, i, 1)?; }
  set
}

// 对map的range做project_static_dim
fn project_static_dim_map(mut map: Map) -> Map {
  let n = map.dim(DimType::Out) as u32;
  debug_assert_eq!(n % 2, 1);
  for i in 0..n / 2 { map = map.project_out(DimType::Out, i, 1)?; }
  map
}
