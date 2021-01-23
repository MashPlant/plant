use crate::*;

#[derive(Debug)]
pub struct Comp {
  pub ctx: CtxRef,
  pub func: P<Func>,
  pub expr: Expr,
  // schedule将表示原始循环迭代范围的set映射到调度后的循环迭代范围的set，同时还包含循环间的顺序关系
  // in dim名字是Comp的名字，out dim名字是空的，#out = #in * 2 + 1
  // out dim分为static和dynamic dim，从循环层次i到包围它的static dim：i * 2；从循环层次i到它的dynamic dim：i * 2 + 1
  // isl_basic_set表示可以用一组仿射约束的交集定义的集合，isl_set表示一组isl_basic_set的并集
  // 这里需要用到isl_set，例如i <= max(x, y)这样的约束无法用交集表示，必须是i <= x or i <= y
  pub schedule: Map,
  pub store: Option<Map>,
  pub pred: Option<P<Comp>>,
  pub succ: HashMap<P<Comp>, u32>,
  pub dim_tags: Vec<Option<DimTag>>,
}

impl_try!(&Comp);

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum DimTag { Parallel, GPUBlockX, GPUBlockY, GPUBlockZ, GPUThreadX, GPUThreadY, GPUThreadZ }

// 将Comp传给接受impl IntoExpr的地方时，是将它作为Param表达式，而非Access表达式
// 用法区别请看Comp::as_param的注释
impl IntoExpr for &Comp {
  fn expr(self) -> Expr { self.as_param() }
}

// 用来实现类似重载的效果，为Func::comp提供“默认参数”
pub trait CompBuilder {
  fn comp(self, f: &Func) -> &Comp;
}

impl<E1: IntoExpr, E2: IntoExpr, E3: IntoExpr> CompBuilder for (&str, &[(E1, E2)], E3) {
  fn comp(self, f: &Func) -> &Comp { f.comp(self.0, self.1, self.2) }
}

impl<E1: IntoExpr> CompBuilder for (&str, E1) {
  fn comp(self, f: &Func) -> &Comp { f.comp(self.0, EMPTY2, self.1) }
}

impl<E1: IntoExpr, E2: IntoExpr, E3: IntoExpr> CompBuilder for (&[(E1, E2)], E3) {
  fn comp(self, f: &Func) -> &Comp {
    let e = self.1.expr();
    f.comp(&f.auto_comp_name(&e), self.0, e)
  }
}

impl<E1: IntoExpr> CompBuilder for E1 {
  fn comp(self, f: &Func) -> &Comp {
    let e = self.expr();
    f.comp(&f.auto_comp_name(&e), EMPTY2, e)
  }
}

impl Func {
  pub fn comp(&self, name: &str, ranges: &[(impl IntoExpr, impl IntoExpr)], expr: impl IntoExpr) -> &Comp {
    // 很多时候调用方可以提供&[(Expr, Expr)]，这里的拷贝是多余的，但这点浪费可以忽略
    let ranges = ranges.iter().map(|(lb, ub)| (lb.clone_expr(), ub.clone_expr())).collect::<Vec<_>>();
    let expr = expr.expr();
    let mut params = HashSet::default();
    // 收集ranges，expr中的所有Param
    let ref mut vis = |e: &Expr| if let &Param(x) = e { params.insert(x); };
    for (lb, ub) in &ranges {
      lb.visit(vis);
      ub.visit(vis);
    }
    expr.visit(vis);
    let s = format!("[{}] -> {{ {}{}: {} }}\0", comma_sep(params.iter().map(|c| c.name())),
      name, i0_in(ranges.len() as _),
      sep(ranges.iter().enumerate().map(|(i, (lb, ub))| fn2display(move |f|
        write!(f, "{} <= i{} < {}", lb, i, ub))), " and "));
    debug!("comp: domain = {}", s);
    self.comp_raw(self.ctx.set_read_from_str(cstr(&s))?, expr)
  }

  pub fn comp_raw(&self, domain: Set, expr: Expr) -> &Comp {
    // set_read_from_str生成的set可能有冗余，例如为i <= min(x, y)生成两个BasicSet，其实一个就可以表示，coalesce就是试图合并BasicSet
    let schedule = identity_schedule(domain.coalesce()?);
    debug!("comp_raw: initial identity schedule = {}", schedule);
    let comp = box Comp { ctx: *self.ctx, func: self.into(), expr, schedule, store: None, pred: None, succ: HashMap::default(), dim_tags: Vec::new() };
    debug_assert!(self.find_comp(comp.name()).is_none()); // 不允许相同名字的Comp
    let ret = R::new(&*comp);
    self.p().comps.push(comp);
    ret.get()
  }
}

impl Comp {
  // 返回的字符串来源于cstr，[len()]位置是\0，可以传入ISL的接口，更直接的是使用name_cstr
  pub fn name(&self) -> &str { self.name_cstr().unwrap().as_str() }

  pub fn name_cstr(&self) -> Option<CStr> { self.schedule.get_tuple_name(DimType::In) }

  pub fn n_dim(&self) -> u32 { self.schedule.dim(DimType::In) as _ }

  pub fn sch_dim(&self) -> u32 { self.schedule.dim(DimType::Out) as _ }

  // 表示原始循环迭代范围的Set
  pub fn domain(&self) -> Set { self.schedule.copy()?.domain()? }

  // 表示调度后循环迭代范围的Set，注意需要设置名字，schedule的out dim名字为空
  pub fn schedule(&self) -> Set {
    self.schedule.copy()?.range()?.set_tuple_name(self.name_cstr())?
  }

  // 输出[逗号分隔的params列表]
  pub fn params<'a>(&'a self) -> impl Display + 'a {
    fn2display(move |f| write!(f, "[{}]", comma_sep((0..self.schedule.dim(DimType::Param) as u32)
      .map(|i| self.schedule.get_dim_name(DimType::Param, i).unwrap()))))
  }

  pub fn set_expr(&self, expr: Expr) { self.p().expr = expr; }

  // 将自身作为一个Param表达式，Expr中直接通过它的名字使用它
  // 当且仅当它没有store的时候，生成的代码会在计算发生的地方定义一个对应名字的局部变量，所以它必须没有store
  // 如果Comp的计算结果用于其他Comp中的循环范围，Access下标，则必须用as_param
  // 如果只是用于普通运算，可以用as_param或at，as_param会往domain/schedule中引入一个参数，应该没有什么好处
  pub fn as_param(&self) -> Expr { Param(self.into()) }

  pub fn at(&self, idx: &[impl IntoExpr]) -> Expr {
    debug_assert_eq!(idx.len() as u32, self.n_dim());
    Access(self.into(), idx.iter().map(|e| e.clone_expr()).collect())
  }

  pub fn at_inline(&self, idx: &[impl IntoExpr]) -> Expr {
    debug_assert_eq!(idx.len() as u32, self.n_dim());
    let mut expr = self.expr.clone();
    expr.visit_mut(&mut |e| if let Iter(_, x) = &e { *e = idx[*x as usize].clone_expr(); });
    expr
  }
}

// Bound(zero, min, max)，意义是循环i及外层循环的空间上的0，i的最小值，最大值 + 1 (min <= i < max)
// 最小值和最大值用包围i的循环变量表示
pub struct Bound(Aff, PwAff, PwAff);
impl_try!(Bound);

impl Comp {
  pub fn tile(&self, i: u32, j: u32, tile_i: u32, tile_j: u32) -> &Comp {
    debug_assert!(i < j);
    self.split(i, tile_i).split(j + 1, tile_j).reorder(i + 1, j + 1)
  }

  pub fn tile_3(&self, i: u32, j: u32, k: u32, tile_i: u32, tile_j: u32, tile_k: u32) -> &Comp {
    debug_assert!(i < j && j < k);
    self.split(i, tile_i).split(j + 1, tile_j).split(k + 2, tile_k)
      // i0 i1 j0 j1 k0 k1 -> i0 j0 k0 i1 j1 k1
      .reorder_n(&[(i + 1, j + 1), (j + 1, k + 2), (j + 2, i + 1), (k + 2, j + 2)])
  }

  pub fn split(&self, i: u32, factor: u32) -> &Comp {
    let (n, i) = (self.sch_dim(), i * 2 + 1);
    let s = format!("{{ {} -> [{}]: i{i0} = floor(i{i} / {f}) and i{i1} = i{i} % {f} }}\0", i0_in(n),
      comma_sep((0..n + 2).map(|x| fn2display(move |f|
        if x == i + 1 { f.write_str("0") } else {
          write!(f, "i{}", if x < i { x } else if x == i { n } else if x == i + 2 { n + 1 } else { x - 2 })
        }))),
      i0 = n, i1 = n + 1, i = i, f = factor);
    debug!("split: {}", s);
    self.apply_sch_raw(&s)
  }

  // 交换循环层次i和j
  pub fn reorder(&self, i: u32, j: u32) -> &Comp {
    self.reorder_n(&[(i, j), (j, i)])
  }

  // 对于map中每个元素(old, new)，将循环层次old用new代替
  // 用户须保证map是一一映射，即old不重复，new不重复，且old构成的集合与new构成的集合相等
  pub fn reorder_n(&self, map: &[(u32, u32)]) -> &Comp {
    let n = self.sch_dim();
    let s = format!("{{ {} -> [{}] }}\0", i0_in(n),
      comma_sep((0..n).map(|x| fn2display(move |f|
        write!(f, "i{}", map.iter().find(|&&(old, _)| x == old * 2 + 1)
          .map(|&(_, new)| new * 2 + 1).unwrap_or(x))
      ))));
    debug!("reorder_n: {}", s);
    self.apply_sch_raw(&s)
  }

  pub fn skew(&self, i: u32, j: u32, factor: u32) -> &Comp {
    debug_assert!(i < j);
    let (n, i, j) = (self.sch_dim(), i * 2 + 1, j * 2 + 1);
    let s = format!("{{ {} -> [{}]: i{j1} = {f} * i{i} + i{j} }}\0", i0_in(n),
      comma_sep((0..n).map(|x| fn2display(move |f|
        write!(f, "i{}", if x < j { x } else if x == j { n } else { x - 1 })))),
      j1 = n, f = factor, i = i, j = j);
    debug!("skew: {}", s);
    self.apply_sch_raw(&s)
  }

  pub fn shift(&self, i: u32, n: i32) -> &Comp {
    // 设置i_out = i_in + n
    self.schedule.write(map_set_eq(self.schedule.read(), i * 2 + 1, -1, n));
    debug!("shift: {}", self.schedule);
    self
  }

  pub fn separate(&self, i: u32, factor: u32) -> Option<&Comp> {
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
    debug!("separate: sep = {}", sep);
    let ge = self.schedule.copy()?.intersect_range(i_aff.copy()?.ge_set(sep.copy()?)?
      .add_dims(DimType::Out, n - pos - 1)?)?;
    // 被separate出去的计算迭代域是空集则不生成Comp
    if ge.is_empty()? {
      debug!("separate: dup comp has empty domain");
      return None;
    }
    let name = format!("_{}_sep{}\0", self.name(), self.func.new_comp_id());
    let dup = box Comp {
      ctx: self.ctx,
      func: self.func,
      expr: self.expr.clone(),
      schedule: ge.set_tuple_name(DimType::In, cstr(&name))?,
      store: if let Some(x) = &self.store { Some(x.copy()?.set_tuple_name(DimType::In, cstr(&name))?) } else { None },
      pred: None,
      succ: HashMap::default(),
      dim_tags: self.dim_tags.clone(),
    };
    // 上面的所有操作都只修改局部变量(修改Func::comp_cnt可以接受)，因此错误可以恢复，从这里开始的错误不能恢复
    dup.after(self, i);
    self.schedule.write(self.schedule.read().intersect_range(i_aff.lt_set(sep).unwrap()
      .add_dims(DimType::Out, n - pos - 1).unwrap()).unwrap());
    debug!("separate: created dup comp {}, sch = {}", dup.name(), dup.schedule);
    let ret = R::new(&*dup);
    self.p().func.comps.push(dup);
    Some(ret.get())
  }

  // 将循环层次i和i + 1合并成一个循环
  // 循环i + 1的static dim被丢弃；after，tag之类的都不会特殊处理，调用fuse前应保证它们不存在
  pub fn fuse(&self, i: u32) -> &Comp {
    // min1 <= i1 < max1, min2 <= i2 < max2, 令j = (i1 - min1) * (max2 - min2) + (i2 - min2)
    // 则i1 = floor(j / (max2 - min2)) + min1, i2 = j % (max2 - min2) + min2 (这由ISL自己推导)
    let Bound(_, min1, _) = self.extract_bound(i);
    let Bound(zero2, min2, max2) = self.extract_bound(i + 1);
    let (n, pos) = (self.sch_dim(), i * 2 + 1);
    let min1 = min1.add_dims(DimType::In, 2)?;
    let i1 = zero2.copy()?.set_coefficient_si(DimType::In, pos as _, 1)?.pw_aff_from_aff()?;
    let i2 = zero2.copy()?.set_coefficient_si(DimType::In, (pos + 2) as _, 1)?.pw_aff_from_aff()?;
    let j = zero2.set_coefficient_si(DimType::In, pos as _, 1)?.pw_aff_from_aff()?;
    let mut trans = i1.sub(min1)?.mul(max2.sub(min2.copy()?)?)?.add(i2.sub(min2)?)?.eq_map(j)?
      .add_dims(DimType::In, n - pos - 3)?.add_dims(DimType::Out, n - pos - 5)?;
    // 设置其他维度为恒等映射
    for i in 0..n - 2 {
      if i == pos { continue; }
      // out dim比in dim少2，pos + 1和pos + 2在out dim中不存在，所以i >= pos时in_pos要+ 2
      let in_pos = if i < pos { i } else { i + 2 };
      let cst = trans.get_space()?.local_space_from_space()?.constraint_alloc_equality()?
        .set_coefficient_si(DimType::In, in_pos as _, -1)?
        .set_coefficient_si(DimType::Out, i as _, 1)?;
      trans = trans.add_constraint(cst)?;
    }
    debug!("fuse: trans = {}", trans);
    self.schedule.write(self.schedule.read().apply_range(trans.align_params(self.schedule.get_space()?)?)?);
    debug!("fuse: schedule = {}", self.schedule);
    self
  }

  // 提取循环层次i的迭代范围，Bound的意义见Bound自身的注释
  pub fn extract_bound(&self, i: u32) -> Bound {
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
    let (min, max) = (min?, max?);
    let zero = dom.get_space()?.aff_zero_on_domain_space()?;
    let one = zero.copy()?.set_constant_si(1)?.pw_aff_from_aff()?;
    let max = max.add(one)?; // max += 1，循环范围是min <= i < max
    debug!("extract_bound: min = {}, max = {}", min, max);
    Bound(zero, min, max)
  }

  pub fn apply_sch_raw(&self, s: &str) -> &Comp {
    debug_assert!(s.ends_with('\0'));
    let t = self.ctx.map_read_from_str(cstr(&s))?.align_params(self.schedule.get_space()?)?;
    self.schedule.write(self.schedule.read().apply_range(t)?);
    self
  }
}

// 这个trait只是为了给Vec<&Comp>定义separate函数
pub trait Separate {
  fn separate(&self, i: u32, factor: u32);
}

impl Separate for Vec<&Comp> {
  // 对自身已有元素都执行Comp::separate(i, factor)，把生成的新Comp加到自身中(虽然参数是&self，但会修改自身)
  fn separate(&self, i: u32, factor: u32) {
    let mut s = self.p();
    for idx in 0..s.len() { // 不能用迭代器
      if let Some(x) = s[idx].separate(i, factor) { s.push(x); }
    }
  }
}

impl Comp {
  // 如果用来添加GPU相关的tag，需要保证tag的几个维度是完美嵌套的
  // 例如for (i) { for (j) { C1 }  C2 }，如果tag_dim(i, GPUBlockX)，tag_dim(j, GPUThreadX)
  // 生成的kernel会执行C1和C2，等价于for (i) { for (j) { C1 C2 } }，语义是错误的
  pub fn tag_dim(&self, at: u32, tag: DimTag) -> &Comp {
    let at = at as usize;
    let dim_tags = &mut self.p().dim_tags;
    if dim_tags.len() <= at { dim_tags.resize(at + 1, None); }
    dim_tags[at] = Some(tag);
    self
  }

  pub fn store(&self, buf: &Buf) -> &Comp {
    let store = identity_map(self.domain())
      .set_tuple_name(DimType::Out, cstr(&format!("{}\0", buf.name)))?;
    debug!("store: {}", store);
    self.p().store = Some(store);
    self
  }

  pub fn store_at(&self, buf: &Buf, idx: &[impl IntoExpr]) -> &Comp {
    let s = format!("{{ {}{} -> {}[{}] }}\0", self.name(), i0_in(self.n_dim()),
      buf.name, comma_sep(idx.iter().map(|e| e.clone_expr())));
    debug!("store_at: {}", s);
    self.p().store = Some(self.ctx.map_read_from_str(cstr(&s))?);
    self
  }
}

impl Comp {
  // at的意义是在包围at层循环的static dim上，self在after之后
  // A.after(B, i).after(C, j)的链式调用，语义是A在B后，B在C后
  pub fn after<'a>(&self, other: &'a Comp, at: u32) -> &'a Comp {
    let mut other = other.p();
    let old_level = other.succ.entry(self.into()).or_insert(at);
    *old_level = at.max(*old_level);
    if let Some(mut p) = self.pred {
      if p != other { p.succ.remove(&self.into()); }
    }
    self.p().pred = Some(other);
    other.get()
  }

  // A.before(B, i).before(C, j)的链式调用，语义是A在B前，B在C前
  pub fn before<'a>(&self, other: &'a Comp, at: u32) -> &'a Comp {
    other.after(self, at);
    other
  }

  // 用schedule中的static dim来实现after的逻辑。可以直接使用它，但多个Comp间的关系不一定可以保留
  // 例如A.after(B, i); B.after(C, i); 最终会正确生成A在B后，B在C后
  // 但A.after_raw(B, i); B.after_raw(C, i); 假设一开始static dim都是0，则最终A和B的都是1，分不出先后
  // 此外还须保证事先调用`Func::align_schedule`
  pub fn after_raw(&self, other: &Comp, at: u32) -> Unit {
    debug_assert_eq!(self.sch_dim(), other.sch_dim());
    // 理论上只需要将other.schedule中pos处的constraint + 1即可，但是ISL不提供这样的操作，必须重新构建
    for pos in (0..self.sch_dim()).step_by(2) {
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
      let order = order? + (pos == at * 2) as i32;
      self.schedule.write(map_set_eq(self.schedule.read(), pos, 0, order));
    }
    debug!("after_raw: {}", self.schedule);
    Unit
  }
}

// 输出[i0, ..., i{n-1}]
pub(crate) fn i0_in(n: u32) -> impl Display {
  fn2display(move |f| write!(f, "[{}]",
    comma_sep((0..n).map(|i| fn2display(move |f| write!(f, "i{}", i))))))
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

// 从set生成一对一的map，map的in dim名字为set名字，out dim名字为空
// 注意ISL中名字是空字符串和名字是空指针被认为是不一样的，用reset_tuple_id将名字赋成空指针，或set_tuple_name传None
pub(crate) fn identity_map(set: Set) -> Map {
  set.get_space()?.add_dims(DimType::In, set.n_dim() as _)?
    .set_tuple_name(DimType::In, set.get_tuple_name())?.reset_tuple_id(DimType::Out)?
    .map_identity()?.intersect_domain(set)?
}

pub(crate) fn identity_schedule(domain: Set) -> Map {
  let n_dim = domain.n_dim() as u32;
  let mut sch = identity_map(domain);
  for i in 0..=n_dim { // 在0，2，4...，2 * n_dim下标处插入0
    let pos = 2 * i;
    sch = sch.insert_dims(DimType::Out, pos, 1)?;
    sch = map_add_constraint(sch, pos, 0, 0);
  }
  sch
}
