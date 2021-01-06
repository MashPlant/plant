use isl::{CtxRef, Set, Map, DimType, CStr};
use std::fmt::Display;
use crate::*;

#[derive(Debug)]
pub struct Comp {
  pub ctx: CtxRef,
  pub func: P<Func>,
  // isl_basic_set表示可以用一组仿射约束的交集定义的集合，isl_set表示一组isl_basic_set的并集
  // 这里需要用到isl_set，例如i <= max(x, y)这样的约束无法用交集表示，必须是i <= x or i <= y
  pub domain: Set,
  pub expr: Expr,
  // in dim名字是Comp的名字，out dim名字是空的
  // 从循环层次i到包围它的static dim：i * 2；从循环层次i到它的dynamic dim：i * 2 + 1
  pub schedule: Map,
  pub store: Option<Map>,
  pub pred: Option<P<Comp>>,
  pub succ: HashMap<P<Comp>, u32>,
  pub dim_tags: Vec<Option<DimTag>>,
}

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum DimTag { Parallel, GPUBlockX, GPUBlockY, GPUBlockZ, GPUThreadX, GPUThreadY, GPUThreadZ }

// 将Comp传给接受impl IntoExpr的地方时，是将它作为Param表达式，而非Access表达式
// 用法区别请看Comp::as_param的注释
impl IntoExpr for &Comp {
  fn expr(self) -> Expr { self.as_param() }
}

impl Func {
  pub fn comp(&self, name: &str, ranges: &[(impl IntoExpr, impl IntoExpr)], expr: impl IntoExpr) -> &Comp {
    // 很多时候调用方可以提供&[(Expr, Expr)]，这里的拷贝是多余的，但这点浪费可以忽略
    let ranges = ranges.iter().map(|(lb, ub)| (lb.clone_expr(), ub.clone_expr())).collect::<Vec<_>>();
    let expr = expr.expr();
    let mut params = HashSet::<&str>::default();
    // 收集ranges，expr中的所有Param
    let ref mut vis = |e: &Expr| if let Param(x) = &e.1 { params.insert(x.get()); };
    for (lb, ub) in &ranges {
      lb.visit(vis);
      ub.visit(vis);
    }
    expr.visit(vis);
    let s = format!("[{}] -> {{ {}{}: {} }}\0", comma_sep(params.iter()), name, i0_in(ranges.len() as _),
      sep(ranges.iter().enumerate().map(|(i, (lb, ub))| fn2display(move |f|
        write!(f, "{} <= i{} < {}", lb, i, ub))), " and "));
    debug!("comp: domain = {}", s);
    let domain = self.ctx.set_read_from_str(cstr(&s)).unwrap();
    self.comp_raw(domain, expr)
  }

  pub fn comp_raw(&self, domain: Set, expr: Expr) -> &Comp {
    // set_read_from_str生成的set可能有冗余，例如为i <= min(x, y)生成两个BasicSet，其实一个就可以表示，coalesce就是试图合并BasicSet
    let domain = domain.coalesce().unwrap();
    let schedule = identity_schedule(&domain).unwrap();
    debug!("comp_raw: initial identity schedule = {}", schedule);
    let comp = box Comp { ctx: *self.ctx, func: self.into(), domain, expr, schedule, store: None, pred: None, succ: HashMap::default(), dim_tags: Vec::new() };
    assert!(self.find_comp(comp.name()).is_none()); // 不允许相同名字的Comp
    let ret = R::new(&*comp);
    P::new(self).comps.push(comp);
    ret.get()
  }
}

impl Comp {
  // 返回的字符串来源于cstr，[len()]位置是\0，可以传入ISL的接口，更直接的是使用name_cstr
  pub fn name(&self) -> &str { self.domain.get_tuple_name().unwrap().as_str() }

  pub fn name_cstr(&self) -> Option<CStr> { self.domain.get_tuple_name() }

  pub fn n_dim(&self) -> u32 { self.domain.n_dim() }

  pub fn sch_dim(&self) -> u32 { self.schedule.dim(DimType::Out) }

  // 输出[逗号分隔的params列表]
  pub fn params<'a>(&'a self) -> impl Display + 'a {
    fn2display(move |f| write!(f, "[{}]", comma_sep((0..self.domain.dim(DimType::Param))
      .map(|i| self.domain.get_dim_name(DimType::Param, i).unwrap()))))
  }

  pub fn set_expr(&self, expr: Expr) { P::new(self).expr = expr; }

  // 将自身作为一个Param表达式，Expr中直接通过它的名字使用它
  // 当且仅当它没有store的时候，生成的代码会在计算发生的地方定义一个对应名字的局部变量，所以它必须没有store
  // 如果Comp的计算结果用于其他Comp中的循环范围，Access下标，则必须用as_param
  // 如果只是用于普通运算，可以用as_param或at，as_param会往domain/schedule中引入一个参数，应该没有什么好处
  pub fn as_param(&self) -> Expr {
    Expr(self.expr.0, Param(self.name().into()))
  }

  pub fn at(&self, idx: &[impl IntoExpr]) -> Expr {
    assert_eq!(idx.len() as u32, self.n_dim());
    Expr(self.expr.0, Access(self.into(), idx.iter().map(|e| e.clone_expr()).collect()))
  }

  pub fn at_inline(&self, idx: &[impl IntoExpr]) -> Expr {
    assert_eq!(idx.len() as u32, self.n_dim());
    let mut expr = self.expr.clone();
    expr.visit_mut(&mut |e| if let Iter(x) = &mut e.1 { *e = idx[*x as usize].clone_expr(); });
    expr
  }
}

impl Comp {
  pub fn tile(&self, i: u32, j: u32, tile_i: u32, tile_j: u32) -> &Comp {
    assert!(i < j);
    self.split(i, tile_i).split(j + 1, tile_j).reorder(i + 1, j + 1)
  }

  pub fn tile_3(&self, i: u32, j: u32, k: u32, tile_i: u32, tile_j: u32, tile_k: u32) -> &Comp {
    assert!(i < j && j < k);
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
    self.apply_sch_raw(&s).unwrap()
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
    self.apply_sch_raw(&s).unwrap()
  }

  pub fn skew(&self, i: u32, j: u32, factor: u32) -> &Comp {
    assert!(i < j);
    let (n, i, j) = (self.sch_dim(), i * 2 + 1, j * 2 + 1);
    let s = format!("{{ {} -> [{}]: i{j1} = {f} * i{i} + i{j} }}\0", i0_in(n),
      comma_sep((0..n).map(|x| fn2display(move |f|
        write!(f, "i{}", if x < j { x } else if x == j { n } else { x - 1 })))),
      j1 = n, f = factor, i = i, j = j);
    debug!("skew: {}", s);
    self.apply_sch_raw(&s).unwrap()
  }

  pub fn shift(&self, i: u32, n: i32) -> &Comp {
    // 设置i_out = i_in + n
    self.schedule.write(map_set_eq(self.schedule.read(), i * 2 + 1, -1, n).unwrap());
    debug!("shift: {}", self.schedule);
    self
  }

  pub fn separate(&self, i: u32, factor: u32) -> Option<&Comp> {
    let (n, pos) = (self.sch_dim(), i * 2 + 1);
    // 通过投影去掉i之后的维度，用i前面的维度表示i的上下界
    // remove_divs会去掉类似floor(x / 2) + floor(y / 3) + 4 <= 0这样的约束
    // 因为后面提取上下界的时候也处理不了这样的约束，所以就在这里去掉，这个近似不会造成结果出错
    // todo: 这个apply过程可以提取一下，codegen中也用到了
    let dom = self.domain.copy()?.apply(self.schedule.copy()?)?.set_tuple_name(cstr(self.name()))?
      .project_out(DimType::Out, pos + 1, n - pos - 1)?.remove_divs()?;
    debug!("separate: projected dom = {}", dom);
    // 只支持由一个BasicSet组成的Set上的separate，否则情况太复杂
    if dom.n_basic_set() > 1 {
      warn!("separate: #bset > 1 not supported");
      return None;
    }
    let csts = dom.get_basic_set_list()?.get_basic_set(0)?.get_constraint_list()?;
    let (mut min, mut max) = (None, None);
    for i in 0..csts.n_constraint() {
      let cst = csts.get_constraint(i)?;
      let k = cst.get_coefficient_val(DimType::Out, pos as _)?.get_num_si();
      if k != 0 {
        // 如果这个维度可以用前面的维度用等式表示，那它就只有这一个取值，循环变换没有意义
        if cst.is_equality()? {
          debug!("separate: dim has equality constraint");
          return None;
        }
        let aff = cst.get_bound(DimType::Out, pos as _)?.pw_aff_from_aff()?;
        // ISL中inequality约束一定是... >= 0
        if k > 0 {
          min = Some(if let Some(x) = min { aff.max(x)? } else { aff });
        } else {
          max = Some(if let Some(x) = max { aff.min(x)? } else { aff });
        }
      }
    }
    let (min, max) = (min?, max?);
    let factor = self.ctx.val_int_from_ui(factor as _)?;
    // 有没有简单一点的方法定义i和1...
    let zero = dom.get_space()?.local_space_from_space()?.aff_zero_on_domain()?;
    let i_aff = zero.copy()?.set_coefficient_si(DimType::In, pos as _, 1)?.pw_aff_from_aff()?;
    let one = zero.set_constant_si(1)?.pw_aff_from_aff()?;
    let max = max.add(one)?; // max += 1，循环范围变成min <= i < max
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
    if ge.is_empty()? { return None; }
    let name = self.func.new_comp_name();
    let dup = box Comp {
      ctx: self.ctx,
      func: self.func,
      domain: self.domain.copy()?.set_tuple_name(cstr(&name))?,
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
    debug!("separate: created dup comp = {}", dup.name());
    let ret = R::new(&*dup);
    P::new(self).func.comps.push(dup);
    Some(ret.get())
  }

  pub fn apply_sch_raw(&self, s: &str) -> Option<&Comp> {
    debug_assert!(s.ends_with('\0'));
    let t = self.ctx.map_read_from_str(cstr(&s))?.align_params(self.schedule.get_space()?)?;
    self.schedule.write(self.schedule.read().apply_range(t)?);
    Some(self)
  }
}

impl Comp {
  pub fn tag_dim(&self, at: u32, tag: DimTag) -> &Comp {
    let at = at as usize;
    let dim_tags = &mut P::new(self).dim_tags;
    if dim_tags.len() <= at { dim_tags.resize(at + 1, None); }
    dim_tags[at] = Some(tag);
    self
  }

  pub fn store(&self, buf: &Buf) -> &Comp {
    let mut store = identity_map(&self.domain).unwrap();
    store = store.set_tuple_name(DimType::Out, cstr(&format!("{}\0", buf.name))).unwrap();
    debug!("store: {}", store);
    P::new(self).store = Some(store);
    self
  }

  pub fn store_at(&self, buf: &Buf, idx: &[impl IntoExpr]) -> &Comp {
    let s = format!("{{ {}{} -> {}[{}] }}\0", self.name(), i0_in(self.n_dim()),
      buf.name, comma_sep(idx.iter().map(|e| e.clone_expr())));
    debug!("store_at: {}", s);
    P::new(self).store = Some(self.ctx.map_read_from_str(cstr(&s)).unwrap());
    self
  }
}

impl Comp {
  // at的意义是在包围at层循环的static dim上，self在after之后
  // A.after(B, i).after(C, j)的链式调用，语义是A在B后，B在C后
  pub fn after<'a>(&self, other: &'a Comp, at: u32) -> &'a Comp {
    let mut other = P::new(other);
    let old_level = other.succ.entry(self.into()).or_insert(at);
    *old_level = at.max(*old_level);
    if let Some(mut p) = self.pred {
      if p != other { p.succ.remove(&self.into()); }
    }
    P::new(self).pred = Some(other);
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
  pub fn after_raw(&self, other: &Comp, at: u32) {
    debug_assert_eq!(self.sch_dim(), other.sch_dim());
    // 理论上只需要将other.schedule中pos处的constraint + 1即可，但是ISL不提供这样的操作，必须重新构建
    for i in (0..self.sch_dim()).step_by(2) {
      // 在other的对应位置上static dim上 + 1，其余不变
      let order = get_static_dim(&other.schedule, i).unwrap() + (i == at * 2) as i32;
      self.schedule.write(map_set_eq(self.schedule.read(), i, 0, order).unwrap());
    }
    debug!("after_raw: {}", self.schedule);
  }
}

// 输出[i0, ..., i{n-1}]
pub(crate) fn i0_in(n: u32) -> impl Display {
  fn2display(move |f| write!(f, "[{}]",
    comma_sep((0..n).map(|i| fn2display(move |f| write!(f, "i{}", i))))))
}

// 在pos处添加约束: k_in * i_in + i_out + val == 0; 如果k_in传0，就是设置out维度中pos处值为val
// 如果已经存在对这个位置约束则不能使用它
pub(crate) fn map_add_constraint(map: Map, pos: u32, k_in: i32, val: i32) -> Option<Map> {
  let pos = pos as i32;
  let mut cst = map.get_space()?.local_space_from_space()?.constraint_alloc_equality()?;
  // 我的应用中out dim总是多于in dim，所以只需要检查pos是否在in dim范围内
  if pos < cst.dim(DimType::In) {
    cst = cst.set_coefficient_si(DimType::In, pos, k_in)?;
  }
  cst = cst.set_coefficient_si(DimType::Out, pos, 1)?.set_constant_si(-val)?;
  map.add_constraint(cst)
}

// k_in，val语义和map_add_constraint中相同
// 可以处理已经存在对这个位置约束的情形，但比`map_add_constraint`开销更大
pub(crate) fn map_set_eq(map: Map, pos: u32, k_in: i32, val: i32) -> Option<Map> {
  let mut sp = map.get_space()?;
  let (n_in, n_out) = (sp.dim(DimType::In), sp.dim(DimType::Out));
  sp = sp.add_dims(DimType::In, n_out - n_in)?;
  let mut trans = sp.map_universe()?;
  for i in 0..n_out {
    // 除pos外其他维度，包括static和dynamic dim，都是恒等映射
    let (k_in, val) = if i == pos { (k_in, val) } else { (-1, 0) };
    trans = map_add_constraint(trans, i, k_in, val)?;
  }
  map.apply_range(trans)
}

// 从set生成一对一的map，map的in dim名字为set名字，out dim名字为空
// 注意ISL中名字是空字符串和名字是空指针被认为是不一样的，用reset_tuple_id将名字赋成空指针，或set_tuple_name传None
pub(crate) fn identity_map(set: &Set) -> Option<Map> {
  let sp = set.get_space()?.add_dims(DimType::In, set.n_dim())?
    .set_tuple_name(DimType::In, set.get_tuple_name())?
    .reset_tuple_id(DimType::Out)?;
  sp.map_identity()?.intersect_domain(set.copy()?)
}

pub(crate) fn identity_schedule(domain: &Set) -> Option<Map> {
  let mut sch = identity_map(domain)?;
  for i in 0..=domain.n_dim() { // 在0，2，4...，2 * n_dim下标处插入0
    let pos = 2 * i;
    sch = sch.insert_dims(DimType::Out, pos, 1)?;
    sch = map_add_constraint(sch, pos, 0, 0)?;
  }
  Some(sch)
}

// 获取map中对应位置的static dim
fn get_static_dim(map: &Map, pos: u32) -> Option<i32> {
  let mut ret = None;
  // Map由多个BasicMap的并集组成，理论上不同的BasicMap的同一个out dim处可以有不同的constraint
  // 但在这里不会发生，这里获取的是static dim，应该保证所有BasicMap的static dim都是相同的
  map.foreach_basic_map(&mut |m| {
    let csts = m.get_constraint_list()?;
    for i in 0..csts.n_constraint() {
      let cst = csts.get_constraint(i)?;
      let k = cst.get_coefficient_val(DimType::Out, pos as _)?;
      if k.is_one()? {
        let val = -cst.get_constant_val()?.get_num_si() as _;
        debug_assert!(ret == None || ret == Some(val));
        ret = Some(val);
      }
    }
    Some(())
  })?;
  ret
}
