use isl::{CtxRef, BasicSet, BasicMap, DimType};
use std::fmt::{Write, Display};
use crate::*;

#[derive(Debug)]
pub struct Comp {
  pub ctx: CtxRef,
  pub func: P<Func>,
  // isl_basic_set表示可以用一组仿射约束的交集定义的集合，isl_set表示一组isl_basic_set的并集
  // 这里用不到isl_set，因为多面体就是可以用一组仿射约束的交集定义的，不需要并集
  pub domain: BasicSet,
  // Expr和Type中都有Type，都表示计算结果的类型，Type表示不参与调度，作为输入
  pub expr: OptionExpr,
  // in dim名字是Comp的名字，out dim名字是空的
  // 从循环层次i到它包围的static dim：i * 2；从循环层次i到它的dynamic dim：i * 2 - 1 (要求i >= 1)
  pub schedule: BasicMap,
  pub store: Option<BasicMap>,
  pub pred: Option<P<Comp>>,
  pub succ: HashMap<P<Comp>, u32>,
}

#[derive(Debug)]
pub enum OptionExpr { Expr(Expr), Type(Type) }

impl<T: IntoExpr> From<T> for OptionExpr { fn from(t: T) -> OptionExpr { OptionExpr::Expr(t.expr()) } }

impl From<Type> for OptionExpr { fn from(t: Type) -> OptionExpr { OptionExpr::Type(t) } }

impl Func {
  pub fn comp(&self, name: &str, ranges: &[(impl IntoExpr, impl IntoExpr)], expr: impl Into<OptionExpr>) -> Option<R<Comp>> {
    let expr = expr.into();
    let mut d = String::new();
    if let OptionExpr::Expr(expr) = &expr {
      let mut params = HashSet::default();
      expr.visit(&mut |e| if let Param(x) = &e.1 { params.insert(x); });
      if !params.is_empty() {
        write!(d, "[{}] -> ", comma_sep(params.iter())).ok()?;
      }
    }
    write!(d, "{{ {}{}: {} }}\0", name, i0_in(ranges.len() as _),
      sep(ranges.iter().enumerate().map(|(i, (lb, ub))| fn2display(move |f| {
        write!(f, "{} <= i{} < {}", lb.clone_expr(), i, ub.clone_expr())
      })), " and ")).ok()?;
    debug!("constructed domain str: {}", d);
    let domain = self.ctx.basic_set_read_from_str(d.as_str().into())?;
    self.comp_raw(domain, expr)
  }

  pub fn comp_raw(&self, domain: BasicSet, expr: OptionExpr) -> Option<R<Comp>> {
    let schedule = identity_schedule(&domain)?;
    debug!("initial identity schedule: {}", schedule);
    let comp = box Comp { ctx: *self.ctx.as_ref(), func: self.into(), domain, expr, schedule, store: None, pred: None, succ: HashMap::default() };
    assert!(self.find_comp(comp.name()).is_none()); // 不允许相同名字的Comp
    let ret = R::new(&*comp);
    P::new(self).comps.push(comp);
    Some(ret)
  }
}

impl Comp {
  pub fn name(&self) -> &str { self.domain.get_tuple_name().unwrap().as_str() }

  pub fn n_dim(&self) -> u32 { self.domain.n_dim() }

  pub fn sch_dim(&self) -> u32 { self.schedule.dim(DimType::Out) }

  pub fn ty(&self) -> Type { match self.expr { OptionExpr::Expr(Expr(t, _)) | OptionExpr::Type(t) => t } }

  pub fn set_expr(&self, expr: Expr) { P::new(self).expr = OptionExpr::Expr(expr); }

  pub fn at(&self, idx: &[impl IntoExpr]) -> Expr {
    assert_eq!(idx.len() as u32, self.n_dim());
    Expr(self.ty(), Access(self.into(), idx.iter().map(|e| e.clone_expr()).collect()))
  }

  pub fn at_inline(&self, idx: &[impl IntoExpr]) -> Expr {
    assert_eq!(idx.len() as u32, self.n_dim());
    let mut expr = match &self.expr { OptionExpr::Expr(e) => e.clone(), _ => panic!("`at_inline` on comp with no expr") };
    expr.visit_mut(&mut |e| if let Iter(x) = &mut e.1 { *e = idx[*x as usize].clone_expr(); });
    expr
  }
}

impl Comp {
  pub fn tile(&self, i: u32, j: u32, tile_i: u32, tile_j: u32) -> &Comp {
    assert!(i < j);
    self.split(i, tile_i).split(j + 1, tile_j).reorder(i + 1, j + 1)
  }

  pub fn split(&self, i: u32, factor: u32) -> &Comp {
    let (n, i) = (self.sch_dim(), i * 2 - 1);
    let s = format!("{{ {} -> [{}]: i{i0} = floor(i{i} / {f}) and i{i1} = i{i} % {f} }}\0", i0_in(n),
      comma_sep((0..n + 2).map(|x| fn2display(move |f|
        if x == i + 1 { f.write_str("0") } else {
          write!(f, "i{}", if x < i { x } else if x == i { n } else if x == i + 2 { n + 1 } else { x - 2 })
        }))),
      i0 = n, i1 = n + 1, i = i, f = factor);
    debug!("split: {}", s);
    self.apply_sch_raw(&s)
  }

  pub fn reorder(&self, i: u32, j: u32) -> &Comp {
    let (n, i, j) = (self.sch_dim(), i * 2 - 1, j * 2 - 1);
    let s = format!("{{ {} -> [{}] }}\0", i0_in(n),
      comma_sep((0..n).map(|x| fn2display(move |f|
        write!(f, "i{}", if x == i { j } else if x == j { i } else { x })))));
    debug!("reorder: {}", s);
    self.apply_sch_raw(&s)
  }

  pub fn skew(&self, i: u32, j: u32, factor: u32) -> &Comp {
    assert!(i < j);
    let (n, i, j) = (self.sch_dim(), i * 2 - 1, j * 2 - 1);
    let s = format!("{{ {} -> [{}]: i{j1} = {f} * i{i} + i{j} }}\0", i0_in(n),
      comma_sep((0..n).map(|x| fn2display(move |f|
        write!(f, "i{}", if x < j { x } else if x == j { n } else { x - 1 })))),
      j1 = n, f = factor, i = i, j = j);
    debug!("skew: {}", s);
    self.apply_sch_raw(&s)
  }

  pub fn apply_sch_raw(&self, s: &str) -> &Comp {
    debug_assert!(s.ends_with('\0'));
    let t = self.ctx.basic_map_read_from_str(s.into()).unwrap();
    self.schedule.write(self.schedule.read().apply_range(t).unwrap());
    self
  }
}

impl Comp {
  pub fn store(&self, buf: &Buf) -> &Comp {
    let mut store = identity_map(&self.domain).unwrap();
    store = store.set_tuple_name(DimType::Out, format!("{}\0", buf.name).as_str().into()).unwrap();
    debug!("store: {}", store);
    P::new(self).store = Some(store);
    self
  }

  pub fn store_at(&self, buf: &Buf, idx: &[impl IntoExpr]) -> &Comp {
    let s = format!("{{ {} -> {}[{}] }}\0", i0_in(self.n_dim()),
      buf.name, comma_sep(idx.iter().map(|e| e.clone_expr())));
    debug!("store_at: {}", s);
    P::new(self).store = Some(self.ctx.basic_map_read_from_str(s.as_str().into()).unwrap());
    self
  }
}

impl Comp {
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
      self.schedule.write(map_set_eq(self.schedule.read(), i, order).unwrap());
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
pub(crate) fn map_add_constraint(map: BasicMap, pos: u32, k_in: i32, val: i32) -> Option<BasicMap> {
  let pos = pos as i32;
  let lsp = map.get_space()?.local_space_from_space()?;
  let mut cst = lsp.constraint_alloc_equality()?;
  // 我的应用中out dim总是多于in dim，所以只需要检查pos是否在in dim范围内
  if pos < cst.dim(DimType::In) {
    cst = cst.set_coefficient_si(DimType::In, pos, k_in)?;
  }
  cst = cst.set_coefficient_si(DimType::Out, pos, 1)?;
  cst = cst.set_constant_si(-val)?;
  map.add_constraint(cst)
}

// 设置out维度中pos处值为val，可以处理已经存在对这个位置约束的情形，但比`map_add_constraint`开销更大
pub(crate) fn map_set_eq(map: BasicMap, pos: u32, val: i32) -> Option<BasicMap> {
  let mut sp = map.get_space()?;
  let (n_in, n_out) = (sp.dim(DimType::In), sp.dim(DimType::Out));
  sp = sp.add_dims(DimType::In, n_out - n_in)?;
  let mut trans = sp.basic_map_universe()?;
  for i in 0..n_out {
    // 除pos外其他维度，包括static和dynamic dim，都是恒等映射
    let (k_in, val) = if i == pos { (0, val) } else { (-1, 0) };
    trans = map_add_constraint(trans, i, k_in, val)?;
  }
  map.apply_range(trans)
}

// 从set生成一对一的map，map的in dim名字为set名字，out dim名字为空
pub(crate) fn identity_map(set: &BasicSet) -> Option<BasicMap> {
  let sp = set.get_space()?.add_dims(DimType::In, set.n_dim())?
    .set_tuple_name(DimType::In, set.get_tuple_name()?)?
    .set_tuple_name(DimType::Out, "\0".into())?;
  sp.basic_map_identity()?.intersect_domain(set.copy()?)
}

pub(crate) fn identity_schedule(domain: &BasicSet) -> Option<BasicMap> {
  let mut sch = identity_map(domain)?;
  for i in 0..domain.n_dim() + 1 { // 在0，2，4...，2 * n_dim下标处插入0
    let pos = 2 * i;
    sch = sch.insert_dims(DimType::Out, pos, 1)?;
    sch = map_add_constraint(sch, pos, 0, 0)?;
  }
  Some(sch)
}

// 获取map中对应位置的static dim
fn get_static_dim(map: &BasicMap, pos: u32) -> Option<i32> {
  let csts = map.get_constraint_list()?;
  for i in 0..csts.n_constraint() {
    let cst = csts.get_constraint(i)?;
    let k = cst.get_coefficient_val(DimType::Out, pos as _)?;
    if k.is_one()? { return Some(-cst.get_constant_val()?.get_num_si() as _); }
  }
  None
}
