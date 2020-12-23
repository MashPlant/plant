use ptr::*;
use isl::{CtxRef, Set, Map, DimType};
use std::fmt::Write;
use crate::*;

#[derive(Debug)]
pub struct Comp {
  pub ctx: CtxRef,
  pub func: P<Func>,
  pub domain: Set,
  // expr为None表示input，不参与调度
  pub expr: Option<Expr>,
  pub schedule: Map,
  pub access: Option<Map>,
  pub pred: Option<P<Comp>>,
  pub succ: HashMap<P<Comp>, u32>,
}

impl Func {
  pub fn new_comp(&mut self, name: &str, iters: &[Var], expr: Option<Expr>) -> Option<R<Comp>> {
    let mut d = String::new();
    if let Some(expr) = &expr {
      let mut vars = HashSet::default();
      expr.visit(&mut |e| if let Var(x) = &e.1 {
        if iters.iter().find(|i| i.name == x.name).is_none() {
          vars.insert(&x.name);
        }
      });
      if !vars.is_empty() {
        write!(d, "[{}] -> ", comma_sep(vars.iter())).ok()?;
      }
    }
    write!(d, "{{ {}[", name).ok()?;
    if iters.is_empty() { d.push('0'); } else {
      write!(d, "{}", comma_sep(iters.iter().map(|i| &i.name))).ok()?;
    }
    d.push_str("] : ");
    write!(d, "{}", sep(
      iters.iter().filter(|Var { range: box [lb, ub], .. }| lb.is_some() || ub.is_some())
        .map(|Var { name, range: box [lb, ub], .. }| fn2display(move |f| {
          if let Some(lb) = lb { write!(f, "{} <= ", lb)?; }
          f.write_str(name)?;
          if let Some(ub) = ub { write!(f, " < {}", ub)?; }
          Ok(())
        })), " and ")).ok()?;
    d.push_str(" }");
    debug!("constructed domain str: {}", d);
    d.push('\0');
    let domain = self.ctx.set_read_from_str(d.as_str().into())?;
    self.new_comp_raw(domain, expr)
  }

  pub fn new_comp_raw(&mut self, mut domain: Set, expr: Option<Expr>) -> Option<R<Comp>> {
    for i in 0..domain.n_dim() {
      if !domain.has_dim_name(DimType::Set, i)? {
        domain = domain.set_dim_name(DimType::Set, i, self.new_var_name().as_str().into())?;
      }
    }
    let schedule = identity_schedule(domain.copy()?)?;
    debug!("initial identity schedule: {}", schedule);
    let comp = box Comp { ctx: *self.ctx.as_ref(), func: self.into(), domain, expr, schedule, access: None, pred: None, succ: HashMap::default() };
    assert!(self.find_comp(comp.name()).is_none()); // 不允许相同名字的Comp
    let ret = R::new(&*comp);
    self.comps.push(comp);
    Some(ret)
  }
}

impl Comp {
  pub fn name(&self) -> &str { self.domain.get_tuple_name().unwrap().as_str() }

  pub fn n_dim(&self) -> usize { self.domain.n_dim() as _ }

  pub fn sch_dim(&self) -> usize { self.schedule.dim(DimType::Out) as _ }

  pub fn get(&self, indices: Box<[Expr]>) -> Expr {
    assert_eq!(indices.len(), self.n_dim());
    let expr = self.expr.as_ref().unwrap();
    Expr(expr.0, Access(self.name().into(), indices))
  }

  pub fn get_inline(&self, indices: Box<[Expr]>) -> Expr {
    assert_eq!(indices.len(), self.n_dim());
    let mut expr = self.expr.as_ref().unwrap().clone();
    let mut map = HashMap::with_capacity_and_hasher(indices.len(), Default::default());
    for (i, e) in indices.into_vec().into_iter().enumerate() {
      map.insert(self.domain.get_dim_name(DimType::Set, i as _).unwrap().as_str(), e);
    }
    expr.visit_mut(&mut |e| if let Var(x) = &mut e.1 {
      if let Some(x) = map.get(x.name.as_ref()) { *e = x.clone(); }
    });
    expr
  }
}

impl Comp {
  pub fn find_loop_level(&self, var: &str) -> u32 {
    // 这里不使用isl_map_find_dim_by_name，因为它只能查找cstr，而是手动实现其中遍历查找的逻
    for i in 0..self.sch_dim() as u32 {
      let iter = self.schedule.get_dim_name(DimType::Out, i).unwrap();
      if iter.as_str() == var {
        return i;
      }
    }
    // 所有使用场景中都期望这个变量存在，否则无法继续，所以不需要返回Option，直接在这里panic
    panic!("loop level of `{}` not found", var)
  }
}

impl Comp {
  // 允许A.after(B, i).after(C, j)的链式调用，语义是A在B后，B在C后
  pub fn after<'a>(&mut self, other: &'a mut Comp, at: &str) -> &'a mut Comp { self.after_level(other, self.find_loop_level(at)) }

  pub fn after_level<'a>(&mut self, other: &'a mut Comp, at: u32) -> &'a mut Comp {
    let old_level = other.succ.entry(self.into()).or_insert(at);
    *old_level = at.max(*old_level);
    assert!(self.pred == None || self.pred == Some(other.into()));
    self.pred = Some(other.into());
    other
  }

  pub fn before(&mut self, other: &mut Comp, at: &str) { self.before_level(other, self.find_loop_level(at)) }

  pub fn before_level(&mut self, other: &mut Comp, at: u32) { other.after_level(self, at); }

  pub fn between(&mut self, before: &mut Comp, before_at: &str, after: &mut Comp, after_at: &str) {
    self.between_level(before, self.find_loop_level(before_at), after, self.find_loop_level(after_at))
  }

  pub fn between_level(&mut self, before: &mut Comp, before_at: u32, after: &mut Comp, after_at: u32) {
    // 如果after是before的后继，取消这条边
    if before.succ.remove(&after.into()).is_some() { after.pred = None; }
    self.after_level(before, before_at);
    after.after_level(self, after_at);
  }
}

// 设置map的out维度中pos处为0
pub(crate) fn map_out_eq0(map: Map, pos: u32) -> Option<Map> {
  let lsp = map.get_space()?.local_space_from_space()?;
  let mut cst = lsp.constraint_alloc_equality()?;
  cst = cst.set_coefficient_si(DimType::Out, pos as _, 1)?;
  cst = cst.set_constant_si(0)?;
  map.add_constraint(cst)
}

pub(crate) fn identity_schedule(domain: Set) -> Option<Map> {
  let sp = domain.get_space()?;
  let n = domain.n_dim() + 1;
  let mut sch = sp.map_from_set()?.map_identity()?;
  sch = sch.intersect_domain(domain)?;
  sch = sch.coalesce()?;
  for i in 0..n { // 在0，2，4...，2 * n_dim下标处插入0
    let pos = 2 * i;
    sch = sch.insert_dims(DimType::Out, pos, 1)?;
    sch = map_out_eq0(sch, pos)?;
  }
  // 每次insert_dims后out name会清空，所以放在循环后
  let in_name = sch.get_tuple_name(DimType::In)?;
  sch.set_tuple_name(DimType::Out, in_name)
}
