use ptr::*;
use isl::{Ctx, Set, Map, DimType};
use std::io;
use crate::*;

#[derive(Debug)]
pub struct Func {
  // 限制符号常量/参数取值范围
  pub func_ctx: Option<Set>,
  pub name: Box<str>,
  pub comps: Vec<Box<Comp>>,
  pub v_cnt: u32,
  pub c_cnt: u32,
  pub iter_ty: Type,
  // Ctx必须在所有引用Ctx的成员析构后析构
  pub ctx: Ctx,
}

impl Func {
  pub fn new(name: Box<str>) -> Option<Box<Func>> {
    Some(box Func { func_ctx: None, name, comps: Vec::new(), v_cnt: 0, c_cnt: 0, iter_ty: I32, ctx: Ctx::new()? })
  }

  pub fn find_comp(&self, name: &str) -> Option<P<Comp>> {
    self.comps.iter().find(|c| c.name() == name).map(|c| P::new(&**c))
  }

  pub fn add_ctx_constraint(&mut self, ctx: Set) {
    if let Some(x) = &mut self.func_ctx {
      x.write(x.read().intersect(ctx).unwrap());
    } else { self.func_ctx = Some(ctx); }
  }

  pub fn align_schedule(&mut self) {
    let max_dim = self.comps.iter().map(|c| c.sch_dim() as u32).max().unwrap_or(0);
    for c in &mut self.comps {
      c.schedule.write(align_dim(c.schedule.read(), max_dim).unwrap());
      debug!("aligned schedule: {}", c.schedule);
    }
  }

  pub fn codegen(&mut self) -> io::Result<()> {
    Ok(())
  }
}

fn align_dim(mut map: Map, max_dim: u32) -> Option<Map> {
  let orig_dim = map.dim(DimType::Out);
  assert!(max_dim >= orig_dim);
  map = map.add_dims(DimType::Out, max_dim - orig_dim)?;
  for i in orig_dim..max_dim { map = map_out_eq0(map, i)?; }
  let in_name = map.get_tuple_name(DimType::In)?;
  map.set_tuple_name(DimType::Out, in_name)
}

impl Func {
  pub(crate) fn new_var_name(&mut self) -> String { format!("t{}\0", (self.v_cnt, self.v_cnt += 1).0) }

  pub(crate) fn new_comp_name(&mut self) -> String { format!("C{}\0", (self.c_cnt, self.c_cnt += 1).0) }
}
