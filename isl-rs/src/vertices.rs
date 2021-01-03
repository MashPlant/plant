use crate::*;

extern "C" {
  pub fn isl_vertex_get_ctx(vertex: VertexRef) -> Option<CtxRef>;
  pub fn isl_vertex_get_id(vertex: VertexRef) -> c_int;
  pub fn isl_vertex_get_domain(vertex: VertexRef) -> Option<BasicSet>;
  pub fn isl_vertex_get_expr(vertex: VertexRef) -> Option<MultiAff>;
  pub fn isl_vertex_free(vertex: Vertex) -> ();
  pub fn isl_basic_set_compute_vertices(bset: BasicSetRef) -> Option<Vertices>;
  pub fn isl_vertices_get_ctx(vertices: VerticesRef) -> Option<CtxRef>;
  pub fn isl_vertices_get_n_vertices(vertices: VerticesRef) -> c_int;
  pub fn isl_vertices_foreach_vertex(vertices: VerticesRef, fn_: unsafe extern "C" fn(vertex: Vertex, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_vertices_free(vertices: Vertices) -> *mut c_void;
  pub fn isl_cell_get_ctx(cell: CellRef) -> Option<CtxRef>;
  pub fn isl_cell_get_domain(cell: CellRef) -> Option<BasicSet>;
  pub fn isl_cell_foreach_vertex(cell: CellRef, fn_: unsafe extern "C" fn(vertex: Vertex, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
  pub fn isl_cell_free(cell: Cell) -> ();
  pub fn isl_vertices_foreach_cell(vertices: VerticesRef, fn_: unsafe extern "C" fn(cell: Cell, user: *mut c_void) -> Stat, user: *mut c_void) -> Stat;
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Vertex(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct VertexRef(pub NonNull<c_void>);

impl Vertex {
  #[inline(always)]
  pub fn read(&self) -> Vertex { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Vertex) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<VertexRef> for Vertex {
  #[inline(always)]
  fn as_ref(&self) -> &VertexRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for Vertex {
  type Target = VertexRef;
  #[inline(always)]
  fn deref(&self) -> &VertexRef { self.as_ref() }
}

impl To<Option<Vertex>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Vertex> { NonNull::new(self).map(Vertex) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Cell(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct CellRef(pub NonNull<c_void>);

impl Cell {
  #[inline(always)]
  pub fn read(&self) -> Cell { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Cell) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<CellRef> for Cell {
  #[inline(always)]
  fn as_ref(&self) -> &CellRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for Cell {
  type Target = CellRef;
  #[inline(always)]
  fn deref(&self) -> &CellRef { self.as_ref() }
}

impl To<Option<Cell>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Cell> { NonNull::new(self).map(Cell) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Vertices(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct VerticesRef(pub NonNull<c_void>);

impl Vertices {
  #[inline(always)]
  pub fn read(&self) -> Vertices { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Vertices) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<VerticesRef> for Vertices {
  #[inline(always)]
  fn as_ref(&self) -> &VerticesRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for Vertices {
  type Target = VerticesRef;
  #[inline(always)]
  fn deref(&self) -> &VerticesRef { self.as_ref() }
}

impl To<Option<Vertices>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Vertices> { NonNull::new(self).map(Vertices) }
}

impl BasicSetRef {
  #[inline(always)]
  pub fn compute_vertices(self) -> Option<Vertices> {
    unsafe {
      let ret = isl_basic_set_compute_vertices(self.to());
      (ret).to()
    }
  }
}

impl Cell {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_cell_free(self.to());
      (ret).to()
    }
  }
}

impl CellRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_cell_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_domain(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_cell_get_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_vertex<F1: FnMut(Vertex) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(Vertex) -> Option<()>>(vertex: Vertex, user: *mut c_void) -> Stat { (*(user as *mut F))(vertex.to()).to() }
    unsafe {
      let ret = isl_cell_foreach_vertex(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
}

impl Vertex {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_vertex_free(self.to());
      (ret).to()
    }
  }
}

impl VertexRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_vertex_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_id(self) -> c_int {
    unsafe {
      let ret = isl_vertex_get_id(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_domain(self) -> Option<BasicSet> {
    unsafe {
      let ret = isl_vertex_get_domain(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_expr(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_vertex_get_expr(self.to());
      (ret).to()
    }
  }
}

impl Vertices {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_vertices_free(self.to());
      (ret).to()
    }
  }
}

impl VerticesRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_vertices_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_n_vertices(self) -> c_int {
    unsafe {
      let ret = isl_vertices_get_n_vertices(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_vertex<F1: FnMut(Vertex) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(Vertex) -> Option<()>>(vertex: Vertex, user: *mut c_void) -> Stat { (*(user as *mut F))(vertex.to()).to() }
    unsafe {
      let ret = isl_vertices_foreach_vertex(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn foreach_cell<F1: FnMut(Cell) -> Option<()>>(self, fn_: &mut F1) -> Option<()> {
    unsafe extern "C" fn fn1<F: FnMut(Cell) -> Option<()>>(cell: Cell, user: *mut c_void) -> Stat { (*(user as *mut F))(cell.to()).to() }
    unsafe {
      let ret = isl_vertices_foreach_cell(self.to(), fn1::<F1>, fn_ as *mut _ as _);
      (ret).to()
    }
  }
}

impl Drop for Cell {
  fn drop(&mut self) { Cell(self.0).free() }
}

impl Drop for Vertex {
  fn drop(&mut self) { Vertex(self.0).free() }
}

impl Drop for Vertices {
  fn drop(&mut self) { Vertices(self.0).free() }
}

