use crate::*;

extern "C" {
  pub fn isl_mat_get_ctx(mat: MatRef) -> Option<CtxRef>;
  pub fn isl_mat_alloc(ctx: CtxRef, n_row: c_uint, n_col: c_uint) -> Option<Mat>;
  pub fn isl_mat_extend(mat: Mat, n_row: c_uint, n_col: c_uint) -> Option<Mat>;
  pub fn isl_mat_identity(ctx: CtxRef, n_row: c_uint) -> Option<Mat>;
  pub fn isl_mat_copy(mat: MatRef) -> Option<Mat>;
  pub fn isl_mat_free(mat: Mat) -> *mut c_void;
  pub fn isl_mat_rows(mat: MatRef) -> c_int;
  pub fn isl_mat_cols(mat: MatRef) -> c_int;
  pub fn isl_mat_get_element_val(mat: MatRef, row: c_int, col: c_int) -> Option<Val>;
  pub fn isl_mat_set_element_si(mat: Mat, row: c_int, col: c_int, v: c_int) -> Option<Mat>;
  pub fn isl_mat_set_element_val(mat: Mat, row: c_int, col: c_int, v: Val) -> Option<Mat>;
  pub fn isl_mat_swap_cols(mat: Mat, i: c_uint, j: c_uint) -> Option<Mat>;
  pub fn isl_mat_swap_rows(mat: Mat, i: c_uint, j: c_uint) -> Option<Mat>;
  pub fn isl_mat_vec_product(mat: Mat, vec: Vec) -> Option<Vec>;
  pub fn isl_vec_mat_product(vec: Vec, mat: Mat) -> Option<Vec>;
  pub fn isl_mat_vec_inverse_product(mat: Mat, vec: Vec) -> Option<Vec>;
  pub fn isl_mat_aff_direct_sum(left: Mat, right: Mat) -> Option<Mat>;
  pub fn isl_mat_diagonal(mat1: Mat, mat2: Mat) -> Option<Mat>;
  pub fn isl_mat_left_hermite(M: Mat, neg: c_int, U: *mut Mat, Q: *mut Mat) -> Option<Mat>;
  pub fn isl_mat_lin_to_aff(mat: Mat) -> Option<Mat>;
  pub fn isl_mat_inverse_product(left: Mat, right: Mat) -> Option<Mat>;
  pub fn isl_mat_product(left: Mat, right: Mat) -> Option<Mat>;
  pub fn isl_mat_transpose(mat: Mat) -> Option<Mat>;
  pub fn isl_mat_right_inverse(mat: Mat) -> Option<Mat>;
  pub fn isl_mat_right_kernel(mat: Mat) -> Option<Mat>;
  pub fn isl_mat_normalize(mat: Mat) -> Option<Mat>;
  pub fn isl_mat_normalize_row(mat: Mat, row: c_int) -> Option<Mat>;
  pub fn isl_mat_drop_cols(mat: Mat, col: c_uint, n: c_uint) -> Option<Mat>;
  pub fn isl_mat_drop_rows(mat: Mat, row: c_uint, n: c_uint) -> Option<Mat>;
  pub fn isl_mat_insert_cols(mat: Mat, col: c_uint, n: c_uint) -> Option<Mat>;
  pub fn isl_mat_insert_rows(mat: Mat, row: c_uint, n: c_uint) -> Option<Mat>;
  pub fn isl_mat_move_cols(mat: Mat, dst_col: c_uint, src_col: c_uint, n: c_uint) -> Option<Mat>;
  pub fn isl_mat_add_rows(mat: Mat, n: c_uint) -> Option<Mat>;
  pub fn isl_mat_insert_zero_cols(mat: Mat, first: c_uint, n: c_uint) -> Option<Mat>;
  pub fn isl_mat_add_zero_cols(mat: Mat, n: c_uint) -> Option<Mat>;
  pub fn isl_mat_insert_zero_rows(mat: Mat, row: c_uint, n: c_uint) -> Option<Mat>;
  pub fn isl_mat_add_zero_rows(mat: Mat, n: c_uint) -> Option<Mat>;
  pub fn isl_mat_col_add(mat: MatRef, dst_col: c_int, src_col: c_int) -> ();
  pub fn isl_mat_unimodular_complete(M: Mat, row: c_int) -> Option<Mat>;
  pub fn isl_mat_row_basis(mat: Mat) -> Option<Mat>;
  pub fn isl_mat_row_basis_extension(mat1: Mat, mat2: Mat) -> Option<Mat>;
  pub fn isl_mat_from_row_vec(vec: Vec) -> Option<Mat>;
  pub fn isl_mat_concat(top: Mat, bot: Mat) -> Option<Mat>;
  pub fn isl_mat_vec_concat(top: Mat, bot: Vec) -> Option<Mat>;
  pub fn isl_mat_is_equal(mat1: MatRef, mat2: MatRef) -> Bool;
  pub fn isl_mat_has_linearly_independent_rows(mat1: MatRef, mat2: MatRef) -> Bool;
  pub fn isl_mat_rank(mat: MatRef) -> c_int;
  pub fn isl_mat_initial_non_zero_cols(mat: MatRef) -> c_int;
  pub fn isl_mat_print_internal(mat: MatRef, out: *mut FILE, indent: c_int) -> ();
  pub fn isl_mat_dump(mat: MatRef) -> ();
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Mat(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct MatRef(pub NonNull<c_void>);

impl Mat {
  #[inline(always)]
  pub fn read(&self) -> Mat { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Mat) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<MatRef> for Mat {
  #[inline(always)]
  fn as_ref(&self) -> &MatRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for Mat {
  type Target = MatRef;
  #[inline(always)]
  fn deref(&self) -> &MatRef { self.as_ref() }
}

impl To<Option<Mat>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Mat> { NonNull::new(self).map(Mat) }
}

impl CtxRef {
  #[inline(always)]
  pub fn mat_alloc(self, n_row: c_uint, n_col: c_uint) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_alloc(self.to(), n_row.to(), n_col.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mat_identity(self, n_row: c_uint) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_identity(self.to(), n_row.to());
      (ret).to()
    }
  }
}

impl Mat {
  #[inline(always)]
  pub fn extend(self, n_row: c_uint, n_col: c_uint) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_extend(self.to(), n_row.to(), n_col.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_mat_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_element_si(self, row: c_int, col: c_int, v: c_int) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_set_element_si(self.to(), row.to(), col.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_element_val(self, row: c_int, col: c_int, v: Val) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_set_element_val(self.to(), row.to(), col.to(), v.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn swap_cols(self, i: c_uint, j: c_uint) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_swap_cols(self.to(), i.to(), j.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn swap_rows(self, i: c_uint, j: c_uint) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_swap_rows(self.to(), i.to(), j.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn vec_product(self, vec: Vec) -> Option<Vec> {
    unsafe {
      let ret = isl_mat_vec_product(self.to(), vec.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn vec_inverse_product(self, vec: Vec) -> Option<Vec> {
    unsafe {
      let ret = isl_mat_vec_inverse_product(self.to(), vec.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn aff_direct_sum(self, right: Mat) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_aff_direct_sum(self.to(), right.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn diagonal(self, mat2: Mat) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_diagonal(self.to(), mat2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn left_hermite(self, neg: c_int) -> Option<(Mat, Mat, Mat)> {
    unsafe {
      let ref mut U = 0 as *mut c_void;
      let ref mut Q = 0 as *mut c_void;
      let ret = isl_mat_left_hermite(self.to(), neg.to(), U as *mut _ as _, Q as *mut _ as _);
      (ret, *U, *Q).to()
    }
  }
  #[inline(always)]
  pub fn lin_to_aff(self) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_lin_to_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn inverse_product(self, right: Mat) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_inverse_product(self.to(), right.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn product(self, right: Mat) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_product(self.to(), right.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn transpose(self) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_transpose(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn right_inverse(self) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_right_inverse(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn right_kernel(self) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_right_kernel(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn normalize(self) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_normalize(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn normalize_row(self, row: c_int) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_normalize_row(self.to(), row.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_cols(self, col: c_uint, n: c_uint) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_drop_cols(self.to(), col.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn drop_rows(self, row: c_uint, n: c_uint) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_drop_rows(self.to(), row.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_cols(self, col: c_uint, n: c_uint) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_insert_cols(self.to(), col.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_rows(self, row: c_uint, n: c_uint) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_insert_rows(self.to(), row.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn move_cols(self, dst_col: c_uint, src_col: c_uint, n: c_uint) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_move_cols(self.to(), dst_col.to(), src_col.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_rows(self, n: c_uint) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_add_rows(self.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_zero_cols(self, first: c_uint, n: c_uint) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_insert_zero_cols(self.to(), first.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_zero_cols(self, n: c_uint) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_add_zero_cols(self.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn insert_zero_rows(self, row: c_uint, n: c_uint) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_insert_zero_rows(self.to(), row.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn add_zero_rows(self, n: c_uint) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_add_zero_rows(self.to(), n.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn unimodular_complete(self, row: c_int) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_unimodular_complete(self.to(), row.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn row_basis(self) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_row_basis(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn row_basis_extension(self, mat2: Mat) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_row_basis_extension(self.to(), mat2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn concat(self, bot: Mat) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_concat(self.to(), bot.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn vec_concat(self, bot: Vec) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_vec_concat(self.to(), bot.to());
      (ret).to()
    }
  }
}

impl MatRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_mat_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn copy(self) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_copy(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn rows(self) -> c_int {
    unsafe {
      let ret = isl_mat_rows(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn cols(self) -> c_int {
    unsafe {
      let ret = isl_mat_cols(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_element_val(self, row: c_int, col: c_int) -> Option<Val> {
    unsafe {
      let ret = isl_mat_get_element_val(self.to(), row.to(), col.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn col_add(self, dst_col: c_int, src_col: c_int) -> () {
    unsafe {
      let ret = isl_mat_col_add(self.to(), dst_col.to(), src_col.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_equal(self, mat2: MatRef) -> Option<bool> {
    unsafe {
      let ret = isl_mat_is_equal(self.to(), mat2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_linearly_independent_rows(self, mat2: MatRef) -> Option<bool> {
    unsafe {
      let ret = isl_mat_has_linearly_independent_rows(self.to(), mat2.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn rank(self) -> c_int {
    unsafe {
      let ret = isl_mat_rank(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn initial_non_zero_cols(self) -> c_int {
    unsafe {
      let ret = isl_mat_initial_non_zero_cols(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_internal(self, out: *mut FILE, indent: c_int) -> () {
    unsafe {
      let ret = isl_mat_print_internal(self.to(), out.to(), indent.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn dump(self) -> () {
    unsafe {
      let ret = isl_mat_dump(self.to());
      (ret).to()
    }
  }
}

impl Vec {
  #[inline(always)]
  pub fn mat_product(self, mat: Mat) -> Option<Vec> {
    unsafe {
      let ret = isl_vec_mat_product(self.to(), mat.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn mat_from_row_vec(self) -> Option<Mat> {
    unsafe {
      let ret = isl_mat_from_row_vec(self.to());
      (ret).to()
    }
  }
}

impl Drop for Mat {
  fn drop(&mut self) { Mat(self.0).free() }
}

