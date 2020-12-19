use crate::*;

extern "C" {
  pub fn isl_version() -> Option<CStr>;
}

