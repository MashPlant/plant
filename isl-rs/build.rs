fn main() {
  println!("cargo:rustc-link-search=./isl-rs/isl/build/lib");
  println!("cargo:rustc-link-lib=static=isl");
  println!("cargo:rustc-link-lib=gmp");
}
