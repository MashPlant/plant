fn main() {
  println!("cargo:rustc-link-search=./isl/build/lib");
  println!("cargo:rustc-link-lib=isl");
}
