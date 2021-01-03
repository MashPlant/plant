use crate::*;

extern "C" {
  pub fn isl_token_get_val(ctx: CtxRef, tok: TokenRef) -> Option<Val>;
  pub fn isl_token_get_str(ctx: CtxRef, tok: TokenRef) -> Option<CString>;
  pub fn isl_token_get_type(tok: TokenRef) -> c_int;
  pub fn isl_token_free(tok: Token) -> ();
  pub fn isl_stream_new_file(ctx: CtxRef, file: *mut FILE) -> Option<Stream>;
  pub fn isl_stream_new_str(ctx: CtxRef, str: CStr) -> Option<Stream>;
  pub fn isl_stream_free(s: Stream) -> ();
  pub fn isl_stream_get_ctx(s: StreamRef) -> Option<CtxRef>;
  pub fn isl_stream_error(s: StreamRef, tok: TokenRef, msg: CStr) -> ();
  pub fn isl_stream_next_token(s: StreamRef) -> Option<Token>;
  pub fn isl_stream_next_token_on_same_line(s: StreamRef) -> Option<Token>;
  pub fn isl_stream_next_token_is(s: StreamRef, type_: c_int) -> c_int;
  pub fn isl_stream_push_token(s: StreamRef, tok: Token) -> ();
  pub fn isl_stream_flush_tokens(s: StreamRef) -> ();
  pub fn isl_stream_eat_if_available(s: StreamRef, type_: c_int) -> c_int;
  pub fn isl_stream_read_ident_if_available(s: StreamRef) -> Option<CString>;
  pub fn isl_stream_eat(s: StreamRef, type_: c_int) -> c_int;
  pub fn isl_stream_is_empty(s: StreamRef) -> c_int;
  pub fn isl_stream_skip_line(s: StreamRef) -> c_int;
  pub fn isl_stream_register_keyword(s: StreamRef, name: CStr) -> TokenType;
  pub fn isl_stream_read_obj(s: StreamRef) -> Obj;
  pub fn isl_stream_read_val(s: StreamRef) -> Option<Val>;
  pub fn isl_stream_read_multi_aff(s: StreamRef) -> Option<MultiAff>;
  pub fn isl_stream_read_map(s: StreamRef) -> Option<Map>;
  pub fn isl_stream_read_set(s: StreamRef) -> Option<Set>;
  pub fn isl_stream_read_pw_qpolynomial(s: StreamRef) -> Option<PwQpolynomial>;
  pub fn isl_stream_read_union_set(s: StreamRef) -> Option<UnionSet>;
  pub fn isl_stream_read_union_map(s: StreamRef) -> Option<UnionMap>;
  pub fn isl_stream_read_schedule(s: StreamRef) -> Option<Schedule>;
  pub fn isl_stream_yaml_read_start_mapping(s: StreamRef) -> c_int;
  pub fn isl_stream_yaml_read_end_mapping(s: StreamRef) -> c_int;
  pub fn isl_stream_yaml_read_start_sequence(s: StreamRef) -> c_int;
  pub fn isl_stream_yaml_read_end_sequence(s: StreamRef) -> c_int;
  pub fn isl_stream_yaml_next(s: StreamRef) -> c_int;
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Token(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct TokenRef(pub NonNull<c_void>);

impl Token {
  #[inline(always)]
  pub fn read(&self) -> Token { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Token) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<TokenRef> for Token {
  #[inline(always)]
  fn as_ref(&self) -> &TokenRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for Token {
  type Target = TokenRef;
  #[inline(always)]
  fn deref(&self) -> &TokenRef { self.as_ref() }
}

impl To<Option<Token>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Token> { NonNull::new(self).map(Token) }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Stream(pub NonNull<c_void>);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct StreamRef(pub NonNull<c_void>);

impl Stream {
  #[inline(always)]
  pub fn read(&self) -> Stream { unsafe { ptr::read(self) } }
  #[inline(always)]
  pub fn write(&self, x: Stream) { unsafe { ptr::write(self as *const _ as _, x) } }
}

impl AsRef<StreamRef> for Stream {
  #[inline(always)]
  fn as_ref(&self) -> &StreamRef { unsafe { mem::transmute(self) } }
}

impl std::ops::Deref for Stream {
  type Target = StreamRef;
  #[inline(always)]
  fn deref(&self) -> &StreamRef { self.as_ref() }
}

impl To<Option<Stream>> for *mut c_void {
  #[inline(always)]
  unsafe fn to(self) -> Option<Stream> { NonNull::new(self).map(Stream) }
}

impl CtxRef {
  #[inline(always)]
  pub fn token_get_val(self, tok: TokenRef) -> Option<Val> {
    unsafe {
      let ret = isl_token_get_val(self.to(), tok.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn token_get_str(self, tok: TokenRef) -> Option<CString> {
    unsafe {
      let ret = isl_token_get_str(self.to(), tok.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn stream_new_file(self, file: *mut FILE) -> Option<Stream> {
    unsafe {
      let ret = isl_stream_new_file(self.to(), file.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn stream_new_str(self, str: CStr) -> Option<Stream> {
    unsafe {
      let ret = isl_stream_new_str(self.to(), str.to());
      (ret).to()
    }
  }
}

impl Stream {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_stream_free(self.to());
      (ret).to()
    }
  }
}

impl StreamRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_stream_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn error(self, tok: TokenRef, msg: CStr) -> () {
    unsafe {
      let ret = isl_stream_error(self.to(), tok.to(), msg.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn next_token(self) -> Option<Token> {
    unsafe {
      let ret = isl_stream_next_token(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn next_token_on_same_line(self) -> Option<Token> {
    unsafe {
      let ret = isl_stream_next_token_on_same_line(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn next_token_is(self, type_: c_int) -> c_int {
    unsafe {
      let ret = isl_stream_next_token_is(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn push_token(self, tok: Token) -> () {
    unsafe {
      let ret = isl_stream_push_token(self.to(), tok.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flush_tokens(self) -> () {
    unsafe {
      let ret = isl_stream_flush_tokens(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eat_if_available(self, type_: c_int) -> c_int {
    unsafe {
      let ret = isl_stream_eat_if_available(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn read_ident_if_available(self) -> Option<CString> {
    unsafe {
      let ret = isl_stream_read_ident_if_available(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn eat(self, type_: c_int) -> c_int {
    unsafe {
      let ret = isl_stream_eat(self.to(), type_.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn is_empty(self) -> c_int {
    unsafe {
      let ret = isl_stream_is_empty(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn skip_line(self) -> c_int {
    unsafe {
      let ret = isl_stream_skip_line(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn register_keyword(self, name: CStr) -> TokenType {
    unsafe {
      let ret = isl_stream_register_keyword(self.to(), name.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn read_obj(self) -> Obj {
    unsafe {
      let ret = isl_stream_read_obj(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn read_val(self) -> Option<Val> {
    unsafe {
      let ret = isl_stream_read_val(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn read_multi_aff(self) -> Option<MultiAff> {
    unsafe {
      let ret = isl_stream_read_multi_aff(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn read_map(self) -> Option<Map> {
    unsafe {
      let ret = isl_stream_read_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn read_set(self) -> Option<Set> {
    unsafe {
      let ret = isl_stream_read_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn read_pw_qpolynomial(self) -> Option<PwQpolynomial> {
    unsafe {
      let ret = isl_stream_read_pw_qpolynomial(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn read_union_set(self) -> Option<UnionSet> {
    unsafe {
      let ret = isl_stream_read_union_set(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn read_union_map(self) -> Option<UnionMap> {
    unsafe {
      let ret = isl_stream_read_union_map(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn read_schedule(self) -> Option<Schedule> {
    unsafe {
      let ret = isl_stream_read_schedule(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn yaml_read_start_mapping(self) -> c_int {
    unsafe {
      let ret = isl_stream_yaml_read_start_mapping(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn yaml_read_end_mapping(self) -> c_int {
    unsafe {
      let ret = isl_stream_yaml_read_end_mapping(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn yaml_read_start_sequence(self) -> c_int {
    unsafe {
      let ret = isl_stream_yaml_read_start_sequence(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn yaml_read_end_sequence(self) -> c_int {
    unsafe {
      let ret = isl_stream_yaml_read_end_sequence(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn yaml_next(self) -> c_int {
    unsafe {
      let ret = isl_stream_yaml_next(self.to());
      (ret).to()
    }
  }
}

impl Token {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_token_free(self.to());
      (ret).to()
    }
  }
}

impl TokenRef {
  #[inline(always)]
  pub fn get_type(self) -> c_int {
    unsafe {
      let ret = isl_token_get_type(self.to());
      (ret).to()
    }
  }
}

impl Drop for Stream {
  fn drop(&mut self) { Stream(self.0).free() }
}

impl Drop for Token {
  fn drop(&mut self) { Token(self.0).free() }
}

