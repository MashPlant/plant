use crate::*;

extern "C" {
  pub fn isl_printer_to_file(ctx: CtxRef, file: *mut FILE) -> Option<Printer>;
  pub fn isl_printer_to_str(ctx: CtxRef) -> Option<Printer>;
  pub fn isl_printer_free(printer: Printer) -> *mut c_void;
  pub fn isl_printer_get_ctx(printer: PrinterRef) -> Option<CtxRef>;
  pub fn isl_printer_get_file(printer: PrinterRef) -> *mut FILE;
  pub fn isl_printer_get_str(printer: PrinterRef) -> Option<CString>;
  pub fn isl_printer_set_indent(p: Printer, indent: c_int) -> Option<Printer>;
  pub fn isl_printer_indent(p: Printer, indent: c_int) -> Option<Printer>;
  pub fn isl_printer_set_output_format(p: Printer, output_format: c_int) -> Option<Printer>;
  pub fn isl_printer_get_output_format(p: PrinterRef) -> c_int;
  pub fn isl_printer_set_yaml_style(p: Printer, yaml_style: c_int) -> Option<Printer>;
  pub fn isl_printer_get_yaml_style(p: PrinterRef) -> c_int;
  pub fn isl_printer_set_indent_prefix(p: Printer, prefix: CStr) -> Option<Printer>;
  pub fn isl_printer_set_prefix(p: Printer, prefix: CStr) -> Option<Printer>;
  pub fn isl_printer_set_suffix(p: Printer, suffix: CStr) -> Option<Printer>;
  pub fn isl_printer_set_isl_int_width(p: Printer, width: c_int) -> Option<Printer>;
  pub fn isl_printer_has_note(p: PrinterRef, id: IdRef) -> Bool;
  pub fn isl_printer_get_note(p: PrinterRef, id: Id) -> Option<Id>;
  pub fn isl_printer_set_note(p: Printer, id: Id, note: Id) -> Option<Printer>;
  pub fn isl_printer_start_line(p: Printer) -> Option<Printer>;
  pub fn isl_printer_end_line(p: Printer) -> Option<Printer>;
  pub fn isl_printer_print_double(p: Printer, d: c_double) -> Option<Printer>;
  pub fn isl_printer_print_int(p: Printer, i: c_int) -> Option<Printer>;
  pub fn isl_printer_print_str(p: Printer, s: CStr) -> Option<Printer>;
  pub fn isl_printer_yaml_start_mapping(p: Printer) -> Option<Printer>;
  pub fn isl_printer_yaml_end_mapping(p: Printer) -> Option<Printer>;
  pub fn isl_printer_yaml_start_sequence(p: Printer) -> Option<Printer>;
  pub fn isl_printer_yaml_end_sequence(p: Printer) -> Option<Printer>;
  pub fn isl_printer_yaml_next(p: Printer) -> Option<Printer>;
  pub fn isl_printer_flush(p: Printer) -> Option<Printer>;
}

impl CtxRef {
  #[inline(always)]
  pub fn printer_to_file(self, file: *mut FILE) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_to_file(self.to(), file.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn printer_to_str(self) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_to_str(self.to());
      (ret).to()
    }
  }
}

impl Printer {
  #[inline(always)]
  pub fn free(self) -> () {
    unsafe {
      let ret = isl_printer_free(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_indent(self, indent: c_int) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_set_indent(self.to(), indent.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn indent(self, indent: c_int) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_indent(self.to(), indent.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_output_format(self, output_format: c_int) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_set_output_format(self.to(), output_format.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_yaml_style(self, yaml_style: c_int) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_set_yaml_style(self.to(), yaml_style.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_indent_prefix(self, prefix: CStr) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_set_indent_prefix(self.to(), prefix.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_prefix(self, prefix: CStr) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_set_prefix(self.to(), prefix.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_suffix(self, suffix: CStr) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_set_suffix(self.to(), suffix.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_isl_int_width(self, width: c_int) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_set_isl_int_width(self.to(), width.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn set_note(self, id: Id, note: Id) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_set_note(self.to(), id.to(), note.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn start_line(self) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_start_line(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn end_line(self) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_end_line(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_double(self, d: c_double) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_double(self.to(), d.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_int(self, i: c_int) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_int(self.to(), i.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn print_str(self, s: CStr) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_print_str(self.to(), s.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn yaml_start_mapping(self) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_yaml_start_mapping(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn yaml_end_mapping(self) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_yaml_end_mapping(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn yaml_start_sequence(self) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_yaml_start_sequence(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn yaml_end_sequence(self) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_yaml_end_sequence(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn yaml_next(self) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_yaml_next(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn flush(self) -> Option<Printer> {
    unsafe {
      let ret = isl_printer_flush(self.to());
      (ret).to()
    }
  }
}

impl PrinterRef {
  #[inline(always)]
  pub fn get_ctx(self) -> Option<CtxRef> {
    unsafe {
      let ret = isl_printer_get_ctx(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_file(self) -> *mut FILE {
    unsafe {
      let ret = isl_printer_get_file(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_str(self) -> Option<CString> {
    unsafe {
      let ret = isl_printer_get_str(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_output_format(self) -> c_int {
    unsafe {
      let ret = isl_printer_get_output_format(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_yaml_style(self) -> c_int {
    unsafe {
      let ret = isl_printer_get_yaml_style(self.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn has_note(self, id: IdRef) -> Option<bool> {
    unsafe {
      let ret = isl_printer_has_note(self.to(), id.to());
      (ret).to()
    }
  }
  #[inline(always)]
  pub fn get_note(self, id: Id) -> Option<Id> {
    unsafe {
      let ret = isl_printer_get_note(self.to(), id.to());
      (ret).to()
    }
  }
}

impl Drop for Printer {
  fn drop(&mut self) { Printer(self.0).free() }
}

