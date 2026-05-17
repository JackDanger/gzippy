#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod igzip_lib;

/// Direct FFI to ISA-L internals that the upstream `igzip_lib.h` header
/// doesn't expose but our patched `igzip_inflate.c` does (the `static`
/// keyword was stripped from the relevant function definitions).
///
/// Mirrors the symbols rapidgzip uses in
/// `vendor/rapidgzip/.../huffman/HuffmanCodingISAL.hpp` for ~340 MB/s/thread
/// Huffman decode (vs ~14 MB/s for pure-Rust).
pub mod isal_internals {
    use crate::igzip_lib::bindings::inflate_huff_code_large;

    /// Mirror of ISA-L's `struct huff_code` (huff_codes.h:105). Union
    /// packing all subfields into a single 32-bit slot. Access via
    /// `code_and_length`; `make_inflate_huff_code_lit_len` reads/writes
    /// internally.
    #[repr(C)]
    #[derive(Copy, Clone, Default)]
    pub struct huff_code {
        pub code_and_length: u32,
    }

    unsafe extern "C" {
        /// `set_and_expand_lit_len_huffcode` (igzip_inflate.c:281 after
        /// patch). Builds the huff_code_table from the count/expand_count
        /// histograms and emits a sorted code_list. Returns 0 on success,
        /// non-zero on failure.
        pub fn set_and_expand_lit_len_huffcode(
            lit_len_huff: *mut huff_code,
            table_length: u32,
            count: *mut u16,
            expand_count: *mut u16,
            code_list: *mut u32,
        ) -> i32;

        /// `make_inflate_huff_code_lit_len` (igzip_inflate.c:387 after
        /// patch). Populates `result` with the short_code_lookup +
        /// long_code_lookup tables that ISA-L uses for fast decode.
        pub fn make_inflate_huff_code_lit_len(
            result: *mut inflate_huff_code_large,
            huff_code_table: *mut huff_code,
            table_length: u32,
            count_total: *const u16,
            code_list: *mut u32,
            multisym: u32,
        );
    }
}
