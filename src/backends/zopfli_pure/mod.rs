//! Pure-Rust port of Google Zopfli. Originally built bottom-up and
//! oracle-tested against the vendored C library; after the cutover the
//! permanent regression test in `tests.rs` is the long-term guard.

#![allow(dead_code)]

pub mod blocksplitter; // Step 13
pub mod cache; // Step 5
pub mod deflate; // Steps 14-15 (built incrementally)
pub mod deflate_size; // Step 9
pub mod gzip; // Step 16
pub mod hash; // Step 4
pub mod katajainen; // Step 2
pub mod lz77; // Steps 6-8 (built incrementally)
pub mod squeeze; // Steps 10-12 (built incrementally)
pub mod symbols; // Step 1
pub mod tree; // Step 3
pub mod zlib; // Step 17 (folded in from Step 20 — trivial port)

#[cfg(test)]
mod tests; // Step 23 — permanent regression fixtures (replace oracle harness)

/// Options used throughout the program. Mirrors C `ZopfliOptions`. Default
/// matches `ZopfliInitOptions`.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct ZopfliOptions {
    pub verbose: i32,
    pub verbose_more: i32,
    pub numiterations: i32,
    pub blocksplitting: i32,
    pub blocksplittinglast: i32,
    pub blocksplittingmax: i32,
}

impl Default for ZopfliOptions {
    fn default() -> Self {
        Self {
            verbose: 0,
            verbose_more: 0,
            numiterations: 15,
            blocksplitting: 1,
            blocksplittinglast: 0,
            blocksplittingmax: 15,
        }
    }
}

/// Output container format. Mirrors C `ZopfliFormat` from `zopfli.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZopfliFormat {
    Gzip,
    Zlib,
    Deflate,
}

/// Top-level dispatcher: wraps `in_` in the requested container and
/// returns the compressed bytes. Mirrors C `ZopfliCompress`.
pub fn compress(options: &ZopfliOptions, format: ZopfliFormat, in_: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    match format {
        ZopfliFormat::Gzip => gzip::gzip_compress(options, in_, &mut out),
        ZopfliFormat::Zlib => zlib::zlib_compress(options, in_, &mut out),
        ZopfliFormat::Deflate => {
            deflate::deflate(options, 2, true, in_, 0, &mut out);
        }
    }
    out
}
