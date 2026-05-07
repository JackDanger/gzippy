//! Zopfli compression backend
//!
//! Provides FFI bindings to the vendored Google Zopfli C library for true zopfli compression.
//! Zopfli achieves slightly better compression than libdeflate's exhaustive search
//! at the cost of much longer runtime (suitable for L11).

use crate::cli::GzippyArgs;
use std::ffi::c_int;

const ZOPFLI_FORMAT_GZIP: c_int = 0;
#[allow(dead_code)]
const ZOPFLI_FORMAT_ZLIB: c_int = 1;
const ZOPFLI_FORMAT_DEFLATE: c_int = 2;

/// Zopfli compression options
#[repr(C)]
struct ZopfliOptions {
    verbose: c_int,
    verbose_more: c_int,
    numiterations: c_int,
    blocksplitting: c_int,
    blocksplittinglast: c_int,
    blocksplittingmax: c_int,
}

extern "C" {
    fn ZopfliInitOptions(options: *mut ZopfliOptions);
    fn ZopfliCompress(
        options: *const ZopfliOptions,
        output_type: c_int,
        input: *const u8,
        insize: usize,
        output: *mut *mut u8,
        outsize: *mut usize,
    );
}

/// Tuning parameters for zopfli compression
#[derive(Clone, Debug)]
pub struct ZopfliTuning {
    /// Number of iterations (default 15; higher = better compression, slower)
    pub iterations: u32,
    /// Enable block splitting (default true)
    pub block_splitting: bool,
    /// Maximum blocks to split into (default 15; 0 = unlimited)
    pub block_splitting_max: u32,
}

impl Default for ZopfliTuning {
    fn default() -> Self {
        Self {
            iterations: 15,
            block_splitting: true,
            block_splitting_max: 15,
        }
    }
}

impl ZopfliTuning {
    /// Create from CLI arguments
    pub fn from_args(args: &GzippyArgs) -> Self {
        Self {
            iterations: args.zopfli_iterations.unwrap_or(15),
            block_splitting: !args.zopfli_no_split,
            block_splitting_max: args.zopfli_split_max.unwrap_or(15),
        }
    }
}

/// Compress data with zopfli in GZIP format
pub fn compress_gzip(data: &[u8], tuning: &ZopfliTuning) -> Vec<u8> {
    compress_internal(data, tuning, ZOPFLI_FORMAT_GZIP)
}

/// Compress data with zopfli in raw DEFLATE format (no gzip header/trailer)
pub fn compress_deflate(data: &[u8], tuning: &ZopfliTuning) -> Vec<u8> {
    compress_internal(data, tuning, ZOPFLI_FORMAT_DEFLATE)
}

fn compress_internal(data: &[u8], tuning: &ZopfliTuning, format: c_int) -> Vec<u8> {
    unsafe {
        let mut opts = ZopfliOptions {
            verbose: 0,
            verbose_more: 0,
            numiterations: tuning.iterations as c_int,
            blocksplitting: if tuning.block_splitting { 1 } else { 0 },
            blocksplittinglast: 0,
            blocksplittingmax: tuning.block_splitting_max as c_int,
        };
        ZopfliInitOptions(&mut opts);

        let mut out: *mut u8 = std::ptr::null_mut();
        let mut outsize: usize = 0;

        ZopfliCompress(
            &opts,
            format,
            data.as_ptr(),
            data.len(),
            &mut out,
            &mut outsize,
        );

        if out.is_null() {
            return Vec::new();
        }

        // Copy C-allocated memory to Rust Vec
        let result = std::slice::from_raw_parts(out, outsize).to_vec();
        // Free the C-allocated memory
        libc::free(out as *mut std::ffi::c_void);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_gzip_basic() {
        let input = b"hello world";
        let tuning = ZopfliTuning::default();
        let output = compress_gzip(input, &tuning);

        // Verify output is valid gzip (starts with magic bytes)
        assert!(output.len() >= 18);
        assert_eq!(output[0], 0x1f);
        assert_eq!(output[1], 0x8b);
    }

    #[test]
    fn test_compress_deflate_basic() {
        let input = b"hello world";
        let tuning = ZopfliTuning::default();
        let output = compress_deflate(input, &tuning);

        // DEFLATE format doesn't have magic bytes, just verify output exists
        assert!(!output.is_empty());
        assert!(output.len() < input.len() * 2); // Reasonable bound
    }
}
