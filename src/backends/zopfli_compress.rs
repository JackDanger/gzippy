//! Zopfli compression backend.
//!
//! Thin tuning wrapper around the pure-Rust port at
//! [`crate::backends::zopfli_pure`]. Zopfli achieves slightly better
//! compression than libdeflate's exhaustive search at the cost of much
//! longer runtime (suitable for L11).

use crate::backends::zopfli_pure::{compress, ZopfliFormat, ZopfliOptions};
use crate::cli::GzippyArgs;

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

/// Compress data with zopfli in GZIP format.
pub fn compress_gzip(data: &[u8], tuning: &ZopfliTuning) -> Vec<u8> {
    compress(&tuning_to_options(tuning), ZopfliFormat::Gzip, data)
}

/// Compress data with zopfli in raw DEFLATE format (no gzip header/trailer).
pub fn compress_deflate(data: &[u8], tuning: &ZopfliTuning) -> Vec<u8> {
    compress(&tuning_to_options(tuning), ZopfliFormat::Deflate, data)
}

fn tuning_to_options(tuning: &ZopfliTuning) -> ZopfliOptions {
    ZopfliOptions {
        verbose: 0,
        verbose_more: 0,
        numiterations: tuning.iterations as i32,
        blocksplitting: if tuning.block_splitting { 1 } else { 0 },
        blocksplittinglast: 0,
        blocksplittingmax: tuning.block_splitting_max as i32,
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
