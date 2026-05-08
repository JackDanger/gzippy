//! Pure-Rust port of Google Zopfli. Output is pinned to Google's reference
//! implementation by the regression fixtures in `tests.rs`.

pub mod blocksplitter;
pub mod cache;
pub mod deflate;
pub mod deflate_size;
pub mod gzip;
pub mod hash;
pub mod katajainen;
pub mod lz77;
pub mod squeeze;
pub mod symbols;
pub mod tree;
pub mod zlib;

#[cfg(test)]
mod tests;

/// Compression knobs. The first five fields mirror Google Zopfli's
/// `ZopfliOptions`; `thread_budget` is a Rust-only knob that bounds
/// intra-block parallelism so callers with their own outer pool don't
/// oversubscribe (`1` = serial, anything else = one thread per chunk).
#[derive(Clone, Debug)]
pub struct ZopfliOptions {
    pub verbose: i32,
    pub verbose_more: i32,
    pub numiterations: i32,
    pub blocksplitting: i32,
    pub blocksplittingmax: i32,
    pub thread_budget: u32,
}

impl Default for ZopfliOptions {
    fn default() -> Self {
        Self {
            verbose: 0,
            verbose_more: 0,
            numiterations: 15,
            blocksplitting: 1,
            blocksplittingmax: 15,
            thread_budget: 0,
        }
    }
}

/// Output container format. Mirrors C `ZopfliFormat` from `zopfli.h`.
/// `Zlib` isn't reached from gzippy proper — it stays in the dispatcher
/// (and is exercised by `tests::fixture_alphabet_zlib_iter15`) so the
/// crate still presents the full zopfli surface to other consumers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZopfliFormat {
    Gzip,
    #[allow(dead_code)]
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
