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

// Note: the `oracle` corpus test that compares against vendored C zopfli lives
// at `src/oracle_tests.rs` and is included only by `src/lib.rs` (via
// `#[path]`), not by `src/main.rs`. Declaring it inside this `mod backends;`
// tree would cause the bin's `mod backends;` to also compile it, and the
// resulting bin-test binary fails to link against `zopfli_oracle` (build.rs
// links it into the lib's targets via cc::Build, not into a transient bin
// test that doesn't directly use the package's library target).

/// Compression knobs. The first five fields mirror the *active* knobs in
/// Google Zopfli's `ZopfliOptions`. The C struct also has a sixth field
/// `blocksplittinglast` that the C upstream documents as "No longer used,
/// left for compatibility"; we omit it deliberately — production code
/// never reads it and the corpus-oracle FFI shim in `oracle_tests.rs`
/// declares its own `#[repr(C)]` mirror that matches the full C layout.
/// `thread_budget` is a Rust-only knob that bounds intra-block
/// parallelism so callers with their own outer pool don't oversubscribe
/// (`1` = serial, anything else = one thread per chunk).
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
/// `Gzip` and `Zlib` aren't reached from gzippy proper — `ZopfliGzEncoder`
/// writes its own gzip header and calls into the `Deflate` arm — but
/// they stay in the dispatcher because (a) they're the public surface of
/// `zopfli_pure` for any other consumer, and (b) the regression fixtures
/// in `tests` and the Phase 11.2 corpus oracle use them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZopfliFormat {
    #[allow(dead_code)]
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
