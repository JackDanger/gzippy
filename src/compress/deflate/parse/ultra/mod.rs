//! The "crown" engine — a pure-Rust port of Google Zopfli, extended with
//! ECT-grade wins (LzFind BT4 full-Pareto matchfinder, multi-seed iterated
//! squeeze, uncapped exact-cost recursive block splitter). Output is pinned
//! to Google's reference implementation by the regression fixtures in
//! `tests.rs`; reached only via the CLI's `-F`/`-I`/`-J` flags (see
//! [`encoder::ZopfliGzEncoder`] and [`tuning::ZopfliTuning`]).
//!
//! See `docs/compressor-architecture.md` for the full module map and the
//! staged migration plan this module is Stage A of.

pub mod blocksplit;
pub mod cache;
pub mod deflate;
pub mod deflate_size;
pub mod encoder;
pub mod gzip;
pub mod hash;
pub mod lz77;
pub mod squeeze;
pub mod symbols;
mod tuning;
pub mod zlib;

pub use tuning::{compress_deflate, ZopfliTuning};

#[cfg(test)]
mod tests;

// Note: the `oracle` corpus test that compares against vendored C zopfli lives
// at `oracle_tests.rs` and is included only by `src/lib.rs` (via `#[path]`),
// not by `src/main.rs`. Declaring it inside this `mod compress;` tree would
// cause the bin's `mod compress;` to also compile it, and the resulting
// bin-test binary fails to link against `zopfli_oracle` (build.rs links it
// into the lib's targets via cc::Build, not into a transient bin test that
// doesn't directly use the package's library target).

/// Compression knobs. The first four fields mirror the *active* knobs in
/// Google Zopfli's `ZopfliOptions`. The C struct also has two more fields —
/// `blocksplittinglast` (which the C upstream documents as "No longer used,
/// left for compatibility") and `blocksplittingmax` (superseded here: the
/// splitter has been unconditionally uncapped since the crown-caps change,
/// see `deflate.rs`'s "UNCAP" comment, so a cap knob has no effect) — that we
/// omit deliberately; production code never reads either, and the
/// corpus-oracle FFI shim in `oracle_tests.rs` declares its own
/// `#[repr(C)]` mirror that matches the full C layout for the FFI call.
/// `thread_budget` is a Rust-only knob that bounds intra-block
/// parallelism so callers with their own outer pool don't oversubscribe
/// (`1` = serial, anything else = one thread per chunk).
#[derive(Clone, Debug)]
pub struct ZopfliOptions {
    pub verbose: i32,
    pub verbose_more: i32,
    pub numiterations: i32,
    pub blocksplitting: i32,
    pub thread_budget: u32,
}

impl Default for ZopfliOptions {
    fn default() -> Self {
        Self {
            verbose: 0,
            verbose_more: 0,
            numiterations: 15,
            blocksplitting: 1,
            thread_budget: 0,
        }
    }
}

/// Output container format. Mirrors C `ZopfliFormat` from `zopfli.h`.
/// `Gzip` and `Zlib` aren't reached from gzippy proper — `ZopfliGzEncoder`
/// writes its own gzip header and calls into the `Deflate` arm — but
/// they stay in the dispatcher because (a) they're the public surface of
/// this module for any other consumer, and (b) the regression fixtures
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
