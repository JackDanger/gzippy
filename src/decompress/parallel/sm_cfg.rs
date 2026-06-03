//! Compile-time gates for the parallel single-member production path.

/// True when the parallel-SM inner decode is pure Rust (no ISA-L C in the wrapper).
#[allow(dead_code)]
pub const PURE_RUST_INFLATE_DECODE: bool = cfg!(pure_inflate_decode);

/// Parallel SM orchestration (chunk_fetcher, block_finder, etc.) is available.
///
/// The `parallel_sm` cfg is emitted by `build.rs::emit_parallel_sm_cfgs` and is
/// true when EITHER `x86_64 + (isal-compression | pure-rust-inflate)` OR
/// `aarch64 + pure-rust-inflate`. arm64 always uses the pure-Rust inner decoder
/// (ISA-L's C library is x86-only); see `inflate_wrapper::IsalInflateWrapper`'s
/// `#[cfg(pure_inflate_decode)]` backend.
pub const PARALLEL_SM: bool = cfg!(parallel_sm);

/// Post-bootstrap + bootstrap DYNAMIC table build use patched ISA-L C code.
#[allow(dead_code)]
pub const USE_ISAL_INFLATE: bool = cfg!(all(
    feature = "isal-compression",
    not(feature = "pure-rust-inflate"),
    target_arch = "x86_64"
));
