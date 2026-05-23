//! Compile-time gates for the parallel single-member production path.

/// Parallel SM orchestration (chunk_fetcher, block_finder, etc.) is available.
pub const PARALLEL_SM: bool = cfg!(all(
    target_arch = "x86_64",
    any(feature = "isal-compression", feature = "pure-rust-inflate")
));

/// Post-bootstrap + bootstrap DYNAMIC table build use patched ISA-L C code.
#[allow(dead_code)]
pub const USE_ISAL_INFLATE: bool = cfg!(all(
    feature = "isal-compression",
    not(feature = "pure-rust-inflate"),
    target_arch = "x86_64"
));
