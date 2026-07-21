// Decode-FFI differential ORACLES — compiled for TESTS ONLY, never reachable
// from the production decode graph (which is pure-Rust end-to-end).
#[cfg(test)]
pub mod inflate_bit;
#[cfg(test)]
pub mod libdeflate;

// Increment 7: `isal_compress` (ISA-L *compression*, C-FFI) is OFF the
// production compress routing graph. It is retained as a differential oracle,
// compiled only for tests, the `ffi-oracle` feature, or an explicit
// `isal-compression` build (never the default binary → zero ISA-L compressor).
// `isal_decompress` is the ISA-L FFI *decode* oracle used by differential tests
// + the resumable from-bit oracle export in lib.rs; it is NOT on the production
// decode path (decode FFI is out of Increment-7 scope and stays as-is).
#[cfg(any(test, feature = "ffi-oracle", feature = "isal-compression"))]
pub mod isal_compress;
pub mod isal_decompress;
