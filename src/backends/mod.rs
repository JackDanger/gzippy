// Decode-FFI differential ORACLES — compiled for TESTS ONLY, never reachable
// from the production decode graph (which is pure-Rust end-to-end).
#[cfg(test)]
pub mod inflate_bit;
#[cfg(test)]
pub mod libdeflate;

// ISA-L COMPRESSION backend (production, x86_64 L0–L3). `isal_decompress` is the
// ISA-L FFI *decode* oracle used by differential tests + the resumable
// from-bit oracle export in lib.rs; it is NOT on the production decode path.
pub mod isal_compress;
pub mod isal_decompress;
pub mod zopfli_compress;
pub mod zopfli_pure;
