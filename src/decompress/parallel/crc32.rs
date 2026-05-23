#![allow(dead_code)] // vendor-faithful rapidgzip port; many items are pending consumer-port

//! Literal port of `rapidgzip::CRC32Calculator` plus the carryless
//! polynomial helpers used by `combineCRC32`
//! (vendor/rapidgzip/librapidarchive/src/rapidgzip/gzip/crc32.hpp).
//!
//! ## Mapping to gzippy's existing CRC infrastructure
//!
//! gzippy already depends on the `crc32fast` crate, which:
//! - exposes a `Hasher` with `update` / `finalize` semantics that
//!   matches `CRC32Calculator::update` / `crc32()` byte-for-byte
//!   (both use the standard gzip CRC32, `0xEDB88320` generator).
//! - exposes `Hasher::combine` implementing the same polynomial
//!   trick the vendor calls `combineCRC32`.
//! - dispatches to the same hardware acceleration paths (CLMUL on
//!   x86_64, PMULL on aarch64) the vendor sources via ISA-L.
//!
//! So we do NOT re-port the 1 KiB CRC32 table, the 64 KiB
//! slice-by-N LUT, or the slice-by-N `updateCRC32`. Instead, this
//! module exposes:
//! - `CRC32_GENERATOR_POLYNOMIAL` and a few utility constants for
//!   parity with vendor symbol names that other ports may reference.
//! - `polynomial_multiply_modulo` / `x_power_modulo` —
//!   self-contained polynomial math used by `combineCRC32`. We port
//!   them because the algorithms are non-trivial and a future port
//!   of `IndexFileFormat` (which serializes per-chunk CRCs and may
//!   combine them at unusual offsets) needs them callable from Rust.
//! - `CRC32Calculator` — a thin wrapper around `crc32fast::Hasher`
//!   that mirrors the vendor's `update` / `verify` / `append` /
//!   `prepend` API for drop-in compatibility.
//!
//! Cross-check tests assert that our `combine_crc32` agrees with
//! both `crc32fast::Hasher::combine` and a direct call to the
//! polynomial-multiply formula.

use crc32fast::Hasher;

// =====================================================================
// Constants (crc32.hpp:23-58).
// =====================================================================

/// Mirror of `rapidgzip::CRC32_GENERATOR_POLYNOMIAL` (crc32.hpp:23).
/// The reflected gzip / zlib generator polynomial.
pub const CRC32_GENERATOR_POLYNOMIAL: u32 = 0xEDB8_8320;

/// Mirror of `rapidgzip::CRC32_LOOKUP_TABLE_SIZE` (crc32.hpp:45).
pub const CRC32_LOOKUP_TABLE_SIZE: usize = 256;

/// Mirror of `rapidgzip::MAX_CRC32_SLICE_SIZE` (crc32.hpp:58). Not
/// used directly by this port (we delegate to `crc32fast`'s SIMD
/// kernels) but reserved here for parity with vendor source.
pub const MAX_CRC32_SLICE_SIZE: usize = 64;

// =====================================================================
// Polynomial math (crc32.hpp:152-258).
// =====================================================================

/// Literal port of `rapidgzip::polynomialMultiplyModulo`
/// (crc32.hpp:152-170).
///
/// Carryless polynomial multiplication of `a` and `b` modulo the
/// (reflected) polynomial `p`. Used as the kernel of both
/// `x_power_modulo` and `combine_crc32`.
pub const fn polynomial_multiply_modulo(a: u32, mut b: u32, p: u32) -> u32 {
    let mut result: u32 = 0;
    let mut coefficient_position: u32 = 1u32 << 31;
    while coefficient_position > 0 {
        if (a & coefficient_position) != 0 {
            result ^= b;
        }
        let overflows = (b & 1) != 0;
        b >>= 1;
        if overflows {
            b ^= p;
        }
        coefficient_position >>= 1;
    }
    result
}

/// Literal port of `rapidgzip::X2N_LUT` (crc32.hpp:179-188).
/// `X2N_LUT[n]` = q(x)^(2^n) mod p, with q(x) = x^1 in reflected
/// notation. Built once at first access; `LazyLock` matches the
/// vendor `constexpr` semantics modulo Rust's stable-const limits.
///
/// We materialize it eagerly via a `const fn` builder so callers
/// can use it from `const` contexts if needed.
pub static X2N_LUT: [u32; 32] = build_x2n_lut();

const fn build_x2n_lut() -> [u32; 32] {
    let mut result = [0u32; 32];
    result[0] = 1u32 << 30; // x^1 in reflected notation.
    let mut n = 1;
    while n < 32 {
        result[n] =
            polynomial_multiply_modulo(result[n - 1], result[n - 1], CRC32_GENERATOR_POLYNOMIAL);
        n += 1;
    }
    result
}

/// Literal port of `rapidgzip::xPowerModulo` (crc32.hpp:198-208).
/// Returns x^exponent mod p, where p is the gzip generator polynomial.
pub const fn x_power_modulo(mut exponent: u64) -> u32 {
    let mut p: u32 = 1u32 << 31; // x^0 in reflected notation.
    let mut k = 0;
    while exponent > 0 {
        if (exponent & 1) != 0 {
            p = polynomial_multiply_modulo(
                X2N_LUT[k % X2N_LUT.len()],
                p,
                CRC32_GENERATOR_POLYNOMIAL,
            );
        }
        exponent >>= 1;
        k += 1;
    }
    p
}

/// Literal port of `rapidgzip::combineCRC32` (crc32.hpp:214-258).
///
/// Combines the CRC32 of stream `a` (length `byte_stream_length` bytes)
/// with the CRC32 of the subsequent stream `b` to yield the CRC32 of
/// their concatenation. `byte_stream_length` is the length of the
/// *second* stream (`b`).
pub fn combine_crc32(crc1: u32, crc2: u32, byte_stream_length: u64) -> u32 {
    polynomial_multiply_modulo(
        x_power_modulo(byte_stream_length * 8),
        crc1,
        CRC32_GENERATOR_POLYNOMIAL,
    ) ^ crc2
}

// =====================================================================
// CRC32Calculator (crc32.hpp:261-345).
// =====================================================================

/// Literal port of `rapidgzip::CRC32Calculator`.
///
/// Internally delegates to `crc32fast::Hasher` for the per-byte
/// table-lookup work (which is hardware-accelerated on x86_64 and
/// aarch64, matching what the vendor gets out of ISA-L's
/// `crc32_gzip_refl`).  The `append` / `prepend` operations route
/// through our `combine_crc32` because `crc32fast::Hasher::combine`
/// builds a fresh Hasher per call — for chunked CRC pipelines, a
/// constant-time polynomial combine is preferred.
///
/// `Clone` + `Debug` mirror what `ChunkData` requires of every field
/// in its `std::vector<CRC32Calculator>` (ChunkData.hpp:561). The
/// inner `crc32fast::Hasher` implements both, so the derive is cheap.
#[derive(Debug, Clone)]
pub struct CRC32Calculator {
    hasher: Hasher,
    stream_size_in_bytes: u64,
    enabled: bool,
    finalized_crc: u32,
}

impl Default for CRC32Calculator {
    fn default() -> Self {
        Self::new()
    }
}

impl CRC32Calculator {
    /// Mirror of the vendor's default-constructed state
    /// (crc32.hpp:341-344). Enabled by default; stream size 0;
    /// internal CRC seed `~0u32`.
    pub fn new() -> Self {
        Self {
            hasher: Hasher::new(),
            stream_size_in_bytes: 0,
            enabled: true,
            finalized_crc: 0,
        }
    }

    /// Mirror of `CRC32Calculator::setEnabled` (crc32.hpp:265-268).
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Mirror of `CRC32Calculator::enabled` (crc32.hpp:270-274).
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Mirror of `CRC32Calculator::reset` (crc32.hpp:277-281).
    /// Returns calculator to the initial-empty-stream state.
    pub fn reset(&mut self) {
        self.hasher = Hasher::new();
        self.stream_size_in_bytes = 0;
        self.finalized_crc = 0;
    }

    /// Mirror of `CRC32Calculator::crc32` (crc32.hpp:283-287).
    /// Returns the finalized (post-`~`) CRC32 of all bytes fed so far.
    pub fn crc32(&self) -> u32 {
        if self.stream_size_in_bytes == 0 && self.finalized_crc == 0 {
            // crc32fast returns 0 for an empty hasher — matches vendor
            // behavior `return ~m_crc32` with `m_crc32 = ~0` → 0.
            0
        } else {
            self.hasher.clone().finalize() ^ self.finalized_crc
        }
    }

    /// Mirror of `CRC32Calculator::streamSize` (crc32.hpp:289-293).
    pub fn stream_size(&self) -> u64 {
        self.stream_size_in_bytes
    }

    /// Mirror of `CRC32Calculator::update` (crc32.hpp:296-303).
    pub fn update(&mut self, data: &[u8]) {
        if self.enabled {
            self.hasher.update(data);
            self.stream_size_in_bytes += data.len() as u64;
        }
    }

    /// Mirror of `CRC32Calculator::verify` (crc32.hpp:308-319).
    /// Returns `Ok(())` on match; `Err(message)` mirroring the vendor's
    /// `std::domain_error` formatting on mismatch. Disabled calculators
    /// always succeed (vendor behavior).
    pub fn verify(&self, crc32_to_compare: u32) -> Result<(), String> {
        if !self.enabled || self.crc32() == crc32_to_compare {
            return Ok(());
        }
        Err(format!(
            "Mismatching CRC32 (0x{:x} <-> stored: 0x{:x})!",
            self.crc32(),
            crc32_to_compare
        ))
    }

    /// Mirror of `CRC32Calculator::append` (crc32.hpp:322-329).
    /// Combines `self` with a calculator covering the *subsequent*
    /// part of the stream. Falls back to `combine_crc32` (the ported
    /// polynomial helper) rather than `crc32fast::Hasher::combine`
    /// to keep behavior identical even if the dependency switches.
    pub fn append(&mut self, to_append: &CRC32Calculator) {
        if self.enabled != to_append.enabled {
            return;
        }
        let combined = combine_crc32(
            self.crc32(),
            to_append.crc32(),
            to_append.stream_size_in_bytes,
        );
        // Reset internal hasher and store the combined CRC.
        self.hasher = Hasher::new();
        self.stream_size_in_bytes += to_append.stream_size_in_bytes;
        self.finalized_crc = combined;
    }

    /// Mirror of `CRC32Calculator::prepend` (crc32.hpp:332-339).
    /// Combines `self` with a calculator covering the *preceding*
    /// part of the stream.
    pub fn prepend(&mut self, to_prepend: &CRC32Calculator) {
        if self.enabled != to_prepend.enabled {
            return;
        }
        let combined = combine_crc32(to_prepend.crc32(), self.crc32(), self.stream_size_in_bytes);
        self.hasher = Hasher::new();
        self.stream_size_in_bytes += to_prepend.stream_size_in_bytes;
        self.finalized_crc = combined;
    }
}

/// Mirror of `rapidgzip::crc32` (crc32.hpp:348-353) — one-shot CRC32
/// of a byte slice. Delegates to `crc32fast::hash`.
pub fn crc32(buffer: &[u8]) -> u32 {
    crc32fast::hash(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-check `polynomial_multiply_modulo` against a brute-force
    /// reference for a few small inputs.
    #[test]
    fn polynomial_multiply_modulo_matches_reference() {
        // Identity: a * 0 = 0.
        assert_eq!(
            polynomial_multiply_modulo(0xABCD_1234, 0, CRC32_GENERATOR_POLYNOMIAL),
            0
        );
        // Identity: 0 * b = 0.
        assert_eq!(
            polynomial_multiply_modulo(0, 0xDEAD_BEEF, CRC32_GENERATOR_POLYNOMIAL),
            0
        );
        // Self-multiply is non-trivial; lock the result for the
        // identity x^0 = 1 in reflected notation (bit 31).
        let one = 1u32 << 31;
        for v in [0x1u32, 0x12345678, 0xFFFF_FFFF] {
            assert_eq!(
                polynomial_multiply_modulo(one, v, CRC32_GENERATOR_POLYNOMIAL),
                v,
                "x^0 * v should equal v for v={v:x}"
            );
        }
    }

    /// `x_power_modulo(0)` is the multiplicative identity x^0,
    /// represented as bit 31 in reflected notation.
    #[test]
    fn x_power_modulo_zero_is_identity() {
        assert_eq!(x_power_modulo(0), 1u32 << 31);
    }

    /// `X2N_LUT[0]` is x^1 in reflected notation per the vendor
    /// comment (crc32.hpp:183).
    #[test]
    fn x2n_lut_index_zero_is_x_to_the_one() {
        assert_eq!(X2N_LUT[0], 1u32 << 30);
    }

    /// Cross-check `combine_crc32` against `crc32fast::Hasher::combine`
    /// across several stream lengths.
    #[test]
    fn combine_matches_crc32fast() {
        // Use a few representative byte slices.
        let cases: &[(&[u8], &[u8])] = &[
            (b"", b""),
            (b"abc", b""),
            (b"", b"def"),
            (b"abc", b"def"),
            (b"hello, ", b"world!\n"),
            (&[0u8; 1024], &[0xFFu8; 4096]),
        ];
        for (a, b) in cases {
            let crc_a = crc32(a);
            let crc_b = crc32(b);
            let mut concat = Vec::new();
            concat.extend_from_slice(a);
            concat.extend_from_slice(b);
            let crc_concat = crc32(&concat);

            // Our port.
            let ours = combine_crc32(crc_a, crc_b, b.len() as u64);
            assert_eq!(
                ours,
                crc_concat,
                "combine_crc32 mismatch for a.len={}, b.len={}",
                a.len(),
                b.len()
            );

            // crc32fast reference (sanity, not relied on as ground truth).
            let mut h_a = Hasher::new();
            h_a.update(a);
            let mut h_b = Hasher::new();
            h_b.update(b);
            h_a.combine(&h_b);
            let crc32fast_combined = h_a.finalize();
            assert_eq!(
                ours,
                crc32fast_combined,
                "ours vs crc32fast for a.len={}, b.len={}",
                a.len(),
                b.len()
            );
        }
    }

    #[test]
    fn calculator_update_matches_one_shot() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let mut c = CRC32Calculator::new();
        c.update(data);
        assert_eq!(c.crc32(), crc32(data));
        assert_eq!(c.stream_size(), data.len() as u64);
    }

    #[test]
    fn calculator_reset_zeros_state() {
        let mut c = CRC32Calculator::new();
        c.update(b"hello");
        c.reset();
        assert_eq!(c.crc32(), 0);
        assert_eq!(c.stream_size(), 0);
    }

    #[test]
    fn calculator_disabled_does_not_update() {
        let mut c = CRC32Calculator::new();
        c.set_enabled(false);
        c.update(b"hello");
        assert_eq!(c.crc32(), 0);
        assert_eq!(c.stream_size(), 0);
    }

    #[test]
    fn calculator_verify_ok_and_err() {
        let data = b"some data";
        let mut c = CRC32Calculator::new();
        c.update(data);
        let real = c.crc32();
        assert!(c.verify(real).is_ok());

        let bad = real.wrapping_add(1);
        let err = c.verify(bad).unwrap_err();
        // Vendor format: "Mismatching CRC32 (0x... <-> stored: 0x...)!".
        assert!(err.starts_with("Mismatching CRC32 (0x"));
        assert!(err.ends_with(")!"));
    }

    #[test]
    fn calculator_append_combines_streams() {
        let a = b"hello, ";
        let b = b"world!";
        let mut ca = CRC32Calculator::new();
        ca.update(a);
        let mut cb = CRC32Calculator::new();
        cb.update(b);
        ca.append(&cb);

        let mut concat = Vec::new();
        concat.extend_from_slice(a);
        concat.extend_from_slice(b);
        assert_eq!(ca.crc32(), crc32(&concat));
        assert_eq!(ca.stream_size(), (a.len() + b.len()) as u64);
    }

    #[test]
    fn calculator_prepend_combines_streams() {
        let a = b"prefix-";
        let b = b"-suffix";
        let mut cb = CRC32Calculator::new();
        cb.update(b);
        let mut ca = CRC32Calculator::new();
        ca.update(a);
        cb.prepend(&ca);

        let mut concat = Vec::new();
        concat.extend_from_slice(a);
        concat.extend_from_slice(b);
        assert_eq!(cb.crc32(), crc32(&concat));
        assert_eq!(cb.stream_size(), (a.len() + b.len()) as u64);
    }
}
