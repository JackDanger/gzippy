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
// CRC32 byte-fold kernel (dispatched).
// =====================================================================
//
// GATE-2 / asm-confirmed motivation (aarch64, 2026-06-21): crc32fast 1.5.0's
// aarch64 path threads ONE `crc32x` accumulator (`crc32x w8, w8, xN` 8-way
// "unrolled" but a single w8 dependency chain) → latency-bound at ~8.5 GB/s on
// M-series (CRC32 instr ~3-cycle latency). The removal-oracle sized this at
// ~25 ms (9-20% of the T1 wall) on silesia/nasa; libdeflate uses PMULL
// multi-accumulator folding that breaks the chain. This kernel matches the
// throughput technique on the HW crc32 unit: THREE independent `crc32x`
// accumulators over three contiguous thirds, folded with the already-tested
// GF(2) `combine_crc32`. Bytes are unchanged (it is the same gzip CRC32); the
// gzip-trailer verify is a loud correctness check, and a differential test
// pins it byte-for-byte against crc32fast across sizes/seeds/alignments.

/// Continue the finalized gzip CRC32 `crc` over `data`. `crc` is the value a
/// previous fold returned (0 for an empty prefix). Single source of truth for
/// `CRC32Calculator`'s running checksum.
#[inline]
pub(crate) fn crc32_fold(crc: u32, data: &[u8]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        if !crc_legacy() {
            // PMULL carry-less fold (libdeflate crc32_arm_pmullx12 structure):
            // PMULL is a SEPARATE execution resource from the `crc32x` unit on
            // Apple Silicon, so a multi-accumulator PMULL fold can exceed the
            // 1-crc32x/cycle ceiling that bounds `fold3`. Default ON when the
            // CPU has BOTH the `aes` (PMULL) and `crc` extensions; force OFF for
            // a one-binary A/B with `GZIPPY_CRC_PMULL=0`.
            if hw_pmull::available() && crc_pmull_enabled() {
                // SAFETY: gated on runtime `aes`+`crc` feature detection.
                return unsafe { hw_pmull::crc32_arm_pmull(crc, data) };
            }
            if hw_crc::available() {
                // SAFETY: gated on runtime `crc` feature detection.
                return unsafe { hw_crc::fold3(crc, data) };
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if vpclmul::available() && !crc_legacy() {
            return vpclmul::crc32_vpclmul(crc, data);
        }
    }
    let mut h = Hasher::new_with_initial(crc);
    h.update(data);
    h.finalize()
}

/// `GZIPPY_CRC_LEGACY=1` forces the crc32fast kernel so the faster HW-CRC fold
/// (aarch64 3-way `crc32x`, x86_64 VPCLMULQDQ 256-bit fold) stays re-verifiable
/// by an interleaved A/B on ONE binary (controls for build variance). Read
/// once. Default OFF = fast path.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline]
fn crc_legacy() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| {
        matches!(
            std::env::var("GZIPPY_CRC_LEGACY").ok().as_deref(),
            Some("1")
        )
    })
}

/// `GZIPPY_CRC_PMULL=0` forces the `crc32x` 3-way fold (`hw_crc::fold3`) so the
/// PMULL fold stays re-verifiable by a one-binary A/B (controls for build
/// variance). Read once. Default ON = PMULL fold when `aes`+`crc` are present.
#[cfg(target_arch = "aarch64")]
#[inline]
fn crc_pmull_enabled() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| !matches!(std::env::var("GZIPPY_CRC_PMULL").ok().as_deref(), Some("0")))
}

#[cfg(target_arch = "aarch64")]
mod hw_crc {
    use super::combine_crc32;
    use core::arch::aarch64::{__crc32b, __crc32d};
    use std::sync::OnceLock;

    #[inline]
    pub fn available() -> bool {
        static A: OnceLock<bool> = OnceLock::new();
        *A.get_or_init(|| std::arch::is_aarch64_feature_detected!("crc"))
    }

    /// Single `crc32x` chain (matches crc32fast's kernel) — used for the head
    /// alignment, the post-3way tail, and small inputs.
    #[target_feature(enable = "crc")]
    unsafe fn fold1(crc: u32, data: &[u8]) -> u32 {
        let mut c = !crc;
        let (pre, mid, post) = data.align_to::<u64>();
        for &b in pre {
            c = __crc32b(c, b);
        }
        for &w in mid {
            c = __crc32d(c, w);
        }
        for &b in post {
            c = __crc32b(c, b);
        }
        !c
    }

    /// THREE independent `crc32x` accumulators over three contiguous,
    /// equal, word-aligned thirds, GF(2)-combined. Breaks the single-chain
    /// latency bottleneck (3 crc32x in flight hides the ~3-cycle latency).
    #[target_feature(enable = "crc")]
    pub unsafe fn fold3(crc: u32, data: &[u8]) -> u32 {
        const STRIDE: usize = 8; // bytes per crc32d
        let n = data.len();
        // Small inputs: the combine fixed-cost outweighs the ILP win.
        if n < 3 * 1024 {
            return fold1(crc, data);
        }
        // words processed per stream; block = bytes per stream (mult. of 8).
        let words = n / (3 * STRIDE);
        let block = words * STRIDE;
        let ptr = data.as_ptr();

        // Internal (pre-final-inversion) accumulators. Stream A carries the
        // incoming finalized `crc` (→ `!crc`); B and C start fresh (`!0`).
        let mut a = !crc;
        let mut b = !0u32;
        let mut c = !0u32;
        for i in 0..words {
            let off = i * STRIDE;
            let wa = (ptr.add(off) as *const u64).read_unaligned();
            let wb = (ptr.add(block + off) as *const u64).read_unaligned();
            let wc = (ptr.add(2 * block + off) as *const u64).read_unaligned();
            a = __crc32d(a, wa);
            b = __crc32d(b, wb);
            c = __crc32d(c, wc);
        }
        // Finalize each stream, then fold A++B++C (each `block` bytes).
        let fa = !a;
        let fb = !b;
        let fc = !c;
        let ab = combine_crc32(fa, fb, block as u64);
        let abc = combine_crc32(ab, fc, block as u64);
        // Tail past the three equal thirds.
        fold1(abc, &data[3 * block..])
    }
}

// =====================================================================
// aarch64 PMULL carry-less CRC32 fold (libdeflate crc32_arm_pmullx12 port).
// =====================================================================
//
// GATE motivation (M1, 2026-06-28): gz's arm64 CRC = `hw_crc::fold3`, three
// interleaved `crc32x` chains, bounded by the M1's 1-crc32x/cycle throughput.
// libdeflate uses PMULL (carry-less multiply, a SEPARATE execution resource on
// Apple Silicon) folding 192 bytes (12 independent 128-bit accumulators) per
// iteration, then a `crc32x` reduction of the final 128-bit accumulator + tail.
// This is a faithful port of libdeflate's `crc32_arm` extra-wide template
// (vendor/libdeflate/lib/arm/crc32_pmull_wide.h, stride 12) — the variant
// libdeflate itself selects for the M1. The pointer-alignment step is omitted
// (we use unaligned 128-bit loads, equally correct: CRC is a function of bytes,
// not of how they are blocked), so the result is byte-identical to crc32fast,
// pinned by the differential test across sizes/alignments/seeds.
//
// Fold multipliers are x^N mod G(x) (reflected gzip generator), N = 128·D ± 31/33
// for a D-vector fold, derived at COMPILE TIME by the same `x_power_modulo`
// routine that builds `combine_crc32`, and asserted to reproduce libdeflate's
// published `crc32_multipliers.h` constants — no magic numbers trusted unverified.
#[cfg(target_arch = "aarch64")]
mod hw_pmull {
    use super::x_power_modulo as xpow;
    use core::arch::aarch64::{__crc32b, __crc32d, __crc32h, __crc32w, vmull_p64};
    use std::sync::OnceLock;

    #[inline]
    pub fn available() -> bool {
        static A: OnceLock<bool> = OnceLock::new();
        *A.get_or_init(|| {
            std::arch::is_aarch64_feature_detected!("aes")
                && std::arch::is_aarch64_feature_detected!("crc")
        })
    }

    // ---- compile-time fold-multiplier derivation (reflected GF(2)) ----
    // For a D-vector (D·16-byte) fold the reflected multiplier pair is
    // {low = x^(128·D + 31) mod G, high = x^(128·D − 33) mod G}.
    const fn m(d: u64) -> (u64, u64) {
        (xpow(128 * d + 31) as u64, xpow(128 * d - 33) as u64)
    }
    const M12: (u64, u64) = m(12);
    const M6: (u64, u64) = m(6);
    const M4: (u64, u64) = m(4);
    const M3: (u64, u64) = m(3);
    const M2: (u64, u64) = m(2);
    const M1: (u64, u64) = m(1);

    // Assert the derived multipliers reproduce libdeflate's published
    // crc32_multipliers.h constants (provenance: each is `x^N mod G(x)`).
    const _: () = {
        assert!(M12.0 == 0x596c8d81 && M12.1 == 0xf5e48c85); // X1567 / X1503
        assert!(M6.0 == 0xdf068dc2 && M6.1 == 0x57c54819); // X799  / X735
        assert!(M4.0 == 0x8f352d95 && M4.1 == 0x1d9513d7); // X543  / X479
        assert!(M3.0 == 0x3db1ecdc && M3.1 == 0xaf449247); // X415  / X351
        assert!(M2.0 == 0xf1da05aa && M2.1 == 0x81256527); // X287  / X223
        assert!(M1.0 == 0xae689191 && M1.1 == 0xccaa009e); // X159  / X95
    };

    /// Load 16 unaligned bytes as a little-endian 128-bit polynomial
    /// (lane0 = first 8 bytes, lane1 = next 8 bytes — matches `vld1q_u8`).
    #[inline(always)]
    unsafe fn ld(p: *const u8) -> u128 {
        (p as *const u128).read_unaligned()
    }

    /// libdeflate `fold_vec`: reflected fold of `src` (one 16-byte vector)
    /// `mult` distance ahead into `dst`:
    ///   clmul(src.lo, mult.lo) ^ clmul(src.hi, mult.hi) ^ dst.
    #[inline]
    #[target_feature(enable = "neon,aes")]
    unsafe fn fold(src: u128, dst: u128, mult: (u64, u64)) -> u128 {
        let lo = vmull_p64(src as u64, mult.0);
        let hi = vmull_p64((src >> 64) as u64, mult.1);
        lo ^ hi ^ dst
    }

    /// Reduce a final 128-bit accumulator to 32 bits via two `crc32x`.
    #[inline]
    #[target_feature(enable = "crc")]
    unsafe fn reduce128(crc: u32, v: u128) -> u32 {
        let c = __crc32d(crc, v as u64);
        __crc32d(c, (v >> 64) as u64)
    }

    /// Continue the finalized gzip CRC32 `crc_in` over `data`, byte-identical to
    /// `crc32fast`, via the PMULL extra-wide fold. Internal (`!crc`) domain
    /// throughout; inverted at the boundaries like `fold3`.
    #[target_feature(enable = "neon,aes,crc")]
    pub unsafe fn crc32_arm_pmull(crc_in: u32, data: &[u8]) -> u32 {
        let mut crc = !crc_in;
        let mut p = data.as_ptr();
        let mut len = data.len();

        if len >= 3 * 192 {
            // ----- extra-wide path: 12 independent 128-bit accumulators -----
            let mut v0 = ld(p) ^ (crc as u128);
            let mut v1 = ld(p.add(16));
            let mut v2 = ld(p.add(32));
            let mut v3 = ld(p.add(48));
            let mut v4 = ld(p.add(64));
            let mut v5 = ld(p.add(80));
            let mut v6 = ld(p.add(96));
            let mut v7 = ld(p.add(112));
            let mut v8 = ld(p.add(128));
            let mut v9 = ld(p.add(144));
            let mut v10 = ld(p.add(160));
            let mut v11 = ld(p.add(176));
            p = p.add(192);
            len -= 192;
            while len >= 192 {
                v0 = fold(v0, ld(p), M12);
                v1 = fold(v1, ld(p.add(16)), M12);
                v2 = fold(v2, ld(p.add(32)), M12);
                v3 = fold(v3, ld(p.add(48)), M12);
                v4 = fold(v4, ld(p.add(64)), M12);
                v5 = fold(v5, ld(p.add(80)), M12);
                v6 = fold(v6, ld(p.add(96)), M12);
                v7 = fold(v7, ld(p.add(112)), M12);
                v8 = fold(v8, ld(p.add(128)), M12);
                v9 = fold(v9, ld(p.add(144)), M12);
                v10 = fold(v10, ld(p.add(160)), M12);
                v11 = fold(v11, ld(p.add(176)), M12);
                p = p.add(192);
                len -= 192;
            }
            // Fold v0..v11 down to v0, consuming up to 144 more bytes.
            v0 = fold(v0, v6, M6);
            v1 = fold(v1, v7, M6);
            v2 = fold(v2, v8, M6);
            v3 = fold(v3, v9, M6);
            v4 = fold(v4, v10, M6);
            v5 = fold(v5, v11, M6);
            if len >= 96 {
                v0 = fold(v0, ld(p), M6);
                v1 = fold(v1, ld(p.add(16)), M6);
                v2 = fold(v2, ld(p.add(32)), M6);
                v3 = fold(v3, ld(p.add(48)), M6);
                v4 = fold(v4, ld(p.add(64)), M6);
                v5 = fold(v5, ld(p.add(80)), M6);
                p = p.add(96);
                len -= 96;
            }
            v0 = fold(v0, v3, M3);
            v1 = fold(v1, v4, M3);
            v2 = fold(v2, v5, M3);
            if len >= 48 {
                v0 = fold(v0, ld(p), M3);
                v1 = fold(v1, ld(p.add(16)), M3);
                v2 = fold(v2, ld(p.add(32)), M3);
                p = p.add(48);
                len -= 48;
            }
            v0 = fold(v0, v1, M1);
            v0 = fold(v0, v2, M1);
            crc = reduce128(0, v0);
        } else if len >= 64 {
            // ----- medium path: 4 accumulators, 64 bytes/iter -----
            let mut v0 = ld(p) ^ (crc as u128);
            let mut v1 = ld(p.add(16));
            let mut v2 = ld(p.add(32));
            let mut v3 = ld(p.add(48));
            p = p.add(64);
            len -= 64;
            while len >= 64 {
                v0 = fold(v0, ld(p), M4);
                v1 = fold(v1, ld(p.add(16)), M4);
                v2 = fold(v2, ld(p.add(32)), M4);
                v3 = fold(v3, ld(p.add(48)), M4);
                p = p.add(64);
                len -= 64;
            }
            v0 = fold(v0, v2, M2);
            v1 = fold(v1, v3, M2);
            if len >= 32 {
                v0 = fold(v0, ld(p), M2);
                v1 = fold(v1, ld(p.add(16)), M2);
                p = p.add(32);
                len -= 32;
            }
            v0 = fold(v0, v1, M1);
            crc = reduce128(0, v0);
        }

        // ----- tail (< 64 bytes remaining) via crc32x instructions -----
        if len & 32 != 0 {
            crc = __crc32d(crc, (p as *const u64).read_unaligned());
            crc = __crc32d(crc, (p.add(8) as *const u64).read_unaligned());
            crc = __crc32d(crc, (p.add(16) as *const u64).read_unaligned());
            crc = __crc32d(crc, (p.add(24) as *const u64).read_unaligned());
            p = p.add(32);
        }
        if len & 16 != 0 {
            crc = __crc32d(crc, (p as *const u64).read_unaligned());
            crc = __crc32d(crc, (p.add(8) as *const u64).read_unaligned());
            p = p.add(16);
        }
        if len & 8 != 0 {
            crc = __crc32d(crc, (p as *const u64).read_unaligned());
            p = p.add(8);
        }
        if len & 4 != 0 {
            crc = __crc32w(crc, (p as *const u32).read_unaligned());
            p = p.add(4);
        }
        if len & 2 != 0 {
            crc = __crc32h(crc, (p as *const u16).read_unaligned());
            p = p.add(2);
        }
        if len & 1 != 0 {
            crc = __crc32b(crc, *p);
        }
        !crc
    }
}

// =====================================================================
// x86_64 VPCLMULQDQ CRC32 fold (DIVERGENCE from the crc32fast crate kernel).
// =====================================================================
//
// GATE-1 motivation (Intel i7-13700T raptorlake, 2026-06-26, /dev/shm
// microbench, byte-exact vs crc32fast across sizes/aligns/seeds): crc32fast
// 1.5.0's x86 kernel is a 128-bit fold-by-4 (SSE PCLMULQDQ, 64 B/iter). On
// cache-resident data (16 KiB–1 MiB — the regime per-block CRC runs in) it
// measures ~11 GB/s; ISA-L's crc32_gzip_refl is the SAME (~11 GB/s); but
// libdeflate's CRC reaches ~22 GB/s = 2.0x. The win is a WIDER fold: 256-bit
// VEX-encoded VPCLMULQDQ (`_mm256_clmulepi64_epi128`, 2 carryless mults/insn)
// folding 128 B/iter with 4 YMM accumulators hides PCLMUL latency. This pure-
// Rust port matches libdeflate (~21.4–21.8 GB/s, 2.0x crc32fast on hot data).
// Bytes are unchanged (same reflected gzip CRC32); the gzip-trailer verify is
// a loud correctness check and the differential test pins it byte-for-byte.
// Fold constants are derived at COMPILE TIME from the reflected GF(2) routine
// above (same algorithm as `polynomial_multiply_modulo`) and asserted to
// reproduce crc32fast's published 64 B / 16 B keys before yielding the new
// 128 B-stride keys — so no magic numbers are trusted unverified.
#[cfg(target_arch = "x86_64")]
mod vpclmul {
    use core::arch::x86_64 as arch;
    use crc32fast::Hasher;
    use std::sync::OnceLock;

    /// Runtime feature gate (cached). Requires the full VPCLMULQDQ + AVX2 set.
    #[inline]
    pub fn available() -> bool {
        static A: OnceLock<bool> = OnceLock::new();
        *A.get_or_init(|| {
            is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("vpclmulqdq")
                && is_x86_feature_detected!("pclmulqdq")
                && is_x86_feature_detected!("sse4.1")
                && is_x86_feature_detected!("sse2")
        })
    }

    // ---- compile-time fold-key derivation (reflected GF(2)) ----
    const POLY: u32 = super::CRC32_GENERATOR_POLYNOMIAL; // 0xEDB88320

    const fn pmm(a: u32, mut b: u32) -> u32 {
        let mut result: u32 = 0;
        let mut cp: u32 = 1u32 << 31;
        while cp > 0 {
            if (a & cp) != 0 {
                result ^= b;
            }
            let of = (b & 1) != 0;
            b >>= 1;
            if of {
                b ^= POLY;
            }
            cp >>= 1;
        }
        result
    }

    /// x^exponent mod P in reflected form (binary exponentiation; const-usable).
    const fn x_pow_mod(mut e: u64) -> u32 {
        let mut result: u32 = 1u32 << 31; // x^0
        let mut base: u32 = 1u32 << 30; // x^1
        while e > 0 {
            if e & 1 != 0 {
                result = pmm(result, base);
            }
            base = pmm(base, base);
            e >>= 1;
        }
        result
    }

    /// PCLMULQDQ key for x^e mod P (33-bit reflected encoding = residue << 1).
    const fn key(e: u64) -> i64 {
        ((x_pow_mod(e) as u64) << 1) as i64
    }

    // For an S-byte fold stride: low key (× a.low , clmul 0x00) = x^(8S+32),
    // high key (× a.high, clmul 0x11) = x^(8S-32). Verified vs crc32fast below.
    const K1_64: i64 = key(544); // 64 B stride, low
    const K2_64: i64 = key(480); // 64 B stride, high
    const K1_128: i64 = key(1056); // 128 B stride, low
    const K2_128: i64 = key(992); // 128 B stride, high
    const K3: i64 = key(160); // 16 B stride, low
    const K4: i64 = key(96); // 16 B stride, high
    const K5: i64 = key(64); // step-3
    const P_X: i64 = 0x1DB7_10641;
    const U_PRIME: i64 = 0x1F70_11641;

    const _: () = {
        assert!(K1_64 == 0x1544_42bd4);
        assert!(K2_64 == 0x1c6e_41596);
        assert!(K3 == 0x1751_997d0);
        assert!(K4 == 0x0cca_a009e);
        assert!(K5 == 0x163c_d6124);
    };

    /// crc32fast continuation — the project's trusted scalar/short-input path.
    #[inline]
    fn fallback(crc: u32, data: &[u8]) -> u32 {
        let mut h = Hasher::new_with_initial(crc);
        h.update(data);
        h.finalize()
    }

    /// Continue the finalized gzip CRC32 `crc` over `data`, byte-identical to
    /// `crc32fast`. Dispatches to the 256-bit fold when the buffer is large
    /// enough; small/short inputs go through crc32fast.
    #[inline]
    pub fn crc32_vpclmul(crc: u32, data: &[u8]) -> u32 {
        if data.len() < 128 {
            return fallback(crc, data);
        }
        // SAFETY: `available()` (checked at the call site in crc32_fold) proves
        // the CPU supports every feature the inner fn enables.
        unsafe { crc32_vpclmul_inner(crc, data) }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    #[target_feature(enable = "avx2,vpclmulqdq,pclmulqdq,sse4.1,sse2")]
    unsafe fn crc32_vpclmul_inner(crc: u32, mut data: &[u8]) -> u32 {
        debug_assert!(data.len() >= 128);

        let mut y3 = get256(&mut data);
        let mut y2 = get256(&mut data);
        let mut y1 = get256(&mut data);
        let mut y0 = get256(&mut data);

        // Fold the incoming finalized CRC into the lowest dword (first byte).
        y3 = arch::_mm256_xor_si256(y3, arch::_mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, !crc as i32));

        let k128 = arch::_mm256_set_epi64x(K2_128, K1_128, K2_128, K1_128);
        while data.len() >= 128 {
            y3 = reduce256(y3, get256(&mut data), k128);
            y2 = reduce256(y2, get256(&mut data), k128);
            y1 = reduce256(y1, get256(&mut data), k128);
            y0 = reduce256(y0, get256(&mut data), k128);
        }

        // Reduce the 8 128-bit sub-lanes (stream order) to one with the 16 B key.
        let k3k4 = arch::_mm_set_epi64x(K4, K3);
        let mut x = reduce128(low128(y3), high128(y3), k3k4);
        x = reduce128(x, low128(y2), k3k4);
        x = reduce128(x, high128(y2), k3k4);
        x = reduce128(x, low128(y1), k3k4);
        x = reduce128(x, high128(y1), k3k4);
        x = reduce128(x, low128(y0), k3k4);
        x = reduce128(x, high128(y0), k3k4);

        // Also fold any remaining 64 B groups with the 64 B key for parity with
        // crc32fast's fold-by-1 step (here just the 16 B fold-by-1 loop).
        let _ = (K1_64, K2_64);
        while data.len() >= 16 {
            x = reduce128(x, get128(&mut data), k3k4);
        }

        // Step 3 (128->64) + Barrett (64->32), verbatim from crc32fast 1.5.0.
        let x = arch::_mm_xor_si128(
            arch::_mm_clmulepi64_si128::<0x10>(x, k3k4),
            arch::_mm_srli_si128::<8>(x),
        );
        let x = arch::_mm_xor_si128(
            arch::_mm_clmulepi64_si128::<0x00>(
                arch::_mm_and_si128(x, arch::_mm_set_epi32(0, 0, 0, !0)),
                arch::_mm_set_epi64x(0, K5),
            ),
            arch::_mm_srli_si128::<4>(x),
        );
        let pu = arch::_mm_set_epi64x(U_PRIME, P_X);
        let t1 = arch::_mm_clmulepi64_si128::<0x10>(
            arch::_mm_and_si128(x, arch::_mm_set_epi32(0, 0, 0, !0)),
            pu,
        );
        let t2 = arch::_mm_clmulepi64_si128::<0x00>(
            arch::_mm_and_si128(t1, arch::_mm_set_epi32(0, 0, 0, !0)),
            pu,
        );
        let c = arch::_mm_extract_epi32::<1>(arch::_mm_xor_si128(x, t2)) as u32;

        arch::_mm256_zeroupper();

        if data.is_empty() {
            !c
        } else {
            fallback(!c, data)
        }
    }

    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn reduce256(a: arch::__m256i, b: arch::__m256i, keys: arch::__m256i) -> arch::__m256i {
        let t1 = arch::_mm256_clmulepi64_epi128::<0x00>(a, keys);
        let t2 = arch::_mm256_clmulepi64_epi128::<0x11>(a, keys);
        arch::_mm256_xor_si256(arch::_mm256_xor_si256(b, t1), t2)
    }

    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn reduce128(a: arch::__m128i, b: arch::__m128i, keys: arch::__m128i) -> arch::__m128i {
        let t1 = arch::_mm_clmulepi64_si128::<0x00>(a, keys);
        let t2 = arch::_mm_clmulepi64_si128::<0x11>(a, keys);
        arch::_mm_xor_si128(arch::_mm_xor_si128(b, t1), t2)
    }

    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn get256(data: &mut &[u8]) -> arch::__m256i {
        debug_assert!(data.len() >= 32);
        let out = arch::_mm256_loadu_si256(data.as_ptr() as *const arch::__m256i);
        *data = &data[32..];
        out
    }

    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn get128(data: &mut &[u8]) -> arch::__m128i {
        debug_assert!(data.len() >= 16);
        let out = arch::_mm_loadu_si128(data.as_ptr() as *const arch::__m128i);
        *data = &data[16..];
        out
    }

    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn low128(value: arch::__m256i) -> arch::__m128i {
        arch::_mm256_castsi256_si128(value)
    }

    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn high128(value: arch::__m256i) -> arch::__m128i {
        arch::_mm256_extracti128_si256::<1>(value)
    }
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
    /// Finalized gzip CRC32 of all bytes fed via `update` since the last
    /// reset/append/prepend (0 for an empty run). Replaces the former inner
    /// `crc32fast::Hasher` so the fold kernel is dispatchable (`crc32_fold`).
    running: u32,
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
            running: 0,
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
        self.running = 0;
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
            self.running ^ self.finalized_crc
        }
    }

    /// Mirror of `CRC32Calculator::streamSize` (crc32.hpp:289-293).
    pub fn stream_size(&self) -> u64 {
        self.stream_size_in_bytes
    }

    /// Mirror of `CRC32Calculator::update` (crc32.hpp:296-303).
    pub fn update(&mut self, data: &[u8]) {
        if self.enabled {
            self.running = crc32_fold(self.running, data);
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
        // Reset the running fold and store the combined CRC.
        self.running = 0;
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
        self.running = 0;
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

    /// BYTE-EXACT pin for the aarch64 PMULL fold: `hw_pmull::crc32_arm_pmull`
    /// MUST equal crc32fast (the trusted oracle) AND `hw_crc::fold3` across every
    /// size that exercises the extra-wide (≥576 B), medium (64–575 B), and
    /// crc32x-tail paths, at every head/tail misalignment and seed. A fast CRC
    /// with wrong checksums is a FAIL, so this is the gate the perf claim rides on.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn crc32_pmull_matches_crc32fast_and_fold3_all_sizes() {
        if !hw_pmull::available() {
            eprintln!("skip: no aes+crc on this aarch64 host");
            return;
        }
        let mut big = vec![0u8; 200_000 + 64];
        let mut x: u32 = 0x0BAD_F00D;
        for b in big.iter_mut() {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *b = (x >> 16) as u8;
        }
        // Sizes straddle the 64-byte and 576-byte path thresholds, the 192/96/48
        // wide-fold edges, the 64/32 medium edges, and every tail bit (32/16/8/4/2/1).
        let sizes = [
            0usize, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 47, 48, 63, 64, 65, 95, 96, 127, 128,
            143, 144, 191, 192, 193, 287, 288, 383, 384, 575, 576, 577, 768, 1151, 1152, 1153,
            4096, 9999, 65536, 131073, 200000,
        ];
        let seeds = [0u32, 1, 0xFFFF_FFFF, 0xDEAD_BEEF, 0x8000_0001, 0x0000_0001];
        for &align in &[0usize, 1, 3, 5, 7, 8, 9] {
            for &size in &sizes {
                if align + size > big.len() {
                    continue;
                }
                let data = &big[align..align + size];
                for &seed in &seeds {
                    let want = {
                        let mut h = Hasher::new_with_initial(seed);
                        h.update(data);
                        h.finalize()
                    };
                    let pmull = unsafe { hw_pmull::crc32_arm_pmull(seed, data) };
                    assert_eq!(
                        pmull, want,
                        "PMULL!=crc32fast size={size} align={align} seed={seed:#x}"
                    );
                    if hw_crc::available() {
                        let f3 = unsafe { hw_crc::fold3(seed, data) };
                        assert_eq!(
                            pmull, f3,
                            "PMULL!=fold3 size={size} align={align} seed={seed:#x}"
                        );
                    }
                }
            }
        }
        // Chained PMULL folds == single fold over the concatenation.
        let parts = [
            &big[0..7],
            &big[7..600],
            &big[600..50000],
            &big[50000..131073],
        ];
        let mut running = 0u32;
        for q in parts {
            running = unsafe { hw_pmull::crc32_arm_pmull(running, q) };
        }
        let whole = unsafe { hw_pmull::crc32_arm_pmull(0, &big[0..131073]) };
        assert_eq!(running, whole, "chained PMULL fold != single fold");
    }

    /// Standalone GB/s microbench: PMULL fold vs `crc32x` fold3 on M1.
    /// IGNORED in normal runs; invoke with
    ///   cargo test --release --no-default-features --features pure-rust-inflate \
    ///     -- --ignored --nocapture crc_pmull_vs_crc32x_gbps
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore]
    fn crc_pmull_vs_crc32x_gbps() {
        if !hw_pmull::available() {
            eprintln!("skip: no aes+crc");
            return;
        }
        use std::time::Instant;
        // ~4 MiB working set (cache-resident, the per-chunk CRC regime).
        let n = 4 << 20;
        let mut buf = vec![0u8; n];
        let mut x: u32 = 0x1357_9BDF;
        for b in buf.iter_mut() {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *b = (x >> 16) as u8;
        }
        let iters = 200usize;
        // Warm + correctness cross-check before timing.
        let a = unsafe { hw_pmull::crc32_arm_pmull(0, &buf) };
        let b = unsafe { hw_crc::fold3(0, &buf) };
        assert_eq!(a, b, "bench inputs disagree — refusing to report GB/s");

        let bench = |f: &dyn Fn(u32, &[u8]) -> u32| -> f64 {
            // best-of-7 to suppress scheduler noise. The running `acc` is fed
            // back as the seed AND the buffer is black_box'd each iteration so
            // the optimizer cannot CSE/hoist the pure call out of the loop
            // (the phantom-GB/s trap). The seed dependency serializes calls.
            let mut best = f64::INFINITY;
            for _ in 0..7 {
                let t = Instant::now();
                let mut acc = 0u32;
                for _ in 0..iters {
                    acc = f(acc, std::hint::black_box(&buf));
                }
                std::hint::black_box(acc);
                let s = t.elapsed().as_secs_f64();
                if s < best {
                    best = s;
                }
            }
            (n as f64 * iters as f64) / best / 1e9
        };
        let pmull = bench(&|c, d| unsafe { hw_pmull::crc32_arm_pmull(c, d) });
        let crc32x = bench(&|c, d| unsafe { hw_crc::fold3(c, d) });
        eprintln!(
            "CRC32 standalone microbench (M1, {} MiB x{iters}, best-of-7):",
            n >> 20
        );
        eprintln!("  crc32x fold3  : {crc32x:7.2} GB/s");
        eprintln!("  PMULL  fold12 : {pmull:7.2} GB/s");
        eprintln!("  PMULL/crc32x  : {:.3}x", pmull / crc32x);
    }

    /// The dispatched `crc32_fold` (aarch64 3-way `crc32x` interleave + tail,
    /// else crc32fast continuation) MUST equal crc32fast byte-for-byte across
    /// sizes that exercise the small-input path, the 3-way body, the
    /// post-3way tail, and every head/tail (mis)alignment. This is the
    /// correctness pin for the GATE-2 CRC throughput lever.
    #[test]
    fn crc32_fold_matches_crc32fast_all_sizes_seeds_alignments() {
        // LCG-filled buffer so content varies; bytes don't matter for the
        // poly but exercise real values.
        let mut big = vec![0u8; 200_000 + 64];
        let mut x: u32 = 0x1234_5678;
        for b in big.iter_mut() {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *b = (x >> 16) as u8;
        }
        // Sizes around the 3*1024 small/large threshold and 3*8 block edges.
        let sizes = [
            0usize, 1, 7, 8, 9, 23, 24, 25, 64, 1023, 3071, 3072, 3073, 3095, 4096, 9999, 65536,
            131071, 131072, 131073, 200000,
        ];
        let seeds = [0u32, 1, 0xFFFF_FFFF, 0xDEAD_BEEF, 0x8000_0001];
        for &align in &[0usize, 1, 3, 5, 7] {
            for &size in &sizes {
                if align + size > big.len() {
                    continue;
                }
                let data = &big[align..align + size];
                for &seed in &seeds {
                    let got = crc32_fold(seed, data);
                    let mut h = Hasher::new_with_initial(seed);
                    h.update(data);
                    let want = h.finalize();
                    assert_eq!(
                        got, want,
                        "fold mismatch size={size} align={align} seed={seed:#x}"
                    );
                }
            }
        }
        // Chained folds (multiple update calls) must equal one fold over the
        // concatenation — the running-CRC invariant CRC32Calculator relies on.
        let parts = [
            &big[0..7],
            &big[7..3000],
            &big[3000..50000],
            &big[50000..131073],
        ];
        let mut running = 0u32;
        for p in parts {
            running = crc32_fold(running, p);
        }
        let whole = crc32_fold(0, &big[0..131073]);
        assert_eq!(running, whole, "chained fold != single fold");
    }

    /// `CRC32Calculator::update` (now backed by `crc32_fold`) must still equal
    /// the standalone `crc32()` of the same bytes after the hasher->running
    /// refactor, for a large multi-update stream.
    #[test]
    fn calculator_update_matches_reference_after_refactor() {
        let mut buf = vec![0u8; 300_003];
        let mut x: u32 = 0xCAFE_BABE;
        for b in buf.iter_mut() {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *b = (x >> 8) as u8;
        }
        let mut c = CRC32Calculator::new();
        // Feed in irregular chunks crossing the 3-way and tail boundaries.
        for chunk in buf.chunks(7919) {
            c.update(chunk);
        }
        assert_eq!(c.crc32(), crc32(&buf));
        assert_eq!(c.stream_size(), buf.len() as u64);
    }

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

    // =================================================================
    // STAGE-1 CRC-KERNEL HARDENING (merge-blocker for the VPCLMULQDQ /
    // 3-way HW-CRC fold). A wrong fold is SILENT corruption, so the
    // dispatched `crc32_fold` MUST equal BOTH crc32fast AND an
    // independent bit-serial reference across every length×alignment
    // that exercises the kernel's structural boundaries:
    //   x86 VPCLMULQDQ: <128 fallback, the 128 B/iter fold-by-4 body
    //     (128·k ± 1), the 16 B fold-by-1 tail, the <16 scalar tail.
    //   aarch64 3-way:  <3072 fold1, the 3·8 block edges, the post-3way
    //     tail.
    // crc32fast is the project's trusted kernel; the independent
    // bit-serial reference removes the "both wrap the same crate" blind
    // spot. This test FIRES the new kernel on x86_64 (vpclmul) and
    // aarch64 (hw_crc) because `crc32_fold` dispatches to it.
    // =================================================================

    /// Textbook reflected (LSB-first) bit-serial CRC32 continuation —
    /// independent of crc32fast, used as a second oracle.
    fn crc32_ref_continue(crc: u32, data: &[u8]) -> u32 {
        let mut c = !crc;
        for &b in data {
            c ^= b as u32;
            for _ in 0..8 {
                c = (c >> 1) ^ (CRC32_GENERATOR_POLYNOMIAL & (!(c & 1)).wrapping_add(1));
            }
        }
        !c
    }

    #[test]
    fn crc32_ref_self_consistency() {
        // Independent reference must match crc32fast on a known vector.
        // "123456789" CRC32 == 0xCBF43926 (the canonical check value).
        assert_eq!(crc32_ref_continue(0, b"123456789"), 0xCBF4_3926);
        let mut h = Hasher::new();
        h.update(b"123456789");
        assert_eq!(h.finalize(), 0xCBF4_3926);
    }

    /// THE merge-blocker. Exhaustive-ish length × alignment sweep of the
    /// dispatched fold kernel against TWO oracles. Prints the pass count.
    #[test]
    fn crc_kernel_hardening_lengths_x_alignments() {
        // LCG-filled buffer; content varies so the poly sees real bytes.
        const MAXLEN: usize = 70_001;
        const PAD: usize = 64; // alignment headroom
        let mut buf = vec![0u8; MAXLEN + PAD];
        let mut x: u32 = 0x9E37_79B9;
        for b in buf.iter_mut() {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *b = (x >> 13) as u8;
        }

        // Build the length set: exhaustive small, all 128 B/iter
        // boundaries ±1 (x86 fold-by-4), the 16 B tail edges, the aarch64
        // 3·8 / 3072 edges, and large spot lengths.
        let mut lengths: Vec<usize> = (0..=600).collect();
        for k in 1..=549 {
            // 128·k − 1, 128·k, 128·k + 1  (fold-by-4 body, up to ~70272)
            let base: usize = 128 * k;
            for d in [base.wrapping_sub(1), base, base + 1] {
                if d <= MAXLEN {
                    lengths.push(d);
                }
            }
        }
        // 16 B tail edges and aarch64 thresholds.
        for &e in &[
            15usize, 16, 17, 31, 32, 33, 127, 128, 129, 143, 144, 145, 3071, 3072, 3073, 3095,
            3096, 3097, 6143, 6144, 6145, 9215, 9216, 9217, 65535, 65536, 65537, 70000, 70001,
        ] {
            if e <= MAXLEN {
                lengths.push(e);
            }
        }
        lengths.sort_unstable();
        lengths.dedup();

        let seeds = [0u32, 1, 0xFFFF_FFFF, 0xDEAD_BEEF, 0x8000_0001, 0xCBF4_3926];

        let mut pass: u64 = 0;
        let mut fail: u64 = 0;
        let mut first_fail: Option<(usize, usize, u32)> = None;

        for align in 0usize..=63 {
            for &len in &lengths {
                if align + len > buf.len() {
                    continue;
                }
                let data = &buf[align..align + len];
                // crc32fast (trusted crate kernel) is the oracle for ALL
                // lengths. The O(n·8) bit-serial reference is run only for
                // len <= 4096 (still covers <128 fallback + the first 32
                // fold-by-4 iters + all 16 B-tail edges) to keep the
                // sweep fast; crc32fast covers the large body.
                let do_ref = len <= 4096;
                for &seed in &seeds {
                    let got = crc32_fold(seed, data);
                    let mut h = Hasher::new_with_initial(seed);
                    h.update(data);
                    let want_fast = h.finalize();
                    let ref_ok = !do_ref || got == crc32_ref_continue(seed, data);
                    if got == want_fast && ref_ok {
                        pass += 1;
                    } else {
                        fail += 1;
                        if first_fail.is_none() {
                            first_fail = Some((len, align, seed));
                        }
                    }
                }
            }
        }

        // Chained-fold invariant across the kernel boundaries (the
        // running-CRC contract CRC32Calculator relies on): many small
        // updates == one big fold.
        let split = [0usize, 1, 15, 16, 128, 129, 3072, 20000, 40000, 70001];
        let mut running = 0u32;
        for w in split.windows(2) {
            running = crc32_fold(running, &buf[w[0]..w[1]]);
        }
        let whole = crc32_fold(0, &buf[0..70001]);
        let chained_ok = running == whole;
        if chained_ok {
            pass += 1;
        } else {
            fail += 1;
        }

        eprintln!(
            "CRC-KERNEL-HARDENING: pass={pass} fail={fail} \
             lengths={} alignments=64 seeds={} chained_ok={chained_ok} \
             first_fail={:?}",
            lengths.len(),
            seeds.len(),
            first_fail
        );
        assert_eq!(
            fail, 0,
            "CRC fold mismatch; first_fail (len,align,seed)={first_fail:?}"
        );
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(4000))]

        /// Property: the dispatched fold == crc32fast == independent
        /// bit-serial reference for ARBITRARY bytes, lengths and seeds,
        /// at every alignment 0..64. Random fuzz on top of the
        /// structured sweep above.
        #[test]
        fn prop_crc_fold_matches_both_oracles(
            mut bytes in proptest::collection::vec(proptest::prelude::any::<u8>(), 0..4200usize),
            seed in proptest::prelude::any::<u32>(),
            align in 0usize..64,
        ) {
            // Prepend `align` filler bytes, then fold the aligned tail so
            // the kernel sees a non-zero start offset.
            let mut padded = vec![0u8; align];
            padded.append(&mut bytes);
            let data = &padded[align..];
            let got = crc32_fold(seed, data);
            let mut h = Hasher::new_with_initial(seed);
            h.update(data);
            let want_fast = h.finalize();
            let want_ref = crc32_ref_continue(seed, data);
            proptest::prop_assert_eq!(got, want_fast, "fold != crc32fast");
            proptest::prop_assert_eq!(got, want_ref, "fold != bit-serial ref");
        }
    }
}
