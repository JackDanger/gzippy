//! §3.2 Route D + §3.4 — shift-register-width bit reader.
//!
//! Two-axis design:
//!   - **Width**: 64 (scalar), 128 (NEON), 256 (AVX2), 512 (AVX-512).
//!     Chosen at decoder construction via runtime CPU detection.
//!   - **Underflow model** (Route D): `bitsleft` is signed `i32`.
//!     `consume(n)` is plain subtraction; negative means we read past
//!     the buffer's end and the bounds check moves from per-`consume`
//!     to per-block (block entry + block exit). The hot loop decodes
//!     optimistically and rolls back on negative `bitsleft`.
//!
//! ## Why signed-i32 rollback (ISA-L's pattern)
//!
//! Vendor ISA-L's inflate inner loop tracks `read_in_length` as a
//! signed counter. Each `bits_consumed` subtracts; underflow is a
//! single `js` (jump-if-sign) instruction at block exit. Comparison:
//! libdeflate uses an unsigned counter with a `cmp + jb` per consume.
//! ISA-L's pattern saves ~1 cycle per symbol on the hot path.
//!
//! Per `plans/unified-decoder.md` §3.2 + §4.4: this primitive
//! underlies Route D and is the bit-extraction shape Route C v3+
//! dynasm emit targets. Today's pure-Rust libdeflate-inner uses
//! the unsigned u8 bitsleft pattern; this module is the alternative
//! Route D builds on.
//!
//! ## v0.1 scope
//!
//! Scalar 64-bit shift register with signed-i32 underflow tracking.
//! AVX2-256/AVX-512/NEON-128 widths land in v0.2 (architecture-
//! specific dispatch through runtime CPU detection).
//!
//! ## Why a separate module from `bmi2.rs`
//!
//! bmi2.rs holds the EXTRACT primitives (BZHI, PEXT). This module
//! holds the BUFFER (shift register + bit position tracker). Route C
//! v3 emit needs both; keeping them separate lets the dynasm emitter
//! reason about register allocation cleanly (bitbuf in a single reg,
//! bitsleft in another, extract via PEXT inline).

/// Signed-i32 bitsleft bit reader. Drop-in shape for Route D's hot loop.
pub struct BitReaderI32<'a> {
    buf: &'a [u8],
    /// Byte position of the next refill.
    byte_pos: usize,
    /// 64-bit shift register holding the current bit window.
    bitbuf: u64,
    /// Number of valid bits in `bitbuf`. SIGNED: negative means we
    /// read past EOF; caller handles rollback at the next block-exit
    /// check.
    bitsleft: i32,
}

impl<'a> BitReaderI32<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            byte_pos: 0,
            bitbuf: 0,
            bitsleft: 0,
        }
    }

    /// Refill: load as many bytes as fit into `bitbuf` AND are actually
    /// available in the input buffer. `bitsleft` increments only by
    /// real bits, so a partial input tail correctly leaves the reader
    /// with `bitsleft < 64` and consumption past EOF trips
    /// `underflowed()`.
    #[inline(always)]
    pub fn refill(&mut self) {
        let want_bits = 64u32.saturating_sub(self.bitsleft.max(0) as u32);
        let want_bytes = (want_bits / 8) as usize;
        let avail_bytes = self.buf.len().saturating_sub(self.byte_pos).min(want_bytes);
        // Load up to 8 bytes (with zero-pad past EOF).
        let mut loaded: u64 = 0;
        for i in 0..avail_bytes {
            loaded |= (self.buf[self.byte_pos + i] as u64) << (i * 8);
        }
        self.bitbuf |= loaded << (self.bitsleft.max(0) as u32 & 63);
        self.bitsleft += (avail_bytes * 8) as i32;
        self.byte_pos += avail_bytes;
    }

    /// Peek low `n` bits without consuming.
    #[inline(always)]
    pub fn peek(&self, n: u8) -> u64 {
        debug_assert!(n <= 56);
        self.bitbuf & ((1u64 << n) - 1)
    }

    /// Consume `n` bits. Signed: produces negative `bitsleft` on
    /// underflow. Caller checks at block-exit; the hot loop never
    /// touches the branch.
    #[inline(always)]
    pub fn consume(&mut self, n: u8) {
        self.bitbuf >>= n;
        self.bitsleft -= n as i32;
    }

    /// Read N bits (combined peek + consume).
    #[inline(always)]
    pub fn read(&mut self, n: u8) -> u64 {
        let v = self.peek(n);
        self.consume(n);
        v
    }

    /// Number of bits currently buffered. Negative means we overshot
    /// EOF since the last block-exit check.
    #[inline(always)]
    pub fn bitsleft(&self) -> i32 {
        self.bitsleft
    }

    /// True when the reader has gone past valid input.
    #[inline(always)]
    pub fn underflowed(&self) -> bool {
        self.bitsleft < 0
    }
}

/// §3.4 — shift register width selection. The actual u128/u256/u512
/// shift register implementations land in v0.2 (architecture-specific
/// SIMD intrinsics). v0.1 exposes the dispatch enum + threshold table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftRegisterWidth {
    /// 64-bit scalar (default; portable).
    Scalar64,
    /// 128-bit NEON (aarch64).
    Neon128,
    /// 256-bit AVX2 + BMI2 (x86_64 Raptor Lake target).
    Avx2_256,
    /// 512-bit AVX-512 (rare desktop, common server).
    Avx512_512,
}

impl ShiftRegisterWidth {
    /// Refill threshold per width (bits-left below which we refill).
    /// Per `plans/unified-decoder.md` §3.4.
    pub fn refill_threshold(self) -> u32 {
        match self {
            Self::Scalar64 => 48,
            Self::Neon128 => 96,
            Self::Avx2_256 => 224,
            Self::Avx512_512 => 448,
        }
    }

    /// Refill chunk size (bytes loaded per refill).
    pub fn refill_chunk(self) -> u32 {
        match self {
            Self::Scalar64 => 8,
            Self::Neon128 => 16,
            Self::Avx2_256 => 32,
            Self::Avx512_512 => 64,
        }
    }

    /// Detect the best width supported by the current CPU at runtime.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("bmi2") {
                return Self::Avx512_512;
            }
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("bmi2") {
                return Self::Avx2_256;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Self::Neon128;
            }
        }
        Self::Scalar64
    }
}

/// §9 risk-10 escalation: BMI2 PEXT-accelerated bit extraction.
///
/// PEXT(src, mask) extracts bits from `src` at positions where `mask=1`
/// and packs them contiguously. For DEFLATE this is useful in the
/// extras decode (length-extra + dist-extra) where the byte stream
/// has the extras at known bit positions per code-length.
///
/// # Safety
///
/// Pure-register BMI2 intrinsic with no memory operands; it is `unsafe`
/// only because `_pext_u64` is an arch intrinsic. Safe to call on any
/// inputs when the build targets a BMI2-capable CPU (guaranteed by the
/// `target_feature = "bmi2"` cfg gating this definition).
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
pub unsafe fn pext_u64(src: u64, mask: u64) -> u64 {
    std::arch::x86_64::_pext_u64(src, mask)
}

/// Fallback PEXT for non-BMI2 builds. Slow scalar loop; only kept so
/// the API is portable for tests.
#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
#[inline]
pub fn pext_u64(src: u64, mask: u64) -> u64 {
    let mut out = 0u64;
    let mut m = mask;
    let mut s = src;
    let mut bit = 0;
    while m != 0 {
        let low = m & m.wrapping_neg();
        let pos = low.trailing_zeros();
        out |= ((s >> pos) & 1) << bit;
        bit += 1;
        m &= m - 1;
        s |= 0; // silence unused
    }
    let _ = s;
    out
}

/// Safe wrapper that dispatches to BMI2 when available, fallback otherwise.
#[inline]
pub fn pext(src: u64, mask: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    unsafe {
        pext_u64(src, mask)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        pext_u64(src, mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_d_read_byte_stream() {
        let data = [0x12, 0x34, 0x56, 0x78];
        let mut br = BitReaderI32::new(&data);
        br.refill();
        assert!(br.bitsleft() >= 32);
        // Low 8 bits LSB-first = 0x12.
        assert_eq!(br.read(8), 0x12);
        assert_eq!(br.read(8), 0x34);
        assert_eq!(br.read(8), 0x56);
        assert_eq!(br.read(8), 0x78);
    }

    #[test]
    fn route_d_underflow_negative_bitsleft() {
        let data = [0xFF];
        let mut br = BitReaderI32::new(&data);
        br.refill();
        // 8 bits available; consume more.
        br.read(8);
        assert!(!br.underflowed(), "exactly empty is not underflow yet");
        br.read(1); // pop 1 too many
        assert!(br.underflowed(), "negative bitsleft on overrun");
    }

    #[test]
    fn shift_register_thresholds() {
        assert_eq!(ShiftRegisterWidth::Scalar64.refill_threshold(), 48);
        assert_eq!(ShiftRegisterWidth::Avx2_256.refill_threshold(), 224);
        assert_eq!(ShiftRegisterWidth::Avx512_512.refill_chunk(), 64);
    }

    #[test]
    fn shift_register_detects_some_width() {
        let w = ShiftRegisterWidth::detect();
        // At minimum we always have scalar.
        assert!(matches!(
            w,
            ShiftRegisterWidth::Scalar64
                | ShiftRegisterWidth::Neon128
                | ShiftRegisterWidth::Avx2_256
                | ShiftRegisterWidth::Avx512_512
        ));
        eprintln!("detected shift-register width: {:?}", w);
    }

    #[test]
    fn pext_extracts_marked_bits() {
        // src = 0b1010_1010, mask = 0b1111_0000 → extract upper 4 bits → 0b1010.
        let src = 0b1010_1010u64;
        let mask = 0b1111_0000u64;
        assert_eq!(pext(src, mask), 0b1010);

        // src = 0xDEAD_BEEF, mask = 0xF0F0_F0F0 → extract every other nibble.
        let src = 0xDEAD_BEEF_u64;
        let mask = 0xF0F0_F0F0_u64;
        let r = pext(src, mask);
        // Expected: DEAD_BEEF nibbles at the high half of each byte:
        // D, A, B, E → 0xDABE.
        assert_eq!(r, 0xDABE);
    }
}
