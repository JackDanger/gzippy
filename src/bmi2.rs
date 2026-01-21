//! BMI2 intrinsics for x86_64 bit manipulation
//!
//! BMI2 provides hardware acceleration for bit extraction and manipulation.
//! Key instruction: BZHI (Zero High Bits) - zeroes bits starting from a specified position.
//!
//! Example: _bzhi_u64(0xFFFF, 8) = 0xFF (keep low 8 bits)
//!
//! This is used for extracting extra bits from saved_bitbuf in Huffman decoding.

#![allow(dead_code)]

/// Extract low `n` bits from `val` using branchless mask
/// Fallback implementation for non-BMI2 targets
#[inline(always)]
pub fn extract_bits_fallback(val: u64, n: u8) -> u64 {
    if n >= 64 {
        val
    } else {
        let mask = (1u64 << n).wrapping_sub(1);
        val & mask
    }
}

/// Check if BMI2 is available at runtime (cached)
#[cfg(target_arch = "x86_64")]
pub fn has_bmi2() -> bool {
    #[cfg(target_feature = "bmi2")]
    {
        true
    }
    #[cfg(not(target_feature = "bmi2"))]
    {
        std::is_x86_feature_detected!("bmi2")
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn has_bmi2() -> bool {
    false
}

/// Extract low `n` bits from `val` using BMI2 BZHI instruction
/// This is typically 1 cycle vs 3+ cycles for shift-and-mask
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
pub unsafe fn extract_bits_bmi2(val: u64, n: u32) -> u64 {
    use std::arch::x86_64::_bzhi_u64;
    _bzhi_u64(val, n)
}

/// Wrapper that uses BMI2 when available, fallback otherwise
#[inline(always)]
pub fn extract_bits(val: u64, n: u8) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        // When compiled with target_feature = "bmi2", use directly
        unsafe { std::arch::x86_64::_bzhi_u64(val, n as u32) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        extract_bits_fallback(val, n)
    }
}

/// Decode extra bits from saved_bitbuf after codeword
/// `saved_bitbuf`: bit buffer state when entry was looked up
/// `codeword_bits`: how many bits the codeword consumed
/// `extra_bits`: how many extra bits to extract
///
/// Inlined and branchless for maximum performance
#[inline(always)]
pub fn decode_extra_bits(saved_bitbuf: u64, codeword_bits: u8, extra_bits: u8) -> u64 {
    // Branchless: shift right to position extra bits at bit 0
    let shifted = saved_bitbuf >> codeword_bits;

    // Branchless mask: when extra_bits is 0, mask is 0
    // When extra_bits is 64, we'd overflow, but DEFLATE max is 13 extra bits
    let mask = (1u64 << extra_bits).wrapping_sub(1);
    shifted & mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_bits_fallback() {
        assert_eq!(extract_bits_fallback(0xFFFF, 8), 0xFF);
        assert_eq!(extract_bits_fallback(0xFFFF, 4), 0xF);
        assert_eq!(extract_bits_fallback(0b11111111, 3), 0b111);
        assert_eq!(extract_bits_fallback(0b10101010, 4), 0b1010);
        assert_eq!(extract_bits_fallback(u64::MAX, 0), 0);
        assert_eq!(extract_bits_fallback(u64::MAX, 64), u64::MAX);
    }

    #[test]
    fn test_decode_extra_bits() {
        // codeword takes 7 bits, 1 extra bit follows
        let saved = 0b1_0000000u64; // bit 7 is 1 (the extra bit)
        assert_eq!(decode_extra_bits(saved, 7, 1), 1);

        // codeword takes 5 bits, 3 extra bits follow
        let saved = 0b101_00000u64; // bits 5-7 are 101 = 5
        assert_eq!(decode_extra_bits(saved, 5, 3), 5);

        // No extra bits
        assert_eq!(decode_extra_bits(0xFFFF, 8, 0), 0);
    }

    #[test]
    fn test_bmi2_detection() {
        let has = has_bmi2();
        eprintln!("BMI2 available: {}", has);
        // This test just ensures detection doesn't crash
    }

    #[test]
    fn bench_extract_bits() {
        use std::time::Instant;

        let iterations = 10_000_000u64;
        let mut sum = 0u64;

        // Benchmark fallback
        let start = Instant::now();
        for i in 0..iterations {
            sum = sum.wrapping_add(extract_bits_fallback(i, (i & 0x3F) as u8));
        }
        let fallback_time = start.elapsed();

        // Benchmark extract_bits (may use BMI2)
        let start = Instant::now();
        for i in 0..iterations {
            sum = sum.wrapping_add(extract_bits(i, (i & 0x3F) as u8));
        }
        let extract_time = start.elapsed();

        eprintln!("\nExtract bits benchmark ({} iterations):", iterations);
        eprintln!("  Fallback: {:.2}ms", fallback_time.as_secs_f64() * 1000.0);
        eprintln!(
            "  extract_bits: {:.2}ms",
            extract_time.as_secs_f64() * 1000.0
        );
        eprintln!(
            "  Speedup: {:.1}x",
            fallback_time.as_secs_f64() / extract_time.as_secs_f64()
        );
        eprintln!("  BMI2 available: {}", has_bmi2());
        eprintln!("  (sum to prevent optimization: {})", sum);
    }
}
