//! Shared Lempel-Ziv matchfinding primitives.
//!
//! Port of libdeflate `vendor/libdeflate/lib/matchfinder_common.h`: the hash
//! (`lz_hash`, :168-172), word-at-a-time match extension (`lz_extend`,
//! :178-222), and the saturating-`i16` position-table init/rebase
//! (`matchfinder_init` / `matchfinder_rebase`, :106-159, plus `MATCHFINDER_INITVAL`
//! at :51). No actual match finder is built yet — these are the shared building
//! blocks, unit-tested here.

/// Standard DEFLATE matchfinder window order (32 KiB window).
pub const MATCHFINDER_WINDOW_ORDER: u32 = 15;
pub const MATCHFINDER_WINDOW_SIZE: i32 = 1 << MATCHFINDER_WINDOW_ORDER;
/// `MATCHFINDER_INITVAL = -WINDOW_SIZE`, which is exactly `i16::MIN` for a
/// 32 KiB window.
pub const MATCHFINDER_INITVAL: i16 = (-MATCHFINDER_WINDOW_SIZE) as i16;

const WORDBYTES: usize = 8;

/// The matchfinder hash: multiply the low bits of a sequence by a large
/// constant and take the top `num_bits` bits of the 32-bit product.
///
/// `lz_hash(seq, bits) = (seq * 0x1E35A7BD) >> (32 - bits)` with wrapping
/// multiply.
#[inline(always)]
pub fn lz_hash(seq: u32, num_bits: u32) -> u32 {
    seq.wrapping_mul(0x1E35A7BD) >> (32 - num_bits)
}

/// Unaligned little-endian 4-byte load from `base.add(off)` (port of libdeflate
/// `get_unaligned_le32`). Raw-pointer codegen — no slice bounds check, mirroring
/// the C hot loop.
///
/// # Safety
/// The caller MUST guarantee the 4 bytes `[off, off + 4)` are within the
/// allocation `base` points into. Every call site in the matchfinder pairs this
/// with a `debug_assert!(off + 4 <= buf.len())` proving that from the module's
/// soundness invariant.
#[inline(always)]
pub unsafe fn load_u32(base: *const u8, off: usize) -> u32 {
    // read_unaligned tolerates any alignment; `.to_le()`/`from_le` normalizes to
    // little-endian to match `get_unaligned_le32` on every target.
    u32::from_le(core::ptr::read_unaligned(base.add(off) as *const u32))
}

/// Unaligned little-endian low-3-byte load (port of `load_u24_unaligned`):
/// a `load_u32` masked to its low 24 bits.
///
/// # Safety
/// Same contract as [`load_u32`]: the 4 bytes `[off, off + 4)` must be in bounds
/// (the u24 path always has >= 4 readable bytes at the call site).
#[inline(always)]
pub unsafe fn load_u24(base: *const u8, off: usize) -> u32 {
    load_u32(base, off) & 0x00FF_FFFF
}

/// Unaligned little-endian 8-byte load from `base.add(off)`.
///
/// # Safety
/// The 8 bytes `[off, off + 8)` must be within the allocation `base` points into.
#[inline(always)]
unsafe fn load_u64(base: *const u8, off: usize) -> u64 {
    u64::from_le(core::ptr::read_unaligned(base.add(off) as *const u64))
}

/// Return the length of the match between the bytes at `str_pos` and
/// `match_pos` within `data`, starting from `start_len` already-matched bytes
/// and never exceeding `max_len`.
///
/// Word-at-a-time XOR + trailing-zero-count fast path (port of `lz_extend`).
/// Contract: `str_pos + max_len <= data.len()` and `match_pos + max_len <=
/// data.len()`, so the word loads never read out of bounds.
#[inline(always)]
pub fn lz_extend(
    data: &[u8],
    str_pos: usize,
    match_pos: usize,
    start_len: u32,
    max_len: u32,
) -> u32 {
    let mut len = start_len as usize;
    let max = max_len as usize;
    let base = data.as_ptr();

    // SAFETY: the contract is `str_pos + max_len <= data.len()` and
    // `match_pos + max_len <= data.len()` (documented above; upheld by
    // `adjust_max_and_nice_len` clamping max_len to the remaining input). Every
    // load below reads bytes at offset `< str_pos + max` or `< match_pos + max`,
    // all `<= data.len()`. The debug_asserts trap any caller that breaks it.
    debug_assert!(str_pos + max <= data.len());
    debug_assert!(match_pos + max <= data.len());
    unsafe {
        while len + WORDBYTES <= max {
            // Reads [match_pos+len, +8) and [str_pos+len, +8); len+8 <= max.
            let v = load_u64(base, match_pos + len) ^ load_u64(base, str_pos + len);
            if v != 0 {
                // Little-endian: the first differing byte is at trailing_zeros/8.
                return (len + (v.trailing_zeros() as usize >> 3)) as u32;
            }
            len += WORDBYTES;
        }

        while len < max
            && *data.get_unchecked(str_pos + len) == *data.get_unchecked(match_pos + len)
        {
            len += 1;
        }
    }
    len as u32
}

/// Initialize a matchfinder position table to `MATCHFINDER_INITVAL`
/// (port of `matchfinder_init`).
#[inline]
pub fn matchfinder_init(data: &mut [i16]) {
    for d in data.iter_mut() {
        *d = MATCHFINDER_INITVAL;
    }
}

/// Slide the matchfinder by one window: subtract `WINDOW_SIZE` from each entry
/// with signed saturation to `-WINDOW_SIZE`. Branchless 32768-window form from
/// `matchfinder_rebase`:
/// `data[i] = 0x8000 | (data[i] & ~(data[i] >> 15))`.
#[inline]
pub fn matchfinder_rebase(data: &mut [i16]) {
    for d in data.iter_mut() {
        let v = *d;
        // v >> 15 is an arithmetic shift: -1 for negatives, 0 for non-negatives.
        *d = (0x8000u16 as i16) | (v & !(v >> 15));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lz_hash_matches_formula() {
        for &seq in &[0u32, 1, 0xDEADBEEF, 0x0000_0001, 0xFFFF_FFFF] {
            for bits in 1..=16u32 {
                let expected = seq.wrapping_mul(0x1E35A7BD) >> (32 - bits);
                assert_eq!(lz_hash(seq, bits), expected);
                assert!(lz_hash(seq, bits) < (1 << bits));
            }
        }
    }

    #[test]
    fn lz_extend_full_and_partial() {
        // Two identical regions of 40 bytes.
        let mut data = vec![0u8; 100];
        for i in 0..50 {
            data[i] = (i % 7) as u8;
            data[50 + i] = (i % 7) as u8;
        }
        // str at 0, match at 50, both 50 bytes identical => bounded by max_len.
        let len = lz_extend(&data, 0, 50, 0, 40);
        assert_eq!(len, 40);
    }

    #[test]
    fn lz_extend_stops_at_first_difference() {
        let mut data = vec![0u8; 64];
        // Fill two windows identically for 11 bytes, then diverge at byte 11.
        for i in 0..32 {
            data[i] = i as u8;
            data[32 + i] = i as u8;
        }
        data[32 + 11] = 0xFF; // diverge at offset 11 in the second window
        let len = lz_extend(&data, 0, 32, 0, 32);
        assert_eq!(len, 11);
    }

    #[test]
    fn lz_extend_word_boundary_difference() {
        // Difference exactly at offset 8 (second word) exercises the word loop.
        let mut data = vec![0u8; 64];
        for i in 0..24 {
            data[i] = 0xAB;
            data[24 + i] = 0xAB;
        }
        data[24 + 8] = 0x00;
        let len = lz_extend(&data, 0, 24, 0, 24);
        assert_eq!(len, 8);
    }

    #[test]
    fn lz_extend_respects_start_len() {
        let data = vec![0x5Au8; 64];
        // start_len=4 means "assume 4 already matched"; identical data => max.
        assert_eq!(lz_extend(&data, 0, 8, 4, 20), 20);
    }

    #[test]
    fn matchfinder_init_sets_initval() {
        let mut tab = vec![0i16; 16];
        matchfinder_init(&mut tab);
        assert!(tab.iter().all(|&x| x == MATCHFINDER_INITVAL));
        assert_eq!(MATCHFINDER_INITVAL, i16::MIN);
        assert_eq!(MATCHFINDER_INITVAL, -32768);
    }

    #[test]
    fn matchfinder_rebase_saturating_subtract() {
        let mut tab: Vec<i16> = vec![0, 5, 100, 32767, -1, -100, -32768, MATCHFINDER_INITVAL];
        matchfinder_rebase(&mut tab);
        // Non-negative v => v - 32768; negative v => saturated to -32768.
        assert_eq!(tab[0], -32768); // 0 - 32768
        assert_eq!(tab[1], -32763); // 5 - 32768
        assert_eq!(tab[2], -32668); // 100 - 32768
        assert_eq!(tab[3], -1); // 32767 - 32768
        assert_eq!(tab[4], -32768); // negative saturates
        assert_eq!(tab[5], -32768);
        assert_eq!(tab[6], -32768);
        assert_eq!(tab[7], -32768);
    }

    #[test]
    fn matchfinder_rebase_matches_general_branch() {
        // Cross-check the branchless form against the plain saturating subtract
        // for every representable i16.
        for v in i16::MIN..=i16::MAX {
            let mut one = [v];
            matchfinder_rebase(&mut one);
            let expected: i16 = if v >= 0 {
                (v as i32 - MATCHFINDER_WINDOW_SIZE) as i16
            } else {
                (-MATCHFINDER_WINDOW_SIZE) as i16
            };
            assert_eq!(one[0], expected, "v={v}");
        }
    }
}
