//! SIMD-accelerated marker replacement.
//!
//! Phase 2 of the marker-based parallel single-member decoder: walk a chunk's
//! `Vec<u16>` output, replace each marker (value ≥ 32768) with the corresponding
//! byte from the predecessor chunk's last 32 KB window. Bytes already in
//! 0..=255 are left untouched.
//!
//! Marker encoding (matches the convention in `marker_decode.rs`):
//! ```text
//!   marker_value = 32768 + offset
//!   resolved_byte = window[window.len() - 1 - offset]
//! ```
//! So offset = 0 → last byte of window (most recently emitted), offset increases
//! going backward. The deflate max back-reference distance is 32768, so offset
//! fits in 15 bits and `marker_value` always has the high bit set.
//!
//! This file is the resurrected SIMD body from the deleted
//! `src/hyper_parallel.rs` (commit `3eba641^`). Repurposed unchanged for the
//! v0.5.x marker pipeline.

#![allow(dead_code)]

/// Values at or above this are markers; values below are literal bytes.
pub const MARKER_BASE: u16 = 32768;

/// Resolve every marker in `data` against `window`. Bytes already in 0..=255
/// are left as-is. Markers whose offset would read past the start of `window`
/// (shouldn't happen for a well-formed deflate stream + 32 KB window) are left
/// at their marker value — phase 3 will detect this via CRC mismatch.
pub fn replace_markers(data: &mut [u16], window: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 just detected at runtime.
            unsafe { replace_markers_avx2(data, window) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        replace_markers_neon(data, window);
        return;
    }
    #[allow(unreachable_code)]
    replace_markers_scalar(data, window);
}

#[inline]
fn replace_markers_scalar(data: &mut [u16], window: &[u8]) {
    for val in data.iter_mut() {
        if *val >= MARKER_BASE {
            let offset = (*val - MARKER_BASE) as usize;
            if offset < window.len() {
                *val = window[window.len() - 1 - offset] as u16;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn replace_markers_avx2(data: &mut [u16], window: &[u8]) {
    use std::arch::x86_64::*;

    if data.is_empty() || window.is_empty() {
        // Still walk to satisfy callers in degenerate cases — but with no
        // window there's nothing to substitute.
        return;
    }

    // Lane-wide mask: high bit set → marker. Avoids the signed comparison
    // pitfall with `_mm256_cmpgt_epi16` (markers ≥ 32768 are negative as i16).
    let marker_bit = _mm256_set1_epi16(i16::MIN); // 0x8000

    let mut i = 0;
    let simd_end = data.len().saturating_sub(16);
    while i < simd_end {
        let v = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);
        let masked = _mm256_and_si256(v, marker_bit);
        let cmp = _mm256_cmpeq_epi16(masked, marker_bit);
        let any_markers = _mm256_movemask_epi8(cmp);

        if any_markers != 0 {
            // At least one of the next 16 lanes is a marker. Fall through to
            // a scalar fix-up for those exact lanes; the common case (no
            // markers in the next 16) is a single AVX2 vector test.
            for j in 0..16 {
                let val = data[i + j];
                if val >= MARKER_BASE {
                    let offset = (val - MARKER_BASE) as usize;
                    if offset < window.len() {
                        data[i + j] = window[window.len() - 1 - offset] as u16;
                    }
                }
            }
        }
        i += 16;
    }

    // Tail.
    for val in &mut data[i..] {
        if *val >= MARKER_BASE {
            let offset = (*val - MARKER_BASE) as usize;
            if offset < window.len() {
                *val = window[window.len() - 1 - offset] as u16;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn replace_markers_neon(data: &mut [u16], window: &[u8]) {
    use std::arch::aarch64::*;

    if data.is_empty() || window.is_empty() {
        return;
    }

    let mut i = 0;
    let simd_end = data.len().saturating_sub(8);

    // 8×u16 per lane.
    let marker_threshold = unsafe { vdupq_n_u16(MARKER_BASE - 1) };

    while i < simd_end {
        unsafe {
            let v = vld1q_u16(data.as_ptr().add(i));
            // vcgtq_u16: unsigned compare > threshold → all-ones for markers.
            let mask = vcgtq_u16(v, marker_threshold);
            // Reduce: any lane set → at least one marker in this block.
            let any_markers = vmaxvq_u16(mask);

            if any_markers != 0 {
                for j in 0..8 {
                    let val = data[i + j];
                    if val >= MARKER_BASE {
                        let offset = (val - MARKER_BASE) as usize;
                        if offset < window.len() {
                            data[i + j] = window[window.len() - 1 - offset] as u16;
                        }
                    }
                }
            }
        }
        i += 8;
    }

    for val in &mut data[i..] {
        if *val >= MARKER_BASE {
            let offset = (*val - MARKER_BASE) as usize;
            if offset < window.len() {
                *val = window[window.len() - 1 - offset] as u16;
            }
        }
    }
}

/// Convert a `Vec<u16>` (with all markers already resolved, so every value
/// fits in u8) to a `Vec<u8>`. Returns Err if any value still has the marker
/// bit set — callers should call this only after `replace_markers`.
pub fn u16_to_u8(data: &[u16]) -> Result<Vec<u8>, usize> {
    if let Some(bad) = data.iter().position(|&v| v >= MARKER_BASE) {
        return Err(bad);
    }
    Ok(data.iter().map(|&v| v as u8).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_resolves_simple_marker() {
        let mut data = vec![b'a' as u16, MARKER_BASE, MARKER_BASE + 1, b'd' as u16];
        let window = b"xy";
        replace_markers(&mut data, window);
        // offset 0 → window[1] = 'y'
        // offset 1 → window[0] = 'x'
        assert_eq!(
            data,
            vec![b'a' as u16, b'y' as u16, b'x' as u16, b'd' as u16]
        );
    }

    #[test]
    fn no_markers_passes_through() {
        let mut data: Vec<u16> = b"hello world".iter().map(|&b| b as u16).collect();
        let copy = data.clone();
        replace_markers(&mut data, b"window");
        assert_eq!(data, copy);
    }

    #[test]
    fn empty_window_leaves_markers_alone() {
        let mut data = vec![b'a' as u16, MARKER_BASE, MARKER_BASE + 5, b'b' as u16];
        replace_markers(&mut data, &[]);
        // Markers untouched; phase 3 will detect via CRC.
        assert_eq!(data[1], MARKER_BASE);
        assert_eq!(data[2], MARKER_BASE + 5);
    }

    #[test]
    fn handles_size_below_simd_step() {
        // SIMD path skips the last vector-width chunk; this guarantees the
        // tail loop covers correctness for small inputs.
        let mut data: Vec<u16> = (0..7u16).map(|i| MARKER_BASE + i).collect();
        let window = b"abcdefghij";
        replace_markers(&mut data, window);
        // offset 0..6 → window[9..3] = j,i,h,g,f,e,d
        let expected: Vec<u16> = b"jihgfed".iter().map(|&b| b as u16).collect();
        assert_eq!(data, expected);
    }

    #[test]
    fn handles_long_runs_with_intermittent_markers() {
        let mut data: Vec<u16> = Vec::with_capacity(1000);
        for i in 0..1000 {
            if i % 7 == 0 {
                data.push(MARKER_BASE + (i % 10) as u16);
            } else {
                data.push(b'.' as u16);
            }
        }
        let window: Vec<u8> = (0..32u8).collect();
        replace_markers(&mut data, &window);
        for (i, &v) in data.iter().enumerate() {
            if i % 7 == 0 {
                let offset = i % 10;
                assert_eq!(v, window[window.len() - 1 - offset] as u16);
            } else {
                assert_eq!(v, b'.' as u16);
            }
        }
    }

    #[test]
    fn u16_to_u8_rejects_unresolved_markers() {
        let data = vec![b'a' as u16, MARKER_BASE + 3, b'c' as u16];
        match u16_to_u8(&data) {
            Err(1) => {}
            other => panic!("expected Err(1), got {:?}", other),
        }
    }

    #[test]
    fn u16_to_u8_passes_clean_input() {
        let data: Vec<u16> = b"hello".iter().map(|&b| b as u16).collect();
        assert_eq!(u16_to_u8(&data).unwrap(), b"hello".to_vec());
    }
}
