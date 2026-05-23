#![allow(dead_code)] // vendor-faithful rapidgzip port; many items are pending consumer-port

//! SIMD-accelerated marker replacement.
//!
//! Literal port of rapidgzip's `MapMarkers` semantics
//! (vendor/rapidgzip/.../MarkerReplacement.hpp:24-42):
//! ```text
//!   marker_value <= 255      → literal byte
//!   marker_value >= 32768    → window[marker_value - MARKER_BASE]   (index from OLDEST byte)
//!   256 <= marker_value < 32768 → invalid (rapidgzip throws)
//! ```
//! The window is a fixed 32 KiB buffer; shorter predecessor windows must
//! be padded with leading zeros by the caller (matches rapidgzip's
//! `DecodedData::getWindowAt` prefill at DecodedData.hpp:418-432).
//!
//! Switched from the old "offset from newest byte" encoding to align with
//! rapidgzip's `MapMarkers` so any downstream rapidgzip code that depends
//! on marker semantics ports byte-for-byte.

/// Values at or above this are markers; values below are literal bytes.
pub const MARKER_BASE: u16 = 32768;

/// Resolve every marker in `data` against `window`. Bytes already in 0..=255
/// are left as-is. Markers whose offset would read past the start of `window`
/// (shouldn't happen for a well-formed deflate stream + 32 KB window) are left
/// at their marker value — phase 3 will detect this via CRC mismatch.
///
/// **Vendor's branchless LUT path** (`DecodedData.hpp:314-338`): for
/// large marker buffers (≥ 128 KiB), vendor builds a 64 KiB lookup
/// table where slots `[0..256)` are iota-initialized (identity for
/// literals), `[256..32768)` are zero (unused range — well-formed
/// streams never produce these values), and `[32768..65536)` hold
/// the window bytes. The hot loop becomes `target[i] =
/// fullWindow[chunk[i]]` — a single branchless load per element.
///
/// The threshold matters: building the 64 KiB LUT costs ~64 KiB of
/// stack zeroing + a 32 KiB window copy ≈ 96 KiB written. For tiny
/// marker buffers, the per-element branch is cheaper than building
/// the LUT. Vendor's 128 KiB threshold is empirical.
///
/// Falls back to the per-element branch path for small markers (and
/// is the only path on hosts without runtime AVX2 detection or with
/// the lookup-table size budget exceeded — currently 64 KiB on stack
/// is always available).
pub fn replace_markers(data: &mut [u16], window: &[u8]) {
    // Vendor's threshold (`DecodedData.hpp:315`): `markerCount >= 128_Ki`.
    // gzippy's `dataWithMarkers` is u16, so a u16 element count of
    // 128 KiB corresponds to 256 KiB of bytes; we use the element
    // count directly (matches vendor's `markerCount` semantic which
    // is element count of the `Vec<uint16_t>` markers).
    const LUT_THRESHOLD_ELEMENTS: usize = 128 * 1024;
    if data.len() >= LUT_THRESHOLD_ELEMENTS && window.len() == 32768 {
        replace_markers_lut(data, window);
        return;
    }

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

/// Branchless lookup-table marker replacement. Literal port of
/// vendor's `DecodedData::applyWindow` body at
/// `DecodedData.hpp:316-337`. Builds a 64 KiB stack-allocated LUT
/// where:
///   - `lut[0..256]` = identity (`0u8..=255u8`) — literal bytes pass
///     through unchanged.
///   - `lut[256..32768]` = 0 — unused range; well-formed streams
///     never produce these values.
///   - `lut[32768..65536]` = `window[0..32768]` — markers (with
///     `MARKER_BASE` offset already encoded in the high bit).
///
/// Hot loop: `data[i] = lut[data[i] as usize] as u16` — one load,
/// zero branches. Auto-vectorizable.
///
/// Requires `window.len() == 32768`. Caller's responsibility (the
/// `replace_markers` dispatcher checks).
fn replace_markers_lut(data: &mut [u16], window: &[u8]) {
    debug_assert_eq!(
        window.len(),
        32768,
        "vendor LUT path requires 32 KiB window"
    );
    let mut lut = [0u8; 65536];
    // [0..256): iota — literal byte passthrough.
    for (i, slot) in lut[0..256].iter_mut().enumerate() {
        *slot = i as u8;
    }
    // [256..32768): zero (already initialized to 0).
    // [32768..65536): the predecessor window.
    lut[MARKER_BASE as usize..MARKER_BASE as usize + 32768].copy_from_slice(window);

    // Hot loop. Vendor's comment at DecodedData.hpp:327-331 explicitly
    // avoids `std::transform` because out-of-order execution could
    // overwrite slots before they're read — but for our in-place u16
    // mutation, the read of `data[i]` happens before the write, so
    // out-of-order is safe. We use a tight indexed loop to give LLVM
    // the freedom to auto-vectorize.
    for val in data.iter_mut() {
        *val = lut[*val as usize] as u16;
    }
}

#[inline]
fn replace_markers_scalar(data: &mut [u16], window: &[u8]) {
    for val in data.iter_mut() {
        if *val >= MARKER_BASE {
            let offset = (*val - MARKER_BASE) as usize;
            if offset < window.len() {
                *val = window[offset] as u16;
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
                        data[i + j] = window[offset] as u16;
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
                *val = window[offset] as u16;
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
                            data[i + j] = window[offset] as u16;
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
                *val = window[offset] as u16;
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
        // MapMarkers semantics: marker_value - MARKER_BASE indexes into
        // window from the oldest byte.
        // marker 0 → window[0] = 'x'
        // marker 1 → window[1] = 'y'
        assert_eq!(
            data,
            vec![b'a' as u16, b'x' as u16, b'y' as u16, b'd' as u16]
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
        // MapMarkers: marker i → window[i] for i = 0..6 → a,b,c,d,e,f,g
        let expected: Vec<u16> = b"abcdefg".iter().map(|&b| b as u16).collect();
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
                let idx = i % 10;
                assert_eq!(v, window[idx] as u16);
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

    /// Vendor-faithfulness gate for the LUT path
    /// (`DecodedData.hpp:314-338`). Exercises the ≥ 128 KiB threshold
    /// that triggers `replace_markers_lut` and asserts byte-equivalent
    /// output against the scalar reference path. Previously NO test
    /// hit the LUT branch — flagged by advisor #10 after commit
    /// ec55351 landed.
    #[test]
    fn lut_path_matches_scalar_above_threshold() {
        // 200K-element buffer: well over the 128 KiB threshold.
        const N: usize = 200_000;
        // Deterministic mixed marker / literal pattern. Use a PRNG-shaped
        // sequence to hit both literal and marker values across the
        // full marker offset range [0, 32767].
        let mut input = Vec::with_capacity(N);
        let mut x: u32 = 0x9E37_79B9; // golden-ratio seed
        for i in 0..N {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            // 70% markers, 30% literals — typical compressed-text mix
            if x % 10 < 7 {
                input.push(MARKER_BASE + ((x >> 8) % 32768) as u16);
            } else {
                input.push((i % 256) as u16);
            }
        }
        let window: Vec<u8> = (0..32768).map(|i| (i * 37 + 13) as u8).collect();

        // Run the scalar path for ground truth.
        let mut scalar_out = input.clone();
        replace_markers_scalar(&mut scalar_out, &window);

        // Run the dispatching `replace_markers` — should pick the LUT path.
        let mut lut_out = input.clone();
        replace_markers(&mut lut_out, &window);

        // Byte-equivalent on the LOW BYTE of each u16 (the only byte
        // production code reads via `*v as u8` narrow). The LUT path
        // writes the FULL u16 with high byte = 0; the scalar path
        // leaves the high byte at whatever the input had — so they
        // can differ in high byte for slots that stayed literal.
        // We assert the low-byte equivalence which is what matters.
        assert_eq!(scalar_out.len(), lut_out.len());
        for (i, (s, l)) in scalar_out.iter().zip(lut_out.iter()).enumerate() {
            assert_eq!(
                *s as u8, *l as u8,
                "scalar/lut diverge at index {}: scalar=0x{:04x} lut=0x{:04x}",
                i, s, l
            );
            // After replacement, no value above MARKER_BASE should remain
            // (window covers the full 32 KiB → every marker resolves).
            assert!(
                *l < MARKER_BASE,
                "unresolved marker in LUT output at {i}: 0x{:04x}",
                l
            );
        }
    }
}
