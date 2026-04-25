//! Universal inflate-from-bit backend.
//!
//! On x86_64 + ISA-L: delegates to isal_decompress (~1500 MB/s).
//! On all other platforms (including arm64): uses libz-ng inflatePrime (~600 MB/s).
//!
//! Both backends implement the same inflatePrime pattern:
//!   inflateInit2(-15)           — raw deflate, no gzip/zlib wrapper
//!   inflateSetDictionary(dict)  — prime the LZ77 sliding window
//!   inflatePrime(bits, value)   — pre-load partial byte for non-byte-aligned start
//!   inflate loop                — decode to output buffer
//!   inflatePrime(-1, 0)         — query remaining bits in bit buffer
//!
//! end_bit formula (valid for both bit_skip=0 and bit_skip>0):
//!   end_bit = data.len()*8 - avail_in*8 - bits_in_buffer
//! Derivation: total input bits = initial_avail_in*8 + pre_loaded_bits.
//! Remaining bits = final_avail_in*8 + bits_in_buffer.
//! end_bit = start_bit + consumed = simplifies to the formula above regardless of bit_skip.

/// Always true — zlib-ng is a static dep available on all platforms.
#[inline]
pub fn is_available() -> bool {
    true
}

// ── ISA-L path (x86_64 only) ─────────────────────────────────────────────────

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub fn decompress_deflate_from_bit(
    data: &[u8],
    bit_offset: usize,
    dict: &[u8],
    min_output: usize,
) -> Option<Vec<u8>> {
    crate::backends::isal_decompress::decompress_deflate_from_bit(
        data, bit_offset, dict, min_output,
    )
}

#[cfg(all(feature = "isal-compression", target_arch = "x86_64"))]
pub fn decompress_deflate_from_bit_with_end(
    data: &[u8],
    bit_offset: usize,
    dict: &[u8],
    max_output: usize,
) -> Option<(Vec<u8>, usize)> {
    crate::backends::isal_decompress::decompress_deflate_from_bit_with_end(
        data, bit_offset, dict, max_output,
    )
}

// ── zlib-ng path (all platforms, primary on arm64) ───────────────────────────

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
pub fn decompress_deflate_from_bit(
    data: &[u8],
    bit_offset: usize,
    dict: &[u8],
    min_output: usize,
) -> Option<Vec<u8>> {
    const MAX_CAP: usize = 512 * 1024 * 1024;
    let cap = min_output.clamp(256 * 1024, MAX_CAP);
    let (out, _) = decompress_zng(data, bit_offset, dict, cap)?;
    if out.len() >= min_output {
        Some(out)
    } else {
        None
    }
}

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
pub fn decompress_deflate_from_bit_with_end(
    data: &[u8],
    bit_offset: usize,
    dict: &[u8],
    max_output: usize,
) -> Option<(Vec<u8>, usize)> {
    // Cap to avoid allocation overflow when callers pass usize::MAX.
    const MAX_CAP: usize = 512 * 1024 * 1024;
    decompress_zng(
        data,
        bit_offset,
        dict,
        max_output.clamp(256 * 1024, MAX_CAP),
    )
}

#[cfg(not(all(feature = "isal-compression", target_arch = "x86_64")))]
fn decompress_zng(
    data: &[u8],
    bit_offset: usize,
    dict: &[u8],
    cap: usize,
) -> Option<(Vec<u8>, usize)> {
    use libz_ng_sys as zng;
    use std::mem;
    use std::ptr;

    let byte_idx = bit_offset / 8;
    let bit_skip = bit_offset % 8;

    if byte_idx >= data.len() {
        return None;
    }

    // z_stream has fn-pointer fields (zalloc/zfree) that Rust treats as NonNull,
    // but zlib interprets null as "use default malloc/free". Use write_bytes to
    // zero-fill without triggering Rust's validity checks.
    let mut strm: zng::z_stream = unsafe {
        let mut m = mem::MaybeUninit::<zng::z_stream>::uninit();
        ptr::write_bytes(m.as_mut_ptr(), 0, 1);
        m.assume_init()
    };

    // Raw deflate: windowBits = -15 (no gzip/zlib wrapper)
    let ret = unsafe {
        zng::inflateInit2_(
            &mut strm,
            -15,
            ptr::null(),
            mem::size_of::<zng::z_stream>() as i32,
        )
    };
    if ret != zng::Z_OK {
        return None;
    }

    // Prime the LZ77 sliding window. Gzip streams start with an empty (zeroed)
    // 32KB window; mid-stream decodes need the actual preceding output.
    // When no dict is given, use zeros — this mirrors ISA-L's zeroed output buffer
    // and prevents Z_DATA_ERROR on back-references before position 0.
    static ZERO_WINDOW: [u8; 32768] = [0u8; 32768];
    let window = if dict.is_empty() {
        &ZERO_WINDOW[..]
    } else {
        dict
    };
    {
        let ret =
            unsafe { zng::inflateSetDictionary(&mut strm, window.as_ptr(), window.len() as u32) };
        if ret != zng::Z_OK {
            unsafe { zng::inflateEnd(&mut strm) };
            return None;
        }
    }

    // Pre-load the upper (8-bit_skip) bits of the partial byte via inflatePrime,
    // then point next_in past that byte.
    if bit_skip > 0 {
        let bits = (8 - bit_skip) as i32;
        let value = (data[byte_idx] >> bit_skip) as i32;
        let ret = unsafe { zng::inflatePrime(&mut strm, bits, value) };
        if ret != zng::Z_OK {
            unsafe { zng::inflateEnd(&mut strm) };
            return None;
        }
        strm.next_in = unsafe { data.as_ptr().add(byte_idx + 1) as *mut _ };
        strm.avail_in = (data.len() - byte_idx - 1) as u32;
    } else {
        strm.next_in = unsafe { data.as_ptr().add(byte_idx) as *mut _ };
        strm.avail_in = (data.len() - byte_idx) as u32;
    }

    let mut output = vec![0u8; cap];
    let mut out_pos = 0usize;

    loop {
        let remaining = cap - out_pos;
        if remaining == 0 {
            break;
        }

        strm.next_out = unsafe { output.as_mut_ptr().add(out_pos) };
        strm.avail_out = remaining as u32;

        let ret = unsafe { zng::inflate(&mut strm, zng::Z_NO_FLUSH) };
        let written = remaining - strm.avail_out as usize;
        out_pos += written;

        if ret == zng::Z_STREAM_END {
            break;
        } else if ret == zng::Z_OK {
            if written == 0 && strm.avail_in == 0 {
                break;
            }
        } else {
            // Z_DATA_ERROR, Z_STREAM_ERROR, Z_BUF_ERROR, etc.
            if out_pos == 0 {
                unsafe { zng::inflateEnd(&mut strm) };
                return None;
            }
            break;
        }
    }

    if out_pos == 0 {
        unsafe { zng::inflateEnd(&mut strm) };
        return None;
    }

    // inflatePrime(-1, 0): query bits still in internal bit buffer (not yet consumed by inflate)
    let bits_in_buffer = unsafe { zng::inflatePrime(&mut strm, -1, 0) }.max(0) as usize;

    // end_bit = data.len()*8 - avail_in*8 - bits_in_buffer
    // Works for both bit_skip=0 (next_in = byte_idx) and bit_skip>0 (next_in = byte_idx+1)
    // because data.len() encompasses the full slice including the pre-loaded partial byte.
    let end_bit = data.len() * 8 - strm.avail_in as usize * 8 - bits_in_buffer;

    unsafe { zng::inflateEnd(&mut strm) };

    output.truncate(out_pos);
    Some((output, end_bit))
}
