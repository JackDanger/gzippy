//! RFC 1950 zlib container helpers — header, Adler-32, and trailer framing.
//!
//! The gzip containers live inline in the encoders (header functions +
//! `crc32fast`); zlib needs the same services with one difference: a 2-byte
//! CMF/FLG header and a big-endian Adler-32 trailer instead of the 10-byte
//! gzip header and CRC32/ISIZE trailer. The DEFLATE payload between them is
//! produced by the same engines either way.
//!
//! `adler32` is the vanilla per-customer port of Google Zopfli's
//! `ZlibAdler32` (see `deflate/parse/ultra/zlib.rs`, which now reuses this
//! implementation), including the 5550-element chunking that keeps `s2 =
//! s2 + s1` free of 32-bit overflow.

/// Which container wraps the DEFLATE stream. Wired through
/// [`crate::compress::pipelined::PipelinedGzEncoder`] so T>1 stays parallel
/// for both containers (issue #303); the engine and scheduling are
/// container-independent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContainerKind {
    /// RFC 1952 gzip (the default; the historic behaviour, byte-identical).
    #[default]
    Gzip,
    /// RFC 1950 zlib (the `-z/--zlib` CLI flag; issue #303).
    Zlib,
}

/// The fixed 2-byte zlib header: CM=8 (deflate), CINFO=7 (32 KiB window),
/// FLEVEL=3, FDICT=0, FCHECK adjusted so `(CMF<<8 | FLG) % 31 == 0`.
/// Port of the zopfli `zlib_container.c` header recipe (zlib's own
/// construction). FLEVEL is advisory metadata for the decoder; a constant
/// value keeps every level's header identical so only the DEFLATE payload
/// varies.
#[inline]
pub(crate) fn header_bytes() -> [u8; 2] {
    const CMF: u16 = 0x78; // CM = 8, CINFO = 7
    const FLEVEL: u16 = 3;
    const FDICT: u16 = 0;
    let cmfflg = (CMF * 256) + FDICT * 32 + FLEVEL * 64;
    let fcheck = 31 - (cmfflg % 31);
    let v = cmfflg + fcheck;
    [(v >> 8) as u8, (v & 0xff) as u8]
}

/// True when `data` opens with a valid zlib header this crate can decode:
/// CM = 8, no preset dictionary, and the FCHECK sum divisible by 31. Sniffs
/// content, not the file name — used by the CLI's `-t` and by the
/// decompress dispatcher when the extension does not already identify .zz.
pub(crate) fn is_zlib_stream(data: &[u8]) -> bool {
    // `decompress_zlib_turbo`'s documented floor: 2 header + 4 ADLER32 and
    // at least one DEFLATE byte.
    if data.len() < 6 {
        return false;
    }
    let cmf = data[0] as u16;
    let flg = data[1] as u16;
    if cmf & 0x0f != 8 {
        return false; // CM must be 8 (deflate)
    }
    if cmf >> 4 > 7 {
        return false; // CINFO window size > 32 KiB is invalid
    }
    if flg & 0x20 != 0 {
        return false; // FDICT: our inflate path has no preset-dictionary step
    }
    (cmf * 256 + flg).is_multiple_of(31)
}

/// Vanilla Adler-32 (RFC 1950): `s1 = 1 + Σ bytes`, `s2 = Σ s1 - returns`,
/// both mod 65521, packed `(s2 << 16) | s1`. Mirrors the C inner loop
/// exactly, including the 5550-element sum-overflow chunking (the largest
/// count of bytes that can be added without overflowing 32-bit
/// `s2 = s2 + s1`).
pub(crate) fn adler32(data: &[u8]) -> u32 {
    const SUMS_OVERFLOW: usize = 5550;
    let mut s1: u32 = 1;
    let mut s2: u32 = 0;
    let mut rest = data;
    while !rest.is_empty() {
        let amount = rest.len().min(SUMS_OVERFLOW);
        for &b in &rest[..amount] {
            s1 += b as u32;
            s2 += s1;
        }
        s1 %= 65521;
        s2 %= 65521;
        rest = &rest[amount..];
    }
    (s2 << 16) | s1
}

/// Fold the Adler-32 of the NEXT chunk into that of the concatenation so
/// far. `acc` is the forged Adler over the stream prefix, `chunk` the naive
/// `adler32` of the plain chunk bytes, `chunk_len` the chunk's byte length.
///
/// Derivation (all mod 65521), where a stream state is `s1 = 1 + Σ bytes`
/// and `s2 = Σ running s1`:
/// * `s1(A‖B) = s1(A) - 1 + s1(B)` — s1(B) already includes its own 1.
/// * A byte at position p in B sits at L + p in the concatenation, and its
///   running sum picks up `s1(A) - 1` extra, so `s2(A‖B) = s2(A) + s2(B) +
///   L·(s1(A) - 1)`.
/// Used by the T>1 pipeline, which computes the per-chunk Adler in the
/// worker and folds them in block order after the join (the same shape as
/// the gzip path's CRC32 combine), so no sequential pass over the input is
/// added. Pinned equal to a whole-input [`adler32`] pass by the unit test
/// below.
pub(crate) fn adler32_combine(acc: u32, chunk: u32, chunk_len: usize) -> u32 {
    const M: u64 = 65521;
    let s1_acc = (acc & 0xffff) as u64;
    let s2_acc = (acc >> 16) as u64;
    let s1_chunk = (chunk & 0xffff) as u64;
    let s2_chunk = (chunk >> 16) as u64;
    let s1 = (s1_acc + s1_chunk + M - 1) % M;
    let s2 = (s2_acc + s2_chunk + (chunk_len as u64) * ((s1_acc + M - 1) % M)) % M;
    ((s2 << 16) | s1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adler32_matches_known_values() {
        // Reference values from RFC 1950 / common implementations.
        assert_eq!(adler32(b""), 0x0000_0001);
        assert_eq!(adler32(b"a"), 0x0062_0062);
        assert_eq!(adler32(b"abc"), 0x024d_0127);
        // Hand-computed: s1 = 1+sum(bytes) = 920 = 0x398;
        // s2 = sum of running s1 values = 4582 = 0x11E6.
        assert_eq!(adler32(b"Wikipedia"), 0x11e6_0398);
    }

    #[test]
    fn header_modulo_31_is_zero_and_cm_is_deflate() {
        let [cmf, flg] = header_bytes();
        assert_eq!(cmf & 0x0f, 8, "CM must be deflate");
        assert_eq!(cmf >> 4, 7, "CINFO must be the 32 KiB window");
        assert_eq!(flg & 0x20, 0, "FDICT must be clear (we never preset one)");
        let cmfflg = ((cmf as u16) << 8) | flg as u16;
        assert_eq!(cmfflg % 31, 0, "RFC 1950 FCHECK");
    }

    #[test]
    fn chunked_adler32_combine_equals_scalar_over_whole_input() {
        // Deterministic pseudo-random buffer, then folded combines over
        // assorted chunk grids — including the 5550 overflow boundary and a
        // 0-length trailing chunk — must equal the scalar whole-input pass.
        let mut data = Vec::with_capacity(1 << 18);
        let mut x: u64 = 0x243f_6a88_85a3_08d3;
        while data.len() < 256 * 1024 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            data.push((x >> 33) as u8);
        }
        let whole = adler32(&data);
        for &chunk_size in &[1usize, 2, 7, 100, 5549, 5550, 5551, 65536, 200_000] {
            let mut combined = 1u32; // adler32 of the empty prefix (s1 = 1)
            let mut rest = data.as_slice();
            while !rest.is_empty() {
                let take = rest.len().min(chunk_size);
                let chunk_adler = adler32(&rest[..take]);
                combined = adler32_combine(combined, chunk_adler, take);
                rest = &rest[take..];
            }
            assert_eq!(
                combined, whole,
                "chunked combine diverged at chunk size {chunk_size}"
            );
        }
        // The empty tail must be a no-op.
        assert_eq!(adler32_combine(whole, 1, 0), whole);
    }

    #[test]
    fn is_zlib_stream_accepts_our_header_and_rejects_gzip_and_truncations() {
        // ≥ 6 bytes: header + at least one DEFLATE byte + trailer space.
        let ok = [header_bytes()[0], header_bytes()[1], 0x03, 0, 0, 0];
        assert!(is_zlib_stream(&ok));
        // gzip magic must not sniff as zlib.
        assert!(!is_zlib_stream(&[0x1f, 0x8b, 0x08, 0, 0, 0]));
        // Truncations fall out (and never panic).
        for n in 0..6 {
            assert!(!is_zlib_stream(&ok[..n]));
        }
        // An FDICT flag with a matching FCHECK is rejected (we cannot preset
        // a dictionary). FLG must satisfy (0x78*256 + FLG) % 31 == 0, and
        // with bit 5 (FDICT) set that is FLG = 32.
        let mut fdict = ok;
        fdict[1] = 32;
        assert!((0x7800u16 + 32).is_multiple_of(31));
        assert!(
            !is_zlib_stream(&fdict),
            "FDICT streams must not sniff as ours"
        );
        // An FCHECK-violating header is not zlib.
        assert!(!is_zlib_stream(&[0x78, 0x00, 0, 0, 0, 0]));
        // A CINFO above 7 (window > 32 KiB) is invalid per RFC 1950. FLG=28
        // would satisfy the FCHECK sum for CMF=0x88, so this proves the
        // CINFO check bites, not just the checksum.
        assert!(!is_zlib_stream(&[0x88, 0x1c, 0, 0, 0, 0]));
    }
}
