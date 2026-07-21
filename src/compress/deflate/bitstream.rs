//! Word-oriented DEFLATE bit writer.
//!
//! Port of the libdeflate output-bitstream machinery
//! (`vendor/libdeflate/lib/deflate_compress.c`, the `BITBUF_NBITS` /
//! `CAN_BUFFER` / `ADD_BITS` / `FLUSH_BITS` macros, ~:669-751).
//!
//! Bits are packed little-endian, LSB-first, exactly as DEFLATE requires: the
//! first bit of a codeword goes into the least-significant free bit position of
//! the accumulator. `flush_bits()` drains whole bytes out of the 64-bit
//! accumulator in one branchless `to_le_bytes` copy — the "flush a whole word"
//! fast path from `FLUSH_BITS()`.

/// Capacity of the bit accumulator, one less than the real width to avoid UB on
/// a `>> (bitcount & !7)` of a full-width value (`BITBUF_NBITS`).
pub const BITBUF_NBITS: u32 = 8 * std::mem::size_of::<u64>() as u32 - 1;

/// Can `n` more bits always be added after a flush (which can leave up to 7)?
/// Mirrors the `CAN_BUFFER(n)` macro.
#[inline]
pub const fn can_buffer(n: u32) -> bool {
    7 + n <= BITBUF_NBITS
}

/// A growable, little-endian DEFLATE bit sink backed by a `Vec<u8>`.
pub struct BitWriter {
    /// Bits not yet written to `out` (valid bits are the low `bitcount`).
    bitbuf: u64,
    /// Number of valid bits currently held in `bitbuf` (0..=BITBUF_NBITS).
    bitcount: u32,
    out: Vec<u8>,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    #[inline]
    pub fn new() -> Self {
        BitWriter {
            bitbuf: 0,
            bitcount: 0,
            out: Vec::new(),
        }
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        BitWriter {
            bitbuf: 0,
            bitcount: 0,
            out: Vec::with_capacity(cap),
        }
    }

    /// Adopt `out` as the (byte-aligned) sink and append the DEFLATE stream
    /// directly to it. The caller's existing bytes are preserved as the prefix;
    /// [`finish`](Self::finish) hands the same `Vec` back with the stream
    /// appended. This is the write-through constructor that lets the encoder emit
    /// straight into the caller's output buffer instead of building a second Vec
    /// and copying it over.
    #[inline]
    pub fn from_vec(out: Vec<u8>) -> Self {
        BitWriter {
            bitbuf: 0,
            bitcount: 0,
            out,
        }
    }

    /// Number of bits currently buffered (not yet flushed to bytes).
    #[inline]
    pub fn buffered_bits(&self) -> u32 {
        self.bitcount
    }

    /// Add the low `n` bits of `val` to the stream.
    ///
    /// `add_bits`/`ADD_BITS` in libdeflate requires the caller to flush often
    /// enough that `bitcount + n <= BITBUF_NBITS`. We keep the same LSB-first
    /// packing but auto-flush first whenever the accumulator could not
    /// otherwise hold `n` more bits, so the primitive is safe for any `n <= 56`.
    #[inline]
    pub fn add_bits(&mut self, val: u64, n: u32) {
        debug_assert!(n <= 57, "add_bits n={n} too large for a single word");
        debug_assert!(
            n == 64 || val < (1u64 << n),
            "add_bits value has bits above n"
        );
        if self.bitcount + n > BITBUF_NBITS {
            self.flush_bits();
        }
        self.bitbuf |= val << self.bitcount;
        self.bitcount += n;
    }

    /// `ADD_BITS` — pure accumulate with no flush and no bounds check.
    ///
    /// This is the libdeflate hot-path primitive (C:717-722): the caller is
    /// responsible for having flushed recently enough that `bitcount + n <=
    /// BITBUF_NBITS`. Used by the block-body emitter, which interleaves these
    /// with [`flush_word_unchecked`](Self::flush_word_unchecked) at the exact
    /// points the vendor's `WRITE_MATCH` / 4-literal loop flush.
    #[inline]
    pub fn add_bits_raw(&mut self, val: u64, n: u32) {
        debug_assert!(
            self.bitcount + n <= BITBUF_NBITS,
            "add_bits_raw overflow: bitcount={} n={n}",
            self.bitcount
        );
        debug_assert!(
            n == 0 || val < (1u64 << n),
            "add_bits_raw value has bits above n"
        );
        self.bitbuf |= val << self.bitcount;
        self.bitcount += n;
    }

    /// Reserve space for at least `additional` more output bytes.
    ///
    /// Callers that then use [`flush_word_unchecked`](Self::flush_word_unchecked)
    /// must reserve enough that every flush in the batch has 8 spare bytes.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.out.reserve(additional);
    }

    /// Branchless whole-word `FLUSH_BITS` (C:734-751), assuming buffer room.
    ///
    /// Emits the complete bytes of the accumulator with a SINGLE unaligned
    /// 64-bit store (not a per-byte loop nor a variable-length `extend`), then
    /// keeps only the partial trailing byte. After this at most 7 bits remain.
    ///
    /// # Safety
    /// The output buffer must have at least 8 bytes of spare capacity
    /// (`self.out.capacity() - self.out.len() >= 8`). The batch emitter guards
    /// this by [`reserve`](Self::reserve)-ing `6 * tokens + slack` up front (a
    /// token codes to at most 47 bits < 6 bytes), which bounds the total bytes
    /// written so every intermediate flush stays 8 bytes inside capacity.
    #[inline]
    pub unsafe fn flush_word_unchecked(&mut self) {
        let len = self.out.len();
        debug_assert!(
            self.out.capacity() - len >= 8,
            "flush_word_unchecked without 8 spare bytes (len={len} cap={})",
            self.out.capacity()
        );
        // SAFETY: caller guarantees capacity >= len + 8, so the 8-byte
        // unaligned store lands entirely inside the allocation. We store the
        // native-order representation of `bitbuf.to_le()`, i.e. the little-endian
        // byte sequence DEFLATE requires, on both endiannesses.
        let dst = self.out.as_mut_ptr().add(len) as *mut u64;
        dst.write_unaligned(self.bitbuf.to_le());
        let nbytes = (self.bitcount >> 3) as usize;
        // SAFETY: nbytes <= 7 (bitcount <= 63) and len + nbytes <= len + 8 <= cap;
        // those bytes were just initialized by the store above.
        self.out.set_len(len + nbytes);
        // `bitcount & !7` is at most 56, so the shift is well-defined.
        self.bitbuf >>= self.bitcount & !7;
        self.bitcount &= 7;
    }

    /// Flush whole bytes from the accumulator into the output buffer.
    ///
    /// This is the branchless whole-word flush: write the eight little-endian
    /// bytes of `bitbuf`, keep only the `bitcount >> 3` that are complete, then
    /// shift the remaining partial byte (0..8 bits) down. After this, at most 7
    /// bits remain buffered.
    #[inline]
    pub fn flush_bits(&mut self) {
        let nbytes = (self.bitcount >> 3) as usize;
        let word = self.bitbuf.to_le_bytes();
        self.out.extend_from_slice(&word[..nbytes]);
        // `bitcount & !7` is at most 56, so the shift is always well-defined.
        self.bitbuf >>= self.bitcount & !7;
        self.bitcount &= 7;
    }

    /// Pad with zero bits up to the next byte boundary and flush.
    ///
    /// The pad bits are implicitly zero because only the low `bitcount` bits of
    /// `bitbuf` are ever set.
    #[inline]
    pub fn align_to_byte(&mut self) {
        let pad = (8 - (self.bitcount & 7)) & 7;
        self.bitcount += pad;
        self.flush_bits();
        debug_assert_eq!(self.bitcount, 0);
    }

    /// Append raw bytes. The stream MUST already be byte-aligned (call
    /// [`align_to_byte`](Self::align_to_byte) first). Used by stored blocks.
    #[inline]
    pub fn write_aligned_bytes(&mut self, bytes: &[u8]) {
        debug_assert_eq!(self.bitcount, 0, "write_aligned_bytes on unaligned stream");
        self.out.extend_from_slice(bytes);
    }

    /// Append a little-endian `u16` on a byte-aligned stream.
    #[inline]
    pub fn write_u16_le(&mut self, v: u16) {
        debug_assert_eq!(self.bitcount, 0, "write_u16_le on unaligned stream");
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    /// Finish the stream: flush full bytes, emit any final partial byte
    /// (zero-padded), and return the accumulated output.
    pub fn finish(mut self) -> Vec<u8> {
        self.flush_bits();
        if self.bitcount > 0 {
            debug_assert!(self.bitcount < 8);
            self.out.push(self.bitbuf as u8);
            self.bitbuf = 0;
            self.bitcount = 0;
        }
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_buffer_matches_definition() {
        assert!(can_buffer(56));
        assert!(!can_buffer(57));
        assert_eq!(BITBUF_NBITS, 63);
    }

    #[test]
    fn single_byte_lsb_first() {
        // Writing 0b1 then 0b0 then 0b1 (three bits) => byte 0b101 = 0x05.
        let mut w = BitWriter::new();
        w.add_bits(1, 1);
        w.add_bits(0, 1);
        w.add_bits(1, 1);
        assert_eq!(w.finish(), vec![0x05]);
    }

    #[test]
    fn known_multibit_pattern() {
        // Add a 3-bit value 0b110 (=6, LSB first: bits 0,1,1) then 5-bit 0b10011
        // (=19). The byte packs low bits first: value 6 occupies bits 0..3,
        // value 19 occupies bits 3..8 => (19 << 3) | 6 = 152+6 = 0x9E.
        let mut w = BitWriter::new();
        w.add_bits(6, 3);
        w.add_bits(19, 5);
        assert_eq!(w.finish(), vec![0x9E]);
    }

    #[test]
    fn crosses_word_boundary() {
        // Write 100 single 1-bits. Result: 12 full 0xFF bytes and a final byte
        // with the low 4 bits set (100 = 12*8 + 4) => 0x0F.
        let mut w = BitWriter::new();
        for _ in 0..100 {
            w.add_bits(1, 1);
        }
        let out = w.finish();
        assert_eq!(out.len(), 13);
        assert_eq!(&out[..12], &[0xFFu8; 12]);
        assert_eq!(out[12], 0x0F);
    }

    #[test]
    fn align_then_raw_bytes() {
        let mut w = BitWriter::new();
        w.add_bits(0b101, 3); // 3 bits pending
        w.align_to_byte(); // pad to a full byte => 0b00000101 = 0x05
        w.write_aligned_bytes(&[0xDE, 0xAD]);
        assert_eq!(w.finish(), vec![0x05, 0xDE, 0xAD]);
    }

    #[test]
    fn large_add_bits_autoflushes() {
        // Fill near capacity, then keep adding: must never panic and must
        // reproduce a clean bit pattern.
        let mut w = BitWriter::new();
        for _ in 0..40 {
            w.add_bits(0xABCDE, 20); // 20-bit chunks
        }
        let out = w.finish();
        // 40 * 20 = 800 bits = 100 bytes.
        assert_eq!(out.len(), 100);
        // Re-read the first 20 bits back out and confirm they equal 0xABCDE.
        let mut acc: u64 = 0;
        for (i, b) in out.iter().take(3).enumerate() {
            acc |= (*b as u64) << (8 * i);
        }
        assert_eq!(acc & 0xFFFFF, 0xABCDE);
    }
}
