#![allow(dead_code)] // vendor-faithful rapidgzip port; many items are pending consumer-port

//! Fast LSB-first DEFLATE bit reader — port of rapidgzip's `core/BitReader.hpp`.
//!
//! Shared across the parallel modules (block-boundary validation, the Huffman
//! decode tables, and the candidate finder). Previously this struct lived inside
//! `blockfinder_validation.rs` (now [`super::blockfinder_validation`]); it was moved here
//! so the bit-primitive no longer hides inside a validation module and the rg
//! `core/BitReader.hpp` correspondence is explicit.

/// Misaligned, single-load DEFLATE bit reader (rg `core/BitReader.hpp`).
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_buf: u64,
    bits_available: u8,
}

impl<'a> BitReader<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        let mut reader = Self {
            data,
            byte_pos: 0,
            bit_buf: 0,
            bits_available: 0,
        };
        reader.refill();
        reader
    }

    #[inline]
    pub fn seek_to_bit(&mut self, bit_offset: usize) {
        self.byte_pos = bit_offset / 8;
        self.bit_buf = 0;
        self.bits_available = 0;
        self.refill();
        let skip = (bit_offset % 8) as u8;
        if skip > 0 {
            self.bit_buf >>= skip;
            self.bits_available = self.bits_available.saturating_sub(skip);
        }
    }

    #[inline]
    fn refill(&mut self) {
        // Vendor-parity 8-byte misaligned refill. Previously this loop
        // OR'd one byte at a time, up to 7 iterations to refill a low
        // bit_buf. Vendor (`BitReader.hpp`) issues a single unaligned
        // 8-byte little-endian load and OR-shifts it in.
        //
        // The shift may discard the top `bits_available` bits of the
        // loaded value (they overflow the u64); we compensate by
        // advancing `byte_pos` by ONLY `(64 - bits_available) / 8`
        // bytes — the discarded high bits are re-read on the next refill
        // from the not-yet-advanced byte position.
        if self.bits_available <= 56 && self.byte_pos + 8 <= self.data.len() {
            // SAFETY: bounds-checked above; little-endian byte order
            // matches our bit-numbering (low bit of byte 0 → bit 0 of bit_buf).
            let next8: u64 = unsafe {
                core::ptr::read_unaligned(self.data.as_ptr().add(self.byte_pos) as *const u64)
            }
            .to_le();
            self.bit_buf |= next8 << self.bits_available;
            let bytes_consumed = (64 - self.bits_available as usize) / 8;
            self.bits_available += (bytes_consumed * 8) as u8;
            self.byte_pos += bytes_consumed;
        }
        // Tail: byte-by-byte for the last <8 bytes of input (and the
        // path taken when bits_available > 56 already, which is a no-op).
        while self.bits_available <= 56 && self.byte_pos < self.data.len() {
            self.bit_buf |= (self.data[self.byte_pos] as u64) << self.bits_available;
            self.bits_available += 8;
            self.byte_pos += 1;
        }
    }

    /// Returns true if at least `n` bits are available (refilling first).
    #[inline]
    pub fn can_read(&mut self, n: u8) -> bool {
        if self.bits_available < n {
            self.refill();
        }
        self.bits_available >= n
    }

    /// Peek up to `n` bits. Caller must ensure `bits_available >= n` —
    /// use `peek_refilled` for the safe variant.
    #[inline]
    pub fn peek(&self, n: u8) -> u64 {
        self.bit_buf & ((1u64 << n) - 1)
    }

    /// Refill if needed, then peek. Use for reads larger than the
    /// refill watermark (e.g. 57-bit precode reads).
    #[inline]
    pub fn peek_refilled(&mut self, n: u8) -> u64 {
        if self.bits_available < n {
            self.refill();
        }
        self.peek(n)
    }

    #[inline]
    pub fn skip(&mut self, n: u8) {
        self.bit_buf >>= n;
        self.bits_available = self.bits_available.saturating_sub(n);
        if self.bits_available < 32 {
            self.refill();
        }
    }

    /// Read up to `n` bits, refilling first if necessary. Safe for
    /// any n ≤ 57.
    #[inline]
    pub fn read(&mut self, n: u8) -> u64 {
        if self.bits_available < n {
            self.refill();
        }
        let val = self.peek(n);
        self.skip(n);
        val
    }

    #[inline]
    pub fn bit_position(&self) -> usize {
        self.byte_pos * 8 - self.bits_available as usize
    }

    #[inline]
    pub fn is_eof(&self) -> bool {
        self.byte_pos >= self.data.len() && self.bits_available == 0
    }
}
