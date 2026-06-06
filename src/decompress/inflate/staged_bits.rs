//! 128 KiB input staging for `ResumableInflate2`.
//!
//! Vendor `IsalInflateWrapper::refillBuffer` (`gzip/isal.hpp:228-250`) copies
//! at most 128 KiB from the `BitReader` into `m_buffer` before each
//! `isal_inflate` call. Decoding multi-MiB mmap'd input without chunking is
//! ~4× slower per the vendor comment at `isal.hpp:205-206`.

use std::io::{Error, ErrorKind, Result};

use super::consume_first_decode::Bits;

/// Vendor `m_buffer` capacity (`gzip/isal.hpp:207`).
pub const INPUT_STAGING_BYTES: usize = 128 * 1024;

/// Bit reader backed by a fixed 128 KiB staging window into a larger input.
pub struct StagedBitInput<'a> {
    full: &'a [u8],
    until_bits: usize,
    // Heap-backed: 128 KiB on the worker stack regressed x86 CLI decode
    // (BTYPE=11 / InvalidLookback) while lib tests passed — see
    // test_silesia_parallel_sm_mmap_fd_cli_shape vs gzippy binary.
    buf: Box<[u8; INPUT_STAGING_BYTES]>,
    buf_len: usize,
    /// Absolute byte offset in `full` where `buf[0]` lives.
    buf_base_byte: usize,
    pub(crate) inner: Bits<'a>,
}

impl<'a> StagedBitInput<'a> {
    pub fn with_until_bits(full: &'a [u8], bit_offset: usize, until_bits: usize) -> Result<Self> {
        let until_bits = until_bits.min(full.len().saturating_mul(8));
        if bit_offset > until_bits {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "bit_offset past end of input",
            ));
        }
        let mut s = Self {
            full,
            until_bits,
            buf: Box::new([0; INPUT_STAGING_BYTES]),
            buf_len: 0,
            buf_base_byte: 0,
            inner: Bits::new(&[]),
        };
        s.reload_at_bit(bit_offset)?;
        Ok(s)
    }

    #[inline]
    pub fn full_input(&self) -> &'a [u8] {
        self.full
    }

    #[inline]
    #[allow(dead_code)] // vendor parity / cap widening
    pub fn until_bits(&self) -> usize {
        self.until_bits
    }

    /// Widen the logical input cap (e.g. after `advance_input` skips a gzip
    /// header). Reloads staging when the new cap exposes bytes beyond the
    /// current window.
    pub fn set_until_bits(&mut self, until_bits: usize) {
        let until_bits = until_bits.min(self.full.len().saturating_mul(8));
        if until_bits <= self.until_bits {
            return;
        }
        self.until_bits = until_bits;
        if self.buf_base_byte.saturating_add(self.buf_len) < self.until_byte() {
            let _ = self.reload_at_bit(self.bit_position());
        }
    }

    /// Absolute bit position of the next bit to consume.
    #[inline]
    pub fn bit_position(&self) -> usize {
        let unread = (self.inner.bitsleft as u8).min(64) as isize;
        let rel = (self.inner.pos as isize) * 8 - unread;
        let abs = (self.buf_base_byte as isize) * 8 + rel;
        abs.max(0) as usize
    }

    #[inline]
    pub fn available(&self) -> u32 {
        self.inner.available()
    }

    #[inline]
    pub fn bitbuf(&self) -> u64 {
        self.inner.bitbuf
    }

    #[inline]
    #[allow(dead_code)] // staged_bits unit tests
    pub fn bitsleft(&self) -> u32 {
        self.inner.bitsleft
    }

    #[inline]
    #[allow(dead_code)] // staged_bits unit tests
    pub fn pos(&self) -> usize {
        self.inner.pos
    }

    pub fn refill(&mut self) {
        if self.inner.pos < self.buf_len {
            self.inner.refill();
        }
        if self.buf_base_byte.saturating_add(self.buf_len) >= self.until_byte() {
            return;
        }
        if self.inner.pos < self.buf_len {
            return;
        }
        let _ = self.advance_staging_window();
    }

    pub fn consume(&mut self, n: u32) {
        self.inner.consume(n);
    }

    #[allow(dead_code)] // vendor Bits parity
    pub fn peek(&self) -> u64 {
        self.inner.peek()
    }

    pub fn read_u16(&mut self) -> u16 {
        self.inner.read_u16()
    }

    pub fn align_to_byte(&mut self) {
        self.inner.align_to_byte();
    }

    /// Seek to an absolute byte offset in `full` (clears bit buffer).
    pub fn seek_abs_byte(&mut self, byte: usize) {
        let bit = byte.saturating_mul(8).min(self.until_bits);
        let _ = self.reload_at_bit(bit);
    }

    /// Rebuild staging from the full-input oracle at `bit_pos`. Used when the
    /// Huffman fastloop mirrors `Bits` on the mmap'd `full` slice and must
    /// push absolute progress back into `StagedBitInput` for block-header
    /// entry (`try_enter_next_block`).
    pub(crate) fn sync_at_absolute_bit(&mut self, bit_pos: usize) {
        let _ = self.reload_at_bit(bit_pos);
    }

    /// Reload staging once every byte in the current chunk is in `inner.pos`
    /// (vendor: `avail_in == 0`). Returns `false` when no forward progress is
    /// possible (EOF under cap).
    #[allow(dead_code)] // exercised by unit tests; production uses `refill()`
    pub fn reload_if_exhausted(&mut self, in_pos: usize) -> bool {
        if in_pos < self.buf_len {
            return true;
        }
        debug_assert!(
            in_pos >= self.buf_len,
            "reload_if_exhausted called before staging bytes consumed"
        );
        let prev_base = self.buf_base_byte;
        if self.buf_base_byte.saturating_add(self.buf_len) >= self.until_byte() {
            return false;
        }
        if self.advance_staging_window().is_err() {
            return false;
        }
        self.buf_base_byte != prev_base
    }

    /// Slide when the bit accumulator still holds unread bits (vendor
    /// `refillBuffer`); full oracle reload otherwise (block headers / seeks).
    fn advance_staging_window(&mut self) -> Result<()> {
        let bl = (self.inner.bitsleft as u8).min(64);
        if bl > 0 && bl < 64 {
            self.slide_staging_window()?;
            if bl < 56 {
                self.inner.refill();
            }
        } else {
            self.reload_at_bit(self.bit_position())?;
        }
        Ok(())
    }

    /// Advance to the next staging window at `buf_base + buf_len`, preserving
    /// `bitbuf`/`bitsleft` (vendor `refillBuffer` after `avail_in==0`).
    fn slide_staging_window(&mut self) -> Result<()> {
        let until_byte = self.until_byte();
        let next_base = self.buf_base_byte.saturating_add(self.buf_len);
        if next_base >= until_byte {
            return Ok(());
        }
        let max_bytes = (until_byte - next_base).min(INPUT_STAGING_BYTES);
        self.buf[..max_bytes].copy_from_slice(&self.full[next_base..next_base + max_bytes]);
        if max_bytes < INPUT_STAGING_BYTES {
            self.buf[max_bytes..].fill(0);
        }
        self.buf_len = max_bytes;
        self.buf_base_byte = next_base;
        let slice: &'a [u8] = unsafe { std::mem::transmute(&self.buf[..self.buf_len]) };
        self.inner.data = slice;
        self.inner.pos = 0;
        Ok(())
    }

    fn until_byte(&self) -> usize {
        self.until_bits.div_ceil(8).min(self.full.len())
    }

    /// Port of vendor `refillBuffer` byte-load path (`isal.hpp:228-250`).
    fn reload_at_bit(&mut self, bit_pos: usize) -> Result<()> {
        let until_byte = self.until_byte();
        let start_byte = (bit_pos / 8).min(until_byte);

        if start_byte >= until_byte {
            self.buf_len = 0;
            self.buf_base_byte = start_byte;
            self.inner = Bits::at_bit_offset(&[], 0);
            return Ok(());
        }

        // Reconstruct reader state from the full input oracle so reload is
        // bit-identical to direct `Bits` at the same absolute position.
        let oracle = Bits::at_bit_offset(self.full, bit_pos);
        let rel_pos = oracle.pos.saturating_sub(start_byte);

        // Skip the 128 KiB memcpy when the window is already anchored at
        // `start_byte` and still covers the oracle byte position (+8 for
        // branchless refill).
        if start_byte == self.buf_base_byte
            && self.buf_len > 0
            && rel_pos.saturating_add(8) <= self.buf_len
        {
            let slice: &'a [u8] = unsafe { std::mem::transmute(&self.buf[..self.buf_len]) };
            self.inner = Bits {
                data: slice,
                pos: rel_pos,
                bitbuf: oracle.bitbuf,
                bitsleft: oracle.bitsleft,
            };
            return Ok(());
        }

        let max_bytes = (until_byte - start_byte).min(INPUT_STAGING_BYTES);
        self.buf[..max_bytes].copy_from_slice(&self.full[start_byte..start_byte + max_bytes]);
        // Pad tail: branchless `Bits::refill` may read up to 7 bytes past the
        // last logical input byte when `pos` is near `buf_len`.
        if max_bytes < INPUT_STAGING_BYTES {
            self.buf[max_bytes..].fill(0);
        }
        self.buf_len = max_bytes;
        self.buf_base_byte = start_byte;

        if rel_pos > self.buf_len {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "staging reload window does not cover oracle byte position",
            ));
        }

        // SAFETY: `inner` only reads `buf` while we own it; reload reconstructs
        // `inner` before any caller can hold an aliasing `&[u8]`.
        let slice: &'a [u8] = unsafe { std::mem::transmute(&self.buf[..self.buf_len]) };
        self.inner = Bits {
            data: slice,
            pos: rel_pos,
            bitbuf: oracle.bitbuf,
            bitsleft: oracle.bitsleft,
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::DeflateEncoder;
    use flate2::Compression;
    use std::io::Write;

    fn make_deflate(payload: &[u8]) -> Vec<u8> {
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::new(6));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    fn make_stored_deflate(payload: &[u8]) -> Vec<u8> {
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::new(0));
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn staged_reader_tracks_direct_bits_through_refills() {
        let payload = b"hello world hello world".repeat(500);
        let deflate = make_deflate(&payload);
        for &bit_off in &[0usize, 17, 64] {
            let mut staged =
                StagedBitInput::with_until_bits(&deflate, bit_off, deflate.len() * 8).unwrap();
            let mut direct = Bits::at_bit_offset(&deflate, bit_off);
            for _ in 0..10_000 {
                if staged.bit_position() >= deflate.len() * 8 {
                    break;
                }
                assert_eq!(
                    staged.bit_position(),
                    direct.bit_position(),
                    "bit_position drift at offset {bit_off}"
                );
                assert_eq!(staged.bitbuf(), direct.bitbuf, "bitbuf drift");
                assert_eq!(staged.bitsleft(), direct.bitsleft, "bitsleft drift");
                if staged.available() >= 5 {
                    staged.consume(5);
                    direct.consume(5);
                } else {
                    staged.refill();
                    direct.refill();
                }
            }
        }
    }

    #[test]
    fn bit_position_matches_bits_at_offset() {
        let payload = vec![0xABu8; 300_000];
        let deflate = make_deflate(&payload);
        for bit_offset in [0usize, 17, 64, 8000] {
            if bit_offset > deflate.len() * 8 {
                continue;
            }
            let direct = Bits::at_bit_offset(&deflate, bit_offset);
            let staged =
                StagedBitInput::with_until_bits(&deflate, bit_offset, deflate.len() * 8).unwrap();
            assert_eq!(
                staged.bit_position(),
                bit_offset,
                "initial position mismatch at offset {bit_offset}"
            );
            assert_eq!(
                staged.bit_position(),
                direct.bit_position(),
                "vs direct Bits at offset {bit_offset}"
            );
        }
    }

    #[test]
    fn staged_matches_direct_on_fixed_block_fixture() {
        let payload = b"hello world hello world hello world";
        let deflate = make_deflate(payload);
        let mut direct = Bits::at_bit_offset(&deflate, 0);
        let mut staged = StagedBitInput::with_until_bits(&deflate, 0, deflate.len() * 8).unwrap();
        for _ in 0..50_000 {
            if direct.bit_position() >= deflate.len() * 8 {
                break;
            }
            assert_eq!(staged.bit_position(), direct.bit_position());
            assert_eq!(staged.bitbuf(), direct.bitbuf);
            assert_eq!(staged.bitsleft(), direct.bitsleft);
            if direct.available() >= 10 {
                direct.consume(10);
                staged.consume(10);
            } else {
                direct.refill();
                staged.refill();
            }
        }
        assert_eq!(staged.bit_position(), direct.bit_position());
    }

    #[test]
    fn silesia_staged_reader_tracks_direct_to_eof() {
        let large = std::path::Path::new("benchmark_data/silesia-large.gz");
        let tar = std::path::Path::new("benchmark_data/silesia-gzip.tar.gz");
        let path = if large.exists() {
            large
        } else if tar.exists() {
            tar
        } else {
            eprintln!("skip: silesia fixture missing");
            return;
        };
        let gzip = std::fs::read(path).unwrap();
        let header = crate::decompress::format::parse_gzip_header_size(&gzip).unwrap_or(10);
        let deflate = &gzip[header..gzip.len() - 8];
        let until = deflate.len() * 8;
        let mut staged = StagedBitInput::with_until_bits(deflate, 0, until).unwrap();
        let mut direct = Bits::at_bit_offset(deflate, 0);
        let mut steps = 0usize;
        while staged.bit_position() < until && steps < 50_000_000 {
            assert_eq!(
                staged.bit_position(),
                direct.bit_position(),
                "drift at step {steps}"
            );
            assert_eq!(
                staged.bitbuf(),
                direct.bitbuf,
                "bitbuf drift at step {steps}"
            );
            assert_eq!(
                staged.bitsleft(),
                direct.bitsleft,
                "bitsleft drift at step {steps}"
            );
            if staged.available() >= 10 {
                direct.consume(10);
                staged.consume(10);
            } else {
                direct.refill();
                staged.refill();
            }
            steps += 1;
        }
        assert_eq!(staged.bit_position(), direct.bit_position());
    }

    #[test]
    fn reload_at_bit_skips_memcpy_when_anchor_unchanged() {
        let payload = vec![0xABu8; 300_000];
        let deflate = make_stored_deflate(&payload);
        let until = deflate.len() * 8;
        for &bit_pos in &[0usize, 17, 8000, 64_000] {
            if bit_pos >= until {
                continue;
            }
            let mut staged = StagedBitInput::with_until_bits(&deflate, bit_pos, until).unwrap();
            let oracle = Bits::at_bit_offset(&deflate, bit_pos);
            let buf_snapshot = *staged.buf;
            staged.reload_at_bit(bit_pos).unwrap();
            assert_eq!(*staged.buf, buf_snapshot, "memcpy at bit_pos {bit_pos}");
            assert_eq!(staged.bit_position(), oracle.bit_position());
            assert_eq!(staged.bitbuf(), oracle.bitbuf);
        }
    }

    #[test]
    fn reload_at_bit_matches_full_oracle_through_many_reloads() {
        let payload = vec![0xABu8; 300_000];
        let deflate = make_stored_deflate(&payload);
        let until = deflate.len() * 8;
        let mut staged = StagedBitInput::with_until_bits(&deflate, 0, until).unwrap();
        let mut direct = Bits::at_bit_offset(&deflate, 0);
        for step in 0..200_000 {
            if direct.bit_position() >= until {
                break;
            }
            let bit_pos = direct.bit_position();
            let oracle = Bits::at_bit_offset(&deflate, bit_pos);
            staged.reload_at_bit(bit_pos).unwrap();
            assert_eq!(
                staged.bit_position(),
                oracle.bit_position(),
                "bit_position drift at step {step}"
            );
            assert_eq!(
                staged.bitbuf(),
                oracle.bitbuf,
                "bitbuf drift at step {step}"
            );
            assert_eq!(
                staged.bitsleft(),
                oracle.bitsleft,
                "bitsleft drift at step {step}"
            );
            if direct.available() >= 5 {
                direct.consume(5);
            } else {
                direct.refill();
            }
        }
    }

    #[test]
    fn reload_advances_across_staging_boundary() {
        // Incompressible payload so compressed size exceeds 2× staging.
        let payload = vec![0xABu8; 300_000];
        let deflate = make_stored_deflate(&payload);
        assert!(
            deflate.len() > INPUT_STAGING_BYTES + 4096,
            "fixture must span multiple staging windows"
        );
        let mut staged = StagedBitInput::with_until_bits(&deflate, 0, deflate.len() * 8).unwrap();
        assert!(staged.buf_len <= INPUT_STAGING_BYTES);
        let jump_byte = INPUT_STAGING_BYTES - 64;
        let jump_bit = jump_byte * 8;
        staged.reload_at_bit(jump_bit).unwrap();
        assert_eq!(staged.bit_position(), jump_bit);
        assert!(staged.reload_if_exhausted(staged.pos()));
        assert!(staged.buf_base_byte >= jump_byte);
    }
}
