//! DEFLATE encoder. Built across plan Steps 14 (`BitWriter`) and
//! 15 (tree emission, block emission, deflate driver).
//!
//! Port of the bit-emitting half of vendor/zopfli/src/zopfli/deflate.c.

#![allow(dead_code)]

/// Bit-level writer that appends to a byte buffer. The C code uses a
/// `(out, outsize, bp)` triple where `bp` is the bit pointer in `[0, 7]`;
/// when `bp == 0` a fresh zero byte is appended before any bits land.
///
/// First-port goal (Step 14): one bit per iteration, byte-identical to C.
/// Phase-10 optimisation (Step 27) replaces this with a 64-bit accumulator.
pub struct BitWriter<'a> {
    pub out: &'a mut Vec<u8>,
    pub bp: u8,
}

impl<'a> BitWriter<'a> {
    /// Creates a new writer wrapping `out`. `bp` is the bit pointer for
    /// the *current* tail byte: 0 means the next bit will allocate a new
    /// byte; 1..=7 means there is a half-filled tail byte already.
    pub fn new(out: &'a mut Vec<u8>, bp: u8) -> Self {
        debug_assert!(bp < 8);
        Self { out, bp }
    }

    /// Returns the bit pointer (modular position of the next bit in the
    /// last byte, in `[0, 7]`).
    #[inline]
    pub fn bp(&self) -> u8 {
        self.bp
    }

    /// Append a single bit (0 or 1).
    pub fn add_bit(&mut self, bit: u8) {
        debug_assert!(bit <= 1);
        if self.bp == 0 {
            self.out.push(0);
        }
        let n = self.out.len();
        self.out[n - 1] |= bit << self.bp;
        self.bp = (self.bp + 1) & 7;
    }

    /// Append `length` bits of `symbol`, LSB-first. Used for everything
    /// except canonical Huffman codes.
    pub fn add_bits(&mut self, symbol: u32, length: u32) {
        for i in 0..length {
            let bit = ((symbol >> i) & 1) as u8;
            if self.bp == 0 {
                self.out.push(0);
            }
            let n = self.out.len();
            self.out[n - 1] |= bit << self.bp;
            self.bp = (self.bp + 1) & 7;
        }
    }

    /// Append `length` bits of `symbol`, MSB-first. The DEFLATE spec
    /// requires this orientation for canonical Huffman codes.
    pub fn add_huffman_bits(&mut self, symbol: u32, length: u32) {
        for i in 0..length {
            let bit = ((symbol >> (length - i - 1)) & 1) as u8;
            if self.bp == 0 {
                self.out.push(0);
            }
            let n = self.out.len();
            self.out[n - 1] |= bit << self.bp;
            self.bp = (self.bp + 1) & 7;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a writer, run a closure, return the resulting (bytes, bp).
    fn run<F: FnOnce(&mut BitWriter<'_>)>(f: F) -> (Vec<u8>, u8) {
        let mut out = Vec::new();
        let mut w = BitWriter::new(&mut out, 0);
        f(&mut w);
        let bp = w.bp;
        (out, bp)
    }

    #[test]
    fn add_bit_packs_lsb_first() {
        let (bytes, bp) = run(|w| {
            // Write 1, 0, 1, 1 → byte = 0b0000_1101 = 0x0D, bp = 4.
            w.add_bit(1);
            w.add_bit(0);
            w.add_bit(1);
            w.add_bit(1);
        });
        assert_eq!(bytes, vec![0x0D]);
        assert_eq!(bp, 4);
    }

    #[test]
    fn add_bits_lsb_first_eight_bits_makes_one_byte() {
        let (bytes, bp) = run(|w| {
            // 0xA5 = 0b1010_0101 — written LSB-first as bits 1,0,1,0,0,1,0,1.
            w.add_bits(0xA5, 8);
        });
        assert_eq!(bytes, vec![0xA5]);
        // 8 bits modulo 8 == 0 → bp wraps back to 0.
        assert_eq!(bp, 0);
    }

    #[test]
    fn add_bits_spans_two_bytes() {
        let (bytes, bp) = run(|w| {
            // 12 bits of 0xABC = 0b1010_1011_1100, LSB-first.
            // First 8 bits land in byte 0: bits 0..8 of 0xABC = 0xBC.
            // Next 4 bits land in byte 1, low nibble: 0xA.
            w.add_bits(0xABC, 12);
        });
        assert_eq!(bytes, vec![0xBC, 0x0A]);
        assert_eq!(bp, 4);
    }

    #[test]
    fn add_huffman_bits_msb_first() {
        let (bytes, bp) = run(|w| {
            // Symbol 0b101, length 3. MSB-first → emit 1, 0, 1.
            // LSB-stuffed into byte 0 → bits 0..3 = 1,0,1 → 0b0000_0101 = 0x05.
            w.add_huffman_bits(0b101, 3);
        });
        assert_eq!(bytes, vec![0x05]);
        assert_eq!(bp, 3);
    }

    #[test]
    fn matches_c_addbits_addhuffmanbits_for_random_sequence() {
        // Mirror the C routines bit-for-bit using a literal port; build a
        // reference byte stream and assert our writer matches.
        fn ref_add_bits(out: &mut Vec<u8>, bp: &mut u8, symbol: u32, length: u32) {
            for i in 0..length {
                let bit = ((symbol >> i) & 1) as u8;
                if *bp == 0 {
                    out.push(0);
                }
                let n = out.len();
                out[n - 1] |= bit << *bp;
                *bp = (*bp + 1) & 7;
            }
        }
        fn ref_add_huffman(out: &mut Vec<u8>, bp: &mut u8, symbol: u32, length: u32) {
            for i in 0..length {
                let bit = ((symbol >> (length - i - 1)) & 1) as u8;
                if *bp == 0 {
                    out.push(0);
                }
                let n = out.len();
                out[n - 1] |= bit << *bp;
                *bp = (*bp + 1) & 7;
            }
        }

        // Deterministic LCG so the test is reproducible.
        let mut s: u32 = 0xDEADBEEF;
        let mut next = || -> u32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            s
        };

        let mut ref_out = Vec::new();
        let mut ref_bp: u8 = 0;
        let mut got_out = Vec::new();
        let got_bp;
        {
            let mut w = BitWriter::new(&mut got_out, 0);
            for _ in 0..200 {
                let r = next();
                let length = (r % 16) + 1; // 1..=16 bits
                let symbol = next() & ((1u32 << length) - 1);
                if r & 0x10 == 0 {
                    ref_add_bits(&mut ref_out, &mut ref_bp, symbol, length);
                    w.add_bits(symbol, length);
                } else {
                    ref_add_huffman(&mut ref_out, &mut ref_bp, symbol, length);
                    w.add_huffman_bits(symbol, length);
                }
            }
            got_bp = w.bp;
        }
        assert_eq!(got_out, ref_out);
        assert_eq!(got_bp, ref_bp);
    }
}
