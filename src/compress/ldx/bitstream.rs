//! C: `vendor/libdeflate/lib/deflate_compress.c:667-750` — the output bitstream.
//!
//! # Why the state is copied into locals and written back at the end
//!
//! The C's comment on `deflate_flush_block` explains it: "It is hard to get compilers
//! to understand that writes to `os->next` don't alias `os`. That hurts performance
//! significantly, as everything in `os` would keep getting re-loaded. (`restrict`
//! *should* do the trick, but it's unreliable.) Therefore, we keep all the output
//! bitstream state in local variables, and output bits using macros."
//!
//! Rust's aliasing rules make this provable rather than hopeful — `&mut [u8]` cannot
//! alias the `&mut DeflateOutputBitstream` that owns it — but the SHAPE is kept
//! anyway. Codegen is half the goal here ("performing exactly the same"), and a
//! structurally different emitter is a structurally different inner loop.
//!
//! # `next`/`end` are indices, not pointers
//!
//! The C carries `u8 *next` and `u8 * const end`. We carry a `&mut [u8]` plus a
//! `next` index, so `end` is `buf.len()`. Every pointer comparison in the C
//! (`out_next < out_fast_end`, `os->end - out_next`) becomes the corresponding index
//! comparison. This is not a semantic change: the C never forms a pointer outside
//! `[buf, buf+len]`, which is exactly the range an index can address.

/// C: `typedef machine_word_t bitbuf_t` (:670)
///
/// The type for the bitbuffer variable, which temporarily holds bits that are being
/// packed into bytes and written to the output buffer. For best performance, this
/// should have size equal to a machine word.
///
/// # The u128 accumulator (MEASUREMENT-ONLY, `ladder-tune`)
///
/// Under `ladder-tune` the accumulator widens to `u128` — two 64-bit registers
/// (aarch64 `ldp`/`stp` pairs) or `shld`/`shrd` pairs (x86) — lifting
/// [`BITBUF_NBITS`] 63→127. This is NOT the falsified 2026-07-22 batch-widening:
/// that experiment tried to widen a flush batch **within** the 64-bit buffer and
/// failed because the bound is set by a block's WORST single symbol
/// (`anatomy_counters.rs` `emit_body_bits`/`bitstream_flush_word_calls` note).
/// A wider accumulator is a different lever the same counters kept open: every
/// worst-case flush group the C's `CAN_BUFFER` glue branches on must fit the
/// widened form or it is not compile-time-true, and the only group the 64-bit
/// buffer pinned at its ceiling was the 4-literal batch (4×14 + 7 = 63).
///
/// Byte identity is by CONSTRUCTION, not by check: the buffer holds an LSB-first
/// bit string; every flush advances `bitcount >> 3` whole bytes off its low end
/// and keeps `bitcount & 7` bits, so the sequence of bytes that reaches the
/// output is independent of WHERE between symbols flushes are taken. Shipped
/// builds keep the C's `machine_word_t` bit for bit — the feature is a
/// measurement build (docs/board/sprint-2026-09-25.md, the cadence card whose
/// +18.8% wall quote this lever is priced against).
#[cfg(not(feature = "ladder-tune"))]
pub(crate) type BitbufT = usize;
#[cfg(feature = "ladder-tune")]
pub(crate) type BitbufT = u128;

/// C: `#define WORDBYTES sizeof(machine_word_t)`
pub(crate) const WORDBYTES: usize = core::mem::size_of::<BitbufT>();

/// C: `#define BITBUF_NBITS (8 * sizeof(bitbuf_t) - 1)` (:677)
///
/// The capacity of the bitbuffer, in bits. This is **1 less** than the real size, in
/// order to avoid undefined behavior when doing `bitbuf >>= bitcount & ~7`.
pub(crate) const BITBUF_NBITS: u32 = (8 * WORDBYTES - 1) as u32;

/// C: `#define CAN_BUFFER(n) (7 + (n) <= BITBUF_NBITS)` (:683)
///
/// Can the specified number of bits always be added to `bitbuf` after any pending
/// bytes have been flushed? There can be up to 7 bits remaining after a flush, so the
/// count must not exceed `BITBUF_NBITS` after adding `n` more bits.
pub(crate) const fn can_buffer(n: u32) -> bool {
    7 + n <= BITBUF_NBITS
}

/// C: `struct deflate_output_bitstream` (:689)
///
/// Structure to keep track of the current state of sending bits to the compressed
/// output buffer.
pub(crate) struct DeflateOutputBitstream<'a> {
    /// Bits that haven't yet been written to the output buffer.
    pub(crate) bitbuf: BitbufT,

    /// Number of bits currently held in `bitbuf`. This can be between 0 and
    /// `BITBUF_NBITS` in general, or between 0 and 7 after a flush.
    pub(crate) bitcount: u32,

    /// C: `u8 *next` — index in `buf` at which the next byte should be written.
    pub(crate) next: usize,

    /// The output buffer. C: `next` and `end` point into this; `end` is `buf.len()`.
    pub(crate) buf: &'a mut [u8],

    /// True if the output buffer ran out of space.
    pub(crate) overflow: bool,
}

impl<'a> DeflateOutputBitstream<'a> {
    pub(crate) fn new(buf: &'a mut [u8]) -> Self {
        Self {
            bitbuf: 0,
            bitcount: 0,
            next: 0,
            buf,
            overflow: false,
        }
    }

    /// C: `os->end` — one past the last writable byte.
    #[inline(always)]
    pub(crate) fn end(&self) -> usize {
        self.buf.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `CAN_BUFFER` answers that `deflate_flush_block` branches on. On a 64-bit
    /// target every one of them is true, which is why the C's fast paths are taken;
    /// pinning them documents WHICH shape we are actually compiling. The
    /// `ladder-tune` u128 accumulator compiles the SAME test points on the widened
    /// buffer, where every worst-case group the glue branches on fits outright.
    #[test]
    fn can_buffer_answers_on_this_target() {
        if WORDBYTES == 16 {
            // The u128 lever: everything the 64-bit build pinned as its exact
            // ceiling is now true with room to spare, flush frequencies fall
            // only where flush.rs re-groups against the new const bound.
            assert_eq!(BITBUF_NBITS, 127);
            // A litlen codeword (max 14) plus its extra length bits (max 5).
            assert!(can_buffer(14 + 5));
            // A whole match: litlen + extra len + offset codeword + extra offset.
            assert!(can_buffer(14 + 5 + 15 + 13));
            // Four literals — one bit AT the 63-bit ceiling before, loose now.
            assert!(can_buffer(4 * 14));
            // The batch the within-64-bit falsification could not admit.
            assert!(can_buffer(8 * 14));
            // 18 of the 19 precode lengths merged with the preceding header
            // fields, and even the 19th — false on the 64-bit buffer.
            assert!(can_buffer(3 * 18));
            assert!(can_buffer(3 * 19));
            // The C's formula admits nothing past `BITBUF_NBITS - 7` bits.
            assert!(can_buffer(127 - 7));
            assert!(!can_buffer(127 - 6));
        } else if WORDBYTES == 8 {
            assert_eq!(BITBUF_NBITS, 63);
            // A litlen codeword (max 14) plus its extra length bits (max 5).
            assert!(can_buffer(14 + 5));
            // A whole match: litlen + extra len + offset codeword + extra offset.
            assert!(can_buffer(14 + 5 + 15 + 13));
            // Four literals at a time — exactly at the limit, 7 + 56 == 63.
            assert!(can_buffer(4 * 14));
            assert!(!can_buffer(4 * 14 + 1));
            // 18 of the 19 precode lengths merged with the preceding header fields.
            assert!(can_buffer(3 * 18));
            assert!(!can_buffer(3 * 19));
        } else {
            assert_eq!(WORDBYTES, 4);
            assert_eq!(BITBUF_NBITS, 31);
            assert!(can_buffer(14 + 5));
            assert!(!can_buffer(4 * 14));
        }
    }
}
