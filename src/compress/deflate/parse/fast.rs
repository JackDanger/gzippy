//! Level-1 igzip-class one-pass FAST parser.
//!
//! The match finder is a port of igzip's level-0/1 deflate body
//! (`vendor/isa-l/igzip/igzip_base.c:isal_deflate_body_base`, :27-113): a
//! chainless single-probe hash with `LIMIT_HASH_UPDATE`, the igzip-class speed
//! lever. Where this path DIFFERS from the greedy/lazy/near-optimal parsers is
//! only in the finder — it does one probe per position, no chains, no depth
//! loop — NOT in how blocks are coded.
//!
//! For coding it reuses the SHARED L2-9 backend ([`Sink`] + [`emit_block`]):
//! tokens stream into the sequence buffer while litlen + offset frequency
//! histograms accumulate as-you-go (no extra pass), a block is flushed every
//! [`FAST_BLOCK_LENGTH`] input bytes (and at end-of-input), and each flush picks
//! the cheapest of a per-block DYNAMIC Huffman code, the RFC-1951 static code, or
//! a STORED block. The old fast path direct-emitted one whole-input static block:
//! fast, but 1.2-1.3x larger than `gzip -1` on some corpora, and it could not
//! escape to STORED on incompressible data. Per-block dynamic Huffman + the
//! stored escape recover competitive ratio while the single-probe finder keeps
//! the speed.
//!
//! The mechanisms ported from igzip (finder only):
//!   1. **Chainless single-probe hash** (`igzip_base.c:60-64`): one small head
//!      table storing the last position per hash; overwrite on collision; ONE
//!      candidate per position — no chains, no depth loop.
//!   2. **LIMIT_HASH_UPDATE** (`igzip_base.c:71-86`): over an accepted match,
//!      insert the hash for only the first ~3 interior positions, then jump the
//!      cursor by the whole match length (skip the interior stores).
//!   3. **compare258 match-extend** (`huffman.h:260-314`): 8-byte XOR +
//!      trailing-zero-count, reusing Increment 1's [`lz_extend`].
//!
//! Block emission (mechanisms 4-5: full-length-codeword LUT, 4-literals-per-
//! flush, branchless word-store FLUSH, cheapest-of-{dynamic,static,stored}) all
//! lives in the shared [`emit_block`]/`emit_tokens` machinery — this file does
//! not duplicate it.

use super::super::bitstream::BitWriter;
use super::super::matchfinder::common::{load_u32, lz_extend, lz_hash};
use super::super::tables::DEFLATE_MAX_MATCH_LEN;
use super::{emit_block, Sink, StaticCodes};

/// Log2 of the head-table size. igzip's level-0 hash table is
/// `IGZIP_LVL0_HASH_SIZE = 8 * 1024 = 1 << 13` (`igzip_lib.h:121-125`).
const HASH_BITS: u32 = 13;
const HASH_SIZE: usize = 1 << HASH_BITS;

/// Sentinel head-table entry meaning "no position stored yet". Any position we
/// store is `< in_end <= u32::MAX`, so the sentinel never collides with a real
/// index, and its computed distance always fails the window test.
const NO_POS: u32 = u32::MAX;

/// igzip `SHORTEST_MATCH` (`huff_codes.h:89`): the fast path only emits matches
/// of length >= 4 (its hash keys 4 bytes), coding anything shorter as literals.
const SHORTEST_MATCH: u32 = 4;

/// DEFLATE sliding-window size — the largest legal back-reference distance.
const WINDOW: usize = 32768;

/// Input bytes covered per DEFLATE block in the fast path.
///
/// A per-block dynamic Huffman code adapts to LOCAL statistics, so a moderate
/// block (vs one whole-input block) improves ratio on heterogeneous input; at
/// 64 KiB the ~dozens-of-bytes dynamic header amortizes to well under 1%. This
/// is the fast path's one ratio/speed tuning knob — it does not affect
/// correctness (any block boundary roundtrips).
const FAST_BLOCK_LENGTH: usize = 1 << 16;

/// Run the one-pass fast encoder over `buf[data_start..in_end]`, appending one
/// or more DEFLATE blocks to `bw`.
///
/// `buf` MUST carry at least [`super::BUF_PAD`] trailing pad bytes beyond
/// `in_end` (upheld by the caller in `deflate::compress_block`) so the
/// speculative 4-byte hash loads and 8-byte match-extend loads never read out of
/// bounds. `buf[..data_start]` is an optional preset dictionary: its positions
/// are seeded into the head table so matches may reference it, but it is not
/// coded.
pub(super) fn run(
    buf: &[u8],
    data_start: usize,
    in_end: usize,
    statics: &StaticCodes,
    bw: &mut BitWriter,
    is_last: bool,
) {
    debug_assert!(in_end > data_start, "empty data handled by the caller");
    debug_assert!(buf.len() >= in_end + super::BUF_PAD);

    // Chainless head table: one slot per hash, holding the most recent position.
    let mut head = vec![NO_POS; HASH_SIZE];
    let base = buf.as_ptr();

    // Seed the preset dictionary (positions < data_start) into the head table.
    // Each has >= 4 readable bytes because data follows the dict in `buf`.
    let mut p = 0usize;
    while p < data_start {
        // SAFETY: p < data_start <= in_end, and buf has BUF_PAD >= 16 bytes past
        // in_end, so [p, p+4) is in bounds.
        let seq = unsafe { load_u32(base, p) };
        head[lz_hash(seq, HASH_BITS) as usize] = p as u32;
        p += 1;
    }

    // Per-block accumulator: tokens + litlen/offset histograms built as-you-go.
    let mut sink = Sink::new();
    let mut pos = data_start;

    loop {
        // Start a new block. It ends after FAST_BLOCK_LENGTH input bytes (a match
        // straddling the boundary is allowed to overrun it slightly) or at EOF.
        let block_begin = pos;
        sink.begin();
        let block_end_target = (block_begin + FAST_BLOCK_LENGTH).min(in_end);

        while pos < in_end {
            let remaining = in_end - pos;
            let max_len = if remaining > DEFLATE_MAX_MATCH_LEN as usize {
                DEFLATE_MAX_MATCH_LEN
            } else {
                remaining as u32
            };

            if max_len >= SHORTEST_MATCH {
                // SAFETY: max_len >= 4 implies pos + 4 <= in_end, in bounds.
                let seq = unsafe { load_u32(base, pos) };
                let h = lz_hash(seq, HASH_BITS) as usize;
                let cand = head[h];
                head[h] = pos as u32;

                // `pos - cand`; a wrapping sub keeps a sentinel/stale entry out of
                // the window range instead of panicking on underflow.
                let dist = pos.wrapping_sub(cand as usize);
                if (1..=WINDOW).contains(&dist) {
                    let cand_pos = cand as usize;
                    // Byte-exact extend (never trusts the hash): a spurious
                    // candidate simply yields length < SHORTEST_MATCH -> literal.
                    let length = lz_extend(buf, pos, cand_pos, 0, max_len);
                    if length >= SHORTEST_MATCH {
                        // LIMIT_HASH_UPDATE: insert the hash for only positions
                        // pos+1, pos+2 (igzip's `end = next_hash + 3`), then jump
                        // the cursor over the whole match. length >= 4 guarantees
                        // these interior positions are inside the match.
                        let limit = pos + 3;
                        let mut nh = pos + 1;
                        while nh < limit {
                            // SAFETY: nh <= pos+2 < pos+length <= in_end, and buf's
                            // pad covers the 4-byte load past in_end.
                            let s = unsafe { load_u32(base, nh) };
                            head[lz_hash(s, HASH_BITS) as usize] = nh as u32;
                            nh += 1;
                        }

                        sink.push_match(length, dist as u32);
                        pos += length as usize;
                        if pos >= block_end_target {
                            break;
                        }
                        continue;
                    }
                }
            }

            // Literal.
            sink.push_literal(buf[pos]);
            pos += 1;
            if pos >= block_end_target {
                break;
            }
        }

        // Flush the block: cheapest of per-block dynamic / static / stored.
        // The BFINAL bit is set only on the last internal block AND only when
        // this chunk is the last chunk of the whole stream (`is_last`). A
        // non-final chunk's blocks stay BFINAL=0 so the sync-flush marker the
        // caller appends can close the chunk on a clean boundary.
        emit_block(
            bw,
            buf,
            block_begin,
            &sink,
            statics,
            is_last && pos == in_end,
        );
        if pos == in_end {
            break;
        }
    }
}
