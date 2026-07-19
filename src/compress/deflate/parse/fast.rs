//! Level-1 igzip-class one-pass FAST parser.
//!
//! Port of igzip's level-0/1 deflate body
//! (`vendor/isa-l/igzip/igzip_base.c:isal_deflate_body_base`, :27-113, and the
//! matching `isal_deflate_finish_base` tail, :114-215). Unlike the
//! greedy/lazy/near-optimal parsers (which buffer a token stream and build a
//! dynamic Huffman code per block), this path is a single streaming pass that
//! emits DIRECTLY into the bitstream through a FIXED (RFC 1951 static) Huffman
//! table. That trades a little ratio (static vs dynamic Huffman) for igzip-class
//! throughput — the whole point of the fast mode.
//!
//! The mechanisms ported from igzip:
//!   1. **Chainless single-probe hash** (`igzip_base.c:60-64`): one small head
//!      table storing the last position per hash; overwrite on collision; ONE
//!      candidate per position — no chains, no depth loop.
//!   2. **LIMIT_HASH_UPDATE** (`igzip_base.c:71-86`): over an accepted match,
//!      insert the hash for only the first ~3 interior positions, then jump the
//!      cursor by the whole match length (skip the interior stores).
//!   3. **compare258 match-extend** (`huffman.h:260-314`): 8-byte XOR +
//!      trailing-zero-count, reusing Increment 1's [`lz_extend`].
//!   4. **Direct-emit via a precomputed Huffman LUT** (`igzip_lib.h:409-413`):
//!      a literal is one LUT read + one `add_bits_raw`; a match is a
//!      full-length-codeword LUT read + a distance-codeword read + `add_bits_raw`.
//!   5. **64-bit branchless bit buffer**: the shared [`BitWriter`] fast path
//!      (`add_bits_raw` / `flush_word_unchecked`).
//!
//! The whole input is coded as ONE static-Huffman block (BFINAL=1, BTYPE=01),
//! so there is no per-block tree build or block-split statistic at all.

use super::super::bitstream::BitWriter;
use super::super::matchfinder::common::{load_u32, lz_extend, lz_hash};
use super::super::tables::{offset_slot, DEFLATE_END_OF_BLOCK};
use super::super::tables::{DEFLATE_MAX_MATCH_LEN, OFFSET_EXTRA_BITS, OFFSET_SLOT_BASE};
use super::{FullLenCodewords, StaticCodes};

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

/// Run the one-pass fast encoder over `buf[data_start..in_end]`, appending a
/// single static-Huffman DEFLATE block to `bw`.
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
) {
    debug_assert!(in_end > data_start, "empty data handled by the caller");
    debug_assert!(buf.len() >= in_end + super::BUF_PAD);

    let litcode = &statics.litcode;
    let offcode = &statics.offcode;
    // Precompute the "litlen symbol + extra length bits" packed codeword per
    // match length, so a match's length field is a single LUT read + one add.
    let full_len = FullLenCodewords::build(litcode);

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

    // Reserve worst-case output capacity up front (every byte a 9-bit literal =
    // input * 9/8 bytes, plus header/EOB slack). This bounds the total bytes
    // written so every `flush_word_unchecked` below keeps its required 8 spare
    // bytes: capacity - out.len() stays >= 8 for the whole batch.
    let data_len = in_end - data_start;
    bw.reserve(data_len + data_len / 8 + 256);

    // Single static-Huffman block header: BFINAL=1, BTYPE=01. Uses the
    // auto-flushing `add_bits`; only 3 bits, so afterwards bitcount == 3 (<= 7),
    // satisfying the raw fast-path invariant with no explicit drain.
    bw.add_bits(1, 1);
    bw.add_bits(1, 2);

    let mut pos = data_start;
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
                    // pos+1, pos+2 (igzip's `end = next_hash + 3`), then jump the
                    // cursor over the whole match. length >= 4 guarantees these
                    // interior positions are inside the match.
                    let limit = pos + 3;
                    let mut nh = pos + 1;
                    while nh < limit {
                        // SAFETY: nh <= pos+2 < pos+length <= in_end, and buf's
                        // pad covers the 4-byte load past in_end.
                        let s = unsafe { load_u32(base, nh) };
                        head[lz_hash(s, HASH_BITS) as usize] = nh as u32;
                        nh += 1;
                    }

                    emit_match(bw, &full_len, offcode, length, dist as u32);
                    pos += length as usize;
                    continue;
                }
            }
        }

        // Literal.
        let b = buf[pos];
        bw.add_bits_raw(
            litcode.codewords[b as usize] as u64,
            litcode.lens[b as usize] as u32,
        );
        // SAFETY: reserve() above guarantees 8 spare bytes for every flush; a
        // literal adds <= 9 bits to a buffer holding <= 7, well under 63.
        unsafe { bw.flush_word_unchecked() };
        pos += 1;
    }

    // End-of-block symbol, then a final flush.
    bw.add_bits_raw(
        litcode.codewords[DEFLATE_END_OF_BLOCK] as u64,
        litcode.lens[DEFLATE_END_OF_BLOCK] as u32,
    );
    // SAFETY: see above; the reserved slack covers this final flush.
    unsafe { bw.flush_word_unchecked() };
}

/// Emit one back-reference: the packed length codeword, then the distance
/// codeword + extra distance bits, then a single whole-word flush.
///
/// The accumulator holds <= 7 bits on entry; a match adds at most
/// `14 (full len) + 5 (dist sym) + 13 (dist extra) = 32` bits, so 7 + 32 = 39
/// stays under the 63-bit ceiling — one flush per match, no intermediate drain.
#[inline]
fn emit_match(
    bw: &mut BitWriter,
    full_len: &FullLenCodewords,
    offcode: &super::super::huffman::HuffmanCode,
    length: u32,
    dist: u32,
) {
    debug_assert!((1..=WINDOW as u32).contains(&dist));
    bw.add_bits_raw(
        full_len.codewords[length as usize] as u64,
        full_len.lens[length as usize] as u32,
    );
    let os = offset_slot(dist) as usize;
    bw.add_bits_raw(offcode.codewords[os] as u64, offcode.lens[os] as u32);
    bw.add_bits_raw(
        (dist - OFFSET_SLOT_BASE[os]) as u64,
        OFFSET_EXTRA_BITS[os] as u32,
    );
    // SAFETY: the caller (`run`) reserved worst-case output capacity, so 8 spare
    // bytes are available for this flush.
    unsafe { bw.flush_word_unchecked() };
}
