//! Level-1 parser over the 2-way hash-table matchfinder (`deflate_compress_fastest`).
//!
//! Faithful transliteration of libdeflate `deflate_compress_fastest`
//! (`vendor/libdeflate/lib/deflate_compress.c:2452-2521`) over
//! [`super::super::matchfinder::ht::HtMatchfinder`]. It is deliberately a
//! SEPARATE parser from [`super::fast`] rather than a new branch inside it.
//!
//! # Why a separate parser, and why transliterate before deleting
//!
//! `docs/encoder-campaign-plan.md` §3 records the method that has actually won
//! here, counted: vendor structural diff/port is 9 wins and 0 falsifications;
//! shaving our own profile's top line is 0 wins and >= 17 falsifications. It
//! also records the sequencing rule — *"Converge on the vendor's structure
//! BEFORE deleting anything. Decode halved the igzip gap by faithful
//! transliteration; only then did single-op deletions become visible and worth
//! 5-10% each. Deleting first is what produced this campaign's
//! falsifications."*
//!
//! So this is step one of two, and step one is on purpose:
//!   1. **(this)** stand up the vendor's shape beside ours, take the size
//!      verdict on the real corpus. `super::fast` is untouched, so L0 and the
//!      shipped L1 are byte-identical and every other level is unaffected.
//!   2. **only if the size verdict holds**, delete what it makes dead — the
//!      `head3` side table, `Hash3Cfg`'s `gated`/`gate_*` fields,
//!      `L1_HASH3_GATE_LIT_THRESHOLD_PCT` (a constant fitted two points off one
//!      file's cliff, which `CLAUDE.md` non-negotiable #3 orders deleted),
//!      `LAZY_PEEK_GATE*`, and the `l1-tune` knob module.
//!
//! # Structural differences from `super::fast`, all of them libdeflate's
//!
//! | | `super::fast` (shipped L1) | this parser |
//! |---|---|---|
//! | candidates per position | 1 (`head`) + a 3-byte-keyed `head3` probe | 2, inline in one bucket |
//! | working set | 256 KiB + 128 KiB | **128 KiB** |
//! | min match | 3 (via `head3`) | 4 |
//! | block end | 65,536 B | 65,535 B **OR 8,192 seqs** |
//! | match accept | min-length heuristic + offset guard + lazy peek | **any match at all** |
//! | content gates | hash3 gate, lazy-peek gate | **none** |
//!
//! The accept rule is the part most likely to surprise a reader: `fastest` takes
//! whatever the matchfinder returns, with no `calculate_min_match_len`, no
//! `offset <= 4096` guard for length-3 matches (it has no length-3 matches to
//! guard), and no lazy peek. All of the selectivity lives in the matchfinder's
//! `MIN_MATCH_LEN 4` and its `nice_len` cutoff.
//!
//! # Bounds
//!
//! This parser is bounded TWICE — 65,535 input bytes and 8,192 sequences per
//! block — so it cannot reach the overflow condition the FALSIFY note on
//! `SEQ_STORE_CAPACITY` (`super::SEQ_STORE_CAPACITY`) warns about. That note
//! records that `fast` and `near_optimal` never call `continue_block` and are
//! bounded only by their block span, which is why the store is sized for the
//! worst case any parser can produce; the seq cap here is far below it.

use super::super::bitstream::BitWriter;
use super::super::huffman::HeaderScratch;
use super::super::level::LevelParams;
use super::super::matchfinder::ht::{HtMatchfinder, HT_REQUIRED_NBYTES};
use super::super::tables::DEFLATE_MAX_MATCH_LEN;
use super::{emit_block, BlockRole, Sink, StaticCodes};

/// `FAST_SOFT_MAX_BLOCK_LENGTH` (`deflate_compress.c:102`). Deliberately below
/// the regular `SOFT_MAX_BLOCK_LENGTH`, per libdeflate's own comment.
pub(super) const FAST_SOFT_MAX_BLOCK_LENGTH: usize = 65_535;
/// `FAST_SEQ_STORE_LENGTH` (`deflate_compress.c:108`).
pub(super) const FAST_SEQ_STORE_LENGTH: usize = 8_192;

/// `choose_max_block_end` at the fast path's soft maximum.
///
/// Distinct from [`super::choose_max_block_end`], which bakes in the regular
/// 300,000 B `SOFT_MAX_BLOCK_LENGTH`. Same shape: if the tail is shorter than
/// one soft-max plus one minimum block, take it all rather than leaving a runt.
#[inline]
fn choose_max_block_end_fast(block_begin: usize, in_end: usize) -> usize {
    if in_end - block_begin < FAST_SOFT_MAX_BLOCK_LENGTH + super::MIN_BLOCK_LENGTH {
        in_end
    } else {
        block_begin + FAST_SOFT_MAX_BLOCK_LENGTH
    }
}

/// Compress `buf[data_start..in_end]` at level 1 into `bw`.
///
/// `data_start > 0` seeds a preset dictionary: those positions are inserted into
/// the matchfinder so matches may reference them, but they are not coded.
pub(super) fn run(
    buf: &[u8],
    data_start: usize,
    in_end: usize,
    params: &LevelParams,
    statics: &StaticCodes,
    bw: &mut BitWriter,
    is_last: bool,
) {
    let mut mf = HtMatchfinder::acquire();
    let mut in_base = 0usize;
    // libdeflate initialises `next_hash = 0`, so the FIRST position probes
    // bucket 0 with a hash that does not describe its bytes. That is harmless
    // and intentional: the bucket only proposes candidates, and the 4-byte
    // `seq` compare in `longest_match` is what establishes a real match. Kept
    // identical rather than "fixed" so parse decisions match the vendor's.
    let mut next_hash = 0u32;
    // Second key, for the length-3 singleton table. Same `0` seeding convention and
    // the same reason it is harmless — the table only proposes a candidate, and the
    // 3-byte compare in `longest_match` is what establishes a real match.
    let mut next_hash3 = 0u32;
    let mut sink = Sink::new();
    // One dynamic-header scratch for the WHOLE call, reused across every
    // internal block, instead of `build_dynamic_header` allocating per block.
    let mut header_scratch = HeaderScratch::new();

    let mut in_next = data_start;
    if data_start > 0 {
        mf.skip_bytes(
            buf,
            &mut in_base,
            0,
            in_end,
            data_start as u32,
            &mut next_hash,
            &mut next_hash3,
        );
    }

    // Declared OUTSIDE the block loop and mutated, exactly as in the C: once the
    // tail shortens them they stay shortened for the rest of the input.
    let mut max_len = DEFLATE_MAX_MATCH_LEN;
    let mut nice_len = params.nice_match_length.min(max_len);

    let role = if is_last {
        BlockRole::Final
    } else {
        BlockRole::Interior
    };

    // An empty input still has to emit one (final, empty) block.
    if in_next == in_end {
        sink.begin();
        sink.block_length = 0;
        emit_block(
            bw,
            buf,
            in_next,
            &sink,
            statics,
            role.is_final(),
            &mut header_scratch,
        );
        return;
    }

    loop {
        // ---- starting a new DEFLATE block ----
        let block_begin = in_next;
        let in_max_block_end = choose_max_block_end_fast(in_next, in_end);
        sink.begin();

        // `anatomy-wall` region `parse_match`: one timer per INTERNAL BLOCK,
        // matching the convention in `fast.rs` and `greedy.rs` — probe and
        // emission are fused here for the same reason they are there (no call
        // boundary inside the token loop to split them at without going to
        // per-position granularity). Zero cost when `anatomy-wall` is off.
        in_next = crate::anatomy_wall_time!(parse_match_ns, parse_match_calls, {
            let mut pos = in_next;
            loop {
                let remaining = in_end - pos;
                if remaining < DEFLATE_MAX_MATCH_LEN as usize {
                    max_len = remaining as u32;
                    if max_len < HT_REQUIRED_NBYTES {
                        // Fewer than 5 bytes left: the matchfinder cannot be
                        // called at all, so drain the tail as literals.
                        let mut n = max_len;
                        while n > 0 {
                            sink.push_literal_fast(buf[pos]);
                            pos += 1;
                            n -= 1;
                        }
                        break;
                    }
                    nice_len = nice_len.min(max_len);
                }

                let (length, offset) = mf.longest_match(
                    buf,
                    &mut in_base,
                    pos,
                    max_len,
                    nice_len,
                    &mut next_hash,
                    &mut next_hash3,
                );

                if length != 0 {
                    // `fastest` accepts ANY match the finder returns — no
                    // min-length heuristic, no short-match offset guard, no
                    // lazy peek. Selectivity lives in MIN_MATCH_LEN 4/nice_len.
                    sink.push_match_fast(length, offset);
                    mf.skip_bytes(
                        buf,
                        &mut in_base,
                        pos + 1,
                        in_end,
                        length - 1,
                        &mut next_hash,
                        &mut next_hash3,
                    );
                    pos += length as usize;
                } else {
                    sink.push_literal_fast(buf[pos]);
                    pos += 1;
                }

                // Time to close the block? Bounded by input span AND seq count.
                if pos >= in_max_block_end || sink.nseqs >= FAST_SEQ_STORE_LENGTH {
                    break;
                }
            }
            pos
        });

        // The `_fast` push variants skip per-token bookkeeping, so the block's
        // input span is derived once here (see `Sink::push_literal_fast`).
        sink.block_length = in_next - block_begin;
        emit_block(
            bw,
            buf,
            block_begin,
            &sink,
            statics,
            role.is_final() && in_next == in_end,
            &mut header_scratch,
        );
        if in_next >= in_end {
            return;
        }
    }
}
