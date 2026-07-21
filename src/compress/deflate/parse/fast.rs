//! Level-0/1 igzip-class one-pass FAST parser.
//!
//! Shared by `Strategy::Fast0` (L0) and `Strategy::Fast` (L1): both use the
//! IDENTICAL chainless single-probe matchfinder below; they differ only in
//! `use_dynamic` at the block-emit step (L1 evaluates a per-block dynamic
//! Huffman code, L0 does not — see `super::emit_block_static_or_stored`'s doc
//! comment for why that is the L0/L1 cost/ratio trade).
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
//!   1. **Chainless single-probe hash** (`igzip_base.c:60-64`): one head table
//!      storing the last position per hash; overwrite on collision; ONE
//!      candidate per position — no chains, no depth loop. **DEVIATION from
//!      igzip:** the table is `1 << 16` (64K) slots, not igzip's `1 << 13` (8K).
//!      Because there is only ONE probe, a wider table is the cheapest way to
//!      cut hash collisions — the single candidate is far less often an
//!      unrelated position — and it closes the last ~1% ratio gap vs `pigz -1`
//!      on `text`/`bin` at ~zero speed cost (still one load + one compare per
//!      position). See [`HASH_BITS`].
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
use super::super::matchfinder::common::{load_u32, lz_extend, lz_hash, prefetch_write};
use super::super::tables::DEFLATE_MAX_MATCH_LEN;
use super::{emit_block, emit_block_static_or_stored, Sink, StaticCodes};

/// Log2 of the head-table size. igzip's level-0 hash table is
/// `IGZIP_LVL0_HASH_SIZE = 8 * 1024 = 1 << 13` (`igzip_lib.h:121-125`); we widen
/// it to `1 << 16` (64K) because the finder is single-probe — a wider table
/// spreads the 4-byte keys over 8× more slots, so the ONE candidate we keep per
/// hash is far less likely to be an unrelated collision, and it recovers the
/// last ~1% ratio gap vs pigz-1 on `text`/`bin` at near-zero speed cost (same
/// one load, one compare per position). See Lever 1.
const HASH_BITS: u32 = 16;
const HASH_SIZE: usize = 1 << HASH_BITS;

/// Software-pipeline distance for the head-table prefetch (SF1-C). Each fastloop
/// iteration prefetches the head slot for the position it will probe `PF_DIST`
/// steps ahead, so the dependent `head[h]` load — cachegrind-named as 69% of the
/// L1 fast path's D1 read misses, and perf-confirmed as the IPC collapse vs igzip
/// on binary data (IPC 1.32 vs 2.46) — is already warm when consumed. Pure hint:
/// it warms a cache line, never changes a value the finder reads, so output stays
/// byte-identical. Same technique as the hc.rs chain-walk prefetch.
const PF_DIST: usize = 4;

/// LIMIT_HASH_UPDATE: number of match-interior positions whose hash is inserted
/// into the head table before the cursor jumps over the rest of the match
/// (`igzip_base.c:71-86`, igzip uses ~3). Denser interior inserts seed more
/// candidates for later matches (better ratio) at the cost of more hash stores;
/// `usize::MAX` means "insert EVERY interior position" (zlib-ng style).
const LIMIT_HASH_UPDATE_INSERTS: usize = 2;

/// Sentinel head-table entry meaning "no position stored yet". Any position we
/// store is `< in_end <= u32::MAX`, so the sentinel never collides with a real
/// index, and its computed distance always fails the window test.
const NO_POS: u32 = u32::MAX;

/// L0-only search acceleration (the `ACCEL` const generic on [`run`]), an
/// LZ4-`LZ4_compress_fast`-style scan-step ramp: no vendor DEFLATE encoder
/// counterpart, a novel technique for this chainless single-probe finder.
/// The scan step is `1 + (consecutive_misses >> ACCEL_SHIFT)`, capped at
/// `ACCEL_MAX_STEP`: every `1 << ACCEL_SHIFT` further consecutive-miss
/// positions, the per-position hash lookup/insert is skipped for a growing
/// number of subsequent positions — those bytes are coded as literals
/// directly with no finder work at all. Any match resets the miss counter
/// (and so the step) back to 1. This trades some missed matches (ratio) for
/// skipping finder work outright (speed) — but ONLY on long literal runs,
/// which is exactly where L1's exhaustive per-position search is least
/// likely to pay off. `run::<false>` (L1 / `Strategy::Fast`) monomorphizes
/// with this whole mechanism compiled away — this is strictly an L0
/// (`Strategy::Fast0`) lever.
const ACCEL_SHIFT: u32 = 1;
/// Consecutive-miss count at which the ramp arms (below this, `step` stays 1
/// and the per-literal cost is just the counter increment + one
/// well-predicted comparison — see the arming check's doc comment in
/// [`run`]).
const ACCEL_ARM_THRESHOLD: u32 = 1 << ACCEL_SHIFT;
/// Cap on the accelerated scan step (bytes skipped per ramped-up jump).
/// Bounded well under the `DEFLATE_MAX_MATCH_LEN` (258) fastloop safety
/// margin so the skip can never run the cursor past `in_end`.
const ACCEL_MAX_STEP: usize = 8;

/// igzip `SHORTEST_MATCH` (`huff_codes.h:89`): the fast path only emits matches
/// of length >= 4 (its hash keys 4 bytes), coding anything shorter as literals.
const SHORTEST_MATCH: u32 = 4;

/// DEFLATE sliding-window size — the largest legal back-reference distance.
const WINDOW: usize = 32768;

/// Input bytes covered per DEFLATE block in the fast path (L1 value —
/// [`super::compress`] passes a caller-chosen `block_length`, see
/// [`FAST0_BLOCK_LENGTH`] for the L0 value).
///
/// A per-block dynamic Huffman code adapts to LOCAL statistics, so a moderate
/// block (vs one whole-input block) improves ratio on heterogeneous input; at
/// 64 KiB the ~dozens-of-bytes dynamic header amortizes to well under 1%. This
/// is the fast path's one ratio/speed tuning knob — it does not affect
/// correctness (any block boundary roundtrips).
pub(super) const FAST_BLOCK_LENGTH: usize = 1 << 16;

/// L0's block length. The per-block dynamic-Huffman build (canonical code +
/// length-limiting over up to 288+32 symbols) costs roughly the SAME whether
/// the block covers 64 KiB or 1 MiB — it is a function of alphabet size, not
/// byte count — so widening the block cuts the number of builds (and their
/// bit-cost evaluations / header emissions) roughly proportionally with
/// little further ratio loss beyond less-local adaptation. 16x L1's block
/// gives L0 ~1/16th the per-block overhead while keeping most of L1's ratio
/// (measured: far closer to L1 than the static-Huffman-only alternative,
/// which gave up ~20%+ on text).
pub(super) const FAST0_BLOCK_LENGTH: usize = 1 << 20;

/// Run the one-pass fast encoder over `buf[data_start..in_end]`, appending one
/// or more DEFLATE blocks to `bw`.
///
/// `buf` MUST carry at least [`super::BUF_PAD`] trailing pad bytes beyond
/// `in_end` (upheld by the caller in `deflate::compress_block`) so the
/// speculative 4-byte hash loads and 8-byte match-extend loads never read out of
/// bounds. `buf[..data_start]` is an optional preset dictionary: its positions
/// are seeded into the head table so matches may reference it, but it is not
/// coded.
///
/// `block_length` is the input-byte span covered by each internal block
/// (L1 passes [`FAST_BLOCK_LENGTH`], L0 passes [`FAST0_BLOCK_LENGTH`]).
/// `use_dynamic` selects the block emitter: `true` evaluates a per-block
/// dynamic Huffman code (cheapest of dynamic/static/stored, [`emit_block`]);
/// `false` skips that build entirely (cheapest of static/stored only,
/// [`emit_block_static_or_stored`]).
///
/// `ACCEL` is a CONST generic (not a runtime `bool`) so L1's call
/// (`run::<false>`) monomorphizes to code with the accel state/arithmetic
/// compiled away entirely — L1's fastloop is exactly the code that existed
/// before the accel lever was added, not "the same logic with a runtime
/// branch". Only the `run::<true>` instantiation (L0) carries the ramp.
pub(super) fn run<const ACCEL: bool>(
    buf: &[u8],
    data_start: usize,
    in_end: usize,
    statics: &StaticCodes,
    bw: &mut BitWriter,
    is_last: bool,
    block_length: usize,
    use_dynamic: bool,
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
    // Consecutive-miss counter driving the `ACCEL` scan-step ramp (see
    // `ACCEL_SHIFT`'s doc comment). Dead (never read) when `ACCEL` is
    // `false` — that monomorphization compiles it away.
    #[allow(unused_assignments)]
    let mut literal_run: u32 = 0;

    loop {
        // Start a new block. It ends after `block_length` input bytes (a match
        // straddling the boundary is allowed to overrun it slightly) or at EOF.
        let block_begin = pos;
        sink.begin();
        let block_end_target = (block_begin + block_length).min(in_end);

        // FASTLOOP / TAIL split (igzip loop2 shape). While
        // `pos < in_end - DEFLATE_MAX_MATCH_LEN`, `remaining` strictly exceeds
        // the longest possible match, so `max_len == DEFLATE_MAX_MATCH_LEN` is a
        // constant and the per-position `remaining`/`max_len`/`>= SHORTEST_MATCH`
        // computations fold away; folding the block-end break into the loop
        // condition removes the second per-token branch. Token equivalence with
        // the previous single loop: a token starting at `p` was processed iff
        // `p < block_end_target` (the break was checked AFTER each token, and
        // `block_end_target <= in_end`), which is exactly the pre-checked loop
        // condition here; positions below `fast_end` additionally satisfied
        // `max_len == DEFLATE_MAX_MATCH_LEN`. Identical token decisions ⇒
        // identical bytes.
        let fast_end = block_end_target.min(in_end.saturating_sub(DEFLATE_MAX_MATCH_LEN as usize));
        while pos < fast_end {
            // SF1-C software-pipeline: warm the head-table line for the position
            // PF_DIST ahead. `pos + PF_DIST` is speculative (a match jumps the
            // cursor past it); a wrong prefetch only wastes bandwidth. SAFETY:
            // `pos < fast_end <= in_end - 258` and `PF_DIST` is small, so
            // `pos + PF_DIST + 4 <= in_end`, in bounds for the 4-byte load; the
            // prefetch address is a pure hint that never faults.
            unsafe {
                let fseq = load_u32(base, pos + PF_DIST);
                let fh = lz_hash(fseq, HASH_BITS) as usize;
                prefetch_write(head.as_ptr().add(fh) as *const u8);
            }
            // SAFETY: `pos < fast_end <= in_end - 258`, so `pos + 4 <= in_end`.
            let seq = unsafe { load_u32(base, pos) };
            let h = lz_hash(seq, HASH_BITS) as usize;
            // SAFETY: `lz_hash(_, HASH_BITS)` output is `< 2^16 == HASH_SIZE`.
            let cand = unsafe { *head.get_unchecked(h) };
            unsafe { *head.get_unchecked_mut(h) = pos as u32 };

            // `pos - cand`; a wrapping sub keeps a sentinel/stale entry out of
            // the window range instead of panicking on underflow.
            let dist = pos.wrapping_sub(cand as usize);
            if (1..=WINDOW).contains(&dist) {
                let cand_pos = cand as usize;
                // Byte-exact extend (never trusts the hash): a spurious
                // candidate simply yields length < SHORTEST_MATCH -> literal.
                let length = lz_extend(buf, pos, cand_pos, 0, DEFLATE_MAX_MATCH_LEN);
                if length >= SHORTEST_MATCH {
                    // LIMIT_HASH_UPDATE (see the tail loop for the full note).
                    let match_end = pos + length as usize;
                    let insert_end = if LIMIT_HASH_UPDATE_INSERTS == usize::MAX {
                        match_end
                    } else {
                        (pos + 1 + LIMIT_HASH_UPDATE_INSERTS).min(match_end)
                    };
                    let mut nh = pos + 1;
                    while nh < insert_end {
                        // SAFETY: nh < match_end = pos+length <= in_end, and
                        // buf's pad covers the 4-byte load past in_end.
                        let s = unsafe { load_u32(base, nh) };
                        // SAFETY: `lz_hash` output `< HASH_SIZE`, as above.
                        unsafe {
                            *head.get_unchecked_mut(lz_hash(s, HASH_BITS) as usize) = nh as u32
                        };
                        nh += 1;
                    }

                    sink.push_match_fast(length, dist as u32);
                    pos += length as usize;
                    literal_run = 0;
                    continue;
                }
            }

            // Literal.
            // SAFETY: `pos < fast_end <= in_end <= buf.len()`.
            sink.push_literal_fast(unsafe { *buf.get_unchecked(pos) });

            // Common-case-cheap ramp: below `ACCEL_ARM_THRESHOLD` consecutive
            // misses, the ONLY added-vs-L1 cost is the `literal_run += 1` and
            // a (well-predicted-not-taken, on any corpus with normal match
            // density) comparison — the shift/min/extra-literal-copy work
            // only runs once we're actually inside a long literal run, which
            // is the ONLY time it can pay for itself. This keeps L0 from
            // regressing vs L1 on compressible corpora where the ramp rarely
            // arms (measured: computing `step` unconditionally cost L0 ~2-3%
            // vs L1 on `text6`, even fully branch-eliminated for L1 via the
            // `ACCEL` const generic — the tax was the arithmetic itself, not
            // dispatch).
            let mut step = 1usize;
            if ACCEL {
                literal_run += 1;
                if literal_run >= ACCEL_ARM_THRESHOLD {
                    // Scan-step ramp: skip the hash lookup/insert for
                    // `step - 1` further positions, coding them as literals
                    // directly. `pos + step <= pos + ACCEL_MAX_STEP <
                    // fast_end + ACCEL_MAX_STEP <= in_end` (fast_end's
                    // 258-byte margin dwarfs ACCEL_MAX_STEP), so every extra
                    // literal index stays in bounds.
                    step = (1 + (literal_run >> ACCEL_SHIFT) as usize).min(ACCEL_MAX_STEP);
                    let mut i = 1;
                    while i < step {
                        // SAFETY: see the bounds note above.
                        sink.push_literal_fast(unsafe { *buf.get_unchecked(pos + i) });
                        i += 1;
                    }
                }
            }
            pos += step;
        }

        // TAIL: the last <= DEFLATE_MAX_MATCH_LEN bytes of input (or of the
        // block), where `max_len` must be clamped per position.
        while pos < block_end_target {
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
                        // LIMIT_HASH_UPDATE: insert the hash for the first
                        // LIMIT_HASH_UPDATE_INSERTS match-interior positions
                        // (igzip inserts ~3), then jump the cursor over the whole
                        // match. usize::MAX means "insert every interior position"
                        // (zlib-ng style). length >= 4 guarantees at least the
                        // first interior positions are inside the match; the
                        // `.min(match_end)` clamp keeps every insert inside it.
                        let match_end = pos + length as usize;
                        let insert_end = if LIMIT_HASH_UPDATE_INSERTS == usize::MAX {
                            match_end
                        } else {
                            (pos + 1 + LIMIT_HASH_UPDATE_INSERTS).min(match_end)
                        };
                        let mut nh = pos + 1;
                        while nh < insert_end {
                            // SAFETY: nh < match_end = pos+length <= in_end, and
                            // buf's pad covers the 4-byte load past in_end.
                            let s = unsafe { load_u32(base, nh) };
                            head[lz_hash(s, HASH_BITS) as usize] = nh as u32;
                            nh += 1;
                        }

                        sink.push_match_fast(length, dist as u32);
                        pos += length as usize;
                        continue;
                    }
                }
            }

            // Literal.
            sink.push_literal_fast(buf[pos]);
            pos += 1;
        }

        // The fast-path pushes skip per-push `block_length` bookkeeping; the
        // covered length is exactly the cursor distance walked this block.
        sink.block_length = pos - block_begin;

        // Flush the block. The BFINAL bit is set only on the last internal
        // block AND only when this chunk is the last chunk of the whole
        // stream (`is_last`). A non-final chunk's blocks stay BFINAL=0 so the
        // sync-flush marker the caller appends can close the chunk on a clean
        // boundary.
        let is_final = is_last && pos == in_end;
        if use_dynamic {
            // L1: cheapest of per-block dynamic / static / stored.
            emit_block(bw, buf, block_begin, &sink, statics, is_final);
        } else {
            // L0: cheapest of static / stored only — no per-block dynamic
            // Huffman build (see `emit_block_static_or_stored`'s doc comment
            // for why this is the L0-vs-L1 cost/ratio trade).
            emit_block_static_or_stored(bw, buf, block_begin, &sink, statics, is_final);
        }
        if pos == in_end {
            break;
        }
    }
}
