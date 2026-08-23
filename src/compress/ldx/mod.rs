//! `ldx` — a FAITHFUL, function-by-function port of libdeflate's compressor
//! (`vendor/libdeflate/lib/deflate_compress.c`, 4,155 lines) into pure Rust.
//!
//! Upstream license: libdeflate is Copyright 2016 Eric Biggers, Copyright
//! 2024 Google LLC, MIT — see `THIRD_PARTY_NOTICES.md` at the repo root.
//!
//! # Why this module exists, and why it is SEPARATE from `super::deflate`
//!
//! Our shipping encoder is not libdeflate's. It is FOUR lineages stitched together
//! behind our own dispatch:
//!
//! | module | lines | lineage |
//! |---|---|---|
//! | `deflate/parse/fast.rs` | 3,380 | **igzip** — this is L1, not libdeflate at all |
//! | `deflate/parse/mod.rs` | 2,522 | ours (dispatch, seq store, block emission) |
//! | `deflate/matchfinder/hc.rs` | 1,422 | libdeflate port |
//! | `deflate/parse/near_optimal.rs` + `costs.rs` | 1,462 | libdeflate port |
//! | `deflate/matchfinder/{bt,ht,common,lzfind}.rs` | 1,991 | libdeflate + our own `lzfind` |
//! | `deflate/huffman/{fast,optimal,header}.rs` | 1,479 | mixed |
//! | `deflate/parse/ultra/*` | 5,000+ | **zopfli** |
//!
//! ~17,000 lines of core deflate path doing what libdeflate does in 4,155. That
//! structural fact — not any single number — is the gap. It is why L1 diverges,
//! why the level->config map is inherited while the code around it is not, and why
//! byte-identity with libdeflate holds on 154 of 198 T1 cells but not the rest.
//!
//! **GOAL (user, 2026-08-01): perfectly copy libdeflate's exact implementation in
//! pure Rust, performing exactly the same and producing the same output.**
//!
//! This is built as a NEW module rather than a refactor of `super::deflate` for two
//! reasons that are both about keeping a measurement honest:
//!
//! 1. **A clean oracle.** Every function here can be differentially tested against
//!    the C directly. A refactor-in-place has no oracle — you cannot tell a port bug
//!    from a pre-existing divergence.
//! 2. **It cannot regress a closed cell while it is being built.** We are
//!    byte-identical to libdeflate on 154 T1 cells with ZERO tolerance (clause 3 is
//!    absolute). Nothing here is routed until it is proven byte-identical.
//!
//! # Porting rules — these are what make it a COPY and not a rewrite
//!
//! * **Same functions, same names, same order as the C.** Each item carries a
//!   `C:` comment naming its source line in `deflate_compress.c`. If you cannot
//!   point at the C, it does not belong here.
//! * **Same arithmetic, same types.** `u32` stays `u32`; C's wrapping/truncating
//!   semantics are reproduced with `wrapping_*` / `as` casts, never "improved".
//!   An overflow the C relies on is behaviour, not a bug to fix.
//! * **No idiomatic cleanups.** No iterators where the C has an index loop, no
//!   `Option` where the C uses a sentinel. Divergence in shape is divergence in
//!   codegen, and codegen is half the goal ("performing exactly the same").
//! * **Port the comments too.** The C's comments explain non-obvious invariants
//!   (e.g. why the heap is 1-indexed, why lengths are capped before sorting). Losing
//!   them is how a later "obvious simplification" breaks byte-identity.
//! * **No env knobs, no content detection** — CLAUDE.md clause 3, unchanged here.
//!
//! # Verification ladder (each rung must pass before the next is attempted)
//!
//! 1. **Unit-level differential**: for each ported function, identical inputs ->
//!    identical outputs vs the C (drive the C through a small harness).
//! 2. **Block-level**: `deflate_flush_block` emits byte-identical bits for a given
//!    sequence store + frequency table.
//! 3. **Whole-stream**: every corpus file, every level 0-12, `sha256(ours) ==
//!    sha256(libdeflate-gzip)`. This is the real gate — `wc -c` never counts.
//! 4. **Wall**: only once 3 passes. `fulcrum ab paired`, read `aa_bias`.
//!
//! Nothing routes into the shipping path until rung 3 is green for that level, and
//! `scripts/campaign/tie-guard.sh` runs before any change that alters T1 output.
//!
//! # Port status
//!
//! See `PORT_STATUS.md` in the repo root for the live checklist. Order is dependency
//! order, which is roughly the C's own file order:
//!
//! * [x] constants (this file)
//! * [ ] heap sort + `sort_symbols` + `build_tree` + `compute_length_counts`
//! * [ ] `gen_codewords` + `deflate_make_huffman_code(s)`
//! * [ ] precode: `deflate_compute_precode_items` / `deflate_precompute_huffman_header`
//! * [ ] `deflate_flush_block` (the big one — 334 lines of C)
//! * [ ] sequence store + `deflate_choose_literal` / `deflate_choose_match`
//! * [ ] block-split stats (`init_block_split_stats` .. `should_end_block`)
//! * [ ] matchfinders: `hc_matchfinder.h`, `ht_matchfinder.h`, `bt_matchfinder.h`
//! * [ ] `deflate_compress_none` / `_fastest` / `_greedy` / `_lazy_generic`
//! * [ ] near-optimal: costs, `deflate_find_min_cost_path`, `_compress_near_optimal`
//! * [ ] `libdeflate_alloc_compressor_ex` (the level -> config map, ported verbatim)

#![allow(dead_code)] // Ports land bottom-up; unused until the driver is ported.

mod bitstream;
mod codes;
pub mod compress;
mod compress_fastest;
mod compress_greedy;
mod compress_lazy;
mod flush;
mod hc_matchfinder;
mod heap;
mod ht_matchfinder;
mod huffman;
mod length;
mod matchfinder_common;
mod min_match;
mod precode;
mod sequences;
mod split;
mod tables;

// ---------------------------------------------------------------------------
// C: deflate_compress.c:118-120 — libdeflate's OWN codeword length limits, which
// are not the same as the format's.
// ---------------------------------------------------------------------------

/// C: `#define MAX_LITLEN_CODEWORD_LEN 14` (:118)
///
/// **14, not the format's 15, and that is deliberate.** libdeflate caps litlen
/// codewords one bit below the RFC limit so that a litlen codeword plus its extra
/// length bits plus an offset codeword plus its extra offset bits always fit in one
/// 64-bit bitbuffer refill (see `WRITE_MATCH`, :1662). Raising it to 15 emits a legal
/// but different stream, and costs the emitter a flush.
pub const MAX_LITLEN_CODEWORD_LEN: usize = 14;

/// C: `#define MAX_OFFSET_CODEWORD_LEN DEFLATE_MAX_OFFSET_CODEWORD_LEN` (:119)
pub const MAX_OFFSET_CODEWORD_LEN: usize = DEFLATE_MAX_OFFSET_CODEWORD_LEN as usize;

/// C: `#define MAX_PRE_CODEWORD_LEN DEFLATE_MAX_PRE_CODEWORD_LEN` (:120)
pub const MAX_PRE_CODEWORD_LEN: usize = DEFLATE_MAX_PRE_CODEWORD_LEN as usize;

// ---------------------------------------------------------------------------
// C: deflate_compress.c:44-115 — block and sequence-store lengths.
//
// These are the parameters CLAUDE.md clause 5 is about: test the INVARIANT, not
// the VALUE. Nothing here asserts a specific number except where the number is
// a consequence of the others (MAX_BLOCK_LENGTH) or of RFC 1951 (65535).
// ---------------------------------------------------------------------------

/// C: `#define MIN_BLOCK_LENGTH 5000` (:67)
///
/// Defining a fixed minimum block length is needed in order to guarantee a
/// reasonable upper bound on the compressed size. It's also needed because the block
/// splitting algorithm doesn't work well on very short blocks.
pub const MIN_BLOCK_LENGTH: usize = 5000;

/// C: `#define SOFT_MAX_BLOCK_LENGTH 300000` (:82)
///
/// For the greedy, lazy, lazy2, and near-optimal compressors: the soft maximum block
/// length, in uncompressed bytes. The compressor will try to end blocks at this
/// length, but it may go slightly past it if there is a match that straddles this
/// limit or if the input data ends soon after this limit.
pub const SOFT_MAX_BLOCK_LENGTH: usize = 300_000;

/// C: `#define SEQ_STORE_LENGTH 50000` (:94)
///
/// The maximum number of matches that can be used in a block. If the sequence store
/// fills up, the compressor is forced to end the block early.
pub const SEQ_STORE_LENGTH: usize = 50_000;

/// C: `#define FAST_SOFT_MAX_BLOCK_LENGTH 65535` (:103)
///
/// `deflate_compress_fastest` doesn't use the regular block splitting algorithm; it
/// only ends blocks when they reach this many bytes or `FAST_SEQ_STORE_LENGTH`
/// matches.
pub const FAST_SOFT_MAX_BLOCK_LENGTH: usize = 65_535;

/// C: `#define FAST_SEQ_STORE_LENGTH 8192` (:109)
pub const FAST_SEQ_STORE_LENGTH: usize = 8192;

/// C: `#define MAX_BLOCK_LENGTH ...` (:188)
///
/// `MAX(SOFT_MAX_BLOCK_LENGTH + MIN_BLOCK_LENGTH - 1, SOFT_MAX_BLOCK_LENGTH + 1 +
/// DEFLATE_MAX_MATCH_LEN)` — the two ways a block can overrun its soft maximum.
pub const MAX_BLOCK_LENGTH: usize = {
    let a = SOFT_MAX_BLOCK_LENGTH + MIN_BLOCK_LENGTH - 1;
    let b = SOFT_MAX_BLOCK_LENGTH + 1 + DEFLATE_MAX_MATCH_LEN as usize;
    if a > b {
        a
    } else {
        b
    }
};

/// C: `#define NUM_OBSERVATIONS_PER_BLOCK_CHECK 512` (:444)
pub const NUM_OBSERVATIONS_PER_BLOCK_CHECK: u32 = 512;

// ---------------------------------------------------------------------------
// C: vendor/libdeflate/lib/deflate_constants.h
// Ported verbatim. Names match the C exactly (minus the DEFLATE_ prefix where the
// module path already supplies it) so a reader can grep either tree.
// ---------------------------------------------------------------------------

// Block types.
pub const DEFLATE_BLOCKTYPE_UNCOMPRESSED: u32 = 0;
pub const DEFLATE_BLOCKTYPE_STATIC_HUFFMAN: u32 = 1;
pub const DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN: u32 = 2;

// Match constraints.
pub const DEFLATE_MIN_MATCH_LEN: u32 = 3;
pub const DEFLATE_MAX_MATCH_LEN: u32 = 258;
pub const DEFLATE_MAX_MATCH_OFFSET: u32 = 32768;
pub const DEFLATE_WINDOW_ORDER: u32 = 15;

// Alphabet sizes.
pub const DEFLATE_NUM_PRECODE_SYMS: usize = 19;
pub const DEFLATE_NUM_LITLEN_SYMS: usize = 288;
pub const DEFLATE_NUM_OFFSET_SYMS: usize = 32;
pub const DEFLATE_MAX_NUM_SYMS: usize = 288;
pub const DEFLATE_NUM_LITERALS: usize = 256;
pub const DEFLATE_END_OF_BLOCK: u32 = 256;
pub const DEFLATE_FIRST_LEN_SYM: u32 = 257;

// Codeword length limits.
pub const DEFLATE_MAX_PRE_CODEWORD_LEN: u32 = 7;
pub const DEFLATE_MAX_LITLEN_CODEWORD_LEN: u32 = 15;
pub const DEFLATE_MAX_OFFSET_CODEWORD_LEN: u32 = 15;
pub const DEFLATE_MAX_CODEWORD_LEN: u32 = 15;

/// Maximum number of extra lengths that a run of precode symbol 16/17/18 can
/// write past the end of the lens array. C: `DEFLATE_MAX_LENS_OVERRUN`.
pub const DEFLATE_MAX_LENS_OVERRUN: usize = 137;

pub const DEFLATE_MAX_EXTRA_LENGTH_BITS: u32 = 5;
pub const DEFLATE_MAX_EXTRA_OFFSET_BITS: u32 = 13;

/// Compress `input` at `level` through the ported path and return the raw DEFLATE
/// bytes, or `None` if that level is not ported yet.
///
/// **This is a test/differential entry point, not a shipping one.** It exists so the
/// port's rung-3 gate — byte-for-byte against libdeflate's own
/// `libdeflate_deflate_compress` — can be run from `examples/ldxdump.rs`. Nothing in
/// `src/compress/deflate` calls it, and nothing routes here.
/// PRODUCTION entry point: compress `input` and APPEND the raw DEFLATE bytes to
/// `out`, with no scratch buffer and no copy.
///
/// [`compress_for_diff`] is the divergence ORACLE and allocates
/// `vec![0u8; input.len() * 2 + 65536]` — it ZEROES twice the input plus 64 KB on
/// every call, then the caller copies the result out again. That is fine for a
/// test; shipping it cost a 24.4 MB memset per compression on a 12 MB input, and
/// it is a FIXED cost, so it dominated the fast levels and vanished at L9 —
/// exactly the level pattern we could not explain (we lost 1.07-1.15x at L1-L6
/// and WON 0.84-0.96x at L9 against the very C we are a port of).
///
/// The bound is libdeflate's own worst case for its stored fallback: the input
/// plus 5 bytes of block header per 65535-byte sub-block plus framing slack.
/// `spare_capacity_mut` hands the compressor uninitialised bytes — nothing is
/// zeroed — and `set_len` commits only what it wrote.
/// C: `c->max_passthrough_size = 55 - (compression_level * 4);` (:3919).
///
/// Inputs at or below this length never reach a parser in libdeflate — it emits an
/// uncompressed block. Our streaming encoder does not, and is 3 bytes smaller there
/// (e.g. n=1: ours 3, port 6). The router uses this to keep the port off inputs where
/// it is strictly worse, which is what makes routing a level here a ZERO-size change
/// rather than a trade.
#[inline]
pub fn max_passthrough_size(level: u32) -> usize {
    55usize.saturating_sub((level as usize) * 4)
}

pub fn compress_into(level: u32, input: &[u8], out: &mut Vec<u8>) -> bool {
    let Some(mut c) = compress::LdxCompressor::new(level) else {
        return false;
    };
    let bound = input.len() + input.len().div_ceil(65535) * 5 + 64;
    let start = out.len();
    out.reserve(bound);
    let n = {
        let spare = out.spare_capacity_mut();
        // SAFETY: `spare` is `[MaybeUninit<u8>]` with at least `bound` elements.
        // `compress` writes only within the slice it is given and returns how
        // many bytes it wrote; we commit exactly that many below. Writing
        // uninitialised bytes through a `&mut [u8]` view is sound here because
        // the compressor never READS the buffer before writing it — it is a
        // pure output sink (the C takes a raw `void *out` for the same reason).
        let buf = unsafe { core::slice::from_raw_parts_mut(spare.as_mut_ptr() as *mut u8, bound) };
        c.compress(input, input.len(), buf)
    };
    if n == 0 {
        return false;
    }
    // SAFETY: `compress` reported writing `n` bytes into the reserved region.
    unsafe { out.set_len(start + n) };
    true
}

pub fn compress_for_diff(level: u32, input: &[u8]) -> Option<Vec<u8>> {
    let mut c = compress::LdxCompressor::new(level)?;
    let mut out = vec![0u8; input.len() * 2 + 65536];
    let n = c.compress(input, input.len(), &mut out);
    if n == 0 {
        return None;
    }
    out.truncate(n);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// These are wire-format constants from RFC 1951, not tunables. Pinning them
    /// is legitimate (CLAUDE.md clause 5 forbids pinning knobs DECLARED FREE to
    /// change; a format constant is the opposite of that).
    #[test]
    fn constants_match_rfc1951() {
        assert_eq!(DEFLATE_MIN_MATCH_LEN, 3);
        assert_eq!(DEFLATE_MAX_MATCH_LEN, 258);
        assert_eq!(DEFLATE_MAX_MATCH_OFFSET, 1 << DEFLATE_WINDOW_ORDER);
        assert_eq!(DEFLATE_NUM_LITLEN_SYMS, 288);
        assert_eq!(DEFLATE_NUM_OFFSET_SYMS, 32);
        assert_eq!(DEFLATE_NUM_PRECODE_SYMS, 19);
        assert_eq!(DEFLATE_END_OF_BLOCK as usize, DEFLATE_NUM_LITERALS);
        assert_eq!(DEFLATE_FIRST_LEN_SYM, DEFLATE_END_OF_BLOCK + 1);
        assert_eq!(DEFLATE_MAX_NUM_SYMS, DEFLATE_NUM_LITLEN_SYMS);
    }
}
