//! `ldx` — a FAITHFUL, function-by-function port of libdeflate's compressor
//! (`vendor/libdeflate/lib/deflate_compress.c`, 4,155 lines) into pure Rust.
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

mod codes;
mod heap;
mod huffman;
mod length;
mod precode;
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
