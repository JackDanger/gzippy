//! Lempel-Ziv matchfinding.
//!
//! # The four tiers (Stage D, `docs/compressor-architecture.md` §5)
//!
//! gzippy runs FOUR distinct matchfinders, one per speed/quality tier. This is
//! a deliberate outcome, not unfinished consolidation: each finder's shape is
//! dictated by what its consumer needs, and the three real modules already
//! share every primitive that can be shared WITHOUT touching a hot call site
//! (see [`common`]). A single generic `trait Matchfinder` over all four was
//! evaluated and rejected — `hc`/`bt`/`lzfind` return structurally different
//! shapes (one best match vs. a sorted list vs. packed interleaved pairs) for
//! reasons documented per-tier below, and forcing them through one interface
//! would mean either (a) allocating/copying to a common shape on every call
//! (a real cost in loops that run once per input byte) or (b) an enum/dyn
//! dispatch boundary that risks losing the `#[inline(always)]` LTO the SF-chain
//! speed work depends on. Both were judged not worth the "one trait" prize —
//! see the SF1-A/C prefetch/software-pipeline commit history on `hc`/`fast.rs`
//! for what that inlining is worth in practice.
//!
//! | Tier | Module | Levels | Consumer(s) | Output shape | Position/window model |
//! |---|---|---|---|---|---|
//! | **ht** (chainless single-probe) | fused into [`crate::compress::deflate::parse::fast`] — see below | L0-1 | `parse::fast::run` | inline locals (no return type at all) | `u32` head table, no chain, no `mf_pos_t` sentinel |
//! | **hc** (hash chains) | [`hc`] | L2-9 | `parse::greedy::run`, `parse::lazy::run`; Stage D adds `parse::ultra::greedy_hc` | `(length, offset)` — the ONE best match | `i16` `mf_pos_t` relative to a sliding `in_base`, saturating rebase |
//! | **bt** (binary tree, single-best) | [`bt`] | L10-12 | `parse::near_optimal::run` | `&mut [LzMatch]`, sorted by strictly-increasing length | same `i16`/`in_base` model as `hc` |
//! | **lzfind** (ECT BT4, full-Pareto) | [`lzfind`] | Ultra (`-F`/`-I`/`-J`) | `parse::ultra::squeeze::get_best_lengths` (the DP driver) | packed `&mut [u16]`, interleaved `len, dist, len, dist, …` | `u32` positions in a private padded owned buffer, no sentinel/rebase |
//!
//! **Why `ht` is not extracted into this module.** The level-0/1 fast path
//! (`parse/fast.rs`) inlines its chainless single-probe hash table directly
//! into the fastloop body — head-table lookup, insert, and the `LIMIT_HASH_UPDATE`
//! interior-hash-insert skip are all interleaved with the token-emit and the
//! SF1-C software-pipelined prefetch in one function (`run::<ACCEL>`). Pulling
//! the finder out into its own type (mirroring `hc`/`bt`) would put a function
//! boundary between the prefetch issue and its consumer load, which is exactly
//! the dependency the prefetch exists to hide — a real codegen risk for a
//! module that exists ONLY to satisfy "one shared shape," not a measured need.
//! Per the Stage D charter ("do NOT genericize hot call sites if it changes
//! codegen"), `ht` stays fused; this doc entry + the module-doc note atop
//! `fast.rs` IS its documented home in the tier table.
//!
//! **Why `hc`/`bt`/`lzfind` stay three separate types instead of trait objects
//! over one shared vocabulary.** [`common::LzMatch`] IS the shared vocabulary
//! type where two signatures already agree (`bt`'s list output; a future
//! list-returning tier could reuse it for free) — see its doc comment for why
//! `hc` (single match, no list) and `lzfind` (packed `u16` pairs, no per-match
//! struct) are NOT forced onto it. Every position-arithmetic primitive that
//! doesn't change a finder's algorithmic shape already lives in [`common`]:
//! `lz_hash`, `lz_extend`, the unaligned loads, the `i16` sentinel/rebase pair,
//! and prefetch hints. `bt` and `lzfind` are ALSO two independently-evolved
//! binary-tree matchfinders (module doc pointer: `bt.rs`'s header notes they
//! are "siblings behind one trait; converging them is a gated follow-up, not
//! part of the structural move") — `bt` is libdeflate's single-best-match
//! near-optimal tree, `lzfind` is ECT's full-Pareto-frontier tree with a
//! DIFFERENT hash (portable CRC-mix, `HASH_LOG=15` vs `bt`'s `lz_hash`
//! order-16) and a different tree-node layout (`son: Vec<u32>` cyclic buffer
//! vs `bt`'s contiguous `[hash3|hash4|child]` slab). Converging THOSE two is
//! the real remaining opportunity but is explicitly out of scope for Stage D
//! (measured/gated separately per the header note) — this module doc records
//! it as the "gated follow-up," not silent unfinished work.
//!
//! Increment 1 landed the shared primitives in [`common`]; the hash-chain /
//! binary-tree / hash-table finders came in later increments. Stage D
//! (`docs/compressor-architecture.md` §5-D) is the measured retirement of the
//! FIFTH (legacy zopfli chain) finder that used to live in `parse::ultra::{hash,
//! cache, lz77::find_longest_match}` — see `parse::ultra::greedy_hc` and that
//! module's doc for the replacement and its score-gated verdict.

pub mod bt;
pub mod common;
pub mod hc;
/// ECT-class BT4 full-Pareto matchfinder — the crown engine's (`parse::ultra`)
/// seed matchfinder for the multi-seed iterated squeeze DP. Not used by any
/// other level tier. See the module doc above for why it stays a distinct
/// type from [`bt`] rather than merging behind one trait.
pub mod lzfind;
