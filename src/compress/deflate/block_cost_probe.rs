//! Per-block block-TYPE cost probe (`block-cost-probe`, DEFAULT OFF).
//!
//! THE BLOCKED QUESTION (2026-08-22, `probe/stored-blocks`): `examples/blockcensus`
//! shows we emit ZERO stored (BTYPE=00) blocks at L3 on all 23 corpus files while
//! `gzip -3` emits stored blocks on 4 of them (`sil40` 1, `movie.mp4` 146,
//! `data.csv.gz` 8, `tool.bin` 2). A zero count has two possible causes that a
//! census CANNOT distinguish:
//!
//!   (a) the stored candidate is costed on every block and loses on every block, or
//!   (b) the stored candidate is unreachable / mis-costed on the path those files take.
//!
//! Distinguishing them needs the LOSING candidate's cost, which no output stream
//! carries. This feature prints one line per emitted block:
//!
//!   BLOCKCOST len=<input bytes> stored=<bits> static=<bits> dyn=<bits> win=<s|f|d> site=<e|s>
//!
//! `dyn=-1` on the L0 (`emit_block_static_or_stored`) site, which never builds the
//! dynamic candidate. `site=e` is `emit_block`, `site=s` is
//! `emit_block_static_or_stored`.
//!
//! With the feature OFF every call site compiles to nothing — the same pattern as
//! `anatomy-counters` / `anatomy-wall`, and for the same reason (non-negotiable #3:
//! no env var, no production-path behaviour change). Verified inert: the feature-off
//! release build is byte-identical to `main` on all 115 cells of
//! 23 corpus files x levels 1,2,3,6,9 (sha256 of every stream).
//!
//! ⚠ READ THE LINES PER *DECISION*, NOT PER SHIPPED BLOCK. At the pick-min levels
//! (L1/L2/L4) several parse arms run and only the winner's bytes ship, so the line
//! count exceeds the shipped block count — `movie.mp4 -1` emits 396 lines for 203
//! shipped blocks. Every line is still a real block-type decision made on real
//! frequencies; only the shipped SUBSET is smaller.
//!
//! WHAT IT MEASURED (2026-08-22, 53,737 decisions over 23 files x L1/L2/L3/L6/L9,
//! `-p1`; aggregation scripts in the session scratchpad):
//!
//!   * The stored candidate is costed on 100% of decisions — `site=e` on every one.
//!     It is NOT dead code: it WINS 26 times at L1 (`movie.mp4` 22, `tool.bin` 4).
//!   * `site=s` fires ZERO times, at every level including `-0`: the shipped `-0`
//!     route is the single-shot stored path, not `emit_block_static_or_stored`, which
//!     emits no `BLOCKCOST` line on any corpus file at any level 0-9.
//!   * The pick is OPTIMAL. Summing `max(0, min(static,dynamic) - stored)` over every
//!     decision gives 0 bits at L2, L3, L6 and L9, and 3,223 bits at L1 — all of
//!     which the 26 stored wins already collect. There is no block anywhere on the
//!     corpus where a cheaper stored block was available and not taken.
//!   * Nothing is close, either. Smallest positive gap `stored - min(static,dynamic)`:
//!     L1 3 bits, L2 12, L3 628, L6 145, L9 595. Only 4 decisions in 53,737 sit
//!     within 8 bits, all at L1 on `movie.mp4`.
//!
//! WHY, MECHANICALLY. Stored costs 8.00061 bits/byte at a 65,535-byte grid. At L2-L9
//! our blocks run 240-300 KB (vs L1's 65,536) and the deeper parse pulls the emitted
//! rate BELOW 8 on every near-incompressible file — max emitted rate is 7.997
//! (`movie.mp4` L3), 7.981 (`data.csv.gz`), 7.987 (`photo.jpg`), 7.978
//! (`data.parquet`), 7.904 (`sil40`), 7.850 (`tool.bin`), 7.446
//! (`weights.safetensors`). A block can only go stored above 8.00061, so zero stored
//! blocks at L2+ is the correct answer, not a missed one.
//!
//! COST FORMULA vs libdeflate (`deflate_compress.c:1797-1801`,
//! `uncompressed_cost += (-(bitcount + 3) & 7) + 32 + 40*(DIV_ROUND_UP(len, UINT16_MAX) - 1) + 8*len`
//! on top of a 3-bit seed): `stored_block_bits` matches it term for term EXCEPT the
//! byte-alignment pad, which we hold at a constant 5 bits where libdeflate uses the
//! live `bitcount`. That is a +5/-2 bit error on the first sub-block only, and the gap
//! histogram above prices it: at most 2 decisions in 53,737 could flip, worth < 2 bytes
//! corpus-wide. Both implementations break ties toward stored (`<=`), and both compare
//! the same three candidates against the same denominator (each seeded with its own
//! 3 header bits, each including EOB).

/// Print one block's candidate costs. Compiles to nothing when the
/// `block-cost-probe` feature is off.
#[macro_export]
macro_rules! block_cost_probe {
    ($len:expr, $stored:expr, $static_:expr, $dyn_:expr, $win:expr, $site:expr) => {
        #[cfg(feature = "block-cost-probe")]
        {
            eprintln!(
                "BLOCKCOST len={} stored={} static={} dyn={} win={} site={}",
                $len, $stored, $static_, $dyn_, $win, $site
            );
        }
        #[cfg(not(feature = "block-cost-probe"))]
        {
            let _ = (&$len, &$stored, &$static_, &$dyn_);
        }
    };
}
