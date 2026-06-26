//! ASM-campaign rung (c) — the FULL contig clean fast loop as ONE
//! `core::arch::asm!` region (git history (campaign plan, removed) §2(c), §5 VAR_VIII salvage).
//!
//! Increment (a) (commit `b5c3f7c4`, REVERTED, recoverable) proved the
//! per-symbol asm seam costs ~1.4 cyc/crossing with zero latency recovered
//! (charter §8). Rung (c) is the only shape that amortizes the seam: the asm
//! owns the per-symbol hot path of `Block::decode_clean_into_contig`'s fast
//! loop (marker_inflate.rs:2466-2827) with the back-edge INSIDE the asm;
//! Rust is re-entered only at fast-loop boundaries (out-of-room, input-low,
//! EOB, rare/exceptional packets).
//!
//! ───────────────────────────────────────────────────────────────────────────
//! # EXIT-STATE CONTRACT (the hard part — enumerated BEFORE the asm exists)
//!
//! The asm region and the pure-Rust fast loop are bound forever by this
//! contract; the differential suite must hold it at every exit.
//!
//! ## Entry preconditions (guaranteed by the dispatcher + call site)
//!  E1. The bit cursor (`lb.pos`, `lb.bitbuf`, `lb.bitsleft`) sits exactly
//!      BEFORE a fresh, un-consumed litlen packet (the same invariant the
//!      Rust fast loop holds at its iteration top).
//!  E2. `lb.bitsleft` is a clean u32 in `[0, 63]` (no garbage above the low
//!      byte — every in-loop producer is `refill`'s `|56` or a budgeted
//!      subtract), AND `bitsleft >= 48` OR the state is bit-for-bit what the
//!      entry/bottom refill left (post-refill `< 48` implies
//!      `pos == data.len()`, the `decode_prefilled` precondition argument).
//!  E3. `dst = base + *pos`; `emitted == (dst - (base + pos_entry))` — the
//!      loop's `*pos += k; emitted += k;` lockstep makes ONE cursor
//!      sufficient inside the asm.
//!  E4. `ctx.out_lim = base + pos_entry + local_cap - FAST_OUT_SLOP`
//!      (`local_cap > FAST_OUT_SLOP` enforced at dispatch), so
//!      `dst < out_lim` ⇔ the Rust guard `emitted + FAST_OUT_SLOP <
//!      local_cap`. `ctx.in_lim = in_end - IN_MARGIN` (saturating), so
//!      `pos < in_lim` ⇒ the Rust guard `pos + FAST_IN_SLOP < in_end` AND
//!      every refill the asm can issue in one iteration stays on the fast
//!      8-byte path (see IN_MARGIN proof below).
//!  E5. The block's litlen LUT and the libdeflate-shape `DistTable` are
//!      built and immutable for the whole region (`block_huffman_luts_ready`,
//!      `dist_tbl.is_some()`); the asm only READS them.
//!  E6. No measurement knob is active (`contig_prof`, `slow_knob` spins,
//!      removal-oracle replay/nostore/record) and
//!      `track_backreferences == false` — `dispatch_allowed` is the single
//!      predicate (charter §3.5: hooks cannot fire inside asm; a silently
//!      unperturbable region would poison every causal instrument).
//!
//! ## Exit postconditions (EVERY exit code)
//!  X1. `(bitbuf, bitsleft, pos)` and `dst` written back are EXACTLY the
//!      values the pure-Rust loop holds at the same logical point. This is
//!      stronger than "same logical bit position": commit paths record
//!      `(pos, bitbuf, bitsleft)` verbatim, so REFILL PLACEMENT must be
//!      identical. Consequences the asm honors:
//!        * the chain arm's `decode()` backstop refill (`available < 32 →
//!          refill`) fires BEFORE the gate, INCLUDING for the carried
//!          (gate-failed) packet — bit-for-bit `Bits::refill` fast form;
//!        * the post-chain / pre-copy / bottom `< 48`-threshold refills are
//!          replicated at the same program points;
//!        * long litlen codes are resolved INSIDE the asm (a bail-to-Rust
//!          there would change LIT_CHAIN_MAX accounting and hence refill
//!          placement — see the chain-count analysis in the c2 commit);
//!        * the asm NEVER runs a slow refill: `IN_MARGIN` makes the fast
//!          `pos + 8 <= len` precondition structural (proof below).
//!  X2. The cursor sits BEFORE a fresh un-consumed packet — never
//!      mid-packet. Rare/exceptional packets are handed to Rust UN-consumed
//!      (EXIT_RECLASS) and re-executed there from the identical state, so
//!      error/EOB commits inherit byte-exact state by re-execution. The
//!      backref arms consume litlen bits (and the multi-with-trailing arm
//!      also stores the packed prefix and advances dst) before the dist
//!      decode can bail; they spill `(bitbuf, bitsleft, dst)` at arm entry
//!      and RESTORE all three before EXIT_RECLASS (`pos` untouched by then
//!      — the first refill in the arm comes after all rare checks; rolled-
//!      back packed-store bytes are X3 garbage above the exit dst).
//!  X3. Every byte in `[entry dst, exit dst)` is final, byte-identical
//!      output; bytes at/above the exit `dst` may hold speculative garbage
//!      exactly like the Rust loop's packed-store overshoot (never read
//!      back: back-ref sources are `< dst`).
//!  X4. `self.*` (`at_end_of_block`, `decoded_bytes`, `backreferences`),
//!      `orec`, prof counters: NEVER touched by the asm. Rust owns every
//!      commit. Effect/coverage counters are wrapper-side, gated on
//!      `stats_enabled()`, accumulated from the emitted delta.
//!  X5. `bitsleft >= 48` at exit, OR the cursor is unchanged since entry
//!      (E2 carries through) — so the caller may re-derive the carried
//!      packet with `decode_prefilled` (purity: a litlen decode reads at
//!      most 21 bits < 32/48, and refills are append-only, so re-decoding
//!      after the asm's refills equals the Rust loop's carried `pre`).
//!  X6. `exit dst == entry dst` ⇔ the ENTIRE state is unchanged (no bits
//!      consumed) — the caller uses this to skip the `pre` re-derivation.
//!
//! ## Exit codes
//!  * `EXIT_BOUNDARY` (1): an asm guard failed (`dst >= out_lim` or
//!    `pos >= in_lim`). Both are monotone within one call, so the caller
//!    disables re-entry (`asm_on = false`) and the Rust loop finishes the
//!    tail under its own (looser) guards.
//!  * `EXIT_RECLASS` (0, and reason-tagged 2/3/4 — semantically
//!    identical): the next packet needs Rust — EOB (lone or trailing,
//!    tag 2), `bit_count == 0` (invalid), symbol > MAX_LIT_LEN_SYM (lone
//!    plain, trailing tag 3), a backref (lone OR multi-with-trailing)
//!    whose dist decode hit a subtable pointer (rare; charter §5-R1) or
//!    `raw == 0` / `distance == 0` / `distance > 32768` /
//!    `distance > *pos` (restored incl. dst, X2; tag 4). The caller
//!    leaves `asm_on = true`; the Rust loop handles exactly that packet and
//!    the next `continue 'fast` re-enters the asm.
//!
//! ## IN_MARGIN proof (no slow refill inside the asm)
//! After the branchless-classify rewrite the speculative literal chain is
//! GONE, so one asm iteration issues at most ONE refill (the pure-literal
//! bottom — now UNCONDITIONAL, igzip-style every-iteration, the software-
//! pipelined `loop_block` shape; or the backref pre-copy `< 48`). Each fast
//! refill advances
//! `pos` by ≤ 7 and reads 8 bytes, so the deepest read inside an iteration
//! touches `pos_top + 7 + 8 = pos_top + 15 <= pos_top + IN_MARGIN - 1 <
//! in_end` for `IN_MARGIN = 40` (`pos_top < in_lim = in_end - 40`). Hence
//! `pos + 8 <= len` at every asm refill — the Rust fast form, bit-for-bit,
//! and the `> 64` underflow check is dead (E2: bitsleft ≤ 63 always).
//! IN_MARGIN stays 40 (the pre-rewrite value): the contract only got
//! looser, so no caller change is needed.
//!
//! ## Store-shape equivalence (branchless rewrite)
//! Pure-literal pack (lone OR multi, unified): ONE 8-byte speculative store
//! of `entry & 0x00FF_FFFF` (up to 3 packed literal bytes), advance by
//! `sym_count`. This is the libdeflate/igzip fastloop shape and folds the
//! lone-literal case into the multi path so the per-symbol classify needs
//! no `sym_count` branch (the old 1-byte lone store / advisor-Q3 special
//! case is retired — the extra store-buffer width is dominated by the
//! branch-misprediction it removes). Bytes above `dst` are X3 overshoot,
//! never read back. Backref copy (igzip-full-rewrite): a faithful 16-byte
//! MOVDQU port of igzip `large_byte_copy` (603-612) + `small_byte_copy`
//! (614-627) with `COPY_SIZE == 16` — load 16B from src, `distance >=
//! min(16, length)` ⇒ large (one-or-more MOVDQU, src += 16/iter); else
//! grow the period by store+double-distance until >= COPY_SIZE then large.
//! Output in `[dst, dst+length)` is byte-identical to the scalar
//! `emit_backref_contig` walk (overshoot above dst is X3). `length > 240`
//! (mean back-ref len 6.3, so ~never) keeps the proven scalar P3.4 shape
//! (dist ≥ 8 burst-5 + stride-8 words; dist == 1 RLE broadcast word; 2..=7
//! stride-dist words; `length > 40` prefetch) so the write extent stays
//! inside the MAX_RUN_LENGTH+8 ring envelope.
//!
//! ## Dispatch policy (charter §4)
//! Cargo feature `asm-kernel`, x86_64-only call sites; pure-Rust loop ALWAYS
//! compiled and reachable. Runtime: BMI2 detect (`shrx`/`shlx`/`bzhi`) +
//! `GZIPPY_ASM_KERNEL=0` kill-switch (OnceLock, one predictable branch per
//! contig call). DEFAULT-ON on x86_64: `pure-rust-inflate` (the shipped
//! native + isal feature set) pulls in `asm-kernel`, so on a BMI2 host this
//! region IS the production clean fast loop.

#![allow(dead_code)]

#[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
use crate::decompress::inflate::consume_first_decode::Bits;

/// Stricter input guard margin (bytes) — see IN_MARGIN proof in the module
/// doc. The Rust loop's own guard is `FAST_IN_SLOP = 8`.
pub const IN_MARGIN: usize = 40;

/// Exit codes (module doc "Exit codes"). The caller's contract tests ONLY
/// `== EXIT_BOUNDARY`; every other value is a RECLASS. Values >= 2 are
/// reason-tagged RECLASSes for the effect/coverage instrument (decide.sh
/// EFFECT predicate + the F-c asm_frac diagnosis) — semantically identical
/// to `EXIT_RECLASS`.
pub const EXIT_RECLASS: u64 = 0; // invalid / oversize / lone EOB-class top bail
pub const EXIT_BOUNDARY: u64 = 1;
pub const EXIT_RECLASS_EOB: u64 = 2; // lone EOB packet
pub const EXIT_RECLASS_MULTI_TRAIL: u64 = 3; // multi-literal packet with trailing non-literal
pub const EXIT_RECLASS_DIST: u64 = 4; // backref arm dist-side bail (subtable/raw0/validity)

// NIGHT15 DECOMPOSITION INSTRUMENT (byte-transparent; the caller collapses every
// non-BOUNDARY exit to EXIT_RECLASS, so these sub-tags change ONLY which stat
// counter increments — Rust handling + the 85: re-read reconstruct are identical
// for all of them). They split the lumped EXIT_RECLASS_DIST (reclass_dist=6410 on
// silesia, NIGHT14) into its four mutually-exclusive causes to TEST the task's
// load-bearing premise that subtable-dist DOMINATES (vs the out-of-scope
// window-absent marker case `src < out_base`, which keeps the D-1 anchor alive).
pub const EXIT_RECLASS_RAW0: u64 = 5; // dist entry raw==0 (hole / code 30,31 — invalid)
pub const EXIT_RECLASS_SUBTABLE: u64 = 6; // dist subtable pointer (the removable bucket)
pub const EXIT_RECLASS_BADDIST: u64 = 7; // distance==0 or >MAX_WINDOW (invalid)
pub const EXIT_RECLASS_MARKER: u64 = 8; // src<out_base (window-absent backref — OUT OF SCOPE)

/// The single dispatch predicate for the knob-exclusion rule (charter §3.5)
/// — pure so it is unit-testable. `enabled()` (env + CPU) is checked
/// separately by the call site; this covers the per-call state.
#[inline(always)]
#[allow(clippy::too_many_arguments)] // mirrors the knob list 1:1 (charter §3.5)
pub fn dispatch_allowed(
    prof_on: bool,
    oracle_nostore: bool,
    orec_active: bool,
    dec_spin: u64,
    st_spin: u64,
    track_backrefs: bool,
    local_cap: usize,
    fast_out_slop: usize,
) -> bool {
    !prof_on
        && !oracle_nostore
        && !orec_active
        && dec_spin == 0
        && st_spin == 0
        && !track_backrefs
        && local_cap > fast_out_slop
}

/// Loop-invariant context + INLINE Huffman tables for the asm region —
/// ELEMENT A: igzip's single-state-base register discipline
/// (`igzip_decode_block_stateless.asm`: ONE `state` base in r11, tables via
/// struct offsets `[state+_lit_huff_code+...]` / `[state+_dist_huff_code+...]`;
/// struct `igzip_lib.h:515-524`). The litlen LUT and the dist table are
/// CO-LOCATED INLINE in this ONE boxed struct so the asm addresses them off
/// the single `ctx` base — `[{ctx}+ASM_LIT_SHORT_OFF+idx*4]` and
/// `[{ctx}+ASM_DIST_OFF+idx*4]` — instead of two extra pinned base registers
/// ({short_tbl}, {dtbl}). Those 2 freed GP registers carry the iteration-top
/// `p0`/`d0` un-consume anchor IN-REGISTER, deleting the 2 per-iteration anchor
/// STORES (former `[ctx+56]`/`[ctx+64]`; D-1 ledger).
///
/// ZERO-COPY: the tables are BUILT IN PLACE inside this boxed struct (held in
/// the decoder `self`, stable address) — a per-region copy (~7465 blocks ×
/// ~27 KiB on silesia) would negate the win. `in_ptr` stays a separate
/// register (it indexes the compressed input window `[in_ptr+pos]`, not state
/// — igzip likewise keeps `next_in` in its own register).
///
/// `#[repr(C)]`, header first at the SAME low offsets the asm guards read
/// (`[ctx+8]`/`[ctx+16]`/`[ctx+24]`), then the inline tables. Offsets are
/// compile-asserted below; the table offsets are passed to the asm as `const`
/// operands computed from `offset_of!` (layout-driven, not hand-numbered).
#[repr(C)]
pub struct AsmState {
    /// +0: `bits.data.as_ptr()` — refill word loads.
    pub in_ptr: u64,
    /// +8: `in_end - IN_MARGIN` (saturating) — top guard `pos < in_lim`.
    pub in_lim: u64,
    /// +16: `base + pos_entry + local_cap - FAST_OUT_SLOP` — top guard
    /// `dst < out_lim` (E4: exactly the Rust out guard).
    pub out_lim: u64,
    /// +24: `base` as integer — c3 backref validity (`src >= out_base` ⇔
    /// `distance <= *pos`).
    pub out_base: u64,
    /// +32: litlen LUT, INLINE (short_code_lookup `[u32;4096]` at +32 then
    /// long_code_lookup `[u16;1264]`). Built in place by
    /// `LutLitLenCode::rebuild_from`; read by the asm via the single base.
    pub lut_litlen: super::lut_huffman::LutLitLenCode,
    /// dist `DistTable`, INLINE (`entries: [DistEntry;DIST_CAP]` first). Built
    /// in place by `DistTable::rebuild`; read by the asm via the single base.
    pub dist: crate::decompress::inflate::libdeflate_entry::DistTable,
}

/// Asm `const`-operand displacements (single-base addressing). Layout-driven
/// via nested `offset_of!` — never hand-numbered.
pub const ASM_LIT_SHORT_OFF: usize =
    std::mem::offset_of!(AsmState, lut_litlen.table.short_code_lookup);
pub const ASM_LIT_LONG_OFF: usize =
    std::mem::offset_of!(AsmState, lut_litlen.table.long_code_lookup);
pub const ASM_DIST_OFF: usize = std::mem::offset_of!(AsmState, dist.entries);

const _: () = assert!(std::mem::offset_of!(AsmState, in_ptr) == 0);
const _: () = assert!(std::mem::offset_of!(AsmState, in_lim) == 8);
const _: () = assert!(std::mem::offset_of!(AsmState, out_lim) == 16);
const _: () = assert!(std::mem::offset_of!(AsmState, out_base) == 24);
// Header is exactly 32 bytes; the inline litlen short table starts right after.
const _: () = assert!(ASM_LIT_SHORT_OFF == 32);
const _: () = assert!(ASM_LIT_LONG_OFF == 32 + 4096 * 4);
// Dist entries land after the whole litlen LUT (short 16384 + long 2528) plus
// LutLitLenCode's two Box fields + valid (padded). Asserted to pin layout.
const _: () = assert!(ASM_DIST_OFF > ASM_LIT_LONG_OFF);

/// LUT-layout constants mirrored from `lut_huffman.rs` / `libdeflate_entry.rs`
/// (compile-checked so drift between the asm immediates and the table
/// builders is impossible).
const _: () = assert!(super::lut_huffman::LARGE_FLAG_BIT == 1 << 25);
const _: () = assert!(super::lut_huffman::LARGE_SHORT_CODE_LEN_OFFSET == 28);
const _: () = assert!(super::lut_huffman::LARGE_SYM_COUNT_OFFSET == 26);
const _: () = assert!(super::lut_huffman::LARGE_SYM_COUNT_MASK == 3);
const _: () = assert!(super::lut_huffman::LARGE_SHORT_SYM_MASK == 0x1FF_FFFF);
const _: () = assert!(super::lut_huffman::LARGE_SHORT_MAX_LEN_OFFSET == 26);
const _: () = assert!(super::lut_huffman::LARGE_LONG_CODE_LEN_OFFSET == 10);
const _: () = assert!(super::lut_huffman::LARGE_LONG_SYM_MASK == 0x3FF);
const _: () = assert!(super::lut_huffman::ISAL_DECODE_LONG_BITS == 12);
const _: () = assert!(super::lut_huffman::MAX_LIT_LEN_SYM == 512);
const _: () =
    assert!(crate::decompress::inflate::libdeflate_entry::HUFFDEC_SUBTABLE_POINTER == 0x4000);
// The x86 asm kernel hardcodes 9-bit dist-table indexing ([{ctx}+DIST_OFF+idx*4]).
// On aarch64 the dist table is 8 (engine-A convergence to libdeflate OFFSET_TABLEBITS),
// and the asm kernel does not run there, so this invariant is x86-only.
#[cfg(target_arch = "x86_64")]
const _: () = assert!(crate::decompress::inflate::libdeflate_entry::DistTable::TABLE_BITS == 9);

#[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
mod imp {
    use super::{AsmState, Bits, ASM_DIST_OFF, ASM_LIT_LONG_OFF, ASM_LIT_SHORT_OFF};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    /// Effect/coverage counters (decide.sh EFFECT predicate + the asm_frac
    /// validity precondition F-c requires). Wrapper-side only (X4), gated on
    /// `stats_enabled()` so the measured hot path carries zero atomics.
    pub static KERN_ENTRIES: AtomicU64 = AtomicU64::new(0);
    pub static KERN_EXIT_BOUNDARY: AtomicU64 = AtomicU64::new(0);
    pub static KERN_EXIT_RECLASS: AtomicU64 = AtomicU64::new(0);
    pub static KERN_ASM_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static KERN_RECLASS_EOB: AtomicU64 = AtomicU64::new(0);
    pub static KERN_RECLASS_MULTI_TRAIL: AtomicU64 = AtomicU64::new(0);
    pub static KERN_RECLASS_DIST: AtomicU64 = AtomicU64::new(0);
    // NIGHT15 decomposition of KERN_RECLASS_DIST (byte-transparent stat split).
    pub static KERN_RECLASS_RAW0: AtomicU64 = AtomicU64::new(0);
    pub static KERN_RECLASS_SUBTABLE: AtomicU64 = AtomicU64::new(0);
    pub static KERN_RECLASS_BADDIST: AtomicU64 = AtomicU64::new(0);
    pub static KERN_RECLASS_MARKER: AtomicU64 = AtomicU64::new(0);

    /// Flip-precondition 3 (the permanent ON-vs-OFF fuzz net): in-process
    /// dispatch override. The env kill-switch is a process-wide OnceLock
    /// read, so a same-binary ON/OFF differential cannot toggle it; the
    /// fuzz test toggles HERE instead. 0 = no override (production
    /// semantics), 1 = force-disabled, 2 = force-enabled (still requires
    /// BMI2 — the asm cannot execute without it). Compiled ONLY into the
    /// lib test harness (`cfg(test)`); production binaries carry no trace.
    #[cfg(test)]
    pub static TEST_FORCE: AtomicU64 = AtomicU64::new(0);
    /// Engagement counter for the fuzz net (cfg(test) only): incremented at
    /// every `run_contig` entry, independent of `GZIPPY_ASM_STATS`, so the
    /// ON arm can PROVE the asm actually executed (effect-verified, not
    /// assumed) and the OFF arm can prove it did not.
    #[cfg(test)]
    pub static TEST_RUN_CONTIG_CALLS: AtomicU64 = AtomicU64::new(0);

    /// Runtime dispatch: ON when compiled in, unless `GZIPPY_ASM_KERNEL=0`
    /// (kill-switch) or the CPU lacks BMI2 (`shrx`/`shlx`/`bzhi`).
    pub fn enabled() -> bool {
        #[cfg(test)]
        match TEST_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return std::arch::is_x86_feature_detected!("bmi2"),
            _ => {}
        }
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            let killed = std::env::var("GZIPPY_ASM_KERNEL").is_ok_and(|v| v == "0");
            !killed && std::arch::is_x86_feature_detected!("bmi2")
        })
    }

    /// `GZIPPY_ASM_STATS=1` — effect-verification counters.
    pub fn stats_enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("GZIPPY_ASM_STATS").is_ok_and(|v| v == "1"))
    }

    /// The rung-(c) kernel region.
    ///
    /// BRANCHLESS-CLASSIFY (this commit): the LITERAL hot path is a FLAT
    /// igzip/libdeflate-style dispatch — iteration-top guards, raw short-LUT
    /// gather, long-code resolution (`20:`), then ONE data-dependent branch
    /// (`cmp 255` on the uniformly-extracted TRAILING packed symbol, the
    /// analog of igzip's `cmp 0x100`). Lone AND multi literals are unified
    /// into one pure-literal path (single 8-byte packed store, advance by
    /// `sym_count`); the old multi-stage `sym_count` cascade and the
    /// speculative literal chain that RE-RAN the classify for the 2nd/3rd
    /// literal are GONE — one packet per top-iteration, one bottom `< 48`
    /// refill, and the back-edge.
    ///
    /// STAGE c3 (this commit): the BACKREF arm lives inside too (`50:`) —
    /// a lone length code (short or long path) consumes the litlen packet,
    /// SPECULATIVELY PRELOADS the dist short entry from the post-consume
    /// bitbuf at the consume point (igzip `decode_next_dist` 457 + load
    /// 550-552: the dist code immediately follows the litlen, so its L1
    /// gather is issued early and overlaps the cnt-branch / prefix store /
    /// length lea — `dpre`), then at `58:` decodes the distance from the
    /// libdeflate-shape `DistTable` (the preloaded entry + in-register decode:
    /// `consume_entry` == `shrx` by the entry low byte, `decode_distance` ==
    /// `bzhi(saved, total_bits) >> cw_bits`
    /// + base), validates (raw==0 / subtable / dist==0 / >32768 / > *pos),
    /// runs the X1 pre-copy `< 48` refill + carried-packet preload (the
    /// P3.5 c1 hoist), and copies with the P3.4 `emit_backref_contig`
    /// shape transliterated verbatim (dist>=8 burst-5 + stride-8 words;
    /// dist==1 RLE broadcast word; 2..=7 stride-dist words; `length > 40`
    /// prefetch). The X2 spill/restore (`80:`) makes every rare bail
    /// pre-consume. Remaining `EXIT_RECLASS` packets: lone EOB, invalid,
    /// oversize (>512), dist-subtable (rare; charter §5-R1), dist validity
    /// errors (Rust re-executes to the identical error commit), and
    /// multi-with-trailing (builder-impossible, kept for defense) — all at
    /// an iteration top, cursor before a fresh un-consumed packet.
    /// Bit-budget proof (E2 stays clean inside the arm): an iteration top
    /// holds >= 48 bits, the litlen packet consumes <= 21, a main-table
    /// dist entry consumes <= 9 + 13 = 22 <= the >= 27 remaining — no
    /// underflow; the subtable case (the only one that could) exits to
    /// Rust.
    ///
    /// Asm↔Rust line map (byte-exactness audit; Rust mirror =
    /// `run_contig_ref_biased`, pinned asm==ref by the c2/c3 differentials):
    ///   top guards        ↔ ref `*dst >= out_lim || pos >= in_lim`
    ///   prologue/carried gather + `20:` long resolve
    ///                     ↔ lut_huffman.rs `decode_prefilled`
    ///   late discriminator (`cmp {t5}, 256` on the extracted trailing →
    ///                      `2b`/non-literal arm) ↔ ref `trailing > 255`
    ///                      (igzip loop_block 555-556 runtime classify; no
    ///                      build-time class flag — gz's dead bit-24
    ///                      TRAILING_NONLIT_FLAG was removed to match igzip's
    ///                      table build. The c2/c3 differential pins the same
    ///                      contract on the runtime-extracted trailing.)
    ///   pure-literal pack (lone+multi unified) ↔ ref 8-byte packed store +
    ///                       `*dst += cnt` (one packet/iter, NO chain)
    ///   bottom `6:`/`63:` ↔ ref `< 48` threshold refill
    ///   non-literal arm `50:` (cnt-split lone/multi-trailing; dist entry
    ///                       PRELOADED post-consume — igzip decode_next_dist)
    ///                     ↔ ref `trailing > 255` arm (EOB/oversize gates,
    ///                       cnt>=2 literal-prefix store, post-consume
    ///                       `dist.lookup` (== the asm preload), validity,
    ///                       X2 restore on bail)
    ///   copy `71:`-`74:`  ↔ igzip large/small_byte_copy (16-byte MOVDQU);
    ///                       `70:`-`59:` scalar fallback (length > 240)
    ///                       ↔ marker_inflate.rs `emit_backref_contig`
    ///   REFILL sequence   ↔ consume_first_decode.rs `Bits::refill` fast
    ///                       form (`pos + 8 <= len` structural via
    ///                       IN_MARGIN; `>64` underflow check dead via E2)
    ///
    /// # Safety
    /// Caller upholds E1-E6 (module doc). The asm reads `ctx`, the litlen +
    /// dist tables and the input window; it writes only `[entry dst,
    /// exit dst + copy-overshoot)` inside the reserved out_room envelope
    /// (E4; the copy overshoot <= max(40, length+7) <= 265 < the
    /// MAX_RUN_LENGTH + 8 = 266 reservation, the P3.4 envelope) and the
    /// pinned registers.
    ///
    /// `#[inline(never)]`: c2's frozen A/B measured OFF-vs-base +36 ms —
    /// inlining the (cold-when-disabled) region into the hot Rust loop
    /// taxes its layout even when killed. One CALL per region run is
    /// amortized by the in-asm back-edge.
    #[inline(never)]
    pub unsafe fn run_contig(ctx: &AsmState, lb: &mut Bits<'_>, dst: *mut u8) -> (u64, *mut u8) {
        #[cfg(test)]
        TEST_RUN_CONTIG_CALLS.fetch_add(1, Ordering::Relaxed);
        let mut bitbuf = lb.bitbuf;
        let mut bitsleft: u64 = lb.bitsleft as u64;
        let mut pos: u64 = lb.pos as u64;
        let mut dst_c = dst;
        let ret: u64;
        unsafe {
            core::arch::asm!(
                // ── prologue: speculative preload of the CURRENT packet's
                //    short entry (no consume — harmless table read, exactly the
                //    bits `decode_prefilled` reads). Carried in {t1} = igzip
                //    `next_sym`, the preloaded symbol re-loaded at the bottom of
                //    every iteration (igzip 502 / 540 / 580-582).
                "mov {t1:e}, {bitbuf:e}",
                "and {t1:e}, 0xFFF",
                "mov {t1:e}, dword ptr [{ctx} + {t1}*4 + {lit_off}]",
                // ── NIGHT36: HOIST the resumable BOUNDARY set `ret=1` OUT of the
                //    hot literal loop top. On the literal path `{ret}` is written
                //    but never read until a guard-exit (`9f`); it is CLOBBERED only
                //    in the two backref arms (`74:` lea/`70:` mov reuse {ret} as the
                //    copy cursor — 15-GP-operand ceiling). So set it ONCE here, and
                //    re-set it ONLY on the backref completion paths (74:/59:) before
                //    they loop to `2b`. Removes 1 instr/iter from the hot literal
                //    path (the ~0.20 instr/B `ret=1` bucket, NIGHT34 ledger);
                //    long-literal paths (64:/20:/21:) never touch {ret}. {ret} is
                //    NOT loop-carried (dedicated `out(reg)`, dead on the literal
                //    path) so this adds NO live-range across the refill (the NIGHT32
                //    hazard) and lengthens NO loop-carried chain. Byte-exact: the
                //    {ret} VALUE at every exit is unchanged (boundary=1, reclass tags
                //    set in their own arms) ⇒ ref model needs no change.
                "mov {ret}, 1",                       // speculative BOUNDARY (hoisted, once)
                // ── iteration top: guards (E4) ──────────────────────────
                "2:",
                "cmp {dst}, qword ptr [{ctx} + 16]",  // dst vs out_lim
                "jae 9f",
                "cmp {pos}, qword ptr [{ctx} + 8]",   // pos vs in_lim
                "jae 9f",
                // ── NIGHT11 MINIMAL UN-CONSUME ANCHOR (the night9 4-store X2
                //    snapshot is DELETED — see DIVERGENCE LEDGER). The late-
                //    discriminator body below CONSUMES + REFILLS + speculatively
                //    STORES before it knows the packet is non-literal, so a rare
                //    bail must hand Rust the packet UN-consumed (X2). Instead of
                //    snapshotting all four of (bitbuf,bitsleft,pos,dst), save the
                //    TWO values a bail cannot otherwise recover:
                //      [ctx+56] = p0 = pos*8 - bitsleft  (iteration-top BIT
                //        position). REFILL-INVARIANT (Bits::refill's |56 + pos
                //        advance preserve pos*8-bitsleft), so it is the un-
                //        consumed packet start regardless of the body's
                //        consume/refill. A bail re-reads the whole cursor from it
                //        (the consumed low bits are shifted out — unrecoverable
                //        from post-consume registers).
                //      {d0} = dst (iteration-top dst; un-consume target).
                //    ELEMENT A: with {short_tbl}+{dtbl} folded into the single
                //    {ctx} base (inline tables), the 2 freed GP registers carry
                //    p0/d0 IN-REGISTER — the 2 per-iteration anchor STORES
                //    (former `mov [ctx+56],p0` / `mov [ctx+64],dst`) are DELETED
                //    (igzip stateless single-base; D-1 ledger). The bails read
                //    p0/d0 from registers at 85:.
                // ── NIGHT40: HOIST the {d0} dst anchor OFF the hot literal path
                //    (the NIGHT36 ret=1 pattern applied to the dst un-consume
                //    target). {d0} is read ONLY on a non-literal/long bail (85:),
                //    never on the literal back-edge (`jb 2b`), so the per-iteration
                //    `mov {d0},{dst}` on the HOT literal path is pure waste. It is
                //    moved to the two cold points where a bail can first occur,
                //    reconstructed as `dst - cnt` (= iteration-top dst, since the
                //    speculative store advanced dst by cnt): the short non-literal
                //    fall-through (post-discriminator, `{t3}`=cnt still live —
                //    captured BEFORE the dist decode clobbers `{t3}` at saved_bitbuf
                //    `94:`) and the long lone non-literal `21:` (dst un-advanced
                //    there, so {d0}={dst}). Byte-exact: the {d0} VALUE at every bail
                //    is identical to the old top capture (iteration-top dst), so the
                //    ref model (`run_contig_ref_biased`, d0=*dst at top) needs NO
                //    change. The {p0} bit anchor (lea+sub) STAYS on the hot path:
                //    moving it would need `bc` reconstructed at the bail, but `bc`
                //    (`{t2}`) is DESTROYED by the refill (`6:` reuses {t2}) and
                //    keeping it live across the refill is the NIGHT32 trap (register
                //    live-range / schedule hazard → cyc/B regress). cnt survives the
                //    refill ({t3} untouched), bc does not — that asymmetry is why d0
                //    is hoistable and p0 is not.
                "lea {p0}, [{pos} * 8]",
                "sub {p0}, {bitsleft}",
                // ── decode_next_lit_len on the preloaded entry {t1} (igzip
                //    322-372 / loop_block 515): extract bit_count + sym_count.
                //    LARGE_FLAG → long table (cold); bc==0 → invalid (cold).
                "test {t1:e}, 0x2000000",             // LARGE_FLAG_BIT → long (cold)
                "jnz 20f",
                "mov {t2:e}, {t1:e}",
                "shr {t2:e}, 28",                     // bc = bit_count
                "jz 86f",                             // invalid (bc==0) → Rust (UN-consumed, no re-read)
                "mov {t3:e}, {t1:e}",
                "shr {t3:e}, 26",
                "and {t3:e}, 3",                      // cnt = sym_count (1/2/3)
                // ── STEP-2(b) MASK-ONCE (igzip decode_next_lit_len macro 341
                //    `and next_sym, LARGE_SHORT_SYM_MASK`): clear bc/cnt/
                //    LARGE_FLAG (bits 25-31) from {t1} IN PLACE — all already
                //    extracted (bc→{t2} @449, cnt→{t3} @451-453, LARGE_FLAG
                //    tested @446). The single masked {t1} (= the bits-0..24
                //    packed-symbol field) is then REUSED for BOTH the trailing
                //    shrx (igzip 521) AND the speculative store (igzip 518) —
                //    deleting the two scratch copies (`mov {t5},{t1}` +
                //    `mov {t4},{t1}`) and the per-use store mask (`and 0xFFFFFF`).
                //    {t1} is dead after the store until reloaded at the bottom
                //    preload (497-499); the non-literal arm re-derives its needs
                //    from {t5}/{t3}, so the in-place mask is safe. (`and r32`
                //    zero-extends, so {t1}'s high 32 bits are 0 for the store.)
                "and {t1:e}, 0x1FFFFFF",              // LARGE_SHORT_SYM_MASK (ONCE)
                // CONSUME-REORDER (candidate #1, igzip 370-371 order): consume the
                // litlen bits HERE (right after mask), BEFORE the trailing extract +
                // spec store, matching igzip decode_next_lit_len (consume before the
                // store 518). Byte-exact: nothing between mask and the old consume
                // site reads {bitbuf} (trailing uses {t1}; store uses {t1}/{dst}/{t3});
                // {t2}=bc is live from classify and dead after. Lets the loop-carried
                // bitbuf->index->refill->entry-load recurrence start earlier.
                // Schedule-only; ref model (functional) unchanged.
                "shrx {bitbuf}, {bitbuf}, {t2}",      // consume litlen (moved up)
                "sub {bitsleft}, {t2}",
                // ── TRAILING EXTRACT (igzip 520-521), UNCONDITIONAL every
                //    iteration: next_sym2 = (masked >> 8*(cnt-1)) & 0xFFFF. The
                //    post-shrx & 0xFFFF was DELETED (NIGHT34, igzip _04 0x38d25
                //    convergence): the mask-once `and 0x1FFFFFF` clears class
                //    bits 25-31 and the LUT build zero-fills unused symbol slots,
                //    so bits above the trailing are already 0 for every cnt. {t4}
                //    is the scratch shift.
                "lea {t4:e}, [{t3:e}*8 - 8]",         // shift = 8*(cnt-1)
                "shrx {t5}, {t1}, {t4}",              // {t5} = trailing (clean; NIGHT34 igzip _04 0x38d25 converge: & 0xFFFF removed)
                // ── SPECULATIVE STORE + advance by cnt (igzip 518-519),
                //    UNCONDITIONAL: store the masked {t1} (bits 0..24) directly
                //    and advance dst by the full sym_count assuming a pure-
                //    literal pack. A trailing length over-advances dst by 1 (its
                //    low byte stored as garbage); `decode_len_dist` fixes it with
                //    one `dec dst` so the copy overwrites it (igzip's symmetric
                //    next_out `lea +repeat_length-1`). {t1} byte 3 (bit24) is X3
                //    overshoot — dst advances by cnt ≤ 3 so byte 3 is never a
                //    final output byte (identical overshoot semantics to the old
                //    `and 0xFFFFFF` store, which only differed in byte 3).
                "mov qword ptr [{dst}], {t1}",        // speculative 8-byte store
                "add {dst}, {t3}",                    // advance by cnt
                // ── SPLIT REFILL + SOFTWARE-PIPELINED PRELOAD CADENCE (NIGHT19:
                //    converge run_contig's per-iteration SCHEDULE on igzip's EXACT
                //    loop_block straight-line order 524-552 — element F of the
                //    KERNEL-CONVERGENCE map, the remaining divergence in the
                //    software-pipelined shape that owns the loop-carried
                //    dependency chain / hot-loop IPC, NIGHT18). Three coupled
                //    reorders vs the prior contiguous `6:` block, all byte-exact
                //    (refill is append-only; ref model is functional, unchanged):
                //      (1) igzip 524-525: extract the NEXT litlen INDEX from the
                //          POST-CONSUME bitbuf BEFORE the refill OR, into a
                //          SEPARATE reg {t4}. The OR sets only bits >= bitsleft,
                //          and post-consume bitsleft >= 48-21 = 27 > 12, so the
                //          low-12 index is identical pre-/post-OR; extracting it
                //          early lets the index compute overlap the OR and widens
                //          the index→load distance.
                //      (2) igzip 528-530: refill part A (OR new high bits).
                //      (3) igzip 540: load the litlen ENTRY from the early index,
                //          AFTER the OR (load-use distance), into {t1}.
                //      (4) igzip 543-547: refill part B (next_in/len advance)
                //          DEFERRED past the litlen load so the loop-carried `pos`
                //          update overlaps that load-use latency — the crux of
                //          igzip's pipelining (gz's prior contiguous form
                //          serialized pos-advance ahead of the preload).
                //    `or bitsleft,56` exact for bitsleft∈[0,63]; tops bitsleft to
                //    ≥56 before the discriminator so the backref dist decode
                //    (≤28 bits) is in budget. {t2} free post-consume; {t4} scratch
                //    (dead after the trailing shrx @491).
                "6:",
                // (1) igzip 524-525 EXACT micro-schedule: `mov tmp3, CONST` /
                //    `and tmp3, read_in`. The mask CONSTANT goes in {t4} FIRST
                //    (no source dependency → OFF the loop-carried critical path);
                //    the single `and {t4}, {bitbuf}` is then the ONLY critical-path
                //    op feeding the entry load. The prior `mov {t4},{bitbuf}` copied
                //    the just-consumed (loop-carried) bitbuf — an EXTRA dependent op
                //    on the recurrence shrx→COPY→and→load. AND is commutative so the
                //    index value (bitbuf & 0xFFF) is byte-identical; pure schedule
                //    convergence (NIGHT24 located the divergence: objdump c33b5
                //    `mov %r8d,%r13d` vs igzip's off-path const-mov).
                "mov {t4:e}, 0xFFF",                  // (1) igzip 524: mask CONST (off critical path)
                "and {t4:e}, {bitbuf:e}",             //     igzip 525: index = bitbuf & 0xFFF (single crit op)
                "mov {t2}, qword ptr [{in_ptr} + {pos}]", // (2) igzip 528-530: OR
                "shlx {t2}, {t2}, {bitsleft}",
                "or {bitbuf}, {t2}",
                "mov {t1:e}, dword ptr [{ctx} + {t4}*4 + {lit_off}]", // (3) igzip 540: entry load
                "mov {t4:e}, 63",                     // (4) igzip 543-547: ptr/len advance DEFERRED
                "sub {t4}, {bitsleft}",
                "shr {t4}, 3",
                "add {pos}, {t4}",
                "or {bitsleft}, 56",
                // ── SPECULATIVE preload of the NEXT dist entry, EVERY iteration
                //    (igzip 550-552): the dist code follows the just-consumed
                //    litlen, so its short entry is at the low 9
                //    (DistTable::TABLE_BITS) bits of the refilled bitbuf. On a
                //    literal this load is discarded; on a length the backref arm
                //    finds it already in flight (load latency hidden). PURE table
                //    read — byte-exact (the ref models the same post-consume
                //    `dist.lookup`).
                // igzip 550-551 EXACT: `mov next_bits2, CONST` / `and next_bits2,
                //    read_in` — mask CONST off-path, single `and` critical op (same
                //    convergence as the litlen index above; byte-identical, AND
                //    commutative).
                "mov {dpre:e}, 0x1FF",                // igzip 550: mask CONST (off critical path)
                "and {dpre:e}, {bitbuf:e}",           // igzip 551: dist index = bitbuf & 0x1FF
                "mov {dpre:e}, dword ptr [{ctx} + {dpre}*4 + {dist_off}]",
                // ── LATE DISCRIMINATOR (igzip 555-556): trailing < 256 ⇒
                //    literal ⇒ back-edge (the HOT exit); ≥ 256 ⇒ length/EOB ⇒
                //    fall through to the non-literal arm. THE core igzip-shape
                //    change: all the speculation above ran before this single
                //    data-dependent branch resolves.
                "cmp {t5:e}, 256",
                "jb 2b",                              // literal → loop (HOT)
                // ── NIGHT40: short non-literal — capture the hoisted {d0} dst
                //    anchor HERE (off the hot literal path). dst = base+cnt (the
                //    speculative store advanced by cnt), so iteration-top dst =
                //    dst - cnt = dst - {t3}. {t3}=cnt is still live (set pre-refill,
                //    untouched by `6:`; the dist decode at `94:` clobbers it later,
                //    so it MUST be captured before `58:`). Covers EOB (82:),
                //    oversize (30:), and every dist bail (90/92/93) on the short
                //    path. Byte-exact == old top capture (iteration-top dst).
                // FLAG-SAFE snapshot: `mov` does NOT touch flags, so the
                //    `cmp {t5},256` result still drives the `je` (EOB) below.
                //    d0 := dst (= top+cnt here); the -cnt is applied later where
                //    flags are dead (length path) or re-set (short EOB/oversize
                //    stubs `81:`/`30:`). The long path (`21:`) captures d0=dst
                //    directly (dst un-advanced there, so already = top).
                "mov {d0}, {dst}",                    // d0 = top+cnt (flag-safe)
                // ── non-literal arm: EOB / oversize → restore + RECLASS; else
                //    length → decode_len_dist. dst = base+cnt → `dec` = copy
                //    start (base for a lone backref; base+(cnt-1) past the
                //    literal prefix for a pack). {dpre} carries the preloaded
                //    dist entry into `58:`.
                "je 81f",                             // EOB (short) → sub cnt @81 then RECLASS tag 2
                "cmp {t5:e}, 512",                    // MAX_LIT_LEN_SYM
                "ja 30f",                             // oversize (short) → sub cnt @30 then RECLASS
                "sub {d0}, {t3}",                     // length path: d0 = top+cnt - cnt = top (flags dead)
                "dec {dst}",                          // base+cnt → copy start
                "lea {t2:e}, [{t5:e} - 254]",         // length = trailing - 254
                "jmp 58f",                            // → shared dist+copy body
                // ── cold bottom (long-literal path): unconditional refill +
                //    preload, NO store (the long path stored its 1 byte already)
                "64:",
                "mov {t2}, qword ptr [{in_ptr} + {pos}]",
                "shlx {t2}, {t2}, {bitsleft}",
                "or {bitbuf}, {t2}",
                "mov {t5:e}, 63",
                "sub {t5}, {bitsleft}",
                "shr {t5}, 3",
                "add {pos}, {t5}",
                "or {bitsleft}, 56",
                "mov {t1:e}, {bitbuf:e}",
                "and {t1:e}, 0xFFF",
                "mov {t1:e}, dword ptr [{ctx} + {t1}*4 + {lit_off}]",
                "jmp 2b",
                // ── long code at top (decode_prefilled long path; rare/cold).
                //    The iteration-top p0/d0 anchor ([ctx+56]/[ctx+64]) covers
                //    its bails (reconstruct at 85:). {t2} = bc throughout.
                "20:",
                "mov {t2:e}, {t1:e}",
                "shr {t2:e}, 26",                     // long_max_len (≤21)
                "bzhi {t3}, {bitbuf}, {t2}",
                "shr {t3}, 12",                       // >> ISAL_DECODE_LONG_BITS
                "and {t1:e}, 0x1FFFFFF",
                "add {t1:e}, {t3:e}",                 // long_idx
                "lea {t2}, [{ctx} + {llong_off}]",    // litlen long table (inline, single base)
                "movzx {t1:e}, word ptr [{t2} + {t1}*2]",
                "mov {t2:e}, {t1:e}",
                "shr {t2:e}, 10",                     // bc
                "jz 86f",                             // invalid (bc==0) → Rust (UN-consumed, no re-read)
                "and {t1:e}, 0x3FF",                  // symbol
                "cmp {t1:e}, 255",
                "ja 21f",                             // lone non-literal → backref arm
                "shrx {bitbuf}, {bitbuf}, {t2}",      // lone literal via long path
                "sub {bitsleft}, {t2}",
                "mov byte ptr [{dst}], {t1:l}",
                "inc {dst}",
                "jmp 64b",                            // → cold bottom: refill + preload + loop
                // ── long lone non-literal (cnt=1; dst still = base = copy
                //    start, no spec store on the long path). Consume the long
                //    litlen, refill (dist budget), EOB/oversize gate, preload
                //    the dist entry, length, → 58: (no `dec` — dst is already
                //    the copy start).
                "21:",
                // NIGHT40: long lone non-literal — dst was NOT advanced on the
                //    long path (no speculative store), so iteration-top dst == dst.
                //    Capture the hoisted {d0} here (covers long EOB/oversize 82:/8:
                //    and long dist bails 90/92/93). Byte-exact == old top capture.
                "mov {d0}, {dst}",
                "mov {t5:e}, {t1:e}",                 // trailing = code (clean, no flag bit)
                "shrx {bitbuf}, {bitbuf}, {t2}",      // consume long litlen
                "sub {bitsleft}, {t2}",
                "mov {t2}, qword ptr [{in_ptr} + {pos}]",
                "shlx {t2}, {t2}, {bitsleft}",
                "or {bitbuf}, {t2}",
                "mov {t4:e}, 63",
                "sub {t4}, {bitsleft}",
                "shr {t4}, 3",
                "add {pos}, {t4}",
                "or {bitsleft}, 56",
                "cmp {t5:e}, 256",
                "je 82f",                             // long EOB → restore + RECLASS
                "cmp {t5:e}, 512",
                "ja 8f",                              // long oversize lone → restore + RECLASS tag 0
                "mov {dpre:e}, {bitbuf:e}",
                "and {dpre:e}, 0x1FF",                // DistTable::TABLE_BITS = 9
                "mov {dpre:e}, dword ptr [{ctx} + {dpre}*4 + {dist_off}]",
                "lea {t2:e}, [{t5:e} - 254]",         // length
                "jmp 58f",                            // dst = base = copy start
                // ── oversize trailing (>512): restore + RECLASS, tag by cnt
                //    (lone → 0 via 8:, multi → 3 via 83:).
                "30:",
                "sub {d0}, {t3}",                     // NIGHT40 short oversize: d0 = top (cmp below re-sets flags)
                "cmp {t3:e}, 1",
                "je 8f",                              // lone oversize → tag 0
                "jmp 83f",                            // multi oversize → tag 3
                // ── shared backref body (igzip decode_len_dist). {t2}=length,
                //    {dpre}=dist entry preloaded in the main body (igzip
                //    550-552) or at `21:` for the long path, {dst}=copy start.
                //    The base+index gather (the L1-latency load) already ran;
                //    the in-register entry decode follows with latency hidden.
                //    X2: the iteration-top p0/d0 anchor makes every dist-side
                //    bail un-consume (tag at `80:` → reconstruct at `85:`). ──
                "58:",
                "mov {t1:e}, {dpre:e}",               // preloaded dist entry → t1
                // FLOOR ORACLE (NIGHT16): subtable leaf re-enters here (the
                // inline subtable decode at 91: loads the leaf into {t1} and
                // jumps back to 94b). On a leaf the two tests below are
                // harmless: a valid leaf is nonzero (raw0 test falls through)
                // and bit14 (0x4000) is always 0 for a distance entry (the
                // distance base occupies bits 16-30, codeword_bits bits 8-11),
                // so it falls straight into the normal in-register entry decode.
                "94:",
                "test {t1:e}, {t1:e}",
                "jz 90f",                             // raw == 0 (hole/code 30/31) → bail (tag 5)
                "test {t1:e}, 0x4000",                // HUFFDEC_SUBTABLE_POINTER
                "jnz 91f",                            // subtable dist → INLINE decode at 91:
                "mov {t3}, {bitbuf}",                 // saved_bitbuf
                "shrx {bitbuf}, {bitbuf}, {t1}",      // consume_entry: >>= raw as u8 (= total_bits <= 31)
                "mov {t4:e}, {t1:e}",
                "and {t4:e}, 0x1F",
                "sub {bitsleft}, {t4}",               // -= raw & 0x1F (no underflow: budget proof)
                "bzhi {t3}, {t3}, {t1}",              // saved & ((1 << total_bits) - 1)
                "mov {t4:e}, {t1:e}",
                "shr {t4:e}, 8",
                "and {t4:e}, 0xF",                    // codeword_bits
                "shrx {t3}, {t3}, {t4}",              // extra value
                "shr {t1:e}, 16",                     // distance base
                "add {t1:e}, {t3:e}",                 // distance
                // CORRECTNESS-NEUTRAL DEAD-CHECK REMOVAL (Intel-silesia lever):
                // the `distance==0` and `distance>32768` bails are UNREACHABLE — a
                // decoded distance = LutDistCode base (1..24577) + bounded extra is
                // always in [1,32768] by construction, and an invalid dist code is
                // already caught by the raw==0 entry bail (`test {t1},{t1}; jz 90f`).
                // Removing these 3 hot back-ref instructions is BYTE-EXACT on valid
                // input (full differential suite incl. prop_near_max_distance +
                // diff_max_distance_backrefs_l9, 19/19) and corruption-behavior-
                // NEUTRAL (base-vs-after crash CONTROL identical on adversarial
                // mutants); the kept `src<out_base` bound (93f) + final CRC32 remain
                // the backstops. Gated Intel: gz/igzip 1.023->1.003 (instr/B -0.254
                // RESOLVED, 19/21 faster).
                "mov {t4}, {dst}",
                "sub {t4}, {t1}",                     // src = dst - distance
                "cmp {t4}, qword ptr [{ctx} + 24]",
                "jb 93f",                             // src < out_base ⇔ distance > *pos → restore (tag 8, marker)
                // RANK-2 UNCONDITIONAL pre-copy refill (igzip-converge: igzip
                // refills UNCONDITIONALLY; the `cmp bitsleft,48; jae 51f` skip is
                // DELETED to remove the STEP-1-located silesia 36%-mispredict
                // branch). Byte-exact: the refill is append-only and the fast
                // form is bit-for-bit `Bits::refill` (the ref-model pre-copy
                // refill `run_contig_ref_biased` is made unconditional in
                // lockstep, so the c2/c3 cursor-equality differential still
                // holds). When bitsleft>=48 the load+shlx appends >=0 high bytes
                // and pos advances by (63-bitsleft)>>3 in [0,1] — the cursor
                // stays self-consistent (more bits buffered, pos points past
                // them), decoding the identical symbol stream. OOB-safe: pos <=
                // iteration-top_pos + 8 < in_lim+8 = len-IN_MARGIN+8, so pos+8 <=
                // len (fast form only; the IN_MARGIN proof is preserved — the
                // refill was already issued conditionally here, this only removes
                // the skip). The carried gather below is pure.
                // CURSOR2 (single-refill convergence to igzip): DELETE the SECOND
                // per-back-ref refill (the `or {bitbuf}` + input load + pos-advance
                // + `or {bitsleft},56`). igzip refills ONCE per loop_block iteration
                // (528-530 / 543-547), covering BOTH the litlen and dist consumes
                // from that one full-budget refill, doing only a table preload before
                // the copy (575-577). gz had been refilling AGAIN here after the dist
                // consume — an EXTRA `or {bitbuf}` on the LOOP-CARRIED bitbuf
                // recurrence (bitbuf feeds the next index→entry-load) for ~30% of
                // iterations. That on-chain edge is the gz-specific critical-path
                // lengthening behind the +9% cyc/B vs igzip (gz IPC is already HIGHER
                // and mispredicts FEWER; gated trainer 2026-06-23: prior
                // instr-count cuts were cyc-FLAT/slack ⇒ the loop is LATENCY-bound,
                // so only chain-shortening moves the wall).
                //
                // BYTE-EXACT BUDGET PROOF: after `6:` real bitsleft F∈[56,63] (the
                // `or 56` accounting is exact for byte-granular loads). Dist consume
                // removes D bits; DEFLATE max D = 15-bit code + 13 extra = 28 (incl.
                // subtable). (a) carried index read needs ≥12: F−D ≥ 56−28 = 28 ✓.
                // (b) leave bitsleft=F−D, pos UN-advanced; bit pos pos*8−bitsleft is
                // unchanged (consume-only) so the next `6:` refill loads the correct
                // bytes. Next top: SHORT litlen bc≤12 (12-bit short table; >12-bit
                // symbols take the LONG path 20: which bzhi’s ≤21<28 then self-
                // refills) ⇒ bitsleft ≥ 28−12 = 16 ≥ 12 for the next `6:` index ✓.
                // Ref model `run_contig_ref_biased` drops its pre-copy refill in
                // lockstep (c2/c3 cursor differential preserved).
                "mov {t3:e}, {bitbuf:e}",
                "and {t3:e}, 0xFFF",
                "mov {t3:e}, dword ptr [{ctx} + {t3}*4 + {lit_off}]",  // carried preload only (igzip 575-577; no 2nd refill)
                // ── 16-byte MOVDQU back-ref copy (igzip large_byte_copy
                //    603-612 + small_byte_copy 614-627, COPY_SIZE = 16) ──
                //    t1=distance t2=length t4=src dst=dest {ret}=end {t3}=carried
                //    Faithful igzip mechanism: load 16B from src; if
                //    `distance >= min(16, length)` the 16-byte window never
                //    reads an un-finalized byte ⇒ large copy (one-or-more
                //    MOVDQU, src+=16/iter); else GROW THE PERIOD by storing
                //    then doubling the distance (small copy) until the period
                //    >= COPY_SIZE, then fall through to large. The bytes in
                //    [dst, dst+length) are byte-identical to the scalar
                //    `emit_backref_contig` walk; bytes above dst are X3
                //    overshoot (never read back). `length > 240` (vanishingly
                //    rare — mean back-ref len 6.3) takes the proven scalar
                //    path so the write extent stays inside the
                //    MAX_RUN_LENGTH+8 ring envelope: MOVDQU tail overshoot
                //    <= 15, so dst+240+15 = dst+255 <= cap-12 (the scalar
                //    path's own bound is dst+264 <= cap-3,
                //    marker_inflate.rs:2959-2962).
                // RANK-3 (B3) libdeflate overshoot-burst routing
                // (decompress_template.h:590-622): short-medium matches (<=40 B —
                // the nasa-dominant variable-trip-count case) take the scalar
                // 5-word (40 B) UNCONDITIONAL burst at `70:` — one shot, NO
                // per-16B `sub {t2},16; jle` trip-count loop (removes the nasa
                // 48%-mispredict B3 branch for the common length range). Long
                // matches (41..240) keep the 16-B MOVDQU SIMD loop (SIMD
                // efficiency where the loop is amortized); >240 also takes the
                // scalar path. Byte-exact: both copy paths are proven equivalent
                // to emit_backref_contig for every distance/overlap (the scalar
                // path IS the production >240 path), and a back-ref copy touches
                // NO bit-cursor (bitbuf/bitsleft/pos), so the c2/c3 cursor
                // differential + ref model are unchanged. Envelope-safe: <=40
                // scalar extent <=48 B, 41..240 MOVDQU extent <=255 B, >240
                // scalar extent <=265 B — all < FAST_OUT_SLOP=282.
                // COPYFLOOR-C (VEX-converge): all back-ref copy SIMD ops are
                // VEX `vmovdqu` (igzip emits VEX vmovdqu; gz had emitted legacy
                // SSE `movdqu`). Byte-exact (identical low-128 semantics; ymm0
                // upper is dead scratch). llvm-mca/chainlat rates legacy-SSE and
                // VEX movdqu identical (copy loop cyc/iter Δ=0.000), but legacy
                // SSE mixed with the AVX2 code elsewhere in the binary
                // (memchr/memmove avx2) can incur AVX-SSE dirty-upper
                // false-dependency penalties llvm-mca cannot see. VEX avoids it
                // and matches igzip's encoding faithfully.
                // EDIT1 (dispatch-delete): kill the `cmp {t2},40` mispredict (9% cyc,
                // ~10% br-miss). dist<16 → scalar overlap; len>48 → MOVDQU loop (RARE,
                // >99% not-taken at mean len 6.3); else the proven branchless 3x MOVDQU
                // burst (covers len<=48; 48B write = the threshold). No straddling branch.
                "cmp {t1}, 16",
                "jb 60f",                             // dist<16 → scalar overlap (period-growth)
                "cmp {t2}, 48",
                "ja 75f",                             // len>48 → MOVDQU loop (rare)
                "vmovdqu xmm0, [{t4}]",
                "vmovdqu [{dst}], xmm0",
                "vmovdqu xmm0, [{t4} + 16]",
                "vmovdqu [{dst} + 16], xmm0",
                "vmovdqu xmm0, [{t4} + 32]",
                "vmovdqu [{dst} + 32], xmm0",
                "add {dst}, {t2}",
                "mov {t1:e}, {t3:e}",
                "jmp 2b",
                "75:",                                // len 49..240 → MOVDQU loop; >240 → scalar
                "cmp {t2}, 240",
                "ja 60f",
                "lea {ret}, [{dst} + {t2}]",          // ret = end = dst + length
                "vmovdqu xmm0, [{t4}]",                // load 16 from src
                "mov {t5:e}, 16",
                "cmp {t5}, {t2}",
                "cmovg {t5}, {t2}",                   // t5 = min(16, length)
                "cmp {t1}, {t5}",
                "jb 72f",                             // distance < min → overlap (small)
                "71:",                                // large_byte_copy (igzip 603-612)
                "vmovdqu [{t4} + {t1}], xmm0",         // store 16 at src+distance (= dst run)
                "sub {t2}, 16",
                "jle 74f",
                "add {t4}, 16",
                "vmovdqu xmm0, [{t4}]",
                "jmp 71b",
                "72:",                                // small_byte_copy_pre (igzip 614-616)
                "add {t2}, {t1}",                     // repeat_length += distance
                "73:",                                // small_byte_copy (igzip 617-623)
                "vmovdqu [{t4} + {t1}], xmm0",
                "shl {t1}, 1",                        // distance *= 2 (grow the period)
                "vmovdqu xmm0, [{t4}]",
                "cmp {t1}, 16",
                "jl 73b",
                "sub {t2}, {t1}",                     // repeat_length -= distance
                "jg 71b",                             // remainder (> 0) → large
                "74:",
                "mov {dst}, {ret}",                   // dst advances by exactly length
                "mov {ret}, 1",                       // NIGHT36: restore hoisted BOUNDARY ({ret} clobbered as copy cursor)
                "mov {t1:e}, {t3:e}",                 // carried packet → top classify
                "jmp 2b",
                // RANK-2 (project_rank2_instr_locate_2026_06_23): the dist>=8
                // 5-word SCALAR burst below is 19.6% of all gz instructions.
                // For the dist>=16 AND len<=40 sub-bucket, replace it with an
                // UNCONDITIONAL 3x 16-byte MOVDQU burst (48 B written, NO
                // trip-count loop). PRESERVES B3's branchlessness (no nasa
                // `sub;jle` mispredict re-introduced) AND cuts the dominant
                // copy-path instruction count (3 SIMD stores vs ~10 scalar mov
                // pairs). Cite igzip large_byte_copy
                // (igzip_decode_block_stateless.asm:603-612, COPY_SIZE=16).
                // BYTE-EXACT ONLY for dist>=16: each sequential 16-byte load
                // stays at-or-behind the write frontier (frontier +16/store,
                // load offset +16) so every load reads already-finalized bytes
                // — the canonical libdeflate/igzip large-copy invariant.
                // dist<16 (the 1..7 period-growth + 8..15 word arms) and len>40
                // (incl. the rare >240) fall through to the UNTOUCHED scalar
                // path at 60:. The 48 B extent <= FAST_OUT_SLOP=282; src+48 <=
                // dst+32 stays inside the same envelope; the copy touches NO
                // bit cursor so the c2/c3 cursor differential + ref model are
                // unchanged. {ret} (hoisted BOUNDARY=1) is NOT clobbered here.
                "60:",                                // dist<16 OR len>40 → B3 scalar path
                "mov {ret}, {dst}",
                "prefetcht0 byte ptr [{t4} + 40]",    // EDIT1: unconditional (was cmp 40 gate)
                "52:",
                "add {t2}, {dst}",                    // t2 = end = dst + length
                "cmp {t1}, 8",
                "jb 55f",
                // dist >= 8: unconditional 5-word burst, then stride-8
                "mov {t5}, qword ptr [{t4}]",
                "mov qword ptr [{ret}], {t5}",
                "mov {t5}, qword ptr [{t4} + 8]",
                "mov qword ptr [{ret} + 8], {t5}",
                "mov {t5}, qword ptr [{t4} + 16]",
                "mov qword ptr [{ret} + 16], {t5}",
                "mov {t5}, qword ptr [{t4} + 24]",
                "mov qword ptr [{ret} + 24], {t5}",
                "mov {t5}, qword ptr [{t4} + 32]",
                "mov qword ptr [{ret} + 32], {t5}",
                "add {ret}, 40",
                "cmp {ret}, {t2}",
                "jae 59f",                            // length <= 40 → done
                "add {t4}, 40",
                "53:",
                "mov {t5}, qword ptr [{t4}]",
                "mov qword ptr [{ret}], {t5}",
                "add {t4}, 8",
                "add {ret}, 8",
                "cmp {ret}, {t2}",
                "jb 53b",
                "jmp 59f",
                "55:",
                "cmp {t1:e}, 1",
                "jne 56f",
                // dist == 1: RLE broadcast-word fill while dst < end
                "movzx {t5:e}, byte ptr [{t4}]",
                "mov {t1}, 0x0101010101010101",
                "imul {t5}, {t1}",
                "54:",
                "mov qword ptr [{ret}], {t5}",
                "add {ret}, 8",
                "cmp {ret}, {t2}",
                "jb 54b",
                "jmp 59f",
                // 2 <= dist <= 7: stride-dist words, 4 unconditional then
                // while dst < end
                "56:",
                "mov {t5}, qword ptr [{t4}]",
                "mov qword ptr [{ret}], {t5}",
                "add {t4}, {t1}",
                "add {ret}, {t1}",
                "mov {t5}, qword ptr [{t4}]",
                "mov qword ptr [{ret}], {t5}",
                "add {t4}, {t1}",
                "add {ret}, {t1}",
                "mov {t5}, qword ptr [{t4}]",
                "mov qword ptr [{ret}], {t5}",
                "add {t4}, {t1}",
                "add {ret}, {t1}",
                "mov {t5}, qword ptr [{t4}]",
                "mov qword ptr [{ret}], {t5}",
                "add {t4}, {t1}",
                "add {ret}, {t1}",
                "57:",
                "cmp {ret}, {t2}",
                "jae 59f",
                "mov {t5}, qword ptr [{t4}]",
                "mov qword ptr [{ret}], {t5}",
                "add {t4}, {t1}",
                "add {ret}, {t1}",
                "jmp 57b",
                "59:",
                "mov {dst}, {t2}",                    // dst advances by exactly length
                "mov {ret}, 1",                       // NIGHT36: restore hoisted BOUNDARY ({ret} clobbered as scalar cursor)
                "mov {t1:e}, {t3:e}",                 // carried packet → top classify
                "jmp 2b",
                // ── RECLASS un-consume via FROM-DATA RE-READ (X2). The igzip-
                //    shape body consumes+refills+spec-stores before the late
                //    discriminator, so a rare exit hands Rust the packet UN-
                //    consumed. Each bail sets {ret}=tag and jumps to 85: which
                //    reconstructs the cursor from the iteration-top bit anchor
                //    [ctx+56]=p0 (NOT a 4-value snapshot):
                //      byte = p0>>3 ; skip = p0&7
                //      bitbuf = load_u64(in_ptr+byte) >> skip   (low 64-skip bits = the
                //                                                bitstream from p0)
                //      bitsleft = 64 - skip   (∈[57,64] ⇒ X5 bitsleft>=48 holds)
                //      pos = byte + 8 ; dst = [ctx+64]=d0
                //    8-byte read in-range by IN_MARGIN (p0>>3 <= pos < in_lim =
                //    len-40 ⇒ byte+8 < len). Byte-exact: p0 is the exact un-
                //    consumed packet start; the re-read yields the SAME low bits
                //    decode_prefilled reads (a different but equivalent cursor
                //    REPRESENTATION). The ref model uses the identical re-read in
                //    lockstep; the caller only re-runs decode_prefilled(&lb),
                //    never inspecting the cursor shape. igzip has no counterpart
                //    (stateless — it never un-consumes).
                // NIGHT15 decomposed dist-bail labels (replace the lumped 80:).
                // All reconstruct identically at 85: (re-read from p0, dst=d0);
                // the only difference is the stat tag in {ret}. Byte-transparent.
                "90:",
                "mov {ret}, 5",                       // RECLASS raw==0 (hole/invalid)
                "jmp 85f",
                // FLOOR ORACLE (NIGHT16): INLINE subtable-dist (igzip
                // decode_next_dist long path, igzip_decode_block_stateless.asm
                // macro 396-440; mirror of production careful path
                // marker_inflate.rs:3447-3449 `if is_subtable_ptr { consume(9);
                // lookup_subtable_direct }`). At 91:: {dpre}=={t1}=subtable
                // pointer entry, {t2}=length (PRESERVE), bitbuf/bitsleft are
                // post-litlen+refill, t3/t4 free. Consume the 9 main DistTable
                // bits, index the subtable, load the leaf into {t1}, re-enter
                // 94b — where the leaf's total_bits/codeword_bits (= len-9 +
                // extra, relative to the post-9-consume bitbuf, since the
                // builder stores subtable_len = len-table_bits) are decoded by
                // the unchanged in-register entry path. Byte-exact (ref lockstep
                // in run_contig_ref_biased).
                "91:",
                "shr {bitbuf}, 9",                    // consume DistTable::TABLE_BITS=9
                "sub {bitsleft}, 9",
                "mov {t3:e}, {dpre:e}",
                "shr {t3:e}, 8",
                "and {t3:e}, 0xF",                    // subtable_bits (DistEntry bits 11-8)
                "bzhi {t4}, {bitbuf}, {t3}",          // idx = bitbuf & ((1<<sb)-1)   (sb<=6)
                "mov {t3:e}, {dpre:e}",
                "shr {t3:e}, 16",                     // subtable_start (DistEntry bits 31-16)
                "add {t4:e}, {t3:e}",
                "mov {t1:e}, dword ptr [{ctx} + {t4}*4 + {dist_off}]",   // leaf entry
                "jmp 94b",
                "92:",
                "mov {ret}, 7",                       // RECLASS bad-distance (0 / >window)
                "jmp 85f",
                "93:",
                "mov {ret}, 8",                       // RECLASS marker (src<out_base; out of scope)
                "jmp 85f",
                "81:",                                // NIGHT40 short-EOB stub: d0 = top (was top+cnt snapshot)
                "sub {d0}, {t3}",
                "82:",
                "mov {ret}, 2",                       // RECLASS, lone-EOB tag (long EOB enters here, d0 already top)
                "jmp 85f",
                "83:",
                "mov {ret}, 3",                       // RECLASS, multi-trailing tag
                "jmp 85f",
                // ── exits ───────────────────────────────────────────────
                "8:",
                "mov {ret}, 0",                       // RECLASS (lone/long oversize — CONSUMED)
                "jmp 85f",                            // → reconstruct
                "85:",
                "mov {t2}, {p0}",                     // p0 (in-register anchor)
                "mov {t1}, {t2}",
                "and {t1:e}, 7",                      // skip = p0 & 7
                "shr {t2}, 3",                        // byte = p0 >> 3
                "mov {bitbuf}, qword ptr [{in_ptr} + {t2}]",
                "shrx {bitbuf}, {bitbuf}, {t1}",      // >> skip
                "mov {bitsleft:e}, 64",
                "sub {bitsleft}, {t1}",               // bitsleft = 64 - skip
                "lea {pos}, [{t2} + 8]",              // pos = byte + 8
                "mov {dst}, {d0}",                    // dst = d0 (in-register anchor)
                "jmp 9f",
                // ── invalid (bc==0) exit: reached PRE-consume (short 438 /
                //    long 546), so the cursor is ALREADY at the un-consumed
                //    packet start and dst is still d0 — leave both UNCHANGED
                //    (X6), no re-read. ──
                "86:",
                "mov {ret}, 0",                       // RECLASS (invalid)
                "9:",
                // ELEMENT A single base: {ctx} addresses the inline litlen +
                // dist tables via const-operand displacements (igzip single
                // state base). {short_tbl} + {dtbl} are GONE (2 GP freed);
                // {p0}/{d0} carry the un-consume anchor in those freed regs.
                ctx = in(reg) ctx as *const AsmState,
                in_ptr = in(reg) ctx.in_ptr,
                dpre = out(reg) _,                    // preloaded dist entry carried 50: → 58:
                p0 = out(reg) _,                      // iteration-top bit anchor (was [ctx+56])
                d0 = out(reg) _,                      // iteration-top dst anchor (was [ctx+64])
                lit_off = const ASM_LIT_SHORT_OFF,    // [ctx + idx*4 + lit_off]
                llong_off = const ASM_LIT_LONG_OFF,   // lea [ctx + llong_off]
                dist_off = const ASM_DIST_OFF,        // [ctx + idx*4 + dist_off]
                bitbuf = inout(reg) bitbuf,
                bitsleft = inout(reg) bitsleft,
                pos = inout(reg) pos,
                dst = inout(reg) dst_c,
                ret = out(reg) ret,
                t1 = out(reg) _,
                t2 = out(reg) _,
                t3 = out(reg) _,
                t4 = out(reg) _,
                t5 = out(reg) _,
                out("xmm0") _, // MOVDQU back-ref copy scratch (16-byte SIMD)
                options(nostack),
            );
        }
        lb.bitbuf = bitbuf;
        lb.bitsleft = bitsleft as u32;
        lb.pos = pos as usize;
        (ret, dst_c)
    }

    pub fn note_exit(exit: u64, delta: usize) {
        if !stats_enabled() {
            return;
        }
        KERN_ENTRIES.fetch_add(1, Ordering::Relaxed);
        KERN_ASM_BYTES.fetch_add(delta as u64, Ordering::Relaxed);
        match exit {
            super::EXIT_BOUNDARY => &KERN_EXIT_BOUNDARY,
            super::EXIT_RECLASS_EOB => &KERN_RECLASS_EOB,
            super::EXIT_RECLASS_MULTI_TRAIL => &KERN_RECLASS_MULTI_TRAIL,
            super::EXIT_RECLASS_DIST => &KERN_RECLASS_DIST,
            super::EXIT_RECLASS_RAW0 => &KERN_RECLASS_RAW0,
            super::EXIT_RECLASS_SUBTABLE => &KERN_RECLASS_SUBTABLE,
            super::EXIT_RECLASS_BADDIST => &KERN_RECLASS_BADDIST,
            super::EXIT_RECLASS_MARKER => &KERN_RECLASS_MARKER,
            _ => &KERN_EXIT_RECLASS,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    pub fn dump_if_enabled() {
        if !stats_enabled() {
            return;
        }
        eprintln!(
            "[asm-kernel:c] enabled={} entries={} exit_boundary={} exit_reclass={} reclass_eob={} reclass_multi_trail={} reclass_dist={} reclass_raw0={} reclass_subtable={} reclass_baddist={} reclass_marker={} asm_bytes={}",
            enabled(),
            KERN_ENTRIES.load(Ordering::Relaxed),
            KERN_EXIT_BOUNDARY.load(Ordering::Relaxed),
            KERN_EXIT_RECLASS.load(Ordering::Relaxed),
            KERN_RECLASS_EOB.load(Ordering::Relaxed),
            KERN_RECLASS_MULTI_TRAIL.load(Ordering::Relaxed),
            KERN_RECLASS_DIST.load(Ordering::Relaxed),
            KERN_RECLASS_RAW0.load(Ordering::Relaxed),
            KERN_RECLASS_SUBTABLE.load(Ordering::Relaxed),
            KERN_RECLASS_BADDIST.load(Ordering::Relaxed),
            KERN_RECLASS_MARKER.load(Ordering::Relaxed),
            KERN_ASM_BYTES.load(Ordering::Relaxed),
        );
    }
}

#[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
pub use imp::{dump_if_enabled, enabled, note_exit, run_contig, stats_enabled};
#[cfg(all(test, feature = "asm-kernel", target_arch = "x86_64"))]
pub use imp::{TEST_FORCE, TEST_RUN_CONTIG_CALLS};

/// Non-asm builds: constant-false dispatch, no-op dump — call sites fold away.
#[cfg(not(all(feature = "asm-kernel", target_arch = "x86_64")))]
pub fn dump_if_enabled() {}

/// Pure-Rust REFERENCE MODEL of the asm region's contract — the exact
/// fast-loop subset of `decode_clean_into_contig` the asm owns (same
/// guards, same decode/backstop/threshold refill placement, same stores,
/// same copy shape via the production `emit_backref_contig`, same
/// RECLASS/restore rules), expressed through the production primitives
/// (`LutLitLenCode::{decode, decode_prefilled}`, `DistTable`, `Bits`). The
/// differential test pins asm == ref on (exit, bit cursor, dst, output
/// bytes); the ref's own equivalence to the production loop is by line-map
/// inspection (see `run_contig` doc) and by the guest suite/sha grid.
#[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
pub fn run_contig_ref(
    lut: &super::lut_huffman::LutLitLenCode,
    dist: &crate::decompress::inflate::libdeflate_entry::DistTable,
    lb: &mut Bits<'_>,
    out: &mut [u8],
    dst: &mut usize,
    out_lim: usize,
    in_lim: usize,
) -> u64 {
    let mut stats = RefArmStats::default();
    run_contig_ref_biased::<0>(lut, dist, lb, out, dst, out_lim, in_lim, &mut stats)
}

/// Arm-level coverage counters for the REFERENCE model — flip-precondition 2
/// (campaign §9 gate): the differential must PROVE it exercises the asm's
/// `25:` multi-literal+trailing-LENGTH arm (99.6% of c3a crossings — the
/// make-or-break arm of the F-c coverage gate), not silently degrade to
/// literals-only/lone-backref traffic after a refactor.
///
/// The ref counts are a valid proxy for the ASM arm: the differential pins
/// asm == ref on (exit class, full cursor, dst, bytes), so any asm misroute
/// of a packet the ref runs through this arm (e.g. `25:` redirected to a
/// bail) diverges in exit/dst/cursor and fails the equality asserts first.
/// (The asm-side KERN_RECLASS_MULTI_TRAIL counter tags only the BAIL
/// crossings; completions never cross the seam — they are countable only on
/// the ref side.)
#[derive(Default)]
pub struct RefArmStats {
    /// Packets entering the multi-with-trailing-length path past the
    /// EOB/oversize gates (the asm `25:` X2-spill point).
    pub multi_trail: u64,
    /// Those whose dist decode + copy completed in-region (the asm
    /// `25:` → `58:` → copy success route).
    pub multi_trail_completed: u64,
}

/// `run_contig_ref` with a test-injected CONSUME BIAS — the flip-precondition
/// positive control for the bit-consume failure class (asm-campaign §9 flip
/// gate, precondition 1).
///
/// `CONSUME_BIAS` is added to the lone-literal arm's litlen consume count
/// (`lb.consume(pre.bit_count + BIAS)`), modeling exactly an off-by-one in
/// the asm's consume pair (`shrx {bitbuf}, {bitbuf}, {t2}` +
/// `sub {bitsleft}, {t2}`). The differential harness compares the REF cursor
/// against the ASM cursor field-for-field, so a consume off-by-one on EITHER
/// side produces the same inequality — biasing the ref proves the harness's
/// cursor-equality asserts fire on this failure class without duplicating
/// the 450-line asm region. `BIAS = 0` is bit-for-bit the production
/// reference model (the wrapper above delegates here), so the control shares
/// every instruction with the real harness.
#[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
pub fn run_contig_ref_biased<const CONSUME_BIAS: u32>(
    lut: &super::lut_huffman::LutLitLenCode,
    dist: &crate::decompress::inflate::libdeflate_entry::DistTable,
    lb: &mut Bits<'_>,
    out: &mut [u8],
    dst: &mut usize,
    out_lim: usize,
    in_lim: usize,
    stats: &mut RefArmStats,
) -> u64 {
    loop {
        if *dst >= out_lim || lb.pos >= in_lim {
            return EXIT_BOUNDARY;
        }
        // NIGHT11 MINIMAL UN-CONSUME ANCHOR (lockstep with the asm's
        // [ctx+56]=p0 / [ctx+64]=d0 capture): the night9 4-value snapshot is
        // DELETED. Capture only the iteration-top BIT POSITION `p0` (refill-
        // invariant) and `d0`; a rare bail reconstructs the whole cursor from
        // `p0` via the identical from-data re-read (`reclass_reread`).
        let p0: usize = lb.pos * 8 - lb.bitsleft as usize;
        let d0: usize = *dst;
        let pre = lut.decode_prefilled(lb);
        if pre.bit_count == 0 {
            // invalid (asm `86:`): decode_prefilled consumed nothing, so the
            // cursor is already at the un-consumed packet start and dst is still
            // d0 — leave both UNCHANGED (X6), no re-read. (p0/d0 used by the
            // consumed-bail paths below.)
            return EXIT_RECLASS;
        }
        let cnt = pre.sym_count as usize;
        // TRAILING extract (asm 520-521), UNCONDITIONAL: the asm tests the
        // BUILD-TIME trailing-class flag bit but the resulting predicate is
        // exactly `trailing >= 256` (lut_huffman flag invariant); this ref
        // extracts the trailing packed symbol the way the asm now does in the
        // body. The `& 0xFFFF` was removed (NIGHT34 igzip-converge): redundant
        // given the 0x01FF_FFFF mask + the LUT build's zero-filled upper slots.
        let trailing = (pre.symbol & 0x01FF_FFFF) >> (8 * (cnt - 1));
        // SPECULATIVE STORE + advance by cnt (asm 518-519), UNCONDITIONAL — the
        // packed literal bytes go down assuming a pure-literal pack; a trailing
        // length over-advances dst by 1 (fixed below with `*dst -= 1`).
        let packed = (pre.symbol & 0x00FF_FFFF) as u64;
        out[*dst..*dst + 8].copy_from_slice(&packed.to_le_bytes());
        *dst += cnt;
        // CONSUME + REFILL every iteration (asm consume + main-body refill).
        // CONSUME_BIAS != 0 only in the positive-control test (off-by-one model
        // of a wrong asm consume count); 0 == production reference.
        lb.consume(pre.bit_count + CONSUME_BIAS);
        lb.refill();
        // The asm's speculative dist-entry preload (550-552) is a timing-only
        // pure table read; the ref models it lazily at `dist.lookup` below.
        // LATE DISCRIMINATOR (asm 555-556).
        if trailing <= 255 {
            continue; // literal → loop (the hot back-edge)
        }
        // ── non-literal arm: EOB / oversize → un-consume + RECLASS ──
        if trailing == 256 || trailing > super::lut_huffman::MAX_LIT_LEN_SYM {
            reclass_reread(lb, p0);
            *dst = d0;
            return EXIT_RECLASS;
        }
        if cnt >= 2 {
            stats.multi_trail += 1;
        }
        // decode_len_dist: dst = base+cnt → copy start base+cnt-1 (asm `dec`).
        *dst -= 1;
        let length = trailing as usize - 254;
        // FLOOR ORACLE (NIGHT16): inline subtable-dist in lockstep with the asm
        // 91:→94b path (mirror of careful path marker_inflate.rs:3447-3449).
        // After consuming the 9 main DistTable bits, the leaf entry's
        // total_bits/codeword_bits are relative to the post-9-consume bitbuf, so
        // `decode_distance(saved=post-9 bitbuf)` and `consume_entry(leaf.raw())`
        // reproduce the asm exactly. raw==0 (hole/codes 30,31) still bails.
        let e0 = dist.lookup(lb.bitbuf);
        let e = if e0.is_subtable_ptr() {
            lb.consume(crate::decompress::inflate::libdeflate_entry::DistTable::TABLE_BITS as u32);
            dist.lookup_subtable_direct(e0, lb.bitbuf)
        } else {
            e0
        };
        let bail = if e.raw() == 0 {
            true
        } else {
            let saved = lb.bitbuf;
            lb.consume_entry(e.raw());
            let distance = e.decode_distance(saved) as usize;
            if distance == 0 || distance > 32768 || distance > *dst {
                true
            } else {
                // CURSOR2 lockstep: the asm DROPS the second per-back-ref refill
                // (single-refill convergence to igzip), so the ref model drops its
                // pre-copy `lb.refill()` here too — keeping the c2/c3 cursor-
                // equality differential (asm.pos/bitbuf/bitsleft == ref) exact.
                // Budget proof in the asm comment: F−D ≥ 28 bits remain, enough for
                // the next short decode + `6:` index; the next-iteration refill tops
                // up. No refill here.
                unsafe {
                    super::marker_inflate::emit_backref_contig(
                        out.as_mut_ptr(),
                        dst,
                        distance,
                        length,
                    );
                }
                if cnt >= 2 {
                    stats.multi_trail_completed += 1;
                }
                false
            }
        };
        if bail {
            reclass_reread(lb, p0);
            *dst = d0;
            return EXIT_RECLASS;
        }
    }
}

/// X2 un-consume reconstruction (lockstep with the asm `85:` block). Given the
/// iteration-top absolute BIT position `p0` (= `pos*8 - bitsleft`, refill-
/// invariant), re-read the bit cursor from the input so it points at the
/// un-consumed packet start with `bitsleft >= 48` (X5). The consumed low bits
/// were shifted out by `consume` and are unrecoverable from the post-consume
/// cursor, so a from-data re-read is the only faithful un-consume. The result
/// is a different but equivalent cursor REPRESENTATION than the pre-consume
/// state; correctness rests on `p0` being the exact packet start and the caller
/// only re-running `decode_prefilled(&lb)` (never inspecting the cursor shape).
/// `byte + 8 <= data.len()` is guaranteed by IN_MARGIN inside the asm region
/// (`p0>>3 <= pos < in_lim = len-40`).
#[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
#[inline]
fn reclass_reread(lb: &mut Bits<'_>, p0: usize) {
    let byte = p0 >> 3;
    let skip = (p0 & 7) as u32;
    let w = u64::from_le_bytes(lb.data[byte..byte + 8].try_into().unwrap());
    lb.bitbuf = w >> skip;
    lb.bitsleft = 64 - skip;
    lb.pos = byte + 8;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_allowed_excludes_every_knob() {
        // Charter §3.5: any active measurement knob must force the pure-Rust
        // path so causal instruments can still perturb the region.
        let ok = |p, n, o, d, s, t, cap| dispatch_allowed(p, n, o, d, s, t, cap, 8);
        assert!(ok(false, false, false, 0, 0, false, 9));
        assert!(!ok(true, false, false, 0, 0, false, 9), "contig_prof");
        assert!(!ok(false, true, false, 0, 0, false, 9), "oracle nostore");
        assert!(!ok(false, false, true, 0, 0, false, 9), "oracle recorder");
        assert!(!ok(false, false, false, 7, 0, false, 9), "decode slow knob");
        assert!(!ok(false, false, false, 0, 7, false, 9), "store slow knob");
        assert!(!ok(false, false, false, 0, 0, true, 9), "track_backrefs");
        assert!(!ok(false, false, false, 0, 0, false, 8), "no out headroom");
    }

    /// c1 seam round-trip: the asm region must return with the cursor and
    /// dst bit-for-bit unchanged and a valid exit code, for both guard
    /// outcomes. (Executes only where the asm path is compiled; the guest
    /// gauntlet is the authoritative run.)
    #[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
    #[test]
    fn c1_seam_roundtrip_state_unchanged() {
        let data = vec![0xA5u8; 256];
        let mut lb = Bits {
            data: &data,
            pos: 17,
            bitbuf: 0x0123_4567_89AB_CDEF,
            bitsleft: 56,
        };
        let mut out = vec![0u8; 64];
        let dst = out.as_mut_ptr();
        // ELEMENT A: zeroed inline tables in one AsmState (the prologue reads
        // the short table once; guards fail before any decode).
        use crate::decompress::inflate::libdeflate_entry::DistTable;
        use crate::decompress::parallel::lut_huffman::LutLitLenCode;
        let mut ctx = Box::new(AsmState {
            in_ptr: data.as_ptr() as u64,
            in_lim: (data.len() - IN_MARGIN) as u64,
            out_lim: dst as u64 + 32,
            out_base: dst as u64,
            lut_litlen: LutLitLenCode::new_empty(),
            dist: DistTable::new_empty(),
        });
        // Guards pass → RECLASS, state unchanged.
        let (exit, ndst) = unsafe { run_contig(&ctx, &mut lb, dst) };
        assert_eq!(exit, EXIT_RECLASS);
        assert_eq!(ndst, dst, "X6: dst unchanged");
        assert_eq!(lb.pos, 17);
        assert_eq!(lb.bitbuf, 0x0123_4567_89AB_CDEF);
        assert_eq!(lb.bitsleft, 56);
        // Out guard fails → BOUNDARY, state unchanged.
        ctx.out_lim = dst as u64;
        let (exit, ndst) = unsafe { run_contig(&ctx, &mut lb, dst) };
        assert_eq!(exit, EXIT_BOUNDARY);
        assert_eq!(ndst, dst);
        assert_eq!(lb.pos, 17);
        assert_eq!(lb.bitbuf, 0x0123_4567_89AB_CDEF);
        assert_eq!(lb.bitsleft, 56);
        // In guard fails → BOUNDARY.
        ctx.out_lim = dst as u64 + 32;
        ctx.in_lim = 17;
        let (exit, _) = unsafe { run_contig(&ctx, &mut lb, dst) };
        assert_eq!(exit, EXIT_BOUNDARY);
        assert_eq!(lb.bitsleft, 56);
    }

    /// Stage c2 differential: the asm region vs the pure-Rust reference
    /// model over random bitstreams × three table shapes (fixed-Huffman
    /// with length codes → RECLASS coverage; dense short codes → multi-sym
    /// packing + chain coverage; 13-bit literal codes → long-path
    /// resolution at top AND inside the chain) × varying out budgets
    /// (BOUNDARY coverage). Compares exit code, full bit cursor
    /// (pos/bitbuf/bitsleft — X1 state equality), dst advance, and every
    /// output byte below dst (X3). Skips the asm half without BMI2 (local
    /// Rosetta); the guest gauntlet is authoritative.
    #[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
    #[test]
    fn c2_differential_asm_vs_ref_random_streams() {
        use crate::decompress::parallel::lut_huffman::LutLitLenCode;
        if !std::arch::is_x86_feature_detected!("bmi2") {
            eprintln!("SKIP c2 differential: no BMI2 on this host (run on guest)");
            return;
        }
        // Fixed-Huffman litlen lens (RFC1951 §3.2.6) — 7-9-bit codes,
        // length/EOB symbols present.
        let mut fixed = vec![8u8; 288];
        fixed[144..256].iter_mut().for_each(|x| *x = 9);
        fixed[256..280].iter_mut().for_each(|x| *x = 7);
        // Dense short codes (Kraft-complete) — exercises 2-3-sym packing
        // and the literal chain: lens {2,2,3,3,4,4,4(eob)} + 4×6.
        let mut dense = vec![0u8; 286];
        dense[0] = 2;
        dense[1] = 2;
        dense[2] = 3;
        dense[3] = 3;
        dense[4] = 4;
        dense[5] = 4;
        dense[256] = 4;
        dense[6] = 6;
        dense[7] = 6;
        dense[8] = 6;
        dense[9] = 6;
        // Long-code-heavy (Kraft-complete): lens 1..6 for a few hot
        // literals + EOB, remainder 1/64 as 128 literals at 13 bits —
        // populates long_code_lookup with LITERALS (long resolve at top
        // and inside the chain).
        let mut longy = vec![0u8; 286];
        longy[0] = 1;
        longy[1] = 2;
        longy[2] = 3;
        longy[3] = 4;
        longy[4] = 5;
        longy[256] = 6;
        for s in 5..133 {
            longy[s] = 13;
        }
        let lens_set: [&[u8]; 3] = [&fixed, &dense, &longy];

        let mut x: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        };
        // All-holes dist table: every lone length code takes the c3
        // spill/restore path and RECLASSes (preserves this test's c2
        // literal-only semantics and small-buffer envelope, while now
        // covering the X2 restore on every backref candidate).
        use crate::decompress::inflate::libdeflate_entry::DistTable;
        let dist_holes = DistTable::build(&[0u8; 30]).expect("holes dist table");
        for (ti, lens) in lens_set.iter().enumerate() {
            // ELEMENT A: tables INLINE in one AsmState — the asm reads them off
            // the single `ctx` base; the ref reads the SAME storage
            // (`st.lut_litlen` / `st.dist`), so contents are identical and the
            // differential is valid.
            let mut st = Box::new(AsmState {
                in_ptr: 0,
                in_lim: 0,
                out_lim: 0,
                out_base: 0,
                lut_litlen: {
                    let mut c = LutLitLenCode::new_empty();
                    assert!(c.rebuild_from(lens), "test table lens must be valid");
                    c
                },
                dist: dist_holes.clone(),
            });
            for trial in 0..4000 {
                // Random input stream, long enough for the margin scheme.
                let n = 96 + (next() as usize % 400);
                let data: Vec<u8> = (0..n).map(|_| next() as u8).collect();
                let in_lim = data.len().saturating_sub(IN_MARGIN);
                // Varying out budget: small values force BOUNDARY exits.
                let out_lim = 4 + (next() as usize % 240);
                let buf_len = out_lim + 16; // FAST_OUT_SLOP-class envelope
                let mut out_ref = vec![0u8; buf_len];
                let mut out_asm = vec![0u8; buf_len];

                let mut lb_ref = Bits::new(&data);
                let mut lb_asm = Bits {
                    data: &data,
                    pos: lb_ref.pos,
                    bitbuf: lb_ref.bitbuf,
                    bitsleft: lb_ref.bitsleft,
                };

                let mut dref = 0usize;
                let ref_exit = run_contig_ref(
                    &st.lut_litlen,
                    &st.dist,
                    &mut lb_ref,
                    &mut out_ref,
                    &mut dref,
                    out_lim,
                    in_lim,
                );

                let dst0 = out_asm.as_mut_ptr();
                st.in_ptr = data.as_ptr() as u64;
                st.in_lim = in_lim as u64;
                st.out_lim = dst0 as u64 + out_lim as u64;
                st.out_base = dst0 as u64;
                let (asm_exit, dst1) = unsafe { run_contig(&st, &mut lb_asm, dst0) };
                let dasm = (dst1 as usize) - (dst0 as usize);

                let tag = format!("table {ti} trial {trial}");
                // Ref reports the exit CLASS; the asm additionally tags
                // RECLASS reasons (codes >= 2) for the effect instrument.
                let asm_class = if asm_exit == EXIT_BOUNDARY {
                    EXIT_BOUNDARY
                } else {
                    EXIT_RECLASS
                };
                assert_eq!(ref_exit, asm_class, "exit diverged ({tag})");
                assert_eq!(dref, dasm, "dst advance diverged ({tag})");
                assert_eq!(lb_ref.pos, lb_asm.pos, "pos diverged ({tag})");
                assert_eq!(lb_ref.bitbuf, lb_asm.bitbuf, "bitbuf diverged ({tag})");
                assert_eq!(
                    lb_ref.bitsleft, lb_asm.bitsleft,
                    "bitsleft diverged ({tag})"
                );
                assert_eq!(
                    &out_ref[..dref],
                    &out_asm[..dasm],
                    "output bytes diverged ({tag})"
                );
            }
        }
    }

    /// Flip-precondition 1 (campaign §9 gate): POSITIVE CONTROL for the
    /// bit-consume failure class. A test-injected off-by-one in the
    /// lone-literal consume count (`run_contig_ref_biased::<1>` — the ref
    /// mirror of corrupting the asm's `shrx`/`sub bitsleft` pair) MUST trip
    /// the differential's cursor-equality asserts; the bias-0 arm on the
    /// SAME inputs must stay divergence-free (so the divergence is caused
    /// by the mutation, not the inputs). Proves the harness is live for
    /// consume-count drift — the one failure class the three c3 controls
    /// (lit store, dist base shift, multi advance) did not cover.
    #[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
    #[test]
    fn positive_control_consume_off_by_one_trips_cursor_asserts() {
        use crate::decompress::inflate::libdeflate_entry::DistTable;
        use crate::decompress::parallel::lut_huffman::LutLitLenCode;
        if !std::arch::is_x86_feature_detected!("bmi2") {
            eprintln!("SKIP consume-bias control: no BMI2 on this host (run on guest)");
            return;
        }
        // Fixed-Huffman litlen (8-9-bit literal codes → lone-literal-dominant,
        // the arm the bias mutates) + all-holes dist (every backref candidate
        // RECLASSes pre-consume — c2 envelope).
        let mut fixed = vec![8u8; 288];
        fixed[144..256].iter_mut().for_each(|x| *x = 9);
        fixed[256..280].iter_mut().for_each(|x| *x = 7);
        let mut lut = LutLitLenCode::new_empty();
        assert!(lut.rebuild_from(&fixed), "fixed table lens must be valid");
        let dist_holes = DistTable::build(&[0u8; 30]).expect("holes dist table");
        // ELEMENT A: tables INLINE in one AsmState; asm + both ref arms read the
        // SAME storage (st.lut_litlen / st.dist).
        let mut st = Box::new(AsmState {
            in_ptr: 0,
            in_lim: 0,
            out_lim: 0,
            out_base: 0,
            lut_litlen: lut,
            dist: dist_holes,
        });

        let mut x: u64 = 0xD1B54A32D192ED03;
        let mut next = move || {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        };
        const TRIALS: usize = 2000;
        let mut cursor_diverged = 0usize;
        for trial in 0..TRIALS {
            let n = 96 + (next() as usize % 400);
            let data: Vec<u8> = (0..n).map(|_| next() as u8).collect();
            let in_lim = data.len().saturating_sub(IN_MARGIN);
            let out_lim = 4 + (next() as usize % 240);
            let buf_len = out_lim + 16;

            // Arm A: asm (the system under test in the real harness).
            let mut out_asm = vec![0u8; buf_len];
            let mut lb_asm = Bits::new(&data);
            let dst0 = out_asm.as_mut_ptr();
            st.in_ptr = data.as_ptr() as u64;
            st.in_lim = in_lim as u64;
            st.out_lim = dst0 as u64 + out_lim as u64;
            st.out_base = dst0 as u64;
            let (asm_exit, dst1) = unsafe { run_contig(&st, &mut lb_asm, dst0) };
            let dasm = (dst1 as usize) - (dst0 as usize);

            // Arm B (negative control): bias-0 ref — MUST match the asm on
            // every compared field (the live harness, unmutated).
            let mut out_ref0 = vec![0u8; buf_len];
            let mut lb_ref0 = Bits::new(&data);
            let mut dref0 = 0usize;
            let mut st0 = RefArmStats::default();
            let ref0_exit = run_contig_ref_biased::<0>(
                &st.lut_litlen,
                &st.dist,
                &mut lb_ref0,
                &mut out_ref0,
                &mut dref0,
                out_lim,
                in_lim,
                &mut st0,
            );
            let asm_class = if asm_exit == EXIT_BOUNDARY {
                EXIT_BOUNDARY
            } else {
                EXIT_RECLASS
            };
            assert_eq!(ref0_exit, asm_class, "bias-0 exit diverged (trial {trial})");
            assert_eq!(dref0, dasm, "bias-0 dst diverged (trial {trial})");
            assert_eq!(
                lb_ref0.pos, lb_asm.pos,
                "bias-0 pos diverged (trial {trial})"
            );
            assert_eq!(
                lb_ref0.bitbuf, lb_asm.bitbuf,
                "bias-0 bitbuf diverged (trial {trial})"
            );
            assert_eq!(
                lb_ref0.bitsleft, lb_asm.bitsleft,
                "bias-0 bitsleft diverged (trial {trial})"
            );
            assert_eq!(
                &out_ref0[..dref0],
                &out_asm[..dasm],
                "bias-0 bytes diverged (trial {trial})"
            );

            // Arm C (positive control): bias-1 ref — the injected off-by-one.
            // Count trials where the CURSOR-equality fields (pos/bitbuf/
            // bitsleft) detect it.
            let mut out_ref1 = vec![0u8; buf_len];
            let mut lb_ref1 = Bits::new(&data);
            let mut dref1 = 0usize;
            let mut st1 = RefArmStats::default();
            let _ = run_contig_ref_biased::<1>(
                &st.lut_litlen,
                &st.dist,
                &mut lb_ref1,
                &mut out_ref1,
                &mut dref1,
                out_lim,
                in_lim,
                &mut st1,
            );
            if lb_ref1.pos != lb_asm.pos
                || lb_ref1.bitbuf != lb_asm.bitbuf
                || lb_ref1.bitsleft != lb_asm.bitsleft
            {
                cursor_diverged += 1;
            }
        }
        // Trials whose region exits before any lone literal (first packet is
        // a length/EOB candidate, ~22% on the fixed table) consume nothing
        // and legitimately cannot diverge; every trial that consumes a
        // literal must. Floor at 50% — observed rate is ~78%+.
        assert!(
            cursor_diverged >= TRIALS / 2,
            "positive control DEAD: consume off-by-one tripped the cursor \
             asserts in only {cursor_diverged}/{TRIALS} trials"
        );
        eprintln!(
            "[control] consume off-by-one: cursor asserts fired in \
             {cursor_diverged}/{TRIALS} trials (bias-0 arm clean)"
        );
    }

    /// Stage c3 differential: asm vs ref over WINDOWED buffers (32 KiB
    /// random prefill shared by both arms, dst starting at the window
    /// edge so distances <= 32768 are valid) × litlen tables WITH length
    /// codes × dist-table shapes chosen for arm coverage:
    ///   * `lenny` litlen (≈half length codes; 281..284 @8 + 4-5 extra
    ///     bits exceed the 12-bit short table → LONG-path length codes,
    ///     covering the `21:` → `50:` route) — heavy backref traffic;
    ///   * `dist_small` (complete, dsym 0..7) — dist 1..16: RLE (dist 1),
    ///     stride (2..7), burst overlap (8+), short lengths;
    ///   * `dist_fixed5` ([5;30], 2/32 holes) — full distance range: most
    ///     backrefs invalid-dist or > *pos → restore/RECLASS coverage;
    ///   * `dist_sub` (complete, 10-bit tails) — subtable-pointer bails.
    /// Compares exit, full cursor, dst advance, and every byte above the
    /// shared window. Coverage floor asserted on the (lenny × small) cell
    /// so the test cannot silently degrade to literals-only.
    #[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
    #[test]
    fn c3_differential_asm_vs_ref_windowed_backrefs() {
        use crate::decompress::inflate::libdeflate_entry::DistTable;
        use crate::decompress::parallel::lut_huffman::LutLitLenCode;
        if !std::arch::is_x86_feature_detected!("bmi2") {
            eprintln!("SKIP c3 differential: no BMI2 on this host (run on guest)");
            return;
        }
        const WIN: usize = 32 * 1024;
        // Litlen with length codes (Kraft-complete):
        //   lits 0..3 @3, lengths 257..264 @5, 265..272 @6, 273..280 @7,
        //   281..284 @8, EOB @8, lits 10..31 @9.
        let mut lenny = vec![0u8; 286];
        for s in 0..4 {
            lenny[s] = 3;
        }
        for s in 257..265 {
            lenny[s] = 5;
        }
        for s in 265..273 {
            lenny[s] = 6;
        }
        for s in 273..281 {
            lenny[s] = 7;
        }
        for s in 281..285 {
            lenny[s] = 8;
        }
        lenny[256] = 8;
        for s in 10..32 {
            lenny[s] = 9;
        }
        // RFC-fixed-shape litlen (lengths via 7-8-bit codes).
        let mut fixed = vec![8u8; 288];
        fixed[144..256].iter_mut().for_each(|x| *x = 9);
        fixed[256..280].iter_mut().for_each(|x| *x = 7);
        // Dist tables.
        let dist_small = DistTable::build(&[2, 2, 2, 3, 4, 5, 6, 6]).expect("small dist table");
        let dist_fixed5 = DistTable::build(&[5u8; 30]).expect("fixed5 dist table");
        let dist_sub = DistTable::build(&[1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 10, 10])
            .expect("subtable dist table");
        // ELEMENT A: pairs carry the litlen LENS (a per-pair AsmState rebuilds
        // the inline litlen table) + the dist table (cloned into the inline slot).
        let pairs: [(&[u8], &DistTable, &str); 5] = [
            (&lenny, &dist_small, "lenny×small"),
            (&lenny, &dist_fixed5, "lenny×fixed5"),
            (&lenny, &dist_sub, "lenny×sub"),
            (&fixed, &dist_fixed5, "fixed×fixed5"),
            (&fixed, &dist_small, "fixed×small"),
        ];

        let mut x: u64 = 0x243F6A8885A308D3;
        let mut next = move || {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        };
        // Shared window prefill (identical in both arms).
        let window: Vec<u8> = (0..WIN).map(|_| next() as u8).collect();
        let mut small_pair_bytes = 0u64;
        let mut small_pair_trials = 0u64;
        // Flip-precondition 2: per-cell arm coverage (see RefArmStats doc —
        // ref counts are a valid proxy for the asm arms under the equality
        // asserts below).
        let mut cell_stats: [RefArmStats; 5] = Default::default();
        for (pi, (lens, dt, tag0)) in pairs.iter().enumerate() {
            // Inline litlen + dist in ONE AsmState (zero-copy single base);
            // asm + ref share the storage (st.lut_litlen / st.dist).
            let mut st = Box::new(AsmState {
                in_ptr: 0,
                in_lim: 0,
                out_lim: 0,
                out_base: 0,
                lut_litlen: {
                    let mut c = LutLitLenCode::new_empty();
                    assert!(c.rebuild_from(lens), "test table lens must be valid");
                    c
                },
                dist: (*dt).clone(),
            });
            for trial in 0..3000 {
                let n = 150 + (next() as usize % 450);
                let data: Vec<u8> = (0..n).map(|_| next() as u8).collect();
                let in_lim = data.len().saturating_sub(IN_MARGIN);
                let budget = 16 + (next() as usize % 368);
                let out_lim = WIN + budget;
                // Envelope: copy overshoot <= max(40, length+7) <= 265.
                let buf_len = out_lim + 300;
                let mut out_ref = vec![0u8; buf_len];
                let mut out_asm = vec![0u8; buf_len];
                out_ref[..WIN].copy_from_slice(&window);
                out_asm[..WIN].copy_from_slice(&window);

                let mut lb_ref = Bits::new(&data);
                let mut lb_asm = Bits {
                    data: &data,
                    pos: lb_ref.pos,
                    bitbuf: lb_ref.bitbuf,
                    bitsleft: lb_ref.bitsleft,
                };

                let mut dref = WIN;
                let ref_exit = run_contig_ref_biased::<0>(
                    &st.lut_litlen,
                    &st.dist,
                    &mut lb_ref,
                    &mut out_ref,
                    &mut dref,
                    out_lim,
                    in_lim,
                    &mut cell_stats[pi],
                );

                let base0 = out_asm.as_mut_ptr();
                let dst0 = unsafe { base0.add(WIN) };
                st.in_ptr = data.as_ptr() as u64;
                st.in_lim = in_lim as u64;
                st.out_lim = base0 as u64 + out_lim as u64;
                st.out_base = base0 as u64;
                let (asm_exit, dst1) = unsafe { run_contig(&st, &mut lb_asm, dst0) };
                let dasm = WIN + ((dst1 as usize) - (dst0 as usize));

                let tag = format!("pair {pi} ({tag0}) trial {trial}");
                // Ref reports the exit CLASS; the asm additionally tags
                // RECLASS reasons (codes >= 2) for the effect instrument.
                let asm_class = if asm_exit == EXIT_BOUNDARY {
                    EXIT_BOUNDARY
                } else {
                    EXIT_RECLASS
                };
                assert_eq!(ref_exit, asm_class, "exit diverged ({tag})");
                assert_eq!(dref, dasm, "dst advance diverged ({tag})");
                assert_eq!(lb_ref.pos, lb_asm.pos, "pos diverged ({tag})");
                assert_eq!(lb_ref.bitbuf, lb_asm.bitbuf, "bitbuf diverged ({tag})");
                assert_eq!(
                    lb_ref.bitsleft, lb_asm.bitsleft,
                    "bitsleft diverged ({tag})"
                );
                assert_eq!(
                    &out_ref[WIN..dref],
                    &out_asm[WIN..dasm],
                    "output bytes diverged ({tag})"
                );
                if pi == 0 {
                    small_pair_bytes += (dasm - WIN) as u64;
                    small_pair_trials += 1;
                }
            }
        }
        // Coverage floor: the lenny×small cell must emit real backref
        // volume (literal-only c2 averaged ~1.3 B/exit on silesia; with
        // valid small distances the average must far exceed that).
        let avg = small_pair_bytes as f64 / small_pair_trials as f64;
        assert!(
            avg > 30.0,
            "c3 coverage degraded: lenny×small avg bytes/run = {avg:.1}"
        );
        // Flip-precondition 2 (campaign §9 gate): the `25:` multi-literal+
        // trailing-LENGTH arm — 99.6% of c3a crossings, the F-c coverage
        // make-or-break — must be PROVEN exercised, both the completion
        // route (`25:`→`58:`→copy) and the X2 dst-rollback bail route. The
        // floors are ~25% of the observed counts (lenny×small completes
        // 7,735; bails total 373 — lenny×fixed5 344 + lenny×sub 29), so a
        // refactor that silently starves the arm fails loudly while seed/
        // table tweaks within reason pass.
        for (pi, st) in cell_stats.iter().enumerate() {
            eprintln!(
                "[c3 coverage] cell {pi} ({}): multi_trail={} completed={}",
                pairs[pi].2, st.multi_trail, st.multi_trail_completed
            );
        }
        assert!(
            cell_stats[0].multi_trail_completed >= 2_000,
            "25:-arm coverage degraded: lenny×small completed only {} \
             multi-with-trailing-length backrefs (floor 2000)",
            cell_stats[0].multi_trail_completed
        );
        let bails: u64 = cell_stats
            .iter()
            .map(|s| s.multi_trail - s.multi_trail_completed)
            .sum();
        assert!(
            bails >= 90,
            "25:-arm bail (X2 dst-rollback) coverage degraded: only {bails} \
             multi-trailing dist bails across all cells (floor 90)"
        );
    }
}
