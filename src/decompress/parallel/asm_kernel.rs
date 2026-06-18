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
//! bottom `< 48`, or the backref pre-copy `< 48`). Each fast refill advances
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
//! never read back. Backref (c3): the P3.4 `emit_backref_contig` shape
//! transliterated (dist ≥ 8 burst-5 + stride-8 words; dist == 1 RLE
//! broadcast word fill; 2..=7 stride-dist words; `length > 40` prefetch) —
//! NOT VAR_VIII's byte-copy-all-overlaps D loop (refuted by P3.4's −87 ms).
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

/// Loop-invariant context for the asm region (VAR_VIII KernCtx salvage —
/// cold invariants via `[ctx + off]` memory operands keep register pressure
/// inside the 12-operand envelope). Field order is the asm ABI: offsets are
/// compile-asserted below and referenced as literal displacements.
#[repr(C)]
pub struct KernCtx {
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
    /// +32: litlen long-code table (`long_code_lookup.as_ptr()`, u16s).
    pub long_tbl: u64,
    /// +40: dist `DistTable` entries base (`*const DistEntry` = u32s, c3).
    pub dist_tbl: u64,
    /// +48: backref-arm spill: saved bitbuf (X2 restore).
    pub save_bitbuf: u64,
    /// +56: backref-arm spill: saved bitsleft.
    pub save_bitsleft: u64,
    /// +64: litlen short table (`short_code_lookup.as_ptr()`) — passed to
    /// the asm as a pinned REGISTER; kept here so the ctx is self-contained.
    pub short_tbl: u64,
    /// +72: backref-arm spill: saved dst (X2 restore — the multi-with-
    /// trailing path advances dst by the literal prefix BEFORE the dist
    /// decode can bail; the restore rolls dst back, and the already-stored
    /// packed bytes become X3 overshoot garbage above the exit dst).
    pub save_dst: u64,
}

const _: () = assert!(std::mem::offset_of!(KernCtx, in_ptr) == 0);
const _: () = assert!(std::mem::offset_of!(KernCtx, in_lim) == 8);
const _: () = assert!(std::mem::offset_of!(KernCtx, out_lim) == 16);
const _: () = assert!(std::mem::offset_of!(KernCtx, out_base) == 24);
const _: () = assert!(std::mem::offset_of!(KernCtx, long_tbl) == 32);
const _: () = assert!(std::mem::offset_of!(KernCtx, dist_tbl) == 40);
const _: () = assert!(std::mem::offset_of!(KernCtx, save_bitbuf) == 48);
const _: () = assert!(std::mem::offset_of!(KernCtx, save_bitsleft) == 56);
const _: () = assert!(std::mem::offset_of!(KernCtx, short_tbl) == 64);
const _: () = assert!(std::mem::offset_of!(KernCtx, save_dst) == 72);

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
const _: () = assert!(crate::decompress::inflate::libdeflate_entry::DistTable::TABLE_BITS == 9);

#[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
mod imp {
    use super::{Bits, KernCtx};
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
    /// decodes the distance from the libdeflate-shape `DistTable` (ONE load
    /// + in-register entry decode: `consume_entry` == `shrx` by the entry
    /// low byte, `decode_distance` == `bzhi(saved, total_bits) >> cw_bits`
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
    ///   flat classify (`cmp 255` on trailing, `50f`) ↔ ref `trailing > 255`
    ///   pure-literal pack (lone+multi unified) ↔ ref 8-byte packed store +
    ///                       `*dst += cnt` (one packet/iter, NO chain)
    ///   bottom `6:`/`63:` ↔ ref `< 48` threshold refill
    ///   non-literal arm `50:` (cnt-split lone/multi-trailing)
    ///                     ↔ ref `trailing > 255` arm (EOB/oversize gates,
    ///                       cnt>=2 literal-prefix store, dist single-lookup,
    ///                       validity, X2 restore on bail)
    ///   copy `52:`-`59:`  ↔ marker_inflate.rs `emit_backref_contig`
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
    pub unsafe fn run_contig(ctx: &mut KernCtx, lb: &mut Bits<'_>, dst: *mut u8) -> (u64, *mut u8) {
        #[cfg(test)]
        TEST_RUN_CONTIG_CALLS.fetch_add(1, Ordering::Relaxed);
        let mut bitbuf = lb.bitbuf;
        let mut bitsleft: u64 = lb.bitsleft as u64;
        let mut pos: u64 = lb.pos as u64;
        let mut dst_c = dst;
        let ret: u64;
        unsafe {
            core::arch::asm!(
                // ── prologue: speculative gather of the packet at the
                //    cursor (no consume — harmless table read, exactly the
                //    bits `decode_prefilled` reads).
                "mov {t1:e}, {bitbuf:e}",
                "and {t1:e}, 0xFFF",
                "mov {t1:e}, dword ptr [{short_tbl} + {t1}*4]",
                // ── iteration top: guards (E4) ──────────────────────────
                "2:",
                "mov {ret}, 1",                       // speculative BOUNDARY
                "cmp {dst}, qword ptr [{ctx} + 16]",  // dst vs out_lim
                "jae 9f",
                "cmp {pos}, qword ptr [{ctx} + 8]",   // pos vs in_lim
                "jae 9f",
                // ── FLAT CLASSIFY (igzip/libdeflate fastloop shape): the
                //    preloaded short entry {t1} is dispatched with ONE
                //    data-dependent branch — literal-vs-non-literal on the
                //    TRAILING packed symbol (`cmp 255`, the analog of
                //    igzip's `cmp 0x100`). The LARGE_FLAG (long) and bc==0
                //    (invalid) branches are perfectly-predicted on real
                //    streams (rare, monotone). The old multi-stage cascade
                //    (sym_count branch) AND the speculative literal chain
                //    that RE-RAN the whole classify for the 2nd/3rd literal
                //    are GONE: lone + multi literals are unified into one
                //    flat path, one packet per top-iteration.
                "test {t1:e}, 0x2000000",             // LARGE_FLAG_BIT → long (cold)
                "jnz 20f",
                "mov {t2:e}, {t1:e}",
                "shr {t2:e}, 28",                     // bc = bit_count
                "jz 8f",                              // invalid (bc==0) → Rust (pre-consume)
                "mov {t3:e}, {t1:e}",
                "shr {t3:e}, 26",
                "and {t3:e}, 3",                      // cnt = sym_count (1/2/3)
                "lea {t4:e}, [{t3:e} - 1]",
                "shl {t4:e}, 3",                      // shift = 8*(cnt-1)  (0/8/16)
                "mov {t5:e}, {t1:e}",
                "and {t5:e}, 0x1FFFFFF",              // strip flag/cnt/bc → packed syms only
                "shrx {t5}, {t5}, {t4}",              // trailing symbol → low bits (high bits 0)
                "cmp {t5:e}, 255",
                "ja 50f",                             // trailing non-literal → backref/EOB arm
                // ── pure-literal pack (cnt literals, all ≤255): consume the
                //    whole packet, ONE speculative 8-byte store of the up-to-3
                //    packed bytes (libdeflate fastloop; the bytes above dst
                //    are X3 overshoot, never read back), advance by cnt. Lone
                //    AND multi unified — no sym_count branch, no chain.
                "shrx {bitbuf}, {bitbuf}, {t2}",
                "sub {bitsleft}, {t2}",
                "mov {t4:e}, {t1:e}",
                "and {t4:e}, 0xFFFFFF",               // up to 3 packed literal bytes
                "mov qword ptr [{dst}], {t4}",        // speculative 8-byte store
                "add {dst}, {t3}",                    // advance by cnt
                // ── bottom: `< 48` refill + preload + back-edge ──────────
                "6:",
                "cmp {bitsleft}, 48",
                "jae 63f",
                "mov {t3}, qword ptr [{in_ptr} + {pos}]",
                "shlx {t3}, {t3}, {bitsleft}",
                "or {bitbuf}, {t3}",
                "mov {t4:e}, 63",
                "sub {t4}, {bitsleft}",
                "shr {t4}, 3",
                "add {pos}, {t4}",
                "or {bitsleft}, 56",
                "63:",
                "mov {t1:e}, {bitbuf:e}",
                "and {t1:e}, 0xFFF",
                "mov {t1:e}, dword ptr [{short_tbl} + {t1}*4]",
                "jmp 2b",
                // ── long code at top (decode_prefilled long path; rare) ──
                "20:",
                "mov {t2:e}, {t1:e}",
                "shr {t2:e}, 26",                     // long_max_len (≤21)
                "bzhi {t3}, {bitbuf}, {t2}",
                "shr {t3}, 12",                       // >> ISAL_DECODE_LONG_BITS
                "and {t1:e}, 0x1FFFFFF",
                "add {t1:e}, {t3:e}",                 // long_idx
                "mov {t2}, qword ptr [{ctx} + 32]",   // long_tbl
                "movzx {t1:e}, word ptr [{t2} + {t1}*2]",
                "mov {t2:e}, {t1:e}",
                "shr {t2:e}, 10",                     // bc
                "jz 8f",                              // invalid → Rust
                "and {t1:e}, 0x3FF",                  // symbol
                "cmp {t1:e}, 255",
                "ja 21f",                             // lone non-literal → backref arm
                "shrx {bitbuf}, {bitbuf}, {t2}",      // lone literal via long path
                "sub {bitsleft}, {t2}",
                "mov byte ptr [{dst}], {t1:l}",
                "inc {dst}",
                "jmp 6b",                             // → bottom refill + preload + loop
                "21:",                                // long lone non-literal
                "mov {t5:e}, {t1:e}",                 // trailing = code
                "mov {t3:e}, 1",                      // cnt = 1 (long codes are never packed)
                // fall through to the unified non-literal arm
                // ── unified non-literal arm (replaces lone `50:` + multi
                //    `25:`): t3=cnt, t2=bc, t5=trailing(>255), t1=raw short
                //    entry. Lone backref (cnt==1) and multi-with-trailing-
                //    length (cnt>=2, literal prefix + trailing length) share
                //    the dist+copy body at `58:`. X2 spill/restore makes
                //    every dist-side bail pre-consume.
                "50:",
                "cmp {t5:e}, 256",
                "je 82f",                             // EOB (lone or trailing) → Rust, tag 2
                "cmp {t5:e}, 512",                    // MAX_LIT_LEN_SYM
                "ja 30f",                             // oversize → pre-consume RECLASS
                "mov qword ptr [{ctx} + 48], {bitbuf}",   // X2 spill (incl. dst → 80:)
                "mov qword ptr [{ctx} + 56], {bitsleft}",
                "mov qword ptr [{ctx} + 72], {dst}",
                "shrx {bitbuf}, {bitbuf}, {t2}",      // consume the whole litlen packet
                "sub {bitsleft}, {t2}",
                "cmp {t3:e}, 1",
                "je 31f",                             // lone: no literal prefix
                "mov {t4:e}, {t1:e}",
                "and {t4:e}, 0xFFFFFF",               // packed literal prefix
                "mov qword ptr [{dst}], {t4}",        // speculative prefix store
                "lea {t4:e}, [{t3:e} - 1]",
                "add {dst}, {t4}",                    // advance by lit-prefix (cnt-1)
                "31:",
                "lea {t2:e}, [{t5:e} - 254]",         // length = code - 254 (3..=258)
                "jmp 58f",                            // → shared dist+copy body
                "30:",                                // oversize trailing (>512): pre-consume RECLASS
                "cmp {t3:e}, 1",
                "je 8f",                              // lone oversize → tag 0
                "jmp 83f",                            // multi oversize → tag 3
                // ── shared backref body: t2 = length; X2 state spilled ──
                "58:",
                // dist decode: ONE main-table load + in-register entry decode
                // (libdeflate DistTable shape; subtable → Rust, §5-R1).
                "mov {t3}, qword ptr [{ctx} + 40]",   // dist entries base
                "mov {t1:e}, {bitbuf:e}",
                "and {t1:e}, 0x1FF",                  // DistTable::TABLE_BITS = 9
                "mov {t1:e}, dword ptr [{t3} + {t1}*4]",
                "test {t1:e}, {t1:e}",
                "jz 80f",                             // raw == 0 (hole/code 30/31) → restore
                "test {t1:e}, 0x4000",                // HUFFDEC_SUBTABLE_POINTER
                "jnz 80f",                            // subtable dist → restore
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
                "jz 80f",                             // distance == 0 → restore
                "cmp {t1:e}, 32768",
                "ja 80f",                             // > MAX_WINDOW_SIZE → restore
                "mov {t4}, {dst}",
                "sub {t4}, {t1}",                     // src = dst - distance
                "cmp {t4}, qword ptr [{ctx} + 24]",
                "jb 80f",                             // src < out_base ⇔ distance > *pos → restore
                // X1 pre-copy `< 48` refill (production refills BEFORE the
                // copy — P3.5 c1; the carried gather below is pure).
                "cmp {bitsleft}, 48",
                "jae 51f",
                "mov {t3}, qword ptr [{in_ptr} + {pos}]",
                "shlx {t3}, {t3}, {bitsleft}",
                "or {bitbuf}, {t3}",
                "mov {t5:e}, 63",
                "sub {t5}, {bitsleft}",
                "shr {t5}, 3",
                "add {pos}, {t5}",
                "or {bitsleft}, 56",
                "51:",
                "mov {t3:e}, {bitbuf:e}",
                "and {t3:e}, 0xFFF",
                "mov {t3:e}, dword ptr [{short_tbl} + {t3}*4]",  // carried preload
                // ── copy: emit_backref_contig transliterated ────────────
                //    t1=distance t2=length t4=src {ret}=cursor t5=word
                //    {t3}=carried entry (LIVE across the copy)
                "mov {ret}, {dst}",
                "cmp {t2}, 40",
                "jbe 52f",
                "prefetcht0 byte ptr [{t4} + 40]",    // length > 40 (P3.4 item 3B)
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
                "mov {t1:e}, {t3:e}",                 // carried packet → top classify
                "jmp 2b",
                // ── restore + RECLASS (X2: un-consume the whole packet) ─
                "80:",
                "mov {bitbuf}, qword ptr [{ctx} + 48]",
                "mov {bitsleft}, qword ptr [{ctx} + 56]",
                "mov {dst}, qword ptr [{ctx} + 72]",  // multi path: rolls back lit_prefix
                "mov {ret}, 4",                       // RECLASS, dist-bail tag
                "jmp 9f",
                "82:",
                "mov {ret}, 2",                       // RECLASS, lone-EOB tag
                "jmp 9f",
                "83:",
                "mov {ret}, 3",                       // RECLASS, multi-trailing tag
                "jmp 9f",
                // ── exits ───────────────────────────────────────────────
                "8:",
                "mov {ret}, 0",                       // RECLASS (pre-consume)
                "9:",
                ctx = in(reg) ctx as *mut KernCtx,
                short_tbl = in(reg) ctx.short_tbl,
                in_ptr = in(reg) ctx.in_ptr,
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
            _ => &KERN_EXIT_RECLASS,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    pub fn dump_if_enabled() {
        if !stats_enabled() {
            return;
        }
        eprintln!(
            "[asm-kernel:c] enabled={} entries={} exit_boundary={} exit_reclass={} reclass_eob={} reclass_multi_trail={} reclass_dist={} asm_bytes={}",
            enabled(),
            KERN_ENTRIES.load(Ordering::Relaxed),
            KERN_EXIT_BOUNDARY.load(Ordering::Relaxed),
            KERN_EXIT_RECLASS.load(Ordering::Relaxed),
            KERN_RECLASS_EOB.load(Ordering::Relaxed),
            KERN_RECLASS_MULTI_TRAIL.load(Ordering::Relaxed),
            KERN_RECLASS_DIST.load(Ordering::Relaxed),
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
        let pre = lut.decode_prefilled(lb);
        if pre.bit_count == 0 {
            return EXIT_RECLASS;
        }
        // FLAT CLASSIFY (mirrors the rewritten asm top): extract the TRAILING
        // packed symbol uniformly (`(syms >> 8*(cnt-1)) & 0xFFFF`, after
        // stripping the flag/cnt/bc bits with `& 0x01FF_FFFF`) and dispatch
        // on literal-vs-non-literal with ONE branch. Lone + multi literals
        // are unified; there is NO speculative literal chain (one packet per
        // iteration), exactly as the asm now does.
        let cnt = pre.sym_count as usize;
        let trailing = ((pre.symbol & 0x01FF_FFFF) >> (8 * (cnt - 1))) & 0xFFFF;
        if trailing > 255 {
            // Unified non-literal arm (asm `50:`): EOB/oversize pre-consume;
            // a multi pack stores its literal PREFIX then shares the dist+
            // copy body; every dist-side bail restores the spilled cursor
            // AND dst (X2). `multi_trail*` count only the cnt>=2 path (the
            // asm `25:`-equivalent crossings the coverage gate asserts).
            if trailing == 256 || trailing > super::lut_huffman::MAX_LIT_LEN_SYM {
                return EXIT_RECLASS;
            }
            if cnt >= 2 {
                stats.multi_trail += 1;
            }
            let (sb, sl, sd) = (lb.bitbuf, lb.bitsleft, *dst);
            lb.consume(pre.bit_count);
            if cnt >= 2 {
                let packed = (pre.symbol & 0x00FF_FFFF) as u64;
                out[*dst..*dst + 8].copy_from_slice(&packed.to_le_bytes());
                *dst += cnt - 1;
            }
            let length = trailing as usize - 254;
            let e = dist.lookup(lb.bitbuf);
            let bail = if e.raw() == 0 || e.is_subtable_ptr() {
                true
            } else {
                let saved = lb.bitbuf;
                lb.consume_entry(e.raw());
                let distance = e.decode_distance(saved) as usize;
                if distance == 0 || distance > 32768 || distance > *dst {
                    true
                } else {
                    if (lb.bitsleft as u8) < 48 {
                        lb.refill();
                    }
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
                lb.bitbuf = sb;
                lb.bitsleft = sl;
                *dst = sd;
                return EXIT_RECLASS;
            }
            continue;
        }
        // Pure-literal pack (asm pure-literal path): consume the whole packet,
        // one speculative 8-byte store of the up-to-3 packed bytes, advance by
        // cnt. CONSUME_BIAS != 0 only in the positive-control test (off-by-one
        // model of a wrong asm consume count); 0 == production reference.
        lb.consume(pre.bit_count + CONSUME_BIAS);
        let packed = (pre.symbol & 0x00FF_FFFF) as u64;
        out[*dst..*dst + 8].copy_from_slice(&packed.to_le_bytes());
        *dst += cnt;
        if (lb.bitsleft as u8) < 48 {
            lb.refill();
        }
    }
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
        let short = vec![0u32; 1 << 12];
        let long = vec![0u16; 1264];
        let mut ctx = KernCtx {
            in_ptr: data.as_ptr() as u64,
            in_lim: (data.len() - IN_MARGIN) as u64,
            out_lim: dst as u64 + 32,
            out_base: dst as u64,
            long_tbl: long.as_ptr() as u64,
            dist_tbl: 0,
            save_bitbuf: 0,
            save_bitsleft: 0,
            save_dst: 0,
            short_tbl: short.as_ptr() as u64,
        };
        // Guards pass → RECLASS, state unchanged.
        let (exit, ndst) = unsafe { run_contig(&mut ctx, &mut lb, dst) };
        assert_eq!(exit, EXIT_RECLASS);
        assert_eq!(ndst, dst, "X6: dst unchanged");
        assert_eq!(lb.pos, 17);
        assert_eq!(lb.bitbuf, 0x0123_4567_89AB_CDEF);
        assert_eq!(lb.bitsleft, 56);
        // Out guard fails → BOUNDARY, state unchanged.
        ctx.out_lim = dst as u64;
        let (exit, ndst) = unsafe { run_contig(&mut ctx, &mut lb, dst) };
        assert_eq!(exit, EXIT_BOUNDARY);
        assert_eq!(ndst, dst);
        assert_eq!(lb.pos, 17);
        assert_eq!(lb.bitbuf, 0x0123_4567_89AB_CDEF);
        assert_eq!(lb.bitsleft, 56);
        // In guard fails → BOUNDARY.
        ctx.out_lim = dst as u64 + 32;
        ctx.in_lim = 17;
        let (exit, _) = unsafe { run_contig(&mut ctx, &mut lb, dst) };
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
        let build = |lens: &[u8]| -> LutLitLenCode {
            let mut c = LutLitLenCode::new_empty();
            assert!(c.rebuild_from(lens), "test table lens must be valid");
            c
        };
        let tables = [build(&fixed), build(&dense), build(&longy)];

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
        for (ti, tbl) in tables.iter().enumerate() {
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
                    tbl,
                    &dist_holes,
                    &mut lb_ref,
                    &mut out_ref,
                    &mut dref,
                    out_lim,
                    in_lim,
                );

                let dst0 = out_asm.as_mut_ptr();
                let mut ctx = KernCtx {
                    in_ptr: data.as_ptr() as u64,
                    in_lim: in_lim as u64,
                    out_lim: dst0 as u64 + out_lim as u64,
                    out_base: dst0 as u64,
                    long_tbl: tbl.table.long_code_lookup.as_ptr() as u64,
                    dist_tbl: dist_holes.entries_ptr() as u64,
                    save_bitbuf: 0,
                    save_bitsleft: 0,
                    save_dst: 0,
                    short_tbl: tbl.table.short_code_lookup.as_ptr() as u64,
                };
                let (asm_exit, dst1) = unsafe { run_contig(&mut ctx, &mut lb_asm, dst0) };
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
            let mut ctx = KernCtx {
                in_ptr: data.as_ptr() as u64,
                in_lim: in_lim as u64,
                out_lim: dst0 as u64 + out_lim as u64,
                out_base: dst0 as u64,
                long_tbl: lut.table.long_code_lookup.as_ptr() as u64,
                dist_tbl: dist_holes.entries_ptr() as u64,
                save_bitbuf: 0,
                save_bitsleft: 0,
                save_dst: 0,
                short_tbl: lut.table.short_code_lookup.as_ptr() as u64,
            };
            let (asm_exit, dst1) = unsafe { run_contig(&mut ctx, &mut lb_asm, dst0) };
            let dasm = (dst1 as usize) - (dst0 as usize);

            // Arm B (negative control): bias-0 ref — MUST match the asm on
            // every compared field (the live harness, unmutated).
            let mut out_ref0 = vec![0u8; buf_len];
            let mut lb_ref0 = Bits::new(&data);
            let mut dref0 = 0usize;
            let mut st0 = RefArmStats::default();
            let ref0_exit = run_contig_ref_biased::<0>(
                &lut,
                &dist_holes,
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
                &lut,
                &dist_holes,
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
        let build = |lens: &[u8]| -> LutLitLenCode {
            let mut c = LutLitLenCode::new_empty();
            assert!(c.rebuild_from(lens), "test table lens must be valid");
            c
        };
        let lut_lenny = build(&lenny);
        let lut_fixed = build(&fixed);
        // Dist tables.
        let dist_small = DistTable::build(&[2, 2, 2, 3, 4, 5, 6, 6]).expect("small dist table");
        let dist_fixed5 = DistTable::build(&[5u8; 30]).expect("fixed5 dist table");
        let dist_sub = DistTable::build(&[1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 10, 10])
            .expect("subtable dist table");
        let pairs: [(&LutLitLenCode, &DistTable, &str); 5] = [
            (&lut_lenny, &dist_small, "lenny×small"),
            (&lut_lenny, &dist_fixed5, "lenny×fixed5"),
            (&lut_lenny, &dist_sub, "lenny×sub"),
            (&lut_fixed, &dist_fixed5, "fixed×fixed5"),
            (&lut_fixed, &dist_small, "fixed×small"),
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
        for (pi, (tbl, dt, tag0)) in pairs.iter().enumerate() {
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
                    tbl,
                    dt,
                    &mut lb_ref,
                    &mut out_ref,
                    &mut dref,
                    out_lim,
                    in_lim,
                    &mut cell_stats[pi],
                );

                let base0 = out_asm.as_mut_ptr();
                let dst0 = unsafe { base0.add(WIN) };
                let mut ctx = KernCtx {
                    in_ptr: data.as_ptr() as u64,
                    in_lim: in_lim as u64,
                    out_lim: base0 as u64 + out_lim as u64,
                    out_base: base0 as u64,
                    long_tbl: tbl.table.long_code_lookup.as_ptr() as u64,
                    dist_tbl: dt.entries_ptr() as u64,
                    save_bitbuf: 0,
                    save_bitsleft: 0,
                    save_dst: 0,
                    short_tbl: tbl.table.short_code_lookup.as_ptr() as u64,
                };
                let (asm_exit, dst1) = unsafe { run_contig(&mut ctx, &mut lb_asm, dst0) };
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
