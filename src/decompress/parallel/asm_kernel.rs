//! ASM-campaign rung (c) — the FULL contig clean fast loop as ONE
//! `core::arch::asm!` region (plans/asm-campaign.md §2(c), §5 VAR_VIII salvage).
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
//!      backref arm consumes litlen bits before its rare checks; it spills
//!      `(bitbuf, bitsleft)` at arm entry and RESTORES them before
//!      EXIT_RECLASS (`pos`/`dst` untouched by then — first refill in that
//!      arm comes after all rare checks).
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
//!  * `EXIT_RECLASS` (0): the next packet needs Rust — lone EOB,
//!    `bit_count == 0` (invalid), symbol > MAX_LIT_LEN_SYM, a multi-symbol
//!    packet whose trailing element is non-literal (builder-impossible,
//!    kept for defense), a backref whose dist decode hit `raw == 0` /
//!    `distance > 32768` / `distance > *pos` (restored, X2), and — until
//!    stage c3 lands the backref arm — every lone length code. The caller
//!    leaves `asm_on = true`; the Rust loop handles exactly that packet and
//!    the next `continue 'fast` re-enters the asm.
//!
//! ## IN_MARGIN proof (no slow refill inside the asm)
//! One asm iteration issues at most 4 refills (chain arm: ≤ 3 `decode()`
//! backstops — 2 consumed extras + 1 carried — plus the post-chain `< 48`
//! threshold; the backref/litpack arms issue ≤ 1). Each fast refill
//! advances `pos` by ≤ 7 and reads 8 bytes, so the deepest read inside an
//! iteration touches `pos_top + 4*7 + 8 = pos_top + 36 <= pos_top +
//! IN_MARGIN - 1 < in_end` for `IN_MARGIN = 40` (guard: `pos_top + 40 <=
//! in_end - 1`... `pos_top < in_lim = in_end - 40`). Hence `pos + 8 <= len`
//! at every asm refill — the Rust fast form, bit-for-bit, and the `> 64`
//! underflow check is dead (E2: bitsleft ≤ 63 always).
//!
//! ## Store-shape equivalence (c2/c3)
//! Lone literal: 1-byte store (advisor Q3 — never the 8-byte store).
//! Multi-literal: the 8-byte speculative store of `sym0 & 0x00FF_FFFF`,
//! advance by `sym_count` — identical to the Rust packed arm. Backref (c3):
//! the P3.4 `emit_backref_contig` shape transliterated (dist ≥ 8 burst-5 +
//! stride-8 words; dist == 1 RLE broadcast word fill; 2..=7 stride-dist
//! words; `length > 40` prefetch) — NOT VAR_VIII's byte-copy-all-overlaps D
//! loop (refuted by P3.4's −87 ms).
//!
//! ## Dispatch policy (charter §4)
//! Cargo feature `asm-kernel`, x86_64-only call sites; pure-Rust loop ALWAYS
//! compiled and reachable. Runtime: BMI2 detect (`shrx`/`shlx`/`bzhi`) +
//! `GZIPPY_ASM_KERNEL=0` kill-switch (OnceLock, one predictable branch per
//! contig call). Default-OFF until the ship gate passes frozen.

#![allow(dead_code)]

#[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
use crate::decompress::inflate::consume_first_decode::Bits;

/// Stricter input guard margin (bytes) — see IN_MARGIN proof in the module
/// doc. The Rust loop's own guard is `FAST_IN_SLOP = 8`.
pub const IN_MARGIN: usize = 40;

/// Exit codes (module doc "Exit codes").
pub const EXIT_RECLASS: u64 = 0;
pub const EXIT_BOUNDARY: u64 = 1;

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
    use super::{Bits, KernCtx, EXIT_BOUNDARY};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    /// Effect/coverage counters (decide.sh EFFECT predicate + the asm_frac
    /// validity precondition F-c requires). Wrapper-side only (X4), gated on
    /// `stats_enabled()` so the measured hot path carries zero atomics.
    pub static KERN_ENTRIES: AtomicU64 = AtomicU64::new(0);
    pub static KERN_EXIT_BOUNDARY: AtomicU64 = AtomicU64::new(0);
    pub static KERN_EXIT_RECLASS: AtomicU64 = AtomicU64::new(0);
    pub static KERN_ASM_BYTES: AtomicU64 = AtomicU64::new(0);

    /// Runtime dispatch: ON when compiled in, unless `GZIPPY_ASM_KERNEL=0`
    /// (kill-switch) or the CPU lacks BMI2 (`shrx`/`shlx`/`bzhi`).
    pub fn enabled() -> bool {
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
    /// STAGE c1 (this commit): the seam itself — the asm carries the FULL
    /// pinned-operand contract (bit cursor in registers, dst cursor, ctx,
    /// short-table base, scratch + ret) and executes ONLY the top guards,
    /// then exits: `EXIT_BOUNDARY` when a guard fails, `EXIT_RECLASS`
    /// otherwise, with NO state change (X6: `dst` returns unchanged). This
    /// proves the operand/clobber contract and the caller-side state
    /// round-trip byte-exact under the production integration — every
    /// packet crosses the seam in the `GZIPPY_ASM_KERNEL` ON arm and output
    /// must stay identical. Stages c2/c3 grow the body inside the SAME
    /// contract.
    ///
    /// # Safety
    /// Caller upholds E1-E6 (module doc). The c1 body only reads `ctx`
    /// fields and registers.
    #[inline]
    pub unsafe fn run_contig(ctx: &mut KernCtx, lb: &mut Bits<'_>, dst: *mut u8) -> (u64, *mut u8) {
        let mut bitbuf = lb.bitbuf;
        let mut bitsleft: u64 = lb.bitsleft as u64;
        let mut pos: u64 = lb.pos as u64;
        let mut dst_c = dst;
        let ret: u64;
        unsafe {
            core::arch::asm!(
                // ── iteration top: guards (E4) ──────────────────────────
                "2:",
                "mov {ret}, 1",                       // speculative BOUNDARY
                "cmp {dst}, qword ptr [{ctx} + 16]",  // dst vs out_lim
                "jae 9f",
                "cmp {pos}, qword ptr [{ctx} + 8]",   // pos vs in_lim
                "jae 9f",
                // c1: hand every in-guard packet to Rust, state unchanged.
                "mov {ret}, 0",                       // RECLASS
                // Dead uses pinning the FULL c2/c3 operand contract (asm!
                // rejects unreferenced named operands; these zero scratch
                // regs / touch flags only — state operands are not written).
                "test {bitbuf}, {bitbuf}",
                "test {bitsleft}, {bitsleft}",
                "test {short_tbl}, {short_tbl}",
                "xor {t1:e}, {t1:e}",
                "xor {t2:e}, {t2:e}",
                "xor {t3:e}, {t3:e}",
                "xor {t4:e}, {t4:e}",
                "9:",
                ctx = in(reg) ctx as *mut KernCtx,
                short_tbl = in(reg) ctx.short_tbl,
                bitbuf = inout(reg) bitbuf,
                bitsleft = inout(reg) bitsleft,
                pos = inout(reg) pos,
                dst = inout(reg) dst_c,
                ret = out(reg) ret,
                t1 = out(reg) _,
                t2 = out(reg) _,
                t3 = out(reg) _,
                t4 = out(reg) _,
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
        if exit == EXIT_BOUNDARY {
            KERN_EXIT_BOUNDARY.fetch_add(1, Ordering::Relaxed);
        } else {
            KERN_EXIT_RECLASS.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn dump_if_enabled() {
        if !stats_enabled() {
            return;
        }
        eprintln!(
            "[asm-kernel:c] enabled={} entries={} exit_boundary={} exit_reclass={} asm_bytes={}",
            enabled(),
            KERN_ENTRIES.load(Ordering::Relaxed),
            KERN_EXIT_BOUNDARY.load(Ordering::Relaxed),
            KERN_EXIT_RECLASS.load(Ordering::Relaxed),
            KERN_ASM_BYTES.load(Ordering::Relaxed),
        );
    }
}

#[cfg(all(feature = "asm-kernel", target_arch = "x86_64"))]
pub use imp::{dump_if_enabled, enabled, note_exit, run_contig, stats_enabled};

/// Non-asm builds: constant-false dispatch, no-op dump — call sites fold away.
#[cfg(not(all(feature = "asm-kernel", target_arch = "x86_64")))]
pub fn dump_if_enabled() {}

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
}
