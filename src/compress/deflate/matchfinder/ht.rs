//! Hash-table matchfinder: 2-entry inline buckets PLUS a length-3 table.
//!
//! Port of `vendor/libdeflate/lib/ht_matchfinder.h` (Eric Biggers, 2022) at
//! `HT_MATCHFINDER_BUCKET_SIZE == 2` — the hand-unrolled arm libdeflate ships at
//! level 1 — **with a `hash3_tab` added in the shape libdeflate's OWN
//! `hc_matchfinder` uses at levels 2-9.** That combination is the point of this
//! module and no vendor ships it.
//!
//! # Why, measured
//!
//! `fulcrum why libdeflate:data.csv:L1:T1:size` (the automated vendor diff,
//! structure layer, 26,500,000 B input) found the L1 gap is the PARSE, not block
//! sizing and not table quality — header bits agreed within 0.37%:
//!
//! | | ours (`parse::fast`) | libdeflate L1 |
//! |---|---|---|
//! | matches | 1,846,129 | **1,930,665** (+4.58%) |
//! | literals | **741,183** | 256,099 (**+189.41%**) |
//! | input covered by matches | 97.20% | **99.03%** |
//!
//! A pure `ht_matchfinder` port then closed 9 libdeflate L1 cells — every one to
//! ratio EXACTLY 1.0000 — and OPENED 7, because `ht_matchfinder` deliberately has
//! no length-3 table ("Due to its focus on speed, the ht_matchfinder doesn't
//! support length 3 matches") and three BINARIES were files where our `head3`
//!
//! So the two properties are complementary, not alternatives: **length-3 matches
//! earn bytes on binaries; 2-way bucketing earns far more on text and structured
//! data.** libdeflate has buckets without hash3 at L1 and hash3 without buckets
//! at L2-9; `parse::fast` has hash3 with a single probe. This has both.
//!
//! # Working set — the arithmetic that makes this a REOPEN
//!
//! `c0f69036` recorded a FROZEN ship gate on `17283ee6` ("insert-depth=8 +
//! bucket2(gate=64)"): SIZE PASSED — "a genuine, confirmed, LARGE ratio-only win"
//! — and WALL FAILED DECISIVELY at a 12-29% self-tax, ~26 standard deviations. Its
//! stated reopen condition is "a materially cheaper way to get the same length-3-8
//! reach / second-candidate signal". Cheaper is the whole claim:
//!
//! | | `parse::fast` (shipped) | falsified `17283ee6` | this |
//! |---|---|---|---|
//! | 4-byte table | `head` 64 K x u32 = 256 KiB | same, kept | **2-way `[[i16;2]; 32 K]` = 128 KiB** |
//! | 3-byte table | `head3` = 128 KiB | same, kept | **`[i16; 32 K]` = 64 KiB** |
//! | second candidate | none | ADDED, gated at 64 | inline in the bucket, ungated |
//! | insert depth | 3 | raised to 8 | 1 per position |
//! | **total** | **384 KiB** | **384 KiB + a third probe** | **192 KiB — exactly half** |
//!
//! Half the memory touched per position, two 4-byte candidates instead of one, and
//! one fewer probe site than the falsified attempt. The `i16` position encoding is
//! what pays for it: libdeflate's `mf_pos_t` is 2 bytes against our `u32` heads.
//!
//! It is also NOT a content detector, which matters because the competing lever is:
//! both tables are read and written at EVERY position with no data-dependent
//! branch, so there is nothing to gate and no threshold to fit. That is why this
//! route can retire `L1_HASH3_GATE_LIT_THRESHOLD_PCT` — a constant fitted two
//! points off a 2-point-wide cliff on the single file `dd79_bin6`, which
//! `CLAUDE.md` non-negotiable #3 orders deleted — rather than join it.
//!
//! **NO WALL CLAIM IS MADE HERE.** The prior falsification in this class died on
//! wall, so the working-set arithmetic above is a REASON TO MEASURE, not evidence.
//! Size is deterministic and free and runs first; a size win obliges a frozen
//! paired wall run on solvency before anything ships.

//! # MEASURED COST PROFILE — the wall regression is WRITE traffic
//!
//! ⚠ **READ THE VENDOR SECTION BELOW BEFORE USING THIS TABLE.** Both arms here are
//! OURS. This table compares this finder against `parse::fast`, our own previous
//! path — it is not, and never was, a comparison against libdeflate. It says how
//! much more this finder costs THAN US. It cannot say whether that cost is
//! intrinsic to the algorithm, and it was read that way for two falsifications.
//!
//! Cachegrind, 6,000,000 B of data.csv at L1 T1 on Zen2, shipped `parse::fast` vs this
//! finder:
//!
//! | | `parse::fast` | this | ratio |
//! |---|---|---|---|
//! | I refs | 92,057,699 | 190,151,913 | 2.07x |
//! | D reads | 17,484,786 | 33,552,458 | 1.92x |
//! | **D writes** | **7,320,227** | **26,927,232** | **3.68x** |
//! | D1 misses | 799,400 | 2,065,774 | 2.58x |
//!
//! # THE VENDOR COMPARISON, measured 2026-08-01
//!
//! The dispatch arm in `parse/mod.rs` cites "the 2.19x instruction gap". **That number
//! IS defined and IS correct**: commit `df475791` (#194) measured "the finder executes
//! 201,359,346 instructions against `parse::fast`'s 92,057,657 — 2.19x, from probing
//! twice and hashing a third table" (201,359,346 / 92,057,657 = 2.1873).
//!
//! ⚠ RETRACTED, same day: an earlier version of this section said "**That number had
//! no defining measurement anywhere in `src/` or `docs/`** — grep it." That is FALSE,
//! and the way it went wrong is worth more than the claim was: **the grep covered the
//! WORKING TREE, and this project keeps its receipts in COMMIT MESSAGES.** `git log
//! -S2.19 --all` finds it in two seconds. Before asserting that a record does not
//! exist, search the history, not just `src/` and `docs/`.
//!
//! What is true, and is the only reason this section exists: 2.19x is `ht` against
//! `parse::fast` — OURS against OURS, like the 2.07x table above. No comparison
//! against libdeflate had been run. That is what follows.
//!
//! And #194 named the MECHANISM correctly at the time — "probing twice and hashing a
//! third table". The ablation below confirms it exactly: the third table is
//! 54,690,301 Ir. #194 was right; it simply never priced the alternative.
//!
//! Cachegrind, the SAME 6,000,000 B of data.csv, L1, `-p1` (see the
//! thread trap below), trainer/Intel, `libdeflate-gzip` built with `-g`:
//!
//! | arm | total Ir | output |
//! |---|---|---|
//! | this finder, `HT_MAX_LEN3_OFFSET = 4096` | 196,854,205 | 881,712 |
//! | this finder, `HT_MAX_LEN3_OFFSET = 0` | 198,467,459 | 881,550 |
//! | **this finder, off=0 AND hash3 MAINTENANCE ablated** | **143,777,158** | **881,550** |
//! | **libdeflate `-1`** | **146,098,195** | **881,550** |
//!
//! **THERE IS NO IMPLEMENTATION GAP. Our faithful `ht_matchfinder` port executes 1.6%
//! FEWER instructions than the vendor's C, on byte-identical output.** The entire
//! excess is one feature: the length-3 table costs **54,690,301 Ir, 104% of the whole
//! 52.4M difference**. `lz_hash` costs 2.0 Ir/call on BOTH sides (libdeflate
//! 11,999,974 Ir / 5,999,987 calls; ours 24.38M / 11,999,974 calls) — we simply call
//! it twice per position, once for the 4-byte key and once for the 3-byte key.
//!
//! ⚠ THE ABLATION IS THE ONLY VALID WAY TO PRICE THIS TABLE, and the reason is the
//! trap that produced a false record here on 2026-08-01 (retracted below):
//! `HT_MAX_LEN3_OFFSET = 0` disables length-3 ACCEPTANCE, not MAINTENANCE. At off=0
//! the second `lz_hash`, the load, the store and the extra 64 KiB rebase per window
//! slide are all still paid, for a table that is never read. Only removing the
//! maintenance prices the feature, and doing it AT off=0 keeps the output identical
//! by construction, so the ablation cannot change what is being compared.
//!
//! ⚠ RETRACTED, same day, by the ablation above: this note previously said "**The
//! length-3 table is not the tax.** Turning it off made us 1.6M instructions SLOWER
//! (198.47M vs 196.85M) ... Any future note blaming hash3 for the L1 wall must beat
//! this measurement first." That is FALSE, and it is false for a nameable reason
//! rather than by bad luck: off=0 was read as "hash3 off" when it only means "hash3
//! unused". The 1.6M is real but it measures the cost of length-3 matches NOT BEING
//! TAKEN while still being maintained; it says nothing about the table's cost. The
//! table's cost is 54.69M.
//!
//! It also previously claimed the gap was "1.35x total and 1.60x in the matchfinder,
//! 89% of it inside the matchfinder". The 1.60x was not apples-to-apples: libdeflate's
//! matchfinder is `ht_matchfinder.h` 77.02M PLUS `matchfinder_common.h` 18.87M,
//! `common_defs.h` 2.81M, `x86/matchfinder_impl.h` 2.64M and `emmintrin.h` 3.00M =
//! 104.34M, against our 151.17M across `ht.rs`, `common.rs`, `uint_macros.rs`,
//! `ptr/mod.rs`, `sse.rs` and `slice/iter/macros.rs`. Like for like that is 1.45x,
//! and the ablation shows even that is the feature and not the implementation.
//!
//! What DOES survive: at `HT_MAX_LEN3_OFFSET = 0` our payload is BYTE-IDENTICAL to
//! libdeflate's — sha256 `62a4e450aca4b522`, full 6 MB multi-block file. Insert counts
//! match exactly (5,507,374 both sides), confirming an identical parse. That is why
//! the ablation is sound, and it is the one claim here that was never in doubt.
//!
//! # THE WALL IS NOT THE INSTRUCTIONS — it is LLC traffic, and `main` already wins
//!
//! `perf stat`, FULL data.csv (26,500,039 B), `taskset -c 2`, L1 T1:
//!
//! | arm | Ir | cycles | IPC | LLC miss | wall |
//! |---|---|---|---|---|---|
//! | this finder, ablated (byte-identical to libdeflate) | 678.1M | 287.2M | 2.36 | **1.366M** | 209.6 ms |
//! | `main` (`parse::fast`) | 444.6M | 242.4M | 1.83 | 1.346M | **176.4 ms** |
//! | libdeflate `-1` | 657.4M | 242.1M | 2.71 | **0.579M** | 176.2 ms |
//!
//! At 1.032x libdeflate's instructions we take 1.186x their cycles, on 2.36x their LLC
//! misses. **`main` is ALREADY at wall parity with libdeflate on this cell**, so routing
//! this finder converts a PASSING wall cell into a ~1.19-1.29x failure — clause 3 is
//! absolute, and closing every instruction named above still leaves ~213 ms against
//! `main`'s 177 ms. Instruction elasticity here is ~0.22, not 1: adding the 53M hash3
//! instructions RAISED IPC to 2.89 and cost only 8% wall.
//!
//! The excess LLC misses scale linearly with input (~2 extra input-sized passes:
//! armexe 3.35, data.csv 1.94, tool.bin 2.15 excess bytes per input byte) and `main`
//! carries the SAME fixed excess. So the L1 wall deficit lives in the buffering/IO
//! path that both arms share — not in the matchfinder, and not in this finder.
//!
//! ⚠ THREAD TRAP: `gzippy -1 -c FILE` with no `-p` uses `num_cpus`. On the 16-core
//! trainer that silently measures T16. Both arms above are `-p1`; the T16 output on
//! the same slice is 881,816, not 881,712, which is how the trap announces itself.
//!
//! ⚠ INSTRUMENT: `fulcrum why`'s layer [2] callgrind reported 2,406,284,393 Ir for
//! the first arm — **12.22x the cachegrind figure** (2,406,284,393 / 196,854,205),
//! matching the recorded parser inflation bug, with `Dr 0` on every row as the
//! corroborating tell. Layer [1] (position counts) is independent of that path and
//! was used; layer [2] was not. (This said "12.66x the cachegrind figure", which is
//! the ratio against the BANKED 190,151,913, not against the cachegrind run it names.
//! Corrected rather than left, because a mislabelled ratio inside a record-hygiene
//! note is the same defect the note exists to fix.)
//!
//! Reads at 1.92x are the expected price of a second candidate. **Writes at 3.68x are
//! the regression**: a 2-entry bucket costs TWO stores per insert (shift + head) and
//! the length-3 table a third, so inserting at every interior position of a ~14-byte
//! average match is ~42 stores per match against `parse::fast`'s ~3 (it ships igzip's
//! `LIMIT_HASH_UPDATE_INSERTS_L1 == 3`).
//!
//! **And the obvious fix is not available.** Limiting the inserts was tried and gives
//!
//! # The elided bounds checks bought NOTHING — measured, keep them elided anyway
//!
//! So this is NOT the explanation for the L1 wall regression recorded at the
//! `Strategy::Fast` dispatch arm, and the next person should not re-try it. The form
//! below is kept because it matches [`super::hc`] and libdeflate and costs nothing —
//! not because it was worth anything.
//!
//! # Soundness of the elided bounds checks
//!
//! The hot loop drops Rust's bounds checks to match libdeflate's C codegen, exactly as
//! [`super::hc`] does and for the same reasons. Every elided check is discharged by
//! construction, not by inspection:
//!
//! * **Bucket / length-3 indices.** `hash = lz_hash(seq, HT_HASH_ORDER)` with
//!   `HT_HASH_ORDER == 15`, so `hash < 2^15 == HT_TAB_LEN`; `hash3 = lz_hash(seq &
//!   0xFF_FFFF, HT_HASH3_ORDER)` with the same order, so `hash3 < 2^15 ==
//!   HT_HASH3_SIZE`. Both come straight out of `lz_hash` — never out of arithmetic —
//!   so `hash_tab`/`hash3_tab` `get_unchecked[_mut]` are in bounds. `debug_assert!`s
//!   pin this in test builds.
//! * **Buffer reads.** The parser pads its working buffer by `BUF_PAD` (16) bytes past
//!   `in_end` and the caller guarantees `max_len >= HT_REQUIRED_NBYTES`; see the
//!   per-call SAFETY comment in [`HtMatchfinder::longest_match`].

use super::common::{
    load_u24, load_u32, lz_extend, lz_hash, matchfinder_rebase, prefetch_write,
    MATCHFINDER_INITVAL, MATCHFINDER_WINDOW_SIZE,
};

/// `HT_MATCHFINDER_HASH_ORDER`.
pub const HT_HASH_ORDER: u32 = 15;
/// `HT_MATCHFINDER_BUCKET_SIZE`. The port below is the hand-unrolled `== 2` arm;
/// changing this constant alone does NOT change the algorithm.
pub const HT_BUCKET_SIZE: usize = 2;
/// Number of buckets.
pub const HT_TAB_LEN: usize = 1 << HT_HASH_ORDER;
/// `HT_MATCHFINDER_MIN_MATCH_LEN`. Asserted throughout the port, exactly as
/// libdeflate's `STATIC_ASSERT` does: the 4-byte `seq` compare depends on it.
pub const HT_MIN_MATCH_LEN: u32 = 4;
/// `HT_MATCHFINDER_REQUIRED_NBYTES` — minimum `max_len` for [`HtMatchfinder::longest_match`].
pub const HT_REQUIRED_NBYTES: u32 = 5;

/// Hash order for the LENGTH-3 singleton table. Same order libdeflate's
/// `hc_matchfinder` uses for its own `hash3_tab` (`HC_HASH3_ORDER` 15).
pub const HT_HASH3_ORDER: u32 = 15;
/// Entries in [`HtMatchfinder::hash3_tab`]. 32,768 x 2 B = 64 KiB.
pub const HT_HASH3_SIZE: usize = 1 << HT_HASH3_ORDER;
/// Longest length-3 match distance worth coding. A length-3 match at a large
/// offset costs more bits than three literals, so libdeflate's greedy parser
/// guards it with `length > DEFLATE_MIN_MATCH_LEN || offset <= 4096`
/// (`deflate_compress.c` `deflate_compress_greedy`). `deflate_compress_fastest`
/// has no such guard only because it has no length-3 matches to guard; adding
/// them means adopting the guard with them.
pub const HT_MAX_LEN3_OFFSET: u32 = 4096;

/// `struct ht_matchfinder`. 128 KiB of inline `[i16; 2]` buckets.
///
/// Inline fixed-size array rather than a `Vec`, for the same reason
/// [`super::hc::HcMatchfinder`] does it: a bucket read becomes `self +
/// const_offset + i*4` in one addressing mode with no table-base register tied
/// up. The struct is >64 KiB, so it is always heap-boxed and never constructed
/// or passed by value.
pub struct HtMatchfinder {
    hash_tab: [[i16; HT_BUCKET_SIZE]; HT_TAB_LEN],
    /// Singleton nodes for LENGTH-3 matches, in the shape libdeflate's
    /// `hc_matchfinder` uses. `ht_matchfinder` has no such table — that is the
    /// documented reason it "doesn't support length 3 matches" — and adding it is
    /// the whole point of this variant. See the module doc.
    hash3_tab: [i16; HT_HASH3_SIZE],
    /// Whether length-3 matches may be ACCEPTED for the block being parsed.
    ///
    /// Set per block by the parser from libdeflate's own literal-count rule
    /// ([`super::super::parse::choose_min_match_len`]): a length-3 match is worth
    /// taking only when literals are expensive, and the number of DISTINCT
    /// literals in the data is libdeflate's proxy for that. Their
    /// `deflate_compress_fastest` needs no such rule because its matchfinder has
    /// no length-3 table to govern; we have one, so we need the rule they apply
    /// at levels 2-9.
    ///
    /// The table is MAINTAINED regardless of this flag, so a block that re-enables
    /// length-3 matching sees a fully populated table. Entries surviving from a
    /// disabled block are still sound: every candidate is re-validated against
    /// `cutoff` and by the 3-byte compare, so a stale entry can only produce a
    /// genuine match or no match at all.
    allow_len3: bool,
}

thread_local! {
    /// One recycled `HtMatchfinder` per thread — see [`HtMatchfinder::acquire`].
    /// Thread-local rather than shared: no cross-thread mutable state, no
    /// synchronization, and T>1 chunk independence is preserved because every
    /// chunk still starts from a fully re-armed table.
    static HT_POOL: std::cell::RefCell<Option<Box<HtMatchfinder>>> =
        const { std::cell::RefCell::new(None) };
}

/// RAII handle from [`HtMatchfinder::acquire`]; returns its box to [`HT_POOL`]
/// on drop instead of freeing it.
pub struct PooledHt(Option<Box<HtMatchfinder>>);

impl std::ops::Deref for PooledHt {
    type Target = HtMatchfinder;
    #[inline]
    fn deref(&self) -> &HtMatchfinder {
        // SAFETY/invariant: `Some` for a `PooledHt`'s whole lifetime outside
        // `Drop::drop`, which is the only place it is taken.
        self.0.as_deref().expect("PooledHt used after drop")
    }
}

impl std::ops::DerefMut for PooledHt {
    #[inline]
    fn deref_mut(&mut self) -> &mut HtMatchfinder {
        self.0.as_deref_mut().expect("PooledHt used after drop")
    }
}

impl Drop for PooledHt {
    fn drop(&mut self) {
        if let Some(b) = self.0.take() {
            HT_POOL.with(|cell| {
                *cell.borrow_mut() = Some(b);
            });
        }
    }
}

impl HtMatchfinder {
    /// `ht_matchfinder_init`: allocate and set every entry to the sentinel.
    ///
    /// Built through `Box::new_uninit` so no 128 KiB temporary lands on the
    /// stack; every entry is written before `assume_init`.
    pub fn new() -> Box<Self> {
        let mut boxed = Box::<Self>::new_uninit();
        // SAFETY: `new_uninit` gives one aligned, fully-owned `HtMatchfinder`.
        // Every `i16` of the only field is written with `MATCHFINDER_INITVAL`
        // (the exact value `matchfinder_init` writes — a `-WINDOW_SIZE`
        // sentinel, NOT zero) before `assume_init`. `addr_of_mut!` avoids
        // forming a reference to uninit memory, and each `.add(i)` for
        // `i < HT_TAB_LEN * HT_BUCKET_SIZE` stays inside the field.
        unsafe {
            let p = boxed.as_mut_ptr();
            let tab = core::ptr::addr_of_mut!((*p).hash_tab) as *mut i16;
            for i in 0..(HT_TAB_LEN * HT_BUCKET_SIZE) {
                tab.add(i).write(MATCHFINDER_INITVAL);
            }
            let t3 = core::ptr::addr_of_mut!((*p).hash3_tab) as *mut i16;
            for i in 0..HT_HASH3_SIZE {
                t3.add(i).write(MATCHFINDER_INITVAL);
            }
            // MUST be written before `assume_init`: `new_uninit` gives uninitialised
            // memory and a `bool` holding anything other than 0 or 1 is UB the moment
            // it is read. `false` is the conservative arm — a parser that forgets to
            // call `set_allow_len3` gets libdeflate's own no-length-3 behaviour, not
            // a coin flip.
            core::ptr::addr_of_mut!((*p).allow_len3).write(false);
            boxed.assume_init()
        }
    }

    /// Re-arm every entry in place, reusing the existing 128 KiB allocation.
    /// Same postcondition as [`Self::new`].
    fn reset(&mut self) {
        // One flat fill: `[[i16; 2]; N]` is contiguous, so this is a single
        // memset-shaped loop rather than N two-element fills.
        let flat: &mut [i16] = self.as_flat_mut();
        flat.fill(MATCHFINDER_INITVAL);
        self.hash3_tab.fill(MATCHFINDER_INITVAL);
    }

    /// The bucket array viewed as one contiguous `[i16]`, for `fill`/rebase.
    #[inline]
    fn as_flat_mut(&mut self) -> &mut [i16] {
        // SAFETY: `[[i16; HT_BUCKET_SIZE]; HT_TAB_LEN]` is a contiguous array
        // of `i16` with no padding (arrays are laid out contiguously and `i16`
        // has no alignment slack inside `[i16; 2]`), so reinterpreting it as
        // `[i16; HT_TAB_LEN * HT_BUCKET_SIZE]` is layout-identical. The length
        // is exactly the element count, and the borrow is unique.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.hash_tab.as_mut_ptr() as *mut i16,
                HT_TAB_LEN * HT_BUCKET_SIZE,
            )
        }
    }

    /// Pooled equivalent of [`Self::new`] — a table reset to its sentinel
    /// state, reusing this thread's allocation when it already has one.
    ///
    /// `run()` is invoked once per CHUNK on the T>1 path and chunk count
    /// exceeds thread count, so without pooling every chunk would pay a fresh
    /// 128 KiB allocation plus sentinel fill. That per-`run()` cadence is a
    /// documented waste class in this codebase (DHAT: "1x / 9-28x = chunk
    /// count"); this avoids re-introducing it.
    pub fn acquire() -> PooledHt {
        let existing = HT_POOL.with(|cell| cell.borrow_mut().take());
        let mut mf = existing.unwrap_or_else(Self::new);
        mf.reset();
        PooledHt(Some(mf))
    }

    /// Set whether [`Self::longest_match`] may return a length-3 match.
    ///
    /// Called once per DEFLATE block by the parser, which owns the bit-cost
    /// question; the matchfinder only enforces the answer. See the field's doc.
    #[inline]
    pub fn set_allow_len3(&mut self, allow: bool) {
        self.allow_len3 = allow;
    }

    /// `ht_matchfinder_slide_window`: rebase every stored position by one window.
    #[inline]
    fn slide_window(&mut self) {
        matchfinder_rebase(self.as_flat_mut());
        matchfinder_rebase(&mut self.hash3_tab[..]);
    }

    /// `ht_matchfinder_longest_match`. Returns `(best_len, offset)`; `best_len ==
    /// 0` means no match and the offset is meaningless.
    ///
    /// Faithful to the `BUCKET_SIZE == 2` hand-unrolled arm, including its two
    /// deliberate quirks, both of which are load-bearing for byte-for-byte
    /// agreement with libdeflate's parse decisions:
    ///   * entry 0 is copied to entry 1 **even when `nice_len` is reached on the
    ///     first candidate** (libdeflate: "this can be slightly faster");
    ///   * the second candidate is pre-screened by comparing the 4 bytes ending
    ///     at `best_len` BEFORE calling `lz_extend`, so a candidate that cannot
    ///     beat the incumbent is rejected without extending.
    ///
    /// Contract, inherited from the C: `max_len >= HT_REQUIRED_NBYTES` (5), and
    /// `buf` is padded so the 4-byte loads at `in_next + 1` and at
    /// `in_next + best_len - 3` stay in bounds.
    #[inline(always)]
    pub fn longest_match(
        &mut self,
        buf: &[u8],
        in_base: &mut usize,
        in_next: usize,
        max_len: u32,
        nice_len: u32,
        next_hash: &mut u32,
        next_hash3: &mut u32,
    ) -> (u32, u32) {
        debug_assert!(max_len >= HT_REQUIRED_NBYTES);
        debug_assert_eq!(HT_MIN_MATCH_LEN, 4);

        let mut best_len: u32 = 0;
        let mut best_match_pos = in_next;
        let mut cur_pos = (in_next - *in_base) as i32;

        if cur_pos == MATCHFINDER_WINDOW_SIZE {
            self.slide_window();
            *in_base += MATCHFINDER_WINDOW_SIZE as usize;
            cur_pos = 0;
        }
        let in_base_now = *in_base;
        let cutoff = cur_pos - MATCHFINDER_WINDOW_SIZE;

        // SAFETY for every `load_u32` below. `load_u32(base, off)` reads 4 bytes
        // at `off`; all offsets used here are `in_next`, `in_next + 1`,
        // `in_next + best_len - 3`, `match_pos`, and `match_pos + best_len - 3`.
        //   * `in_next + 1 + 4 <= buf.len()` because the caller guarantees
        //     `max_len >= HT_REQUIRED_NBYTES == 5` bytes remain at `in_next`,
        //     and `buf` additionally carries `BUF_PAD` (16) trailing pad bytes.
        //   * `best_len <= max_len`, so `in_next + best_len` is in bounds and
        //     `in_next + best_len - 3 + 4 = in_next + best_len + 1` is too. The
        //     `- 3` cannot underflow: this expression is only evaluated after
        //     `best_len` was set by `lz_extend` starting at `HT_MIN_MATCH_LEN`
        //     (4), so `best_len >= 4`.
        //   * `match_pos = in_base_now + cur_node` with `cur_node > cutoff`, and
        //     it is a position we previously inserted, hence `< in_next`; the
        //     same pad argument therefore covers `match_pos + best_len + 1`.
        let base = buf.as_ptr();
        let seq = unsafe { load_u32(base, in_next) };
        // The next position's hashes are computed HERE, one position ahead, so the
        // caller's loop never recomputes them — this is what makes
        // HT_REQUIRED_NBYTES 5 rather than 4. Both keys come from ONE 4-byte load
        // of `in_next + 1`, so the length-3 table costs no extra input read.
        let hash = *next_hash as usize;
        let next_seq = unsafe { load_u32(base, in_next + 1) };
        *next_hash = lz_hash(next_seq, HT_HASH_ORDER);
        debug_assert!((*next_hash as usize) < HT_TAB_LEN);
        // Prefetch the bucket the NEXT position will touch, matching
        // libdeflate's `prefetchw(&mf->hash_tab[*next_hash])`.
        // SAFETY: `lz_hash(_, HT_HASH_ORDER)` returns < 1 << 15 == HT_TAB_LEN,
        // so this stays inside `hash_tab`; and a prefetch is a pure hint that
        // cannot fault regardless.
        unsafe {
            prefetch_write(self.hash_tab.as_ptr().add(*next_hash as usize) as *const u8);
        }

        // Read the length-3 singleton and insert this position, in the shape
        // `hc_matchfinder` uses. Done BEFORE the 4-byte search so the insert
        // happens exactly once per position on every control-flow path — the
        // 4-byte search below has several early exits, and a table that skips
        // inserts on some of them would silently degrade over the file.
        //
        // MAINTENANCE IS GATED WITH ACCEPTANCE. On a block where `allow_len3` is
        // false no candidate this table proposes can be accepted, so the read and
        // the write are pure D1 traffic to a 64 KiB table — measured (cachegrind,
        // both arms, 6 MB data.csv, L1 T1): this finder issued 2.68x libdeflate's
        // data writes and 70% of its D1 write misses, and the length-3 singleton
        // was the excess. Entries NOT written during a disabled block leave the
        // table stale for a later enabled block; stale entries are sound (every
        // candidate is re-validated against `cutoff` and by the 3-byte compare)
        // and the offset guard discards most carried-over entries anyway.
        // The COMPUTATION of the 3-byte key is gated too — the banked ablation
        // priced maintenance + key computation together at ~50M Ir on 6 MB, and
        // "instructions are the binding cost in this finder" (see the packed-store
        // record in `skip_bytes`). On a disabled block `*next_hash3` goes stale;
        // at the first position of a re-enabled block the stale key probes one
        // wrong bucket — the same "harmless wrong hash" convention libdeflate
        // itself uses for `next_hash = 0` at stream start (see `ht_fast::run`),
        // made safe by the same 3-byte compare.
        let cur_node3 = if self.allow_len3 {
            let hash3 = *next_hash3 as usize;
            *next_hash3 = lz_hash(next_seq & 0xFF_FFFF, HT_HASH3_ORDER);
            debug_assert!(hash3 < HT_HASH3_SIZE && (*next_hash3 as usize) < HT_HASH3_SIZE);
            // SAFETY: `hash3 < HT_HASH3_SIZE` — see the module doc's soundness section.
            let n = unsafe { *self.hash3_tab.get_unchecked(hash3) } as i32;
            unsafe { *self.hash3_tab.get_unchecked_mut(hash3) = cur_pos as i16 };
            n
        } else {
            // Sentinel that can never pass the `> cutoff` acceptance test below
            // (which is also gated on `allow_len3`, so this value is never read).
            cutoff
        };

        // The 4-byte search. `break 'four` rather than `return`, so that a miss
        // falls through to the length-3 check instead of discarding it.
        'four: {
            // --- entry 0: read, then insert this position ---
            // SAFETY: `hash < HT_TAB_LEN` — see the module doc's soundness section.
            let mut cur_node = unsafe { self.hash_tab.get_unchecked(hash)[0] } as i32;
            unsafe { self.hash_tab.get_unchecked_mut(hash)[0] = cur_pos as i16 };
            if cur_node <= cutoff {
                break 'four;
            }
            let mut match_pos = in_base_now + cur_node as usize;

            // --- entry 1: shift entry 0 down into it, unconditionally ---
            // libdeflate copies entry 0 into entry 1 even when `nice_len` is reached
            // on the first candidate; keeping that makes the parse decisions match.
            let to_insert = cur_node;
            // SAFETY: as above.
            cur_node = unsafe { self.hash_tab.get_unchecked(hash)[1] } as i32;
            unsafe { self.hash_tab.get_unchecked_mut(hash)[1] = to_insert as i16 };

            unsafe {
                if load_u32(base, match_pos) == seq {
                    best_len = lz_extend(buf, in_next, match_pos, HT_MIN_MATCH_LEN, max_len);
                    best_match_pos = match_pos;
                    if cur_node <= cutoff || best_len >= nice_len {
                        break 'four;
                    }
                    match_pos = in_base_now + cur_node as usize;
                    // Pre-screen: the same 4-byte head AND the 4 bytes ending at the
                    // incumbent's last matched byte. A candidate failing either
                    // cannot beat `best_len`, so it is rejected without extending.
                    if load_u32(base, match_pos) == seq
                        && load_u32(base, match_pos + best_len as usize - 3)
                            == load_u32(base, in_next + best_len as usize - 3)
                    {
                        let len = lz_extend(buf, in_next, match_pos, HT_MIN_MATCH_LEN, max_len);
                        if len > best_len {
                            best_len = len;
                            best_match_pos = match_pos;
                        }
                    }
                } else {
                    if cur_node <= cutoff {
                        break 'four;
                    }
                    match_pos = in_base_now + cur_node as usize;
                    if load_u32(base, match_pos) == seq {
                        best_len = lz_extend(buf, in_next, match_pos, HT_MIN_MATCH_LEN, max_len);
                        best_match_pos = match_pos;
                    }
                }
            }
        }

        // LENGTH-3 CHECK — the addition to libdeflate's `ht_matchfinder`, in the
        // shape its own `hc_matchfinder` uses for `hash3_tab`. Only reached when the
        // 4-byte search found nothing, so it costs one compare on the miss path and
        // nothing at all on the hit path.
        //
        // The offset guard is not optional: a length-3 match beyond
        // HT_MAX_LEN3_OFFSET costs more bits than three literals, which is why
        // `deflate_compress_greedy` carries `length > DEFLATE_MIN_MATCH_LEN ||
        // offset <= 4096`. Applying it here rather than in the parser keeps the
        // parser a faithful `deflate_compress_fastest` (which accepts any match the
        // finder returns) and puts the bit-cost knowledge where the length-3
        // candidate is produced.
        //
        // `allow_len3` is the SECOND half of that bit-cost question, and the offset
        // guard alone cannot answer it: distance decides what a length-3 match
        // COSTS, while the literal distribution decides what the three literals it
        // replaces WOULD have cost. Measured on the corpus, the two questions split
        // the files cleanly and in opposite directions — see the field's doc.
        if best_len == 0 && self.allow_len3 && cur_node3 > cutoff {
            let mp = in_base_now + cur_node3 as usize;
            let off = (in_next - mp) as u32;
            if off <= HT_MAX_LEN3_OFFSET {
                // SAFETY: `cur_node3 > cutoff` so `mp < in_next` and `mp` points into
                // already-processed input; `load_u24` reads 4 bytes at `mp`, and
                // `mp + 4 <= in_next + 4 <= buf.len()` given BUF_PAD.
                let cand = unsafe { load_u24(base, mp) };
                if cand == seq & 0xFF_FFFF {
                    return (3, off);
                }
            }
        }

        if best_len == 0 {
            return (0, 0);
        }
        (best_len, (in_next - best_match_pos) as u32)
    }

    /// `ht_matchfinder_skip_bytes`: insert `count` positions without searching.
    ///
    /// Faithful port, including the early return: if fewer than `count +
    /// HT_REQUIRED_NBYTES` bytes remain, libdeflate inserts NOTHING at all
    /// rather than inserting a truncated run. That asymmetry is part of the
    /// algorithm — it is what keeps the `next_hash` pipeline valid — not an
    /// oversight to tidy up.
    // `cur_pos + count - 1 >= WINDOW_SIZE` is libdeflate's literal form. clippy
    // prefers the arithmetically identical `cur_pos + count > WINDOW_SIZE`; the
    // vendor's is kept so this port stays diffable against
    // `ht_matchfinder_skip_bytes` line for line. Faithfulness to the source we are
    // converging on is worth more here than the lint, and the two forms are equal
    // for all i32 values reachable here (`count >= 1`, so no underflow).
    #[allow(clippy::int_plus_one)]
    #[inline(always)]
    pub fn skip_bytes(
        &mut self,
        buf: &[u8],
        in_base: &mut usize,
        in_next: usize,
        in_end: usize,
        count: u32,
        next_hash: &mut u32,
        next_hash3: &mut u32,
    ) {
        if count as usize + HT_REQUIRED_NBYTES as usize > in_end - in_next {
            return;
        }
        let mut cur_pos = (in_next - *in_base) as i32;
        if cur_pos + count as i32 - 1 >= MATCHFINDER_WINDOW_SIZE {
            self.slide_window();
            *in_base += MATCHFINDER_WINDOW_SIZE as usize;
            cur_pos -= MATCHFINDER_WINDOW_SIZE;
        }

        // SAFETY: the early return above guarantees `count + HT_REQUIRED_NBYTES
        // <= in_end - in_next`, so every `pos` reached below satisfies
        // `pos + 4 <= in_next + count + 5 <= in_end <= buf.len()`.
        let base = buf.as_ptr();
        // One register-resident copy per call: the flag cannot change mid-call, and
        // reading the field inside the loop measurably re-loads it per position.
        let len3 = self.allow_len3;
        let mut hash = *next_hash as usize;
        let mut hash3 = *next_hash3 as usize;
        let mut pos = in_next;
        let mut remaining = count;
        loop {
            debug_assert!(hash < HT_TAB_LEN && hash3 < HT_HASH3_SIZE);
            // Plain two-store shift, faithful to `ht_matchfinder_skip_bytes`.
            //
            // A u32-packed SINGLE store was tried and reverted. It cut D writes
            // 26,927,232 -> 19,730,873 (-26.7%) with bit-identical output, and cost
            // 10,601,417 extra instructions — 1.92 extra instructions per removed write,
            // exactly the shape of load32+shift+or+convert+store32 against a 16-bit
            // copy. The frozen paired wall moved ZERO. Stores are not the binding cost
            // in this finder; instructions are.
            //
            // SAFETY: `hash < HT_TAB_LEN` and `i` ranges over `0..HT_BUCKET_SIZE` — see
            // the module doc's soundness section.
            unsafe {
                let b = self.hash_tab.get_unchecked_mut(hash);
                let mut i = HT_BUCKET_SIZE - 1;
                while i > 0 {
                    b[i] = b[i - 1];
                    i -= 1;
                }
                b[0] = cur_pos as i16;
            }
            // The length-3 table is a SINGLETON per key (no bucket to shift), so a
            // skipped run overwrites rather than shifts — same as
            // `hc_matchfinder_skip_positions` does for its `hash3_tab`. Skipping this
            // insert would leave the length-3 table blind to every position inside a
            // match, which is most of the input on compressible data.
            //
            // Gated with acceptance, same as `longest_match` — see the comment
            // there. Dictionary seeding at T>1 runs with `allow_len3` forced on
            // (see `ht_fast::run`) so an enabled first block sees a full table.
            pos += 1;
            let seq = unsafe { load_u32(base, pos) };
            hash = lz_hash(seq, HT_HASH_ORDER) as usize;
            if len3 {
                unsafe { *self.hash3_tab.get_unchecked_mut(hash3) = cur_pos as i16 };
                hash3 = lz_hash(seq & 0xFF_FFFF, HT_HASH3_ORDER) as usize;
            }
            cur_pos += 1;
            remaining -= 1;
            if remaining == 0 {
                break;
            }
        }
        // SAFETY: as above — `hash < HT_TAB_LEN`, and prefetch cannot fault.
        unsafe {
            prefetch_write(self.hash_tab.as_ptr().add(hash) as *const u8);
        }
        *next_hash = hash as u32;
        if len3 {
            *next_hash3 = hash3 as u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table must start at the sentinel, not at zero: position 0 is a VALID
    /// position, so a zero-initialised table would report a match against
    /// offset 0 on the very first probe.
    #[test]
    fn new_and_reset_arm_the_sentinel() {
        let mut mf = HtMatchfinder::new();
        assert!(mf
            .hash_tab
            .iter()
            .all(|b| b.iter().all(|&e| e == MATCHFINDER_INITVAL)));
        mf.hash_tab[123][0] = 7;
        mf.hash_tab[123][1] = 9;
        mf.reset();
        assert!(mf
            .hash_tab
            .iter()
            .all(|b| b.iter().all(|&e| e == MATCHFINDER_INITVAL)));
    }

    /// `acquire` must hand back a re-armed table even when it reused a box that
    /// a previous run had dirtied — otherwise T>1 chunk independence breaks.
    #[test]
    fn acquire_rearms_a_recycled_table() {
        {
            let mut mf = HtMatchfinder::acquire();
            mf.hash_tab[42][0] = 1234;
        } // returned to the pool DIRTY
        let mf = HtMatchfinder::acquire();
        assert_eq!(mf.hash_tab[42][0], MATCHFINDER_INITVAL);
    }

    /// A second candidate that the first probe would have missed must be found.
    /// This is the whole point of the 2-way bucket, so it is asserted directly
    /// rather than inferred from a corpus size.
    #[test]
    fn finds_the_better_second_candidate() {
        // Two occurrences of "ABCD" that collide in the same bucket, where the
        // OLDER one extends further. A single-probe finder keeps only the most
        // recent and would return the shorter match.
        let mut data = Vec::new();
        data.extend_from_slice(b"ABCDXXXXXXXXXXXX"); // 0: ABCD + long tail
        data.extend_from_slice(b"ABCDY"); //            16: ABCD + short tail
        data.extend_from_slice(b"ABCDXXXXXXXXXXXX"); // 21: the search position
        data.resize(data.len() + 64, 0); // pad for the word-at-a-time loads

        let mut mf = HtMatchfinder::new();
        let mut in_base = 0usize;
        // SAFETY: `data` was padded by 64 bytes above, so a 4-byte load at 0 is
        // trivially in bounds.
        let mut next_hash = lz_hash(unsafe { load_u32(data.as_ptr(), 0) }, HT_HASH_ORDER);
        let mut next_hash3 = lz_hash(
            unsafe { load_u32(data.as_ptr(), 0) } & 0xFF_FFFF,
            HT_HASH3_ORDER,
        );

        // Insert positions 0..21 the way the parse loop would.
        for pos in 0..21 {
            let _ = mf.longest_match(
                &data,
                &mut in_base,
                pos,
                32,
                258,
                &mut next_hash,
                &mut next_hash3,
            );
        }
        let (len, off) = mf.longest_match(
            &data,
            &mut in_base,
            21,
            32,
            258,
            &mut next_hash,
            &mut next_hash3,
        );

        // The best available match at 21 is against position 0 (16 bytes), not
        // the more recent position 16 (4 bytes).
        assert!(
            len >= 8,
            "expected the longer candidate, got len={len} off={off}"
        );
        assert_eq!(off, 21, "expected the offset to name position 0");
    }
}
