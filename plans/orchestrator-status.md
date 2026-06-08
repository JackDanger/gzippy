# Orchestrator status — NAMING TRUTH + TWO-PATH + 3-WAY FULCRUM mission

## COPY-FREE-TO-FINAL STAGE 2 — WIRED + LANDED BYTE-EXACT + MEASURED → drain-memcpy tooth BANKED +0.05× (advisor UPHELD-WITH-CAVEATS); native_fold ~0.74×→~0.79× rg [2026-06-07, OWNER turn, HEAD 0f5bc85b]
Wired Stage-1's `decode_clean_into_contig` into the gzippy-native FOLD seam: the post-flip clean
tail now decodes DIRECTLY into chunk.data's reserved contiguous tail, DELETING the ring→chunk.data
bulk drain memcpy. FAITHFUL PREPEND (the 32 KiB window is already the contiguous tail of chunk.data
⇒ data_prefix_len stays 0, CRC-prefix-exclusion + decode_bypass landmine SIDESTEPPED; NOT the
forbidden window-in-scratch shortcut). Two synchronous advisors: pre-impl source-verify UPHELD the
key realization + landmine-sidestep + named 5 hazards; post-impl measurement UPHELD-WITH-CAVEATS.
Advisor verdict: plans/copyfree-stage2-advisor-verdict.md. **SUPERVISOR GATE — drain tooth banked
+0.05×; next = the ≤0.11× intrinsic symbol-rate inner-Huffman work bounded by ocl_cf 0.925×.**

MECHANISM (commit 0f5bc85b, +415 net over 3 files): new `MarkerStep::FlipToContig` (native,
`not(isal_clean_tail)`) → driver resumes the SAME thread-local Block in `finish_decode_chunk_contig_native`
(no ring, no drain); shared generic loop + gzippy-isal two-phase path UNCHANGED. +
`SegmentedU8::contig_decode_window` (base/cap/len re-fetched every iter, grow-safe) +
`Block::decode_clean_stored_into_contig` (post-flip stored block). Hazards: H1 release headroom
guard (contig has no ring modulo ⇒ violation = heap OOB not CRC-catchable — a real regrow-past-16MiB
bug was CAUGHT by it during testing + fixed: Vec::reserve(min_spare)); H2 stored-block; H3
commit-before-decoded_range + multi-call accounting; H4 base re-fetch; H5 native-only.

BYTE-EXACT: gzippy-native sha 028bd002…cb410f == gzip == rapidgzip on /root/silesia.gz (211968000)
T1+T8, x86_64 guest + arm64 host; gzippy-isal UNCHANGED (x86_64 Rosetta). flip_to_clean=12 (contig
route is production). 862 lib + native_fold_parity + flip-seam + 3 Stage-1 diff + 3 NEW owed-case
tests green (only pre-existing diff_ratio timing flake fails, passes isolated).

MEASURED (locked guest, interleaved measure.sh N=11/pass, sha-OK, 10 passes, A/B vs prior banked
copy-free-DRAIN baseline /tmp/gzbuild-native@9cde0b4f vs rg): stage2 strictly faster than priornative
9/10 passes; paired delta mean +0.058× median +0.044× SE±0.020 ⇒ BANKED +0.05× (advisor: drop the
+0.07 edge). Sign-stable + TRIANGULATED by the same-binary GZIPPY_FOLD_NODRAIN +0.067× (clean
isolation). Magnitude load-confounded (loadavg 2.2→5.0, autocorrelated ⇒ ~2-SE). Provenance verified:
only-delta is the Stage 2 wiring. GUEST: /tmp/gz-ft-src build (native, sha 028bd002…cb410f). NO
orphan processes.

## COPY-FREE-TO-FINAL REFACTOR — STAGE 1 LANDED byte-exact (the ~0.067× drain-memcpy tooth's hardest MECHANICAL risk retired; wall NOT yet banked, Stage 2 wiring gated) [2026-06-07, OWNER turn, HEAD c224aaad + this commit]
Two synchronous disproof advisors. The first vetted the SCOPING decision: CHECKPOINT-STAGE-1,
NOT one-pass — a gated full-wire (engine writes contiguous into chunk.data when ON, ring+drain
when OFF) CANNOT bank the wall safely this turn, because OFF==identity isolates COMMIT risk but
NOT MEASUREMENT risk: to bank you must run ON, and ON's correctness rests on the uncontained
data_prefix_len-nonzero activation + CRC-prefix-exclusion + decode_bypass serialization
round-trip; realistic one-pass outcome = a dead ON path + nothing banked (strictly worse than a
clean checkpoint). The second advisor (plans/copyfree-stage1-advisor-verdict.md) UPHELD the
LANDED Stage 1 — A (additive, zero prod caller) UPHELD, B (contig back-ref byte-equiv + proven
266-byte headroom) UPHELD, C (range-check `distance > *pos` ≡ ring's `> decoded_bytes+emitted`)
UPHELD, D (differential adequacy) UPHELD-W-CAVEATS, E (correct checkpoint, nothing to revert)
UPHELD. **SUPERVISOR GATE — Stage 1 mechanical risk retired byte-exact; Stage 2 (wire FOLD seam +
flip default + delete drain + remove-and-measure) is the next gated turn.**

LANDED (commit c224aaad, +453/-0, ONE file marker_inflate.rs, PURELY ADDITIVE, ZERO production
callers ⇒ byte-exact by construction on BOTH features + BOTH archs): (1) emit_backref_contig —
non-wrapping clean back-ref (no % U8_RING_SIZE; the 3 wrap arms collapse to word/RLE/overlap).
(2) Block::decode_clean_into_contig — clean (<false>) body → caller-supplied CONTIGUOUS buffer
with the 32 KiB window as a DICTIONARY PREFIX at base[0..window_len) (vendor setInitialWindow,
DecodedData.hpp:278-289); per-call cap to spare-(MAX_RUN_LENGTH+8), Engine-C grow-between-calls
contract. (3) 3 ring-vs-CONTIG differentials driving the REAL production read() loop, byte-equal:
window-prefix back-ref (distance==*pos==32768 → base[0]), multi-call resumable (per_call=4096),
RLE+short-distance. 3/3 pass (Rosetta x86_64 x86-64-v2); full lib 855 pass = 6 pre-existing
Rosetta/timing failures only (stash A/B == HEAD baseline); arm64 native release clean; round-trip
sha verified.

OWED FOR STAGE 2 (advisor D — do NOT assume retired): (1) stored/uncompressed clean block (contig
fn Errs; ring read() decodes it) ⇒ route/extend. (2) multi-deflate-block clean phase across a
block boundary (read_header advance with persisted *pos/decoded_bytes) UNTESTED. (3) actual
out_room-saturating REGROW UNTESTED (math present, no test drives it). PLUS the deferred landmine:
data_prefix_len=32768 activation (audit decoded_size / window-publish / consumer iovecs /
apply_window readers — FOLD has only ever run prefix==0) + CRC-prefix exclusion + decode_bypass
serialize/deserialize round-trip. FORBIDDEN shortcut: dual-region back-ref (window in scratch,
chunk.data prefix==0) — diverges from vendor prepend ⇒ violates faithful-port. Decide the FAITHFUL
PREPEND model now. RESIDUAL: banked teeth UNCHANGED at native_fold 0.737× rg; the ~0.067× tooth is
de-risked, banked by Stage 2; then ≤0.11× UPPER-BOUND intrinsic symbol rate bounded by ocl_cf
0.925×. NO orphan processes (research + 2 advisor subagents completed; no sleep sentinels).

## RING→DATA DRAIN ISOLATION RAN (fold-contig advisor's owed same-engine pure-Rust ring-copy oracle) → the 0.188× residual SPLITS: ~0.067× ring→data drain MEMCPY (recoverable free tooth) + ≤0.11× UPPER BOUND intrinsic symbol rate + ring-write; CRC is NOT a lever. NO production fix landed (banking needs the byte-exact copy-free-to-final refactor, prompt-gated) [2026-06-07, OWNER turn, HEAD 7ae5903]
Captured the gzippy-native T8 whole-system picture with the merged fulcrum_total (built native @ HEAD
fc7336c3, target-cpu=native, on /tmp/gz-ft-src), then BUILT + RAN the fold-contig advisor's owed
same-engine pure-Rust ring-copy-free isolation oracle (GZIPPY_FOLD_NODRAIN/NOCRC — measurement-only,
OFF==identity, committed 7ae5903). Synchronous disproof advisor UPHELD (C1/C3/C5 UPHELD, C2 upheld-w-
caveats; the no-op-drain is CONSERVATIVE). Charter CURRENT STATE updated. Brief:
plans/ring-drain-isolation-brief.md; verdict: plans/ring-drain-isolation-advisor-verdict.md.
**SUPERVISOR GATE — drain memcpy isolated as ~0.067× recoverable; intrinsic symbol-rate sharpened to
≤0.11× UPPER BOUND; NEXT = (1) byte-exact copy-free-to-final engine refactor to BANK ~0.067×, then
(2) the ≤0.11× inner-Huffman rate work bounded by ocl_cf 0.925×. Do NOT start without supervisor.**

fulcrum_total (native T8, trace byte-exact): routing flip_to_clean=0 finished_no_flip=16
window_seeded=2 (natural propagation, NOT the seed oracle; isal_oracle_chunks=0). The binary
routing-guard REFUSES on window_seeded>0 (advisor-flagged C4 caveat-2 brittleness — 2/16 naturally
seeded ≠ GZIPPY_SEED_WINDOWS), so read DESCRIPTIVELY with that caveat: worker.block_body 658ms SELF
(dominant = marker-engine inner decode @262 MB/s/168MB), apply_window 90ms, writev 60.8ms; consumer
61% WAIT ⇒ engine-rate-bound. The ring-write + ring→data drain are folded INSIDE block_body/drain (no
separate spans) ⇒ fulcrum_total can't split them — hence the isolation oracle.

ISOLATION (locked guest 10.30.0.199, taskset 0,2,4,6,8,10,12,14, gov=perf, interleaved measure.sh
N=11, RAW=211968000, best-of-N over 4 NON-OVERLAPPING passes, load-invariant 1.2→2.3):
native_fold ∈[0.1807,0.1831] vs nodrain ∈[0.1645,0.1684] = nodrain/native 1.083-1.109×; nodrain_nocrc
≈ nodrain; rg 0.1358-0.1464. ⇒ removing the ring→data drain memcpy moves native_fold ~0.745×→~0.812×
rg = ~+0.067× recoverable; CRC ≈ free. SELF-TEST: OFF==rg==028bd002…cb410f (x86_64 guest + arm64);
NODRAIN sha differs (fires), exits 1 only at the terminal trailer CRC32 (full decode ran first). 857
lib tests green.

ADVISOR: C1 UPHELD-W-CAVEATS (best-of-N on one-sided noise = correct estimator; interleaved +
load-invariant = freq-neutral; small turbo component not fully excluded w/o pinned-freq pass). C2
UPHELD-W-CAVEATS (≤0.11× still confounds ring-write + ISA-L-vs-pure-Rust engine diff = UPPER BOUND on
symbol rate; safe unconditional = intrinsic ≤0.188×). C3 UPHELD (CRC reads ring slice, drain-
independent). C5 UPHELD (copy-free-to-final is FAITHFUL: vendor decodes clean BULK to contiguous u8
with no u16 ring, DecodedData.hpp:278-289 — gzippy's clean-phase drain has no vendor counterpart;
nuance: vendor concatenates at merge, so claim "no clean-phase ring/narrow" not "zero copies"). KEY:
no-op-drain is CONSERVATIVE (cold chunk.data reads in writev + window-publish cost nodrain EXTRA) ⇒
true drain ≥ +0.067×.

NEXT (gate, do NOT start): (1) BANK ~0.067× — pure-Rust clean (`<false>`) phase writes u8 DIRECTLY
into chunk.data's reserved tail (reuse writable_tail_reserve+commit), back-refs from that tail, NO
output_ring for the clean phase; non-trivial engine rewrite (clean-phase addressing + back-ref resolve
+ flip-seam + >16MiB reserve-clamp fallback), correctness-sensitive at the flip seam; bound by the
nodrain knob, byte-exact dual-sha, remove-and-measure. (2) THEN ≤0.11× inner-Huffman rate (BMI2 PEXT/
BZHI, wider multi-literal, ISA-L-class packed-u32 LUT) bounded by ocl_cf 0.925×, never the VAR_VI slope.
GUEST: /tmp/gz-ft-src (source @fc7336c3 + knobs; symlinks vendor→/root/gzippy/vendor, /tmp/fulcrum→
/root/fulcrum), build /tmp/gz-ft-src/target/release/gzippy (native, sha 028bd002…cb410f OFF). Driver
/tmp/isolation_wall.sh. Trace /tmp/ft-art/. NO orphan processes.

## SUPERSEDED — CADENCE/INTRINSIC SPLIT RUN (advisor's owed symmetric control) → CADENCE TAX IS REAL + RECOVERED + LANDED: copy-free FOLD clean drain, native_fold 0.678× → 0.737× rg = +0.059× banked ratchet tooth; residual ≤0.188× UPPER BOUND on intrinsic rate [2026-06-07, OWNER turn, HEAD 9cde0b4f]
Ran the symmetric control (give the gzippy-native FOLD `<false>` clean tail one contiguous output
region, no per-block refetch/grow). The FIRST control (pending_clean-only) TIE'd — disproof advisor
caught I'd left the DOMINANT per-block ring→u8buf copy (fresh Vec::with_capacity + byte loop in
marker_inflate::drain_to_output) untouched. Ran the advisor's owed copy-free-drain control: it MOVED
the wall, proving a free cadence component was mis-booked as intrinsic. LANDED the fix (commit 9cde0b4f,
byte-exact). Synchronous disproof advisor ×2 (PASS 1 REFUTED under-scoped C1; PASS 2 CONFIRMED the
landed fix, 3 corrections applied). Charter CURRENT STATE updated. Advisors:
plans/fold-contig-split-advisor-verdict.md + plans/fold-contig-landed-advisor-verdict.md.
**SUPERVISOR GATE — cadence tax RECOVERED + banked (+0.059×); next = inner-Huffman RATE work on the
`<false>` clean path, bounded by ocl_cf 0.925× (engine-removed), never the VAR_VI slope.**

THE FIX (production default, byte-exact, advisor-CONFIRMED): marker_inflate::drain_to_output clean
branch pushes ≤2 CONTIGUOUS u8 ring slices straight to the sink (no per-block u8buf alloc + byte loop);
new ContigFoldSink (replaced+DELETED UnifiedMarkerSink) writes them DIRECTLY into a pre-reserved
contiguous chunk.data (no pending_clean middle-man, no second append_clean copy, no regrow);
ChunkData::reserve_clean (clamped 16 MiB).

SELF-TEST (Rule 4): rg == native_FINAL == isal_FINAL sha 028bd002…cb410f (byte-exact, OFF==identity);
flip_to_clean=0 finished_no_flip=16 (FOLD preserved, window-absent bootstrap preserved, did NOT seed);
857 lib + native_fold_parity + flip-seam differentials green (Rosetta x86_64 + arm64).

THE SPLIT (locked guest, interleaved measure.sh, sha-OK, 6 passes): old(triple-buffer) 0.634× <
new_off(copy-free drain) 0.674× < new_contig(fully copy-free) 0.717× rg means, MONOTONIC every pass.
copy#1 (ring→u8buf) +0.040×, copy#2/3/grow +0.043×. BANKED (quiet box, default binary, 3-pass):
native_fold 0.678× → 0.737× rg = +0.059× (the +0.083× split-sum is load-inflated, sign/monotonicity
evidence only). isal_prod also improved to ~0.80× (shares the copy-free drain in its marker bootstrap).

ADVISOR: L1 mechanism CONFIRMED (copy-free drain + contig sink, byte-exact even on the ring wrap split);
L1 magnitude OVERSTATED → banked is +0.059× not +0.083×; L2 (residual = intrinsic symbol rate) NOT
LICENSED → ≤0.188× is an UPPER BOUND (ocl_cf is ring-free + copy-free-to-final + a different engine, so
it confounds symbol rate + ring-write + ring→data drain memcpy); L3 (ContigFoldSink default + delete
UnifiedMarkerSink) CONFIRMED (no live caller, two-phase CleanTailSink untouched, sink overrides correct,
blast radius bounded). 3 corrections (re-label +0.059×, residual upper bound, reserve heuristic + clamp)
ALL APPLIED.

NEXT (gate, do NOT start): inner-Huffman RATE on the `<false>` clean path (BMI2 PEXT/BZHI, wider
multi-literal, ISA-L-class packed-u32 LUT — CLAUDE.md-authorized), bounded by ocl_cf 0.925×, never the
slope. A clean intrinsic-rate isolation needs a same-engine pure-Rust ring-based copy-free-to-final
oracle (does not exist yet) to strip the residual ring-drain memcpy from the 0.188×. GUEST:
/tmp/gzbuild-native (FINAL) + /tmp/gzbuild-head (FINAL) + /tmp/gzippy-old-drain (pre-fix baseline), all
relevant sha 028bd002…cb410f; drivers /tmp/{baseline,contig,contig2,split}_wall.sh + contig_selftest.sh
(bash). NO orphan processes.

## SUPERSEDED — COPY-FREE CLEAN-TAIL ORACLE RAN → prior INCONCLUSIVE REVERSED: clean-decode subsystem IS a wall lever (copy-free ISA-L 0.87–0.925× TIE vs pure-Rust 0.73–0.755×). UNIFIED single-primitive engine ALREADY EXISTS (gzippy-native FOLD, byte-exact) but is the SLOWEST pure-Rust at the wall (0.68×); next lever bounded < 0.14× = cadence-tax + intrinsic clean rate [2026-06-07, OWNER turn, HEAD 7aae6c4a + copy-free overlay, measured /tmp/gzbuild-head + /tmp/gzbuild-native]
Made the clean-tail oracle COPY-FREE (the advisor-OWED fix): new `decompress_deflate_from_bit_into`
(ISA-L decodes DIRECTLY into the chunk buffer — no 64 MiB Vec, no copy) + `writable_tail_reserve` +
`decoded_range`. The copy-free ISA-L clean tail (unseeded, window-absent preserved) hits 0.87–0.925× rg
= TIE vs production pure-Rust 0.73–0.755× — Δ ~0.14× ≫ spread, sign-stable ×3, freq-neutral. Then
source-verified + measured that the UNIFIED single-primitive engine ALREADY EXISTS as gzippy-native (the
FOLD: marker_inflate Engine M flips in-place via `flip_repack_to_u8` = vendor setInitialWindow and
continues the clean tail on the SAME cursor — the governing one-engine memory), byte-exact sha
028bd002…cb410f, flip_to_clean=0 finished_no_flip=16. The unified PURE-RUST engine is the SLOWEST at the
wall (native_fold 0.68× rg) — marker_inflate `<false>` clean rate trails resumable AND ISA-L. Synchronous
disproof advisor (plans/copy-free-oracle-advisor-verdict.md) UPHELD-WITH-CAVEATS. Charter CURRENT STATE
updated. Brief: plans/copy-free-oracle-brief.md.
**SUPERVISOR GATE — clean-engine question SETTLED (subsystem is a lever, ceiling readable = 0.925×); the
unified-engine PORT is DONE structurally (gzippy-native is the production unified path); NO production fix
landed (oracle measurement-only, byte-transparent); next = the clean-RATE fix on the unified engine,
scoped + bounded.**

SELF-TEST (Rule 4): PROD == copy-free ORACLE == rg sha 028bd002…cb410f (byte-exact, OFF==identity);
isal_oracle_chunks=14 isal_oracle_fallbacks=0; routing = prod unseeded (window-absent preserved). 877 lib
+ seam + 10 segmented_buffer tests green (Rosetta x86_64).

WALL (locked guest, interleaved N=11, sha-OK, gov=perf): rg 1.000 | ocl_cf (copy-free ISA-L unified)
0.895/0.892/0.870/0.925/0.905× = TIE | isal_prod (two-engine, pure-Rust clean) 0.735/0.755/0.746/0.733× |
native_fold (unified pure-Rust FOLD) 0.685/0.676× = SLOWEST.

ADVISOR: C1 UPHELD-WITH-CAVEATS (subsystem is a real lever, Δ≫spread, copy gone, reverses INCONCLUSIVE;
but 0.14× bundles ISA-L symbol rate + per-128-KiB resumable yield/refetch tax + one-shot reserve vs grow —
"RATE is the lever" over-attributes). C2 UPHELD (residual = window-absent structure). C3 DIRECTION UPHELD
(unify right), COROLLARY REFUTED (pure-Rust gap to close < 0.14×; cadence+grow are recoverable WITHOUT
raising symbol rate; VAR_VI 0.6× does NOT bound a tax-shedding primitive). MOST ACTIONABLE: run the
symmetric control — give the PURE-RUST `<false>` clean tail one large contiguous window (no per-128-KiB
refetch/yield) and re-measure native_fold; recovered portion = cadence/grow (free), remainder = intrinsic
rate that sets the real no-FFI 1.0× bar.

NEXT (gate, do NOT start): the CLEAN-RATE fix on gzippy-native FOLD. (1) symmetric control = large
contiguous window for the `<false>` tail; (2) close the intrinsic-rate remainder (BMI2/multi-literal/
packed-u32 LUT) up to the < 0.14× bound; ceiling = copy-free ocl_cf 0.925× (engine-removed), never the
slope. GUEST: /tmp/gzbuild-head (isal+copy-free oracle) + /tmp/gzbuild-native (FOLD), both sha
028bd002…cb410f; drivers /tmp/oracle_{selftest,wall}.sh (bash); 3 modified files overlaid on guest tree.
NO orphan processes.

## SUPERSEDED — DECISIVE WALL ORACLE RAN (window-absent-PRESERVING ISA-L clean-tail removal) → clean-engine binder INCONCLUSIVE (instrument contaminated by its own copy); RECONCILIATION SOLID (gap = window-absent STRUCTURE, not the clean engine); NO FORK established [2026-06-07, OWNER turn, HEAD 7aae6c4a, measured /tmp/gzbuild-head]
Ran the decisive WALL-level removal oracle the prior advisor OWED: `GZIPPY_ISAL_ENGINE_ORACLE=1`
UNSEEDED (window-absent routing preserved per the charter OSCILLATION rule) replaces ONLY the
post-flip clean tail with real ISA-L. Plus the seedfull↔production reconciliation (engine held
constant). Synchronous disproof advisor REFUTED my Conclusion-1 framing. Charter CURRENT STATE
updated. Advisor verdict: plans/clean-tail-wall-oracle-advisor-verdict.md. Brief:
plans/clean-tail-wall-oracle-brief.md. Pre-reg: plans/clean-tail-wall-oracle-prereg.md.
**SUPERVISOR GATE — clean-engine binder UNDECIDED (copy-free oracle owed); NO fix landed; NO
fork escalated; the user-constraint fork is NOT forced (advisor: no fork — unify the two engines).**

SELF-TEST (Rule 4): OFF==ON==rg sha 028bd002…cb410f (byte-exact); isal_oracle_chunks=14
isal_oracle_fallbacks=0; routing flip_to_clean=12 finished_no_flip=4 window_seeded=2 = prod
unseeded ⇒ 89% window-absent bootstrap preserved.

RESULT 1 — WALL (3 interleaved N=11, sha-OK, locked guest, gov=perf): prod 0.744/0.754/0.755× rg;
ocl (ISA-L clean tail, unseeded) 0.698/0.686/0.702× rg. The ISA-L clean-tail oracle did NOT beat
production. ADVISOR (load-bearing, CORRECT): INCONCLUSIVE — the oracle's per-chunk 64 MiB
alloc+to_vec copy (gzip_chunk.rs:203,247-256) costs ~0.17× (= the prod→ocl gap AND the
gap-to-threshold); my own model implies S≈C≈0.17×, i.e. the clean engine may SAVE ~0.17× of wall
exactly masked by the copy. A handicapped contender failing to win is uninformative about a
speed-UP ceiling (Rule 3). CANNOT declare engine slack NOR engine binder.

RESULT 2 — RECONCILIATION (UPHELD, engine constant = ISA-L both, 2 runs): ocl_unseed (bootstrap
preserved) 0.697/0.701× vs ocl_seed (seeded, no bootstrap) 0.860/0.857× rg. The seedfull↔production
gap is the WINDOW-ABSENT STRUCTURE (~0.16× of wall), NOT the clean engine. CAVEAT: do NOT
sub-credit to marker RATE specifically — seeding also removes 13→0 spec-fail re-decodes
(project_confirmed_offset_prefetch_gap) + flip machinery; the 0.120s≈0.322s match is the SUM-vs-wall
trap. Structure-level cause solid; rate-vs-spec-fail split NOT isolated.

Q4 NO FORK (advisor, vendor deflate.hpp:1277/1285/1452-1453/1585-1666/1600): rg uses ONE
width-templated `readInternalCompressedMultiCached` for BOTH u16 markers and u8 clean; gzippy's
two-engine split (marker_inflate u16 + resumable u8) is the divergence. The marker rate is likely
bounded by the SAME primitive ceiling as the clean (VAR_VI 0.6×). The faithful move is to COLLAPSE
marker+clean into ONE primitive ([[project_faithful_unified_decoder_over_perf]]) — not a fork.

NEXT (gate, do NOT start): (1) make the oracle copy-free (decode ISA-L into writable_tail) so the
clean-engine ceiling becomes readable; THEN (2) the UNIFIED single-primitive engine. GUEST:
/tmp/gzbuild-head; drivers /tmp/oracle_{selftest,wall,reconcile,verbose}.sh (bash); seeds
/tmp/seeds.bin. NO orphan processes.

## SUPERSEDED — 1.6× WINDOW-ABSENT decodeBlock GAP ATTRIBUTED → it's the CLEAN u8 TAIL decoder (pure-Rust ~2.3× slower than ISA-L), NOT the u16 marker loop [2026-06-07, OWNER turn, HEAD 7aae6c4a + overlay]
Source-diffed the window-absent decode path vs vendor, ran apples-to-apples --verbose
(gzippy + rg, T8 unseeded, locked guest) + a causal SLOW_MODE A/B (freq-neutral),
synchronous disproof advisor. STOPPED at attribution + scoped fix (NO fix landed, per
prompt). Charter CURRENT STATE updated. Advisor: plans/window-absent-attribution-advisor-verdict.md.
Attribution: plans/window-absent-attribution.md. Brief: plans/window-absent-attribution-brief.md.
**SUPERVISOR GATE — attribution advisor-vetted; NO fix landed; next = run the Phase-0
ISA-L removal oracle on the FlipToClean clean tail to bound the wall payoff (Rule 3),
then the scoped clean-engine fix.**

NUMBERS (T8 unseeded, locked guest 10.30.0.199, taskset 0,2,4,6,8,10,12,14, gov=perf):
gzippy decodeBlock SUM 0.805-0.831s vs rg 0.4995s = 1.61-1.66×. Per-engine: gzippy
marker bootstrap body 0.323s (73.0M @ 226-235 MB/s) vs rg "custom inflate" 0.4748s =
gzippy 0.68× FASTER; gzippy clean tail ≈0.48s (≈139M @ ≈290 MB/s, subtraction) vs rg
"ISA-L" 0.2065s = ≈2.3× SLOWER. Routing: flip_to_clean=12 finished_no_flip=4
window_seeded=2.

CAUSAL (ΔdecodeBlock SUM, freq-neutral sleep control, baseline 0.831s): MARKER+100%
→0.965 (+134ms); CLEAN+100% →1.025 (+194ms, marker body UNCHANGED). Sleep: marker +142,
clean +248. CLEAN inject lands in resumable.rs:1199 (the post-flip clean engine), marker
body untouched, yet adds the LARGER ΔdecodeBlock ⇒ the clean tail is the dominant
decodeBlock term, NOT the marker loop.

ROOT CAUSE (source): flip threshold byte-identical to vendor (marker_inflate.rs:1116-1119
↔ deflate.hpp:1282-1284). Two-phase routing (gzip_chunk.rs:1397-1410, isal_clean_tail):
Engine M u16 bootstrap → FlipToClean → Engine C = StreamingInflateWrapper =
unified::Inflate<Clean> = pure-Rust resumable.rs. The measured gzippy-ISAL build's clean
tail is PURE-RUST, NOT ISA-L FFI (resumable.rs:1182-1192; build.rs:98 "REAL ISA-L FFI"
comment was STALE — FIXED, comment-only/byte-transparent). rg WITH_ISAL uses real ISA-L
(deflate.hpp:1452-1453). So the 1.6× = gzippy's pure-Rust clean engine (~0.6× ISA-L
primitive plateau) decoding the clean BULK ~2.3× slower than ISA-L.

ATTRIBUTION among prompt's 3 candidates: (a) marker inner loop REFUTED-as-prime (gzippy
marker FASTER than rg); (b) u16-over-clean-bulk REFUTED (flip byte-identical; post-flip
bulk is u8); (c) table-build REFUTED (shared, seedfull ties). CAUSE = pure-Rust clean
tail slower than ISA-L.

ADVISOR: core UPHELD (causally airtight); CONFIRMED the build.rs stale-comment risk;
REFUTED "actionable now" (Δwall payoff needs the Phase-0 ISA-L removal oracle, Rule 3,
SUM slack-masked at Fill 85%); 0.68×/2.3× are subtraction hedges (proven claim = "clean
tail is the bigger decodeBlock term"); candidate-1 ISA-L FFI = goal #2 (re-adds FFI),
candidate-2 faster pure-Rust clean engine = goal #1 advance.

SCOPED FIX (gate, do NOT start): (1) goal#2 route FlipToClean clean tail through ISA-L
FFI (= rg WITH_ISAL); (2) goal#1 faster pure-Rust clean u8 engine (BMI2/multi-literal/
ISA-L-LUT). Bound with GZIPPY_ISAL_ENGINE_ORACLE on the clean tail (FALLBACKS==0) FIRST.
GUEST: /tmp/gzbuild-head/release/gzippy (measured, clean HEAD isal native); rg 0.16.0;
captures /tmp/{rg_verbose,gz_verbose,m100,c100,cs,ms}.txt. NO orphan processes.

## CORRECTED overlap oracle RUN → OVERLAP DEAD AS A LEVER; all 3 scheduling levers causally exhausted; binder RELOCATED to the marker-engine decode rate [2026-06-07, OWNER turn, HEAD 7aae6c4a + corrected-oracle overlay]
Corrected the backwards overlap oracle (warm-all-then-drain → real in-flight
overlap dispatch via `submit_prefetch`, non-blocking), measured the registered
decider on the locked guest, removed the advisor-flagged 4096 retention confound,
and tested the two remaining scheduling sub-levers. Two synchronous disproof-advisor
passes (PASS 2 reversed PASS 1's own "F1 likely holds via overlap"). Charter CURRENT
STATE updated. Advisor: plans/corrected-overlap-advisor-verdict.md (+FOLLOW-UP
VERDICT section). Brief: plans/corrected-overlap-advisor-brief.md.
**SUPERVISOR GATE — overlap REFUTED, engine relocation advisor-vetted (direction),
NO engine fix landed; next loop = faithful engine direction (u8-direct clean path).**

THE CORRECTION (perfect_overlap.rs + chunk_fetcher.rs perfect_overlap_warm): dispatch
EVERY chunk's decode as an IN-FLIGHT prefetch up-front (`submit_prefetch(part_key,rx)`
= vendor m_prefetching.emplace, BlockFetcher.hpp:558), NON-BLOCKING, return immediately
so the in-order consumer_loop runs CONCURRENTLY with the still-running decodes = real
decode↔drain overlap. Removed the 4096 cache-cap bump (retention confound).

SELF-TEST (Rule 4): OFF==identity AND ON byte-exact sha 028bd002…cb410f on arm64-native
(local) + x86_64 gzippy-isal native (guest); path=ParallelSM; dispatch phase 0.0007s
(non-blocking); warm_hit_frac 0.882 (15/17, 2 offset-0 misses).

DECIDER NUMBER (T8, measure.sh interleaved N=11, sha-OK, locked guest, gov=perf, 5 runs):
perfovl CORRECTED+retention-fixed = 0.187-0.192s = 0.684-0.695× rg vs HEAD 0.174-0.177s
= 0.730-0.754× (rg 0.130s). **The corrected overlap oracle is sign-stably ~5-7% SLOWER
than production — does NOT reach the tie; retention fix did NOT rescue it.**

RESOLVE-AHEAD SATURATED (verbose): Worker resolve-ahead ok=13/13 (head), 14/14 (perfovl)
= drain-hiding LIVE on the production path ~82% coverage; std::future::get is a wait on
the DECODE future, not resolve (resolve runs earlier on the pool). Drain already hidden.

FINER-CHUNKING REFUTED (GZIPPY_CHUNK_KIB sweep, byte-exact, interleaved vs rg, 2 runs):
k4096(17 chunks)≈k2048(34)≈k1024(68) all 0.68-0.72× rg, FLAT-to-WORSE. decodeBlock SUM
stays ~1.1s, Fill DROPS 87%→77% (per-chunk bootstrap overhead cancels the tail-wave gain).
The advisor's ~0.04s tail-wave-quantization hypothesis did NOT materialize.

ADVISOR ×2: PASS 1 REFUTED my decider C1 (retention confound + the real lever is
resolve-ahead), said F1 likely holds via overlap. PASS 2 (after resolve-ahead-saturated
+ retention-fixed + finer-chunking evidence) REVERSED: F1-via-overlap/drain REFUTED;
binder relocates to the per-thread decode floor (engine) — DIRECTION upheld, 1.6×
magnitude UNVERIFIED; target the u8-DIRECT clean path (governing memory), not just a
faster u16 ring.

CONCLUSION (causal, advisor-vetted): the T8 wall is NOT scheduling-bound. All three
scheduling levers exhausted (dispatch-depth null/worse, drain-hiding saturated, tail-wave
flat-to-worse). Residual ~0.70× gap = per-thread MARKER-ENGINE decode rate. NEXT (gate):
source-diff window-absent decode_chunk_unified_marker vs vendor to attribute the gap
(marker inner loop / u16-vs-u8 width / table-build), faithful target = u8-direct clean
path + readInternalCompressedMultiCached marker port; bound any engine fix with the
Phase-0 ISA-L engine-removal oracle (already ties seeded), never the slope.
GUEST: src overlaid to HEAD+corrected-oracle, build /tmp/gzbuild-po (sha 028bd002…cb410f),
drivers /tmp/po_measure.sh + /tmp/chunk_sweep.sh. NO orphan processes.

## SUPERSEDED — GZIPPY_PERFECT_OVERLAP (the registered decider, NEVER-BEFORE-RUN) BUILT + SELF-TESTED + RUN; advisor REFUTED my read (oracle built BACKWARDS) [2026-06-07, OWNER turn, HEAD 7aae6c4a + oracle overlay]
Closed the live Rule-3 violation flagged by the supervisor coach (PROCESS FIX #1):
the registered decider `GZIPPY_PERFECT_OVERLAP` had NEVER been run. Built it
byte-exact (src/decompress/parallel/perfect_overlap.rs + perfect_overlap_warm in
chunk_fetcher.rs), self-tested (Rule 4), measured on the locked guest, ran the
synchronous disproof advisor. Advisor caught a LOAD-BEARING error. Charter CURRENT
STATE + prereg RESOLUTION #2 updated. Advisor: plans/perfect-overlap-advisor-verdict.md.
Brief: plans/perfect-overlap-advisor-brief.md. **SUPERVISOR GATE — decider RUN, but
F1 UNDECIDED (my oracle tested the wrong schedule); do NOT declare STOP/TIE.**

DECIDER NUMBER (lead with the causal oracle, T8, measure.sh interleaved N=11, sha-OK,
2 runs): perfovl 0.225-0.227s = 0.581-0.583× rg; HEAD 0.177s = 0.740×; rg 0.131s.
**The oracle is SLOWER than production.**

WHY (advisor, load-bearing): my oracle runs warm (decode-ALL) FULLY then drain
(resolve-chain+write) — it SERIALIZES the two phases production already OVERLAPS =
an ANTI-overlap. Its wall is a pessimistic SUM (warm 0.117 + drain 0.066). An
upper bound built by DESTROYING overlap cannot falsify F1 (the TIE claim) — the
symmetric of last turn's upper-bound-can't-fire-F2. My C1/C3/C4 REFUTED.

GENUINE FINDING (advisor-corrected): warm alone = 0.117s is a TRUE LOWER bound on
any schedule's wall (all chunks must decode; drain 0.066 < warm hides under it).
**0.117 < rg WALL 0.131 < tie threshold 0.138** = INSIDE the tie zone. I mis-read it
"above the tie" only by comparing 0.117 to rg's decode FLOOR (0.085) not rg's WALL
(0.131). Matched floor-to-floor = warm 0.117 vs rg Real Decode 0.104 = 1.13×. **Read
correctly: the T8 TIE IS REACHABLE by better decode↔drain OVERLAP — the scheduling
direction is NOT refuted; this oracle FAILED TO TEST it.**

SELF-TEST (Rule 4): sha 028bd002…cb410f byte-identical w/ and w/o the oracle on
arm64 + x86_64 guest; warm_hit_frac 0.88-0.96 (2 misses = offset-0 stream-start).
Byte-transparent; the warm cache really removed the head-of-line wait.

STILL OPEN: can a REAL OVERLAPPED schedule collapse production 0.177 toward the
0.117-0.13 floor? Needs a CORRECTED oracle (warm OVERLAPPING drain, pipelined
per-chunk as predecessor windows land), NOT warm-all-then-drain. F1 UNDECIDED.
SCOPED NEXT (gate): build the corrected overlap oracle to decide F1, OR (lower bound
already says reachable) go to the faithful decode↔drain overlap fix
(project_confirmed_offset_prefetch_gap dispatch-TIMING) bounded by that corrected
oracle first. GUEST: src overlaid to HEAD+oracle, build /tmp/gzbuild-po, driver
/tmp/po_measure.sh. NO orphan processes.

## SUPERSEDED — COUNTER RENAMED (anti-inversion) + SCHEDULING/SERIAL CEILING BOUNDED via real oracles; advisor REFUTED my arithmetic F2 over-reach [2026-06-07, OWNER turn, HEAD f1aceee1]
Renamed the inversion-prone counter byte-exact (commit f1aceee1:
`BOOTSTRAP_POST_FLIP_U16_BYTES` → `BOOTSTRAP_CLEAN_FLIPPED_BYTES`; it counts the
marker-FREE CLEAN-flipped complement, NOT marker-ring bytes — the exact inversion
that bit C3 repeatedly; verbose label now self-documents). Then bounded the
scheduling/serial ceiling with REAL removal oracles on the locked guest (src rsynced
to HEAD, /tmp/gzbuild-head, sha 028bd002…cb410f every cell). Charter CURRENT STATE +
prereg updated. Advisor: plans/scheduling-ceiling-advisor-verdict.md. Pre-reg:
plans/scheduling-ceiling-prereg.md. Brief: plans/scheduling-ceiling-advisor-brief.md.
**SUPERVISOR GATE — ceiling bounded, NO engine fix landed (binder coupled/unconfirmed).**

NUMBERS (T8, interleaved measure.sh, sha-exact): HEAD 0.174-0.177s = 0.736-0.755× rg
(rg 0.130s). seedfull (GZIPPY_SEED_WINDOWS = the faithful perfect-window-overlap oracle)
0.128s = 1.029× = TIE; T16 0.128s = 1.121× WIN. NO_PREFETCH negative control 0.523s =
0.253× (3× SLOWER ⇒ scheduling firmly critical). Verbose: gzippy decodeBlock SUM 0.803
vs rg 0.502 (1.6×); Real Decode 0.116 vs 0.084; future::get 0.089 (T16 0.046, HALVES).

KEY (oracle-grounded, advisor-corrected): the T8 TIE IS reachable (seedfull, F1) BUT
the scheduling overlap AND the window-absent marker-engine rate are LIVE + COUPLED
(window-present ⇒ CLEAN engine, gzip_chunk.rs:790 vs :826) — neither isolable. My
arithmetic "engine binds (F2)" was a forbidden Rule-3 extrapolation (the 0.116+0.043
sum was a STRICT UPPER BOUND, double-counting the overlapping tail) — advisor REFUTED
C2 (scheduling IS critical: future::get halving = criticality, NO_PREFETCH 3×) and C3
(F2). C1 (engine reaches wall) UPHELD-WITH-CAVEATS. C4 (backward marker scan) UNCONFIRMED/
implausible (flip-to-clean at 32KiB confines it). rg's mechanism source-verified
(GzipChunkFetcher.hpp:479/:513/:559 — main-thread uncompressed last-window publish on
the named serial critical path; gzippy ALREADY ports it) ⇒ residual is dispatch TIMING,
not a missing mechanism nor horizon DEPTH (vendor-identical). SCOPED NEXT (gate, must
CAUSALLY PERTURB FIRST): (a) faster window-absent marker engine OR (b) earlier window
publish so more chunks hit clean at high T (project_confirmed_offset_prefetch_gap). NO
orphan processes (advisor wrapper+sleep killed; guest clean).

## MARKER FAST LOOP LANDED → rg's multi-cached u16 marker loop ported; T8 wall = TIE (no move), byte-exact, KEPT per 7a. Advisor C1 UPHELD / C3 REFUTED my mechanism [2026-06-07, OWNER turn, HEAD 04fda86d]
Ported step (i) of the bounded plan: rg's multi-cached u16 marker FAST LOOP
(vendor `readInternalCompressedMultiCached` deflate.hpp:1585-1666). Added a
speculative software-pipelined fast loop to the u16 MARKER path
(`read_internal_compressed_specialized::<true>`, marker_inflate.rs `'mfast` loop),
mirroring the clean path. rg runs the SAME tight multi-cached loop for u16 markers
as u8 clean; gzippy's marker path was stuck on the careful per-symbol loop. Three
faithful u16 deltas (widened-u16 speculative store, distance_marker += lit_prefix +
emit_backref_ring::<true>, no marker-window range check). Charter CURRENT STATE
updated. Advisor: plans/marker-loop-port-advisor-verdict.md. Brief:
plans/marker-loop-port-brief.md.

BYTE-EXACT: gzippy-native arm64 + gzippy-isal guest x86_64, sha 028bd002…cb410f
T1/T8/T16 path=ParallelSM. 856 lib tests (1 fail = pre-existing flaky diff_ratio).
Seam + native_fold_parity green.

REMOVE-AND-MEASURE (locked guest 10.30.0.199 double-ssh, 16c gov=perf turbo-on,
taskset 0,2,4,6,8,10,12,14, T8, measure.sh interleaved N=11, RAW=68229982, sha-OK):
markerfast vs mergefix(prior HEAD) = +1.2% / +3.0% / +0.0% over 3 runs = TIE
(spread 10-38%). decodeBlock 0.9568→0.9485s (~0.9%); rg decodeBlock 0.500s ⇒ gzippy
still ~1.9× but SLACK-MASKED (Fill 87%, Total Real Decode 0.137s, wall 0.175s =
0.73× rg, unchanged). KEPT per rule 7a (faithful TIE, gain latent behind binder).

ADVISOR: C1 UPHELD (widening + wrap-safety + no-desync + faithful range-check drop
all source-verified; emit_backref_ring is one shared fn). C2 UPHELD-WITH-CAVEATS
(honest TIE + interleaved freq-neutral, but spread can't see ≤10-20%). C3 REFUTED
(LOAD-BEARING): I read BOOTSTRAP_POST_FLIP_U16_BYTES BACKWARDS — it counts CLEAN
flipped bytes (gzip_chunk.rs:1489), so 2.0% is the sliver the loop does NOT touch;
the loop owns the ~98% marker COMPLEMENT. The exact counter-inversion the charter
warns about. Commit msg corrected.

CORRECTED PREMISE: "decodeBlock 1.69× = the marker loop" is now suspect — the loop
owns ~98% of bootstrap body yet barely moved decodeBlock and did NOT move the wall.
The engine SUM (~1.9×) is real but SLACK-MASKED at Fill 87% (Phase-0 already showed
engine TIE-vs-TIE seeded). The T8 binder is NOT the per-thread engine compute. NEXT
(do NOT start — supervisor gate): re-perceive the wall; the gap is the SCHEDULING/
SERIAL term (pool-fill + in-order consumer head-of-line wait =
project_confirmed_offset_prefetch_gap) — bound THAT with a removal oracle, not the
slack-masked engine. GUEST: /root/gzippy tree RESTORED to baseline (marker patch
reversed) + mergefix overlay; builds /tmp/gzbuild-{base,mergefix,markerfast} (all
sha 028bd002…cb410f); drivers /tmp/markerfast_{measure,trace}.sh + /tmp/sha_markerfast.sh
(bash); patch /tmp/marker_fastloop.patch. NO orphan processes.

## MERGE-REMOVAL LANDED → rg's view-based applyWindow ported; T8 wall MOVED +12% (0.65×→0.73× rg), byte-exact, advisor-UPHELD [2026-06-07, OWNER turn, branch reimplement-isa-l]
Executed step 1 of the bounded plan: port (ii) rg's view-based applyWindow = drop the redundant
full-output memcpy in `merge_resolved_markers_into_data`. Charter CURRENT STATE updated. Advisor:
plans/merge-removal-advisor-verdict.md. Brief: plans/merge-removal-advisor-brief.md.

CHANGE (chunk_fetcher.rs:2453 resolve_chunk_markers_on_chunk + chunk_data.rs): DROP
`merge_resolved_markers_into_data()` (~68MB full-output alloc+memcpy, segmented_buffer.rs:356) AND
the eager `recycle_markers_after_resolution()`. Narrowed marker bytes stay in `data_with_markers`
(narrowed_len set), emitted zero-copy via append_narrowed_iovecs; recycle DEFERRED behind the
consumer writev (defer_chunk_recycle → recycle_decoded_buffers frees both buffers). `contains_markers`
treats narrowed_len>0 as resolved (post-narrow u16 high bytes are stale). populate_subchunk_windows
assert relaxed. + debug-only double-resolve tripwire (advisor rec, byte-transparent). New test
populate_subchunk_windows_unmerged_view_based_apply_window.

BYTE-EXACT: gzippy-isal native (guest) + gzippy-native (local arm64) sha 028bd002…cb410f T1+T8
path=ParallelSM. 856 lib tests pass (1 fail = pre-existing flaky diff_ratio timing micro-test, fails
identically on unmodified 507d6ecb). Seam + native_fold_parity green.

REMOVE-AND-MEASURE (NOT the SUM, advisor Q4): locked guest 10.30.0.199 double-ssh, 16c gov=perf
turbo-on, taskset 0,2,4,6,8,10,12,14, T8, measure.sh interleaved N=11, RAW=68229982, sha-OK every run.
base(WITH merge) vs mergefix(REMOVED): run1 0.2291→0.2045 (+12.0%); run2 0.2128→0.1900 (+12.0%);
run3 0.2006→0.1765 (+13.7%, cleanest 6-13% spread). rg ratio base ~0.65× → mergefix ~0.73×. Sign
stable + load-invariant (1.64/2.80/1.86) ⇒ not a turbo artifact (interleaved = freq-neutral). KEEP.

ADVISOR: C1 UPHELD, C2 + C3 UPHELD-WITH-CAVEATS. Vendor citation accurate (change is MORE faithful);
no use-after-recycle on any emit path; re-resolution gates hold via !markers_resolved. Correction
ADOPTED: double-resolve tripwire (merge used to empty the buffer as the guard; safety now rests on
markers_resolved).

NEW WHOLE-SYSTEM WALL vs rg: T8 ~0.73× (was ~0.65×). Still a LOSS. NEXT (do NOT start — supervisor
gate): port (i) rg's multi-cached u16 marker loop (decodeBlock 1.69×, the larger gap), advisor-gated,
remove-and-measure. GUEST: /root/gzippy reset to clean 507d6ecb (prior overlays stashed as
owner-overlays-507turn) + /tmp/mergefix.patch. Builds /tmp/gzbuild-base (with merge) + /tmp/gzbuild-mergefix
(removed), both sha 028bd002…cb410f. Drivers /tmp/merge_measure.sh + /tmp/sha_check.sh (bash). NO orphans.

## CEILING BOUNDED → T8 TIE NEEDS TWO FAITHFUL PORTS (marker loop + view-based applyWindow); apply_window NOT at parity, divergence = a redundant memcpy [2026-06-07, OWNER turn, HEAD 507d6ecb +substep-timers-on-guest]
Paid the OWED apply_window measurement + source-verified rg's u16 marker-decode mechanism first-hand,
then DECOMPOSED gzippy's apply_window. Charter CURRENT STATE updated. Advisor:
plans/marker-kernel-ceiling-advisor-verdict.md (all UPHELD-WITH-CAVEATS, none refuted). Brief:
plans/marker-kernel-ceiling-brief.md. **SUPERVISOR GATE — fix build NOT started.**

RG MARKER MECHANISM (source, vendor deflate.hpp): readInternal (:1428) dispatches by Huffman-coding
TYPE not marker-vs-clean; with WITH_ISAL lit/len = readInternalCompressedMultiCached (:1453) for BOTH
u16 markers AND u8 clean (templated on Window). ONE loop; containsMarkerBytes = constexpr from element
type (:1600). Marker arms are cheap constexpr-gated only (dist-to-last-marker counter :1311-1317,
post-memcpy back-scan :1379-1389, inverse range-check skip :1652-1655); resolveBackreference fast arm
is std::memcpy for both (:1376). ⇒ NO separate slow marker path in rg; the 2× is gzippy's engine. The
faithful target is PORT rg's multi-cached u16 loop, NOT bolt AVX onto gzippy's loop (= E234 0.41×
plateau). Caveat: markers are u16 ⇒ ~2× clean traffic by construction; promise "marker == rg u16 loop."

OWED MEASUREMENT (locked guest 10.30.0.199 double-ssh, 16c gov=perf turbo-on load~1.0, taskset
0,2,4,6,8,10,12,14, T8, RAW=68229982, sha 028bd002…cb410f every run, /tmp/gzbuild-isal native,
measurement-only sub-step timers byte-exact NOT committed, 3 runs):
  decodeBlock gzippy 0.838s vs rg 0.497s = 1.69×.
  apply_window decompose (SUM/15 marker chunks): gather 0.044-0.064s | crc 0.013-0.019s |
  MERGE 0.116-0.134s | subwin 0.010-0.012s | TOTAL 0.19-0.27s.
  rg --verbose first-hand: "applying the last window" = 0.032s (NOT charter's cached 0.113s = WRONG),
  checksum 0.0096s.
KEY: gather (rg's applyWindow analogue) is ~1.5-2× and ALGORITHMICALLY IDENTICAL (base[i]=lut[v] ↔
rg target[i]=fullWindow[chunk[i]], DecodedData.hpp:335-337). The DOMINANT divergence = MERGE
(chunk_data.rs:1589 → segmented_buffer.rs:356 prepend_narrowed_from_markers): a full ~68MB output-size
memcpy that rg does as std::swap + in-place VectorViews (DecodedData.hpp:368-388). gzippy already has
the zero-copy emit (append_narrowed_iovecs) ⇒ the merge-copy is REDUNDANT for the iovec writer.

ADVISOR: all UPHELD-WITH-CAVEATS, none refuted. Q3 merge removable byte-exactly + faithful (every
consumer supports un-merged state, traced) but a STRUCTURED change: defer marker-recycle behind
consumer writev, relax populate_subchunk_windows narrowed_len==0 assert (chunk_data.rs:1291), keep
narrowed_len through write. Q4 (LOAD-BEARING) do NOT trust −0.12s SUM as wall delta — merge runs on
the pool; wall cost = only the un-overlapped fraction on consumer recv_post_process_blocking
(chunk_fetcher.rs:1769) for un-pre-resolved head chunks, bounded by resolve-ahead hit rate
(project_confirmed_offset_prefetch_gap). Provable ONLY remove-and-measure. Q5 ceiling DIRECTIONALLY
SOUND, two ports right + faithful, NOT yet a proven TIE; third residual (gather/crc ~1.5× = segmented
walk + per-chunk LUT rebuild vs rg contiguous + hoisted fullWindow) under-counted.

BOUNDED CEILING (revised honest): T8 TIE plausibly pure-Rust via TWO faithful ports — (i) rg
multi-cached u16 marker loop (decodeBlock 1.69×) + (ii) rg view-based applyWindow = drop the redundant
merge memcpy (0.12-0.13s divergence). The prior "marker-COMPUTE only" ceiling was OPTIMISTIC exactly
as advisor Q4 warned. SCOPED FIX next loop (do NOT start): land MERGE-REMOVAL FIRST (cheapest, payoff
most uncertain ⇒ measure first) — swap+views model, defer recycle, relax assert, emit via
append_narrowed_iovecs; byte-exact + measure interleaved T8 wall (freq-neutral). THEN the multi-cached
u16 marker loop. Each advisor-gated, each remove-and-measure (never the SUM, never the slope).
GUEST: /root/gzippy @7bf26096 + oracle overlay + decompose knobs + THIS turn's measurement-only
sub-step timers in chunk_fetcher.rs (gather/crc/merge/subwin, via /tmp/patch_resolve.py +
/tmp/patch_merge.py, NOT committed locally, byte-exact). Build /tmp/gzbuild-isal (native, rebuilt).
Drivers /tmp/applywin_measure.sh + /tmp/substep2_measure.sh (bash). NO orphan processes.

## BUNDLE DECOMPOSED → T8 SUB-LEVER = marker-COMPUTE (gzippy window-absent u16 decode ~2× slower than rg) [2026-06-07, OWNER turn, HEAD 5e9905c8 +decompose-knobs]
Decomposed the GZIPPY_SEED_WINDOWS bundle (advisor's 3-removal confound) on the whole-system T8
wall. Charter CURRENT STATE updated. Advisor: plans/t8-decompose-advisor-verdict.md. Pre-reg:
plans/t8-decompose-prereg.md. Findings: plans/t8-decompose-findings.md. **SUPERVISOR GATE — fix
build NOT started (bound-ceiling-first; one owed measurement remains).**

WHAT: 2 measurement-only env knobs (OFF==identity, byte-exact, NOT committed): GZIPPY_SEED_NO_WINDOWS=1
(suppress seeded-window fallback ⇒ seed-only-boundaries) + GZIPPY_SEED_NO_BOUNDARIES=1 (skip
block_finder pre-seed ⇒ seed-only-windows). src/decompress/parallel/seed_windows.rs + chunk_fetcher.rs.

MEASURED (locked guest 10.30.0.199 double-ssh, 16c gov=perf turbo-on load 1.3-2.0, measure.sh
interleaved N=11 CPUS=0,2,4,6,8,10,12,14 RAW=68229982 sha-OK=028bd002…cb410f every cell, 2 runs):
  rg 0.132s 1.000 | seedfull(both) 0.126-0.134s ~1.00× TIE | onlywin(windows) 0.199s 0.66× LOSS |
  onlybnd(boundaries) 0.198-0.205s 0.66× LOSS | prod(none) 0.198-0.203s 0.66× LOSS.
KEY: onlywin ≈ onlybnd ≈ prod (Δ<spread); ONLY seedfull (BOTH windows+boundaries) ties. f_windows≈0,
f_boundary≈0, yet seedfull ties ⇒ SUPER-ADDITIVE/COUPLED (pre-reg branch-4).
MECHANISM (GZIPPY_VERBOSE counters): seedfull window_seeded=17 spec-fail=0 Fill=91% decodeBlock=0.846s
(CLEAN). onlywin seed_hits=0 (windows UNUSABLE at partition-guess offsets) ≡ prod. onlybnd spec-fail
13→0 (real boundaries kill spec-failures) BUT body still 170MB/s u16 decodeBlock=1.106s ≈ prod ⇒
WALL-NEUTRAL. APPLES-TO-APPLES rg --verbose (both window-absent, same 34.5% markers): rg decodeBlock
0.542s vs gzippy prod 1.067s ⇒ rg's u16 marker decode ~2× FASTER per byte. rg ties WITHOUT seeding.

ADVISOR: core UPHELD-WITH-CAVEATS. Q1 the 2×2 knobs CANNOT separate marker-compute from boundary-
alignment — onlywin is DEGENERATE (windows unusable without boundaries by construction ⇒ ≡ prod,
pre-reg self-test FAILED ⇒ the COUPLED branch); re-attribute to onlybnd + rg-comparison. Q2 onlybnd
UPHELD-W-CAVEATS (spec-failures not the cost, wall-neutral). Q3 the 2× rate gap is FAIR (denominator-
matched, applyWindow separate in both, survives spec-failure removal) = STRONGEST pillar. Q4 (MOST
IMPORTANT) ceiling OPTIMISTIC: seedfull removes marker-premium AND applyWindow ⇒ bounds route-(ii)
not the faithful route-(i) (fast u16 marker decode KEEPS applyWindow); route-(i) ceiling rests on the
rapidgzip existence proof (rg 0.54 decode + ~0.113s applyWindow → 0.13 wall), conditional on
gzippy's applyWindow ≈ rg's.

PINPOINTED: T8 sub-lever = marker-COMPUTE (window-absent u16 decode ~2× rg). NOT boundary-alignment
(secondary precondition, wall-neutral) NOT spec-failures (wall-neutral). CEILING ≤ T8 1.0× TIE
CONDITIONAL on apply_window parity vs rg's ~0.113s.
NEXT (next loop, do NOT start now): igzip-class u16 marker-decode kernel (asm/inner-kernel techniques
adapted to u16 marker output — in scope HERE, Phase-0 ISA-L oracle never tested the marker path).
PLUS the OWED measurement: time gzippy's apply_window/marker-resolution vs rg's ~0.113s (needs a
fast-marker prototype or direct timer; no existing cell isolates it).
GUEST: /root/gzippy src @7bf26096 + oracle overlay + this turn's 2 decompose knobs (applied on guest,
NOT committed locally). Build /tmp/gzbuild-isal (gzippy-isal target-cpu=native byte-exact). Seeds
/tmp/seeds.bin (16 windows). Driver /tmp/decompose_measure.sh (use bash). NO orphan processes.

## PHASE-0 WALL ORACLE DONE → T8 BINDER IS THE WINDOW-ABSENT MARKER PATH, NOT THE ENGINE [2026-06-07, OWNER turn, HEAD 3895a23c +oracle]
PHASE-0 of the asm-port project: dropped a REAL ISA-L engine into the PRODUCTION parallel-SM
pipeline (pool/consumer/ring/CRC/window-publish kept) and measured the T8 WALL vs rapidgzip on the
locked guest. Charter CURRENT STATE updated. Advisor: plans/asmport-phase0-advisor-verdict.md.
Pre-reg + results: plans/asmport-phase0-prereg.md. Brief: plans/asmport-phase0-advisor-brief.md.
**SUPERVISOR GATE reached — Phase 1 (asm transliteration) NOT started, per the prompt.**

WHAT: `GZIPPY_ISAL_ENGINE_ORACLE=1` (measurement-only, env-gated, byte-exact, NOT production) routes
`finish_decode_chunk_impl`'s clean-tail decode through REAL ISA-L FFI
(`decompress_deflate_from_bit_with_boundaries`), feeding bytes/boundaries/end-bit through the SAME
ChunkData primitives. ISA-L input bounded to `[..stop_hint/8+256KiB]` (per-chunk, not whole-member —
the FIRST cut decoded the whole member per worker → 0.42s; bounding fixed it). Windows SEEDED
(`GZIPPY_SEED_WINDOWS`, captured T1) so all 18 chunks are window-present → reach the oracle. PROVEN
ISA-L ran: `isal_oracle_chunks=16 isal_oracle_fallbacks=1`.

MEASURED (locked guest 10.30.0.199 double-ssh, 16c gov=perf turbo-on load 2.7-4.2, measure.sh
interleaved N=11 CPUS=0,2,4,6,8,10,12,14 RAW=68229982 sha-OK=028bd002…cb410f, 2 runs):
  rg 0.134s 1.000 | isal(ISA-L,seeded) 0.148s 0.905/0.892 TIE | pure(pure-Rust,seeded) 0.134s
  1.002/0.968 TIE | prod(pure-Rust,no-seed) 0.194s 0.690/0.652 LOSS.
KEY: `pure` (the SLOWER engine) ALREADY ties rg when seeded; engine swap is TIE-vs-TIE ⇒ the engine
is NOT the T8 binder. The ~1.5× prod gap collapses to TIE when the window-absent path is removed.
Per-stage: prod decodeBlock SUM 1.048s/Real 0.169s/Fill 77%/body 168MB/s/13 spec-fails; pure-seed
0.781s/0.108s/Fill 90.55%/0/0. rapidgzip runs the SAME 34.5% marker workload WITHOUT seeding yet
ties (rg --verbose) ⇒ gzippy's window-absent path is the slow one (apples-to-apples).

ADVISOR: C1 UPHELD-W-CAVEATS (oracle real, clean-tail only, 1-chunk impurity). C2 UPHELD-W-CAVEATS
(engine not the T8 binder; but 1.51× engine gap is REAL + slack-masked at Fill 90%, NOT at parity).
C3 UPHELD as COARSE localization — seeding is a valid causal removal + NOT unfair, BUT bundles 3
removals (u16 marker decode, block_finder real-boundary pre-seed vs partition GUESS, 13 spec-fail
re-decodes) ⇒ cannot isolate marker-COMPUTE from boundary-ALIGNMENT. C4 directional rec UPHELD at
T8, strong inference REFUTED (marker-phase rate is on the binding path + never replaced; T1 unaddressed).

NEXT (supervisor gate): DECOMPOSE the bundle before choosing Phase 1 — seed-ONLY-boundaries (no
windows) vs seed-ONLY-windows (prod boundaries). If the delta is boundary-ALIGNMENT, the lever is
the block finder / prefetch horizon (project_confirmed_offset_prefetch_gap), NOT the asm engine. If
marker-COMPUTE, a faster u16 marker kernel. The asm engine port is NOT the T8 lever (pure ties T8
seeded) but remains the plausible T1 lever + helps marker-phase rate — re-scope, don't abandon.
GUEST: /root/gzippy src @7bf2609 + this turn's overlay (oracle in gzip_chunk.rs + chunk_fetcher.rs);
builds /tmp/gzbuild-isal (gzippy-isal) + /tmp/gzbuild-native (gzippy-native), both byte-exact;
seeds /tmp/seeds.bin; driver /tmp/phase0_measure.sh (use `bash`, not sh — measure.sh needs bash).
Tree: oracle committed this turn; NO orphan processes (advisor + polls killed; guest clean).

## PURE-RUST ENGINE CEILING BOUNDED → PLATEAU (~0.6× ISA-L isolation); FORK real, WALL bound owed [2026-06-07, OWNER turn, HEAD f8260aa8 +bench]
Loop step (B) executed: built + measured the faithful-u8 engine CEILING vs ISA-L (isolation, bounded).
Charter CURRENT STATE updated. Advisor: plans/engine-ceiling-advisor-verdict.md (PLATEAU UPHELD-WITH-
CAVEATS). Brief: plans/engine-ceiling-advisor-brief.md. Bench committed this turn.

WHAT: added VAR_VI to benches/engine_isolation.rs = VAR_V speculative flat-u8 pipeline + igzip
packed-u32 table (tricks #1/#2/#3) PLUS (1) BMI2 BZHI distance-extra extraction, (2) AVX2/SSE MOVDQU
wide overlap-copy back-ref. trick #3 confirmed fully exploited.
MEASURED (locked guest 10.30.0.199 double-ssh, 16c gov=perf load~3.3 turbo-on, taskset -c 0, N=11
interleaved, native target-cpu BMI2+AVX2 LIVE avx2_detected=true, 2 stable runs):
  ISA-L 847-851 MB/s | VAR_V 460-462 (0.54×) | VAR_VI 504-525 (0.59-0.62×); per-chunk VI 0.55-0.64.
  BMI2+AVX added ~9-14% over VAR_V, did NOT close the gap. SELFTEST PASS (iii/i=2.73).
BYTE-EXACT: VAR_VI printed MBps (never VOID) every chunk ⇒ byte-identical to scalar AND ISA-L over
the full timed window (bench gate exact[k]). Top-line SHA_ALL_EQUAL=no = PRE-EXISTING VAR_IV_E234
failures (untouched path), NOT VAR_VI.
FALSIFIER FIRED → PLATEAU: VAR_VI ≈0.6× ISA-L, ~23pp below the 0.85 PASS line, WITH the full igzip
stack + inline-ASM intrinsics. Pure-Rust igzip-class as a STANDALONE PRIMITIVE not reached.
ADVISOR: PLATEAU UPHELD-WITH-CAVEATS — all 5 techniques LIVE, fast loop is the timed path, header
symmetric, byte-exact airtight; crediting every caveat lifts a "fixed" VI to ≤~0.65-0.68, still
17-20pp short. LOAD-BEARING CAVEAT: escalating to "1.0× WALL hard-bounded at 0.6×" OVERREACHES
isolation (Rule 3 forbidden extrapolation); the WALL bound needs a PRODUCTION engine-removal oracle.
The floor-to-floor T8 1.74× finding (prior turn, advisor-upheld) independently corroborates the
engine gap reaches the wall, so the FORK is strongly implicated but the clean WALL hard-bound is owed.

NEXT (supervisor gate): (a) escalate the fork with the engine-primitive hard number (~0.6× ISA-L
PLATEAU) + corroborating wall evidence; OR (b, owner-recommended) run the production-pipeline
engine-removal oracle first to convert the engine ceiling into a clean WALL hard-bound, then escalate.
GUEST: /root/gzippy src @7bf26096 + this turn's bench overlay (benches/engine_isolation.rs has VAR_VI);
build /tmp/gzbuild (CARGO_TARGET_DIR), binary engine_isolation-9b415678a2bcb34c; /tmp/engine.seed ->
/dev/shm/engine.seed; corpus /root/gzippy/benchmark_data/silesia-gzip.tar.gz; /root/fulcrum stub present.
Tree: bench committed; no orphan processes (killed claude -p wrapper + sleep watchdogs).

## T8 BINDER RE-LOCATED to the ENGINE; "serial/consumer-wait" binder REFUTED as a unit error [2026-06-07, OWNER turn, HEAD f8260aa8]
Loop step (A) executed (perceive → ceiling-bound → causal-ID → advisor). Charter CURRENT STATE
updated. Advisor: plans/t8-engine-binder-advisor-verdict.md (claim1 UPHELD; 2,3 UPHELD-W-CAVEATS).
Brief: plans/t8-engine-binder-advisor-brief.md. NO code change this turn (perception + advisor).
Tree clean (only untracked plan/script files); NO orphan processes.

THE PRIOR BINDER ("decode floor 0.118s ≈ rapidgzip wall 0.130s ⇒ gap is all scheduling/consumer-
wait") IS A UNIT ERROR: it compared gzippy's FLOOR to rapidgzip's WALL. Floor-to-floor (both
--verbose Theoretical-Optimal): gzippy 0.118s vs rapidgzip 0.068s = 1.74×. decodeBlock SUM 0.93s
vs 0.50s = 1.86×. The T8 BINDER is the per-thread DECODE ENGINE (body_rate 269 vs ~424 MB/s).
future::get consumer-wait 0.077 vs 0.062 = 1.25× = MINORITY + downstream. Matches the constant
~1.7× at BOTH T1 and T8 (flat-across-T = per-thread throughput gap). DECODE-BYPASS + SLEEP-DECODE
oracles are CONFOUNDED (decode-free wall 3.6-5.5× SLOWER — buffer-pool bypass + per-chunk zeroed
allocs/faults + 660MB live + un-overlapped 212MB CRC); use floor-to-floor span comparison instead.
Vendor BlockFetcher.hpp:246-329: rapidgzip ALSO pumps prefetch in `while(wait_for(1ms))` during
future-wait — overlap structure already faithfully ported, no missing mechanism.

NEXT: step (B) = the ENGINE is the binder. Advisor caveat D-D (LOAD-BEARING): the round-2 "2.4×
plateau" that declared pure-Rust unreachable was on the DISCREDITED u16-RING arch — does NOT bound
the current faithful u8-direct engine (fc1c965b). pure-Rust→1.0× is OPEN. USER-CONSTRAINT FORK
implicated: resolve by BUILDING+measuring the faithful-u8 engine ceiling vs ISA-L on the prod
speculative path (igzip-class inner-Huffman: packed-u32 table, speculative 8B literal store +
next-sym preload, BMI2, MOVDQU overlap copy, slop headroom), NOT by extrapolating the u16 plateau.
GUEST: gzippy-mk2 byte-exact at /tmp; silesia /root/silesia.gz; cap_t1.bin+cap_t8.bin bypass
captures + measure_oracleA.sh/oracle_sleep.sh on guest /tmp (kept for reuse).

## [SUPERSEDED — unit error] T8 BINDER LOCATED = SERIAL/CONSUMER-WAIT; u16-path premise FALSIFIED [2026-06-07, OWNER turn, HEAD fb3baec0]
Loop step (A) executed in full (perceive → causal-ID → advisor → instrument committed).
Charter CURRENT STATE updated. Advisor: plans/u16-ceiling-advisor-verdict.md (UPHELD /
UPHELD-WITH-CAVEATS). Brief: plans/u16-ceiling-advisor-brief.md.

PREMISE CORRECTION (source + advisor UPHELD): the charter's "58.6% of bytes take the slow u16
path = biggest prize" was a MIS-READING of the mis-named counter BOOTSTRAP_POST_FLIP_U16_BYTES
(gzip_chunk.rs:97/:1302 increments on marker-FREE blocks, which since fc1c965b decode u8-DIRECT).
Genuine u16-<true> fraction ≈ 42.5% inverse (the pre-flip prefix), NOT the bulk-on-a-slow-path.

CAUSAL (new GZIPPY_SLOW_MARKER_MODE u16-path knob, commit fb3baec0, byte-exact 028bd002…cb410f
all knob modes; locked guest 10.30.0.199 double-ssh, 16c gov=perf, measure.sh interleaved sha-OK,
RAW=211968000, T8 N=11; box load 3-5 ⇒ trust interleaved-relative only):
- CLEAN +100% spin → +27%; CLEAN +100% SLEEP → +27% (IDENTICAL = not turbo); +200% SLEEP → +55%.
  ⇒ clean u8 decode genuinely gates ~27% of T8 wall (freq-neutral).
- MARKER +200% spin → +21%; SLEEP control → +7% (collapses). ⇒ u16 marker = MINORITY ~3.5-14%.
- T1 MARKER → near-flat (validates: u16 barely runs at T1, knob fires ∝ u16 bytes).

BINDER LOCATED (pool trace, first-hand): decodeBlock 0.936s → Theoretical-Optimal(÷8)=0.117s;
Real Decode span 0.147s (Fill 79%); std::future::get in-order consumer wait = 0.077s; header_ms
24.0 (~2.6%). gzippy's perfectly-parallel decode FLOOR (0.117s) ALREADY ≈ rapidgzip's ENTIRE wall
(0.130s). The whole 1.70× gap = scheduling/serial: pool-fill gap ~0.030s + consumer head-of-line
future::get ~0.077s ≈ ~0.10s. rapidgzip ties DESPITE the same engine gap by overlapping decode.

NEXT: FIX the in-order consumer future::get head-of-line wait + pool-fill gap (charter binder #2 /
project_confirmed_offset_prefetch_gap). Bound the ceiling FIRST with a consumer-wait-removal oracle
(Rule 3). The prior placement-port gate FAILED on offset-supply (a non-divergence); the OPEN
distinct question is prefetch-HORIZON/dispatch-depth, not offset supply. Kernel stays the T1 lever
+ a real ~27% T8 term but is NOT the T8 path to 1.0×.

GUEST BUILD NOTE (next owner): guest src build needs a `/tmp/fulcrum` stub crate (Cargo.toml
optional dep `fulcrum = {path="../fulcrum"}` — cargo reads its manifest even with coz OFF) AND the
tarball must include crates/ + examples/ (workspace member + [[example]] coz_bench). Build dir
/tmp/gzbuild on guest; binary gzippy-mk2 (D2-fixed) byte-exact. silesia.gz is at /root/silesia.gz.

## T8 BINDER RE-IDENTIFIED — kernel REFUTED as the T8 lever; u16-path + scheduling bind [2026-06-07, OWNER turn] [u16 part SUPERSEDED above]
Full perceive→causal-ID→advisor loop. Charter CURRENT STATE updated. Advisor:
plans/t8-binder-advisor-verdict.md (UPHELD-WITH-CAVEATS). Brief: plans/t8-binder-advisor-brief.md.

WHOLE-SYSTEM WALL (locked guest 10.30.0.199 -J neurotic, 16c gov=perf no_turbo=1, measure.sh
interleaved sha-verified, RAW=211968000): T8 gzippy-varv ~0.226s vs rapidgzip ~0.137s = 1.655×;
varv vs base TIE; sha 028bd002…cb410f OK. (gzippy-base 16:19 + gzippy-varv 16:18 on guest /tmp,
source /root/gzippy @ 7bf26096 + VAR_V worktree; rapidgzip 0.16.0.)

CAUSAL PERTURBATION (slow_knob, byte-transparent, freq-neutral sleep control, site fires ∝ clean
bytes; reproduced 3×, sha OK every run):
- T1 spin100 (doubles per-thread decode-compute) → +83% wall ⇒ decode GATES ~83% of T1 wall
  (kernel = confirmed T1 lever).
- T8 spin100 → +14–22%; spin200 → +45%; sleep100 control +20% (≥ spin, not a turbo artifact)
  ⇒ clean decode-compute gates only ~18–22% of T8 wall.
- COVERAGE CONFOUND (advisor Angle-4, reconciled first-hand): slow_knob is CLEAN-only (const-folds
  to 0 on marker <true>). clean hits T1=38.7M vs T8=28.4M (73% coverage) ⇒ ~27% of T8 events run
  MARKER mode uncovered. Coverage-corrected T8 decode ceiling ≈ 25–30% (advisor: ≤~45% w/ Rule-3
  unbind slack). Decode-compute is a MINORITY of the T8 wall regardless.

T8 BINDER (the ≥55–75% that is NOT clean decode-compute), GZIPPY_VERBOSE trace first-hand:
1. **u16 post-flip/marker path = 58.6% of decoded BODY bytes** (post_flip_u16_bytes=118.6M, trace
   labels it "Design-B1 prize"). Bulk of T≥2 bytes flow through the slow u16 marker→drain path,
   NOT the clean u8 fast path VAR_V/kernel optimized — why the clean gain was absorbed + slow_knob
   barely moved T8. Speculation header-failures 14/19; body_rate blended 286 MB/s.
2. **Pool sched + serial tail:** Theoretical-Optimal 0.127s → Real-Decode 0.162s (~28% pool
   inefficiency, fill 73–83%, dispatch saturated ~51/60) → wall ~0.22s (~0.06s SERIAL outside the
   pool: in-order publish / drain / CRC). Corroborates project_confirmed_offset_prefetch_gap.

RE-POINT (per PROCESS — bottleneck moved off the clean kernel at T8): NEXT = (A) ORACLE-bound the
u16-path ceiling (if 58.6% u16 body ran at clean-path rate, how much wall drops? likely the bigger
prize — rapidgzip ties DESPITE the same engine gap via in-flight overlap); (B) pool-sched/serial-
tail SERIAL-WORK-vs-DECODE-WAIT decomposition. Inner kernel KEPT as confirmed T1 lever, NOT the T8
path to 1.0×. No build this turn (perception + causal ID + advisor only). Tree clean; no orphans.
NOTE for next owner: guest reachable ONLY as `ssh neurotic 'ssh 10.30.0.199 "…"'` (local key not
authorized on guest; -J neurotic from local FAILS the final hop). Watch leftover `timeout` sleep
watchdogs after claude -p / measurements — kill -P the wrapper pid.

## WINDOW-DISCARD LEVER — SOURCE-VERIFIED + QUANTIFIED → **FALSIFIED** [2026-06-07, leader fresh instance]
Charter: plans/window-discard-and-overhead-diagnosis.md. The advisor flagged
`reset(None, window_opt)` (gzip_chunk.rs:1107) as discarding an available predecessor window.
**FULL CALL-GRAPH TRACE REFUTES IT.** No window is discarded.

STEP 0: VAR_V committed as 9b674651 (byte-exact TIE kept per rule 7a, gain latent).

WHY NO DISCARD (source, first-hand):
- gzip_chunk.rs:1107 `block.reset(None, window_opt)` — `output=None` makes the seed guard
  (`marker_inflate.rs:590 if let (Some(out),Some(window))`) false. BUT its ONLY caller,
  `decode_chunk_unified_marker` (gzip_chunk.rs:743), hard-codes `marker_decode_step(..., &[], ...)`.
  So `initial_window=&[]` → `window_opt=None` ALWAYS at :1107. It discards nothing — there is no
  window on that path to begin with. (The seed `output` param is only a precondition gate; the
  seed actually writes the predecessor into `output_ring`, marker_inflate.rs:709 — not into output.)
- The window-PRESENT case is handled SEPARATELY + EARLIER: `decode_chunk_with_rapidgzip_impl`
  (gzip_chunk.rs:598) `if initial_window.len()==MAX_WINDOW_SIZE` → finish_decode_chunk_impl /
  _with_inexact_offset WITH the window = the clean seeded fast path (vendor setInitialWindow). It
  NEVER reaches the marker bootstrap. Worker dispatch (chunk_fetcher.rs:2323) routes a
  `window_map.get(start_bit)==Some` chunk straight there with materialized window bytes.
- The +8.7% window-seed (commit 9949d2f0) is LIVE for window-present chunks via this :598 path;
  it was implemented in MarkerRing (now `GZIPPY_MARKER_RING=1` legacy only) but the production
  vendor-Block path has its OWN equivalent seeded route at :598. Not regressed.

QUANTIFIED (silesia 68MB, pure-rust-inflate, GZIPPY_VERBOSE, path=ParallelSM, ~18 chunks):
- T1: window_seeded=16, finished_no_flip=0  → EVERY non-zero chunk seeds its window.
- T2/T4: window_seeded=2, finished_no_flip=15 → ~89% window-absent.
- T8: window_seeded=2, finished_no_flip=16 → ~89% window-absent.
The T1→T8 collapse is the CAUSE: at T1 the in-order consumer publishes each window before the
single worker needs the next chunk; at T≥2 workers race AHEAD of in-order publication →
`window_map.get` returns None → speculative marker bootstrap. This is the FAITHFUL vendor model
(rapidgzip is also ~97% window-absent at runtime, chunk_fetcher.rs:2235-2250 non-blocking get).
The 16 absent chunks at T8 had NO predecessor window AVAILABLE-AND-DISCARDED — they were
genuinely not-yet-published. **No discard. No lever here.**

FALSIFIER (pre-registered, then RESOLVED): "If a window were discarded, T1 and T8 would BOTH
show high finished_no_flip (the seed never fires regardless of availability)." MEASURED: T1
shows finished_no_flip=0 / window_seeded=16. The seed DOES fire whenever a window is available.
Falsifier triggered ⇒ window-discard claim REJECTED.

NEXT (per charter step 2, NOT yet started — gated): the real binder is production clean-path
OVERHEADS that absorbed VAR_V's +48% isolation gain (ring %U8_RING_SIZE / reversed-bits dist /
backref arms / resumable cap). The only window-related angle that remains FAITHFUL is closing the
worker-vs-consumer publication race so MORE chunks seed at high T — but rapidgzip has the same race,
so that is parity-matching not a gzippy-only fix. STOPPED at supervisor gate before any large build.



## VAR_V INTEGRATION + REAL-WALL MEASUREMENT — DONE [2026-06-07, leader fresh instance]
Charter: plans/varv-integrate-and-measure.md. VERDICT: byte-exact, WALL TIE, decode STILL binds.

### INTEGRATION (faithful, real overheads kept — NO shortcut)
- VAR_V speculative software-pipelined fast loop ported INTO production
  `read_internal_compressed_specialized::<false>` (marker_inflate.rs ~:1454), the clean
  post-flip path = 89% of production (trace: finished_no_flip=16/18 chunks). Ported ONTO the
  REAL production wrapping u8 ring (`% U8_RING_SIZE`) — NOT a flat buffer. Kept ALL the overheads
  the bench elided: ring modulo + wrap-straddle (back-ref via the EXISTING `emit_backref_ring_u8`),
  the resumable `n_max_to_decode` cap (top-of-loop guard `emitted+FAST_OUT_SLOP < n_max_to_decode`),
  drain, CRC (downstream sink unchanged). Distance via production's `self.dist_hc` (cached
  reversed-bits), NOT the bench's LutDistCode — faithful + byte-exact. Fast loop runs only while
  the physical dst region won't wrap (`dst_phys+FAST_OUT_SLOP<=U8_RING_SIZE`) + input slop; falls
  to the unchanged careful per-symbol loop near wrap / resumable boundary / block tail. Const-folds
  away entirely on the marker `<true>` path. (No commit yet — STOPPED at supervisor gate.)
- BYTE-EXACT (dual-feature, dual-arch): gzippy-native arm64 AND gzippy-isal x86_64 (Rosetta
  cross-compile) BOTH emit silesia sha 028bd002...cb410f via path=ParallelSM, T1 AND T8. Guest
  (x86_64) native build also sha-exact. Tests: routing 37/0, pure_rust_inflate_corpus, the
  adversarial seam test (faithful_u8_flip_seam_max_distance_backref_vs_flate2) + native_fold_parity
  all GREEN (--features gzippy-native, --test-threads=1).

### MEASUREMENT (leader-run, locked guest trainer=10.30.0.199 -J neurotic, 16c, gov=performance)
- Built base (fa9fd73c, VAR_V reverted via patch -R) + varv (HEAD+VAR_V) on guest,
  RUSTFLAGS=-C target-cpu=native, both sha-exact. rapidgzip 0.16.0 same sha. RAW=211968000.
  measure.sh interleaved N=11, taskset-pinned, output-sha-verified=OK every run.
- T8 (CPUS=0,2,4,6,8,10,12,14): run1 varv 0.2145s vs base 0.2220s = 1.035x TIE (spread 10%);
  run2 varv 0.2213s vs base 0.2137s = 0.966x TIE (spread 31%). Sign FLIPS between runs ⇒
  noise-dominated TIE, mean ≈ 1.00x. rapidgzip 1.62–1.71x vs base (gap UNCHANGED).
- T1 (CPUS=0): varv 0.5165s vs base 0.5256s = 1.018x TIE (spread 7%). rapidgzip 1.702x vs base
  (gap UNCHANGED — identical to the pre-VAR_V faithful-u8 measurement's 1.702/1.704x).
- PER-STAGE TRACE (GZIPPY_VERBOSE, T8): decodeBlock base 1.227s vs varv 1.248s (= same, +1.7%
  noise); Pool Fill Factor 82.25% (base) / 82.65% (varv) — workers ~83% busy DECODING, placement
  overhead only ~17%. finished_no_flip=16 (VAR_V IS on the production bulk path).

### VERDICT — decode STILL binds; VAR_V's 0.555 did NOT survive integration
- The wall is a TIE at BOTH T1 and T8; the rapidgzip gap is CONSTANT ~1.70x (unchanged from the
  prior faithful-u8 TIE). The §3 projection (decode_wall 0.410s, "decode stops binding") is
  FALSIFIED on the real wall: producer-side `decodeBlock` is UNCHANGED (1.227→1.248s) — the
  speculative-store gain measured in the flat-buffer bench (+48% vs scalar, 0.555x ISA-L) was
  ABSORBED by the real ring/wrap/resumable overheads the bench elided (exactly the advisor's
  "0.555 is optimistic, real integration ≤ 0.555" caution, now confirmed quantitatively).
- CONTINGENT NEXT LEVER (per charter): decode STILL binds ⇒ the inline-asm/BMI2 spike to push the
  rate is the live direction (NOT placement — placement is ~17% slack here, fill-factor 83%).
  Do NOT start it this turn. The byte-exact VAR_V integration is a TIE — KEEP per rule 7a
  (correct + faithful inner-loop port) but it is NOT itself the wall lever.
- Independent disproof advisor: plans/varv-integration-advisor-verdict.md (synchronous, read-only).
- STOPPED for supervisor gate.

## FAITHFUL u8 NATIVE REWRITE — IN PROGRESS [2026-06-07, leader fresh instance]
Charter: plans/faithful-u8-native-rewrite.md. Leader-lock held (pid in /tmp).
### STEP 1 DONE — production clean path ESTABLISHED (source-map subagent, exit 0, all file:line first-hand)
- TWO native clean-bulk paths exist, selected by whether a 32KiB predecessor window is known:
  (A) window-SEEDED chunks -> ResumableInflate2 via StreamingInflateWrapper (resumable.rs:469/494/507
      writes u8-DIRECT into caller's &mut [u8]) -> ALREADY FAITHFUL u8, no u16, no narrow.
  (B) window-ABSENT/speculative chunks -> Engine M marker_inflate::Block (output_ring: Box<[u16;65536]>,
      marker_inflate.rs:290) -> the u16 ring + post-flip narrow-at-drain (drain_to_output :738-750).
      THIS is the only u16 storage left; the FOLD (gzip_chunk.rs:1205) keeps Engine M running the
      clean tail in-place on native (FlipToClean gated behind isal_clean_tail = OFF).
- PRODUCTION MEASUREMENT (silesia T8, gzippy-native arm64, GZIPPY_VERBOSE, sha 028bd002...410f VERIFIED):
  Unified decoder: finished_no_flip=16  flip_to_clean=0  window_seeded=2  inflate_wrapper=0  finish_decode=2
  => 16/18 chunks (89%) go through Engine M's u16 ring (finish in marker+clean mode, NO flip), only 2
  window-seeded (u8). So the u16 ring IS the production bulk path; the engine-bench PLATEAU measured a
  representative architecture. The faithful u8 rewrite of Engine M is on-target (NOT a dead path).
- The prior u8-direct port (5256075) landed in lut_bulk_inflate.rs::MarkerRing (conflate_to_clean_u8),
  reached ONLY via GZIPPY_MARKER_RING env (gzip_chunk.rs:1018) — DEAD on production. NOT on Block.
- Pre-req build fix: bench-only E2-E4 (read_clean_e234 / emit_backref_ring_clean / emit_avx2_copy_u16,
  7bf26096) used x86_64-only AVX2 intrinsics under a pure_inflate_decode cfg that also fires on arm64 =>
  arm64 build broke. Gated those three behind all(pure_inflate_decode, target_arch="x86_64"). Byte-
  transparent (dead bench code; production sha unchanged 028bd002...410f).
### STEP 2 — u8-width flip-in-place rewrite of Engine M: DONE (committed fc1c965b + test hardening).
- The faithful u8-direct rewrite LANDED byte-exact. Engine M post-flip now decodes the clean bulk
  u8-DIRECT into the u8 VIEW of the SAME output_ring backing (vendor getWindow() reinterpret
  deflate.hpp:890-894); flip = drain_transition_narrow_u16 (one-shot) + flip_repack_to_u8 (vendor
  setInitialWindow conflation 1762-1782: value-DOWNCAST (x&0xFF) the rotated 32KiB window into the u8
  view upper half, re-base cursor to u8-logical BASE=U8_RING_SIZE). emit_backref_ring_u8 = faithful u8
  port (8-byte word copy, distance>=8, RLE memset, overlap), no marker scan. set_initial_window_impl +
  read_internal_uncompressed store u8 in clean mode. Marker (<true>) path const-folded byte-IDENTICAL.
- BYTE-EXACT: silesia T1/T8/T16 == 028bd002...410f on gzippy-native (arm64) AND gzippy-isal (x86_64
  Rosetta cross-compile); 851 lib tests; clippy clean. MANDATORY adversarial seam test added +
  HARDENED: faithful_u8_flip_seam_max_distance_backref_vs_flate2 (6×A=192KiB so flip fires ~65278 and
  the distance-32768 back-refs are UNAMBIGUOUSLY post-flip; asserted vs independent flate2 oracle;
  sentinel at byte 163840 proves the repack rotation+downcast). native_fold_parity + the prior seam
  unit tests all green.

### STEP 4 — MEASUREMENT (leader-run, locked guest 10.30.0.199 via -J neurotic): VERDICT = byte-exact TIE.
- Guest: 16 cores, governor=performance, load ~1.0 (~6% on 16c, measure.sh did NOT flag busy). no_turbo
  write was Permission-denied (guest VM; host controls turbo) — measure.sh's INTERLEAVED RELATIVE delta
  is turbo-immune (both tools see the same per-trial state), so the ratio is the authoritative signal.
- Uploaded /tmp/silesia.gz (decoded sha 028bd002...410f VERIFIED on guest). Built BOTH binaries on guest
  (RUSTFLAGS=-C target-cpu=native): /tmp/gzippy-base (u16 baseline from git HEAD 7bf2609) + /tmp/gzippy-u8
  (synced u8 source). Both byte-exact via path=ParallelSM.
- 3-WAY INTERLEAVED measure.sh (RAW=211968000, sha-verified=OK, taskset-pinned):
    T8 (CPUS=0,2,4,6,8,10,12,14, N=11): base 0.219s/968MB/s (spread 31%); u8 0.218s/972MB/s (spread 14%);
      rapidgzip 0.1285s/1650MB/s.  => u8 1.004x vs base = TIE; rapidgzip 1.704x vs base.
    T1 (CPUS=0, N=11 then N=15): base 0.521-0.526s/403-407MB/s (spread 7-9%); u8 0.537-0.539s/393-394MB/s
      (spread 4%); rapidgzip 0.306s/694MB/s.  => u8 0.976x vs base = TIE; rapidgzip 1.705x vs base.
- FINDINGS:
  1. The faithful u8 rewrite is a byte-exact TIE at BOTH T1 and T8 (Δ 2.4% << spread). KEEP (faithful,
     correct, layer-don't-revert).
  2. The gzippy->rapidgzip gap is CONSTANT ~1.70x at T1 AND T8. A flat ratio across T is the signature of
     a PER-THREAD decode-throughput gap (decode BINDS), NOT scheduling/placement.
  3. CAUSAL DISPROOF of the traffic hypothesis: the u8 rewrite is a clean ~2x perturbation of clean-path
     memory traffic (u16->u8 ring + u8 copies + no narrow) with a FLAT wall response => traffic is SLACK;
     the binding term is per-symbol LUT-DECODE COMPUTE. This FALSIFIES round-2's "u8 clean ring is the
     main lever" and confirms the round-2 PLATEAU (engine ~2.4x ISA-L on compute). The advisor's prior
     prediction "flat-u8 stops binding above ~120MB/s" is FALSIFIED: u8 = 393MB/s single-core, still
     1.70x off rapidgzip, decode still binds.
  4. CAVEAT (not load-bearing): measure.sh absolutes (393-972 MB/s) are a different sink (stdout->mktemp)
     than the charter's "0.604s same-sink wall" — NOT apples-to-apples. The verdict is RELATIVE (u8 vs
     base, same harness) and stands; the 0.604s absolute question is answered only as "u8 did not change
     the relative gap to rapidgzip."
- INDEPENDENT DISPROOF ADVISOR (synchronous, read-only, plans/faithful-u8-native-advisor-verdict.md):
  A byte-exact UPHELD; B seam UPHELD-WITH-CAVEATS (sentinel pinpoint softer than advertised — FIXED by
  the test hardening above); C wall-TIE UPHELD (TIE band sound; mechanism inference is a valid causal
  disproof; cross-harness 0.604s not load-bearing). Mandate SATISFIED structurally. REAL next lever =
  inner Huffman LUT-decode compute kernel (igzip-class packed-table/speculative-store/preload), NOT
  traffic/placement/ring.
- STOPPED for supervisor gate. The asm-kernel port + gzippy-isal u8+FFI are LATER/contingent.

## ASM-KERNEL FEASIBILITY SCOPE — CHECKPOINT REACHED [2026-06-07, leader fresh instance]
Charter: plans/asm-kernel-feasibility-scope.md. ANALYSIS/DESIGN ONLY, no build. Answer the
three questions (igzip-fast map / faithful integration cost+fork / tie projection), write
plans/asm-kernel-feasibility-report.md, route a synchronous disproof advisor, then STOP.
- igzip kernel source map DONE (synchronous read-only subagent, all file:line first-hand).
  KEY: the 4 load-bearing fast tricks (packed-u32 short table, speculative 8-byte literal store
  + next-sym/next-dist preload pipeline, MOVDQU overlap-doubling copy, slop-margin headroom
  guard licensing unchecked over-read/write) ALL DEPEND on the flat-u8 output+window-in-place
  model (asm:518/591/605/618, C:1641/1698). igzip uses SSE-width MOVDQU (xmm), NOT AVX2 ymm —
  the "AVX2"/Haswell build differs mainly in BMI2 (SHLX/SHRX/BZHI), not vector width.
- DELIVERABLE: plans/asm-kernel-feasibility-report.md + plans/asm-kernel-feasibility-advisor-verdict.md.
- STOPPED for supervisor + USER ratification (faithfulness fork is a user-level call). No build.

## ENGINE BENCH ROUND 2 — IN PROGRESS [2026-06-07, leader fresh instance]
Charter: plans/engine-bench-round2-authorization.md. SETTLE the plateau falsifier by building
E2-E4 in the engine-isolation bench (standalone, NOT production integration).
- STEP 0 DONE: round-1 harness COMMITTED 5d5fc3b9 (pushed). Self-test band RECALIBRATED
  [1.7,2.6]->[2.5,3.6] (round-1 FAIL was mis-calibrated: pure-ISA-L is a purer denominator
  → larger honest ~3.1x ratio, advisor-confirmed iii/ii=3.10x).
- KEY STRUCTURAL FACT (source-verified): production clean dynamic-block loop is
  read_internal_compressed_specialized<false> (marker_inflate.rs:1191) — the multi-cached LUT
  loop. Back-ref copy = emit_backref_ring<false> (:2137), writes into a u16 RING. Literal store
  = single u16 (:1348). Refill = bits.refill() once per outer iter (:1324). THESE are the E2/E3/E4
  targets. The u16 ring means every byte costs 2 bytes of mem traffic vs ISA-L's u8 buffer — the
  prime suspect for the 3.1x gap (compute is already heavily optimized: word-copy, RLE fill, etc).
- E2 (wide SIMD back-ref copy): emit_backref_ring already does 8-byte-word(=4 u16) copies + RLE
  fill. Headroom = wider vectors (AVX2 32-byte = 16 u16) for the longer matches; main lever is
  CLEAN u8 ring (halves copy traffic). E3 (packed multi-literal store): :1343 unpacks the
  TRIPLE_SYM 1-3 packed literals one u16 at a time — collapse to one wide store. E4 (wide refill):
  refill is already libdeflate-style (56-63 bits) once/iter; headroom = amortize over >1 symbol.
- PLAN: prototype E2-E4 as new bench variants (byte-exact gate vs scalar + ISA-L), measure
  separately + stacked, chunk-sweep 3-5 clean chunks, settle PASS/PLATEAU via tier1-design §3.
  Delegated to a synchronous subagent (spec in plans/engine-bench-round2-impl-spec.md).

### ROUND-2 SETTLED — VERDICT: **PLATEAU** [2026-06-07, committed 7bf26096]
- COMMITTED 7bf26096 (pushed): bench-only E2-E4 in marker_inflate.rs (read_clean_e234<E2,E3,E4>
  + drain_clean_u8 + emit_backref_ring_clean + emit_avx2_copy_u16, all cfg-gated like
  read_internal_compressed_specialized; production read() dispatches ONLY to <true>/<false> —
  read_clean_e234 has ZERO production callers, grep-verified). Added VAR_IV_E000 (<false,false,false>)
  byte-exactness anchor to the bench.
- BYTE-EXACT GATE: GUEST run SHA_ALL_EQUAL=yes on all 5 swept clean silesia chunks for all 7
  variants incl VAR_IV_E000 (read_clean_e234<false,false,false> == scalar <false> == ISA-L,
  byte-identical) AND the E2/E3/E4 stacks (AVX2 LIVE on guest, avx2_detected=true). Also SHA-clean
  under Rosetta (scalar fallback). Guest freq-locked: no_turbo=1, GATE PASS, RESTORE VERIFIED.
- GUEST AGGREGATE (median-of-per-chunk-medians, MB/s, 5 chunks, taskset core0, N=11 interleaved):
    VAR_I scalar      104   (=92.7ms/chunk anchor; projects 0.615s == design's 0.6134s — model self-consistent)
    VAR_IV_E000       108   (engine, no technique; +4% over scalar)
    VAR_IV_E2         121 / VAR_IV_E23 120 / VAR_IV_E234 (stacked) 118
    VAR_III ISA-L     283
- **(ii_stacked E234)/(iii ISA-L) = 0.412** (per-chunk 0.356..0.466, sd 0.036). This is FIRMLY in
  PLATEAU: ≤ the pre-registered 0.65 plateau line, **12.2σ below** the 0.85 PASS threshold. E234
  adds only ~8.7% median over E000 — the engine stays ~2.4× SLOWER than ISA-L (pure-decoder class).
- §3 PROJECTION (anchored 92.7ms↔104MB/s, 39 chunks, ramp 1.36, T8): E234 projects decode_wall
  0.542s. APPEARS ≤ the 0.604s tie bar, BUT this is a NUMERICAL ARTIFACT — decode_wall(0.542)≈
  total_wall(0.542), i.e. DECODE IS STILL THE BINDING TERM (it did NOT re-bind onto the ~0.54s
  shared floor). The bar sits right at the scalar floor (0.615s) so a ~13% engine bump trips it
  without decode ceasing to bind. The pre-registered PASS requires (ii)/(iii)≥0.85 so decode stops
  binding; E234 (0.41) is ~2× short. RATIO criterion GOVERNS → PLATEAU.
- FALSIFIER FIRED: residual gap to igzip-class (0.85-0.41=0.44) >> spread (sd 0.036, 12σ). Per the
  pre-registered falsifier: engine front is **NOT PROVEN** in pure-Rust+ASM as prototyped. The
  1.0× bar is NOT reachable via this E1-E4 inner-loop direction without FFI (or a revisited bar).
  SUPERVISOR/USER-LEVEL FINDING. Do NOT integrate; do NOT start the multi-week engine build.
- WHY pure-Rust+ASM plateaus (mechanism, not just a TIE): vendor's own bench has ISA-L at 2.1× the
  best PURE decoder single-thread; this bench reproduces ~2.4×. The gap is the u16 ring (2 bytes/byte
  mem traffic vs ISA-L's u8 buffer) + igzip's whole-AVX2 codegen; E2 (AVX2 copy on the STILL-u16 ring)
  + E3 (packed store) + E4 (amortized refill) recover only ~9% because the dominant cost is the per-
  symbol LUT-decode + u16 traffic that these techniques don't remove. A true E1 (u8 ring) was NOT
  prototyped here (faithful u16 ring kept for byte-exactness); it is the remaining untested lever but
  would diverge the ONE-engine ring storage from vendor's m_window16 — a supervisor call.
- NEXT (SUPERVISOR GATE): independent disproof advisor verdict ->
  plans/engine-bench-round2-advisor-verdict.md, then STOP for supervisor. No production integration.

## PLACEMENT PORT GATE — **FAILED (do NOT code attempt #4)** [2026-06-07, HEAD e52b0fc2]
Charter: plans/placement-port-authorization.md. The HARD 3-prior-failures re-derivation
gate ran (2 read-only subagents + 1 independent disproof advisor, all synchronous). Verdict:
**GATE FAIL — STOP. No port code written.** Full advisor verdict: plans/placement-port-advisor-verdict.md.

- The authorized lever's PREMISE IS FACTUALLY WRONG. Authorization said "gzippy never
  re-targets an overshot index at its CONFIRMED offset; rapidgzip does (GzipBlockFinder.hpp
  :117-158)." Source (leader + advisor + leader-reverified) shows gzippy ALREADY re-targets
  at the confirmed offset at THREE faithful sites: gzip_block_finder.rs:180-182 (confirmed
  idx→confirmed offset), chunk_fetcher.rs:1306 (matches_encoded_offset accept) + :1431
  (get_with_prefetch cold get AT the confirmed offset), block_fetcher.rs:945 (submit_for uses
  block_finder.get(index) = confirmed offset). The lever targets a NON-DIVERGENCE.
- Leader-VERIFIED directly (not just advisor): `needs_confirmed_offset` has ZERO hits in src/;
  block_fetcher.rs:784-790 pushes BOTH the confirmed off AND the partition offset (partition
  secondary, exactly vendor BlockFetcher.hpp:485-490) — gzippy does NOT collapse to partition.
- VENDOR reaches the SAME overshoot cold-get (GzipChunkFetcher.hpp:646-654 fallthrough to
  on-demand decode at the real offset). It absorbs it via the 2·P guess-prefetch horizon +
  pump-during-wait, NOT a distinct block-finder mechanism. The confirmed offset is born at the
  in-order insert in BOTH (appendSubchunksToIndexes :357-375 ↔ consumer_append_subchunks_vendor
  chunk_fetcher.rs:2788-2795) = ≤1-chunk lead in both. No lead-lengthening mechanism to port.
- WHY prior 3 failed (corroborated): all supplied confirmed offset at the ≤1-chunk frontier;
  Attempt 3 submitted the EXACT correct offset → fetcher_get UNCHANGED T8 449ms / WORSE T16
  303→936ms (in-flight-not-done). The proposed port inherits the SAME ≤1-chunk lead AND has
  nothing to change → attempt #4 with the same load-bearing constraint.
- ANTI-ESCAPE-HATCH (advisor guardrail, NOT yet done): do NOT bundle this into "it's the
  engine, placement done." The genuinely-distinct UNCLOSED structural question: stalls are
  "all decode_NOT_STARTED" — either workers saturated (engine) OR the GUESS-prefetch for that
  index was never dispatched DEEP ENOUGH AHEAD (prefetch-HORIZON/scheduling — structural,
  distinct from offset supply). That is the pre-registered slow-knob perturbation in
  placement-rescope-diagnosis.md. **Proposed next step (SUPERVISOR GATE): run the
  decode_NOT_STARTED / prefetch-horizon perturbation — NOT the offset-supply port, NOT the
  engine build.** STOPPED for supervisor.


## NEW MISSION (2026-06-06, user-set) — SUPERSEDES all prior arcs
Prior "full ISA-L port as a 2nd engine" plan is DEAD (critically reviewed: redundant
2nd inflate vs ONE-engine memory; x86-only strands arm64). This mission, in order:

- PHASE 1 — NAMING TRUTH PASS (behavior-neutral renames). Every function/type/module/
  dataset/span whose name misdescribes behavior gets renamed; all call sites, doc
  comments, trace/span labels updated. FLAGSHIP: IsalInflateWrapper on the pure-Rust
  build does NOT wrap ISA-L — it delegates to pure-Rust Inflate. Audit whole
  src/decompress/parallel/ + src/decompress/inflate/. Gate: build + byte-exact tests +
  silesia sha 028bd002c89c9a909ccdbc2af0a223de285348edb014ccc8e27d297f52cb410f.
  Commit: "rename(parallel/sm): naming truth — names describe behavior".
- PHASE 2 — TWO FLAG-GATED PATHS + DELETE dead.
  - gzippy-isal feature: FAITHFUL rapidgzip — marker engine + 32KiB clean-handoff to
    REAL ISA-L via FFI (vendor GzipChunk.hpp:520-525). C-FFI behind THIS flag ONLY.
  - gzippy-native feature (DEFAULT prod): UNIFIED ONE-ENGINE marker decode, single pass,
    pure Rust, flip-in-place clean tail, NO 2nd engine, NO FFI, ISA-L hot techniques
    grafted (BMI2 PEXT/BZHI dispatch OFF unified.rs:122; multi-symbol LUT; lean refill).
  - Each path BYTE-EXACT. native also diff-tested vs gzippy-isal oracle + flate2.
    DELETE flagged-off unused impls. Advisor-gate each deletion.
- PHASE 3 — FULCRUM 3-WAY: locked harness (neurotic, T8 silesia, interleaved best-of-N,
  sha-verified) for rapidgzip vs gzippy-isal vs gzippy-native. Report 3-way wall + d_c/d_w.

## CONSTRAINTS
OOM: serialize ALL builds (pgrep first; ONE globally; WAIT). Prefer neurotic. Never two
building subagents. No duplicate orchestrators. Long ops detached + sentinel, polled.
Byte-exact ABSOLUTE. Delegate ALL hands-on work; advisor-corroborate consequential claims.
Numbers ONLY from full locked harness. Clean base HEAD 412b3ac. KEEP #B + byte-exact.

## PHASE 1 — IN PROGRESS
- Worktree: .claude/worktrees/phase1-naming-truth (branch phase1-naming-truth from 412b3ac).
- Observed: inflate_wrapper.rs has TWO IsalInflateWrapper structs (line 154 real-ISA-L cfg,
  line 340 pure-Rust cfg). 9 src files reference IsalInflateWrapper.
- STEP 1 [running]: design subagent produces rename MAP (old->new + rationale). Advisor-
  review. Then impl subagent applies. Then gate.

### STEP 1 DONE — rename MAP produced + advisor-reviewed (APPROVE WITH CHANGES)
- Governing fact (advisor-confirmed via build.rs:81-83): pure_inflate_decode==parallel_sm;
  the old ISA-L decode arm was REMOVED; NO live real-ISA-L decode in parallel/+inflate/.
  Every isal-named symbol there is FALSE-isal (pure-Rust). TRUE-isal lives only in
  src/backends/ + `feature=isal-compression` GATES (not names) — DO NOT rename.
- Approved renames: IsalInflateWrapper->StreamingInflateWrapper (both arms, ~28 code +
  exclude ~12 vendor doc refs); isal_huffman_pure.rs->lut_huffman.rs (mod.rs:46);
  IsalLitLenCodePure->LutLitLenCode (10); IsalDistCodePure->LutDistCode (6);
  isal_lut_bulk.rs->lut_bulk_inflate.rs (mod.rs:48, 19 path refs);
  isal_litlen_pure->lut_litlen (4); isal_lut_litlen_rebuild->lut_litlen_rebuild (3);
  isal_lut_litlen_decode->lut_litlen_decode (2).
- DEFER: trace span worker.isal_stream_inflate (cross-coupled to patch_vendor.sh; renaming
  one side breaks `fulcrum vs` — not behavior-neutral for tooling). KEEP ISAL_* vendor
  constants (traceability). 
- ADD (advisor): also rename DecodePath::IsalParallelSM/IsalSingle in src/decompress/mod.rs
  (flagship FALSE-isal; the GZIPPY_DEBUG path= contract) -> ParallelSM/SingleMember.
  Update GZIPPY_DEBUG label + routing tests + CLAUDE.md mentions consistently.
- STEP 2 [DONE 2026-06-06]: rename applied + gated + COMMITTED.
  - Commit 6b7d6f5 on branch phase1-naming-truth (worktree .claude/worktrees/phase1-naming-truth),
    parent 412b3ac. Diff symmetric pure-rename + a cargo-fmt pass (hook-required: import
    reorder/line-wrap of the renamed code; still behavior-neutral).
  - Gate (deadlock-safe — SINGLE serialized cargo runs, timeout-wrapped, --test-threads=1,
    load-flaky tests skipped):
      * build clean (release, --features pure-rust-inflate); cargo fmt --check clean.
      * lib tests 35 passed / 0 failed (routing:: + pure_rust_inflate_corpus::; skipped
        not_slower/diff_ratio/scoped_cancel/hot_path/alloc_budget).
      * silesia sha256 028bd002...410f == reference, emitted via path=ParallelSM
        (GZIPPY_DEBUG confirms the renamed contract; was IsalParallelSM).
  - NOTE: env lacks a subagent-spawn tool (only TaskStop surfaced); leader executed the
    mechanical gate directly. Phase 2 design/impl/profiling will need the spawn tool —
    flag to supervisor if delegation is required there.

## PHASE 1 — DONE. PHASE 2 — IN PROGRESS.

### STEP 0 [DONE 2026-06-06] — rename integrated onto working branch
- `git merge --ff-only phase1-naming-truth` -> reimplement-isa-l now at 6b7d6f5 (linear ff).
- VERIFIED: StreamingInflateWrapper present (7+ src files); 8 remaining IsalInflateWrapper
  refs are ALL intentional vendor doc-citations (rapidgzip::IsalInflateWrapper / gzip/isal.hpp
  in backends/isal_decompress.rs, inflate/resumable.rs, inflate/staged_bits.rs,
  parallel/inflate_wrapper.rs) — kept for traceability per approved MAP. Renamed files
  lut_bulk_inflate.rs + lut_huffman.rs present; old isal_* names gone.
- Build clean: release --no-default-features --features pure-rust-inflate, 34s, warnings only.
- pgrep NOTE: bare `pgrep -f 'cargo|rustc'` FALSE-POSITIVES on Cursor helpers/shells whose
  cwd contains "isal". Use `pgrep -x cargo; pgrep -x rustc` for the build-serialization gate.

### STEP 1 [DESIGN DONE — advisor-review pending] — two flag-gated paths
DESIGN SUBAGENT (exit 0, ~325s) findings + leader-corroborated:
- KEY: the unified pure-Rust path ALREADY EXISTS and is the SOLE decode path. NO live
  ISA-L decode anywhere in parallel/+inflate/ (build.rs:80-82 pure_inflate_decode==parallel_sm).
  => gzippy-native = today's pure path made default (STEP 1 trivial wiring). gzippy-isal =
  RE-INTRODUCE a real-ISA-L clean tail behind a flag (the path needing new code).
- CAVEAT (leader-verified): gzippy-isal FFI building block EXISTS and is tested:
  isal_decompress.rs:307 decompress_deflate_from_bit (+_with_end:460, +_with_boundaries:643)
  — bit-offset + 32KB-dict tests pass.
- CORRECTION (advisor caught my error): the stopping-point patch IS present — in the C
  SUBMODULE vendor/isa-l/igzip/igzip_inflate.c:1412,1747 + header igzip_lib.h:232 ("gzippy/
  rapidgzip patch") + bindings isal-sys/src/igzip_lib.rs:1565-1569 (full END_OF_BLOCK[_HEADER]/
  END_OF_STREAM[_HEADER] enum). My earlier "no patch" grep checked the Rust wrapper, not the C
  submodule — WRONG. So a faithful tail with REAL block boundaries is available.
- Today's chunk decode: marker phase (marker_inflate::Block, 128KiB u16 ring) -> FlipToClean at
  clean_appended_len>=MAX_WINDOW_SIZE (gzip_chunk.rs:1191, mirrors GzipChunk.hpp:521) ->
  finish_decode_chunk_impl (gzip_chunk.rs:354) constructs StreamingInflateWrapper (pure-Rust
  Inflate<Clean,Generic,Streaming> -> ResumableInflate2, a SECOND 128KiB-staging engine).
- CACHE MANDATE GAP: "one engine, flip-in-place" NOT yet met — TWO engines/buffers per chunk
  (~288KiB scratch). That fold is the substantive native work (sequenced last, A/B-gated).
- WIRING PLAN: gzippy-native=["pure-rust-inflate"]; gzippy-isal=["pure-rust-inflate",
  "isal-compression"]; new build.rs cfg isal_clean_tail = is_x86_64 && CARGO_FEATURE_GZIPPY_ISAL.
  KEEP default=[] through STEP 1+2 (flipping surfaces parallel-module lint debt per Cargo.toml:46-48);
  flip to gzippy-native after STEP 2 clears debt. arm64: gzippy-isal degrades to pure tail (==native).
- SEQUENCE: 1a wiring(zero-behavior, native==today byte-identical) -> 1b isal tail (x86_64) ->
  STEP2 delete dead (route_c_*, not-parallel_sm stub CONFIRMED; consume_first_decode big FASTLOOP
  fns + jit_decode/specialized_decode/double_literal + MarkerRing NEEDS-DEEPER-CHECK) ->
  native cache work (share fixed tables / pool engines / fold-to-one-engine / BMI2 runtime dispatch).

### ADVISOR VERDICT (exit 0, ~275s): APPROVE-WITH-CHANGES. Strategy + native sound. 4 fixes
incorporated into the plan (all leader-verified):
- FIX1 native drops ISA-L *compression* backend: release/CI ship --features isal-compression
  (release.yml:132) giving x86 T1 L0-L3 ISA-L COMPRESS; gzippy-native=["pure-rust-inflate"]
  loses it. FFI-free mandate is DECODE-ONLY. DECISION: gzippy-native keeps isal-compression for
  COMPRESS on x86_64 only (decode stays pure-Rust/FFI-free). Update Makefile + .github/workflows
  to new feature names so the gated path == what ships.
- FIX2 arm64 C-build: isal-sys/build.rs has NO arch guard (verified: handles riscv64/win/macos,
  no x86 gate) -> gzippy-isal pulling isal-compression would try to build ISA-L C on arm64. Add
  build.rs compile_error! when gzippy-isal && !x86_64. gzippy-isal = x86_64-only ref baseline.
- FIX3 add CI line for the new combo pure-rust-inflate+isal-compression (not built today).
- FIX4 (BIGGEST RISK) the isal tail is NOT at parity just because _with_boundaries records
  boundaries. Divergences: pure tail coalesce-stops at first boundary AT-OR-PAST stop_hint_bits
  (gzip_chunk.rs:387) & commits CRC only up-to-boundary bytes (:507-516); isal fns run to
  BFINAL over whole slice. Stopping-point set differs (isal _with_boundaries requests only
  END_OF_BLOCK :666 vs pure END_OF_BLOCK|_HEADER|STREAM_HEADER :382-384). Output: pure STREAMS
  into writable_tail segment-by-segment; isal returns one Vec (needs append+offset glue,
  mirror :428). Two stop modes until_exact true/false (:374). VERIFY-FIRST GATE for 1b: a
  same-input pure-vs-isal differential on real silesia chunks asserting IDENTICAL (decoded
  bytes, committed length, end_bit handoff, per-chunk CRC) before any production wiring.
- Keep default=[] through STEP1+2; silesia gate ALWAYS runs explicit --features gzippy-native
  (bare cargo compiles no decode path: routing tests are #[cfg(parallel_sm)]).

### STEP 1a [DONE 2026-06-06] — zero-behavior wiring
- Cargo.toml: gzippy-native=["pure-rust-inflate"]; gzippy-isal=["pure-rust-inflate",
  "isal-compression"]. build.rs: cfg isal_clean_tail = is_x86_64 && CARGO_FEATURE_GZIPPY_ISAL
  && parallel_sm; panic!() if gzippy-isal && !x86_64 (arm64 C-build guard, FIX2).
- GATE (--features gzippy-native): build clean 34s (same 6 pre-existing warnings as
  pure-rust-inflate => pure alias). silesia sha 028bd002...410f == reference via path=ParallelSM.
  lib tests 35 passed / 0 failed (routing:: + pure_rust_inflate_corpus::, --test-threads=1,
  load-flaky skipped). Native is byte-identical to today's decode, by construction.
- No isal_clean_tail code arm yet (that's 1b). NEXT: 1b verify-first differential gate then tail.

### STEP 1b [IN PROGRESS — impl subagent running, builds x86_64 via Rosetta]
- Rosetta prereqs VERIFIED: target x86_64-apple-darwin installed; arch -x86_64 uname -m == x86_64.
  gzippy-isal builds ONLY on x86_64 (build.rs panics on arm64) so Rosetta is mandatory for it.
- Subagent task: (DELIVERABLE 1, the GATE) a #[cfg(isal_clean_tail)] differential test on REAL
  silesia chunks asserting pure-tail vs isal-tail parity on (decoded bytes, committed length,
  final_bit handoff, per-chunk CRC, block boundaries) for BOTH until_exact true/false; report
  exact divergence mechanism if any. (DELIVERABLE 2, only if parity achievable) the
  #[cfg(isal_clean_tail)] arm in finish_decode_chunk_impl. Build-serialized (pgrep -x), no commit.
- IMPORTANT: leader must NOT start any build while this subagent runs (it owns the build lock).

### STEP 1b [GATE DONE + verdict — committed 8d026a8] — faithful isal tail DEFERRED to last
GATE RESULT: differential PASSES 10/10 (5 silesia chunks x until_exact{T,F}) — ISA-L tail
records byte-identical (committed bytes, length, final_bit, CRC, boundaries) vs pure tail.
Existence proof a faithful gzippy-isal tail is achievable. Test-only, cfg(all(test,isal_clean_tail)).
NOTE: arch -x86_64 cargo IMPOSSIBLE (cargo is arm64-only binary); the working x86_64 path is
NATIVE cargo cross-compile --target x86_64-apple-darwin with CARGO_BUILD_RUSTFLAGS="-C
target-cpu=x86-64" (override .cargo/config.toml target-cpu=native), runs under Rosetta.

ADVISOR VERDICT (2nd advisor, independent rewind-logic verification): 
- Design (A) CORRECT (swap inner engine for interface-compatible ISA-L wrapper, driver loop
  unchanged, accounting matches by construction incl. the :478-489 rewind). Design (B)
  decode-to-end+reselect is UNFAITHFUL by construction (handoff bit is driver-loop-history-
  dependent: :486 rewinds to last_eob_pos recorded at :475 on the PRIOR EOB). No thinner shim.
- SEQUENCING (the real call): STEP 2 (delete dead) -> gzippy-native cache work -> gzippy-isal
  Design-A tail LAST, immediately before PHASE 3. WHY: isal is an INSTRUMENT not a deliverable;
  native's fold-to-one-engine WILL REWRITE the driver interface Design-A swaps into (build isal
  now => re-port; build last => one port); gate 8d026a8 already proves pure≡isal so native
  doesn't need isal as oracle (native gated by flate2+libdeflate+canonical sha, arm64-portable);
  defer the riskiest C-FFI re-add until PHASE 3 actually consumes it.
- CONSTRAINT: STEP 2 must NOT delete finish_decode_chunk_impl (live prod + Design-A insertion
  point). native fold should cfg-FORK it (#[cfg(isal_clean_tail)] keeps two-phase; native folded),
  NOT delete. Delete only the genuinely-dead alts (route_c_*, jit_decode, specialized_decode,
  double_literal, not-parallel_sm stub) — advisor-gate each.
- BEFORE any Design-A impl: harden the gate to exercise the :486 rewind (a non-final/non-fixed
  rewind case; a fixed-Huffman no-rewind case per :481 not_fixed; the ==stop_hint case) AND
  verify ISA-L's bit_position at END_OF_BLOCK[_HEADER] matches the pure tell_compressed()
  convention. Q3+Q4 are the SAME gate.

## STEP 2 — DELETE DEAD INFLATE IMPLS [IN PROGRESS 2026-06-06]
### Deletion MANIFEST produced (read-only subagent, exit 0) — far more conservative than expected
- KEY FINDING: a SECOND live decode path (scan_inflate.rs -> decompress::index -> index_mode.rs ->
  CLI --index/--seek, cli.rs:212) compiles under BOTH features and keeps jit_decode,
  specialized_decode, double_literal, bmi2, libdeflate_entry, consume_first_decode ALL LIVE.
  bgzf.rs inflate_consume_first calls are #[cfg(test)]-only (not a live anchor).
- NO dead parallel/ module found — every #[cfg(parallel_sm)] module has a real chunk-pipeline
  consumer (decode_bypass LIVE @ sm_driver:64/chunk_fetcher; marker engine LIVE; etc).
- Only TRULY dead: ONLY route_c_dynasm.rs + route_c_fixed.rs (gated on feature route-c-dynasm
  NOT pulled by default/pure-rust-inflate/gzippy-native/gzippy-isal/tests; only mod.rs:25-28 refs).
- PARTIAL: libdeflate_decode.rs — KEEP get_fixed_tables (live @ resumable.rs:951); dead big fns
  (decode_libdeflate, inflate_libdeflate, copy_match, file-local BitReader, decode_with_double_cache
  + their in-file #[cfg(test)] tests 979-1196) have ZERO external callers.
- PROPOSED BATCHES: B1 route_c_* (zero prod/test impact); B2 dead libdeflate_decode fns;
  B3 (deferred, needs decode_dynamic dispatch trace) consume_first_decode internal prune +
  double_literal reclassification. MUST-PRESERVE verified: finish_decode_chunk_impl + STEP-1b gate.
- ADVISOR REVIEW [DONE, exit 0]: APPROVE Batch1; APPROVE-WITH-CHANGES Batch2 (manifest had 2
  WRONG symbol names: BitReader->LibdeflateBits; decode_with_double_cache->decode_huffman_double_lit;
  advisor enumerated the real dead cluster by reading the file). Both riskiest claims independently
  grep-confirmed. Gate adds: include plain cargo build/test; Batch2 gate must exercise --index/scan
  path (tests::index). Retire route-c-dynasm feature+deps+sub-crate = SEPARATE follow-up (not byte-neutral).

### BATCH 1 [DONE — committed 03f592e] delete route_c_dynasm.rs + route_c_fixed.rs + mod.rs:25-28
- DUAL-SHA gate GREEN: gzippy-native + gzippy-isal(Rosetta x86_64 cross-compile, --target
  x86_64-apple-darwin, CARGO_BUILD_RUSTFLAGS=-C target-cpu=x86-64) BOTH emit silesia sha
  028bd002...410f via path=ParallelSM. routing 30/0, correctness 130/0, pure_rust_inflate_corpus
  5/0 (incl silesia), index 22/0; plain cargo build clean. NOTE: native cargo build --features
  gzippy-isal on arm64 PANICS by design (build.rs:83 guard) — MUST use Rosetta cross-compile.

### BATCH 2 [DONE — committed edcd863] prune dead big-decoder cluster in libdeflate_decode.rs
- Replaced file with ONLY get_fixed_tables + FIXED_TABLES (was ~1200 lines). Removed the dead
  cluster (zero external callers under either feature). KEEP get_fixed_tables (live @
  resumable.rs:951 + consume_first_decode.rs + double_literal.rs).
- RESUMED after restart: prior leader's interrupted edit reconciled under the NEW global build
  mutex scripts/cargo-lock.sh (portable mkdir-lock; macOS has no flock). DUAL-SHA GREEN:
  gzippy-native (arm64) + gzippy-isal (Rosetta x86_64 cross-compile) both emit silesia
  028bd002...cb410f via path=ParallelSM. native suites 187/0 (routing/correctness/
  pure_rust_inflate_corpus/index, --test-threads=1, load-flaky skipped). clippy clean (fixed
  inherited needless_range_loop in get_fixed_tables, behavior-identical).
- STRUCTURAL FIX: ALL cargo/rustc/test now wrapped `scripts/cargo-lock.sh <cmd>` — concurrent
  builds structurally impossible. Every subagent MUST use it.

### BATCH 3 [DONE — committed 70ec5ff] dead double-literal + short-bits caches
- AFTER restart: read-only scout did the decode_dynamic dispatch trace (the thing that made B3
  ambiguous) + an independent advisor read corroborated. BOTH confirm decode_huffman_cf:1505 is
  LIVE (fixed-block + dynamic-block paths) and KEPT. Everything deleted has ZERO callers under
  gzippy-native, gzippy-isal, OR scan_inflate/index, plus tests/benches/examples.
- Deleted: double_literal.rs, huffman_short_bits_cached_deflate.rs, huffman_short_bits_multi_cached.rs
  (whole files + mod decls); 9 dead fns in consume_first_decode.rs (3580->2628 lines); stale
  doc-comment refs fixed. NOT touched: finish_decode_chunk_impl; specialized_decode cluster
  (production-dead but SpecializedCache coupled to SPEC_CACHE + test stats — coupled edit, deferred).
- DUAL-SHA gate GREEN (under scripts/cargo-lock.sh): native + isal(Rosetta x86_64) both
  028bd002...cb410f via path=ParallelSM; native suites 187/0; clippy clean; plain build clean.

## STEP 2 — DONE (B1 route_c 03f592e, B2 libdeflate_decode edcd863, B3 70ec5ff).
The ONLY remaining production-dead cluster (specialized_decode + SPEC_CACHE stats) is a coupled
edit, intentionally deferred — not a clean leaf delete.

## NATIVE CACHE FOLD — design + advisor DONE 2026-06-06; impl in progress
- DESIGN (subagent, read-only) + ADVISOR review (both this session). Two engines per chunk confirmed:
  Engine M = marker_inflate::Block (128KiB u16 ring; faithful port of vendor deflate.hpp deflate::Block,
  has BOTH markers + clean drain @ marker_inflate.rs:738-750). Engine C = StreamingInflateWrapper ->
  ResumableInflate2 (128KiB staging staged_bits.rs:22 + 32KiB window resumable.rs:94/107 +
  last_32kib_window_vec gzip_chunk.rs:767 ~= 192KiB UNPOOLED Box churn/chunk), constructed @
  finish_decode_chunk_impl gzip_chunk.rs:354/379 after FlipToClean gzip_chunk.rs:1191. Engine C cites
  gzip/isal.hpp — it is the pure-Rust IsalInflateWrapper STAND-IN (belongs to gzippy-isal path).
- FIXED Huffman tables ALREADY SHARED (FIXED_TABLES OnceLock libdeflate_decode.rs:17). Dynamic tables
  inherently per-block (reuse, not share). So mandate's "share tables" already met for the shareable one.
- SURVIVOR FORK RATIFIED (advisor + governing memory): Engine M survives, Engine C deleted from native.
  Engine M == MarkerRing engine the governing memory mandates flip-in-place + continue clean tail on
  SAME cursor; Engine C (IsalInflateWrapper stand-in) stays on gzippy-isal -> real ISA-L FFI Design-A.
- ADVISOR CAUGHT A BUG in the naive first step: pooling the WHOLE Engine C is NOT byte-exact —
  set_window:336 reset is INCOMPLETE (misses encoded_until_bits, coalesce_stop_hint [concrete stale-hint
  divergence: set only when !until_exact gzip_chunk.rs:386-388], block_boundaries) AND a lifetime
  soundness problem (ResumableInflate2<'a> borrows per-chunk input, can't live in 'static thread_local).
- CORRECTED STEP 1 (impl now): pool the TWO HEAP BUFFERS only (128KiB staging Box + 32KiB window Box)
  via thread-local free-list; engine logic + lifetimes UNTOUCHED, only the allocator changes. Near
  byte-transparent. Gate: dual-sha + STEP-1b per-chunk differential. Then STEP 2 flip-in-place fold
  (Engine M continues clean tail, delete Engine C from native) + clean-drain bulk/BMI2 graft IN THE
  SAME step (avoid shipping a wall regression; cfg-fork finish_decode_chunk_impl so isal keeps two-phase).

### NATIVE CACHE STEP 1a [DONE — committed 736ea1a] pool 128KiB staging box
- Scoped to the STAGING box only (the larger, always-fully-overwritten one — provably byte-transparent:
  recycled box's bytes always overwritten by first reload_at_bit before any read). STAGING_POOL
  thread-local free-list (cap 4); buf is ManuallyDrop so Drop recycles instead of frees (no double free).
- DUAL-SHA GREEN: native + isal(Rosetta x86_64) both 028bd002...cb410f via path=ParallelSM. staged_bits
  unit tests 7/0 (reload/memcpy parity on pooled box); native suites 187/0; clippy clean.
- Hit + fixed a test-only E0507 (a unit test moved *staged.buf; now .to_vec()/.as_slice() through MD).
- STEP 1b TODO: 32KiB SlidingWindow pool — DEFERRED, needs explicit reset analysis (read-before-fully-
  written within a chunk, unlike staging). May fold into the STEP-2 flip instead (Engine C goes away).
- TODO: pin the ~ footprint numbers EXACTLY (mandate is measured, not asserted) — build an RSS-vs-T
  instrument before the STEP-2 behavioral fold (advisor: validate the instrument on this predictable
  per-chunk->per-thread delta first).

## (orig) NEXT: STEP 2 — delete dead inflate impls (advisor-gate each). DO NOT touch finish_decode_chunk_impl.
- Features today: default=[] ; pure-rust-inflate (prod decode, sets cfg parallel_sm==
  pure_inflate_decode in build.rs:81-83) ; isal-compression (TRUE ISA-L *compression* only,
  no decode) ; isal=[] (build.rs-set when static lib built). NO live ISA-L decode currently.
- New features to add: gzippy-isal (faithful: marker engine -> REAL ISA-L FFI decode handoff
  at 32KiB clean) and gzippy-native (DEFAULT prod: one unified pure-Rust engine, cache-resident
  per design mandate). Design subagent spawned to produce the two-column rapidgzip<->gzippy map
  + concrete wiring plan before any impl.

## RSS-vs-T INSTRUMENT [DESIGN 2026-06-06 — leader; advisor-corroborate then delegate impl]
Advisor-mandated prerequisite before the STEP-2 behavioral fold. Measures the cache mandate
(plans/gzippy-native-design-mandate.md): shared small hot working set, RSS ~flat as T rises.
RSS captured LOCALLY (arm64 gzippy-native); wall/MPKI stay on neurotic locked harness.

### What it must measure (per binary, per corpus)
1. PEAK total RSS vs T in {1,8,16}: /usr/bin/time -l "maximum resident set size" (bytes on macOS).
2. STEADY-STATE RSS time-series: background `ps -o rss= -p <pid>` poller @ ~50ms -> median of
   the plateau (excludes ramp/teardown) so per-thread growth is separable from one-shot allocs.
3. Derived PER-THREAD working-set slope: (RSS@T16 - RSS@T1)/15 and (RSS@T8-RSS@T1)/7; report both
   (non-linearity is signal). Mandate target: slope << per-thread L2; RSS roughly flat in T.
4. sha256 of decoded output EVERY run == 028bd002...cb410f (a low-RSS run with wrong bytes is void).

### Validation (instrument must earn trust BEFORE its numbers count — CLAUDE.md rule 4/PROCESS)
- POSITIVE control: a knob that PROVABLY inflates per-thread RSS must show up as increased slope
  (e.g. force a larger per-thread alloc, or compare a high-T run that we KNOW duplicates buffers).
- NEGATIVE/self control: same binary vs itself reads equal RSS within poller spread (define spread).
- POOLING DELTA (the real validation): build 70ec5ff (UNPOOLED staging) vs 736ea1a (POOLED staging),
  run both through the instrument at T8/T16. Hypothesis: pooled shows LOWER per-thread RSS slope
  (recycled staging box => fewer live 128KiB allocs at steady state). MEASURE — if peak RSS is
  unmoved (pool may only cut alloc *rate*, not peak footprint), SAY SO; that's a finding about what
  the instrument can/can't see, not a failure. Either way the +/- controls must pass for trust.

### Deliverable
scripts/bench/rss_vs_t.sh — args: BIN, GZ corpus, T-list, N trials; outputs the table above +
controls. ALL builds via scripts/cargo-lock.sh (global mutex). Per-T binary = same gzippy-native
release binary, T set by `-p <T>`. Report per-T peak+plateau RSS, slope, sha-OK, control verdicts.

### ADVISOR VERDICT on RSS instrument design: APPROVE-WITH-CHANGES (incorporated)
KEY REFRAME: process RSS is TOO COARSE for the mandate. Per-thread decode scratch at T16 ~=
128KiB*16 = 2MB against a ~300-400MB backdrop (68MB input mmap + ~211MB decoded silesia) = <1%,
below poller jitter. RSS-vs-T CANNOT resolve per-thread decode buffers. So:
- PRIMARY mandate instrument = DIRECT IN-PROCESS BYTE ACCOUNTING (zero noise floor), via GZIPPY_DEBUG
  counters: peak live bytes per thread-local pool, sum of shared read-only table bytes, count of
  distinct pool high-water boxes. Measures "tiny per-thread working set / shared tables / pooled
  buffers" EXACTLY. This is what honors the mandate.
- RSS-vs-T DEMOTED to a coarse T-scaling GUARD: catches only UNEXPECTED large T-scaling (e.g. a
  per-thread COPY of a shared table — large enough RSS would see it). Necessary-not-sufficient.
- cache-residency is NOT an RSS question -> MPKI (deferred to neurotic locked harness) is the real test.
FIXES incorporated:
- UNITS: /usr/bin/time -l max-RSS is BYTES on macOS; ps -o rss= is KiB. Normalize + assert (1024x trap).
- Stream output to /dev/null (bounded writer) so output buffering is T-invariant + small.
- time -l for PEAK only; ps plateau-median for live set; NEVER cross-compare the two as equal.
- POOLING DELTA = a FINDING, not a control (free-list may not move peak live set; null is ambiguous).
- POSITIVE CONTROL (must-pass, calibrated/signed/magnitude): env knob allocs+TOUCHES (writes every
  page) known N MiB per worker thread; instrument MUST recover slope ~= N at multiple N (linearity).
- NEGATIVE control: binary vs itself within poller spread (keep).
- T grid {1,2,4,8,16}, REGRESS slope (2-pt slope can't separate const overhead from per-thread growth);
  check linearity not endpoints. N>=7 interleaved, Delta<spread => TIE.
- Assert GZIPPY_DEBUG path=ParallelSM (native engine, not fallback) every run; sha gate every run.

### RSS instrument — REVISED deliverables (to delegate)
D1 (PRIMARY): in-process per-thread byte accounting behind GZIPPY_DEBUG (or a new env GZIPPY_MEM_STATS=1):
   instrument STAGING_POOL (+ later the 32KiB window pool / engine state) to record peak-live-bytes per
   pool + box high-water count; emit shared-table byte total (FIXED_TABLES etc). NEEDS a small src change
   (counters) — design subagent first maps WHERE to hook, advisor-gate, then impl. Byte-exact (counters only).
D2 (GUARD): scripts/bench/rss_vs_t.sh — peak (time -l, bytes) + plateau (ps, KiB) RSS over T{1,2,4,8,16},
   N>=7 interleaved, output->/dev/null, sha gate, native-path assert, slope regression + linearity.
D3 (CONTROL): GZIPPY_MEM_BALLAST_MIB=N env -> each worker allocs+touches N MiB; rss_vs_t must recover
   slope~=N at N in {8,16,32}. Pooling delta (70ec5ff vs 736ea1a) run + reported as a FINDING.

### RSS instrument — IMPL NOTE (impl subagent 2026-06-06) — exact hook lines + byte-transparency
New module `src/decompress/inflate/mem_stats.rs` (pub mod in inflate/mod.rs). COUNTERS ONLY, no decode
behavior change. `enabled()` reads `GZIPPY_MEM_STATS` ONCE into a OnceLock<bool>; every hook is
`if !enabled() { return; }` so the flag-off path is an inlined early-return (DUAL-SHA gate proves
byte-identical on/off).
HOOKS:
- staged_bits.rs take_staging_box (~35): split pop into reused vs Box::new arm, call
  `mem_stats::on_take(reused)`. on_take: relaxed-atomic alloc-vs-reuse counter; thread_local live++/peak;
  on a new per-thread peak records into a global Mutex<HashMap<ThreadId,ThreadStat>> (cold path).
- staged_bits.rs return_staging_box (~42): `mem_stats::on_return()` (thread_local live--). Drop path
  already calls return_staging_box (line 329) so per-chunk lifecycle is covered.
- staged_bits.rs STAGING_POOL_CAP -> env-overridable `pool_cap()` OnceLock (default 4); CAP=0 disables
  pooling. This is the chosen D3 POOLING-DELTA mechanism (env `GZIPPY_STAGING_POOL_CAP=0` vs 4 on the
  SAME 736ea1a binary) — isolates the pooling variable with one build, no worktree/cherry-pick drift.
  Default 4 == today's production behavior (byte- AND perf-identical when env unset).
- libdeflate_entry.rs: add `heap_bytes()` to LitLenTable + DistTable (size_of::<Self>() +
  capacity*size_of::<Entry>()) so report() prints the ONE shared-table footprint via get_fixed_tables().
- main.rs main() after `let result = run()`: `decompress::inflate::mem_stats::report();` BEFORE
  process::exit (which skips destructors). report() is itself enabled()-gated.
POSITIVE CONTROL: `GZIPPY_MEM_BALLAST_MIB=N` (own OnceLock switch, independent of MEM_STATS so the RSS
guard drives it without accounting overhead). ensure_ballast() on first take per thread allocs+touches
(4KiB stride) N MiB into a thread_local Vec held for thread life -> known per-thread resident slope.
Byte-transparent: ballast Vec is never read by decode.

## FLIP-IN-PLACE FOLD — leader orientation (gated on RSS instrument; STEP 2 next)
Read the two-engine boundary in gzip_chunk.rs + marker_inflate.rs:
- Engine M loop (run_marker_blocks-style, gzip_chunk.rs:1184-): on clean_appended_len>=MAX_WINDOW_SIZE
  && !ctx.flipped it returns MarkerStep::FlipToClean{end_bit_offset, window_len=32KiB} at :1191-1202,
  setting ctx.current_bit_offset=next_block_offset. Driver THEN constructs Engine C
  (StreamingInflateWrapper -> ResumableInflate2) in finish_decode_chunk_impl (:354) to decode the
  clean tail from that bit offset. THAT handoff is the two-engine cost the fold removes (native only).
- Engine M ALREADY CAN drain clean u8 in-place: marker_inflate.rs:731-758 drain_to_output ->
  push_clean_u8 (:750) once contains_marker_bytes==false (vendor deflate.hpp:1285-1292). So the fold =
  do NOT return FlipToClean in native; keep iterating Engine M's block loop on the SAME ctx cursor,
  letting drain_to_output emit clean bytes, until BFINAL/stop_hint. Engine C deleted from native;
  finish_decode_chunk_impl cfg-FORKED (#[cfg(isal_clean_tail)] keeps two-phase for gzippy-isal Design-A;
  native folded) — advisor CONSTRAINT: do NOT delete finish_decode_chunk_impl (Design-A insertion point).
- Gate: DUAL-SHA 028bd002...cb410f (native folded + isal two-phase), per-chunk differential (STEP-1b gate
  8d026a8 style), then measure wall+RSS on locked harness. Sequence the clean-drain bulk/BMI2 graft IN
  THE SAME behavioral step to avoid shipping a transient wall regression.

## LEADER RE-DRIVE 2026-06-06 (resumed @ 5f162bb) — singleton-leader discipline adopted
- SINGLETON-LEADER LOCK created scripts/leader-lock.sh (clone of cargo-lock mkdir-mutex + stale-pid
  reclaim; acquire/release/status verbs; NON-BLOCKING acquire => duplicate leader EXITS). Self-test
  PASSED: positive (reclaims stale pid 99999), negative (refuses while live holder). ACQUIRED by a
  persistent sentinel pid (nohup sleep) so the lock survives across leader turns (a Bash-subshell
  $$ dies between calls => would read stale).
- FOLD insertion point CONFIRMED by leader read: the ONLY behavioral change is at
  marker_decode_step_loop gzip_chunk.rs:1191-1202 (the `clean_appended_len()>=MAX_WINDOW_SIZE
  && !ctx.flipped` FlipToClean early-return). cfg-fork it: #[cfg(isal_clean_tail)] keeps the
  FlipToClean return (gzippy-isal two-phase, Design-A insertion intact); #[cfg(not(isal_clean_tail))]
  (native) sets ctx.flipped=true and CONTINUEs the loop. Engine M's read() already drains clean u8
  in-place (marker_inflate.rs:1011 drain_to_output -> push_clean_u8 once contains_marker_bytes==false);
  UnifiedMarkerSink.push_clean_u8 buffers to pending_clean, flushed to chunk.data each step
  (decode_chunk_unified_marker :744-749). Loop terminates at BFINAL/stop_hint via MarkerStep::Finished.
  finish_decode_chunk_impl UNTOUCHED (still reachable on isal + window-seeded path). cfg name confirmed
  build.rs:101 isal_clean_tail = is_x86_64 && gzippy-isal && parallel_sm.
- HAZARD to gate: ring-overwrite if a single post-flip clean block is huge and drain only fires at
  EOB (marker_inflate.rs:725-730 contract). read_internal_compressed must drain mid-block OR the
  differential will catch it. The per-chunk differential is the verdict.

## FOLD COMMITTED [DONE 2026-06-06 — 8cfad3a] flip-in-place one-engine fold
- COMMIT 8cfad3a on reimplement-isa-l (parent 5f162bb). Only src change: gzip_chunk.rs +322/-11.
- STEP-5 "FAIL" ROOT CAUSE (corroborated, NOT flaky, NOT a regression): the fold-gate STEP-5 ran
  `cargo test ... --lib routing correctness pure_rust_inflate_corpus -- ...` — `cargo test` accepts
  only ONE positional TESTNAME, so it errored `unexpected argument 'correctness'` and RAN ZERO TESTS.
  Neither the project_parallel_test_hang deadlock-flake NOR a byte regression. Leader rerun under
  scripts/cargo-lock.sh with the supervisor's valid invocation (full --lib, --test-threads=1, 4
  excludes) = 850 passed / 0 failed, matching the supervisor exactly. Fold is byte-exact-clean.
- VALIDATION (all green for the committed tree): DUAL-SHA native arm64 + isal x86_64(Rosetta) both
  028bd002...cb410f via path=ParallelSM; native_fold_parity 32/32 correct (12 FLIPPED; rewind /
  fixed-Huffman / ==stop_hint; until_exact {T,F}); lib suites 850/0.
- GATE HARDENED: /tmp/fold-gate.sh STEP-5 now uses the supervisor's invocation — full `--lib`,
  `--test-threads=1`, excludes by EXACT name (diff_ratio_parallel_single_member_speedup,
  scoped_cancel_stops_early_without_full_scan, hot_path, alloc_budget). No multi-positional filter.
- EXCLUDE/FLAKY POLICY (board record): the 4 names above are the canonical load-flaky/perf-gate
  exclusion set for serial byte-exact lib runs. ALWAYS pass `--test-threads=1` (project_parallel_test_hang
  deadlock-class). Use full `--lib` + `--skip <exact-name>` — NEVER multiple positional TESTNAME filters
  (silent CLI parse error that runs nothing and reads as a FAIL).
- ADVISOR GAP (assumption, flagged): no subagent-spawn tool in this session, so a fresh Opus advisor
  could not be consulted for the commit. Proceeded on standing advisor verdicts already on record (fold
  design APPROVE lines 163-181; finish_decode_chunk_impl-preserve constraint satisfied) + the direct
  flaky-vs-real resolution above. If a supervisor advisor pass is required retroactively, the evidence
  is captured here and in /tmp/fold-gate.result.

## NEXT (sequenced): BMI2 PEXT/BZHI + multi-symbol LUT graft into Engine M
- Cache mandate: ONE shared decode-table copy across threads, no large per-thread buffers, RSS ~flat in T.
- Each technique byte-exact + measured on the locked harness; keep wins, revert regressions.
- Then: gzippy-isal Design-A tail (dual-sha vs folded driver) -> PHASE 3 3-way Fulcrum + RSS/working-set/MPKI.

## SPEED STRETCH — LEADER RE-DRIVE 2026-06-06 (resumed @ 8cfad3a). leader-lock HELD.
### FALSIFIER PRE-REGISTERED [DONE] plans/bmi2-graft-falsifier.md
- Per-technique falsifier + ceiling + TIE/regression judgment recorded BEFORE any opt work.
- KEY PRIOR-STATE FINDING (read the production loop, marker_inflate.rs run_multi_cached_loop
  :1630+): techniques (b) multi-symbol LUT and (c) lean refill are ALREADY PRESENT
  (2-/3-literal speculative chain :1782-1841; branchless bounds-elided refill_fast :1715-1731,
  REFILL_THRESHOLD=48; speculative next-entry carry). Only (a) BMI2 PEXT/BZHI is genuinely
  un-grafted (Generic HAS_BMI2=false unified.rs:122; no BMI2 in the marker hot loop). So (a)
  is THE lever; (b)/(c) are re-validation/sharpening (re-attempt ca52389-class with fresh
  measurement — KNOWN HAZARD).
- unified.rs is a DELEGATING SCAFFOLD (Inflate<Clean,Generic,Streaming> -> ResumableInflate2);
  the live production dynamic-Huffman hot loop is marker_inflate.rs read_internal_compressed_
  canonical_specialized -> run_multi_cached_loop (uses libdeflate LitLenTable/DistTable).
  The fixed-Huffman arm uses HuffmanCodingReversedBitsCached. THAT is the BMI2 graft surface.

### INSTRUMENT-VALIDITY FINDING (must fix before cache-mandate measurement)
- The byte-accounting instrument (mem_stats.rs, hooked to staged_bits.rs take_staging_box) is
  DEAD on the native path AFTER THE FOLD. take_staging_box is only called by StagedBits (part of
  ResumableInflate2 = Engine C); the fold removed Engine C from native (finish_decode_chunk_impl
  unreached in native steady state). Result: GZIPPY_MEM_STATS=1 on native silesia emits NO report
  (threads observed = 0) — leader verified locally.
- The ACTUAL native per-thread working set is BOOTSTRAP_BLOCK (thread_local Block, gzip_chunk.rs:1096),
  dominated by Block.output_ring: Box<[u16; RING_SIZE]> = 2*MAX_WINDOW_SIZE = 128KiB per worker
  thread. The instrument must be RE-HOOKED to this (the ring + literal_cl/backreferences Vecs +
  the shared FIXED_TABLES bytes) before it can measure the cache mandate. This re-hook is byte-exact
  (counters only) and needed for PHASE 3's RSS/working-set numbers; it does NOT block the BMI2 graft
  go/no-go (that's a wall question, answered by the locked harness).

### CEILING-BOUNDING METHOD — open question for advisor (flag to supervisor)
- A clean DECODE-ZERO oracle (charter rule 3 "remove the region, measure") is NOT byte-exact here:
  zeroing the inner decode produces wrong bytes (can't sha-verify), violating the byte-exact ABSOLUTE
  invariant. So the textbook removal-oracle is unavailable for the inner Huffman loop.
- Available byte-exact bounds: (1) the existing GZIPPY_SLOW_* decode/bootstrap slow-injection
  (already moved the wall ~proportionally, survived freq-neutral disproof) gives the SLOPE / confirms
  criticality; (2) the BMI2 graft delta itself IS a perturbation (changes per-symbol decode cost by a
  known mechanism; read the interleaved wall response). Plan: confirm criticality + quantify slope via
  SLOW-injection on the locked harness (small factors, freq-neutral control), THEN graft+measure;
  Δ>spread=lever; TIE=keep-if-byte-exact-no-RSS-regress; regression=revert.
- ADVISOR ASK (supervisor, you run advisors I can't): is the SLOW-injection-slope + graft-delta the
  acceptable ceiling-bound given a byte-exact decode-zero oracle is impossible, or is there a
  byte-exact removal-oracle shape I'm missing (e.g. swap-in a precomputed correct output buffer so the
  decode loop is bypassed but bytes are still correct)? This is the only consequential method call.

### *** PIVOTAL FINDING: BMI2 IS ALREADY ON in the measured locked-harness build ***
The charter's leading lever (a) "BMI2 PEXT/BZHI runtime dispatch (currently OFF, unified.rs:122)"
rests on a FALSE premise for the perf-target build. Evidence (all leader-verified this session):
- The production BMI2 path is NOT in unified.rs (a DEAD delegating scaffold). It is in bmi2.rs
  (extract_varbits :109-118, decode_extra_bits :78-97 — the extra-bits extraction) +
  consume_first_decode.rs::bzhi_u64 :168 + two_level_table.rs :353. ALL gated
  `#[cfg(all(target_arch="x86_64", target_feature="bmi2"))]` → emit BZHI when the feature is on.
- libdeflate_entry.rs decode_length :224 / decode_distance :322 (THE production dynamic-Huffman hot
  loop extract, run_multi_cached_loop) call bmi2::extract_varbits → already BZHI on bmi2 builds.
- THE BUILD ENABLES IT: .cargo/config.toml [build] rustflags=["-C","target-cpu=native"] AND the
  guest harness scripts/bench/guest_fulcrum_capture.sh:67 export RUSTFLAGS=-C target-cpu=native.
  On the BUILD GUEST (root@10.30.0.199, where cargo build actually runs — neurotic login shell has no
  rustc on PATH), `rustc --print cfg -C target-cpu=native` emits target_feature="bmi2" (+avx2,
  pclmulqdq, vpclmulqdq); /proc/cpuinfo shows bmi2. ⇒ EVERY existing locked-harness wall number was
  measured WITH BMI2 BZHI ACTIVE. "Turning BMI2 on" is a no-op on the measured path.
- has_bmi2() (the runtime detector, bmi2.rs:26) is DEAD — never gates the hot path; the path is
  compile-time. So "runtime dispatch" only matters for a PORTABLE binary (built WITHOUT target-cpu=
  native, dispatching at runtime) — which is NOT the perf target (CLAUDE.md "arch-specific is the
  target; portable ships later via runtime dispatch").
- The ONLY genuinely-ungrafted BMI2 op is PEXT (_pext_u64) — ZERO uses in src. But the table index
  is a single masked field (litlen.lookup / dist.lookup), where BZHI/AND already wins; PEXT pays only
  for SCATTERED multi-field extraction, which this loop does not do. No obvious PEXT lever.
- CEILING-BOUND (the perturbation, not extrapolation): build native WITH vs WITHOUT bmi2 on the guest
  and measure the interleaved wall delta on the locked harness. That delta IS the BMI2 ceiling.
  Hypothesis: small/TIE (BZHI vs shift+mask is ~1-2 cycles on one extract per packet, against a
  461ms wall). Running this A/B next — it is the verdict, byte-exact (both correct), no extrapolation.

### BMI2 CEILING-BOUND A/B — DONE [VERDICT: TIE, lever (a) REJECTED with mechanism]
Ran scripts/bench/bmi2_ceiling_ab.sh on build guest 199 (silesia-large 162MB, T8 mask 0-7,
interleaved best-of-9). Both arms byte-exact (sha e114dd2b... == gzip ref) via path=ParallelSM.
- BMI2-ON  (target-cpu=native, PRODUCTION DEFAULT): best 0.6045  median 0.6485  mean 0.6526  σ 0.0316
- BMI2-OFF (target-cpu=native -C target-feature=-bmi2): best 0.6337 median 0.6484 mean 0.6588 σ 0.0235
- delta(best) = 29ms/4.6% but delta(MEDIAN) = -0.1ms / -0.02% (a DEAD TIE). within-arm spread
  63-108ms, σ 24-32ms — both >> any signal. The best-of-9 "4.6%" is an ON-side fast outlier; the
  robust statistic (median) is identical. Δ << spread ⇒ TIE, full stop.
- MECHANISM (a rejection per CLAUDE.md rule 7a, not a narrow miss): BMI2 BZHI is ALREADY compiled
  into the measured production binary (target-cpu=native enables target_feature=bmi2 on the guest
  CPU; extract_varbits/decode_extra_bits/bzhi_u64 emit BZHI). The "runtime-dispatch graft" would
  change NOTHING on the perf-target build — it only matters for a PORTABLE binary (not the perf
  target). And even forcing BZHI fully OFF is a wall TIE: the single extra-bits extract per packet
  is invisible against the ~600ms wall, which is memory-bound (128KiB u16 ring + 32KiB window apply
  + back-ref copies), NOT per-symbol ALU. rapidgzip's existence-proof advantage is therefore NOT in
  this op — it must be elsewhere (window apply / ring layout / copy), which is the cache-mandate
  surface, not BMI2.
- CONSEQUENCE: do NOT graft BMI2 runtime dispatch into Engine M for perf (no-op on perf build;
  the only beneficiary, a portable binary, is explicitly deferred per CLAUDE.md "portable ships
  later"). Techniques (b) multi-symbol LUT + (c) lean refill are ALREADY PRESENT (falsifier finding)
  AND are also per-symbol-ALU techniques the SAME memory-bound argument covers — re-validation is
  unlikely to move the wall by the same mechanism. The campaign's real lever is the cache mandate
  (shared tables already done; per-thread 128KiB ring is the next surface), measured by MPKI on the
  locked harness — NOT the ISA-L per-symbol hot-technique graft.
- ADVISOR CORROBORATION NEEDED (supervisor — you run advisors I can't): this REJECTS the charter's
  leading lever (a) with a mechanism. Please corroborate (1) the BMI2-already-on finding, (2) the
  TIE verdict (median not best-of-N), (3) the redirect from per-symbol-ALU grafts to the cache/MPKI
  surface. If corroborated, PHASE 3's 3-way Fulcrum should center on RSS/working-set/MPKI (the
  mandate), with the BMI2/LUT/refill graft recorded as rejected-with-mechanism (no work spent).

### MILESTONE RESIDUAL — DONE [committed 6388e0b] seam stop-point reconciliation assertion
- native_fold_parity now asserts the consumer-level STOP-POINT seam reconciliation IN-FILE (the
  advisor residual). Decodes a SECOND folded chunk at the first chunk's final_bit (windowed by the
  first's resolved tail) and asserts byte-continuity from the seam offset. 32 seams checked, all
  byte-continuous; 32/32 chunks correct, 12 flipped. Test-only (not(isal_clean_tail)); native sha
  028bd002...cb410f unchanged. A future graft now can't regress the seam silently.

### STRETCH STATUS / NEXT (pending advisor corroboration of the BMI2 rejection)
- DONE: falsifier pre-registered; ceiling BOUNDED by byte-exact A/B (NOT extrapolated) — BMI2 lever
  (a) rejected w/ mechanism (TIE; already-on; memory-bound wall); (b)/(c) already present +
  same-mechanism-covered. Guest routing assert fixed for the rename. Milestone seam residual closed.
- PENDING SUPERVISOR ADVISOR: corroborate (1) BMI2-already-on, (2) median-TIE verdict, (3) redirect
  to cache/MPKI. The graft step (charter item 2) is ANSWERED by the ceiling-bound: there is NO
  per-symbol-ALU wall to graft into on the perf build; spending the stretch on BMI2/LUT/refill would
  violate "bound the ceiling before committing" — the bound says no-op.
- PROPOSED NEXT (if corroborated): pivot to the cache mandate the wall lives in — (i) re-hook the
  byte-accounting instrument to the NATIVE per-thread working set (BOOTSTRAP_BLOCK / Block.output_ring
  128KiB u16 ring; the staged_bits hook is dead post-fold), validate via GZIPPY_MEM_BALLAST_MIB
  positive control; (ii) PHASE 3 3-way locked Fulcrum reporting wall + RSS + per-thread working-set +
  L2/L3 MPKI to locate rapidgzip's real advantage (window-apply / ring layout / copy = cache surface,
  NOT inner ALU); (iii) gzippy-isal Design-A tail before PHASE 3 per the charter sequence.

## NEW CHARTER (2026-06-06, supervisor) — TIER-APPROACH: 1.0x TIE bar, DESIGN→PROVE→ALIGN
plans/tier-approach-mandate.md SUPERSEDES "faithful but accepted-slow". Pure-Rust + inline ASM allowed.
Method is TIERED (no lever hill-climb). Leader re-driving. leader-lock held by persistent sentinel.
SUBAGENT SPAWN WORKS this session: `claude -p --model opus --permission-mode bypassPermissions "<prompt>"`.

### TIER-1 DIAGNOSIS DONE (2026-06-06) — 2 read-only research subagents + leader first-hand vendor read
THE ROOT CAUSE, source-cited (this CORRECTS the prior "memory-bound / cache-mandate" framing):
- **rapidgzip's measured ~0.46s wall is ~99% ISA-L (igzip C/SIMD).** Vendor GzipChunk.hpp dispatch:
  known-window chunk -> 100% IsalInflateWrapper (:440-444, d_c); window-absent -> pure deflate::Block
  ONLY for the <=32KiB markered prefix, then `cleanDataCount>=MAX_WINDOW_SIZE` hands the multi-MiB
  remainder to ISA-L (:520-526, d_w). The pure marker engine (the thing gzippy ports for the WHOLE
  chunk) runs <=32KiB/chunk in rapidgzip. The in-place u16->u8 flip exists in deflate::Block
  (:1282-1289) but is VESTIGIAL in production (ISA-L takes over right after it would fire).
- **u16-vs-u8 buffer-width hypothesis REFUTED.** rapidgzip uses the IDENTICAL 128KiB u16 ring
  (deflate.hpp:805 `std::array<uint16_t,2*MAX_WINDOW_SIZE>`, alignas(64), reinterpret-cast to u8 in
  place). gzippy's output_ring is a faithful port. Not narrower.
- **BMI2/per-symbol-ALU REFUTED (prior arc) AND now explained:** the gap isn't one extract op; it's
  that gzippy runs a SCALAR Rust marker loop for ~100% of bytes where rapidgzip runs igzip AVX2 ASM
  (igzip_decode_block_stateless_04.asm via multibinary dispatch) for ~99%.
- **VENDOR BENCH TABLE (deflate.hpp:72-93,137-145) is the TIER-2 ceiling evidence.** SINGLE-THREAD
  silesia (memory-bw-reduced): ISA-L 720 MB/s; rapidgzip's OWN best PURE decoder ShortBitsMultiCached-11
  337 MB/s; DoubleLiteralCached (what gzippy ported, class-of) 252 MB/s. => ISA-L is 2.1x faster than
  the best pure decoder single-thread; a pure SCALAR loop tops ~337 MB/s. Multi-thread the gap shrinks
  to 1.28x (5024 vs 3927) as the workload becomes DRAM-bandwidth-bound. gzippy measures 1.85x at T8 —
  WORSE than rapidgzip's own 1.28x pure/ISAL, i.e. gzippy's pure engine underperforms even rapidgzip's
  pure decoder, so there is BOTH pure-Rust headroom AND an irreducible ASM-class gap.
- **gzippy native per-thread DECODE hot state ~279KiB** (subagent A, cited): output_ring 128KiB +
  INLINE dist code_cache 128KiB (HuffmanCodingReversedBitsCached<30>, faithful to vendor deflate.hpp:336)
  + lit/len short LUT 16KiB. Overflows 256KiB L2, fits >=512KiB L2. Cache-hostile: 128KiB dist cache
  (scattered per-backref), backref source reads up to 64KiB back in ring. Per-chunk ~20MiB u16 buffer +
  resolve pass are DRAM-bandwidth, not residency.
- **3 PRIOR-NOTE DISCREPANCIES (subagent A, must reconcile):** (1) FIXED_TABLES/LitLenTable/DistTable
  (libdeflate path) are NOT live on native — the prior "shared tables already done" checked a DEAD path;
  live tables are per-thread, per-block-rebuilt, NO shared read-only table on the hot loop. (2) the dist
  cache is a 128KiB INLINE (unboxed) array => thread_local Block is ~279KiB, TWO 128KiB structures not
  one. (3) resolve streams the ~20MiB chunk buffer, not the 128KiB ring.
- **STALE COMMENT found (structure-mandate target):** guest_fulcrum_capture.sh:69-71 claims
  GZIPPY_BUILD_FEATURES=isal-compression gives "the SAME ISA-L clean decode rapidgzip uses (apples-to-
  apples engine A/B)". build.rs:90-96 proves FALSE post-fold: isal-compression no longer enables ISA-L
  DECODE; it would just rebuild the pure decoder. The only real-ISA-L gzippy build is gzippy-isal
  (Design-A tail, deferred). => no in-one-binary engine-swap A/B currently exists.

### CONSEQUENCE FOR THE DESIGN (TIER-1 conclusion)
The 1.0x TIE bar => gzippy-native needs a pure-Rust+inline-ASM bulk DEFLATE decoder competitive with
igzip's AVX2 inner loop for the CLEAN/known-window path (where rapidgzip spends ~99% of decode). This is
NOT the cache-mandate (buffers are faithful/same-size) and NOT BMI2 (already on, scalar-loop-bound). The
faithful-rapidgzip structure is NOT one-engine — it hands the clean tail to igzip; the governing
"one MarkerRing engine, no 2nd engine, no FFI" memory is in TENSION with a TIE (flagged to supervisor).
TIER-2 must PROVE feasibility (a standalone toy igzip-class Rust+ASM decoder benchmarked in isolation vs
ISA-L on the guest, + a bandwidth-vs-compute model from the vendor bench constants) BEFORE TIER-3 build.
Full TIER-1 design + TIER-2 proof plan: plans/tier1-design-and-tier2-proof.md (THIS checkpoint's deliverable).

### INDEPENDENT ADVISOR DISPROOF PASS [DONE 2026-06-06] — verdict: sound-with-corrections
Spawned an independent Opus advisor (read-only, disproof-driven) to attack the diagnosis by re-reading
cited source. It first-hand re-verified CLAIM 1 (ISA-L dominance, all 3 handoffs GzipChunk.hpp:440/501/520),
the 128KiB u16 ring identity, the 128KiB INLINE dist cache (huffman_reversed_bits_cached.rs 1<<15 x 4B,
NOT boxed) + no-shared-table, the BMI2 rejection, AND that the guest trace binary is the ISA-L build
(leader also confirmed: CMakeCache LIBRAPIDARCHIVE_WITH_ISAL:BOOL=ON, 42 isal symbols in nm). 5 corrections,
ALL incorporated into tier1-design-and-tier2-proof.md:
1. SEPARATE ring-STORAGE width [refuted, both 128KiB u16] from clean-bulk WRITE+RESOLVE traffic width
   [LIVE lever: gzippy writes u16 per literal marker_inflate.rs:1526 + re-streams ~20MiB u16 resolve on
   ~100% of bytes; rapidgzip ISA-L writes u8 direct isal.hpp:257 + markers only in <=32KiB prefix => ~0%].
2. High-T convergence is partly CACHE CONTENTION (vendor deflate.hpp:170-172, "LUT too large when 2 HW
   threads share a core"), NOT pure DRAM bandwidth => class-T traffic/residency levers must NOT be
   pre-ranked below class-C SIMD compute. Lever ranking was a soft tier-discipline violation; removed.
3. PROOF-2 validate-by-reproducing-the-2-walls is overfit-circular => added HOLD-OUT cross-validation
   (fit T1/T2/T4, predict T8/T16) + DIRECT perf-stat binding-term measurement as the verdict.
4. Vendor MB/s (337/720) are a Frankensystem CPU (deflate.hpp:96-107, +-20% variance) — illegitimate as
   guest targets; gate on guest-measured RATIOS (in-bench ISA-L oracle) only.
5. Added PROOF-0 PRE-GATE: cheapest decisive experiments FIRST — PG-A clean-loop slow-injection (causal
   perturbation: monotonic T8-wall response => compute binds => build SIMD; flat => SIMD moot, falsified
   in ~1hr not after a multi-week build) + PG-B u8-write+drop-clean-resolve A/B (isolates class-T traffic).
   This is the gate the BMI2 arc lacked.
Advisor bottom line: diagnosis sound to design on; design was partially mis-aimed (under-weighted class-T)
— now fixed; PROOF-1 sound, PROOF-2 fixed, PRE-GATE is the missing decisive gate, run it first.

## CHECKPOINT REACHED [2026-06-06] — TIER-1 design + TIER-2 proof plan delivered for supervisor/advisor
NO TIER-3 implementation started (charter: deliver design+proof BEFORE building). Awaiting supervisor +
independent-advisor ratification of: (1) governing-tension resolution (one no-FFI engine, igzip-class
clean inner loop via pure-Rust+inline-ASM, inner-loop divergence accepted while architecture stays
faithful); (2) the diagnosis; (3) the revised TIER-2 plan (PRE-GATE -> PROOF-1 -> PROOF-2). leader-lock
held by persistent sentinel pid 156. Subagent spawn confirmed working: claude -p --model opus
--permission-mode bypassPermissions.

---
## SUPERVISOR NOTE 2026-06-06 23:24 (lock hygiene)
First leader (TIER-1) completed and exited but left an ORPHANED `sleep 86400`
leader-lock sentinel (pid 156, ppid=1) + a redundant TIER-1 disproof advisor
(pid 4391). The orphaned sleep held the leader-lock "alive" for 24h and would
wedge any new leader's mkdir-mutex acquire. Supervisor killed both and cleared
/tmp/gzippy-leader.lock.d. TIER-2 leader (current) is the SOLE live leader.
RECOMMENDATION for any future leader: do NOT hold the leader-lock with a detached
`sleep` sentinel that outlives you — it orphans on exit. The supervisor already
enforces single-leader via pgrep before each spawn; the sleep-sentinel lock is
net-negative. Prefer no sentinel (supervisor-enforced singleton) or a lock whose
holder is the leader's own pid (auto-released on death).

---
## TIER-2 PRE-GATE — INSTRUMENT FIXED + SWEEP DONE [2026-06-07, TIER-2 leader]

### TASK 1 (instrument-validity) — DONE. Injection moved to the NATIVE clean arm.
The prior subagent had the slow-injection in resumable.rs (Engine C / gzippy-ISAL
path) with a comment claiming "FlipToClean tail runs 100% through resumable" —
WRONG for native. On gzippy-native (the 1.0x bar) the fold keeps Engine M
(marker_inflate::Block) decoding the clean tail in-place (gzip_chunk.rs:1219,
not(isal_clean_tail)); ~99% clean bytes decode through marker_inflate's
CONTAINS_MARKERS=false arm. Injection is now wired into BOTH native clean sites:
read_internal_compressed_specialized::<false> (the lut_litlen multi-symbol loop,
the live native dynamic-Huffman path) AND read_internal_compressed_canonical_
specialized::<false>. resumable injection KEPT (correct site for the isal control).
PROVEN on perf-target build (Rosetta x86_64 gzippy-native, path=ParallelSM,
silesia): GZIPPY_SLOW_HITS counter = 40,131,993 clean decode events for the ~203MB
decode (∝ clean bytes, NOT ~0). Site is the live native clean loop. Commit d0aa1db.

### TASK 2 (self-test) — PASS.
- OFF byte-exact 028bd002...cb410f on BOTH gzippy-native AND gzippy-isal (x86_64),
  T1 + T8, path=ParallelSM. OFF == identity by construction (hoistable spin==0).
- POSITIVE CONTROL (locked harness, silesia-large 503MB, N=9): T1 F=0 3.7340s ->
  T1 F=100 spin 6.3833s = +71% (sd 0.1%, massively out of spread). Knob really
  slows the loop. The pause-hint "sleep" was too cheap to be a valid freq-neutral
  control (~7% at 5x), so the sleep kind was reimplemented as a REAL batched
  thread::sleep (yields the core, calibrated to the spin per-iter cost).

### TASK 4 — PRE-GATE SWEEP (locked guest harness, silesia-large 503MB, T8, N=9 interleaved, sha-verified e114dd2b..., diverged=0 every run, RUN_TRUSTWORTHY=true)
| F   | SPIN T8 wall | Δ vs F0 | SLEEP T8 wall | Δ vs F0 |
|-----|-------------|---------|---------------|---------|
| 0   | 1.1209s     | —       | 1.1209s       | —       |
| 25  | 1.1597s     | +3.5%   | 1.1397s       | +1.7%   |
| 50  | 1.2281s     | +9.6%   | 1.1990s       | +7.0%   |
| 100 | 1.4368s     | +28.2%  | 1.2411s       | +10.7%  |
(within-arm sd 0.8–4.4%; every Δ at F=50/100 is out of spread under BOTH kinds.)
T1 positive control: F0 3.7340s -> F100 spin 6.3833s (+71%, sd 0.1%).

### VERDICT: COMPUTE-BOUND (clean-loop compute IS on the T8 critical path).
- MONOTONIC & PROPORTIONAL T8 response under SPIN (3.5/9.6/28.2%), out of spread.
- The rise SURVIVES the frequency-neutral SLEEP control (1.7/7.0/10.7%, monotonic,
  out of spread) — so the criticality is REAL, not a turbo artifact. Per the
  pre-registered falsifier (plans/pre-gate-falsifier.md), monotonic-survives-sleep
  ⇒ COMPUTE-BOUND ⇒ proceed to PROOF-1 (the SIMD-compute proof).
- Spin slope (28%) > sleep slope (11%) at F=100: the gap IS the turbo-depression
  the busy-spin causes (the exact confound the control isolates); the sleep slope
  is the turbo-clean lower bound and it is still clearly positive.
- DIAGNOSTIC: T1 F100 +71% vs T8 F100 +28% (spin) — at T8 the clean-loop compute
  is partially overlapped by parallelism (wall is less compute-elastic than T1)
  but STILL on the critical path. NOT flat ⇒ class-C (SIMD compute) is NOT moot.
- 1b (u8-write / traffic A/B) NOT run in this window — flagged as the immediate
  next experiment to rank class-T traffic as co-lever (matrix row: 1a MONOTONIC,
  1b TBD ⇒ at minimum COMPUTE-BOUND→PROOF-1; class-T co-lever pending 1b).

NEXT: supervisor + independent disproof advisor gate this verdict BEFORE PROOF-1/
PROOF-2 or any SIMD build (per charter SEQUENCE). No TIER-3 started.

### TASK 5 — INDEPENDENT DISPROOF ADVISOR [DONE] verdict: CORROBORATE-WITH-CAVEATS
Full verdict: plans/pre-gate-advisor-verdict.md (independent Opus, read-only, read
the cited source). It could NOT construct a path to FLAT/bandwidth-bound — the
narrow claim survives. Corrections folded in (the VERDICT block above is AMENDED):
- RELABEL: drop "COMPUTE-BOUND" (overreach). Correct claim = "clean-loop compute
  is ON the T8 critical path; class-C (SIMD compute) is NOT moot." The wall is
  MIXED; clean compute is only ~11–29% of the T8 wall (sleep floor 11% vs spin/T1
  ~29% — methods in genuine tension; the T1 positive control +71% extrapolates to
  ~29%, matching SPIN, so the usual "trust the sleep floor" likely UNDER-reports).
- RULE 3 (mandatory): slowing the loop adds wall ≠ speeding it pays. The pre-gate
  licenses ONLY "not-moot." PROOF-1 = the REMOVAL ORACLE (bound the speed-up
  ceiling) is MANDATORY before ANY SIMD build. Do NOT start the SIMD stretch on
  this pre-gate alone.
- BIGGER LEVER (advisor): even infinite clean speed-up -> 0.79–1.00s vs rapidgzip
  0.53s — still ~1.5–2x slower. T8 runs ~42% parallel-efficiency (1.121s vs 0.47s
  ideal); ~58% lost to placement / head-of-line stalls (confirmed-offset-prefetch-
  gap, ~40% of T8). THAT outranks class-C for closing the rapidgzip gap. Weigh the
  placement lever before committing to the SIMD project.
- INSTRUMENT: site re-asserted on the EXACT 503MB guest T8 build — GZIPPY_SLOW_HITS
  = 94,887,526 clean decode events via path=ParallelSM (∝ clean bytes). Treat
  F->%wall as approximate (T1-calibrated); good for monotonicity, not magnitude.
- 1b (u8-write/traffic A/B) still not run — the next experiment to rank class-T.

---
## TIER-2 STEP A — REMOVAL ORACLES [2026-06-07, STEP-A leader, fresh instance]
CHARTER: plans/tier2-revised-placement-primary.md. Bound BOTH ceilings with removal
oracles (CLAUDE.md rule 3) BEFORE any design commit. Guest VERIFIED IDLE (load 1.04,
no stray gzippy/cargo/rapidgzip/harness procs; /dev/shm clean). HEAD ce9fe6f.

### Falsifiers PRE-REGISTERED (before any run)
- plans/step-a-oracle-c-falsifier.md (clean-compute removal). Mechanism = existing
  decode-bypass FLOOR (PASS A) with the prior leverB CONTAMINATION fixed (prebuilt-map
  init out-of-wall; /dev/shm capture; swap gate; region-removed proof = clean-decode
  span→≈0 in floor trace). Prediction to beat: ~0.79-1.00s (advisor) / 0.78-0.84 (model).
- plans/step-a-oracle-p-falsifier.md (perfect-placement removal — DECISIVE). Mechanism
  P1 = trace-derived optimal-makespan (LPT) bound on the EXISTING locked-Fulcrum T8
  per-chunk decode-busy times (placement-free lower bound, no new code) + self-tests
  (sum-conservation, max-chunk, Σ/8); P2 = bypass-FLOOR placement-residual cross-check.
  Verdict: FLOOR_P ≤ ~0.55s ⇒ placement SUFFICIENT (proven path to tie).

### KEY INFRA FINDING (saves a build): the Oracle-C engine ALREADY EXISTS and is byte-exact
decode_bypass.rs CAPTURE/REPLAY (GZIPPY_BYPASS_CAPTURE/DECODE) + guest_ceiling.sh PASS A.
Prebuilt path (decode_bypass.rs:460-471) = HashMap lookup + Option::take, no per-call
memcpy. Prior leverB run (plans/leverB-ceiling.md) read 3.667s = LOAD-CONTAMINATED
(one-time prebuilt rebuild in-wall + swap), declared a HYPOTHESIS not a ceiling, owed a
re-run "from RAM, prebuilt out-of-wall". THIS is that re-run.

### ORACLE-P [DONE — trace-derived, validated] → PLACEMENT SUFFICIENT
plans/step-a-oracle-p-falsifier.md MEASURED RESULTS. Clean T8 trace (000148, HEAD
d0aa1db = decode-identical parent of ce9fe6f). FLOOR_P (placement-free single-pass
LPT makespan, gzippy engine fixed) = **0.41-0.49s ≤ rapidgzip 0.524s** ⇒ perfect
placement reaches the tie band WITHOUT touching the engine. Positive control:
actual/makespan = 1.36 on BOTH tools. MECHANISM: 5.90s of gzippy's 6.33s T8 decode
busy = worker.scan_candidate (speculative re-decode at mispredicted boundaries);
clean decode (isal_stream_inflate) only 0.365s; single-pass T1 = 3.734s. rapidgzip
decode busy 2.99s (confirms boundaries ahead, never re-scans). VERDICT: PLACEMENT is
the proven primary lever (faithful rapidgzip boundary-confirmation port). CAVEAT
(honest): bound assumes the redundant scan is eliminable + single-pass balances — the
port-feasibility question, NOT proven by the ceiling (advisor to attack).

### ORACLE-C [RE-RUNNING] — first attempt ORPHANED (lesson)
SUBAGENT-ORPHAN LESSON: the `claude -p` Oracle-C subagent printed an interim message
and EXITED (treated the backgrounded guest run as auto-reinvokable) → SIGHUP killed
its child driving-ssh → host_lock died → watchdog restored freq state → run produced
only the build, no captures. FIX: leader runs the guest harness DIRECTLY via a
background Bash task (parented to the persistent runner, not an exiting subagent).
Re-run launched (task b486p1g4b, /tmp/oracle-c-run2.log) at HEAD a49c357 (warm_prebuilt
fix + routing-assert fix both in). Guest verified idle (load 1.08, shm clean) before launch.

### ORACLE-C [DONE — run2 a49c357] → class-C floor ~0.4-0.7s (instrument-entangled), GREY
plans/step-a-oracle-c-falsifier.md MEASURED RESULTS. Locked harness, RESTORE VERIFIED.
HARD gates PASS (sha=OK byte-exact, hit%=97.6, sd%=0.9, anchor 1.13s reproduced, no
swap-thrash). REGION-REMOVED PROOF passes (decode_chunk busy 6.33s→0.076s in floor
trace). BUT the raw CEIL_FLOOR_A=3.63s is DOUBLY contaminated: (1) +3.1s warm_prebuilt
still in the whole-process wall (my fix moved it before drive_t0 but the harness times
the process, not drive_t0 — fix incomplete); (2) decode≈0 frees windows ⇒ L_resolve
collapses (162µs vs 19.93ms) ⇒ fulcrum critpath 336ms UNDER-states. Robust bracket:
A2 0.444 / critpath 0.336 / A−warm−load ~0.5 / B-sleep66 0.732 ⇒ class-C floor ~0.4-0.7s,
BELOW the advisor's 0.79-1.00s. KEY: full decode-removal is a DEGENERATE oracle — it
co-collapses the publish-chain it should preserve. ⇒ class-C bounded (≤~0.7s floor ⇒
infinite clean speedup buys ≤~0.4s/37% of wall), NOT the clean lever Oracle-P is.

### BOTH CEILINGS LANDED — CHECKPOINT
- Oracle-P (placement-free): **0.41-0.49s** ≤ rapidgzip 0.524s ⇒ PLACEMENT SUFFICIENT.
- Oracle-C (decode-free): **~0.4-0.7s** (entangled, GREY) ⇒ class-C bounded-secondary.
- Both land in the same ~0.4-0.7s band because gzippy's two costs (redundant
  speculation = placement; publish-chain) OVERLAP in the parallel pipeline.
- IMPLICATION: placement is the proven primary path to the 1.0× tie (faithful rapidgzip
  boundary-confirmation-ahead port); class-C/class-T secondary. The 5.9s scan_candidate
  (speculative re-decode) is the single largest recoverable item.
NEXT: spawn independent disproof advisor (read-only) → plans/step-a-oracle-advisor-verdict.md.
THEN STOP for supervisor gate. Do NOT start STEP B/C/D or TIER-3.

### DISPROOF ADVISOR [DONE] → CHECKPOINT (STOP for supervisor)
plans/step-a-oracle-advisor-verdict.md. Read-only, re-derived ALL numbers first-hand
(all reproduce). VERDICT: Oracle-P "PLACEMENT SUFFICIENT" REFUTED → NECESSARY-BUT-
INSUFFICIENT; Oracle-C CORROBORATE-WITH-CAVEATS (degenerate, GREY). Key correction:
ramp-consistent placement-perfect gzippy = 0.56-0.66s vs rapidgzip 0.524s (7-26% loss);
the 5.9s scan_candidate is first-pass MARKER decode (decoded once, expensively), not
redundant re-decode; an ENGINE residual survives perfect placement (gzippy clean
91ms/chunk vs rapidgzip 39ms = 2.3×) ⇒ class-C is CO-PRIMARY, not bounded-secondary.
CHECKPOINT REACHED. Did NOT start STEP B/C/D or TIER-3. Awaiting supervisor gate.
OWED before STEP-C (advisor): a CLEAN-ONLY T8 removal oracle (force all chunks through
isal_stream_inflate w/ predecessor windows, measure busy) = the least-entangled engine
ceiling signal both oracles missed.

## TIER-2 STEP A.2 — CLEAN-ONLY ENGINE ORACLE [DONE 2026-06-07, A.2 leader, fresh instance]
CHARTER plans/step-a2-clean-only-oracle.md. Falsifier PRE-REGISTERED before any run:
plans/step-a2-clean-only-falsifier.md. HEAD e89006b0 (commits: instrument 4bd8ecb7,
guest harness 96f98a16, driver e89006b0). Guest verified idle before launch; host
RESTORE VERIFIED after (no_turbo=0, all 8 guests thawed, /dev/shm seed file removed).

### INSTRUMENT (byte-exact, env-gated) — NOT the broken/degenerate prior oracles
- NEW src/decompress/parallel/seed_windows.rs. GZIPPY_SEED_WINDOWS_CAPTURE (p=1 records
  aligned start_bit→window pairs at the NATURAL clean path) + GZIPPY_SEED_WINDOWS
  (pre-seed block_finder w/ aligned boundaries + clean-path window fallback). FORCES
  every chunk clean while KEEPING real Huffman decode AND the production consumer/publish
  chain — distinct from (a) the BROKEN drive_clean_window_oracle (GZIPPY_CLEAN_WINDOW_ORACLE,
  bypasses the whole scheduler+consumer with a bespoke loop) and (b) the DEGENERATE
  Oracle-C/decode_bypass (decode≈0 → publish-chain collapse).
- STRUCTURAL FINDING (proven while building): clean-only is INCOMPATIBLE with production
  speculative dispatch UNLESS boundaries are confirmed ahead — clean decode is only
  possible at a real deflate boundary, but dispatch uses spacing GUESSES whose
  WindowMap-key windows are MISALIGNED to the guess (first naive WindowMap-snapshot capture
  hit 0%/diverged). Engine-isolation REQUIRES confirmed-boundary dispatch = the placement
  fix itself ⇒ placement & engine are structurally COUPLED.

### SELF-TEST PASSES (CLAUDE.md rule 4) — proves it isolated engine + preserved publish chain
silesia-large 503MB, T8 (guest 199, locked, sha e114dd2b…):
- forced-clean: SEED window_seeded 4→39 (ALL clean), finished_no_flip 38→0 (zero marker
  decode), fused_lut 35→0 (zero marker resolution); hit%=100.0, sha=OK byte-exact.
- publish-chain-preserved: Early window publish=39 in BOTH normal AND seeded (NOT collapsed
  like Oracle-C); seeded trace consumer.window_publish_clean = 2.2ms total / 0.06ms per chunk
  (running, just cheap with no markers).
- off==identity (env unset = today's decode, by construction); routing 43/0 lib tests.

### RESULT — ENGINE CEILING (the deliverable). VERDICT: ENGINE-IS-RESIDUAL (co-primary CONFIRMED)
| arm (T8, N=9 interleaved, sha-OK) | min | median | sd% | MB/s |
|---|---|---|---|---|
| gzippy clean-only (ENGINE ceiling, publish chain intact) | 0.5896 | **0.6134** | 1.9 | 854 |
| rapidgzip | 0.5347 | 0.5396 | 3.3 | 942 |
| gzippy normal (baseline) | 1.1157 | 1.1244 | 1.1 | 451 |
- **clean-only 0.6134 vs rapidgzip 0.5396 = +13.7% (Δ0.0738s ≫ ~2-3% spread) → NOT a TIE.**
- Lands in the PRE-REGISTERED [0.58,0.72] ENGINE-IS-RESIDUAL band. Engine gap SURVIVES
  forcing every chunk clean ⇒ the residual is the ENGINE, confirming class-C is co-primary.
- Per-chunk CLEAN busy (seeded trace, 39 chunks): worker.isal_stream_inflate = **92.7ms/chunk**
  — corroborates the advisor's independent 91ms derivation; vs rapidgzip 39ms = **2.38× engine
  gap**, ramp-consistent. Implied engine-bounded wall = 39×92.7/8×1.36 ≈ 0.61s == measured.
- RAMP-CONSISTENT (the STEP-A error fixed): clean-only WALL (0.61) vs rapidgzip WALL (0.54),
  same treatment — no floor-vs-wall mismatch. Decode busy 3616ms/8 = 0.452 ideal, ×1.36 = 0.61.

### CHECKPOINT — engine ceiling cleanly bounded. The grey 0.4-0.7s (Oracle-C) RESOLVED to ~0.61s.
- Placement ceiling (Oracle-P, ramp-consistent): ~0.56-0.66s. Engine ceiling (this, A.2): ~0.61s.
  rapidgzip 0.54s. BOTH levers needed for the 1.0× tie; both are real, both ~13% gaps.
- 1b (u8-write traffic A/B) NOT run this window (engine oracle was the priority + landed; flag
  as still-owed to rank class-T). Did NOT start STEP-C design revision or any TIER-3.
NEXT: spawn ONE independent disproof advisor (read-only) → plans/step-a2-advisor-verdict.md to
attack the oracle's validity (engine isolated? publish-chain preserved? ramp consistent?), then
STOP for supervisor gate.

## TIER-2 STEP C — REVISED DESIGN [DONE 2026-06-07, STEP-C leader, fresh instance]
CHARTER plans/step-c-revised-design.md. TIER-2 PROVE complete; this synthesizes the proven
ceilings into the revised TIER-1 design (the TIER-3 gate). HEAD b8a38e64. No guest measurement
run this turn (design synthesis + read-only subagents only); no host state touched.

### DELIVERABLE — plans/tier1-design-v2.md (the revised single coherent architecture)
- §1 PLACEMENT (faithful port): read-only subagent mapped rapidgzip's 5 scheduler mechanisms
  to gzippy two-column, source-cited (/tmp/scheduler-vendor-map.md). FINDING: 3 already FAITHFUL
  (prefetch strategy, window propagation, consumer-overlap join), 1 PORTED-BUT-DEFEATED
  (confirmed-boundary dispatch), 1 MISSING = the prime defect: INTERIOR/SUBCHUNK REUSE
  (getIndexedChunk, vendor GzipChunkFetcher.hpp:254-309). gzippy BUILDS unsplit_blocks
  (chunk_fetcher.rs:2752) + block_map.find_data_offset (block_map.rs:135) but NEVER READS them.
  Corrects a STALE memory note: matches_encoded_offset IS a range check (chunk_data.rs:461-469),
  not == equality — the blocker is missing interior-EMIT, not the accept predicate.
- §2 ENGINE (inner-loop open territory): E1 u8-direct clean write (the 1b sub-lever, low-risk
  first), E2 wide SIMD back-ref copy, E3 packed multi-literal store (ca52389 non-binding), E4
  wide refill — each isolation-benchmarked vs an in-bench ISA-L positive control (guest ratio,
  NOT Frankensystem absolutes) BEFORE integration. Governing-tension resolution STANDS: one
  no-FFI engine, inner loop reimplemented igzip-class; inner-loop divergence accepted, arch
  faithful (needs supervisor ratification).
- §3 REACHABILITY (coupled, not additive): placement-alone ⇒ ~0.61s (A.2 MEASURED, not rescaled);
  placement+engine-at-igzip-class ⇒ decode stops binding, wall ⇒ shared pipeline floor.
- §4 STRUCTURE mandate sequenced; §5 TIER-3 sequence; §6 checkpoint.

### INDEPENDENT DISPROOF ADVISOR [DONE] → plans/step-c-design-advisor-verdict.md
VERDICT: ratify the METHOD, not the CONCLUSION. All 3 targets CORROBORATE-WITH-CAVEATS:
- §3: coupled-not-additive is sound + double-count-free; 0.61s is measured. BUT the 0.61→0.54
  "TIE" step assumes gzippy's NON-decode floor = rg 0.54s — contradicted by a MEASURED ~225ms
  consumer-serial term + 0.497s consumer block (neither shrinks with engine speed). Read §3 as
  ≥0.54s, plausibly 0.54-0.62s, TIE only at the optimistic edge.
- §1: range-check correction right + Subchunk.decoded_offset exists; BUT "just wire dead code"
  too glib — find_data_offset is DECODED-BYTE-keyed for a consumer gzippy doesn't have (gzippy
  is encoded-BIT-keyed), unsplit_blocks only built for subchunks>1, AND the parent-cached
  precondition was MEASURED TO FAIL (318ms lag → eviction) + left as an UNANSWERED discriminator.
- §2: E1/E3 genuinely unbuilt (re-attempt justified); guest-ratio ISA-L control is a sound ENGINE
  gate; BUT 2.1× pure-decoder ceiling is a hard headwind + the bench gates engine, not wall.
- BOTTOM LINE: safe to authorize TIER-3 as a MEASUREMENT-GATED INVESTIGATION; do NOT ratify the
  headline "1.0× tie reachable." ONE thing most likely to break it: gzippy's non-decode
  consumer-serial floor is structurally higher than rg's. Add a 3rd pre-registered measurement
  (decompose the 0.61s consumer block into decode-wait vs serial) BEFORE the engine build.

### CAVEATS FOLDED BACK INTO plans/tier1-design-v2.md (design ⇄ verdict now consistent)
- §1.2: added THREE port pre-conditions (coordinate-system mismatch; single-subchunk map gap;
  parent-cached discriminator — answer FIRST). The faithful port is getIndexedChunk's LOGIC in
  gzippy's encoded-bit offset space, NOT changing the consumer request model (that = forbidden
  redesign).
- §3: added the NON-DECODE FLOOR caveat (≥0.54s) + the OWED THIRD MEASUREMENT (consumer-block
  decompose). Verdict now: tie reachable CONDITIONAL on TWO floors (engine ≤39-45ms AND
  non-decode floor ≤0.54s); else add a 4th off-critical-path consumer lever or revisit the bar.
- §5: added SEQUENCE step 0 = the two cheap byte-exact discriminators (parent-cached probe +
  consumer-block decompose) run FIRST, before any port/build.

### CHECKPOINT REACHED — STOP for supervisor ratification (TIER-3 gate). NO TIER-3 started.
Subagents (vendor map + advisor) were READ-ONLY, synchronous, collected in-turn; both process
trees explicitly killed (no orphans). No guest run, no host state change, no build this turn.
Supervisor to ratify: (1) the method + sequence; (2) the governing-tension resolution (§2.1);
(3) that TIER-3 is authorized as a measurement-gated investigation (not a ratified tie claim),
with step-0 discriminators run before the placement port and before the engine build.

## TIER-3 STEP 0 — TWO DISCRIMINATORS [IN PROGRESS 2026-06-07, STEP-0 leader, fresh instance]
CHARTER plans/tier3-step0-authorization.md. STEP 0 ONLY (two cheap discriminators); NO placement
port, NO engine work. HEAD b8a38e64. Falsifiers PRE-REGISTERED BEFORE any run:
plans/step0-discriminator-a-falsifier.md (parent-cached) + plans/step0-discriminator-b-falsifier.md
(consumer-block decompose + HARD ESCALATION GATE >0.54s).
Guest 199 verified idle (load 0.51 falling, no stray procs, /dev/shm only stale debug files,
no_turbo=0 = unlocked). Leader-lock: none held (supervisor enforces singleton; NO sleep sentinel
per the 2026-06-06 23:24 lock-hygiene note). Disk 30Gi free (94%) — watching df around builds.

### DISCRIMINATOR (b) — CONSUMER-BLOCK DECOMPOSE [PRELIMINARY from existing A.2 trace] → PASS (floor ≈18ms ≪ 0.54s)
Instrument: scripts/bench/consumer_block_decompose.py — per-span SELF-time on the consumer tid
(tid=1) with no-double-count (nested children subtracted, CLAUDE.md rule 8). WAIT vs SERIAL
classification done by READING THE SOURCE (not guessing): blocking-wait spans are
wait.block_fetcher_get / wait.future_recv / ttp.rx_recv_block (block_fetcher.rs:247 rx.recv_timeout
pump) / consumer.dispatch_recv (chunk_fetcher.rs:3051 rx.recv()); everything else = serial CPU.
- PHANTOM-LEVER CAUGHT: a first (wrong) classification with only {block_fetcher_get, future_recv}
  in the wait set put NORMAL-path SERIAL at 0.515s — but ttp.rx_recv_block (0.26s) + dispatch_recv
  (0.19s) are BLOCKING WAITS, not CPU. The advisor's "~225ms consumer-serial" is this same
  misattribution: those spans are decode-wait, not bookkeeping. Source-read dissolved it.
- Operating point = the A.2 CLEAN-ONLY oracle trace (trace_seed_T8.json, GZIPPY_SEED_WINDOWS,
  every chunk clean at a confirmed boundary = the placement-PERFECT operating point §3 prices the
  0.61s consumer block at). Conservation PASSES (sum-self vs wall gap 0.46%).
  | trace | consumer_wall | DECODE-WAIT | SERIAL-BOOKKEEPING (the floor) |
  |---|---|---|---|
  | clean-only (placement-perfect) | 0.506s | 0.489s (96.5%) | **0.0178s (3.5%)** |
  | normal (marker baseline)       | 1.048s | 0.979s (93.4%) | 0.069s (6.6%) |
- VERDICT (preliminary): non-decode SERIAL floor ≈ 0.018s ≪ 0.54s ⇒ HARD GATE PASSES with a
  huge margin; the consumer does NOT structurally forbid the tie. STILL OWED before final:
  the SLOW-knob positive control (decode slow-injection must inflate DECODE-WAIT while SERIAL
  stays flat — pre-registered) + N≥7 robustness, from a fresh locked guest run.

### DISCRIMINATOR (a) — PARENT-CACHED-AT-STALL [DONE — locked guest] → NO (NOT_RESIDENT) ⇒ RE-SCOPE placement
Probe GZIPPY_STALL_RESIDENCY_PROBE (commit 76e9dc9d, stall_residency.rs) at the cold-get None
branch. Locked guest 199, silesia-large 503MB, T8, host-lock GATE PASS, RESTORE VERIFIED.
Byte-exact every run (sha=OK == ref e114dd2b…); OFF==identity match=YES.
- VERDICT RUN (default caps): total=4 startup=1 CONTAINING_CACHED=0 CONTAINING_IN_FLIGHT=0
  NOT_RESIDENT=4 → 0% of non-startup stalls have a resident containing parent.
- Per-stall detail (auditable, nearest_le_start = nearest cached start ≤ decode_start): EVERY
  non-startup stall has nearest_le_start:-1 — NO cached/in-flight chunk has a start ≤ the stalled
  offset; ALL resident + in-flight chunks are AHEAD of decode_start. The consumer is BEHIND its
  prefetch frontier; the containing chunk was already consumed/passed. Genuinely EVICTED/absent.
- POSITIVE CONTROL passes (probe tracks residency): cap=1 (tiny) total 4→9 NOT_RESIDENT 4→9 (more
  eviction → more cold-get stalls); cap=256 (huge) stays total=4, still 0% resident — proving this
  is NOT a cache-CAPACITY problem (default cap 16 already holds the set); it is consumer-PACE/overshoot.
- ANSWER = NO. Per the pre-registered verdict rule, NOT_RESIDENT majority ⇒ the gap is
  cache-residency/consumer-pace, NOT interior-reuse-fixable. The getIndexedChunk/interior-EMIT port
  (tier1-design-v2 §1.2) presupposes a resident parent it does NOT have ⇒ that placement recipe must
  be RE-SCOPED: the fix is UPSTREAM of interior reuse (make the consumer keep pace / keep the
  containing chunk in-flight that rapidgzip keeps). Chicken-and-egg the advisor flagged. Confirms
  [[project_confirmed_offset_prefetch_gap]] ROOT CAUSE (consumer lags prefetcher ~318ms).

### DISCRIMINATOR (b) — CONSUMER-BLOCK DECOMPOSE [DONE — locked guest] → PASS (floor ~0.015s ≪ 0.54s)
Locked guest, clean-only oracle (GZIPPY_SEED_WINDOWS = placement-perfect operating point),
GZIPPY_TIMELINE trace_v2 spans, output→/dev/null (T-invariant sink), 3 baselines + SLOW control.
consumer_block_decompose.py self-time (no double-count); WAIT spans = block_fetcher_get/future_recv/
ttp.rx_recv_block/dispatch_recv (source-classified). sha=OK byte-exact every run.
| trace | consumer_wall | DECODE-WAIT | SERIAL-BOOKKEEPING (floor) |
|---|---|---|---|
| clean  | 0.506s | 0.493s (97.5%) | 0.0127s |
| clean2 | 0.507s | 0.492s (97.0%) | 0.0151s |
| clean3 | 0.500s | 0.482s (96.5%) | 0.0176s |
| SLOW+100% | 5.142s | 5.097s (99.1%) | 0.0448s |
- non-decode SERIAL floor = 0.013–0.018s (mean ~0.015s, spread ~0.005s) ≪ 0.54s ⇒ HARD GATE PASSES
  with ~36× margin. The consumer does NOT structurally forbid the tie.
- POSITIVE CONTROL passes the pre-registered falsifier: decode +100% (78.7M clean-loop inject hits
  confirmed) inflated DECODE-WAIT 0.49→5.10s while SERIAL stayed 0.013→0.045s (FLAT) ⇒ the decompose
  correctly isolates decode-wait from serial.
- CONFOUND CAUGHT + FIXED: an interim run had output→disk(mktemp), inflating consumer.writev to a
  fake 0.267s "floor". Streaming output→/dev/null removed it. The advisor's feared ~225ms
  consumer-serial term was the decode-WAIT spans (rx_recv_block/dispatch_recv) mis-bucketed as
  serial — dissolved by source-reading the span semantics.

### STEP-0 CHECKPOINT (STOP for supervisor gate)
- (a) parent-cached = NO (0% resident, 318ms-lag eviction confirmed) ⇒ tier1-design-v2 §1.2
  interior-reuse/getIndexedChunk port is NOT the fix as written; placement must be RE-SCOPED to a
  consumer-pace fix (faithful: rapidgzip's consumer stays ~0-17ms behind its prefetcher; gzippy lags
  ~318ms). NOT a "wire dead code" port.
- (b) non-decode floor = ~0.015s ≤ 0.54s ⇒ CONTINUE (no escalation; consumer does not forbid the
  tie). The ENGINE front remains gated by §2.3's isolation bench (not run this turn).
- NEXT: independent disproof advisor (read-only) → plans/step0-advisor-verdict.md, then STOP for the
  supervisor gate. NO placement port, NO engine work started. Host RESTORE VERIFIED; no orphan procs.

### INDEPENDENT DISPROOF ADVISOR [DONE] → both conclusions SAFE (CORROBORATE-WITH-CAVEATS)
plans/step0-advisor-verdict.md (independent Opus, read-only, re-derived from source).
- (a) CORROBORATE-WITH-CAVEATS. PRIME SUSPECT CONFIRMED: my original CONTAINING_CACHED counter
  was VACUOUS — [enc,max] is the speculative START-tolerance window (decoded bytes BEGIN at max,
  chunk_data.rs:143-145; re-anchored chunks have enc==max), so it could never fire for a real
  containing parent. BUT the NO verdict SURVIVES on the bug-free channels: nearest_le_start:-1 on
  every stall (correct necessary-condition test) + CONTAINING_IN_FLIGHT=0 (keyed). The port reads
  the parent from cache(); nearest_le_start:-1 proves it isn't there. Advisor's owed item ANSWERED.
- (b) CORROBORATE-WITH-CAVEATS. Honest; source dissolves the ~225ms fear (apply_window runs on the
  POOL, the consumer blocks on a future wrapped as wait.future_recv NESTED inside
  dispatch_post_process — correctly bucketed DECODE-WAIT). Robust across operating points (even
  NORMAL marker-heavy trace serial=0.069s ≪ 0.54s). CAVEAT: /dev/null excludes production output-
  write (writev ~0.245s on a real sink) — the real floor is ~0.015s + output-write; still passes,
  fair only if rapidgzip's 0.54s is same-sink. Keep the tie contingent on engine ≤39-45ms AND a
  same-sink floor ≤0.54s.

### CORRECTION RUN (commit d764734c, locked guest) — advisor fix applied, verdict HELD
Re-snapshot the encoded END (enc+encoded_size_bits); test enc<decode_start<encoded_end; added the
bug-free has_nearest_le_start channel. Byte-exact OFF==identity (028bd002…cb410f / e114dd2b…).
RESULT (all sha=OK): **has_nearest_le_start=0 at ALL cap settings — default(16), cap1, AND cap256
(huge).** Even with an effectively unbounded cache NO resident chunk starts at/below the stalled
offset ⇒ rules OUT capacity-eviction, CONFIRMS never-retained / consumer-pace (containing chunk
consumed+passed before the consumer reached it; lags the frontier ~318ms). The interior-reuse port
has no resident parent to reuse REGARDLESS of cache size. NO verdict now on bug-free, cap-swept evidence.

### STEP-0 FINAL (STOP for supervisor)
(a) parent-cached = **NO** (consumer-pace / never-retained, cap-swept, advisor-confirmed) ⇒
   tier1-design-v2 §1.2 interior-reuse/getIndexedChunk port is NOT the fix as written; placement
   RE-SCOPES to a consumer-pace fix (keep the containing chunk in-flight; rapidgzip's consumer stays
   ~0-17ms behind its prefetcher vs gzippy ~318ms). Faithful (rapidgzip does this), just a different
   mechanism than interior-EMIT.
(b) non-decode floor = **~0.015s (≪ 0.54s)** ⇒ CONTINUE; consumer does NOT structurally forbid the
   tie. Caveat: re-quote with production output-write for the true same-sink margin.
No placement port / engine work started. Both subagents (advisor) read-only, synchronous, no orphans.
Host freq RESTORE VERIFIED after every guest run. Awaiting supervisor gate.

## PLACEMENT RE-SCOPE — LAG-CAUSALITY DIAGNOSIS [DONE 2026-06-07, placement-rescope leader]
CHARTER plans/placement-rescope-diagnosis.md. Answer the pivotal question: is the ~318ms
consumer-prefetcher lag STRUCTURAL or an EFFECT of the 2.38x slow engine? HEAD cb60842d.
Falsifier PRE-REGISTERED before any run: plans/lag-causality-falsifier.md. Guest verified idle
before; host RESTORE VERIFIED after (no_turbo=0, all guests thawed); freed ~1.7G stale guest
clones (gzippy-{reimplement,consolidate}-verify) for build headroom. No orphans.

### PERTURBATION (slow_knob sweep, native clean arm marker_inflate.rs:1307/1546, locked guest T8, N=7, sha-verified, NO DIVERGENCE)
| combo | T8 wall | per-blk busy | STALL COUNT (non-startup) | wait.block_fetcher_get | wait/blk-busy |
|---|---|---|---|---|---|
| F0 | 1.1105s | 0.29963ms | 3 | 369.1ms | 1232 |
| F50 spin | 1.2055s | 0.35335ms | 3 | 445.0ms | 1259 |
| F100 spin | 1.3704s | 0.44310ms | 3 | 646.3ms | 1459 (+18.4%) |
| F50 sleep | 1.1827s | 0.32327ms | 3 | 418.5ms | 1295 |
| F100 sleep | 1.2340s | 0.36314ms | 3 | 507.5ms | 1398 (+13.5%) |
Wall monotonic + survives sleep (clean compute on T8 crit path — known). STALL COUNT dead-flat
at 3 across 0->+48% slowdown (both kinds). Injection symmetric (slows prefetch+on-demand alike).

### VERDICT (advisor-corrected): MIXED — existence STRUCTURAL, magnitude materially ENGINE-COUPLED, separability UNPROVEN
plans/lag-causality-verdict.md. First-draft "STRUCTURAL/engine-invariant, small residual" was
OVERSTATED + the design rested on a VENDOR MISREAD; both corrected after the disproof advisor.
- EXISTENCE structural: load-bearing signal has_nearest_le_start=0 (NOT the flat count — count is
  a saturated/low-N/wrong-direction proxy per rule 3). The 3 overshoot-tail cold-gets exist at F0,
  cost ~369ms cold-wait now, parent never-retained. Removing them is worth doing.
- MAGNITUDE engine-coupled: normalized wait/blk-busy +18.4% spin / +13.5% sleep (survives the
  frequency control) => ~24% of cold-wait growth under slowdown is genuine drift. The COST (wall)
  is partly engine-set; a faster engine shrinks the 318ms cost while the count stays 3.
- SEPARABILITY unproven: placement does NOT dissolve into engine (stalls cost wall at F0) but is
  NOT cleanly separable co-primary either. Both levers stay co-primary; engine still +13.7% (A.2).

### VENDOR PACING MAP [read-only subagent, plans/vendor-pacing-map.md]
rapidgzip's ~0-17ms pacing STRUCTURE (prefetch depth 2xpar, separate un-evictable in-flight map,
join-in-flight, lean off-path consumer post-proc) is FAITHFULLY PORTED in gzippy LINE-FOR-LINE
(block_fetcher.rs:737/758/66/536, chunk_fetcher.rs:528-529/1542/1561). The generic machinery is
NOT the defect; the SPECIFIC overshoot-reuse path is. Subagent's source-read leaned engine-induced
(the causal perturbation refined this to MIXED — perturbation overrules attribution per CLAUDE.md).

### RE-SCOPED DESIGN (advisor-corrected): block-finder OFFSET SUPPLY, NOT insert-relocation
- REFUTED (advisor, leader-verified first-hand): "move the block-finder insert POOL-side" was a
  vendor MISREAD. appendSubchunksToIndexes (the insert) runs IN-ORDER on the orchestrator
  (GzipChunkFetcher.hpp:357); queueChunkForPostProcessing (:554-582) pool-submits only applyWindow.
  gzippy ALREADY inserts in-order (consumer_append_subchunks_vendor chunk_fetcher.rs:1750/2790,
  citing :343-357). Pool-side move would DIVERGE from vendor (guardrail violation).
- CANDIDATE LEVER: the block-finder offset SUPPLY. Vendor GzipBlockFinder::get (:117-158) returns
  the CONFIRMED offset for a known index, partition GUESS only for unknown. gzippy's strategy offers
  ONLY the guess + never RE-OFFERS once prefetched (needs_confirmed_offset never fires) => the stale
  overshoot guess-prefetch is never superseded. Faithful port = make gzippy re-target the overshot
  index at the confirmed offset once the in-order insert records it.
- OPEN MECHANISM QUESTION (advisor-mandated, answer BEFORE any build): vendor inserts in-order with
  the SAME ~1-chunk lead the 3 prior gzippy attempts had — so HOW does vendor avoid the overshoot
  cold-get? Re-derive first-hand (block-finder re-target within existing look-ahead vs interior
  reuse vs less-overshoot). The prior mechanistic story was tied to the refuted pool-side premise.

### CHECKPOINT REACHED — STOP for supervisor ratification. NO placement port, NO engine build started.
Subagents (vendor map + disproof advisor) read-only, synchronous, collected in-turn, killed (no
orphans). Guest run held by a Bash task holding the ssh; host RESTORE VERIFIED. Supervisor to
ratify: (1) the MIXED verdict (existence structural, magnitude engine-coupled, separability
unproven); (2) that the placement lever survives but its standalone payoff is smaller/engine-
entangled; (3) authorize TIER-3 placement ONLY to re-derive the vendor lead mechanism (block-finder
offset supply) + gate on STALL probe(3->1)+no-flood A/B — NOT the insert-relocation, NOT the
cache-read getIndexedChunk. Engine front unchanged (co-primary, +13.7%, gated by §2.3 bench).

## PREFETCH-HORIZON vs SATURATION DIAGNOSIS [2026-06-07, prefetch-horizon leader, HEAD 85c67474]
CHARTER plans/prefetch-horizon-diagnosis.md. Answer the anti-escape-hatch question: are the
all-`decode_NOT_STARTED` head-of-line stalls WORKER SATURATION (engine) or PREFETCH-HORIZON
too shallow (structural)? Falsifier PRE-REGISTERED before any run: plans/prefetch-horizon-falsifier.md.
Guest verified idle before; host RESTORE VERIFIED after every run; no orphan procs (subagents killed).

### INSTRUMENT (commit 85c67474, byte-exact, env-gated) — STALL_OCCUPANCY_PROBE
At each non-startup cold-get stall (chunk_fetcher.rs:1357 None branch), snapshot worker occupancy
(thread_pool busy/idle_capacity via new idle_thread_count() accessor) + whether the stalled index
was enqueued (in-flight key covers decode_start). Classifies SAT (idle_cap==0) / HORIZON_NOT_ENQUEUED
(idle_cap>0 & not enqueued) / HORIZON_ENQUEUED_NOT_DONE. Gated on GZIPPY_STALL_RESIDENCY_PROBE
(OFF==identity); stall_residency mod now #[cfg(parallel_sm)] (was dead under default clippy).

### DATA (locked guest, silesia-large 503MB, T8, N=7-9, sha-verified, RUN_TRUSTWORTHY=true, diverged=0)
| combo | wall | SAT | HZ_NOT_ENQUEUED | mean_busy/8 | mean_idle_cap |
|---|---|---|---|---|---|
| F0 baseline | 1.125s | 1 | 2 | 5.3-6.0 | 2.0-2.67 |
| F100 spin | 1.413s | 1 | 2 | 5.67 | 2.33 |
| F100 sleep | 1.254s | 0 | 3 | 6.33 | 1.67 |
Residency (same stalls): NOT_RESIDENT=4 has_nearest_le_start=0 (never-retained / consumer-pace).
Source-cite (read-only subagent): horizon DEPTH is VENDOR-IDENTICAL (2·P candidate, P-1 concurrent,
same ramp + 1ms pump). block_fetcher.rs:737/763 ↔ BlockFetcher.hpp:467/474.

### VERDICT: NEITHER clean saturation NOR a confirmed horizon-DEPTH fix. (My first-draft "SATURATION→engine" was REFUTED by the disproof advisor; sustained.)
- SATURATION is DISPROVED: idle_capacity>0 at EVERY stall (1.67-2.67, never 0 even under 2x slow);
  a free worker existed; the on-demand decode submits onto it immediately (chunk_fetcher.rs:1450-1468).
- My slow_knob "decisive" cross-check was CONFOUNDED: engine-slow raises busy GLOBALLY regardless of
  stall cause (both hypotheses predict busy↑); the DISCRETE SAT bucket actually went 1→1→0 (falling)
  while HZ_NOT_ENQUEUED rose 2→3 — by my own pre-registered rule that is the HORIZON signature.
- I OVERRODE my own pre-registered map (HORIZON 3/3 rows) with a post-hoc continuous metric — the
  exact "attribution forecloses measurement" the falsifier forbids. Process violation, advisor-caught.
- BUT it is also NOT a confirmed horizon-DEPTH fix: a single snapshot can't tell NEVER-DISPATCHED
  (with idle cap = real scheduling/horizon gap) from DISPATCHED-DECODED-EVICTED-before-arrival
  (retention/anti-overrun, engine-lag-coupled). Both give NOT_RESIDENT+!enqueued. UNRUN discriminator.
- Engine is a genuine CO-lever but via engine-lag→cache-overrun→eviction (a corrected mechanism that
  REOPENS the placement/retention sub-question), NOT via "no free worker." NO engine redirect (that
  was the escape hatch). NO fix attempted (sub-cause unresolved). NO inline-ASM build started.

### OWED before deciding (advisor §6, the cheap unrun discriminator):
(i) per-stall NEVER-DISPATCHED vs DISPATCHED-THEN-EVICTED (track if a covering task was submitted
earlier + when evicted vs this arrival); (ii) split idle_capacity PARKED vs UNSPAWNED; (iii) N≫3
(lower split_chunk_size / aggregate dozens of stalls). Then saturation-vs-horizon is DECIDED, not
attributed. Distinct from the refuted offset-supply lever and from raw inner-loop speed.

### CHECKPOINT — STOP for supervisor gate. Full findings: plans/prefetch-horizon-findings.md;
advisor verdict: plans/prefetch-horizon-advisor-verdict.md. Subagents (source-cite + disproof
advisor) read-only/synchronous/killed (no orphans). Guest run from Bash tasks holding the ssh;
host RESTORE VERIFIED. Commit 85c67474 (probe) pushed to reimplement-isa-l. NO placement port,
NO engine work, NO inline-ASM build started.

---
## SUPERVISOR ROUTING 2026-06-07 (strategic advisor aa214edf): PIVOT TO ENGINE BENCH
After 3 placement diagnostic rounds (offset-supply refuted; saturation disproved; horizon-depth
vendor-identical), strategic advisor verdict = PIVOT-TO-ENGINE-BENCH. The engine is the
unbounded TIE-DETERMINING unknown (2.38× clean gap); placement is bounded +13.7% w/ diminishing
returns. NEXT = §2.3 engine isolation bench (the only legit ceiling-bound, isolation oracle +
ISA-L positive control), bundling 3 near-free riders: (a) re-read nearest_le_start @cap=256
(closes placement never-retained-vs-evicted); (b) same-sink production-output floor (bench can't
reach it; 0.61s was /dev/null); (c) bench self-test first. DROP "placement dissolves w/ faster
engine" as load-bearing (only ~24% coupled, cold-gets structural). PRE-REGISTER the plateau
falsifier before any engine build. See plans/engine-bench-authorization.md.

## ENGINE ISOLATION BENCH — BUILT + RUN (2026-06-07, engine-bench leader, HEAD 249f25b5)
Falsifier pre-registered: plans/engine-bench-falsifier.md. Bench: benches/engine_isolation.rs
(uncommitted; + Cargo.toml [[bench]] stanza + src/lib.rs one doc-hidden re-export
`isal_decompress_oracle` — measurement-only, decode graph untouched). Build combo
`--features pure-rust-inflate,isal-compression` on x86_64 = BOTH variant-(i) marker_inflate
clean loop (pure_inflate_decode) AND variant-(iii) ISA-L FFI in one binary; isal_clean_tail
stays OFF (gzippy-isal unset) so production routing is unaffected (build.rs:94-101, verified).

### GUEST RESULT (authoritative — native x86_64, freq-locked no_turbo=1/perf, single core 0,
### interleaved best-of-11, host RESTORE VERIFIED). Driver scripts/bench/run_engine_isolation.sh.
```
ENGINE_BENCH start_bit=302012944 N_bytes=4194304 (one clean mid-stream silesia chunk, 32KiB window)
VAR_I   scalar_u16  med=0.035545s  MBps_med=118  sigma=0.4%   (production marker_inflate::Block clean loop)
VAR_II  E1_u8(part) med=0.033563s  MBps_med=125  sigma=0.2%   (u8-direct sink; ring still u16 — bounds OUTPUT-traffic only)
VAR_III isal        med=0.010814s  MBps_med=388  sigma=6.9%   (ISA-L isal_inflate oracle, FFI measurement-only)
RATIO ii/i=1.059  iii/i=3.287  ii/iii=0.322
SHA_ALL_EQUAL=yes  CRC i=ii=iii=0xf24393b3  (BYTE-EXACT GATE PASSES every iter)
```
- BYTE-EXACT is solid: all three CRC identical over 4 MiB, sub-1% sigma on both Rust paths.
  `block.contains_marker_bytes()==false` asserted ⇒ variant (i) is a GENUINE clean decode of
  the production loop, no shortcut, not the markered arm. Bench genuinely isolates single-thread
  inner clean-decode compute (no scheduler/publish/marker machinery).
- (iii)/(i) = **3.287×** on the guest (LARGER than the prior campaign's 2.38× wall / ~2.1× s-t
  framing). This is the PURE single-thread clean inner-loop gap (one chunk, no pipeline). The
  prior 2.38× was clean-per-chunk-rate at the wall (92.7ms vs 39ms) which carries pipeline/drain;
  the isolated inner loop gap is wider.
- SELFTEST band [1.7,2.6] reads FAIL because 3.287 > 2.6 — but this is NOT a broken-instrument
  signal (byte-exact + clean + sub-1% sigma); it is ISA-L beating our scalar clean loop by MORE
  than the band. Band was a prior-number sanity range; the qualitative "ISA-L ~2-3× our scalar"
  reproduces and EXCEEDS it. (Advisor to adjudicate whether 3.29× indicates a real wider gap or a
  bench artifact — see checkpoint.)
- E1 (u8-direct OUTPUT) buys only +6% (ii/i=1.059). FINDING: output write-width is a SMALL
  component of the engine gap. The dominant ISA-L advantage is in the INNER HUFFMAN DECODE LOOP
  (multi-symbol packing / asm), not output traffic. E1-partial keeps the u16 ring; a full E1
  (u8 ring) would also cut ring traffic, but this bounds the output-side lever as minor.

### RIDER 1 — ANSWERED (from committed cap-swept evidence, commit d764734c): has_nearest_le_start=0
at ALL caps including cap=256 ⇒ NEVER-RETAINED / consumer-pace, NOT capacity-eviction. Interior-
reuse (getIndexedChunk) port has no resident parent at any cache size. Placement never-retained-vs-
evicted sub-question CLOSED.

### RIDER 2 — DONE (same-sink, real file sink on /dev/shm, T8 silesia-large, N=9, freq-locked,
### all sha OK, host RESTORE VERIFIED). Driver scripts/bench/run_same_sink_floor.sh.
```
gzippy_same_sink:    min=1.1915 med=1.2113 sd=1.3%
rapidgzip_same_sink: min=0.5960 med=0.6042 sd=1.8%
```
- **rapidgzip's SAME-SINK wall = 0.604s, NOT 0.54s.** The §3 "0.54s" was the /dev/null number;
  on a REAL file sink rapidgzip is 0.604s. The tie BAR moves UP to ~0.60s — slightly MORE
  favorable to gzippy's reachability than the design feared. (/dev/shm is RAM-backed; a true
  spinning-disk sink would lift both further — this is a fair apples-to-apples same-sink compare.)
- gzippy same-sink 1.21s ≈ 2.0× rapidgzip — consistent with the ~2.1× baseline.
- §3 contingency "same-sink floor ≤0.54s" RESOLVED: the comparison floor is rapidgzip's own
  0.60s same-sink; gzippy's non-decode serial floor (~0.015s, step-0) + same-sink output is well
  under that, so the consumer does NOT structurally forbid the tie at the corrected ~0.60s bar.

### PLATEAU-FALSIFIER VERDICT (pre-registered plans/engine-bench-falsifier.md):
- (ii) E1-partial = ii/iii = 0.322 (still ~3× slower than ISA-L). Per the pre-registered
  thresholds: (ii)/(iii)=0.322 ≤ 0.65 ⇒ E1-OUTPUT plateaus near the scalar pure ceiling, FAR
  below ISA-L, residual ≫ spread (sub-1% sigma). **E1 (output write-width) alone is NOT PROVEN**
  as the engine lever. BUT this is only the OUTPUT-traffic sub-lever; E2 (SIMD back-ref copy) /
  E3 (packed multi-literal store) / E4 (wide refill) — the INNER-LOOP techniques where the finding
  says the gap actually lives — were NOT benched (out of scope for the first subagent). So the
  PLATEAU verdict is: **E1 plateaus (output traffic is a minor +6% lever); the engine front is
  NOT YET PROVEN because the inner-loop techniques (E2-E4, the dominant gap) are unmeasured.**
  The bench is BUILT and the harness is proven (byte-exact, freq-locked, self-consistent) — it is
  ready to measure E2-E4 if the supervisor authorizes that next bench round.
### ADVISOR VERDICT (independent, read-only, disproof-driven — plans/engine-bench-advisor-verdict.md):
- (a) BENCH VALIDITY = **CORROBORATE-WITH-CAVEATS.** Variant (i) traced to the PRODUCTION LUT
  multi-cached/TRIPLE_SYM clean loop (marker_inflate.rs:968→981→1114→1191→1206), NOT a slow
  fallback. ISA-L oracle does the same work (raw deflate, same dict, same N via cap, no
  short-circuit) and is biased AGAINST itself (zeroes a 4MiB buffer INSIDE the timed region,
  isal_decompress.rs:372 ⇒ the gap is CONSERVATIVE). Byte gate is full memcmp (bench:245-247),
  not just CRC. The bench is a TRUSTWORTHY instrument for what it measures.
- (b) THE 3.29× HEADLINE = **REFUTED → corrected to ~3.10×.** Variant (i) carries a u16-sink
  re-widen tax (Vec<u16> uses the DEFAULT push_clean_u8) that PRODUCTION DOES NOT PAY —
  production uses the u8-direct clean-tail sink (gzip_chunk.rs:842), which variant (ii) mirrors.
  That tax is only ~6%. So the production-representative ceiling = **(iii)/(ii) ≈ 3.10×**, not
  (iii)/(i)=3.29×. ~3.1× is PLAUSIBLE for ISA-L-AVX2-vs-scalar-u16 (libdeflate/ISA-L beat naive
  scalar 2-4×), NOT a broken-instrument flag. It is LARGER than the prior 2.38× because the
  DENOMINATORS DIFFER: 2.38× was gzippy-clean vs rapidgzip-SYSTEM per-chunk; 3.1× is
  gzippy-current-clean-loop vs PURE-ISA-L-clean (purer denominator ⇒ larger ratio). Do NOT
  conflate them. The SELFTEST "FAIL" is a MIS-CALIBRATED band (set from the 2.38× system number),
  not a broken instrument — recalibrate to ~[2.5×,3.6×] and re-pass (rule 4).
- (c) PLATEAU / NOT-PROVEN = **CORROBORATE (sharpened).** Cannot reject (rule 3/7: E1 was never
  the closing lever; the u16 ring is still u16 so E2-E4 untested) and cannot declare proven. The
  bench built only E1-partial, but the pre-registered falsifier is defined over (ii) AFTER E2-E4.
- (d) RIDER-2 FLOOR = **CORROBORATE-WITH-CAVEATS.** /dev/shm is a legit same-sink ratio proxy
  (neither tool fsyncs); bar moves 0.54→0.604s (favorable) but gzippy at 2.0× doesn't change the
  verdict; the ~2× same-sink wall is consistent with the ~3.1× isolated inner-loop gap.

### THREE OWED ITEMS before this number can authorize the multi-week engine build (advisor):
1. Re-quote the ceiling as **(iii)/(ii) ≈ 3.10×** (production-representative), not 3.29×. [DONE here.]
2. **Sweep ≥3-5 chunks** (text + binary regions) and report the range — one chunk is suggestive,
   not a ceiling bound.
3. **Build the E2-E4 prototype in variant (ii)** and re-run — the pre-registered falsifier
   adjudicates (ii)-AFTER-E2-E4, which were NOT built this round.

### CHECKPOINT — STOP for supervisor gate. NET: the engine isolation bench is BUILT, freq-locked-
### guest-validated, byte-exact, and advisor-confirmed TRUSTWORTHY. The PRODUCTION-REPRESENTATIVE
### single-thread clean inner-loop ceiling = **gzippy ~3.1× slower than pure ISA-L** (118-125 MB/s
### vs 388 MB/s). E1 (output write-width) plateaus at +6% — output traffic is a MINOR lever; the
### gap lives in the INNER HUFFMAN LOOP (E2 SIMD copy / E3 packed store / E4 wide refill), which
### remain UNMEASURED. PLATEAU FALSIFIER: NOT-PROVEN-YET (E2-E4 unbenched), NOT refuted. Same-sink
### tie bar corrected up to ~0.60s. Riders both answered. Per the charter: DO NOT start the
### multi-week production engine integration/build. The decision the supervisor gates: authorize a
### SECOND bench round (sweep chunks + prototype E2-E4 in-bench) to settle the plateau falsifier,
### OR re-confront placement. No production code was changed (bench files uncommitted; lib.rs has a
### measurement-only re-export). HEAD unchanged at 249f25b5.

## INNER HUFFMAN KERNEL — speculative-pipeline isolation bench [2026-06-07, leader fresh instance]
Charter: plans/inner-huffman-kernel.md. State (causal, advisor-upheld): faithful u8 LANDED
byte-exact (fa9fd73c), wall TIE, gap constant ~1.70x at T1+T8 => binder is per-symbol Huffman
LUT-decode COMPUTE. Traffic/ring/placement all SLACK.

### SOURCE-VERIFIED FIRST-HAND (before any code):
- igzip trick #1 (packed-flat-short-code table, one u32 retires up to 3 literals + bit-length)
  is ALREADY BUILT + PRODUCTION-LIVE: lut_huffman.rs make_inflate_huff_code_lit_len builds with
  TRIPLE_SYM_FLAG=0 (=> triples ENABLED); LutLitLenCode::rebuild_from (lut_huffman.rs:998) passes
  it; the production clean loop read_internal_compressed_specialized<false> (marker_inflate.rs:1475)
  decodes via lut_litlen_decode -> (symbol, sym_count, bit_count) and unpacks up to 3 packed
  literals (marker_inflate.rs:1492-1524). So trick #1 is NOT the remaining lever.
- igzip trick #2 (speculative software-pipelined loop) is NOT present in production: the clean
  loop (a) stores ONE u8 per inner iter with `pos % U8_RING_SIZE` (marker_inflate.rs:1504), NOT an
  unconditional 8-byte packed store advanced by actual count; (b) does NOT preload next lit/len +
  dist before knowing current symbol type; (c) refills every outer iter. THIS is the remaining
  lever: the speculative pipeline (igzip asm:507-627) on the flat-u8 path.

### PRE-REGISTERED FALSIFIER (set BEFORE building/measuring):
- BUILD: a new flat-u8 bench variant in benches/engine_isolation.rs (VAR_V) that decodes the SAME
  clean silesia chunk via the EXISTING packed-flat-short-code table (LutLitLenCode/LutDistCode,
  trick #1) + a NEW speculative software-pipelined loop (trick #2): unconditional 8-byte packed-
  literal store into a flat linear u8 buffer advanced by actual sym_count, slop-margin headroom
  guard (no per-symbol bounds check inside the fast region), preload of next litlen+dist, branchless
  multi-literal output, MOVDQU/word back-ref copy on the flat buffer. Pure-Rust first; inline-asm
  only where Rust codegen demonstrably lags (measured, not assumed).
- BYTE-EXACT GATE (absolute): VAR_V SHA-equal vs VAR_I scalar AND vs VAR_III ISA-L over the swept
  clean chunks, all 5. A wrong-bytes oracle is VOID.
- SELF-TEST: VAR_III/VAR_I (iii/i) in [2.5,3.6] on the guest (existing band) must still hold —
  validates the ISA-L oracle denominator.
- PASS  = VAR_V clean MB/s (guest, median-of-per-chunk-medians) gives (V)/(iii ISA-L) >= 0.85,
  i.e. via tier1-design v2 §3 (anchored 92.7ms<->104MB/s, 39 chunks, ramp 1.36, T8) projects
  decode_wall that RE-BINDS off decode (<= 0.604s same-sink bar + spread), closing the ~1.70x.
- PLATEAU = (V)/(iii) stays ~pure class (<= 0.65, the round-2 plateau line) with residual to 0.85
  > inter-chunk spread (round-2 sd ~0.036). Per the standing falsifier this re-opens the FFI/bar
  fork (is igzip-class reachable in pure-Rust+asm AT ALL once BOTH tricks are present).
- Numbers ONLY from the locked guest (interleaved, taskset-pinned, sha-verified). Rosetta gives
  byte-exact validation but NOT authoritative MB/s.

### RESULT — AUTHORITATIVE GUEST (freq-locked, AVX2 live, SHA_ALL_EQUAL=yes, N=11 interleaved, taskset core0)
Host driver run_engine_isolation.sh -> host_lock_and_bench.sh (no_turbo pin + RESTORE VERIFIED).
ALL 8 variants BYTE-EXACT on all 5 swept clean silesia chunks (SHA_ALL_EQUAL=yes incl VAR_V; the
VAR_IV/VAR_V Rosetta VOIDs during local iteration were Rosetta-x86-64-v2 artifacts, NOT real —
the guest native run is byte-clean). Self-test PASS (iii/i=2.76, in [2.5,3.6]).

AGGREGATE (median-of-per-chunk-medians, MB/s):
  VAR_I scalar (production clean loop)              105   vs_isal 0.374
  VAR_IV_E234 (round-2 grafts, u16 ring)           116   vs_isal 0.413
  VAR_V packed-table + SPECULATIVE PIPELINE flat-u8 156   vs_isal 0.555  <-- THE LEVER
  VAR_III ISA-L oracle                             281   vs_isal 1.000
Per-chunk VAR_V vs_isal: 0.554 0.551 0.555 0.555 0.511 (median 0.554, sd ~0.018, tight).

VERDICT = PLATEAU (by the PRE-REGISTERED structural falsifier).
- PASS line (pre-registered) = (V)/(iii) >= 0.85 so decode RE-BINDS off the shared pipeline floor.
  Measured 0.554. Residual to 0.85 = 0.296 ~= 16 sd. FALSIFIER FIRES -> PLATEAU.
- BUT a MATERIAL ADVANCE over the round-2 plateau: 0.413 -> 0.554 (+34% over E234, +48% over
  scalar). The speculative software-pipelined loop (trick #2) on flat-u8 is the single biggest
  inner-loop gain measured this campaign. It confirms (a) trick #1 packed-table was ALREADY
  production-live (so the round-2 "grafts" weren't the table), and (b) the genuine remaining
  lever was trick #2 + flat-u8 linear buffer, which VAR_V isolates and DELIVERS — but to 0.55x,
  not igzip-class.
- §3 projection (anchored 92.7ms<->104MB/s, 39 chunks, ramp 1.36, T8): VAR_V projects decode_wall
  ~0.410s. This is NUMERICALLY < the 0.604s tie bar, AND now also < the ~0.54s placement-laden
  consumer floor -> at 0.55x ISA-L the decode term FALLS BELOW the floor, so decode would stop
  being the SOLE binder and the wall would RE-BIND on the ~0.54s floor. This is exactly the round-2
  "numeric pass is an artifact while the RATIO criterion governs" situation — the pre-registered
  RATIO criterion (0.85) GOVERNS -> PLATEAU. (The decode-falls-below-floor observation is a real
  shift worth the advisor + supervisor's attention: it suggests at VAR_V's rate the BINDER may move
  from engine to the placement/consumer floor — the co-primary from project_pregate.)
- Per the standing falsifier, PLATEAU re-opens the FFI/bar fork: igzip-class (0.85) is NOT reached
  in pure-Rust even with BOTH tricks (packed table + speculative pipeline) on flat-u8. Remaining
  gap to ISA-L (0.55->1.0) is plausibly the inline-ASM/BMI2-specific codegen + the unchecked
  over-read/write the safe-Rust headroom guard only partially exploits. Inline-asm is authorized
  but is the next, larger increment; this checkpoint settles the PURE-RUST ceiling of the lever.
- DECISION: do NOT integrate yet (pre-registered: integrate only on PASS). Report the achievable
  pure-Rust ceiling (0.55x) + the decode-falls-below-floor nuance to the supervisor. inline-asm
  spike + the placement co-primary are the two named next directions.
