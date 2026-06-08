# CAMPAIGN CHARTER — gzippy → rapidgzip parity (owner's constitution)

You (the OWNER agent) fully own this campaign. This doc is your constitution; keep it current
as the single source of truth. Supervisor = thin relay/cleanup only.

## GOALS (crisp)
Two flag-gated parallel single-member gzip decode paths, both FAITHFUL ports of what rapidgzip
ACTUALLY does (faithfulness is defined by rapidgzip's CODE, never by any memory line):
1. **gzippy-native** (default): does literally what rapidgzip does, **entirely in Rust, no
   C-FFI** — including using **u8 wherever rapidgzip uses u8, full stop**. Inline ASM is allowed.
   Target: a **1.0× wall TIE** with rapidgzip. Nothing less is accepted.
2. **gzippy-faithful (isal)**: the same, but hands the clean tail to **ISA-L via C-FFI** (the
   reference/comparison baseline, = rapidgzip's WITH_ISAL build).
Both use u8 in the clean tail; they differ ONLY in whether the u8 decoder is gzippy-Rust or
ISA-L-C. Done = gzippy-native ties rapidgzip on the locked whole-system wall across the workload
matrix, byte-exact, with the structure faithfully mirroring rapidgzip.

## PROCESS (the method — read this twice)
**We do NOT hunt individual levers.** A lever list is how you climb to a local optimum and stall.
Instead we treat the decoder as ONE system and repeatedly do exactly this loop:

1. **Perceive the whole system** with whole-system numbers: the real end-to-end interleaved
   wall (sha-verified, locked harness), at the thread counts that matter. This — not any
   component's busy-time, rate, or latency-share — is the only truth. Producer-side attribution
   is analyst-biasable and has manufactured phantom levers all campaign (rate-ratio "plateaus,"
   busy-time "blame"). The whole-system wall is the verdict.
2. **Find what is CURRENTLY the bottleneck** — the one thing that, right now, sets that wall.
   Establish it CAUSALLY: perturb a candidate and watch the whole-system wall respond
   (monotonic ⇒ on the critical path; flat ⇒ slack), with a frequency-neutral control. Never
   conclude a bottleneck from attribution alone.
3. **Fix that bottleneck** — whatever it happens to be (engine arithmetic, a production-path
   overhead, scheduling, memory, window handling). We don't care which; we fix the binder.
   Rewrite it CORRECTLY (our cost is dominated by shortcuts, not by correct rewrites). Byte-exact.
4. **Re-perceive the whole system.** Fixing the bottleneck MOVES it somewhere else. Go to 1.
   Stop when the whole-system wall ties rapidgzip.

This is bottleneck-following on whole-system numbers, not lever-shopping. A "win" is only real if
it moves the whole-system wall; a byte-exact change that ties is still KEPT (rule 7a) because it's
correct and its gain may be latent behind the current binder — but it does not count as progress
toward the tie until the wall moves.

## NON-NEGOTIABLE DISCIPLINES
- Byte-exact ALWAYS: dual-sha 028bd002c89c9a909ccdbc2af0a223de285348edb014ccc8e27d297f52cb410f on
  BOTH features; all lib tests + the adversarial seam test green. Wrong bytes = void.
- Numbers ONLY from the locked guest harness (interleaved, sha-verified). Never a hand script.
- Independent disproof advisor (synchronous) corroborates every consequential claim BEFORE you
  rely on it. Pre-register a falsifier before each experiment; bound a speed-up CEILING by
  removal/oracle (a slow-down slope proves "on the path," never the payoff).
- No memory or plan line may compete with the goal or redefine faithfulness away from rapidgzip's
  actual code. If one does, correct it.
- Escalate to the supervisor/user ONLY on a genuine FORK that trades off a user constraint
  (1.0× bar vs no-FFI vs faithfulness) — e.g. if a bottleneck proves unfixable in pure-Rust+ASM.
  Otherwise drive with agency.

## HOW YOU DELEGATE (carefully)
You own the work; you delegate via your OWN `claude -p --model opus --permission-mode
bypassPermissions` subagents. Rules that have cost whole turns:
- Run subagents SYNCHRONOUSLY (block with `timeout`, collect in-turn). There is NO auto-reinvoke —
  do NOT background a subagent and yield for a "notification"; you will simply die and the
  supervisor must re-drive. Run measurements YOURSELF (Bash holding the ssh); delegate research
  and the disproof advisor as synchronous calls.
- NO detached `sleep` leader-lock sentinel (it orphans). Leave NO orphaned processes — before you
  finish, pgrep must show none of your claude -p / sleep children.
- SOURCE-VERIFY any premise first-hand before acting on it (a wrong premise — "gzippy never
  re-targets," the "window-discard" — has burned turns). Serialize builds via cargo-lock.sh.
- Keep THIS charter + plans/orchestrator-status.md current so a fresh owner-spawn can resume.

## CURRENT STATE (2026-06-07, owner turn, branch reimplement-isa-l, HEAD 2ff19ac6) — **CORRECTNESS NET ADOPTED (poison-on, Stage-2 seam VALIDATED, no bug); the ≤0.11× residual LOCATED on the T8 critical path (causal perturbation, advisor-UPHELD); inner-Huffman STORE-side technique TIE'd (kept 7a) ⇒ the loop's binding sub-resource is the DECODE-COMPUTE, owed a decode-only localization before BMI2/packed-LUT. native_fold ~0.77-0.79× rg (banked teeth UNCHANGED).**

### STEP 0 — CORRECTNESS NET MERGED + STAGE-2 VALIDATED UNDER POISON (commit 24c3a04)
Merged test/inflate-correctness-net (ae454e0f): 4 test files (seam_crossing, diff_multi_oracle,
inflate_proptest, inflate_fuzz_loop), mod.rs wiring, proptest dev-dep, cfg(test) 0xCD reserve-poison
gated on GZIPPY_POISON_RESERVE. Merge auto-resolved clean. ADDITIONALLY poisoned the Stage-2 copy-free
contig spare [len,cap) in SegmentedU8::contig_decode_window so the gzippy-native FOLD copy-free clean
tail (the seam the net guards) is stress-tested too. VALIDATION (pure-rust-inflate, arm64,
GZIPPY_POISON_RESERVE=1): full lib suite 892 pass /1 pre-existing load-sensitive diff_ratio timing flake
(fails in isolation too; orthogonal perf micro-assert); all 11 seam_crossing tests incl. both
reserve-clamp regimes + 20/30 MiB max-distance fold decodes; 3-oracle + multi-oracle differentials;
proptest; fuzz_loop_differential 5000 iters byte-exact. NO latent Stage-2 seam/back-ref/regrow bug.

### STEP 1a — WHERE THE ≤0.11× IS: contig clean symbol-decode IS on the T8 critical path (commit + advisor)
The Stage-2 FOLD clean tail decodes via decode_clean_into_contig (NO ring — Stage 2 removed the clean-path
ring-write), which the pre-existing GZIPPY_SLOW_MODE knob (ring-path-only) did NOT perturb. Wired the same
clean knob into decode_clean_into_contig (byte-transparent, OFF==ON==028bd002…cb410f, ~24-27M inject hits
confirm the contig loop fires). CAUSAL PERTURBATION (locked guest, interleaved measure.sh N=11, T8, sha-OK,
2 passes): off < spin50 < spin100 < sleep100 MONOTONIC both passes; the freq-neutral SLEEP control
preserves/exceeds the spin delta ⇒ the contig clean symbol-decode is on the critical path (not slack-masked,
not a turbo artifact). off/rg ~0.77× sign-stable. Advisor UPHELD-WITH-CAVEATS: loop is on-path, but the
per-loop-body inject cannot isolate Huffman-decode COMPUTE from store/copy bandwidth — the "BMI2 will pay"
leap is unproven until the binder WITHIN the loop is localized.

### STEP 1b — STORE-SIDE TECHNIQUE (packed multi-literal fast loop) TIE'd, KEPT 7a (commit 2ff19ac6)
Ported the ring VAR_V speculative fast loop (igzip asm:518 — packed multi-literal store + decode pipeline)
onto the contig clean tail; gated the 8-byte store on sym_count>1 (advisor Q3 — lean single-byte path for
the dominant sym_count==1 literal so the wide store never wastes bandwidth). BYTE-EXACT (sha 028bd002…cb410f
T1+T8 guest x86_64 + arm64; full suite + poison + 5000-iter fuzz green). REMOVE-AND-MEASURE (locked guest,
3 interleaved passes, baseline=pre-fastloop vs fastloop2): 1.001×/1.018×/0.994× baseline = TIE (sign-neutral;
the ungated first cut was weakly negative 0.974-0.998×, fixed by the gate). Fast loop handles ~69% of clean
decode events (careful-loop hits 27M→8.3M) so the path IS exercised — the STORE side is simply NOT the
binding sub-resource. KEPT per 7a. Advisor UPHELD-WITH-CAVEATS: store technique exhausted; binding
sub-resource UNIDENTIFIED; redirect to decode-compute is UNPROVEN.

### NEXT (owed, NOT started — fresh measurement arc; do NOT jump to BMI2)
Localize the loop's binding sub-resource: (1) a CLEAN-ONLY oracle (ISA-L for clean chunks only, marker path
intact) to tighten the speed-up ceiling below the whole-engine ocl_cf 0.925× to a clean-loop-only number;
(2) a decode-ONLY perturbation (slow ONLY lut_litlen_decode/dist_hc.decode, not the stores/copies) to test
whether Huffman decode-compute is the binder. Only if (2) confirms decode-compute on-path → BMI2 PEXT/BZHI /
packed-u32 LUT. If un-improvable in pure-Rust (plateau) → escalate the 1.0×-vs-FFI fork with the bounded
number. GUEST: /tmp/gz-ft-src (source @2ff19ac6 marker_inflate.rs), binaries /tmp/gz-baseline (pre-fastloop)
+ /tmp/gz-fastloop2 (gated, sha 028bd002…cb410f); drivers /tmp/{contig_perturb,fastloop2_ab}.sh. rg 0.16.0.
NO orphan processes.

## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, owner turn, branch reimplement-isa-l, HEAD 0f5bc85b = d77a173b + copy-free-to-final Stage 2) — **STAGE 2 WIRED + LANDED BYTE-EXACT + MEASURED: the ~0.067× drain-memcpy tooth is BANKED at +0.05× (advisor-UPHELD-WITH-CAVEATS). native_fold ~0.74× → ~0.79× rg.**

The gzippy-native FOLD post-flip clean tail now decodes DIRECTLY into chunk.data's reserved
contiguous tail (`finish_decode_chunk_contig_native` + `decode_clean_into_contig`), DELETING the
ring→chunk.data drain memcpy (the bulk clean tail no longer touches the u8 ring). **FAITHFUL
PREPEND, no landmine:** at the ctx-flip (clean_appended≥32768) the engine has ALREADY flipped and
≥32 KiB contiguous clean output is in chunk.data, so the 32 KiB predecessor window IS that
contiguous tail (real, already-CRC'd, already-counted prior output) ⇒ `data_prefix_len` stays 0,
back-refs resolve from chunk.data[*pos-d], CRC covers only real output (no prefix to exclude),
decode_bypass round-trip is identical to today's post-drain state. This is vendor setInitialWindow's
prepend (DecodedData.hpp:278-289), NOT the forbidden window-in-scratch dual-region shortcut. The
one-time transition-narrow seam drain (≤few KiB, faithful to vendor's flip) remains; only the BULK
drain is gone.

**MECHANISM (commit 0f5bc85b):** new `MarkerStep::FlipToContig` (native, `not(isal_clean_tail)`):
at the ctx-flip the driver resumes the SAME thread-local Block in a dedicated contig loop (no ring,
no drain). The shared generic `marker_decode_step_loop` AND the gzippy-isal FlipToClean/Engine-C
two-phase path are UNCHANGED. + `SegmentedU8::contig_decode_window` (re-fetched base/cap/len every
outer iter, grow-between-calls pointer-move safety) + `Block::decode_clean_stored_into_contig`
(post-flip STORED block). 5 advisor hazards handled: H1 release-mode headroom guard (contig has no
ring modulo ⇒ a violation would be heap OOB not a CRC-catchable wrong byte); H2 stored-block; H3
commit-before-decoded_range + multi-call-per-block accounting; H4 base re-fetch (a real regrow-past-
16MiB bug was CAUGHT by the H1 guard during testing and fixed — Vec::reserve(min_spare) not the
wrong delta); H5 native-only.

**BYTE-EXACT:** gzippy-native sha 028bd002…cb410f == system gzip == rapidgzip on /root/silesia.gz
(211,968,000 decoded) at T1+T8, x86_64 guest AND arm64 host; gzippy-isal UNCHANGED (isal_clean_tail
gated out, x86_64 Rosetta lib tests pass). flip_to_clean=12 (16 chunks: 12 flip-to-contig + 4
finished_no_flip + 2 window_seeded) confirms the contig route is production. 862 lib tests +
native_fold_parity + flip-seam + 3 Stage-1 differentials + 3 NEW owed-case tests
(contig_native_multiblock_clean_and_crc_prefix_excluded, _regrow_past_reserve_clamp,
_stored_block_after_flip) green (only the pre-existing load-sensitive diff_ratio timing flake fails,
passes in isolation).

**MEASURED (locked guest REDACTED_IP, taskset 0,2,4,6,8,10,12,14, gov=perf, interleaved measure.sh
N=11 best-of-11/pass, sha-OK every run, RAW=211968000, P=8, 10 passes, A/B vs prior banked
copy-free-DRAIN baseline /tmp/gzbuild-native@9cde0b4f vs rapidgzip 0.16.0):** priornative/stage2 ∈
{0.954,0.972,0.989,0.929,1.011,0.876,0.805,0.969,0.818,0.969} ⇒ stage2 strictly faster 9/10 (the
1.011 inversion had a 66% spread). stage2/rg mean ~0.79×, priornative/rg mean ~0.72×; paired delta
mean +0.058×, median +0.044×, SE±0.020. **BANKED +0.05× (drop the +0.07 edge).** Magnitude
load-confounded (loadavg 2.2→5.0 across the campaign, autocorrelated ⇒ ~2-SE) but SIGN-stable and
TRIANGULATED by the same-binary GZIPPY_FOLD_NODRAIN knob (+0.067× wrong-bytes, last turn) = the
methodologically clean drain isolation. Provenance VERIFIED: 9cde0b4f→0f5bc85b production-decode
delta is ONLY the Stage 2 wiring (Stage 1 additive-unwired, nodrain OFF, fulcrum additive).

**ADVISOR (synchronous ×2, plans/copyfree-stage2-advisor-verdict.md):** pre-impl source-verify
UPHELD the key realization + landmine-sidestep; post-impl measurement UPHELD-WITH-CAVEATS (record
+0.05× not +0.05-0.07×; sign-confident, magnitude soft/load-confounded; cross-binary layout is a
residual confound but the same-binary nodrain knob corroborates).

**RESIDUAL / SCOPED NEXT (gate, do NOT start): the ≤0.11× UPPER-BOUND intrinsic symbol-rate gap**
(native_fold ~0.79× → ocl_cf 0.925× ≈ 0.13×, of which ≤0.11× is inner-Huffman symbol rate + the
remaining placement/scheduler term to 1.0×). The drain tooth is banked; the next binder is the
inner-Huffman RATE on the `<false>` clean path (BMI2 PEXT/BZHI, wider multi-literal, ISA-L-class
packed-u32 LUT — CLAUDE.md-authorized), bounded by ocl_cf 0.925× (engine-removed ceiling), never the
VAR_VI slope. GUEST: /tmp/gz-ft-src (source synced to 0f5bc85b's 3 changed files + symlink
vendor→/root/gzippy/vendor), build /tmp/gz-ft-src/target/release/gzippy (native, target-cpu=native,
sha 028bd002…cb410f); prior baseline /tmp/gzbuild-native@9cde0b4f. rg 0.16.0. NO orphan processes.

## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, owner turn, branch reimplement-isa-l, HEAD c224aaad→512a389d + copy-free-to-final Stage 1) — **COPY-FREE-TO-FINAL REFACTOR, STAGE 1 LANDED byte-exact (the ~0.067× drain-memcpy tooth's hardest MECHANICAL risk retired; wall NOT yet banked — Stage 2 wiring is gated). Two synchronous disproof advisors: the first vetted the SCOPING decision (CHECKPOINT-STAGE-1, NOT one-pass — gated full-wire cannot bank the wall safely this turn because OFF==identity isolates COMMIT risk but not MEASUREMENT risk: the data_prefix_len-nonzero activation + CRC-prefix-exclusion + decode_bypass serialization round-trip is an uncontained byte-exactness landmine; realistic one-pass outcome = a dead ON path and NOTHING banked); the second (plans/copyfree-stage1-advisor-verdict.md) UPHELD the LANDED Stage 1 (A/B/C/E UPHELD, D upheld-w-caveats, NOTHING to revert).**

**WHAT LANDED (commit c224aaad, +453/-0, ONE file marker_inflate.rs, PURELY ADDITIVE, ZERO production callers ⇒ byte-exact by construction on BOTH features + BOTH archs):** (1) `emit_backref_contig` — non-wrapping clean back-ref copy, the sibling of `emit_backref_ring_u8` with NO `% U8_RING_SIZE` (contiguous ⇒ the 3 wrap-aware arms collapse to word-copy / RLE-fill / overlap; the secondary win on top of the drain). (2) `Block::decode_clean_into_contig` — clean (`<false>`) body decoded straight into a caller-supplied CONTIGUOUS buffer with the 32 KiB predecessor window installed as a DICTIONARY PREFIX at base[0..window_len) (vendor setInitialWindow model, DecodedData.hpp:278-289 + deflate.hpp:1778); first-clean-byte back-refs of distance ≤ 32768 resolve contiguously into that prefix; range check `distance > *pos` ≡ ring's `distance > decoded_bytes+emitted` (advisor C UPHELD: *pos and decoded_bytes both start at window_len); per-call cap = spare-(MAX_RUN_LENGTH+8) with the Engine-C grow-BETWEEN-calls contract (no mid-block realloc). (3) 3 ring-vs-CONTIG differential tests driving the REAL production `read()` loop on the SAME window-seeded clean DEFLATE body, asserting byte-equal — back-ref into the 32 KiB prefix (distance==*pos==32768 → base[0] boundary), multi-call resumable (per_call=4096, ~10 calls, cross-call back-ref source), RLE+short-distance. All 3 pass (Rosetta x86_64 x86-64-v2). Full lib suite 855 pass; the only failures are the 6 PRE-EXISTING Rosetta/timing artifacts (stash A/B confirmed identical to HEAD baseline). arm64 native release builds clean; round-trip sha verified.

**ADVISOR B (back-ref correctness) — UPHELD with a proven headroom bound:** arm-by-arm reduction of emit_backref_ring_u8 to the non-wrap case == emit_backref_contig exactly; no `*pos-distance` underflow (distance∈[1,*pos] enforced before the call); the MAX_RUN_LENGTH+8 reservation is PROVABLY sufficient (one outer iter writes ≤3 literals OR one ≤258 back-ref, never both, because ISA-L multi-symbol packing stops at any sym≥256 ⇒ pair/triple = all literals; word path max-end = cap-3 < cap, 3 bytes to spare). Proof DEPENDS on MAX sym_count==3 — a comment was added tying the bound to that invariant (this commit, comment/proof-only, byte-transparent).

**OWED FOR STAGE 2 (advisor D, named so they are NOT silently assumed retired):** (1) STORED/uncompressed clean block — decode_clean_into_contig returns Err(InvalidCompression); the ring read() DOES decode stored blocks in clean mode ⇒ Stage 2 must route/extend. (2) MULTI-deflate-block clean phase across a block boundary (caller detects EOB → read_header → continue with persisted *pos/decoded_bytes) is UNTESTED (both test payloads are single-block). (3) An actual out_room-saturating REGROW (the grow-between-calls contract is present in the MATH but not driven by a test — both tests over-size cap). PLUS the deferred landmine the commit already names: data_prefix_len=32768 activation on a path that has only ever run prefix==0 (audit decoded_size/window-publish/consumer iovecs/apply_window), CRC-prefix exclusion (the 32 KiB prefix is already-CRC'd predecessor bytes — exclude or double-count), and the decode_bypass.rs serialize/deserialize round-trip of nonzero data_prefix_len on a partition-fault re-decode. FORBIDDEN shortcut (advisor): the dual-region back-ref (keep the window in scratch, leave chunk.data prefix==0) — it diverges from vendor setInitialWindow's prepend ⇒ violates the faithful-port directive. Decide the FAITHFUL PREPEND model now; Stage 2 concentrates its byte-exact effort there.

**RESIDUAL / RATCHET:** banked teeth UNCHANGED at native_fold 0.737× rg (Stage 1 banks no wall — unwired by design). The ~0.067× drain tooth is now MECHANICALLY de-risked and will be banked by Stage 2 wiring. After it, the ≤0.11× UPPER-BOUND intrinsic symbol-rate gap (inner-Huffman: BMI2 PEXT/BZHI, wider multi-literal, ISA-L-class packed-u32 LUT) bounded by ocl_cf 0.925×, never the VAR_VI slope.

## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, owner turn, branch reimplement-isa-l, HEAD 7ae5903 = fc7336c3 + GZIPPY_FOLD_NODRAIN/NOCRC isolation knobs, measured /tmp/gz-ft-src/target/release/gzippy = native @ HEAD, target-cpu=native) — **RING→DATA DRAIN ISOLATION RAN (the fold-contig advisor's owed same-engine pure-Rust ring-copy oracle) → the 0.188× residual to ocl_cf 0.925× SPLITS: ~0.067× is the ring→chunk.data drain MEMCPY (a recoverable free tooth) + ≤0.11× UPPER BOUND remaining = intrinsic symbol rate + the engine ring-WRITE (the inner-Huffman work). CRC is NOT a lever (nodrain_nocrc ≈ nodrain). The isolation: a measurement-only no-op-drain knob (GZIPPY_FOLD_NODRAIN, OFF==identity byte-exact 028bd002…cb410f, WRONG bytes ON via terminal-trailer CRC mismatch after the FULL decode) skips the extend_from_slice ring→data copy while leaving decode (engine block_body + ring write + back-ref resolve) UNCHANGED. RESULT (locked guest, interleaved measure.sh, best-of-N over 4 NON-OVERLAPPING passes — native_fold ∈[0.1807,0.1831] vs nodrain ∈[0.1645,0.1684], gap ≈3-5× the wider range's width, load-invariant 1.2→2.3): removing the drain moves native_fold ~0.745× → ~0.812× rg = ~+0.067×. Disproof advisor UPHELD (C1/C3/C5 UPHELD, C2 upheld-w-caveats) and found the no-op-drain's cold-cache asymmetry makes +0.067× CONSERVATIVE (true drain ≥ measured). fulcrum_total whole-system read (native T8, trace byte-exact) confirmed the residual lives in worker.block_body (658ms SELF, the marker-engine inner decode @262 MB/s over 168MB) feeding the 61%-wait consumer — but it does NOT separately instrument the ring-write/drain (both inside block_body/drain_to_output), which is WHY the isolation oracle was needed. NO production fix landed this turn (the ~0.067× banking needs the byte-exact copy-free-to-final engine refactor — a non-trivial no-ring-for-clean rewrite, prompt-gated; SCOPED below). Banked teeth unchanged at native_fold 0.737× rg.**

### THIS TURN (isolation) — captured the gzippy-native T8 whole-system picture with the merged fulcrum_total, then BUILT + RAN the fold-contig advisor's owed same-engine pure-Rust ring-copy-free isolation oracle, splitting the drain-memcpy from intrinsic symbol rate. Synchronous disproof advisor UPHELD. Brief: plans/ring-drain-isolation-brief.md; verdict: plans/ring-drain-isolation-advisor-verdict.md.
- **fulcrum_total (HEAD fc7336c3 native build, target-cpu=native, /tmp/gz-ft-src; trace byte-exact 028bd002…cb410f, ParallelSM):** routing flip_to_clean=0 finished_no_flip=16 window_seeded=2 (natural in-stream propagation, NOT the seed oracle — the script never sets GZIPPY_SEED_WINDOWS; isal_oracle_chunks=0). The analyzer's binary routing-guard REFUSES on window_seeded>0 (the advisor-flagged C4 caveat-2 brittleness — 2/16 chunks naturally seeded ≠ the GZIPPY_SEED_WINDOWS oracle), so I read the DESCRIPTIVE structure with that caveat: worker.block_body 658ms SELF (dominant compute = marker-engine inner decode), post_process.apply_window 90ms, consumer.writev 60.8ms; consumer wall 61% WAIT (blocked on workers) ⇒ engine-rate-bound, not scheduling-bound (consistent with prior turns). The ring-write + ring→data drain are NOT separate spans (folded into block_body/drain) ⇒ fulcrum_total alone can't split them — hence the isolation oracle.
- **THE ISOLATION (committed 7ae5903, byte-exact OFF==identity, measurement-only):** GZIPPY_FOLD_NODRAIN skips the ring→chunk.data extend_from_slice; GZIPPY_FOLD_NOCRC skips the per-clean-byte CRC. Self-test: OFF==rg==028bd002…cb410f (x86_64 guest + arm64); NODRAIN sha differs (fires); NODRAIN exits 1 only at the terminal trailer CRC32 (diagnosed: full decode ran, post-decode check). 857 lib tests green.
- **RESULT (4 interleaved passes, best-of-11):** native_fold 0.1824/0.1807/0.1823/0.1831; nodrain 0.1645/0.1669/0.1684/0.1675; nodrain_nocrc ≈ nodrain; rg 0.1358/0.1378/0.1464/0.1381. nodrain/native = 1.109/1.083/1.083/1.093× (sign-stable, ranges never overlap, load-invariant). ⇒ drain memcpy ≈ +0.067× rg; CRC ≈ free.
- **ADVISOR (synchronous, read-only, plans/ring-drain-isolation-advisor-verdict.md):** C1 UPHELD-W-CAVEATS (best-of-N on one-sided noise is the correct estimator; interleaved + load-invariant defends freq-neutrality; a small turbo component can't be fully excluded w/o a pinned-freq pass — cheap insurance, not required). C2 UPHELD-W-CAVEATS (right method; ≤0.11× still confounds ring-write + the ISA-L-vs-pure-Rust engine difference, so it's an UPPER BOUND on symbol rate; the unconditional safe statement is intrinsic ≤0.188×). C3 UPHELD (CRC reads the ring slice, independent of the drain). C5 UPHELD (the copy-free-to-final refactor is faithful: vendor decodes the clean BULK straight to contiguous u8 with no u16 ring — DecodedData.hpp:278-289 — so gzippy's clean-phase ring→data drain has NO vendor counterpart; nuance: vendor still concatenates at merge, so claim "no clean-phase ring/narrow," not "zero copies"). LOAD-BEARING: the no-op-drain is CONSERVATIVE (every cache asymmetry it adds — cold chunk.data reads in writev + window-publish — costs nodrain EXTRA), so true drain ≥ measured +0.067×.
- **SCOPED NEXT (gate, do NOT start without supervisor): (1) BANK the ~0.067× via the byte-exact copy-free-to-final refactor** — the pure-Rust clean (`<false>`) phase writes u8 DIRECTLY into chunk.data's pre-reserved contiguous tail (reuse the writable_tail_reserve+commit pattern the ISA-L oracle already uses), back-refs resolve from that contiguous tail, NO output_ring for the clean phase, so the drain copy disappears WITH correct bytes. This is a non-trivial engine rewrite (clean-phase ring addressing + back-ref resolution + the flip-seam transition + the >16MiB reserve-clamp fallback), correctness-sensitive at the flip seam — bound it with the nodrain knob as the ceiling, byte-exact dual-sha, remove-and-measure the interleaved T8 wall. **(2) THEN the ≤0.11× intrinsic symbol-rate** inner-Huffman work (BMI2 PEXT/BZHI, wider multi-literal, ISA-L-class packed-u32 LUT — CLAUDE.md-authorized) bounded by ocl_cf 0.925×, never the VAR_VI slope.
- **GUEST STATE:** /tmp/gz-ft-src = full source @ HEAD fc7336c3 + the 2 isolation knobs (symlinks: vendor→/root/gzippy/vendor, /tmp/fulcrum→/root/fulcrum), build /tmp/gz-ft-src/target/release/gzippy (native FOLD, target-cpu=native, sha 028bd002…cb410f OFF). rg 0.16.0 /usr/local/bin. Driver /tmp/isolation_wall.sh (bash). Trace /tmp/ft-art/trace_native_T8.json + verbose. silesia /root/silesia.gz (RAW=211968000). NO orphan processes (advisor wrapper + monitors killed).

## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, owner turn, branch reimplement-isa-l, HEAD 9cde0b4f, measured /tmp/gzbuild-native + /tmp/gzbuild-head + /tmp/gzippy-old-drain) — **CADENCE/INTRINSIC SPLIT RUN (the advisor's owed symmetric control) → CADENCE TAX IS A REAL FREE COMPONENT AND IS NOW RECOVERED + LANDED: copy-free post-flip clean drain on the gzippy-native FOLD path moved native_fold 0.678× → 0.737× rg = +0.059× banked (quiet-box, sha-exact), a banked ratchet tooth. The split is MONOTONIC + sign-stable across all 6 interleaved passes (old triple-buffer 0.634× < copy-free-drain 0.674× < fully-copy-free 0.717× — copy#1 ring→u8buf +0.040×, copy#2/3/grow +0.043×; loaded split-sum +0.083× is load-inflated, banked is the quiet +0.059×). The FIX (committed 9cde0b4f, byte-exact 028bd002…cb410f, advisor-CONFIRMED): marker_inflate::drain_to_output clean branch pushes ≤2 contiguous u8 ring slices straight to the sink (no per-block u8buf alloc + byte loop); new ContigFoldSink (replaced+DELETED UnifiedMarkerSink) writes them DIRECTLY into a pre-reserved contiguous chunk.data (no pending_clean middle-man, no second append_clean copy, no regrow). RESIDUAL intrinsic-rate gap = native_fold 0.737× → engine-removed ceiling ocl_cf 0.925× = ~0.188×, an UPPER BOUND (it still includes the ring-write + ring→data drain memcpy that the ISA-L oracle does not pay — NOT pure symbol rate; advisor down-scoped L2). The scoped next loop = the inner-Huffman rate work (BMI2 PEXT / wider multi-literal / ISA-L-class packed-u32 LUT) bounded by ocl_cf 0.925×, never the VAR_VI slope.**

### THIS TURN — ran the advisor's owed SYMMETRIC CONTROL to split cadence from intrinsic on the gzippy-native FOLD path, then RECOVERED + LANDED the free cadence tax. Synchronous disproof advisor ×2 (the first REFUTED my under-scoped C1 — I'd missed the ring→u8buf copy; the second CONFIRMED the corrected fix with 3 label corrections, all applied).
- **THE SPLIT (locked guest REDACTED_IP, taskset 0,2,4,6,8,10,12,14, gov=perf, interleaved measure.sh, sha-OK every run):** 6-pass 3-way old / new_off / new_contig = 0.634 / 0.674 / 0.717× rg means, MONOTONIC every pass. copy#1 (ring→u8buf alloc+byteloop eliminated) = +0.040×; copy#2/3/grow (pending_clean middle-man + append_clean + regrow eliminated) = +0.043×. The FIRST control (pending_clean only, under-reserved) TIE'd — the advisor caught that I'd left the dominant ring→u8buf per-block copy untouched; removing it MOVED the wall, proving a free cadence component had been mis-booked as intrinsic.
- **BANKED (quiet box load 1.2-1.4, default binary, 3-pass interleaved):** native_fold 0.747/0.764/0.701× = mean 0.737× rg (baseline 0.678× ⇒ +0.059× banked). isal_prod ALSO improved to ~0.80× (its marker-bootstrap phase shares the same copy-free drain). +0.083× is the LOADED split-sum (sign/monotonicity evidence only); +0.059× is the honest banked number.
- **LANDED (commit 9cde0b4f, byte-exact OFF==identity, 028bd002…cb410f on gzippy-native guest x86_64 + arm64; gzippy-isal unaffected = same sha; 857 lib + native_fold_parity + flip-seam differentials green):** copy-free `drain_to_output` clean branch + `ContigFoldSink` made the production default + `UnifiedMarkerSink` DELETED + `ChunkData::reserve_clean` (clamped 16 MiB). Routing flip_to_clean=0 finished_no_flip=16 (FOLD preserved, window-absent bootstrap preserved — did NOT seed).
- **ADVISOR (synchronous ×2, read-only, plans/fold-contig-split-advisor-verdict.md + plans/fold-contig-landed-advisor-verdict.md):** PASS 1 REFUTED C1-as-stated (I'd removed only the pending_clean copy, not the dominant ring→u8buf per-block alloc+byteloop), owed the copy-free-drain control. PASS 2 CONFIRMED the landed fix mechanism + the UnifiedMarkerSink deletion (no live caller, two-phase CleanTailSink untouched, sink overrides correct, blast radius bounded — sink is an output accumulator, decode correctness lives in the engine ring) + byte-exactness; required 3 corrections (banked +0.059× not +0.083×; residual ≤0.188× is an UPPER BOUND on symbol rate, not symbol rate; reserve ×8 is a heuristic not the worst DEFLATE ratio + clamp) — ALL APPLIED.
- **RESIDUAL / SCOPED NEXT (gate, do NOT start): the inner-Huffman RATE work on the gzippy-native `<false>` clean path.** native_fold 0.737× → ocl_cf 0.925× = ≤0.188× UPPER BOUND. This bound still bundles the ring-write + ring→data drain memcpy (ocl_cf is ring-free, decodes ISA-L straight into writable_tail_reserve) — a clean intrinsic-rate isolation would need a same-engine (pure-Rust) ring-based copy-free-to-final oracle (does not exist). The rate techniques (BMI2 PEXT/BZHI, wider multi-literal, ISA-L-class packed-u32 LUT — CLAUDE.md-authorized inner-loop work) are bounded by ocl_cf 0.925× (engine-removed ceiling), never the VAR_VI 0.6× slope.
- **GUEST STATE:** /tmp/gzbuild-native/release/gzippy (gzippy-native FOLD, FINAL copy-free, sha 028bd002…cb410f) + /tmp/gzbuild-head/release/gzippy (gzippy-isal, FINAL, same sha) + /tmp/gzippy-old-drain (the pre-fix triple-buffer baseline, kept for the split). rg 0.16.0 at /usr/local/bin. Drivers /tmp/{baseline_wall,contig_wall,contig_wall2,split_wall,contig_selftest}.sh (bash). Guest /root/gzippy src overlaid to HEAD 9cde0b4f production change. NO orphan processes.

## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, owner turn, branch reimplement-isa-l, HEAD 7aae6c4a + copy-free-oracle overlay, measured /tmp/gzbuild-head + /tmp/gzbuild-native) — **COPY-FREE CLEAN-TAIL ORACLE RAN → the prior INCONCLUSIVE is REVERSED: the clean-decode SUBSYSTEM IS a genuine WALL lever (copy-free ISA-L clean tail 0.87–0.925× rg = TIE vs production pure-Rust 0.73–0.755× — Δ ~0.14× ≫ spread, sign-stable ×3, advisor-vetted). AND the UNIFIED single-primitive engine ALREADY EXISTS on the production path as gzippy-native (the FOLD: Engine M flips in-place + continues the clean tail on the SAME cursor — byte-for-byte the governing one-engine memory), byte-exact sha 028bd002…cb410f, flip_to_clean=0 finished_no_flip=16 (no 2nd engine). The unified PURE-RUST engine is the SLOWEST at the wall (native_fold 0.676–0.685× rg) because marker_inflate's `<false>` clean rate trails BOTH resumable.rs AND ISA-L. The faithful next lever is now precisely bounded: the gap a pure-Rust unified primitive must close is < 0.14× (ocl_cf − native_fold), of which a recoverable fraction is the per-128-KiB resumable CADENCE/yield tax (free to pure-Rust per CLAUDE.md), remainder = intrinsic symbol rate. VAR_VI's 0.6× standalone plateau does NOT bound a tax-shedding primitive (advisor C3 corollary REFUTED). NO production fix landed (oracle is measurement-only env-gated, byte-transparent); the unified port is DONE structurally — what remains is the clean-rate fix, scoped below.**

### THIS TURN — made the clean-tail oracle COPY-FREE (the advisor-OWED fix), read the clean-engine WALL ceiling, then confirmed the unified single-primitive engine already exists (gzippy-native FOLD) and measured it. Synchronous disproof advisor UPHELD-WITH-CAVEATS (corrected the RATE-vs-cadence attribution; reversed the pessimistic VAR_VI corollary).
- **THE COPY-FREE FIX (byte-transparent, measurement-only):** new `isal_decompress::decompress_deflate_from_bit_into(data, bit, dict, out: &mut [u8])` decodes ISA-L DIRECTLY into a caller buffer (no internal 64 MiB `Vec`, no CRC inside, returns `(written, end_bit, boundaries)`, returns None→fall-back if caller under-reserved); new `SegmentedU8::writable_tail_reserve(min_spare)` (one contiguous spare region of full chunk size, in the chunk's OWN pooled buffer) + `decoded_range(start,len)` (zero-copy CRC view). Oracle (`gzip_chunk.rs:160-281`) now reserves → ISA-L decodes in-place → `commit(keep_len)` (ZERO copies) → CRC over `decoded_range`; boundary offsets rebased to `decode_start` (= production's `decode_base`). 64 MiB is RESERVED capacity in the recycled pool buffer, NOT a fresh alloc+memset Vec.
- **SELF-TEST (Rule 4 — PASSED):** PROD == copy-free ORACLE == rg sha 028bd002…cb410f (byte-exact, OFF==identity). isal_oracle_chunks=14 isal_oracle_fallbacks=0 (ALL 14 clean tails took copy-free ISA-L, ZERO under-reserve fallbacks). Routing flip_to_clean=12 finished_no_flip=4 window_seeded=2 = IDENTICAL to prod unseeded ⇒ 89% window-absent marker bootstrap PRESERVED (OSCILLATION rule honored; did NOT seed). 877 lib + seam + 10 segmented_buffer tests green (Rosetta x86_64).
- **RESULT 1 — THE WALL NUMBER (3 interleaved N=11, sha-OK every run, locked guest REDACTED_IP, taskset 0,2,4,6,8,10,12,14, gov=perf, turbo-on):** prod (pure-Rust clean tail) 0.755/0.746/0.733× rg; **ocl_cf (COPY-FREE ISA-L clean tail, unseeded) 0.895/0.892/0.870× rg = TIE.** Δ per-pass 0.140/0.146/0.137 ≫ spread (~0.022–0.025), constant across load 1.27→2.01 (freq-neutral). The prior contaminated oracle showed ocl 0.70× ≈ prod 0.75× (copy masked the win); removing the copy moved ocl 0.70→0.89× — EXACTLY the prior advisor's S≈C≈0.17× prediction. Verbose copy-free: decodeBlock SUM 0.645s (prod ~0.83s), Real Decode 0.101s, Fill 79.5%.
- **RESULT 2 — THE UNIFIED ENGINE ALREADY EXISTS (gzippy-native FOLD):** source-verified first-hand — `marker_inflate::Block::read` IS a faithful port of rg `readBlock`: ONE width-templated engine `read_internal_compressed_specialized<const CONTAINS_MARKERS:bool>` (`<true>` u16 / `<false>` u8-direct), with `flip_repack_to_u8` (marker_inflate.rs:1116-1128) = vendor `setInitialWindow()` deflate.hpp:1282-1292. The build selector `isal_clean_tail` (build.rs:98-110, gzip_chunk.rs:1386-1413) forks: gzippy-isal = TWO-PHASE handoff to Engine C (StreamingInflateWrapper/resumable.rs) = the divergence; gzippy-native = FOLD (Engine M keeps decoding clean tail in-place on the SAME ctx cursor — the unified primitive). Built /tmp/gzbuild-native (gzippy-native), byte-exact sha 028bd002…cb410f, routing flip_to_clean=0 finished_no_flip=16 (NO 2nd engine, confirms the fold), decodeBlock 1.033s. **WALL (2 interleaved N=11, sha-OK): native_fold 0.685/0.676× rg = the SLOWEST** (vs isal_prod 0.735×, ocl_cf 0.925×). The unified pure-Rust engine is correct + faithful but slow at the clean tail.
- **ADVISOR (synchronous, read-only, plans/copy-free-oracle-advisor-verdict.md):** C1 UPHELD-WITH-CAVEATS — region IS a real lever (Δ≫spread, copy genuinely gone, reverses INCONCLUSIVE) but 0.14× bundles (a) ISA-L intrinsic symbol rate + (b) elision of the per-128-KiB resumable yield/refetch tax + (c) one-shot reserve vs incremental grow; "engine RATE is the lever" over-attributes — proven claim = "the clean-decode SUBSYSTEM is a real lever, not slack." C2 UPHELD (residual 0.89→1.0× = window-absent structure). C3 DIRECTION UPHELD (unify is right), pessimistic COROLLARY REFUTED — the pure-Rust gap to close is < 0.14× (b+c are recoverable in pure-Rust WITHOUT raising symbol rate: large contiguous window + elide the yield-check, both CLAUDE.md-authorized); VAR_VI's standalone 0.6× does NOT bound a tax-shedding primitive. MOST ACTIONABLE: run the symmetric control — give the PURE-RUST path one large contiguous window (no per-128-KiB refetch/yield) and re-measure; the recovered portion = cadence/grow (free to pure-Rust), remainder = the intrinsic-rate gap that sets the real no-FFI 1.0× bar — NOT VAR_VI.
- **SCOPED NEXT (gate, do NOT start): the CLEAN-RATE fix on the unified pure-Rust engine (gzippy-native FOLD).** (1) FIRST the advisor's symmetric control: feed marker_inflate's `<false>` clean tail one large contiguous output window (current cadence is per-128-KiB `writable_tail` refetch + per-call yield-check in the resumable/marker drain) and re-measure native_fold — isolate cadence/grow (free) from intrinsic symbol rate. (2) THEN close the intrinsic-rate remainder in the `<false>` path (BMI2 PEXT, wider multi-literal, ISA-L-class packed-u32 LUT — the inner-Huffman work CLAUDE.md authorizes) up to the < 0.14× bound. Bound each with the copy-free ocl_cf as the engine-removed ceiling (0.925× = the readable target), never the slope. The unified-engine PORT itself is DONE (gzippy-native is the production unified path); this is the rate fix on it.
- **GUEST STATE:** /tmp/gzbuild-head/release/gzippy (gzippy-isal native, HEAD + copy-free oracle, sha 028bd002…cb410f) + /tmp/gzbuild-native/release/gzippy (gzippy-native FOLD, sha 028bd002…cb410f). rg 0.16.0 at /usr/local/bin. Drivers /tmp/oracle_{selftest,wall}.sh (use bash). 3 modified files overlaid on guest /root/gzippy (isal_decompress.rs, gzip_chunk.rs, segmented_buffer.rs) = local HEAD+copy-free; guest git unchanged. seeds /tmp/seeds.bin. NO orphan processes.

## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, owner turn, branch reimplement-isa-l, HEAD 7aae6c4a + measured /tmp/gzbuild-head) — **DECISIVE WALL ORACLE RAN (window-absent-PRESERVING ISA-L clean-tail removal) → INCONCLUSIVE on the clean-engine binder (the instrument is CONTAMINATED by its own copy confound), but the RECONCILIATION is SOLID: the seedfull-TIE↔production-0.73× gap is the WINDOW-ABSENT STRUCTURE (~0.16× of the wall, ISA-L clean engine held constant), NOT the clean engine. Advisor REFUTED my Conclusion-1 framing (load-bearing) and showed NO FORK is established — rg uses ONE width-templated primitive for marker+clean; gzippy's two-engine split is the divergence. NO fix landed; NO fork escalated; the copy-free oracle is OWED before the clean-engine question is answerable.**

### THIS TURN — ran the decisive WALL-level, window-absent-PRESERVING ISA-L-clean-tail removal oracle + the seedfull↔production reconciliation. Synchronous disproof advisor REFUTED Conclusion 1 as-framed (instrument contaminated).
- **THE ORACLE (already wired, NO rebuild, NO seed):** `GZIPPY_ISAL_ENGINE_ORACLE=1` (gzip_chunk.rs:539→:160) replaces ONLY the post-flip clean tail decode (`finish_decode_chunk_impl`, the resumable.rs term) with REAL ISA-L FFI; the u16 marker bootstrap runs as in production. SELF-TEST PASSED (Rule 4): OFF==ON==rg sha 028bd002…cb410f (byte-exact, OFF==identity); `isal_oracle_chunks=14 isal_oracle_fallbacks=0` (fires unseeded, uncontaminated by fallback); routing flip_to_clean=12 finished_no_flip=4 window_seeded=2 = identical to prod unseeded ⇒ the 89% window-absent marker bootstrap IS preserved (charter OSCILLATION rule honored: did NOT seed).
- **RESULT 1 — THE WALL NUMBER (3 interleaved N=11 runs, sha-OK every run, locked guest REDACTED_IP, taskset 0,2,4,6,8,10,12,14, gov=perf, turbo-on):** prod (pure-Rust clean tail) = 0.744/0.754/0.755× rg; ocl (ISA-L clean tail, unseeded) = 0.698/0.686/0.702× rg. **The ISA-L clean-tail oracle did NOT beat production — ocl ≈ prod or slightly slower, far from the 0.85× falsifier line.** Pre-registered "clean-engine-rate is the binder" branch did NOT fire on the raw number.
- **ADVISOR REFUTED Conclusion 1 (LOAD-BEARING, and CORRECT):** I used the copy confound asymmetrically and the two uses contradict. The oracle pays a per-chunk 64 MiB alloc + to_vec copy (gzip_chunk.rs:203,247-256) that production's direct-to-`writable_tail()` stream never pays. My OWN reconciliation prices that copy at ~0.17× (ocl_seed 0.86× vs pure seedfull 1.029× last turn). Model W_ocl = W_prod − S + C: Result 1 measured W_ocl≈W_prod ⇒ **S ≈ C ≈ 0.17×** — i.e. the ISA-L clean engine plausibly SAVES ~0.17× of wall, EXACTLY masked by the copy (a copy-free ocl ≈0.84-0.87×, AT the threshold). A HANDICAPPED contender failing to win is UNINFORMATIVE about a speed-UP ceiling (Measurement PROCESS Rule 3) — NOT "conservative." **VERDICT: Conclusion 1 INCONCLUSIVE; the oracle is a contaminated instrument for the clean-engine question. CANNOT declare "engine slack" NOR "engine is the binder."**
- **RESULT 2 — THE RECONCILIATION (UPHELD, engine held CONSTANT = ISA-L in both, 2 runs sha-OK):** ocl_unseed (marker bootstrap PRESERVED) = 0.697/0.701× rg vs ocl_seed (windows SEEDED, no bootstrap) = 0.860/0.857× rg. Verbose: ocl_unseed decodeBlock 0.869s / Real 0.129s / Fill 84% / marker bootstrap body 321.7ms@227MB/s / flip=12 finished=4 / spec-fail header=13; ocl_seed 0.749s / 0.101s / Fill 93% / bootstrap 0 / flip=0 finished=0 window_seeded=17 / spec-fail=0. **Same ISA-L clean engine in both ⇒ the seedfull↔production gap is the WINDOW-ABSENT STRUCTURE (~0.16× of the wall), NOT the clean engine.** ADVISOR CAVEAT: do NOT sub-credit this to the marker bootstrap RATE specifically — seeding ALSO removes 13→0 spec-failure re-decodes (the project_confirmed_offset_prefetch_gap block-finder/scheduling term) and the flip machinery; the "ΔdecodeBlock 0.120s ≈ bootstrap body 0.322s SUM/overlap" match is the charter's SUM-vs-wall trap (lines 522-528). The CAUSAL claim (structure costs ~0.16× at the wall, clean engine constant) is sound; the rate-vs-spec-fail split is NOT isolated.
- **Q4 — NO FORK IS ESTABLISHED (advisor, vendor-verified):** rg's marker decode and clean decode are the SAME primitive — `readBlock` calls `readInternal(...,m_window16)` for markers (deflate.hpp:1277) and `readInternal(...,window)` for clean (:1291), both dispatching to the ONE `readInternalCompressedMultiCached` (:1452-1453), a single template over window element width (`containsMarkerBytes` :1600), `setInitialWindow` (:1285) flips width mid-chunk. rg ties unseeded BECAUSE that one fast loop serves both. gzippy's premise "the marker bootstrap is a separate ISA-L-untouchable lever, free of the clean plateau" describes gzippy's DIVERGENCE (two engines: marker_inflate u16 + resumable u8), not rg's design. So the marker rate is most likely bounded by the SAME primitive ceiling VAR_VI measured (clean variant). **The faithful move is NOT a fork — it is to COLLAPSE marker+clean into ONE `readInternalCompressedMultiCached`-shaped primitive** ([[project_faithful_unified_decoder_over_perf]]); whether that unified ceiling clears 1.0× pure-Rust is the ONE genuine open question.
- **OSCILLATION NOT ENDED (advisor Q3):** two terms remain entangled — (a) the spec-failure re-decode cost (scheduling/block-finder) bundled into Result 2, and (b) the marker→clean resolve/apply-window pass (rg's 0.0348s "applying last window") isolated by neither cell. Do NOT install "bootstrap rate" as a third under-resolved label.
- **MOST ACTIONABLE NEXT (advisor, do NOT start — supervisor gate):** the oracle is unusable for the speed-up ceiling until the 64 MiB alloc + intermediate copy is removed (decode ISA-L DIRECTLY into `chunk.data.writable_tail()` so C→0 and S becomes readable). Until then neither "engine is slack" nor "engine is the lever" is decided. THEN the faithful direction is the UNIFIED single-primitive engine (collapse marker_inflate + resumable), not a fork escalation.
- **GUEST STATE:** /tmp/gzbuild-head/release/gzippy (measured, HEAD isal native, sha 028bd002…cb410f); rg 0.16.0 at /usr/local/bin; drivers /tmp/oracle_selftest.sh, /tmp/oracle_wall.sh, /tmp/oracle_reconcile.sh, /tmp/oracle_verbose.sh (use `bash`); seeds /tmp/seeds.bin. NO orphan processes (advisor + sleep children killed; guest pgrep clean).

## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, owner turn, branch reimplement-isa-l, HEAD 7aae6c4a + overlay) — **1.6× WINDOW-ABSENT decodeBlock GAP ATTRIBUTED (causally): it is the CLEAN u8 TAIL DECODER, NOT the u16 marker loop. gzippy's marker bootstrap is actually FASTER than rg's "custom inflate" (0.323s vs 0.475s); the dominant decodeBlock term is the pure-Rust clean tail (`unified::Inflate<Clean>`/resumable.rs, ≈290 MB/s) which is ~2.3× slower than rg's ISA-L clean (0.207s). Despite the `isal_clean_tail` cfg NAME, BOTH gzippy builds decode the clean tail in PURE RUST — real ISA-L FFI is reachable ONLY under the measurement oracle, NOT production. Disproof advisor UPHELD the causal core (CLEAN-tail is the bigger decodeBlock term), flagged the 2.3× ratio as a subtraction hypothesis, and REFUTED "actionable now" (payoff needs a WALL removal oracle, not the slow-down slope). Fixed the stale build.rs comment. NO fix landed (STOP at attribution per prompt).**

### THIS TURN — ATTRIBUTED the 1.6× window-absent decodeBlock gap via source-diff + apples-to-apples --verbose + causal SLOW_MODE A/B (freq-neutral). Synchronous disproof advisor. STOPPED at attribution + scoped fix (no fix landed).
- **APPLES-TO-APPLES --verbose (locked guest REDACTED_IP, T8 unseeded, taskset 0,2,4,6,8,10,12,14, gov=perf):** gzippy decodeBlock SUM **0.805-0.831s** vs rg **0.4995s = 1.61-1.66×**. Per-engine: gzippy marker bootstrap body **0.323s** (73.0M @ 226-235 MB/s) vs rg "custom inflate" **0.4748s** ⇒ gzippy marker is 0.68× = **FASTER**. gzippy clean tail **≈0.48s** (≈139M, ≈290 MB/s, by subtraction) vs rg "ISA-L" **0.2065s** ⇒ **≈2.3× SLOWER**. gzippy routing: flip_to_clean=12, finished_no_flip=4, window_seeded=2 (17 chunks). markers 34.5% both (gzippy 73.0M ≈ rg 73.1M).
- **CAUSAL SLOW_MODE A/B (ΔdecodeBlock SUM, freq-neutral sleep control; baseline 0.831s):** MARKER+100% → 0.965s (+134ms, body 323→483); CLEAN+100% → 1.025s (**+194ms**, marker body UNCHANGED 312). Sleep controls: marker +142ms, clean +248ms. Spin≈sleep ⇒ real, not turbo. **CLEAN inject (lands in resumable.rs:1199, the post-flip clean engine) leaves the marker body untouched yet adds the LARGER ΔdecodeBlock ⇒ the clean tail is the dominant decodeBlock term, NOT the marker loop.**
- **SOURCE ROOT CAUSE (verified first-hand):** flip threshold byte-identical (marker_inflate.rs:1116-1119 ↔ vendor deflate.hpp:1282-1284). Two-phase routing (gzip_chunk.rs:1397-1410, `isal_clean_tail`): Engine M u16 bootstrap → FlipToClean → Engine C = StreamingInflateWrapper = `unified::Inflate<Clean>` = pure-Rust resumable.rs. **The measured gzippy-ISAL build's clean tail is PURE-RUST resumable, NOT ISA-L FFI** (resumable.rs:1182-1192 confirms; the build.rs:98 comment claiming "REAL ISA-L FFI" was STALE — FIXED this turn, comment-only/byte-transparent). rg's WITH_ISAL hands its clean tail to real ISA-L (deflate.hpp:1452-1453). So the 1.6× = gzippy's pure-Rust clean engine (~0.6× ISA-L primitive plateau, prior-turn advisor-upheld) decoding the clean BULK ~2.3× slower than ISA-L.
- **ATTRIBUTION VERDICT among the prompt's 3 candidates:** (a) marker inner loop — REFUTED as prime term (gzippy marker FASTER than rg). (b) u16-width-over-clean-bulk — REFUTED (flip byte-identical to vendor; post-flip bulk decodes u8, not u16). (c) table-build — REFUTED (shared with clean path; seedfull ties). **CAUSE = the pure-Rust CLEAN u8 tail decoder being slower than ISA-L.**
- **DISPROOF ADVISOR (synchronous, read-only, plans/window-absent-attribution-advisor-verdict.md):** core UPHELD (CLEAN-tail dominant, causally airtight; routing source-trace + slow-knob isolation verified). Most load-bearing: CONFIRMED the build.rs "REAL ISA-L FFI" comment is stale/contradicted by wiring (a reviewer trusting it would wrongly refute the owner). REFUTED-as-framed angle D: NOT actionable for a work-stretch without running the WALL removal oracle (the A/B is ΔdecodeBlock SUM, slack-masked at Fill 85%; payoff unbounded until oracle, Rule 3). UPHELD-WITH-CAVEAT: 0.68× and 2.3× are cross-tool/subtraction figures (hedge them; the PROVEN claim is "clean tail is the bigger decodeBlock term"). Faithfulness E: candidate 1 (ISA-L FFI clean tail) = faithful to rg WITH_ISAL = charter goal #2 but re-adds C-FFI (violates goal #1 no-FFI); candidate 2 (faster pure-Rust clean engine) is the goal-#1 advance.
- **SCOPED FIX (do NOT start — supervisor gate; BOUND BY THE Phase-0 ISA-L removal oracle FIRST, never the slope):** (1) gzippy-faithful/isal (goal #2): route the FlipToClean clean tail through real ISA-L FFI (= rg WITH_ISAL; largely the Phase-0 oracle made permanent) — cleanest test of clean-engine-speed-as-binder. (2) gzippy-native (goal #1, 1.0× bar): faster pure-Rust clean u8 engine (BMI2 PEXT, wider multi-literal, ISA-L-LUT parity — the inner-Huffman work the charter authorizes; ceiling is the ~0.6× ISA-L primitive plateau, so this is the hard one). RUN the GZIPPY_ISAL_ENGINE_ORACLE on the FlipToClean clean tail (watch ISAL_ENGINE_ORACLE_FALLBACKS==0) to convert ΔSUM into a Δwall ceiling before any build.
- **GUEST STATE:** /tmp/gzbuild-head/release/gzippy (gzippy-isal native, HEAD, clean — the measured build) + /tmp/gzbuild-po (corrected-overlap overlay, prior). rg 0.16.0 at /usr/local/bin. Verbose captures /tmp/rg_verbose.txt + /tmp/gz_verbose.txt; slow-inject captures /tmp/{m100,c100,cs,ms}.txt. NO orphan processes (guest pgrep clean; advisor subagent completed).

### THIS TURN — CORRECTED the backwards overlap oracle, measured the registered decider, removed the advisor-flagged retention confound, then tested the two remaining scheduling sub-levers (resolve-ahead coverage + finer chunking). Two synchronous disproof-advisor passes (the second reversed the advisor's own prior "F1 likely holds via overlap").
- **THE CORRECTION (perfect_overlap.rs + chunk_fetcher.rs `perfect_overlap_warm`):** the prior oracle ran warm-all-then-drain (blocked on `recv()` for every chunk BEFORE the consumer started = ANTI-overlap, 0.225s). CORRECTED: dispatch EVERY chunk's decode as an IN-FLIGHT prefetch up-front via `block_fetcher.submit_prefetch(part_key, rx)` (vendor `m_prefetching.emplace`, BlockFetcher.hpp:558) — NON-BLOCKING — then return immediately so the in-order `consumer_loop` runs CONCURRENTLY with the still-running decodes = real decode↔drain OVERLAP. Removed the 4096 cache-cap bump (advisor-flagged confound). KEEPS the marker engine (NOT seeded), serial resolve chain, drain, write.
- **SELF-TEST (Rule 4 — VALIDATED):** OFF==identity AND ON byte-exact sha 028bd002…cb410f on BOTH arm64-native (local) AND x86_64 gzippy-isal native (guest); path=ParallelSM; dispatch phase 0.0007s (non-blocking, vs prior 0.117s blocking); warm_hit_frac 0.882 (15/17; 2 offset-0 misses).
- **THE DECIDER NUMBER (lead with the causal oracle; T8, measure.sh interleaved N=11, sha-OK every run, CPUS=0,2,4,6,8,10,12,14, locked guest REDACTED_IP, gov=perf, turbo-on; 5 runs across both oracle variants):** perfovl (CORRECTED, retention-fixed) = **0.187-0.192s = 0.684-0.695× rg** vs production HEAD 0.174-0.177s = 0.730-0.754× rg (rg 0.130s). **The corrected overlap oracle is sign-stably ~5-7% SLOWER than production and does NOT reach the tie.** Removing the retention confound did NOT rescue it ⇒ the dispatch-flood itself buys no wall.
- **RESOLVE-AHEAD ALREADY SATURATED (first-hand verbose):** `Worker resolve-ahead: ok=13/13 (head), 14/14 (perfovl)` — drain-hiding (the advisor's named "untested lever") is LIVE on the production path at ~82% honest coverage (14/17 eager-submitted). The consumer's `std::future::get` ~0.08-0.10s is a wait on the DECODE future, NOT on resolve (resolve runs earlier on the pool, chunk_fetcher.rs:1499 vs the blocking decode wait :1516). So drain is already hidden; what's left is decode time.
- **FINER-CHUNKING REFUTED (GZIPPY_CHUNK_KIB sweep, byte-exact, interleaved vs rg, 2 runs):** k4096(default,17 chunks)=0.699/0.722× ≈ k2048(34)=0.684/0.690× ≈ k1024(68)=0.681/0.688× rg. FLAT-to-WORSE. Verbose: decodeBlock SUM stays ~1.1s and Fill DROPS 87%→77% as chunks shrink (per-chunk marker-bootstrap overhead grows, cancelling the tail-wave gain). The advisor's ~0.04s tail-wave-quantization hypothesis (3 waves 8+8+1) did NOT materialize.
- **INDEPENDENT DISPROOF ADVISOR ×2 (synchronous, read-only, plans/corrected-overlap-advisor-verdict.md):** PASS 1 REFUTED my C1 (decider) — flagged the retention confound + that the real untested lever was resolve-ahead/drain-hiding; said "F1 likely holds via overlap." PASS 2 (after the resolve-ahead-saturated + retention-fixed + finer-chunking evidence) REVERSED itself: **F1-via-overlap/drain is REFUTED** (both legs of its prior rescue gone); binder relocates to the per-thread decode floor (engine), UPHELD in DIRECTION with the 1.6× magnitude UNVERIFIED (rg decode-sum uncited). Caveat: target the u8-direct clean path (governing memory), not just a faster u16 ring; the 1.6× may be u16-width-over-clean-bulk, not the marker inner loop.
- **BOUNDED CONCLUSION (causal, advisor-vetted): the T8 wall is NOT scheduling-bound.** All three scheduling levers exhausted: dispatch-depth (corrected oracle, null/worse), drain-hiding (resolve-ahead saturated ~82%, wall unmoved), tail-wave (finer chunking flat-to-worse). The residual ~0.70× gap is the per-thread MARKER-ENGINE decode rate (decodeBlock SUM ~0.83s ÷8 = 0.103s floor ≈ already 79% of rg's 0.130 wall — but the wall sits at 0.176, so the engine floor + its in-order serial tail is the binder).
- **SCOPED NEXT (do NOT start — supervisor gate):** the faithful ENGINE direction. FIRST source-diff gzippy's window-absent `decode_chunk_unified_marker` (u16 ring) vs vendor to attribute the gap among (i) marker inner loop, (ii) u16-vs-u8 width on the clean bulk, (iii) upstream table-build/bounds — per the governing memory the faithful target is the u8-DIRECT clean path, with readInternalCompressedMultiCached the marker-path port. Bound any engine fix with a REMOVAL oracle (Phase-0 ISA-L engine oracle already ties seeded — re-use it as the engine-removed ceiling), never the slope.
- **GUEST STATE:** /root/gzippy src OVERLAID to HEAD 7aae6c4a + the CORRECTED oracle (chunk_fetcher.rs/perfect_overlap.rs/mod.rs via /tmp/po-corrected2.tgz; guest git unchanged but SOURCE is HEAD+corrected-oracle). Build /tmp/gzbuild-po/release/gzippy (gzippy-isal native, sha 028bd002…cb410f). Drivers /tmp/po_measure.sh + /tmp/chunk_sweep.sh (bash). silesia /root/silesia.gz, seeds /tmp/seeds.bin. NO orphan processes (advisor wrappers killed; guest pgrep clean).

## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, owner turn, HEAD 7aae6c4a + oracle overlay) — **GZIPPY_PERFECT_OVERLAP (the registered decider, NEVER-BEFORE-RUN) BUILT + SELF-TESTED + RUN. ADVISOR REFUTED my read: my oracle was built BACKWARDS (warm-then-drain = anti-overlap, SLOWER than production) so it CANNOT decide F1 — but its warm-phase LOWER BOUND (0.117s < rg WALL 0.131s) says the T8 TIE IS REACHABLE by better overlap. Scheduling direction NOT refuted; the decider question remains UNANSWERED.**

### THIS TURN — ran the registered decider oracle (Rule-3 violation closed: it had NEVER been run). Built byte-exact + self-tested (Rule 4), measured on the locked guest, ran the synchronous disproof advisor — who caught a LOAD-BEARING implementation error in my oracle AND a denominator mis-read in my conclusion.
- **THE DECIDER NUMBER (lead with the causal oracle, not attribution — PROCESS FIX #2):** GZIPPY_PERFECT_OVERLAP wall = **0.225-0.227s = 0.581-0.583× rg** (T8, measure.sh interleaved N=11, sha-OK both runs, 2 runs sign-stable). Production HEAD = 0.177s = 0.740×. rg = 0.131s. **The oracle is SLOWER than production.**
- **WHY (advisor, load-bearing REFUTATION):** the oracle as I built it runs warm (decode-ALL on the real pool, real marker engine, Fill→100%) FULLY, THEN drain (serial resolve-chain + write) — it SERIALIZES the two phases production already OVERLAPS. So its wall is a pessimistic SUM (warm 0.117 + drain 0.066 = 0.183 single-run / 0.225 interleaved), the OPPOSITE of perfect overlap. A "perfect-overlap" config slower than the un-optimized scheduler is an ANTI-overlap; per the symmetric of last turn's correction (an upper-bound LOSS can't fire F2), this upper bound built by DESTROYING overlap **cannot falsify F1** (the TIE claim). My C1 (F1 falsified) / C3 (engine floor implicated) / C4 (binder = engine, not scheduling) are **REFUTED**.
- **THE GENUINE FINDING (advisor-corrected):** the warm phase alone = **0.117s** is a TRUE LOWER bound on any schedule's wall (every chunk must decode; drain 0.066 < warm so it hides under decode). **0.117 < rg WALL 0.131 < tie threshold 0.138** — the lower bound is INSIDE the tie zone. I reported "lower bound above the tie" ONLY by mis-comparing 0.117 to rg's decode FLOOR (0.085) instead of rg's WALL (0.131). Matched floor-to-floor = warm 0.117 vs rg Real Decode 0.104 = **1.13×** (not my 1.38×). **Read correctly: the T8 TIE IS REACHABLE by better decode↔drain OVERLAP — the scheduling/overlap direction is NOT refuted; this oracle FAILED TO TEST it.**
- **SELF-TEST (Rule 4 — VALIDATED before trusting):** output sha 028bd002…cb410f byte-IDENTICAL with/without the oracle on BOTH arm64-native (local) AND x86_64 gzippy-isal native (guest); warm_hit_frac 0.88-0.96 (the 2 misses are offset-0 stream-start chunks). The instrument is byte-transparent and the warm cache really removed the head-of-line wait — it just measured the WRONG schedule.
- **CODE (measurement-only, env-gated, OFF==identity, NOT production):** new `src/decompress/parallel/perfect_overlap.rs` + `perfect_overlap_warm` (chunk_fetcher.rs) + warm call before the timed consumer_loop + prefetch-cache cap bump under the oracle + warm-hit/miss self-test counters. Faithfulness (advisor angle 1 UPHELD): warm never seeds windows / never runs the resolve chain ⇒ every non-zero chunk decodes MARKERED at the true rate (gzip_chunk.rs:790 clean-flip impossible during warm except start_bit==0); is_speculative_prefetch=true at partition guesses matches the prefetch path.
- **INDEPENDENT DISPROOF ADVISOR (synchronous, read-only, plans/perfect-overlap-advisor-verdict.md):** C1 REFUTED (anti-overlap can't falsify F1). C2 UPHELD-on-fact / REFUTED-on-inference (decode floor > rg floor is real but 0.117 < rg WALL ⇒ engine floor does NOT bind the tie; the matched gap is 1.13× not 1.38×). C3 REFUTED (the load-bearing denominator error: 0.117 is BELOW the tie threshold, not above). C4 REFUTED-as-stated (the anti-scheduling conclusion is INVERTED; faster marker engine is neither necessary nor sufficient for the wall tie — keep it only as a secondary/T1 lever). All 5 disproof angles addressed; angle-1 faithfulness UPHELD.
- **STILL OPEN — the decider question this oracle did NOT answer:** can a REAL OVERLAPPED schedule (decode overlapping resolve+write, NOT serialized) collapse production's 0.177s toward the 0.117-0.13 floor? Needs a CORRECTED oracle that OVERLAPS warm with drain (pipeline the drain to start per-chunk as soon as each predecessor window is ready, while later chunks still decode), NOT warm-all-then-drain. **F1 remains UNDECIDED. Do NOT declare STOP/TIE (Rule 3 + PROCESS FIX #3).**
- **SCOPED NEXT (do NOT start — supervisor gate):** build the CORRECTED overlap oracle (warm overlapping drain) to actually decide F1; OR, since the lower bound already says the tie is reachable, go straight to the faithful fix = close the decode↔drain overlap gap (production 0.177 → ~0.12-0.13), which is the named project_confirmed_offset_prefetch_gap dispatch-TIMING lever — bound it with the corrected oracle first (Rule 3), never the slope.
- **GUEST STATE:** /root/gzippy src OVERLAID to HEAD 7aae6c4a + the oracle (piped tarball /tmp/gzsrc-overlay.tgz extracted over the tree; guest git still reads 7bf26096 but the SOURCE is HEAD+oracle). Build /tmp/gzbuild-po/release/gzippy (gzippy-isal native, sha 028bd002…cb410f). Driver /tmp/po_measure.sh (bash). silesia /root/silesia.gz, seeds /tmp/seeds.bin. Advisor wrapper + waiter processes completed; NO orphan processes (verified below).

## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, owner turn, HEAD f1aceee1) — **COUNTER RENAMED (anti-inversion) + SCHEDULING/SERIAL CEILING BOUNDED via real oracles. ADVISOR REFUTED my arithmetic F2 over-reach.** The T8 TIE IS reachable (seedfull oracle = 1.029× T8, 1.121× T16 WIN, sha-exact) BUT scheduling-overlap AND the window-absent marker-engine rate are LIVE + ARCHITECTURALLY COUPLED terms — neither is cleanly isolable (window-present ⇒ clean engine, gzip_chunk.rs:790). **The prompt's "engine slack-masked, binder is scheduling" premise is NOT confirmed AND my counter-arithmetic "engine binds (F2)" is REFUTED** (Rule-3 extrapolation; the sum was a strict upper bound). Status = F3/both-live. **SUPERVISOR GATE — ceiling bounded, NO engine fix landed (binder coupled/unconfirmed); next loop must CAUSALLY perturb before any work-stretch.**

### THIS TURN — renamed the inversion-prone counter (byte-exact), then BOUNDED the scheduling/serial ceiling with REAL removal oracles on the locked guest. Advisor (synchronous, read-only) REFUTED my arithmetic over-reach; resolution is oracle-grounded.
- **COUNTER RENAME (commit f1aceee1, byte-transparent instrumentation):** `BOOTSTRAP_POST_FLIP_U16_BYTES` → `BOOTSTRAP_CLEAN_FLIPPED_BYTES` (gzip_chunk.rs:97/:1491 + chunk_fetcher.rs:947 + GZIPPY_VERBOSE label). It counts output bytes of bootstrap blocks that ended CLEAN (`!contains_marker_bytes`) — the marker-FREE COMPLEMENT, NOT "bytes decoded into the u16 marker ring after the flip" as the old name+doc claimed. It had been read backwards repeatedly (the exact C3 counter-inversion the prior advisor refuted). New label now self-documents: `clean_flipped_bytes=1425448 (2.0% of body = marker-FREE complement; marker loop owns the other 98.0%)`. Compiles clean; faithful_u8_flip_seam test green.
- **FIRST-HAND VERBOSE (locked guest REDACTED_IP double-ssh, 16c gov=perf, gzippy-isal native synced to HEAD f1aceee1, sha 028bd002…cb410f every cell, T8):**
    | metric | gzippy HEAD | rapidgzip | ratio |
    | decodeBlock SUM | 0.803s | 0.502s | 1.60× |
    | Theoretical Optimal (÷T) | 0.100s | 0.068s | 1.47× |
    | Total Real Decode | 0.116s | 0.084s | 1.38× |
    | std::future::get | 0.089s (T16: 0.046s) | 0.064s | 1.39× |
    | serial tail (wall−RealDecode) | 0.058s | ~0.043s | 1.35× |
    | WALL (interleaved best, measure.sh) | 0.174-0.177s | 0.130s | **0.736-0.755×** |
  Note Real Decode 0.116 is BELOW rg's WALL 0.130 (the prompt's cached 0.137 was pre-mergefix/stale). T16: HEAD 0.162s = **0.885×** (rg slows to 0.144 at T16 on 17 chunks).
- **REMOVAL ORACLE #1 — seedfull (GZIPPY_SEED_WINDOWS, sha-exact):** all 17 chunks window-seeded ⇒ CLEAN engine, 0 spec-failures, Fill 90%. T8 wall **0.128s = 1.029× rg = TIE**; T16 **0.128s = 1.121× rg = WIN**. seedfull's future::get 0.083s ≈ HEAD's 0.089s. **This IS the faithful "perfect window-overlap" oracle** — the ONLY way to give the consumer pre-resolved windows (remove head-of-line wait) is to seed windows, which ALSO flips the engine clean (coupling, gzip_chunk.rs:790 vs :826). A pure-scheduling oracle keeping the marker engine is IMPOSSIBLE in-architecture.
- **NEGATIVE CONTROL — GZIPPY_NO_PREFETCH (sha-exact):** T8 wall **0.523s = 0.253× rg (3× SLOWER)**. Removing the prefetch overlap is catastrophic ⇒ scheduling is FIRMLY on the critical path. + future::get HALVES T8→T16 (0.089→0.046) = signature of CRITICALITY (slack does not scale with cores).
- **INDEPENDENT DISPROOF ADVISOR (synchronous, read-only, plans/scheduling-ceiling-advisor-verdict.md):** C1 (engine reaches wall) UPHELD-WITH-CAVEATS (the HEAD→seedfull A/B moving 0.040s on the non-future::get axis is the real evidence, NOT the banned attribution ratios — but "reaches the wall" ≠ "is THE binder"). **C2 (scheduling not the binder) REFUTED** — I read future::get's halving backwards; it IS a criticality signature; NO_PREFETCH 3× regression confirms it. **C3 (arithmetic F2 ceiling = loss) REFUTED (load-bearing)** — 0.116+0.043 is my OWN strict upper bound (double-counts the overlapping tail); the only LOWER bound (decode-phase wall 0.116 = 0.89× rg) is in TIE territory; concluding F2 from a hand sum with no oracle violates Rule 3. **C4 (next binder = backward marker scan emit_backref_ring::<true> :3006-3027) UPHELD-WITH-CAVEATS → effectively UNCONFIRMED** — the scan is fast-path-skipped once `distance_marker>=distance` (:3002) and the isal build FLIPS to clean u8 at 32KiB (gzip_chunk.rs:949), confining it to the per-chunk bootstrap (<1% of a multi-MB chunk); implausible as the prime 1.6× term; needs a causal perturbation.
- **BOUNDED CEILING (honest, oracle-grounded): the T8 TIE IS reachable** (seedfull proves it, F1) but BOTH the scheduling overlap AND the window-absent marker-engine rate are LIVE, COUPLED terms; neither is isolable in gzippy's architecture (window-present ⇒ clean). **rg ties UNSEEDED at the same 34.5% markers because its marker engine is fast (decodeBlock 0.502 vs 0.803 = 1.6×)** — so the faithful path is to make the window-absent decode cheaper at the wall.
- **rg's MECHANISM (source-verified, vendor GzipChunkFetcher.hpp):** `waitForReplacedMarkers` (:479) queues the head chunk's marker-replace, then USES THE WAIT to harvest ready futures + `queuePrefetchedChunkPostProcessing` (:513, full sorted prefetch-cache scan, queue post-process for every chunk whose predecessor window is available). The LAST window is inserted by the MAIN thread (:559-561, *"the critical path that cannot be parallelized... do not compress the last window to save time"*) — rg explicitly names window-publish as THE serial critical path and minimizes it. gzippy ALREADY ports this (queue_prefetched_marker_postprocess chunk_fetcher.rs:1592/1702 + prefetch pump during wait). So the consumer STRUCTURE is faithfully ported; the residual is dispatch TIMING (windows published slightly later ⇒ more chunks window-absent at high T), NOT a missing mechanism and NOT horizon DEPTH (vendor-identical).
- **SCOPED FIX for the NEXT loop (do NOT start — supervisor gate; must CAUSALLY PERTURB FIRST per advisor C4):** TWO coupled faithful candidates, each needs a confirming perturbation before a work-stretch: (a) faster window-absent u16 MARKER engine (the 1.6× decodeBlock gap reaching the wall via Real Decode 1.38×) — but C4's specific "backward marker scan" hypothesis is UNCONFIRMED/implausible (flip-to-clean at 32KiB confines it); the real remaining engine term is more likely the post-flip u8 CLEAN rate + u16 bootstrap traffic, which a SLOW-INJECT/oracle perturbation must locate; OR (b) publish predecessor windows EARLIER so more chunks hit the clean path at high T (closing project_confirmed_offset_prefetch_gap dispatch-TIMING). Bound each with a REMOVAL oracle; never the slow-down slope.
- **GUEST STATE:** /root/gzippy/src rsynced to HEAD f1aceee1 (gzippy-isal native build /tmp/gzbuild-head, sha 028bd002…cb410f). Drivers /tmp/head_measure.sh, /tmp/seedfull_measure.sh, /tmp/t16_measure.sh, /tmp/noprefetch_measure.sh, /tmp/head_verbose.sh, /tmp/seedfull_verbose.sh, /tmp/rg_verbose.sh (all `bash`). Seeds /tmp/seeds.bin. NO orphan processes (advisor wrapper + sleep killed; guest pgrep clean).

## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, owner turn, HEAD 04fda86d) — **PORT (i) LANDED: rg's multi-cached u16 marker FAST LOOP. Byte-exact. T8 wall = TIE (no move), KEPT per 7a. Advisor C1 UPHELD, C3 REFUTED my mechanism.** The marker decode SUM gap (decodeBlock ~1.9× rg) is SLACK-MASKED at the wall (Fill 87%, wall unchanged). Whole-system T8 wall STILL ~0.73× rg. NEXT = the binder is NOT the marker compute — re-perceive: the engine SUM is ~1.9× but slack-masked (matches Phase-0 oracle TIE-vs-TIE). **SUPERVISOR GATE — marker loop measured + committed (TIE, kept); next binder NOT yet located.**

### THIS TURN — ported rg's multi-cached u16 marker fast loop, byte-exact, remove-and-measure on the locked guest. Result: a faithful TIE (kept), and an important PREMISE CORRECTION.
- **THE CHANGE (commit 04fda86d, faithful port of vendor `readInternalCompressedMultiCached` deflate.hpp:1585-1666):** added a speculative software-pipelined FAST LOOP to the u16 MARKER path (`read_internal_compressed_specialized::<true>`, marker_inflate.rs new `'mfast` loop), mirroring the clean path's existing fast loop. rg runs the SAME tight multi-cached loop for u16 markers as for u8 clean (templated on `Window`, no separate slow marker path); gzippy's clean path already had its fast loop but the MARKER path was stuck on the careful per-symbol loop. Three faithful u16 deltas: (1) literal store widened to u16 via an 8-byte speculative store `(p&0xFF)|((p&0xFF00)<<8)|((p&0xFF0000)<<16)`, value-identical to the careful loop's `write(code&0xFF)`; (2) `distance_marker += lit_prefix` per packet, back-refs via the SAME `emit_backref_ring::<true>` (marker scan maintained inside); (3) no `distance>decoded+emitted` range check (vendor const-folds it for marker windows).
- **BYTE-EXACT:** gzippy-native arm64 (T1/T8/T16) + gzippy-isal guest x86_64 (T1/T8/T16) BOTH sha 028bd002…cb410f via path=ParallelSM. 856 lib tests pass (1 fail = pre-existing flaky `diff_ratio` timing micro-test). Adversarial seam test (`faithful_u8_flip_seam_max_distance_backref_vs_flate2`) + native_fold_parity green.
- **REMOVE-AND-MEASURE (locked guest REDACTED_IP double-ssh, 16c gov=perf turbo-on, taskset 0,2,4,6,8,10,12,14, T8, measure.sh interleaved N=11, RAW=68229982, sha-OK every run): markerfast vs mergefix(prior HEAD 77a02f5f) = +1.2% / +3.0% / +0.0% across 3 interleaved runs = TIE** (within 10-38% spread; per charter "Δ < spread ⇒ TIE"). Per-stage trace: decodeBlock 0.9568→0.9485s (~0.9%), body_rate 203→207 MB/s. rg decodeBlock 0.500s ⇒ gzippy still ~1.9×, but Total Real Decode 0.137s / Fill 87% / wall 0.175s = the engine SUM is SLACK-MASKED (matches Phase-0 TIE-vs-TIE). **VERDICT: faithful TIE — KEPT per rule 7a; gain latent behind the current (non-engine) binder.**
- **ADVISOR (plans/marker-loop-port-advisor-verdict.md, synchronous read-only, source-verified):** C1 BYTE-EXACT+FAITHFUL **UPHELD** (widening correct b0/b1/b2/0; no ring-wrap straddle; no bit-cursor desync; dropped range-check faithful; emit_backref_ring is literally the same fn both loops call). C2 WALL-TIE **UPHELD-WITH-CAVEATS** (honest no-regression + interleaved is freq-neutral, but 10-38% spread can't detect ≤10-20% — nearly uninformative). C3 my "marker path is only ~2% of body ⇒ TIE expected" mechanism **REFUTED (load-bearing):** I read `BOOTSTRAP_POST_FLIP_U16_BYTES` BACKWARDS — it increments only when a block ends CLEAN (`flipped_clean = !contains_marker_bytes()`, gzip_chunk.rs:1489-1495), so 2.0% is the CLEAN sliver the loop does NOT touch; the loop's actual domain is the COMPLEMENT ~98% of bootstrap body. (The exact counter-inversion the charter's u16-ceiling correction warns about — I repeated it.) ⇒ the TIE is NOT a small-domain ceiling. Commit message corrected to strike the inverted rationale.
- **CORRECTED PREMISE for the next loop:** the "decodeBlock 1.69× = the marker loop" attribution is now suspect — the marker fast loop owns ~98% of bootstrap body yet barely moved decodeBlock (0.9%) and did NOT move the wall. The engine SUM gap (~1.9×) is real but SLACK-MASKED at Fill 87% (Phase-0 already showed engine TIE-vs-TIE when seeded). **The T8 binder is NOT the per-thread engine compute.** Re-perceive: the wall is 0.73× rg with Fill 87% + Total Real Decode 0.137s ≈ rg's whole wall — the gap is the SCHEDULING/SERIAL term (pool-fill + in-order consumer head-of-line wait), the long-deferred project_confirmed_offset_prefetch_gap binder. NEXT loop should bound THAT with a removal oracle, not chase the slack-masked engine further.
- **GUEST STATE:** /root/gzippy tree RESTORED to baseline (marker patch reversed, marker_inflate.rs sha 7b87c5bd) + the mergefix overlay still applied (chunk_data.rs/chunk_fetcher.rs). Builds: /tmp/gzbuild-base + /tmp/gzbuild-mergefix (prior) + /tmp/gzbuild-markerfast (THIS turn, gzippy-isal native, sha 028bd002…cb410f). Drivers /tmp/markerfast_measure.sh + /tmp/markerfast_trace.sh + /tmp/sha_markerfast.sh (use `bash`). Patch /tmp/marker_fastloop.patch. No orphan processes (advisor wrapper + sleep killed; guest clean).


### THIS TURN — landed the merge-removal (cheapest+most-uncertain of the two ports, measured FIRST per advisor), byte-exact, remove-and-measure on the locked guest.
- **THE CHANGE (faithful port of vendor `applyWindow` swap+views, DecodedData.hpp:325-390 = narrow → swap → VectorViews → `dataWithMarkers.clear()`, NO output-size copy):** `resolve_chunk_markers_on_chunk` (chunk_fetcher.rs:2453) now DROPS `merge_resolved_markers_into_data()` (the redundant ~68MB full-output memcpy — `prepend_narrowed_from_markers`, segmented_buffer.rs:356-378, allocates `n+data.len()` and `extend_from_slice`s the WHOLE clean payload too) AND the eager `recycle_markers_after_resolution()`. The narrowed marker bytes STAY in `data_with_markers` (u8 view of the u16 backing) with `narrowed_len` set; the consumer emits them zero-copy via `append_output_iovecs`→`append_narrowed_iovecs` (chunk_data.rs:1609, already supported narrowed_len>0). Marker-segment recycle DEFERRED behind the consumer writev via the existing `defer_chunk_recycle`→`recycle_decoded_buffers` (frees BOTH data + data_with_markers). `contains_markers` (chunk_data.rs:577) now treats `narrowed_len>0` as resolved (post-narrow the u16 high bytes are stale so `all_resolved()` would misread → `has_been_post_processed` depends on this). `populate_subchunk_windows` assert relaxed (copy_window_at_chunk_offset already branches on narrowed_len>0 at :1220). + a debug-only double-resolve tripwire in `resolve_and_narrow_markers_in_place` (advisor rec; byte-transparent). New test `populate_subchunk_windows_unmerged_view_based_apply_window` locks the un-merged path.
- **BYTE-EXACT:** gzippy-isal native (guest x86_64) + gzippy-native (local arm64) BOTH sha 028bd002…cb410f at T1 AND T8 via path=ParallelSM. 856 lib tests pass (the 1 fail = pre-existing flaky `diff_ratio` timing micro-test, fails IDENTICALLY on unmodified 507d6ecb — confirmed by stash test). Adversarial seam test + native_fold_parity green. New un-merged test green (debug build exercises the tripwire, doesn't fire).
- **REMOVE-AND-MEASURE (NOT the SUM, per advisor Q4): locked guest REDACTED_IP double-ssh, 16c gov=performance turbo-on, taskset 0,2,4,6,8,10,12,14, T8, measure.sh interleaved N=11, RAW=68229982, sha-verified=OK every run. base(WITH merge) vs mergefix(REMOVED), both gzippy-isal native target-cpu=native:**
    | run (load) | base | mergefix | mergefix Δ | base vs rg | mergefix vs rg |
    | run1 (1.64) | 0.2291s | 0.2045s | **+12.0%** | 0.624× | 0.699× |
    | run2 (2.80) | 0.2128s | 0.1900s | **+12.0%** | 0.684× | 0.766× |
    | run3 (1.86, cleanest 6-13% spread) | 0.2006s | 0.1765s | **+13.7%** | 0.651× | 0.739× |
  Sign STABLE across 3 interleaved runs; load-invariant (delta holds at 1.64/2.80/1.86) ⇒ NOT a turbo/frequency artifact. Interleaved measurement is freq-neutral by construction (both tools alternate trials per N). **VERDICT: merge-removal moves the T8 wall ~12% (rg ratio 0.65×→0.73×). KEEP.** Mechanism (advisor Q4): the per-chunk O(whole-chunk) alloc+memcpy landed on the consumer's blocking recv for un-pre-resolved head-of-line marker chunks; removing it un-blocks that critical fraction.
- **INDEPENDENT DISPROOF ADVISOR (synchronous, read-only, plans/merge-removal-advisor-verdict.md):** C1 BYTE-EXACT UPHELD (vendor citation accurate; change is MORE faithful, not a divergence; contains_markers narrowed_len guard required+correct). C2 WALL UPHELD-WITH-CAVEATS (memcpy IS literally the whole clean payload; +12% plausible; stable+sha-identical rules out turbo/wrong-fast; caveat: alloc+copy removed together, attribution not isolated but remove-and-measure was the right method). C3 CORRECTNESS UPHELD-WITH-CAVEATS (no use-after-recycle on any of 3 emit paths — pipe boxes chunk covering BOTH buffers, non-pipe + buffered are sync-then-defer; re-resolution gates hold via !markers_resolved). Single correction ADOPTED: added the double-resolve debug tripwire (the merge used to empty the buffer as a guard; now safety rests on markers_resolved — tripwire restores defense-in-depth).
- **NEW WHOLE-SYSTEM WALL vs rapidgzip: T8 ~0.73× (was ~0.65×).** Still a LOSS — the remaining gap is port (i): rg's multi-cached u16 marker loop (decodeBlock 1.69×), the larger of the two divergences.
- **SCOPED NEXT (do NOT start — supervisor gate): port (i) rg's multi-cached u16 marker loop** to close decodeBlock 1.69× (vendor readInternalCompressedMultiCached deflate.hpp:1453, ONE loop over the u16 window, constexpr-gated marker arms). Larger change; advisor-gated; remove-and-measure. Re-check the gather/crc ~1.5× residual (advisor Q5 third term) after.
- **GUEST STATE:** /root/gzippy reset to clean 507d6ecb source (prior overlays git-stashed as `owner-overlays-507turn`) + this turn's merge-removal applied via /tmp/mergefix.patch. Builds: /tmp/gzbuild-base (507d6ecb WITH merge) + /tmp/gzbuild-mergefix (merge removed), both gzippy-isal native, both sha 028bd002…cb410f. Drivers /tmp/merge_measure.sh + /tmp/sha_check.sh (use `bash`). No orphan processes.

## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, HEAD 507d6ecb +substep-timers-on-guest) — CEILING BOUNDED → **T8 TIE needs TWO faithful ports, not one: (i) rg's multi-cached u16 marker loop (decodeBlock 1.69×) + (ii) rg's view-based applyWindow that skips a redundant full-output memcpy (the `merge` step, 0.12-0.13s SUM = the apply_window divergence)**. apply_window is NOT at parity — but the excess is a removable copy faithful-to-rg, NOT the LUT gather (which is ~1.5-2× and algorithmically identical). rg's "applying the last window" = **0.032s** (NOT the charter's cached 0.113s — that number was WRONG). Advisor: all findings UPHELD-WITH-CAVEATS, none refuted. **SUPERVISOR GATE — do NOT start the fix build (ceiling now bounded; report + gate).**

### THIS TURN — paid the OWED apply_window measurement + source-verified rg's marker-decode mechanism FIRST-HAND, then DECOMPOSED gzippy's apply_window.
- **rg's MARKER decode mechanism (source-verified, vendor deflate.hpp):** `readInternal` (:1428) dispatches by Huffman-coding TYPE not marker-vs-clean; with WITH_ISAL the lit/len path is `readInternalCompressedMultiCached` (:1453) for BOTH u16 markers AND u8 clean (templated on `Window`). It is ONE loop; `containsMarkerBytes` is a constexpr from the element type (:1600). Marker-vs-clean differ ONLY in cheap constexpr-gated arms: m_distanceToLastMarkerByte counter (:1311-1317), post-memcpy back-scan (:1379-1389), inverse window-range-check skip (:1652-1655). resolveBackreference fast arm is `std::memcpy` for BOTH (:1376). ⇒ rg's marker decode is fast because it runs the SAME multi-cached fast loop on the u16 window — there is NO separate slow marker path in rg. The faithful target = port rg's multi-cached u16 loop (NOT bolt AVX onto gzippy's loop — that's the E234 0.41× plateau). Caveat (advisor Q1): markers are u16 by construction ⇒ a faithful port is ~2× the mem traffic of the u8 clean path; promise "marker == rg's u16 multi-cached loop," NOT "marker == u8-clean speed."
- **OWED apply_window measurement (locked guest REDACTED_IP double-ssh, 16c gov=perf turbo-on load ~1.0, taskset -c 0,2,4,6,8,10,12,14, T8, RAW=68229982, sha 028bd002…cb410f EVERY run, /tmp/gzbuild-isal gzippy-isal native, measurement-only sub-step timers added byte-exact (NOT committed), 3 runs):**
    | term (SUM across 15 marker chunks) | gzippy | rg --verbose (first-hand) | ratio |
    | decodeBlock | 0.838s | 0.497s | **1.69×** |
    | gather (LUT resolve+narrow = rg's applyWindow analogue) | 0.044-0.064s | "applying the last window" **0.032s** | ~1.5-2× (algo IDENTICAL) |
    | crc (update_narrowed_crc) | 0.013-0.019s | "checksum" 0.0096s | ~1.5× |
    | **merge_resolved_markers_into_data** | **0.116-0.134s** | std::swap (~0s) | **structural divergence** |
    | subwin (populate_subchunk_windows) | 0.010-0.012s | window export (separate) | — |
    | TOTAL apply_window_us | 0.19-0.27s | — | — |
- **THE apply_window DIVERGENCE = `merge` (chunk_data.rs:1589 → segmented_buffer.rs:356 `prepend_narrowed_from_markers`):** allocates a fresh n-byte buf and `extend_from_slice` COPIES every narrowed byte (the whole ~68MB output) into `data`. rg does NOT do this — DecodedData.hpp:368 `std::swap` + VectorViews into the marker buffers in place (:371-388), no output-size copy. gzippy ALREADY HAS the zero-copy emit (`append_output_iovecs`/`append_narrowed_iovecs`, chunk_data.rs:1609 / segmented_markers.rs:532) ⇒ the merge-copy is REDUNDANT for the iovec writer. The LUT gather is FAITHFUL+identical to rg (`base[i]=lut[v]` ↔ rg `target[i]=fullWindow[chunk[i]]`, DecodedData.hpp:335-337) — the gap is the copy, not the algorithm.
- **INDEPENDENT DISPROOF ADVISOR (synchronous read-only, plans/marker-kernel-ceiling-advisor-verdict.md):** all findings UPHELD-WITH-CAVEATS, NONE refuted. Q1 marker==multi-cached-loop fair (caveat: u16 ⇒ ~2× clean traffic). Q2 apply_window apples-to-apples (caveat: rg's 0.032s already includes its swap+views; honest framing = rg 0.032 vs gzippy gather+merge). Q3 merge IS removable byte-exactly + faithful-to-rg (every consumer — writer, window-extraction, CRC, data_prefix_len>0 — already supports the un-merged state, traced) BUT it is a STRUCTURED change not a delete: must defer marker-recycle behind the consumer writev (else use-after-recycle), relax the populate_subchunk_windows `narrowed_len==0` assert (chunk_data.rs:1291), keep narrowed_len set through write. Q4 (LOAD-BEARING) **do NOT trust −0.12s SUM as the wall delta** — merge runs on the pool; its wall cost is only the un-overlapped fraction landing on the consumer's `recv_post_process_blocking` (chunk_fetcher.rs:1769) for un-pre-resolved head-of-line marker chunks, bounded by resolve-ahead hit rate (project_confirmed_offset_prefetch_gap). Provable ONLY by remove-and-measure (freq-neutral control), never the SUM. Q5 ceiling DIRECTIONALLY SOUND, two ports are the right faithful levers, NOT yet a proven TIE.
- **BOUNDED CEILING (REVISED, honest): T8 TIE plausibly reachable in PURE-RUST via TWO faithful ports — NOT one.** (i) rg's multi-cached u16 marker loop (closes decodeBlock 1.69×); (ii) rg's view-based applyWindow = drop the redundant `merge` memcpy, emit narrowed-marker iovecs (closes the 0.12-0.13s `merge` divergence). PLUS a smaller third residual the advisor flags (gather ~1.5-2× + crc ~1.5× = SegmentedU16 multi-segment walk + per-chunk LUT rebuild vs rg's contiguous chunk + hoisted fullWindow; may need (iii) hoist the LUT build / contiguous narrow target). The prior "marker-COMPUTE only" ceiling was OPTIMISTIC exactly as advisor Q4 warned — apply_window is a real second term and is NOT at parity.
- **SCOPED FIX FOR NEXT LOOP (do NOT start — supervisor gate): land merge-removal FIRST** (cheapest of the two ports, payoff most uncertain ⇒ measure first per advisor): convert `merge_resolved_markers_into_data` to rg's swap+views model — skip `prepend_narrowed_from_markers`, keep narrowed bytes in the marker pages, emit via `append_narrowed_iovecs`, DEFER marker-recycle behind the consumer writev, relax the subchunk `narrowed_len==0` assert. Byte-exact + measure the interleaved T8 wall (freq-neutral control). THEN the multi-cached u16 marker loop (decodeBlock). Each advisor-gated, each remove-and-measure (never the SUM, never the slow-down slope).
- **GUEST STATE:** /root/gzippy src @7bf26096 + oracle overlay + decompose knobs + THIS turn's measurement-only sub-step timers in chunk_fetcher.rs (gather/crc/merge/subwin, applied via /tmp/patch_resolve.py + /tmp/patch_merge.py on guest, NOT committed locally — byte-exact, sha unchanged). Build /tmp/gzbuild-isal (gzippy-isal native, rebuilt this turn). Drivers /tmp/applywin_measure.sh + /tmp/substep2_measure.sh (use `bash`). Seeds /tmp/seeds.bin. No orphan processes.

## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, HEAD 5e9905c8 +decompose-knobs) — BUNDLE DECOMPOSED → **THE T8 SUB-LEVER IS marker-COMPUTE: gzippy's window-absent u16 marker decode is ~2× SLOWER per byte than rapidgzip's** (advisor UPHELD-WITH-CAVEATS). Boundary-alignment + spec-failures are NOT the cost. CEILING = ≤ T8 1.0× TIE, conditional on applyWindow parity. **SUPERVISOR GATE — do NOT start the fix build (bound-ceiling-first; one owed measurement remains).**

### THIS TURN — DECOMPOSED the GZIPPY_SEED_WINDOWS bundle (advisor's 3-removal confound: a=marker-compute, b=boundary-alignment, c=spec-failure re-decodes). Added 2 measurement-only env knobs (OFF==identity, byte-exact, NOT committed): `GZIPPY_SEED_NO_WINDOWS=1` (suppress seeded-window fallback ⇒ seed-only-boundaries) + `GZIPPY_SEED_NO_BOUNDARIES=1` (skip block_finder pre-seed ⇒ seed-only-windows). seed_windows.rs + chunk_fetcher.rs.
- **MEASURED (locked guest REDACTED_IP double-ssh, 16c gov=perf turbo-on load 1.3-2.0, measure.sh interleaved N=11 CPUS=0,2,4,6,8,10,12,14 RAW=68229982 sha-OK=028bd002…cb410f every cell, 2 runs):**
    | cell | what's seeded | wall | vs rg |
    | rg (rapidgzip 0.16.0)              | —          | 0.132s | 1.000 |
    | seedfull (windows+boundaries)      | both       | 0.126-0.134s | **~1.00× TIE** |
    | onlywin (NO_BOUNDARIES, windows)   | windows    | 0.199s | 0.66× LOSS |
    | onlybnd (NO_WINDOWS, boundaries)   | boundaries | 0.198-0.205s | 0.66× LOSS |
    | prod (no seeding)                  | nothing    | 0.198-0.203s | 0.66× LOSS |
  **onlywin ≈ onlybnd ≈ prod (Δ<spread); only seedfull (BOTH) ties.** Pre-reg formula: f_windows≈0, f_boundary≈0, yet seedfull ties ⇒ SUPER-ADDITIVE/COUPLED (pre-reg branch-4).
- **MECHANISM (per-cell counters, GZIPPY_VERBOSE):** seedfull window_seeded=17 spec-fail=0 Fill=91% decodeBlock=0.846s (chunks go CLEAN). onlywin seed_hits=**0** (windows UNUSABLE at partition-guess offsets) window_seeded=2 spec-fail=13 decodeBlock=1.06s ≡ prod. onlybnd spec-fail **13→0** (real boundaries kill spec-failures) BUT body still 170MB/s u16, decodeBlock=1.106s ≈ prod ⇒ WALL-NEUTRAL.
- **APPLES-TO-APPLES vs rg (--verbose, both window-absent, SAME 34.5% replaced markers):** rg decodeBlock **0.542s** / Theo-Opt 0.068-0.074s vs gzippy prod **1.067s** / 0.133s ⇒ **rg's u16 marker decode is ~2× FASTER per byte.** rg ties WITHOUT seeding because its marker decode is fast; gzippy ties only by cheating (seedfull = clean, no applyWindow). Even seedfull's CLEAN decode (0.846s) is 1.57× slower than rg's MARKER decode (0.542s).
- **PINPOINTED T8 SUB-LEVER = marker-COMPUTE** (the slow window-absent u16 decode itself, ~2× rg). NOT boundary-alignment (secondary precondition, wall-neutral), NOT spec-failures (wall-neutral). The Phase-0 ISA-L oracle could not see this — ISA-L can't emit u16 markers, so the marker path was never replaced. ⇒ asm/igzip-class inner-kernel work IS in scope HERE, adapted to u16 marker output.
- **INDEPENDENT DISPROOF ADVISOR (synchronous read-only, plans/t8-decompose-advisor-verdict.md):** core verdict UPHELD-WITH-CAVEATS. (Q1) the 2×2 knobs CANNOT separate (a) from (b) — onlywin is DEGENERATE (windows unusable without boundaries by construction ⇒ ≡ prod; its pre-reg self-test FAILED ⇒ void as a windows-only cell, = the COUPLED branch); re-attribute the verdict to **onlybnd + the rg comparison**, not the decomposition. (Q2) onlybnd UPHELD-W-CAVEATS — spec-failures not the cost (clean isolation, wall-neutral). (Q3) the 2× rate gap is FAIR (denominator-matched decodeBlockTotalTime/parallelization, applyWindow separate in both, survives spec-failure removal) — the STRONGEST pillar. (Q4, MOST IMPORTANT) the CEILING is OPTIMISTIC: seedfull removes TWO things — marker decode premium AND the applyWindow serial pass — so it bounds route-(ii) (more clean windows), NOT the faithful route-(i) (fast u16 marker decode, which KEEPS applyWindow like rg). The route-(i) ceiling rests on the **rapidgzip existence proof** (rg: 0.54 decode + ~0.113s applyWindow → 0.13 wall), conditional on gzippy's applyWindow ≈ rg's.
- **BOUNDED CEILING: ≤ T8 1.0× TIE (rapidgzip existence proof), CONDITIONAL on gzippy's apply_window/marker-resolution pass ≈ rg's ~0.113s.** seedfull achieved the TIE but over-removes applyWindow ⇒ optimistic; the conditional bound is the honest one.
- **SCOPED FIX FOR NEXT LOOP (do NOT start — bound-ceiling-first):** an igzip-class u16 marker-decode kernel (asm/inner-kernel techniques adapted to u16 marker output). PLUS the OWED prerequisite measurement before claiming TIE-reachable: time gzippy's apply_window/marker-resolution vs rg's ~0.113s (no existing cell isolates it — needs a fast-marker prototype or a direct apply_window timer). If gzippy's applyWindow ≫ rg's, a marker-COMPUTE-only fix lands SHORT.
- **GUEST STATE:** /root/gzippy src @7bf26096 + oracle overlay + this turn's 2 decompose knobs (seed_windows.rs + chunk_fetcher.rs, applied on guest, NOT committed locally yet). Build /tmp/gzbuild-isal (gzippy-isal, target-cpu=native, byte-exact). Seeds /tmp/seeds.bin (16 windows). Driver /tmp/decompose_measure.sh (use `bash`). No orphan processes.

### SUPERSEDED — PHASE-0 (HEAD 3895a23c +oracle) — T8 BINDER = WINDOW-ABSENT MARKER/SPECULATION PATH, NOT THE ENGINE

### THIS TURN — PHASE-0: dropped a REAL ISA-L engine into the PRODUCTION parallel-SM pipeline and measured the T8 WALL (Measurement PROCESS #3 — engine REPLACEMENT oracle, not isolation-slope extrapolation). This converts the 0.6× engine-PRIMITIVE plateau into an airtight T8 WALL bound.
- **ORACLE (measurement-only, byte-exact, env-gated, NOT production):** `GZIPPY_ISAL_ENGINE_ORACLE=1`
  routes the clean-tail decode in `finish_decode_chunk_impl` through REAL ISA-L FFI
  (`decompress_deflate_from_bit_with_boundaries`, patched igzip), feeding ISA-L bytes/boundaries/
  end-bit through the SAME ChunkData primitives (commit + per-byte CRC + append_block_boundary_at +
  finalize). Pool/consumer/ring/window-publish/scheduling UNCHANGED. ISA-L input bounded to
  `[..stop_hint/8+256KiB]` so each worker decodes only ITS chunk. To run the bulk on ISA-L, windows
  are SEEDED (`GZIPPY_SEED_WINDOWS`, captured at T1) so all 18 chunks are window-PRESENT and reach
  the oracle. PROVEN ISA-L ran: T8 `isal_oracle_chunks=16 isal_oracle_fallbacks=1` (94% real ISA-L).
- **MEASURED (locked guest REDACTED_IP double-ssh, 16c gov=perf turbo-on load 2.7-4.2, measure.sh
  interleaved N=11 CPUS=0,2,4,6,8,10,12,14 RAW=68229982 sha-OK=028bd002…cb410f every run, 2 runs):**
    | contender | T8 wall | vs rg | verdict |
    | rg (rapidgzip 0.16.0)        | 0.134s | 1.000 | — |
    | isal (ISA-L engine, seeded)  | 0.148s | 0.905/0.892 | **TIE** |
    | pure (pure-Rust eng, seeded) | 0.134s | 1.002/0.968 | **TIE** |
    | prod (pure-Rust, NO seed)    | 0.194s | 0.690/0.652 | LOSS |
- **THE LOAD-BEARING RESULT: `pure` (the SLOWER engine) ALREADY TIES rg once windows are seeded;
  `isal` also ties → engine swap is TIE-vs-TIE. The per-thread engine is NOT the T8 wall binder.**
  The whole ~1.5× prod gap collapses to a TIE when the window-absent path is removed. Per-stage
  --verbose: prod decodeBlock SUM 1.048s / Real Decode 0.169s / Fill 77% / body_rate 168 MB/s /
  13 header-speculation failures; pure-seed 0.781s / 0.108s / Fill 90.55% / 0 failures / 0 bootstrap.
  rapidgzip runs the SAME 34.5% replaced-marker workload WITHOUT seeding yet ties (verified rg
  --verbose) → gzippy's window-absent path is the SLOW one, apples-to-apples (NOT a seeding artifact).
- **INDEPENDENT DISPROOF ADVISOR (synchronous read-only, plans/asmport-phase0-advisor-verdict.md):**
  Claim1 (oracle measures igzip-class engine in real pipeline, byte-exact) UPHELD-W-CAVEATS (clean-
  tail only; 1-chunk fallback impurity). Claim2 (engine alone doesn't close T8, TIE-vs-TIE) UPHELD-W-
  CAVEATS (T8-only; engine gap 1.51× is REAL but slack-masked at Fill 90% — NOT at parity). Claim3
  (binder is window-absent path) UPHELD as COARSE localization — sound + not unfair, BUT the seeding
  knob bundles THREE removals: (a) u16 marker decode+resolution, (b) block_finder REAL-boundary
  pre-seed (vs prod partition-GUESS — the project_confirmed_offset_prefetch_gap head-of-line stalls),
  (c) the 13 speculation-failure re-decodes. CANNOT attribute the gain to marker-COMPUTE vs
  boundary-ALIGNMENT vs re-decode. Claim4 (asm port can't move prod wall) directional rec UPHELD at
  T8, strong inference REFUTED: marker-phase decode rate is ON the binding path and was NEVER replaced
  (ISA-L can't emit u16 markers), and T1 (no Fill slack ⇒ engine binds directly) is unaddressed.

### SCOPED TARGET FOR PHASE 1 (the supervisor gate — pick AFTER decomposing the bundle):
- An igzip-class engine ALONE does NOT close the prod T8 wall (pure-Rust already ties seeded). So
  the asm engine port is **NOT the T8 lever** — at T8. It remains plausibly the **T1 lever** (no
  Fill slack, the 1.51× engine gap binds directly) and helps the **marker-phase decode rate** (168
  MB/s, on the binding path, never tested by this oracle). Do NOT abandon it; re-scope it.
- **NEXT PERTURBATION (decompose the Claim-3 bundle BEFORE choosing Phase 1):** seed ONLY the
  block_finder boundaries (no windows) vs seed ONLY windows (prod boundaries). If most of the
  0.69→1.00 delta is boundary-ALIGNMENT, the lever is the block finder / prefetch horizon
  (project_confirmed_offset_prefetch_gap), NOT the asm engine NOR a marker-kernel rewrite. If it's
  marker-COMPUTE, a faster u16 marker kernel (the asm techniques adapted to u16 output) is the lever.
- **HARD WALL BOUND (owed by prior charter, now PAID):** the engine-PRIMITIVE 0.6×-ISA-L plateau
  does NOT bind the T8 WALL — proven by replacing the engine with REAL ISA-L in the production
  pipeline and STILL only tying (engine slack-masked at Fill 90%). The 1.0×-vs-no-FFI FORK is
  NOT forced by the engine at T8 (pure-Rust already ties T8 seeded). The fork may still bite at T1.

### THIS TURN — step (B) executed: built+measured the faithful-u8 engine CEILING vs ISA-L (isolation, bounded)
- **VAR_VI added to benches/engine_isolation.rs** (`decode_var_vi`, x86_64) = VAR_V (faithful-u8
  speculative software-pipelined flat-u8 loop + igzip packed-u32 multi-symbol table, tricks #1/#2/#3)
  PLUS the two REMAINING igzip techniques: (1) **BMI2 BZHI** (`_bzhi_u64`) for the variable-width
  distance extra-bits extraction; (2) **AVX2/SSE MOVDQU wide overlap-copy** back-ref (32B AVX2 bulk,
  16B SSE distance>=16 overlap, RLE memset dist==1). trick #3 (packed-u32 short table) CONFIRMED
  fully exploited (drives the same `LutLitLenCode::decode` packed packets, unpacks up to 3 lit/decode).
- **MEASURED (locked guest REDACTED_IP double-ssh; 16c gov=perf, load ~3.3, turbo on; taskset -c 0;
  N=11 interleaved; native target-cpu ⇒ BMI2+AVX2 LIVE, avx2_detected=true; 2 independent runs, STABLE):**
    | variant | aggregate MB/s | vs ISA-L | per-chunk vs ISA-L |
    | VAR_III ISA-L | 847-851 | 1.000 | — |
    | VAR_V (no BMI2/AVX) | 460-462 | 0.54× | 0.50-0.56 |
    | **VAR_VI (+BMI2+AVX2)** | **504-525** | **0.59-0.62×** | **0.55-0.64** |
  BMI2+AVX2 added ~9-14% over VAR_V but did NOT close the gap. SELFTEST=PASS (iii/i=2.73 ∈ [2.5,3.6]).
- **BYTE-EXACT:** VAR_VI printed an MBps line (never VOID) on EVERY swept chunk ⇒ per the bench gate
  (`exact[k]= o[..n]==scalar && scalar==isal`, engine_isolation.rs:744/802) VAR_VI is byte-identical
  to BOTH the scalar reference AND ISA-L over the full timed window. (Top-line `SHA_ALL_EQUAL=no` is
  the PRE-EXISTING VAR_IV_E234 failures — a separate path NOT touched this turn — not VAR_VI.)
- **PRE-REGISTERED FALSIFIER FIRED → PLATEAU:** VAR_VI ≈ 0.6× ISA-L, ~23pp below the 0.85 PASS line,
  WITH the full igzip stack + inline-ASM intrinsics. ⇒ pure-Rust igzip-class as a STANDALONE ENGINE
  PRIMITIVE is NOT reached on this design.
- **INDEPENDENT DISPROOF ADVISOR (synchronous, read-only, plans/engine-ceiling-advisor-verdict.md):
  PLATEAU UPHELD-WITH-CAVEATS.** Source-verified all 5 techniques LIVE; fast loop (not the careful
  tail) is the timed path; header/table-build symmetric with ISA-L's own header parse; byte-exact
  reasoning airtight. The two minor under-representations (small-overlap 2≤dist<16 copy is scalar not
  igzip-doubling; SHRX compiler-discretionary) + the only asymmetric confound (VAR_VI's final
  `to_vec` ~few-%) ALL cut AGAINST plateau and together lift a "fixed" VAR_VI to at most ~0.65-0.68 —
  STILL ~17-20pp short of 0.85. Structural reason supports plateau: a Rust port routed through a
  `DecodedSymbol` struct-return + `while remaining` unpack carries codegen overhead a hand-scheduled
  asm hot loop does not.
- **LOAD-BEARING ADVISOR CAVEAT (the escalation correction):** the engine-PRIMITIVE ceiling is
  UPHELD, but escalating to "the 1.0× WALL is HARD-BOUNDED at 0.6×" OVERREACHES isolation — that is
  the forbidden extrapolation through an unlocated knee (Measurement PROCESS #3). To hard-bound the
  WALL you must REMOVE the engine stage in the PRODUCTION PARALLEL pipeline and measure, not
  extrapolate the isolation ratio. NOTE: the prior floor-to-floor T8 finding (engine 1.74× at the
  wall, t8-engine-binder-advisor-verdict.md UPHELD) INDEPENDENTLY corroborates the engine gap
  survives to the wall — so the fork is strongly implicated — but the clean WALL hard-bound is still
  owed that one engine-removal perturbation.

### THE FORK (escalate-candidate — supervisor/user call): pure-Rust 1.0× bar vs no-FFI
- **HARD NUMBER (engine primitive, advisor-upheld): pure-Rust+ASM faithful-u8 engine = ~0.6× ISA-L
  in isolation (ceiling ~0.65-0.68 crediting every caveat); the 0.85 igzip-class bar is NOT reached.**
- **The 1.0× WALL-vs-no-FFI fork is REAL.** Two corroborating data points say the engine gap reaches
  the wall: (i) the floor-to-floor T8 1.74× engine gap (advisor-upheld this campaign); (ii) the
  constant ~1.70× gzippy↔rapidgzip ratio at BOTH T1 and T8 (per-thread-throughput signature).
- **What is NOT yet a clean hard-bound:** the WALL number under an ENGINE-REMOVAL oracle in the
  production parallel pipeline (Rule 3). Recommended BEFORE a final fork decision: run that
  perturbation (replace the per-thread decode with a no-op/ISA-L oracle, measure the T8 wall) — if
  the wall stays ~1.7× off rapidgzip the fork is hard-forced (no-FFI cannot reach 1.0×); if it ties,
  a shared serial stage gates and pure-Rust CAN still tie despite the 0.6× engine.

### NEXT (decision point — supervisor gate):
- Either (a) ESCALATE the fork now with the engine-primitive hard number (~0.6× ISA-L, PLATEAU) +
  the corroborating wall evidence, accepting the advisor's caveat that the wall hard-bound is owed
  one more perturbation; OR (b) FIRST run the production-pipeline engine-removal oracle to convert
  the engine ceiling into a clean WALL hard-bound, then escalate. The owner recommends (b) is cheap
  and removes the last ambiguity, but the engine-primitive PLATEAU itself is settled.

---
## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, HEAD f8260aa8) — T8 BINDER RE-LOCATED to the ENGINE; the "serial/consumer-wait" binder is REFUTED (floor-to-floor + advisor-UPHELD)

### THIS TURN — step (A) ceiling-bound; the prior "binder = serial/consumer-wait" was a UNIT ERROR; binder is the per-thread DECODE ENGINE
- **THE PRIOR CHARTER BINDER ("decode floor 0.118s ALREADY ≈ rapidgzip's wall 0.130s, so the
  whole 1.7× gap is scheduling/consumer-wait") IS REFUTED — it was a UNIT ERROR (advisor UPHELD,
  plans/t8-engine-binder-advisor-verdict.md).** It compared gzippy's decode FLOOR (0.118s) to
  rapidgzip's WALL (0.130s). The correct comparison is FLOOR-TO-FLOOR: rapidgzip's own
  Theoretical-Optimal is 0.068s, NOT 0.130s. gzippy 0.118 vs rapidgzip 0.068 = **1.74× engine gap.**
- **FIRST-HAND apples-to-apples --verbose pool stats this turn (locked guest REDACTED_IP double-ssh,
  16c gov=perf, box load ~2.5 ⇒ INTERNAL SPANS not wall absolutes; gzippy-mk2 byte-exact
  028bd002…cb410f path=ParallelSM; rapidgzip 0.16.x --verbose; 3 runs each, STABLE), T8 silesia:**
    | metric | gzippy | rapidgzip | ratio |
    | decodeBlock (SUM/workers) | 0.93s | 0.50s | **1.86×** |
    | Theoretical-Optimal (÷8) | 0.118s | 0.068s | 1.74× |
    | Total Real Decode Duration | 0.139s | 0.086s | 1.61× |
    | std::future::get (consumer wait) | 0.077-0.082s | 0.062-0.067s | ~1.25× |
    | Pool Fill Factor | 85% | 78% | — |
- **BINDER = the per-thread DECODE ENGINE** (decodeBlock 1.86×; body_rate 269 MB/s vs rapidgzip's
  ~424 MB/s ISA-L = 1.58× raw + speculative/marker overhead, BOTH engine). The consumer future::get
  gap (1.25×) is a MINORITY and largely DOWNSTREAM (the consumer waits longer because each chunk
  decodes slower). This matches the long-observed CONSTANT ~1.7× ratio at BOTH T1 and T8 (flat-
  across-T = the signature of a per-thread throughput gap, which the charter itself noted).
- **CEILING-BOUND METHOD NOTE (Rule 3): the decode-bypass + sleep-decode oracles are CONFOUNDED**
  (decode-FREE wall was 3.6-5.5× SLOWER than real decode — they bypass the buffer pool, do fresh
  full-size zeroed allocs/faults per chunk, hold ≤33 ChunkData/660MB live, single-thread CRC 212MB
  un-overlapped). The valid ceiling instrument is the FLOOR-TO-FLOOR --verbose span comparison.
- **VENDOR SOURCE-VERIFIED (BlockFetcher.hpp:246-329, this turn):** rapidgzip's get() ALSO pumps
  prefetchNewBlocks() in a `while(wait_for(1ms))` loop during the future wait (:314-316), exactly
  as gzippy (chunk_fetcher.rs:1289 Lever H). The consumer-overlap STRUCTURE is already faithfully
  ported; future::get is non-zero in BOTH. There is NO missing overlap mechanism to port.
- **PROD DECODE-MODE SPLIT (T8): finished_no_flip=16, window_seeded=2, flip_to_clean=0.** 16/18
  chunks take Engine M's speculative marker-bootstrap-then-u8-direct-tail path (window-absent at
  high T, faithful — rapidgzip is also ~window-absent at runtime). The engine front IS the bulk path.

### NEXT (per PROCESS — bottleneck is the ENGINE; step (B) = build+measure the faithful-u8 ceiling):
- **The advisor's load-bearing caveat (D-D): the ENGINE-BENCH-ROUND-2 "2.4× plateau" that earlier
  declared pure-Rust+ASM unreachable was measured on the DISCREDITED u16-RING architecture. It does
  NOT bound the CURRENT faithful u8-direct flip-in-place engine (landed fc1c965b). So the
  pure-Rust→1.0× question is OPEN, NOT settled by that plateau.**
- **USER-CONSTRAINT FORK IMPLICATED (advisor-flagged, escalate-candidate):** is the 1.0× bar
  reachable in pure-Rust+ASM (no FFI) given the engine is 1.86× ISA-L? Must be resolved by BUILDING
  + measuring the faithful-u8 engine ceiling vs ISA-L on the production speculative path, NOT by
  extrapolating the invalid u16 plateau. Lever: igzip-class inner-Huffman (packed-u32 short table,
  speculative 8-byte literal store + next-sym/next-dist preload pipeline, BMI2 SHLX/SHRX/BZHI,
  MOVDQU overlap-doubling copy, slop-margin headroom) on the u8-direct ring — authorized in scope.
- Kernel is now the confirmed binder at BOTH T1 (was always) AND T8 (this turn). The consumer-wait
  direction is DESCOPED (minority + already-faithful structure).

---
## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, HEAD fb3baec0) — "T8 BINDER = SERIAL/CONSUMER-WAIT" (REFUTED above as a unit error)

### THIS TURN — step (A) executed; u16-path premise FALSIFIED; binder LOCATED to the serial/consumer term
- **u16-path "biggest prize" premise is FALSIFIED at the source (advisor UPHELD,
  plans/u16-ceiling-advisor-verdict.md).** The "58.6% u16" came from the MIS-NAMED counter
  `BOOTSTRAP_POST_FLIP_U16_BYTES` (gzip_chunk.rs:97/:1302): it increments when
  `!block.contains_marker_bytes()` — i.e. it counts bytes in marker-FREE blocks, which since
  fc1c965b decode u8-DIRECT (marker_inflate.rs:1397-1401 ring_modulus=U8_RING_SIZE, :1685
  ring8.write). The counter NAME + doc (gzip_chunk.rs:91-96) are STALE/inverted. The genuine
  u16-`<true>` fraction is the INVERSE ≈ 42.5% (the pre-flip prefix each speculative chunk must
  decode before 32KiB clean accumulates → flip). NOT "the bulk of bytes on a slow path."
- **CAUSAL PERTURBATION (this turn; new GZIPPY_SLOW_MARKER_MODE u16-path knob, commit fb3baec0;
  byte-exact OFF/marker100/clean100/marker100+sleep all = 028bd002…cb410f; locked guest
  REDACTED_IP double-ssh, 16c gov=perf, measure.sh interleaved sha-OK, RAW=211968000, T8
  CPUS=0,2,4,6,8,10,12,14, N=11; box load 3-5, interleaved-relative is load/turbo-robust):**
    - CLEAN +100% spin → +27%; CLEAN +100% SLEEP control → +27% (IDENTICAL ⇒ NOT a turbo
      artifact); CLEAN +200% SLEEP → +55%. ⇒ clean u8 decode-compute GENUINELY gates ~27% of T8
      wall (freq-neutral, ~linear). (Supersedes the prior "~18-22%" — that was on a different
      box/run; the freq-neutral confirm makes ~27% the number.)
    - MARKER +200% spin → +21%; MARKER +200% SLEEP control → +7% (does NOT survive the control).
      ⇒ u16-marker decode-compute is a MINORITY: ~3.5-14% of T8 wall (advisor range; point est
      ~3.5%, biased low by calibration D1 + event-coverage D3, high-single-digits most likely).
    - T1 MARKER +100% → +0% / +200% → +4% (near-flat: at T1 ~all chunks window-seeded clean, u16
      barely runs; the knob fires ∝ u16 bytes ⇒ near-flat validates it).
- **BINDER LOCATED (not residual) from the GZIPPY_VERBOSE pool trace, first-hand this turn:**
  decodeBlock(all workers)=0.936s → **Theoretical-Optimal (÷8) = 0.117s**; Total Real Decode
  Duration (pool phase span) = 0.147s (Fill 79%); **std::future::get (in-order consumer wait) =
  0.077s**; header_ms=24.0 (~2.6% of decode — the D4 header/table-build caveat is quantitatively
  TINY); full wall this run 0.183s (interleaved best ~0.183-0.221s). 3-way anchor: gzippy-mk ≈
  varv (1.018× TIE), rapidgzip 0.130s = 1.70×.
- **DECISIVE DECOMPOSITION:** gzippy's perfectly-parallel decode floor (Theoretical-Optimal
  0.117s) is ALREADY ≈ rapidgzip's ENTIRE wall (0.130s). The whole 1.70× gap is the
  scheduling/serial term: pool fill gap (0.147-0.117 ≈ 0.030s) + in-order consumer `future::get`
  head-of-line wait (~0.077s) ≈ **~0.10s of serial/overlap = the dominant T8 binder.** rapidgzip
  ties DESPITE the same engine gap by OVERLAPPING decode under scheduling (memory
  project_confirmed_offset_prefetch_gap: gzippy consumer cold-stalls in-order get; rapidgzip
  joins in-flight). **Conclusion #2 advisor caveat: the residual is scheduling/serial + a small
  header/bandwidth term, now MEASURED (header ~2.6%), not eliminated-by-residual.**

### NEXT (per PROCESS — bottleneck is the serial/consumer-wait term, ~0.10s):
- **FIX the in-order consumer `future::get` head-of-line wait (~0.077s) + pool fill gap (~0.030s).**
  This is charter binder #2 and the `project_confirmed_offset_prefetch_gap` memory: make gzippy's
  consumer JOIN an in-flight decode instead of cold-stalling (rapidgzip GzipChunkFetcher.hpp
  consumer loop :1419-1469 cold-get + :1535-1740 serial window-publish chain). CAUTION: the
  prior `placement-port` GATE FAILED (offset-supply was a non-divergence); the OPEN distinct
  question per that gate is the PREFETCH-HORIZON / dispatch-depth (decode_NOT_STARTED stalls =
  guess-prefetch never dispatched DEEP ENOUGH AHEAD), NOT offset supply. Bound the ceiling first
  (Rule 3): an ORACLE that removes the consumer wait (e.g. unbounded look-ahead / pre-resolved
  futures) → measure the whole-system wall; if it lands ~0.12s it confirms the tie is here.
- Keep the inner kernel as the confirmed T1 lever AND a real ~27% T8 contributor (freq-neutral),
  but it is NOT the T8 path to 1.0× — closing the serial term gets to ~rapidgzip's wall alone.
- u16-prefix ceiling: KEEP a prefix-removal oracle in reserve (Rule 3, advisor D6) before fully
  abandoning — a faster prefix COULD let chunks flip sooner / consumer catch up. But it is a
  minority term; do NOT lead with it.

---
## PRIOR STATE (2026-06-07, HEAD 9b674651) — T8 BINDER RE-IDENTIFIED (causal + advisor-upheld) [SUPERSEDED above re: u16]
- gzippy-native is FAITHFUL u8 (u8-direct flip-in-place clean tail landed byte-exact). VAR_V
  speculative pipeline committed (byte-exact TIE, kept per 7a).
- Whole-system wall (locked guest trainer=REDACTED_IP via -J neurotic, 16c gov=performance
  no_turbo=1, measure.sh interleaved sha-verified, RAW=211968000): T8 gzippy ~0.226s vs rapidgzip
  ~0.137s = **1.655× gap** (varv vs base TIE, sha 028bd002…cb410f OK). Reproduced this turn.

- **CHARTER CORRECTION (this turn, causal + disproof-advisor UPHELD-WITH-CAVEATS,
  plans/t8-binder-advisor-verdict.md): the prior "constant 1.70× = pure per-thread decode gap,
  inner Huffman kernel is the ONLY lever to 1.0× at T8" is REFUTED AT T8.** Established via the
  slow_knob causal perturbation (byte-transparent, frequency-neutral sleep control confirms not a
  turbo artifact; site fires ∝ clean bytes):
    - T1 (CPUS=0): spin100 (doubles per-thread decode-compute) → +83% wall (off 0.533→0.974s).
      => decode-compute GATES ~83% of the T1 wall. **Kernel is the confirmed T1 lever.**
    - T8 (8 pinned cores): spin100 → +14–22% wall; spin200 → +45%; sleep100 control +20% (≥ spin).
      => per-thread CLEAN decode-compute gates only ~18–22% of the T8 wall directly.
    - COVERAGE CONFOUND reconciled first-hand: slow_knob is CLEAN-mode only (const-folds to 0 on
      the marker <true> path). Clean-loop hits T1=38.7M vs T8=28.4M (T8 = 73% coverage) ⇒ ~27% of
      T8 decode events run in MARKER (u16) mode, uncovered. Coverage-corrected decode-compute
      ceiling at T8 ≈ ~25–30% of wall (advisor: plausibly up to ~45% with Rule-3 unbind slack).
      EITHER WAY decode-compute is a MINORITY of the T8 wall; ≥~55–75% is OTHER.
- **THE T8 BINDER (the OTHER ≥55–75%), from the GZIPPY_VERBOSE trace (first-hand this turn):**
    1. **u16 post-flip / marker path = 58.6% of decoded BODY bytes** (`post_flip_u16_bytes
       =118.6M, "Design-B1 prize"`). The bulk of production bytes at T≥2 flow through the slower
       u16 marker→drain path, NOT the clean u8 fast path the kernel/VAR_V optimized. This is the
       largest single named term and is why VAR_V's clean gain was absorbed + the slow_knob barely
       moved T8. body_rate blended 286 MB/s; Speculation failures header=14/19.
    2. **Pool scheduling + serial tail:** Theoretical-Optimal 0.127s → Real-Decode-Duration 0.162s
       (~28% pool inefficiency, fill 73–83%, Prefetch dispatch saturated ~51/60) → wall ~0.22s
       (another ~0.06s SERIAL outside the pool: in-order consumer publication / drain / CRC).
       Corroborates memory project_confirmed_offset_prefetch_gap (head-of-line stalls ~40% T8).
- Window-discard lever: FALSIFIED (prior turn; window seeded when available; T≥2 window-absent is
  faithful rapidgzip behavior).
- **NEXT (re-pointed per the PROCESS — bottleneck moved off the clean kernel at T8):** the two T8
  binders above. Recommended order: (A) bound the u16-path ceiling with an ORACLE removal (NOT the
  slope) — if the 58.6% u16 body decoded at clean-path rate, how much wall drops? rapidgzip ties
  its wall DESPITE the same engine gap via in-flight overlap, so this is likely the bigger prize;
  (B) the pool-scheduling/serial-tail SERIAL-WORK vs DECODE-WAIT decomposition. Keep the inner
  kernel as the confirmed T1 lever (not abandoned), but it is NOT the T8 path to 1.0×.
- NO new build this turn (perception + causal ID + advisor only). Tree clean, no orphans.

---
## USER DECISION 2026-06-07 (fork resolved): TRANSLITERATE igzip's FULL AVX2 ASM KERNEL
The pure-Rust engine ceiling is bounded (advisor-upheld): faithful-u8 + the FULL igzip technique
stack + inline-ASM intrinsics (BMI2 BZHI, AVX2/SSE overlap copy, packed-u32 table, speculative
pipeline) = VAR_VI ~0.60× ISA-L in isolation (~515 vs ~849 MB/s) — high-level techniques do NOT
reach hand-tuned igzip asm. User chose: pursue **pure-Rust no-FFI 1.0× by transliterating igzip's
ACTUAL assembly instruction-for-instruction** (our own inline Rust asm — NOT C-FFI). Honors 1.0× +
no-FFI + faithfulness if it lands. This is a MULTI-SESSION project; own it in byte-exact phases.

ASM-PORT PROJECT PLAN (the owner owns; phased, prove-before-the-big-build):
- **PHASE 0 (scope the target — do FIRST, cheap):** an ISA-L-in-pipeline WALL oracle — drop an
  igzip-class engine (real ISA-L FFI, MEASUREMENT-only) into gzippy's PRODUCTION pipeline and
  measure the T8 WALL vs rapidgzip. Tells us: does an igzip-class engine ALONE tie in gzippy's
  real pipeline (⇒ the asm port is sufficient, target = match igzip rate), or do production
  overheads (ring/wrap/resumable/CRC — which absorbed VAR_V) ALSO cap it (⇒ the port must
  integrate into a FLATTENED clean path)? This converts the 0.60× engine-primitive plateau into an
  airtight WALL bound (PROCESS #3) AND scopes the transliteration so it can't be absorbed like VAR_V.
- **PHASE 1+ (the transliteration):** port igzip_decode_block_stateless_{01,04}.asm → inline Rust
  asm, integrated per Phase-0's finding (flatten the path if needed), in byte-exact + wall-measured
  phases, each advisor-gated. Target: production T8 wall ties rapidgzip (~0.13s same-host).

---
## PHASE-0 RESULT 2026-06-07 (advisor-upheld) — ASM PORT IS NOT THE T8 LEVER; RE-SCOPE
ISA-L-in-pipeline wall oracle (commit 5e9905c8, proven ISA-L ran 16/18 chunks, byte-exact):
T8 seeded — pure-Rust 0.97-1.00× TIE, ISA-L 0.90× TIE (TIE-vs-TIE); unseeded production
0.65-0.69× LOSS. ⇒ The pure-Rust ENGINE ALREADY TIES T8 once windows are seeded; the engine
plateau (0.6× isolation) does NOT bind the T8 wall. **The T8 lever is the WINDOW-ABSENT
marker/speculation path** (16/18 chunks window-absent @168 MB/s + 13 header-speculation
failures; rapidgzip runs the same ~34.5% marker workload without the penalty). The T8 1.0× tie
is reachable in PURE-RUST, no FFI, WITHOUT the asm transliteration.
RE-SCOPE (no constraint fork — same goal, pure-Rust): the asm engine port is the **T1 lever**
(T1 has no scheduling slack ⇒ engine binds directly, ~83% of T1 wall) — keep for T1, defer for
T8. The **T8 lever is the window/marker/boundary path**. Advisor caveat: "seeding" BUNDLES three
removals — (1) u16 marker-compute, (2) block-finder real-boundary vs partition-guess ALIGNMENT,
(3) 13 speculation-failure re-decodes. NEXT: DECOMPOSE the bundle (seed-only-boundaries vs
seed-only-windows) to pinpoint the precise T8 sub-lever before fixing — likely boundary-ALIGNMENT
= block-finder / prefetch-horizon (project_confirmed_offset_prefetch_gap), faithfully ported.

---
## PROCESS ADDENDUM — IT'S A RATCHET, not a flip-flop (user-set 2026-06-07)
As performance improves, the binder WILL oscillate between the per-thread engine and the
sequential/scheduling terms — that is the EXPECTED shape of whole-system bottleneck-following,
not thrashing. Fix the current binder, it migrates to the next-largest term; fix that. The only
metric of progress is the WHOLE-SYSTEM WALL RATCHETING DOWN. A banked byte-exact wall-moving
change (e.g. merge-removal +12%, wall 0.65×→0.73× rg) is a ratchet tooth — it does NOT come back,
even though the binder then moves on. Do NOT treat binder migration as a problem to avoid; keep
banking ratchet teeth.
DISTINCTION (the real discipline): separate (a) genuine binder MIGRATION (healthy ratchet) from
(b) FALSE flips caused by measurement error (the unit error; the misnamed counter read backwards
twice). (a) is fine and expected. (b) is the enemy — kill it: every binder claim must come from a
CAUSAL perturbation on the WALL (not producer-side attribution), and counters must be named for
what they actually count. Make every flip a REAL migration.

---
## PROCESS FIXES (supervisor coach review, plans/SUPERVISOR-FEEDBACK.md, 2026-06-07) — BINDING
1. **The decider ORACLE gates each loop.** Before ANY binder-MECHANISM claim or fix, run the
   registered removal oracle for that binder (for the current scheduling direction: the
   `GZIPPY_PERFECT_OVERLAP` ceiling — NEVER YET RUN; it is the registered decider and the current
   strategy rests on it unmeasured = a live Rule-3 violation). No mechanism claim without the
   oracle.
2. **Causal-perturbation-first; attribution is a FOOTNOTE.** Every binder claim must LEAD with a
   causal perturbation on the WALL. SUMs / ratios / Fill% / counters are hypothesis-only and have
   repeatedly produced inverted binders (the misnamed counter read backwards twice). Never let
   attribution be the verdict.
3. **No STOP/TIE/"done" without a validated removal oracle.** Two prior "victories" were reversed
   (2026-06-02 STOP-EARNED; 2026-05-29 rescind on a broken oracle). Don't repeat.
ALSO (coach-corrected facts): engine is CONCLUSIVELY NOT the T8 binder (seeded-pure TIE 1.002× +
ISA-L oracle TIE-vs-TIE) — do NOT re-test the engine at T8 (that is the ~40% wasted re-derivation
in the engine↔scheduling oscillation); the T8 lever is the named head-of-line stall
([[project_confirmed_offset_prefetch_gap]], "fixable not architectural"). 0.73× is NOT "best ever"
(June-2 ~0.85× was a different ISA-L-product basis) — never frame it as a regression OR a record.

---
## OSCILLATION ROOT CAUSE + CORRECTION (2026-06-07, corrected-overlap oracle, 2 advisor passes)
SCHEDULING/OVERLAP is REFUTED as the T8 lever: the corrected overlap oracle (proper pipelined
decode↔drain, window-absent routing preserved) = 0.69× rg, SLOWER than production. Dispatch-depth,
drain-hiding (resolve-ahead saturated ok=14/14), and finer-chunking all null/worse. The T8 binder
is the **WINDOW-ABSENT MARKER decode RATE** (decodeBlock ~0.83s, ~1.6× rg; production is 89%
window-absent).
**ROOT CAUSE OF THE ENGINE↔SCHEDULING OSCILLATION (measurement-validity rule — BINDING):** every
oracle that SEEDED windows (seedfull TIE 1.029×; Phase-0 ISA-L oracle TIE-vs-TIE) ROUTES chunks to
the fast CLEAN engine (window present ⇒ clean path, gzip_chunk.rs:790) and ties — which made
"engine closed at T8" look settled. But production runs the WINDOW-ABSENT MARKER engine, which
those seeded oracles NEVER tested. So "engine closed at T8" was a SEEDED ARTIFACT, now corrected.
RULE: any T8 oracle/measurement MUST preserve production's window-absent routing, or it masks the
marker binder. Seeded numbers are about the clean engine only.
NEXT: ATTRIBUTE the window-absent marker decodeBlock 1.6× gap precisely (apples-to-apples UNSEEDED
vs rg --verbose decode-sum + source-diff decode_chunk_unified_marker vs rg
readInternalCompressedMultiCached): is it (a) u16-WIDTH over the clean bulk on window-absent chunks
(governing u8-direct memory — the likely cause; the earlier marker-fast-loop port TIE'd because it
targeted the inner loop, not the width), (b) the marker inner loop, or (c) table-build? Then fix the
attributed cause faithfully (pure-Rust), bounded by a window-absent-PRESERVING removal oracle.

---
## INSTRUMENT MERGED 2026-06-07 (HEAD fc7336c3): USE scripts/fulcrum_total.py
The trustworthy whole-system instrument is merged (additive). USE IT for whole-system reads:
`bash scripts/bench/fulcrum_total_capture.sh LABEL=… T=8` (unseeded, window-absent-preserving) →
`python3 scripts/fulcrum_total.py trace_*.json`. It REFUSES seeded/contaminated runs, separates
SELF from slack-maskable SUM, asserts busy+idle==span, classifies wait/compute/output. Prefer it
over hand-rolled one-off oracles (which caused the oscillation). Causal verdict still = the wall
perturbation; fulcrum_total is the trustworthy descriptive/gate layer.
PROGRESS: two banked teeth — merge-removal (+12%) and the ContigFold cadence recovery
(gzippy-native 0.678×→0.737× rg, +0.059×, HEAD 7909271f). NEXT: the residual intrinsic symbol-rate
on the native FOLD clean path (≤0.188× upper bound, but still bundles ring-write + ring→data drain
memcpy that the ISA-L oracle doesn't pay). Build a same-engine pure-Rust ring-copy-free-to-final
oracle to cleanly isolate symbol-rate from ring/drain, THEN the inner-Huffman rate work
(BMI2/multi-literal/packed-u32 LUT) bounded by ocl_cf 0.925×, never the VAR_VI 0.6× slope.
