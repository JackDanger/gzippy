# TOOLING-AUDIT.md — adversarial audit of the measurement tooling behind the asm A-vs-B decision

**Auditor:** independent tooling auditor (subagent).
**Audited tree:** `reimplement-isa-l` @ HEAD `9ec10a10` (read first-hand via worktree
`/home/user/www/gzippy-reimplement-isal/.claude/worktrees/tooling-audit`, branch
`audit/tooling-audit`). **This file written to** the auditor's pinned worktree
`/home/user/www/gzippy/.claude/worktrees/agent-ab803b9a6121606b2/plans/TOOLING-AUDIT.md`
(harness pinned writes here; supervisor: copy/merge into the campaign `plans/`).
**Date:** 2026-06-08.
**Method:** read every tool first-hand (line-cited), ran `fulcrum_total --selftest`,
re-ran the parity spine on the guest, traced the ocl_cf oracle into source, attacked each
load-bearing number's tooling.

**Host state at audit:** clock-frozen (no_turbo=1, gov=performance, cpu0 max==min==1.4GHz —
the campaign pin; watchdog PID 407344 from a prior turn). **BUT the box was NOT quiet:** 6×
Plex transcoders at ~100% CPU on overlapping cores 0–23, guest loadavg 8.17. Any ABSOLUTE wall
measured now is contention-inflated (my parity re-run showed 31%/22% spread vs the single-digit
spread of a quiet box). RATIO metrics survive (interleaved). I therefore re-confirmed via ratio
only and did **not** force a fresh `gzippy-isal` rebuild (disk 90% full, load 8 — that would
manufacture the very artifact I audit for).

---

## PART A — TOOL VERDICTS

### A1. `scripts/bench/parity.sh` + `scripts/bench/_parity_guest.sh` — SOUND (production spine)
The claim "a contaminated number is structurally impossible" is **substantially TRUE for the
PRODUCTION number**, and I tried to break it:
- **Env scrub is an ALLOWLIST, not a denylist** (`_parity_guest.sh:49-58`): every `GZIPPY_*`
  except the 3 it sets itself is `unset` before measuring. A renamed/new oracle cannot ride in.
- **sha-verify EVERY run** vs `CORPUS_RAW_SHA256` pin, ABORT on any mismatch (`:229-237`).
- **Regular-file sink with FIFO/symlink defense** (`:199-206`) — kills the pipe-backpressure
  writev phantom.
- **Stale-binary guard by CONTENT fingerprint** (sha of src/crates/examples/build.rs/Cargo.*/
  vendor/benches), not mtime (`:75-119`); absent stamp ⇒ ABORT.
- **Host-freeze readback HARD-FAILS on a readable thawed value** (`:156-159`); `HOST_FROZEN=1`
  can only rescue an NA (unreadable) readback, never override a concrete-wrong one. **Strong.**
- **Production-path assert** `path=ParallelSM` (`:130-134`); matched comparator (gzippy `-p T` /
  rg `-P T`, both to a /dev/shm regular file). Ratio = `rg_min/gz_min` is jitter-immune.

**DEFECTS (minor, do not invalidate the production number):**
1. The post-allowlist hard-fail denylist (`:63-66`) only fails on `GZIPPY_SEED*|*ORACLE*|BYPASS*|
   SLEEP_DECODE*|SLOW*`. Other behavior knobs are scrubbed (unset → restores production) but NOT
   failed — safe ONLY because their unset==production; the guard does not verify OFF==identity.
   **Fix:** fail on ANY scrubbed `GZIPPY_*` whose default isn't the production path, or assert
   OFF==identity per knob in a test.
2. `governor`/`no_turbo` are readback but **loadavg is not** — a Plex-loaded frozen box passes
   the guard and prints a high-spread number (as in my re-run). VERDICT widens the TIE margin to
   the spread (won't false-WIN) but prints a noisy absolute. **Fix:** add a loadavg readback warn.

**Verdict: SOUND for the production parity number** — the most-hardened tool in the set.

### A2. `scripts/bench/oracle.sh` + `scripts/bench/_oracle_guest.sh` — WEAKER (the ocl_cf path)
This produced **ocl_cf 0.945×** (kind `engine-isolation`) and is materially **less hardened**:
- **NO env-scrub.** `_oracle_guest.sh` sets only `GZIPPY_ISAL_ENGINE_ORACLE=1` and inherits the
  rest of the environment (`:108`). A leaked `GZIPPY_SEED_WINDOWS`/`GZIPPY_OVERLAP_WRITER` would
  ride in silently. (Parity allowlist-scrubs everything.)
- **NO stale-binary fingerprint guard** — only `[ -x "$GZIPPY_BIN" ]` (`:32`). The exact
  "stale-binary measurement" failure class is possible here.
- **Host-freeze is WARN-only, not FAIL** (`:46-47`): a thawed/loaded host still prints an ocl_cf
  number. This is why the charter's ocl_cf drifts 0.945× ↔ 0.966–0.989× across host states.
- **Coverage (`isal_oracle_fallbacks==0`) is NOT asserted in-script.** It was checked out-of-band
  via `GZIPPY_VERBOSE` (chunk_fetcher.rs:816-818). The byte-exact gate (`CHECK_SHA=1`,
  `:71,118-119`) IS enforced, but a clean tail that silently fell back to pure-Rust would sha-pass
  while contaminating the "ISA-L ceiling". The tool relies on a human reading a counter.

**Fix:** add the parity allowlist scrub + the stale-binary fingerprint guard; ABORT
`engine-isolation` unless `isal_oracle_fallbacks==0`; hard-fail host thaw to match parity.

**Verdict: USABLE but UNDER-GUARDED.** ocl_cf is byte-exact and coverage was hand-checked, but
the tool does not mechanically prevent the contamination classes the parity spine does.

### A3. `scripts/fulcrum_total.py` — SOUND + SELF-VALIDATED (the trustworthy analyzer)
Re-ran `--selftest`: **25/25 PASS**, including:
- `busy+idle==span` is **non-tautological** — fires on a corrupted `covered` AND independently
  on a corrupted `idle` (idle is independently-measured zero-span gap time, not `span-busy`;
  `:325-329,335-346`). The C2 fix that killed the old vacuous check.
- **no-double-count** (negative self-time ⇒ raise; combine_crc "62ms phantom" → 200us self).
- **wait/compute/output** classification correct (`rx_recv_block`/`dispatch_recv`/
  `wait_replaced_markers` → WAIT, the binder-inversion guard).
- **seeding_guard REFUSES** `window_seeded>0`, `isal_oracle_chunks>0`, and no-sidecar runs
  (`:445-479`) — will not certify the ocl_cf trace as production.
- **oracle_overhead_guard** flags `isal_oracle_fallbacks>0` blends and `to_vec/oracle_copy/
  oracle_alloc` spans (`:482-505`) — the copy-contamination catcher.
- Positive control 1.50×, negative control flat, empty-trace RAISES.

**Verdict: TRUSTWORTHY.** The one tool that catches the classes A2 misses. **Route the ocl_cf
oracle measurements THROUGH fulcrum's guards.** (Fulcrum correctly refuses to bless ocl_cf as
production — ocl_cf is a ceiling, used as one.)

### A4. `benches/engine_isolation.rs` — SOUND (cleanest isolation; VAR_VI/VII source)
- **Byte-exact gate is 3-oracle and strict** (`:1644-1653`): every variant must match scalar over
  `[0,n_actual)` AND scalar must match ISA-L; non-exact ⇒ rate VOID, excluded from the aggregate
  (`:1655,1718,1776-1779`). **VAR_VII 0.276× and VAR_VI 0.594× are byte-verified — the NO-GO is
  not a wrong-bytes artifact.**
- **Interleaved** (all variants per ITER, ITERS=11; `:1686-1693`) ⇒ frequency/turbo-robust.
- **NOT single-chunk** — 5-chunk sweep @ 10/30/50/70/90%, median-of-medians (`:1746-1787`);
  skips non-full-window/short chunks (`:1623-1637`).
- **ISA-L self-test ratio [2.5,3.6]** validates the instrument (`:1798`) — SOFT gate (eprintln, no
  abort, `:1805`); guest authoritative, Rosetta drifts.
- **VAR_VII re-entry is real:** actual `core::arch::asm!` literal-run loop (`:1315+`) re-entered
  per back-ref (`'asm_reentry`, `:1287`); coverage counter (`:1238-1239`) + `GZIPPY_VII_CAREFUL_
  ONLY` bracket (`:1310-1312`) prove the asm ran and the rate falls monotonically with coverage.

**SCOPE caveat (not a defect):** VAR_VI/VII are CLEAN INNER-LOOP DECODE RATES in isolation — no
bootstrap/pipeline/scheduling/output. 0.594× isolated ≠ 0.594× whole-system wall; at T8 the
engine is slack-masked. The bench bounds what the engine *could* contribute; the whole-system
decision correctly rests on ocl_cf.

**Verdict: SOUND.** The most trustworthy of the four numbers by construction.

### A5. Oracle knobs (OFF==identity / window-absent-preserving / no-overhead)
- `GZIPPY_ISAL_ENGINE_ORACLE` (gzip_chunk.rs:188-310): **COPY-FREE verified** —
  `writable_tail_reserve`+`commit`, no intermediate Vec/`copy_from_slice` (prior copy-bug FIXED).
  **Window-absent-preserving verified** — falls back to pure-Rust (counted FALLBACKS) for any
  chunk with `initial_window.len() != MAX_WINDOW_SIZE` (`:206-208`); does NOT seed windows; CRC is
  paid (`:284-290`). **RESIDUAL CONFOUND:** reserves **64 MiB per chunk** (gzip_chunk.rs:232 →
  segmented_buffer.rs:243-248 `Vec::reserve`). Production grows incrementally (~chunk size). The
  default pool DROPS the Vec (rpmalloc thread-cache, chunk_buffer_pool.rs:315-316) — amortization
  is unasserted. No memset (no per-byte tax) but the alloc/page-fault cost is real & unpaid by
  production. **Direction: makes ocl_cf PESSIMISTIC ⇒ true ceiling ≥0.945× ⇒ WEAKENS the asm NO-GO
  premise (more potential headroom, not less).**
- `GZIPPY_SEED_WINDOWS` (perfect_overlap.rs:34): correctly **(masks-binder)**; parity+fulcrum
  refuse it; OFF==identity.
- `GZIPPY_SKIP_WRITEV_SYSCALL`, `GZIPPY_FOLD_NODRAIN/NOCRC` (output_writer.rs:8; gzip_chunk.rs:
  151/162): OFF==identity confirmed (OnceLock `is_ok_and(=="1")`); FOLD_NOCRC is wrong-bytes by
  design, documented.
- `GZIPPY_VII_COVERAGE`/`GZIPPY_VII_CAREFUL_ONLY`: bench-only instruments, OFF==identity.

**Verdict: OFF==identity holds for the audited knobs.** The one substantive issue is the ocl_cf
64 MiB reserve — a residual (pessimistic-direction) confound, not the old copy-contamination bug.

---

## PART B — NUMBER VERDICTS

| # | Number | Verdict | Reason |
|---|--------|---------|--------|
| 1 | gzippy-native **~0.86×** rg (prod, matched) | **TRUSTWORTHY** | Full parity-spine hardening. I re-ran on the guest: **ratio 0.819× (31%/22% spread, Plex-loaded), sha=OK, bin_sha 2f86676 == the banked binary.** Reproduces qualitatively; the spread is load, the central tendency is the banked 0.82–0.87×. |
| 2 | **ocl_cf 0.945×** (ISA-L clean-engine ceiling) | **SUSPECT (sound ceiling, soft tooling)** | Byte-exact + coverage 14/0 verified by-hand (GZIPPY_VERBOSE). BUT: under-guarded `_oracle_guest.sh` (no scrub/stale-guard, host-WARN); 64 MiB-reserve makes it PESSIMISTIC; the campaign's OWN advisor flagged it "over-removes — brackets engine-rate + ring/drain; valid ceiling for the ISA-L-class clean PATH, INVALID as the engine-rate-ALONE ceiling" (clean-rate-ceiling-advisor-verdict.md:30). Sound as a bound on "hand-write the whole clean path"; loose as a bound on pure engine arithmetic. NOT re-measured (needs gzippy-isal rebuild on a quiet box). |
| 3 | **36ms engine / 21ms non-engine** split | **ARTIFACT (the numbers); qualitative finding robust** | Banked on a LOADED host (loadavg ~4, **spread 85–131%**, c2-residual-localization-brief.md:117). On the frozen host the "21ms" was **5–14ms / 0.966–0.989×** (lines 66,110). At 85–131% spread the absolute split is unresolvable. The charter ITSELF lists a quiet-box re-measure + turbo-freq confirmation as still-OWED (CAMPAIGN-CHARTER.md:92). Do NOT bank 36/21. The qualitative claim (engine-rate-dominant gap; no faithful non-engine lever) survives. |
| 4 | engine-bench **VAR_VI 0.594×, VAR_VII 0.276×**, low-T 0.55–0.60× | **TRUSTWORTHY (as isolation rates)** | Byte-exact 3-oracle gate, interleaved, 5-chunk sweep, coverage-instrumented, careful-only bracket, ISA-L [2.5,3.6] self-test. NO-GO (asm-reentry < LLVM loop, monotone-down with coverage) is clean and disproven ×2. CAVEAT: CLEAN-INNER-LOOP rates, NOT whole-system walls; slack-masked at T8. NOT re-measured (no pre-built bench bin + bench corpus absent on guest); source audit ⇒ high confidence it reproduces. |

---

## BOTTOM LINE — what the A-vs-B asm decision can safely rest on

**Safe to rest on (re-confirmed or source-clean):**
- **#1 native ~0.86×** — TRUSTWORTHY, re-ran (ratio-confirmed, sha-OK, matched binary).
- **#4 VAR_VI 0.594× / VAR_VII 0.276×** — TRUSTWORTHY as ENGINE-ISOLATION rates. The asm NO-GO
  (hand-asm-reentry < LLVM loop) is the cleanest, best-instrumented result in the campaign and
  does not depend on the oracle-script or the load-noisy split. **This alone is sufficient to
  reject the inline-asm-TRANSLITERATION fork** (the campaign's actual decision).

**Needs re-measurement on a QUIET, frozen box before it bounds a multi-session commitment:**
- **#2 ocl_cf 0.945×** — re-run via a HARDENED oracle path (add scrub + stale-guard +
  fallback==0 assert + host-FAIL, OR route through fulcrum's guards) and bound the 64 MiB-reserve
  confound (it makes the ceiling pessimistic — true headroom may be larger). ocl_cf bounds the
  **full-kernel asm** (rewriting ISA-L), so its softness matters precisely for fork option (B).
  Treat 0.945× as a *floor on the ceiling*, not a tight ceiling.
- **#3 36/21 split** — do NOT use the absolute numbers in any go/no-go. Re-measure on a quiet box
  (charter already owes this). Use only the qualitative claim.

**Process fix:** the production number is bulletproof; the CEILING numbers (ocl_cf, the split)
ride a weaker harness AND a Plex-contended host. Before the user-fork decision: (1) harden
`_oracle_guest.sh` to the parity bar, (2) get a genuinely QUIET box (Plex ate 600% CPU on
overlapping cores during this audit), (3) re-take ocl_cf + the 36/21 split there, ratio AND
absolute. The VAR_VI/VII NO-GO does not need this and can stand now.

**Disk note:** guest `/` is 90% full (3.4 G free). A fresh `gzippy-isal` + bench rebuild risks
the full-disk failure class the campaign already hit — clear `target/` or build on /dev/shm.

---

### Provenance of this audit's own measurement (the one re-run I executed)
`scripts/bench/parity.sh --no-sync -T 8 -N 11` on guest REDACTED_IP, native bin
`bin_sha=2f8667619679b08c` (== banked), corpus `/root/silesia.gz` ref_sha
`028bd002…cb410f` (== pin), rg 0.16.0, gov=performance/no_turbo=1 (frozen), Plex-loaded
(spread 31%/22%): `gzippy=502ms rg=411ms ratio=0.819 sha=OK verdict=TIE`. No orphan processes
left (guest clean; local ssh wrappers killed; host clock-freeze watchdog left intact per spec).
