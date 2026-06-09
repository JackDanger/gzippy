# STATE.md — gzippy decode campaign single-source-of-truth

Synthesized 2026-06-08 from orchestrator-status.md (top + superseded scan), disproof-ledger.md,
CAMPAIGN-CHARTER.md, standing-specialists.md, overnight-autonomy.md, and the 4 advisor verdicts
(lowt-residual-gate, isal-dormancy, converge-bootstrap, liar-sweep). Branch reimplement-isa-l,
HEAD d56cb0f5. Gated-truth only; uncertainty is flagged inline. This doc replaces the need to read
the 2783-line orchestrator-status.md to orient. When it conflicts with a LIVE owner checkpoint, the
owner checkpoint wins for its own region — but check §7 first (several "live" claims are disproven).

---

## 1. THE GOAL + THE BARS

Two flag-gated parallel single-member gzip decode paths, BOTH faithful ports of what rapidgzip's
CODE actually does (faithfulness defined by vendor code, never by a memory line):
- **gzippy-native** (default): does what rapidgzip does, ENTIRELY in Rust, NO C-FFI, u8 wherever
  rapidgzip uses u8. Inline ASM allowed. Clean tail = pure-Rust inner Huffman (`decode_clean_into_contig`).
- **gzippy-isal**: same, but hands the clean tail to ISA-L via C-FFI (= rapidgzip's WITH_ISAL build).
  Reference/comparison baseline.

**BAR-1 (BINDING, user-set 2026-06-08): TIE = >=0.99x at EVERY thread count** (T1, T4, T8, T16…),
interleaved + sha-verified, quiet/frozen box. 0.88x is NOT a tie; "within spread" / "ties at T8" is
REJECTED. Done = a build ties rapidgzip at every cell, byte-exact, structure faithfully mirroring
rapidgzip, AND (for native) pure-Rust is the sole decode path with C-FFI off the decode graph.

Faithful = match rapidgzip's RUNTIME BEHAVIOR. The overnight ASM decision is now the SUPERVISOR's
call (user delegated full authority for the overnight run; resumes in the morning).

---

## 2. VERIFIED SCORECARD (BAR-2, gated)

Frozen guest REDACTED_IP, no_turbo=1, env-unset PRODUCTION (oracle/seed UNSET), N=11 interleaved,
same-sink /dev/shm regular file, sha == 028bd002…cb410f, path=ParallelSM asserted, vs rg 0.16.0.
Binaries: isal 2d317027, native a42d4600. Pass/fail vs the >=0.99x BAR-1.

| cell | gzippy-ISAL | vs 0.99 | gzippy-NATIVE | vs 0.99 |
|------|-------------|---------|---------------|---------|
| T1 | 0.899x (spread 1%) | **FAIL** | 0.608x (1%) | **FAIL** |
| T4 | 0.900x (spread 3%) | **FAIL** | 0.761x (3%) | **FAIL** |
| T8 | 0.990x (spread **9% — soft**) | PASS at threshold | 0.915x (6%) | **FAIL** |

- ISA-L coverage (GZIPPY_VERBOSE, env-unset): isal_chunks = 16/14/14 @ T1/T4/T8, fallbacks 1/0/0.
  ISA-L is ACTIVE on the bulk; NOT dormant. native isal_chunks=0 (structurally impossible to be >0).
- **Neither build is done under BAR-1.** isal passes T8 ONLY, at the 0.990 threshold with a loose 9%
  spread (a tighter T8 re-run is owed to harden it). Both lose T1/T4. isal is 14-48% faster than
  native at every T. **Low-T is the headline.**
- NUMBER-DRIFT CAVEAT (see §7): an earlier shipped-table turn (19add96c) recorded isal T4 0.885x /
  T8 1.030x; the OPEN-1 decomposition turn recorded T4 0.902-0.904x. The CURRENT banked values are
  0.900x / 0.990x (dormancy-reconciliation, BAR-2). The 0.757x "isal" number is REFUTED (DIS-13,
  mislabeled native binary).

---

## 3. LEVER MAP

| lever | status | gated evidence | what would close it |
|-------|--------|----------------|---------------------|
| **Clean-tail ASM** (native engine → rg's ISA-L instr, pure-Rust+inline-asm, no C-FFI) | SUPERVISOR-GATED (overnight call). NECESSARY-not-SUFFICIENT for low-T. | LEV-1: engine swap pure-Rust→ISA-L recovers ~0.139-0.159x of native's T4 deficit (removal oracle ocl_cf 0.899x vs native 0.740/0.761x; env-unset isal-vs-native A/B corroborates). LEV-4: clean-rate ~2.3x slower/byte than rg. But ocl_cf == real ISA-L == 0.899x@T4 ⇒ even a perfect engine LOSES T4 ⇒ asm alone can't reach 0.99. Per-symbol asm transliteration is DEAD (DIS-1); only a FULL-KERNEL asm is untried. | A full-kernel (not per-symbol) `core::arch::asm!` port of igzip's hot loop, byte-exact, isolation-bench then WHOLE-SYSTEM wall. Even success is bounded at 0.899x@T4 — needs the residual too. |
| **Low-T faithful-placement** (port rg's runtime window-map / in-place narrowing) | OPEN, UNSIZED. OPEN-1 oracle RUNNING (owner af441b6). | seed-all removes 231ms@T4 (1.49-1.55x ceiling) — bootstrap is the DOMINANT removable T4 term, FFI is small. BUT seed-all is an OVER-removing masks-binder ceiling (free precomputed windows from an uncounted p=1 pre-pass). | An oracle granting rg-grade placement at RUNTIME (counted) cost while keeping the prefix on the SAME engine as production (no free ISA-L upgrade). Until then the faithful slice is UNSIZED. |
| **Marker-bootstrap COMPUTE** (pure-Rust u16 marker decode+resolve of window-absent prefix) | DOMINANT T4 term but ENTANGLED (advisor FIX-NEEDED — see §7). | Co-primary with placement (no-windows AND no-bounds each ~0 alone; need BOTH). | DISENTANGLE from the gated asm: the 231ms embeds a pure-Rust→ISA-L engine swap on ~3 prefix chunks (isal_chunks 14→17), which IS the gated LEV-4. The genuinely-faithful slice is a strict unsized subset. |
| **FFI-handoff** | NEGLIGIBLE at T4 (bounded small). | seed-all routes ALL 17 chunks through ISA-L FFI yet BEATS rg by 231ms ⇒ FFI can't be a big term. | (closed direction — not a lever.) |
| **JOB-2 SYNC_FLUSH / stored-fixed clean-tail coverage** (gzippy-isal correctness, OPEN-3) | RUNNING (owner a9e7e1a, box-free). Correctness, not perf. | On stored/fixed-block-dense input, ISA-L's END_OF_BLOCK doesn't fire on stored/fixed ⇒ until_exact declines ⇒ isal degrades to pure-Rust (byte-exact, zero ISA-L coverage). silesia (all-dynamic) unaffected. | Relax `gzip_chunk.rs` until_exact EXACT-match accept to coalesce to nearest clean EOB (faithful readStream isal.hpp:255-360, NOT the wrongly-cited readBytes :392-405 / DIS-10). Its own gated turn; risks reviving the 19add96c over-decode mis-seed. |

T1 is THREAD-COUNT-STRUCTURED differently: at T1 windows are always present (sequential) ⇒ NO marker
bootstrap; isal_chunks=16 (ISA-L IS the engine). rg also uses ISA-L at T1 ⇒ engine is MATCHED, yet
gzippy still loses 0.899x ⇒ the T1 deficit is NON-engine (per-chunk FFI handoff + serial-output floor
+ chunk-0 bootstrap). T1 is NOT bootstrap-closable and NOT clean-tail-asm-closable.

---

## 4. DISPROVEN / DEAD (with mechanism)

- **DIS-1 inline-asm per-symbol transliteration** — NO-GO. VAR_VII 78 MB/s (0.276x ISA-L), SLOWER as
  asm coverage rises: per-symbol asm↔Rust re-entry spills (LLVM barrier ×300-460K/chunk, 4 regs to
  `bits`) dominate. Full-kernel asm is a DIFFERENT, untried construct — DIS-1 does NOT bind it.
- **DIS-2 clean-window oracle** — broken instrument; silently RE-RAN the full bootstrap (fixed 64eb6df).
  Killed the decompose-a-slice-and-shave loop it had licensed.
- **DIS-13 "ISA-L runtime-dormant" bombshell** — REFUTED; it measured a gzippy-NATIVE binary MISLABELED
  as gzippy-isal (isal_chunks=0, 654ms=0.757x = byte-for-byte the native signature). isal_chunks=14 is
  structurally impossible on native (stub returns Ok(false), gzip_chunk.rs:390-408 — SOURCE-VERIFIED).
- **DIS-5 / DIS-11 output-overlap writer** — non-faithful (rg's writeFunctor is inline-synchronous, no
  bg thread) AND sub-parity; the 0.88x "ceiling" was CONSTRUCTED by destroying overlap. REFUTED as path.
- **DIS-6 offset-supply / consumer-confirmation prefetch** — premise factually wrong: gzippy ALREADY
  re-targets (gzip_block_finder.rs:180-182). 3 prior attempts failed; decode in-flight-not-done when
  consumer arrives. Do NOT re-attempt offset-supply.
- **clean-only NO-OP instrument bug** — `GZIPPY_SEED_WINDOWS` is read as a FILE PATH, not a bool
  (seed_windows.rs:98-102); `=1` → open("1") → ENOENT → empty store → silent PRODUCTION fallback. So
  `oracle.sh --kind clean-only` measured production-vs-production. INVALIDATES any "clean-only +10ms
  TIE ⇒ placement slack" number. DIS-13/BAR-2/LEV-1 are unaffected (rest on env-unset binaries).
- Also dead: DIS-7 free-decode Oracle-C (frees windows, collapses publish-chain), DIS-8 combine_crc
  62ms phantom (nested-span double-count), DIS-9 C2 21ms scheduling tooth, DIS-10 isal.hpp:392-405
  resync (wrong vendor line). TIE'd (stop re-measuring): TIE-1 marker fast-loop, TIE-2 fastloop2,
  TIE-3 chunk-size, TIE-4 writev granularity, TIE-5 T8-only engine-swap-when-seeded, TIE-6 page-warmth.

---

## 5. INSTRUMENT-TRUST REGISTRY

VALIDATED:
- env-unset isal-vs-native PRODUCTION A/B (two real binaries) — removal-oracle-grade; gives BAR-2.
- ocl_cf ISA-L engine-isolation removal oracle (GZIPPY_ISAL_ENGINE_ORACLE on isal build) — gives LEV-1.
- two-pass seed driver `scripts/bench/{seed_clean,seed_decompose}.sh` (capture at p=1 + `hits>0`
  self-test) — the CORRECT clean-only oracle; gives the 231ms@T4 bootstrap ceiling.
- merge-removal / FOLD-drain teeth (LEV-8/LEV-9) — landed, byte-exact, advisor-upheld.

KNOWN-BUGGY / DISTRUST:
- **clean-only NO-OP** (oracle.sh --kind clean-only sets SEED_WINDOWS=1 = a file path ⇒ no-op). Any
  number banked from it is production-vs-production. HARNESS BUG — flagged to Steward.
- **clean-window oracle (INSTR-1)** re-ran bootstrap; **INSTR-2** emitted empty output; **INSTR-3**
  isal_oracle_chunks= grep read a string the binary never emits (hard-failed silently — fixed to
  `isal_chunks=`); **INSTR-4** Oracle-C degenerate; **INSTR-5** GZIPPY_PERFECT_OVERLAP built backwards.
- **rss_vs_t.sh MPKI label** prints `-nan` (awk divisor bug under paranoid=4 / cpu_core counter prefix);
  RAW counters valid, label only (INSTR-7).
- **seed-all is an UPPER BOUND** that OVER-removes on TWO axes (free precomputed placement + free ISA-L
  engine upgrade on the prefix); 1.55x is NOT a reachable target.

**MANDATORY binary-verification (overnight rule 2):** every isal measurement must read
**isal_chunks >= 14** as the real-ISA-L fingerprint BEFORE trusting it. isal_chunks=0 ⇒ a mislabeled
native binary (the DIS-13 trap). Native stub returns Ok(false) and never increments — VERIFIED
first-hand at gzip_chunk.rs:386 (increment) / :390-408 (stub).

---

## 6. OPEN DECISIONS / OWED

- **ASM scoping** — SUPERVISOR's overnight call. Plan: SCOPE the full-kernel native clean-tail asm
  FIRST (source-map igzip kernel vs gzippy's clean loop, byte-exact harness, feasibility) as a leader
  turn; if tractable, start incrementally with byte-exact + isolation-bench gates; hold the multi-session
  commit if scoping is unfavorable. NOT started.
- **OPEN-1 residual-decomposition** — the owed oracle that sizes the genuinely-faithful placement slice
  apart from the gated engine swap (grant rg-grade placement at RUNTIME cost, prefix on same engine).
  Until it runs, LEV-2 is an upper bound, not a turnable lever. Owner lowt-residual-decomp ran the
  first cut (advisor: FIX-NEEDED on entanglement).
- **OPEN-2 placement-slice** — RE-OPENED (its bounding clean-only oracle was the NO-OP + had a
  mislabeled-native baseline). Must be RE-RUN on the verified isal binary (0.900x baseline). The
  never-dispatched-vs-evicted discriminator is UNRUN.
- **Running owners:** af441b6 = PLACEMENT ORACLE (gate: faithful-placement slice > spread with
  isal_chunks==14 ⇒ port rg window-map; < spread ⇒ low-T is asm-bound). a9e7e1a = JOB-2 coverage
  (box-free; gate byte-exactness + coverage, KEEP if sound).
- **Parallel-path naming** — PENDING (after live measurement owners settle, to avoid churning edited
  paths). Targets: the "oracle" misnomers on PRODUCTION ISA-L (isal_engine_oracle_enabled /
  ISAL_ENGINE_ORACLE_CHUNKS / finish_decode_chunk_isal_oracle gate production on the isal build),
  GZIPPY_SEED_WINDOWS (a file-path, not a bool). Byte-transparent renames; both feature-sets green.
- Owed across the board: a real synchronous Opus disproof + Steward bankability sign-off on the
  checkpoint numbers (the Agent/subagent tool was ABSENT in owner envs ⇒ self-disproof only).

---

## 7. CROSS-DOC INCONSISTENCIES (highest value — verify before trusting)

1. **build.rs comment is STILL FALSE at HEAD — SOURCE-VERIFIED.** orchestrator-status & DIS-12 assert
   "build.rs:98-110 comment corrected (byte-transparent) on branch fix/buildrs-isal-comment." But on
   reimplement-isa-l @ d56cb0f5 (HEAD), build.rs:104-109 STILL reads *"BOTH topologies decode the clean
   tail in PURE RUST … Real ISA-L FFI … NOT on any production path."* That is FALSE: gzippy-isal
   production routes the clean tail through real ISA-L FFI by default (19add96c). The liar-sweep verdict
   independently flags it UNEDITED. **The fix lives only on an unmerged branch; HEAD is still poisoned.**
   Any agent reading build.rs first will be mis-led. Also the stale ":539" citation points at a parameter.

2. **The "marker-bootstrap = faithful-port, INDEPENDENT of the gated asm" framing is WRONG (advisor
   FIX-NEEDED).** The OPEN-1 owner checkpoint (orchestrator-status:36) calls the 231ms bootstrap term
   "additive to, independent of, the user-gated clean-tail asm." The lowt-residual-gate verdict refutes
   this: seed-all confounds bootstrap-removal with a pure-Rust→ISA-L ENGINE SWAP on ~3 prefix chunks
   (isal_chunks 14→17 = the gated LEV-4 asm). The bootstrap is ENTANGLED with the gated engine; the
   genuinely-faithful slice is a strict, UNSIZED subset. Do NOT treat it as a clean, independent,
   owner-turnable lever.

3. **T4/T8 number drift across turns.** isal T4: 0.885x (19add96c shipped table) vs 0.900x (BAR-2,
   current banked) vs 0.902-0.904x (OPEN-1). isal T8: 1.030x (19add96c) vs 0.990x (BAR-2, current).
   rg T4 absolute wanders 490/493/495/496ms. The 0.757x "isal" T4 is REFUTED (DIS-13). CURRENT banked =
   0.900x / 0.990x. The 19add96c table's 1.030x T8 should NOT be cited — it predates the frozen-N=11
   re-establishment and is looser; under BAR-1 the binding T8 number is 0.990x (threshold pass, 9% spread).

4. **The whole "RESIDUAL-ATTRIBUTION" checkpoint (orchestrator-status:84-125) is SUPERSEDED but reads as
   live.** Its load-bearing facts (isal_chunks=0, isal==native 0.757x, "98% marker bootstrap for BOTH
   builds", "OPEN-2 MOOT/within-spread", "56ms residual does NOT exist at HEAD") are ALL the DIS-13
   mislabeled-native error. It is explicitly marked SUPERSEDED, but a cold reader skimming the top
   ~120 lines hits it before the correction. Treat anything sourced to "isal_chunks=0" or "isal T4
   0.757x" or "98% marker bootstrap" as DEAD.

5. **GOAL #2 "completion" wording vs BAR-1.** orchestrator-status:165 / charter SUPERSEDED state
   declare GOAL #2 "SHIPPED / TIES rapidgzip at T8." Under BAR-1 (BAR-0 supersedes the old framings)
   that is a PASS on T8 ONLY — NOT a build-level done. Both builds lose T1/T4. The "done" language is
   pre-BAR-1 and must not be read as campaign-complete.

6. **OPEN-2 = consumer-imminent eviction is asserted, not measured (two advisors agree).** The
   converge-bootstrap verdict (claim 5) and the ledger both note the "eviction" MECHANISM is one of two
   live hypotheses (the other: prefetch horizon too shallow / worker saturation); the decisive
   discriminator is UNRUN. Some checkpoint prose states "eviction" as the cause — that prejudges an
   unrun diagnostic. Present OPEN-2 as an open diagnostic, not a confirmed co-primary lever.

Net: the campaign's banked SPINE (BAR-2 scorecard, LEV-1 engine-swap, FFI-negligible, ISA-L active) is
consistent and survives. The inconsistencies are (a) a stale poisoned build.rs comment at HEAD, (b) an
over-claimed separability of the bootstrap lever from the gated asm, and (c) superseded
mislabeled-native prose that still sits near the top of orchestrator-status.
