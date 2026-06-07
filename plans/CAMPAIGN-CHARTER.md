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

## CURRENT STATE (2026-06-07, HEAD 3895a23c +oracle) — PHASE-0 WALL ORACLE DONE → **THE T8 WALL BINDER IS THE WINDOW-ABSENT MARKER/SPECULATION PATH, NOT THE ENGINE** (advisor-UPHELD). An igzip-class engine ALONE does NOT close the prod T8 gap; the asm port is NOT the T8 lever. **SUPERVISOR GATE — do NOT start Phase 1 until the bundle is decomposed.**

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
