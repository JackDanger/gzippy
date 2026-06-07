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

## CURRENT STATE (2026-06-07, HEAD fb3baec0) — T8 BINDER LOCATED = SERIAL/CONSUMER-WAIT, not u16 (causal + advisor-UPHELD)

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
