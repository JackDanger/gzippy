# LOW-T RESIDUAL GATE — independent Opus disproof verdict

Role: read-only disproof advisor. Branch reimplement-isa-l @ d56cb0f5.
Source-verified first-hand (seed_windows.rs, chunk_fetcher.rs, gzip_chunk.rs,
the two-pass seed driver). Attacked, not ratified. No edits, no orphan processes.

TL;DR: the owner's TOP-LINE decomposition is SOUND (bootstrap dominates T4, FFI is
small, the instrument bug is real and correctly quarantined). But the STRATEGIC
conclusion over-reaches in one load-bearing way: **the 231 ms "marker-bootstrap"
term is NOT purely faithful-port territory — seed-all confounds bootstrap-removal
with a pure-Rust→ISA-L ENGINE SWAP on the prefix (isal_chunks 14→17), which is the
user-GATED LEV-4 asm. So the bootstrap lever is ENTANGLED with the gated engine,
not "additive to and independent of" it, and its faithfully-reachable size is
UNSIZED.** The over-removal caveat is correct and, if anything, understated.

---

## CLAIM 1 — THE SPLIT (bootstrap+alignment co-primary; FFI negligible) — **FIX-NEEDED**

SOUND parts (source + arithmetic confirmed):
- **FFI is small.** Production (549 ms, isal_chunks=14) and seed-all (318 ms,
  isal_chunks=17) BOTH pay ISA-L FFI; seed-all pays it on MORE chunks yet finishes
  231 ms faster. FFI is held constant-or-higher across the delta, so the 231 ms is
  not FFI. The "17 ISA-L chunks yet beats rg" bound is a valid aggregate cap on FFI.
  No objection.
- **Bootstrap dominates the T4 removable term.** The seed-all ceiling (1.49–1.55×)
  is the single large lever; no other oracle this turn moves the wall comparably.

FIX-NEEDED — two confounds the owner under-states:

(1) **seed-all does NOT merely remove marker-COMPUTE; it REPLACES it with ISA-L.**
    Verified: a seeded-window hit forces `decode_mode_clean=true`
    (chunk_fetcher.rs:2512-2513) → the clean arm `decode_chunk_with_until_exact`
    (chunk_fetcher.rs:2582-2593), identical to the published-window arm, which on
    the isal build routes to real ISA-L FFI (DIS-12 chain). The owner's own counter
    proves it: isal_chunks goes **14 (prod) → 17 (seed-all)** — ~3 chunks that were
    pure-Rust u16-marker in production are now ISA-L. So the 231 ms =
    `[~3 prefix chunks: pure-Rust-marker → ISA-L]` (an ENGINE swap = LEV-4, the
    GATED asm) `+ [placement gain]`. Labelling the whole 231 ms "(A) marker COMPUTE,
    faithful-port" mis-attributes an embedded engine-swap (LEV-4 gated) slice as
    faithful-port. This is the central correction.

(2) **The decompose legs are weaker isolations than presented.** `no-bounds`
    (555 ms ≈ prod) is near-TAUTOLOGICAL: seeded windows are keyed by exact
    `start_bit` (seed_windows.rs:280-301), so partition-GUESS offsets can never
    address them → `seed_window_for` always misses → speculation. "Windows alone
    don't help" is therefore largely an artifact of the key-by-boundary instrument
    design, not independent evidence that placement is separately necessary.
    `no-windows` (581 ms) is within run-spread of production (549 ms) → a TIE, fairly
    read as "no help." The CONCLUSION ("need both jointly; co-primary") survives and
    matches LEV-4/LEV-5 — but the legs do not cleanly isolate (A) from (C); they
    show only the joint ceiling.

Net: top-line direction SOUND; the A/B/C attribution mislabels an ISA-L engine-swap
slice as faithful-port "compute." FIX the attribution before treating the bootstrap
as a clean non-engine lever.

## CLAIM 2 — THE OVER-REMOVAL CAVEAT — **SOUND (arguably understated)**

The caveat ("1.55× is an upper bound, not a reachable target; seed-all hands gzippy
precomputed windows from an UNCOUNTED p=1 pre-pass that rg also does at runtime") is
CORRECT. The two-pass driver confirms the mechanism: capture at p=1 (seed_clean
driver line 17-19), replay with a real file + `hits>0` self-test (line 38). The
captured windows + boundaries are genuinely free at replay.

It is in fact **understated**: beyond the free precomputed windows, seed-all also
upgrades the window-absent prefix from pure-Rust marker decode to ISA-L (confound
#1 above) — work rg does NOT get to do (rg marker-decodes its window-absent chunks
speculatively then narrows; CLAUDE.md: rg carries the same u16 machinery, 31.25%
replaced markers, 0.113 s apply-window). So seed-all over-removes on BOTH axes:
free placement AND a free engine upgrade on the prefix.

**Reachable faithful target: UNKNOWN / UNSIZED.** A faithful close would match rg's
window-map placement + marker rate at RUNTIME (counted) cost — not free precomputed
windows and not ISA-L on the prefix. No oracle this turn isolates that slice. So:
- "bootstrap is the DOMINANT T4 term and FFI is not" — PROVEN.
- "closing the bootstrap reaches ≥0.99× at T4" — **UNPROVEN, and not implied by
  1.55×.** The owner correctly disclaims this. Affirmed.

## CLAIM 3 — THE INSTRUMENT BUG — **SOUND / CONFIRMED (rule-4 failure, correctly quarantined)**

Source-confirmed. `GZIPPY_SEED_WINDOWS` is read as a FILE PATH string
(seed_windows.rs:98-102). `=1` → `File::open("1")` → ENOENT → `empty_seed()`
(seed_windows.rs:189-195). `seed_enabled()` is then TRUE but the store is EMPTY, so
`seedable_chunk_starts()` is empty and `seed_window_for` always misses
(seed_windows.rs:296-299) → every chunk falls through to speculation = PRODUCTION.
It prints "SEED_WINDOWS open error" to stderr (not literally silent) but is
behaviorally a no-op. So `oracle.sh --kind clean-only` (which set `=1`) measured
**production-vs-production**. CONFIRMED.

Blast radius (correctly scoped by the owner / ledger):
- INVALIDATES any "clean-only TIE / +10 ms ⇒ placement is slack" banked from
  `--kind clean-only` — i.e. the OPEN-2 "placement recovers ~0" leg. The ledger
  already RE-OPENED OPEN-2 (disproof-ledger:95) on the SEPARATE mislabeled-binary
  ground; this instrument bug is a second, independent reason that leg is void.
- Does NOT touch DIS-13 / BAR-2 / LEV-1: those rest on env-unset isal-vs-native
  PRODUCTION binaries (isal_chunks=14/14 vs 0), not on clean-only. The banked
  scorecard stands.
- The NEW seed_clean driver fixes the instrument (real capture file + `hits>0`
  assert), which is why the 1.55× number is trustworthy where the old "+10 ms" was
  not. Good rule-4 hygiene.

Recommend the harness flag to Steward (the owner already noted it): `oracle.sh
--kind clean-only` must either run the capture step or hard-fail on `hits=0`.

## CLAIM 4 — STRATEGIC (low-T lever = faithful marker-bootstrap; T1 = non-engine floor) — **FIX-NEEDED on separability; T1 reframe SOUND**

- "The dominant T4 low-T lever is the marker-bootstrap, NOT FFI, NOT scheduling" —
  SOUND as DIRECTION.
- "= faithful-port (window-map + in-place narrowing), owner-turnable, ADDITIVE TO
  and INDEPENDENT OF the gated clean-tail asm" — **FIX-NEEDED.** Per confound #1,
  the 231 ms embeds a pure-Rust→ISA-L engine swap on ~3 prefix chunks (LEV-4, the
  GATED asm). So the bootstrap term is ENTANGLED with the gated engine, not cleanly
  additive/independent. The genuinely-faithful slice (placement/window-map precision
  matched to rg at runtime cost) is a SUBSET of 231 ms and is currently unsized.
  "The lever is faithful-port not asm" over-claims separability.
- **T1 is a NON-engine gap — the prompt's reframe is MORE correct than the owner's
  wording.** At T1, sequential ⇒ windows always present ⇒ no marker bootstrap;
  isal_chunks=16, the body decodes via ISA-L (BAR-2). Since rg ALSO uses ISA-L at
  T1, the engine is MATCHED, yet gzippy still loses 0.899× (1032 vs 927 ms). A
  matched engine that still loses ⇒ the deficit is **NON-engine**: per-chunk FFI
  handoff + serial-output floor + chunk-0 bootstrap, NOT "engine symbol-rate" (the
  owner's phrase "engine symbol-rate + … bounded by real ISA-L" is imprecise — at
  T1-isal the engine IS ISA-L, so it cannot be the symbol-rate that binds). The
  "bounded by real ISA-L zero-margin" observation actually SUPPORTS the non-engine
  read: the fastest engine, matched on both sides, still loses ⇒ the loss lives in
  the wrapper, not the kernel. T1 is NOT bootstrap-closable. SOUND.

---

## STRATEGIC ANSWER (the prompt's headline question)

**Is closing the marker-bootstrap a real reachable lever, or do we not yet know its
reachable size?** — WE DO NOT YET KNOW. Confirmed: it is the DOMINANT removable T4
term and FFI is small. Unconfirmed: its faithfully-reachable size, because seed-all
over-removes on TWO axes (uncounted precomputed placement AND an ISA-L engine
upgrade on the prefix rg marker-decodes at runtime). The reachable faithful slice is
a strict, unsized subset of 231 ms, and part of even that subset is the GATED engine
(LEV-4), not faithful-port.

**Owed next instrument (OPEN-1 is still the gate):** an oracle that grants gzippy
rg-equivalent placement/window-map precision computed at RUNTIME (counted) and keeps
the prefix on the SAME engine as production (no free ISA-L upgrade) — i.e. seed
boundaries only, marker-decode the prefix, but with rg-grade placement — to size the
genuinely-faithful placement slice apart from the gated engine swap. Until that runs,
the bootstrap is the right DIRECTION but NOT an authorizable, sized lever, and "T4
reaches 0.99× by faithfully closing the bootstrap" is UNPROVEN.

## PER-CLAIM VERDICTS
- CLAIM 1 (split): **FIX-NEEDED** — FFI-small SOUND; bootstrap-dominant SOUND; but
  "(A) marker COMPUTE faithful-port" mis-attributes an embedded ISA-L engine swap
  (isal_chunks 14→17) and the no-bounds leg is near-tautological.
- CLAIM 2 (over-removal caveat): **SOUND** (understated). Reachable size UNKNOWN;
  1.55× not a target; "reaches 0.99×" correctly disclaimed.
- CLAIM 3 (instrument bug): **SOUND/CONFIRMED.** Real rule-4 no-op; blast radius =
  OPEN-2/clean-only "placement is slack" only; DIS-13/BAR-2/LEV-1 unaffected; new
  driver fixes it.
- CLAIM 4 (strategic): **FIX-NEEDED** on "faithful-port, independent of asm" (it is
  entangled with the gated LEV-4 engine); **SOUND** on "T4 lever direction =
  bootstrap" and on "T1 deficit is NON-engine, not bootstrap-closable."
