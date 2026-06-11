# FULCRUM 2 CHARTER — `fulcrum decide` (committed BEFORE the code; this is the contract)

USER DIRECTIVE: enhance Fulcrum such that ONE RUN yields the numbers needed to
immediately know precisely what to do next. The failure mode this kills: attribution
that did NOT convert at the wall (the 377ms pair-drain phantom, the per-EOB stop cost,
the KEY-MISMATCH re-key lever, the allocator A/B run under wrong placement) and real
levers (phantom-EOS +16%, ratio-reserve +43%, copy-shape -87ms) that were found only
after manual falsifier chains. The instrument must close the attribution→causal gap
ITSELF: every component row either carries a tool-executed causal A/B verdict, or is
explicitly labeled HYPOTHESIS with the exact perturbation that would test it.

## Product

ONE command, host-side, against the bench box (ssh -J neurotic root@REDACTED_IP,
freeze via /root/bench-lock.sh on the HOST):

    scripts/fulcrum decide [--bin /root/bin-p35-native] [--feature gzippy-native]
                           [--cells silesia:1,silesia:4,silesia:8,silesia:16,model:8]
                           [--knob-cells silesia:1,silesia:16] [--no-knobs]
                           [-N 9] [--knob-n 7] [--allow-thaw] [--dry-run]

It produces ONE ranked table; each row a COMPONENT with: cells affected, wall-ms
attribution (canonical mask only), CAUSAL STATUS, distribution health, the exact
re-verify command — and a final `DO THIS NEXT:` line.

## Build-on-validated-pieces policy (extend, don't fork)

- `scripts/fulcrum_total.py` stays the core trace analyzer. `fulcrum_decide.py`
  IMPORTS it (classify/pair_spans/analyze/self-tests); no measurement logic is
  reimplemented in Python.
- `scripts/bench/parity.sh` + `_parity_guest.sh` remain the bankable wall spine,
  UNTOUCHED (they are hash-pinned in plans/instrument-registry.md; editing them
  reverts their validation row). The decide guest runner `_decide_guest.sh` SOURCES
  `scripts/bench/lib_decide_guest.sh`, whose primitives (timed(), regular-file sink
  assert, env scrub, stale-binary fingerprint, freeze readback, quiet gate, stats())
  are copied VERBATIM from `_parity_guest.sh` with `# VERBATIM-FROM _parity_guest.sh:<lines>`
  provenance markers. This is a LABELED copy, not a silent fork: the charter rule is
  that any future change to the spine's primitives must be mirrored here or the
  decide row in the instrument registry reverts to UNVALIDATED.
- Freeze/sha/fingerprint discipline is mandatory for every wall number the tool
  emits. Counter/ratio-only captures (trace, contig_prof, knob effect checks) may
  run without the freeze but are LABELED `unfrozen-counters` in the manifest and
  rendered with that label.

## Component registry

### A. Knob components (tool-run causal A/B; same binary, env-only, byte-exact both arms)

| component | knob (disable/alt arm) | default | effect-verification counter |
|---|---|---|---|
| engine.dist_amort (P3.4 DistTable amortization) | GZIPPY_DIST_AMORT=0 | ON | contig-prof `disttbl: builds=B reuses=R`: OFF arm must show reuses==0 AND builds_off >= builds_on (prof-knob run) |
| engine.stored_flip (M2b stored early-flip) | GZIPPY_NO_STORED_FLIP=1 | ON | none in-tree → EFFECT-UNVERIFIED label |
| engine.seeded_block (M3 seeded chunks on Block) | GZIPPY_SEEDED_BLOCK=0 | ON | verbose `seeded_block=0` AND `seeded_wrapper>0` in OFF arm |
| engine.exact_block (M4 until-exact on Block) | GZIPPY_EXACT_BLOCK=0 | ON | verbose `exact_block=0` AND `exact_wrapper>0` in OFF arm |
| sched.hit_drive (confirmed-offset hit-drive) | GZIPPY_NO_HIT_DRIVE=1 | ON | none cheap → EFFECT-UNVERIFIED label |
| alloc.slab (slab allocator, the reverted lever) | GZIPPY_SLAB_ALLOC=1 (ENABLES; default arm OFF) | OFF | `GZIPPY_RPMALLOC_STATS=1` in effect capture (rpmalloc_alloc.rs:523): knob arm must have `[rpmalloc …]` stats line; base arm must not → EFFECT-VERIFIED |
| sched.eager_postproc | GZIPPY_EAGER_POSTPROC=1 (ENABLES; default arm OFF) | OFF | none cheap → EFFECT-UNVERIFIED label |

NOTE (registry honesty): the directive listed `GZIPPY_PIN_WORKERS`; it does NOT
exist anywhere in src/ at 93a19384 (grep-verified). The registry carries only knobs
the binary actually reads. `GZIPPY_EAGER_POSTPROC` is in-tree and A/B-able, so it
is included in its place.

Knob A/B mechanics (guest): per knob, `--knob-n` interleaved trial pairs
(default-arm run, knob-arm run) on the cell's canonical mask, regular-file sink,
EVERY run sha-verified against the corpus pin; one extra effect-check run with
GZIPPY_VERBOSE (+GZIPPY_CONTIG_PROF for dist_amort) per knob. A knob whose effect
check FAILS (switch didn't disable the feature) is labeled EFFECT-CHECK-FAILED and
its A/B is NOT rendered as causal (the "kill-switches verified to disable fully"
process rule).

Causal statuses:
- CAUSAL-VERIFIED(feature-COSTS Δms max-arm-spread=Xms): knob-arm (feature off) FASTER
  beyond max(spread_base, spread_knob) ⇒ the shipped feature hurts this cell ⇒ actionable.
- CAUSAL-VERIFIED(feature-PAYS Δms max-arm-spread=Xms): knob-arm SLOWER beyond spread ⇒
  the feature's contribution is causally confirmed; not an action, but it converts
  attribution to causation for that component.
- CAUSAL-NULL(bounded ≤ max-arm-spread=Xms): |Δ| within max(spread_base, spread_knob) ⇒
  the component's wall effect in this cell is bounded. Never rendered as a finding beyond
  the bound.
- EFFECT-UNVERIFIED / EFFECT-CHECK-FAILED qualifiers as above.

### B. Trace components (attribution under the canonical mask; HYPOTHESIS unless a knob covers them)

From the fulcrum_total wall-critical-thread decomposition per cell (consumer thread,
leaf attribution, busy+idle==span asserted): `pipeline.consumer.wait`,
`pipeline.consumer.compute`, `pipeline.consumer.output`, `pipeline.consumer.idle`,
plus the top worker-side SELF-time spans. Each row: cells affected, ms on the
wall-critical thread, share of wall. STATUS: HYPOTHESIS + the suggested perturbation
(the in-tree slow-injection knob: GZIPPY_SLOW_MODE=<pct> GZIPPY_SLOW_KIND=spin then
the sleep control — slow_knob.rs; or the matching wait/output knob), never a
recommendation. Free-placement numbers are never rendered: every capture runs under
the cell's canonical mask (T8 ⇒ taskset 0,2,4,6,8,10,12,14; pin_mask() verbatim).

### C. Engine micro-profile rows (contig_prof integration, per corpus)

One GZIPPY_CONTIG_PROF=1 capture per cell (labeled unfrozen-counters; TSC-invariant
shares are valid on the box). Rows: engine.lit1 / engine.litpack / engine.litchn /
engine.backref / engine.careful with iters, cyc/iter, % of classed cycles, plus
disttbl builds/reuses and wrapper-arm counters when present. Headroom rendering:
- BANKED-COMPARATOR (provenance pinned in fulcrum_decide.py): P3.5 @ a9fe662c
  silesia T8: backref 62.6% of classed cycles at 34.9 cyc/iter, litchn 22.9%,
  wrapper calls=0 (contig sole production path), disttbl reuse 3/1768; T1
  symbol-rate gap vs rg ≈ 1.5x (gz 1375ms vs rg ~914-921ms frozen trajectory).
- Each engine row shows measured vs banked and flags DIVERGES-FROM-BANK when the
  share moves >25% relative — then EITHER the tool or the bank is wrong and the row
  says so instead of silently ranking.
- bounded-ms ESTIMATE per engine row = cell wall gap to rg × class share of classed
  cycles, explicitly labeled ESTIMATE (a partition of the gap, not a promise).
  STATUS: HYPOTHESIS + suggested perturbation (slow_knob for the clean loop).

## Distribution health (every wall sample set, both tools)

- N, min, median, IQR, spread% ((max-min)/min).
- RSS per arm on knob rows: `rss base=XMB knob=YMB (+Z%)` — peak RSS from
  `/usr/bin/time -f '%M'` (kilobytes → MB); written to meta.txt by the guest
  and rendered in the table row (timed_masked extension in lib_decide_guest.sh).
- Bimodality: largest-gap heuristic — sort samples; if the largest internal gap
  > BIMODAL_K (3.0) × median of the remaining gaps AND both sides have ≥2 samples
  (degenerate case: all other gaps zero must also pass the ≥2 check — repro:
  [1,1,1,1,1.01] is NOT bimodal because right side has only 1 sample),
  flag BIMODAL (the N=21 silesia-T16 lesson: rg distributions are bimodal/quantized;
  a median can sit on either mode).
- Verdict per cell vs rg: ratio = rg_min/gz_min; PASS at ≥0.99 (the binding TIE bar,
  every thread count); RESOLVED if |gz-rg| > max(spread_gz, spread_rg) in absolute
  ms (or both distributions' IQRs disjoint), else UNRESOLVED with
  N-needed ≈ ceil(N × (spread/|delta|)²) capped at 99.
- A sub-spread delta is NEVER presented as a finding — it renders as
  CAUSAL-NULL/UNRESOLVED with the bound.

## Routing/seeding guard re-derivation (fulcrum_total.py change)

WHY the old guard exists: seeded ORACLE runs (GZIPPY_SEED_WINDOWS replay) route
chunks to the clean engine and MASK the window-absent bootstrap binder; two broken
oracles poisoned the campaign. The old implementation refused on
`window_seeded>0` — but since M3, PRODUCTION chunks legitimately decode seeded
(WINDOW_SEEDED_CHUNKS increments for ANY full-32KiB-window decode, including
WindowMap-published predecessor windows: gzip_chunk.rs:1181, chunk_fetcher.rs:2545).
So the guard OVER-FIRES on every healthy native/isal production run.

Re-derived contamination signals (refuse iff ANY):
1. `SEED_WINDOWS replay: hits=H` line present with H>0 (seed_windows.rs:304-311 —
   printed ONLY when GZIPPY_SEED_WINDOWS mode is on) ⇒ oracle-seeded run.
2. `BYPASS_DECODE replay: hits=H` line present with H>0 (decode_bypass.rs:628 —
   printed ONLY when the decode-bypass replay is active via report_replay_stats()) ⇒
   pre-computed decode results replayed, real engine cost masked (same binder-masking
   class as SEED_WINDOWS).
3. `isal_chunks>0` (ISAL_ENGINE_ORACLE_CHUNKS) on a NATIVE build ⇒ the engine
   oracle ran (on gzippy-isal these are PRODUCTION counters — the guard takes a
   `feature` parameter; isal_chunks>0 on isal is healthy).
3. The runner's env-scrub log reported a seeding/oracle/bypass/slow var.
Production confirmation still required: at least one of finished_no_flip /
flip_to_clean / seeded_block / exact_block / window_seeded fired (else INCONCLUSIVE —
the silently-skipped-bootstrap class). window_seeded>0 alone, without signal 1-3,
is PRODUCTION-SEEDED routing and is ACCEPTED with the counts printed.
All existing fulcrum_total self-tests stay; the seeded-guard tests are UPDATED to
the re-derived rule and new tests cover: production-seeded-accept,
oracle-replay-refuse, isal-on-native-refuse, isal-on-isal-accept.

## Freeze / refusal rules

- Wall + knob A/Bs run inside ONE host freeze acquisition (lib_hostlock.sh,
  TTL 1800s); guest-side freeze readback + instantaneous procs_running quiet gate
  per the spine, recorded in the manifest.
- The analyzer REFUSES to rank any wall/knob number whose manifest shows a thawed/
  loaded/readback-failed condition; `--allow-thaw` downgrades the refusal to an
  UNFROZEN label on every affected row (and on DO-THIS-NEXT if it depends on one).
- Every measured run sha-verified; any mismatch aborts the cell (number VOID).
- Binary identity: sha256 of --bin recorded in manifest + rendered in the header
  (the load-bearing identity; git hash explicitly labeled unreliable on the guest).

## Output schema (the ONE table)

Header: run id, binary sha, feature, corpus pins, freeze status, mask per cell,
guard verdicts per capture.
Rows (rank order): 1) CAUSAL-VERIFIED feature-COSTS by Δms desc; 2) HYPOTHESIS by
bounded-ms desc; 3) CAUSAL-VERIFIED feature-PAYS (confirmations); 4) CAUSAL-NULL.
Columns: COMPONENT | CELLS | ATTRIBUTION (wall-ms @ mask) | CAUSAL STATUS
(max-arm-spread=Xms instead of ±Xms) | DISTRIBUTION (spread/bimodal/RESOLVED-or-N-needed)
| RSS (base=XMB knob=YMB ±Z%; knob rows only) | RE-VERIFY (exact command).
Last line:
    DO THIS NEXT: <top actionable row — the largest CAUSAL-VERIFIED feature-COSTS,
    else the highest-bounded HYPOTHESIS with its pre-registered perturbation command.
    For knobs whose desc contains "reverted": action phrase is
    "reconcile with the prior gated revert + check RSS before flipping" (not fix/condition)>

## Self-tests (fulcrum_decide.py --selftest; all must pass before any run is trusted)

1. knob-A/B harness: synthetic identical arm samples ⇒ CAUSAL-NULL (the known-null
   knob requirement); shifted arms beyond spread ⇒ CAUSAL-VERIFIED with correct sign
   and Δ; shift within spread ⇒ CAUSAL-NULL with the bound.
2. Bimodality: synthetic two-mode sample flagged; unimodal control not flagged.
3. N-needed: monotone (smaller delta ⇒ larger N), never rendered for resolved cells.
4. Guard matrix (via fulcrum_total.seeding_guard): production-seeded native ACCEPT;
   SEED-replay REFUSE; isal_chunks>0 + native REFUSE; isal_chunks>0 + isal ACCEPT;
   no sidecar INCONCLUSIVE.
5. UNFROZEN manifest ⇒ rank refusal (and --allow-thaw label path).
6. contig-prof parser: synthetic dump ⇒ classes, cyc/iter, shares, disttbl counts
   parsed exactly; DIVERGES-FROM-BANK fires on a >25% share move and not on a 5% one.
7. Ranked-table determinism + DO-THIS-NEXT selection (top feature-COSTS beats any
   HYPOTHESIS; absent any causal action, top bounded HYPOTHESIS is picked).
8. fulcrum_total --selftest (updated) passes wholesale — busy+idle==span,
   no-double-count, empty-output, positive/negative controls all retained.

## Runtime budget

Default run (no build; staged --bin): 5 wall cells × N=9 interleaved + captures
(trace, prof) + 7 knobs × 2 knob-cells × 7 pairs ≈ 6–9 min measured + hops; hard
ceiling well under the 25-min freeze TTL. `--cells`/`--knob-cells`/`--no-knobs`
subset further. `--dry-run` prints the full plan + estimated runtime, executes
nothing.

## Acceptance (this run ships with the charter)

End-to-end on the box, NATIVE build (/root/bin-p35-native, sha aed7fd83…), official
silesia cells (T1/T4/T8/T16) + model T8: the actual ranked table with real knob
A/B causal statuses, live contig_prof rows, distribution-health verdicts, and a
DO-THIS-NEXT line. Cross-check against banked P3.5 (backref 62.6%/34.9 cyc/iter,
T1 symbol-rate ~1.5x, T16 sensitivity): contradictions are investigated and called
(tool vs bank) before delivery.
