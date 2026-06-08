# DISPROOF VERDICT — C2 non-engine residual localization + removal bound

Independent disproof advisor, 2026-06-08. Read the brief, both prior advisor
verdicts, the prefetch-gap memory, and the actual code
(`perfect_overlap.rs`, `chunk_fetcher.rs:1367-1445` + `:2109-2212`,
`gzip_chunk.rs:551-583`, `consumer_block_decompose.py`,
`_oclcf_overlap_bound.sh`). Verdict per claim, with the kill-shot first.

---

## The two code facts that govern everything below

**FACT A — `warm_miss` is BLIND to the costly case.** The miss counter
(`chunk_fetcher.rs:1437-1438`) fires whenever `chunk_arc_from_partition ==
None`. That is reached by *two* paths: (i) `try_take_prefetched_pumping`
returns `None` (genuinely-absent prefetch — the cheap offset-0 startup case),
**and** (ii) the prefetch WAS present but `arc.matches_encoded_offset(decode_start)
== false` (line 1394), so the arc is dropped and the consumer cold-re-decodes.
Path (ii) is *exactly* the overshoot-tail stall the memory names as the
candidate's load-bearing mechanism (confirmed offset lands strictly inside a
guess chunk's `[enc, max]` range). **`warm_hit_frac=0.88` cannot tell a (i) from
a (ii).** So the brief's claim that "the 2 residual are the offset-0 startup
misses" is ASSERTED, not instrumented.

**FACT B — PERFECT_OVERLAP dispatches at the PARTITION-GUESS key, not the
confirmed offset** (`perfect_overlap_warm`, `chunk_fetcher.rs:2164` →
`submit_prefetch(part_key, …)` `:2199`). Therefore it CANNOT warm an
overshoot-tail chunk: that chunk's confirmed `decode_start` ≠ its partition
guess, so `matches_encoded_offset` still returns false and the arc is still
discarded (FACT A path ii). The oracle removes the dispatch-**DEPTH** gap (chunk
not yet in flight); it does NOT remove the dispatch-**TARGETING / interior-reuse**
gap, which is the memory's own CONCLUSIVE root cause. These are different levers
sharing one name.

---

## Claim (1) — "named candidate (project_confirmed_offset_prefetch_gap) REFUTED as the 21ms residual"
### Verdict: UPHELD-WITH-CAVEATS (refutes the DEPTH reading; the INTERIOR-REUSE reading is untested, not refuted)

What the oracle validly establishes: removing the dispatch-depth term — making
88% of chunks in-flight from t0 — moved the wall **-0.2% (FLAT)**, ≪ the 14-18%
spread. This is a genuine REMOVAL (Rule 3), and because it is a removal at a
frozen 1.4 GHz / no_turbo host, NOT a slow-injection, the Rule-2 turbo-depression
artifact does not apply. So **"dispatch depth / head-of-line not-yet-dispatched"
is dead as the 21ms lever.** That half is solid.

Why it is NOT the full refutation the brief claims:

- **Per FACT B, the oracle structurally cannot exercise the candidate's real
  mechanism.** The memory's own final sections (CONCLUSIVE 2026-06-04, BILATERAL,
  ROOT CAUSE) abandon the "prefetch deeper" reading and land on overshoot-tail
  *interior reuse* (port rg's `getIndexedChunk` + blockMap subchunk
  `decodedOffset`, `GzipChunkFetcher.hpp:264-288`). PERFECT_OVERLAP dispatches at
  the guess key, so those chunks are STILL discarded and cold-re-decoded in BOTH
  arms A and B. The flat A→B delta therefore holds the interior-reuse term
  *constant (broken)* in both arms — it does not remove it. The brief's framing
  ("the validated removal oracle for the named candidate") **over-claims**: it is
  a removal oracle for dispatch depth, a *control* (not a removal) for interior
  reuse.

- **Could the unremoved 12% be exactly the costly ones? YES — and the instrument
  can't deny it.** By FACT A the 2 residual misses are equally consistent with
  being the overshoot-tail mismatch-discards (the expensive cold re-decodes) as
  with startup. The brief's "they're startup" is unbacked.

Why the candidate is nonetheless effectively dead **on the wall that matters**
(the caveat that rescues the practical conclusion, not the logic):

- On `ocl_cf` the cold re-decode of an overshoot tail is a *clean, windowed*
  decode (predecessor window is known at the consumer frontier) → it routes
  through `finish_decode_chunk` → the ISA-L oracle → **ISA-L-fast**. So even if
  the 12% are the costly stalls in the pure-Rust world, they are cheap here, and
  the aggregate wall is flat. The memory itself records all THREE interior-reuse
  patch attempts measured WORSE and reverted.

**Net:** REFUTED as "dispatch-depth scheduling." NOT refuted as "interior-reuse
data-layout" — that lever survives the oracle untouched, but has near-zero
measured payoff on `ocl_cf` and a documented 3-for-3 failure record, so it is a
*high-risk, low-payoff* lever, not a *refuted* one. The brief should down-state
"the named candidate is refuted" to "the dispatch-depth interpretation is
refuted; interior-reuse is unremoved but bounded near-zero-payoff on the matched
wall."

**To close the gap cleanly:** label each `warm_miss` at the call site
(`decode_start==0` startup vs `matches_encoded_offset==false` discard vs
absent) — a 3-line change — and re-report. If the 12% are startup, claim (1) goes
to full UPHELD; if they are discards, the brief's "startup" sentence is simply
wrong and must be struck.

---

## Claim (2) — "residual is NOT consumer-serial bookkeeping (resolve/publish ~0ms)"
### Verdict: UPHELD

This is the strongest claim and it survives disproof:

- The decompose is a topology fact, not a small delta:
  `window_publish_marker / wait_replaced_markers / get_last_window` are
  ~0.0001s **each** — three orders of magnitude below the 0.39s wall. apply_window
  / marker resolution genuinely runs off the consumer in-order path (faithful to
  rg's pool-side post-process). A single-shot trace is perfectly adequate to read
  "this span is ~0," because no plausible run-to-run variance turns 0.1ms into a
  contributor.
- Conservation gap 0.03% — the self-time decompose is balanced; the SERIAL
  bucket is not hiding mass.
- The `SERIAL_BOOKKEEPING` bucket (0.139s, 35.7%) is **dominated by
  `consumer.writev` (0.12-0.13s, ~87-93% of it)** — correctly attributed to the
  output floor in claim (3), NOT to marker bookkeeping. The script computes SERIAL
  as a residual (`wall − DECODE_WAIT`), which would normally be suspect, but the
  itemized per-span list makes the composition visible, so the residual-bucket
  risk is defused here.

No caveat of substance. The marker-resolve/publish chain is not the residual.

---

## Claim (3) — "residual = marker-region pure-Rust BOOTSTRAP compute + writev/bandwidth floor"
### Verdict: SPLIT — output-floor half UPHELD; bootstrap half REFUTED-AS-UNPROVEN

**Output-floor half (UPHELD):** `consumer.writev` 0.12-0.13s is measured, and the
prior output-reconciliation has rg paying a comparable ~15.7ms exposure. The
shared 211 MiB materialization floor is real and faithfully shared. Fine.

**Bootstrap half (REFUTED-AS-UNPROVEN) — this is the brief's weakest move.** The
attribution is reached by ELIMINATION ("not scheduling, not bookkeeping ⇒ must be
bootstrap"), which is precisely the ANALYST-BIASABLE residual attribution the
CLAUDE.md PROCESS forbids ("NEVER conclude a lever from attribution… the only
verdict is a causal perturbation"). Specifically:

- The ISA-L oracle (`gzip_chunk.rs:568-583`) swaps ONLY the clean tail in
  `finish_decode_chunk_impl`. The pre-flip u16 marker decode is a *separate
  earlier* decode and is NOT touched — VERIFIED. So `ocl_cf` **still pays** the
  bootstrap. That means the 21ms (ocl_cf − rg) contains the bootstrap **only to
  the extent rg pays less of it**, and the brief never bounds that fraction.
- The persistent `DECODE_WAIT` (64% of wall, 0.246s, essentially unchanged
  native→ocl_cf) is attributed to "chunk availability," but the brief never
  decomposes WHY chunks are unavailable — worker-side bootstrap compute vs output
  backpressure vs the overshoot re-decodes are not separated. Pinning it on
  bootstrap is a guess.
- **A real removal oracle for this term EXISTS and was not run.** `perfect_overlap.rs:35-38`
  itself names `GZIPPY_SEED_WINDOWS` (seedfull), which flips every chunk to the
  clean engine and thereby *removes the marker bootstrap*. `ocl_cf` vs
  `ocl_cf+seedfull` (both ISA-L-class clean) bounds the marker-bootstrap pure-Rust
  cost directly. Only `SLOW_MARKER_MODE` (a slope, not a removal) was available to
  the brief — and Rule 3 says a slow-down slope ≠ a speed-up ceiling. The brief's
  own disproof-ask #2 concedes this. Until seedfull is run, "the residual is
  bootstrap compute" is unproven.

**To close:** run `ocl_cf` vs `ocl_cf + GZIPPY_SEED_WINDOWS`, interleaved
best-of-N, sha-verified. The delta is the bootstrap term's true size. Until then,
claim (3) may state only "writev floor (measured) + an *unbounded* engine-class
remainder."

---

## Claim (4) — "no low-risk faithful C2 scheduling tooth; material fork update"
### Verdict: UPHELD-WITH-CAVEATS

The *scheduling* sub-claim is well-supported: the flat removal oracle kills the
dispatch-depth tooth, and that was the lowest-risk faithful candidate. As a fork
input, "don't spend another session on dispatch-timing scheduling" is correct and
material.

Two caveats keep it from being a clean "no faithful C2 tooth exists":

1. **A faithful tooth the oracle can't see remains on the table.** rg DOES
   interior-subchunk reuse (`GzipChunkFetcher.hpp:264-288`) — porting it IS a
   faithful structural convergence move (charter-aligned), and PERFECT_OVERLAP
   structurally cannot evaluate it (FACT B). So "no faithful tooth distinct from
   the engine" overreaches. The honest statement is: the one faithful tooth left
   (interior reuse) is bounded *near-zero-payoff on the matched ocl_cf wall* and
   carries a 3-for-3 failed-attempt record ⇒ high-risk / low-payoff, not
   nonexistent.

2. **The fork's "residual is engine-class" leg still owes the seedfull bound
   (claim 3).** Declaring the residual engine-class before running the available
   bootstrap-removal oracle re-commits the attribution error. The fork decision
   should not lean on "bootstrap = engine-class" until that one run lands.

---

## Bottom line

- The headline result — **dispatch-depth head-of-line scheduling is FLAT and dead
  as the 21ms lever** — is REAL, a valid frozen-host removal (not a slope, not a
  turbo artifact). That much survives disproof cleanly.
- But the brief **over-generalizes a depth-removal into a refutation of the whole
  named candidate** (the interior-reuse mechanism is structurally outside the
  oracle), **mis-reads a blind self-test fraction** (`warm_miss` can't separate
  cheap startup from costly overshoot-discard), and **attributes the remainder to
  marker-bootstrap by elimination without running the seedfull removal oracle it
  already has**.
- None of these flip the *practical* conclusion — on `ocl_cf` the residual is
  writev floor + an engine-class remainder, and there is no demonstrated low-risk
  *wall-moving* scheduling tooth. But two cheap, already-built measurements would
  convert the conclusion from "argued" to "removed": **(1)** label `warm_miss` by
  cause (startup / discard / absent); **(2)** run `ocl_cf` vs
  `ocl_cf+GZIPPY_SEED_WINDOWS` to bound the bootstrap term. Do both before this
  brief is allowed to update the engine-fork decision.

---

## FINAL SIGN-OFF (2026-06-08, after both owed measurements ran)

Both measurements I demanded ran. I take each on its own terms, then rule on the
revised conclusion.

### Owed measurement (1) — label `warm_miss` by cause → CLOSES the FACT-A gap.
Result: `misses=2`, `Prefetch guard-rejects=1` ⇒ **1 startup-absent + 1
overshoot-tail discard.** This does two things:

- It **proves the original brief's sentence wrong** ("the 2 residual are the
  offset-0 startup misses" — one of the two was in fact the costly discard, exactly
  the FACT-A path-(ii) I flagged). The blind self-test was hiding a real discard.
  The revised brief (STEP 3a) corrects the sentence. Good — the overclaim is struck,
  not papered over.
- But the corrected count **closes the gap the other way I left open**: the
  interior-reuse mechanism fires on **exactly 1 chunk of 17**, and the wall was flat
  with that 1 costly stall present in *both* overlap arms. The bound now comes from
  COUNT (1 chunk, whose cold re-decode on `ocl_cf` routes through ISA-L = cheap), not
  from the depth-oracle that structurally couldn't see it (FACT B still holds — this
  is a magnitude argument, not a removal of that chunk). 1-of-17 + flat-in-both-arms +
  the standing 3-for-3 failed-patch record ⇒ **interior reuse is bounded negligible.
  Argued → removed.** Claim (1) goes to full UPHELD.

### Owed measurement (2) — `ocl_cf` vs `ocl_cf+SEED_WINDOWS` → CLOSES the claim-3 bootstrap gap.
This is the genuine Rule-3 removal I said was missing (not the `SLOW_MARKER_MODE`
slope). Result: A−B = −2ms / +8ms ⇒ **marker-bootstrap pure-Rust term = 0–8ms,
small and engine-class.** And the load-bearing finding: **seedfull-removed `ocl_cf`
plateaus at 0.983–0.985×, NOT 1.0×** — so a ~6ms residual *survives* bootstrap
removal. That is the first **positive** (removal-based) confirmation that the
remainder is the writev/bandwidth floor, replacing the by-elimination attribution I
rejected. The bootstrap half of claim (3) moves from REFUTED-AS-UNPROVEN to
**bounded-small-and-engine-class**; the output-floor half is now confirmed by
removal, not just measured in isolation. **Argued → removed.**

### Is the revised conclusion EARNED?
> "C2 has no demonstrated low-risk wall-moving faithful tooth; residual = shared
> output floor + small engine-class bootstrap; scheduling null."

**Yes — now earned, leg by leg:**
- *Scheduling null* — dispatch-DEPTH dead by validated frozen-host removal (was
  already solid); dispatch-TARGETING now bounded to 1 chunk. ✓
- *Small engine-class bootstrap* — bounded 0–8ms by the seedfull removal oracle. ✓
- *Output floor dominant + irreducible-by-C2* — earned by the seedfull plateau
  (~6ms survives full bootstrap removal). ✓

### Remaining overclaims to flag (none flips the verdict)
1. **"shared" output floor is INHERITED, not re-measured here.** The floor's
   shared-with-rg character rests on the earlier output-reconciliation turn (rg
   ~15.7ms), not on anything in this brief. State it as "floor that the prior turn
   established as largely shared," not as freshly proven here.
2. **The seedfull B arm is a masks-binder CEILING (not byte-exact).** So 0–8ms is an
   *upper* bound on the bootstrap term — which only strengthens the conclusion (a
   faithful removal would help even less and still miss 1.0×). Keep the "≤8ms /
   ceiling" qualifier; don't quote 8ms as the true cost.
3. **Loaded host (loadavg ~4, spread 85–131%).** The *absolute* split between the
   0–8ms bootstrap and the ~6ms floor is not trustworthy at this noise level. The
   *qualitative* verdict (every C2 sub-term flat-to-small; floor survives bootstrap
   removal) is robust because it does not depend on the split — even the most
   bootstrap-favorable arm leaves 0.985×. A quiet-box re-run would tighten the split
   but cannot change the sign. Carry the caveat on the number, not the conclusion.

### Bottom line
The two cheap removals I asked for both landed and both convert their claims from
argued to removed. The revised brief's conclusion is **EARNED and signed off.** C2
yields no demonstrated low-risk wall-moving faithful tooth: scheduling teeth null,
the one interior-reuse tooth is a 1-chunk negligible, and the bootstrap tooth is a
≤8ms engine-class term that doesn't reach 1.0× even when removed. The residual is the
(prior-turn-established) shared output floor plus that small engine-class bootstrap.
Two caveats on wording — "shared" is inherited and "≤8ms" is a ceiling — and one on
the absolute split (loaded host). The fork input stands: C2 is removed as the cheaper
faithful alternative; this does not by itself escalate the engine fork.
