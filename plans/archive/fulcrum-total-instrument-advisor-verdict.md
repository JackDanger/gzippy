# Disproof verdict — fulcrum_total whole-system instrument

Independent adversarial review. Every claim source-verified against the actual
files in this worktree. Span taxonomy checked against the wired set enumerated
from `grep -rhoE 'SpanGuard::begin(_with)?\("..."` over
`src/decompress/parallel/*.rs` (30 real B/E span names) plus the manual
`lock.wait`/`lock.held` emits in `trace_v2.rs:291-294`.

Bottom line: the tool is **trustworthy for its DESCRIPTIVE purpose** and its
seeding/oracle gates are sound, but two of its self-described "fail-loud"
assertions do not actually do what the brief claims. One assertion is a
tautology; one wall-selection heuristic can silently invert the headline
WAIT/COMPUTE story. Fix both before quoting any number as the wall.

---

## C1 — NO DOUBLE-COUNT (self-time): **UPHELD**

`self_time_by_name` (lines 241-259) charges each span's `dur` to its **direct
parent** only (`child_time[s["parent"]] += s["dur"]`), then
`self = total - child_time`. That is the correct self-time recurrence (a
grandchild is subtracted from its parent, not the grandparent, because `parent`
is the immediate enclosing open span recorded at `pair_spans` line 222).
`assert_no_double_count` (337-344) flags negative self-time, which is the real
signature of an over-charged parent. Selftest #2 confirms the combine_crc
phantom: SUM=1000us, SELF=200us. The SUM column is labeled
`SUM(!=wall)` / "slack-maskable" at print (579, 584-585). This defeats the
combine_crc-62ms class. **No defect found.**

---

## C2 — busy+idle==span "fails loud if violated": **REFUTED (as an assertion)**

The brief and the docstring (claim C2) say
`assert_busy_plus_idle_equals_span` "FAILS the analyze() call (raises) if
violated." **It cannot fail.** In `per_thread_busy_idle`:

- line 314-316: `busy = compute+output+overhead+outer+unknown+wait`
- line 316: `t["toplevel"] = busy`
- line 317: `t["idle"] = t["span"] - busy`

and the assertion (332) checks `abs((toplevel + idle) - span)`. Substituting:
`toplevel + idle = busy + (span - busy) ≡ span`. The check is **algebraically
identical to `0`** for every thread, modulo float noise well under `tol_us=1.0`.
It is decorative — it can never detect the depth-bookkeeping double-count it
claims to guard.

The *actual* exactness guarantee is real but comes from a **different** mechanism:
the boundary sweep (300-313) attributes each `[prev_time, tm)` slice to exactly
one leaf, so `busy <= span` by construction and `idle >= 0` always. That
construction is sound. But "busy+idle==span is ASSERTED" (printed verbatim at
line 566) overstates it: the equality is *defined into existence*, not *checked*.

**Fix:** compute `idle` **independently** — sum the slices where the active
stack is empty (a real gap measure) — and assert `busy + independent_idle ==
span`. Only then does a double-count (busy > covered time) actually raise.

---

## C3 — WAIT vs COMPUTE vs OUTPUT classification: **UPHELD-WITH-CAVEATS**

I enumerated every wired B/E span and ran each through `classify` (143-164).
Every genuine block-on-another-thread wait is correctly `wait`:
`ttp.rx_recv_block` (the ~97% binder, block_fetcher.rs:248), `ttp.get_if_available`,
`pool.pick.wait`, `lock.wait`, `consumer.wait_replaced_markers`,
`consumer.dispatch_recv` (wait wins over its outer-frame membership because WAIT
is checked at line 154 before the outer check at 156). **No blocking wait is
mislabeled compute** — the inversion the campaign feared is not present.

I verified the two compute-classified spans that *look* like they might block:
- `consumer.block_finder_get` → `gzip_block_finder::get` (gzip_block_finder.rs:176-206)
  is a brief mutex + partition-arithmetic guess; it does **not** block on a finder
  thread (unlike vendor's blocking `BlockFinder::get`). `compute` is correct.
- `consumer.get_last_window` (chunk_fetcher.rs:1687) is `materialize_window` +
  publish — local compute, not a wait. Correct.
- `ttp.rx_recv_block` wraps a `recv_timeout(1ms){ pump_prefetch() }` loop
  (block_fetcher.rs:256-261). `pump_prefetch` opens `coord.prefetch_call`
  children, so by leaf attribution the real dispatch work is charged to
  `coord.prefetch` (compute), not buried as wait. Good.

Caveats (none change a verdict, all worth fixing):
1. **Comment/code mismatch.** Lines 90-93 claim `ttp.take_prefetch` is
   "classified as wait"; the code puts it in `COMPUTE_PREFIXES` (122) → compute.
   Harmless (take_prefetch is a non-blocking handle handoff, block_fetcher.rs:239)
   but the comment is wrong and will mislead a future reader.
2. **`ttp.get_if_available` is non-blocking** (a ready-probe, block_fetcher.rs:233-234)
   yet classified `wait`. Conservative and negligible in magnitude, but it
   slightly inflates "wait" — not strictly a block-on-another-thread.
3. **`blockfetcher.cache`** appears only as a doc-comment example
   (trace_v2.rs:258), not a wired span → never reaches `classify` → no impact.
   (If `lock_span!` is ever called with a real name, that name flows through
   `lock.wait`/`lock.held`, both already classified.)

No real engine cost is hidden as wait; no wait reads as serial compute. Upheld.

---

## C4 — WINDOW-ABSENT-PRESERVING routing guard: **UPHELD-WITH-CAVEATS**

The regexes (367-374) match the real emit verbatim:
`chunk_fetcher.rs:750` prints `... window_seeded=N bad_seed_resync=N ...` and
`:760` prints `isal_oracle_chunks=N isal_oracle_fallbacks=N`. The whole block is
gated by `if std::env::var("GZIPPY_VERBOSE").is_ok()` (chunk_fetcher.rs:691),
which the capture sets (`GZIPPY_VERBOSE=1`, capture line 79). `seeded>0 ⇒ REFUSE`,
`oracle>0 ⇒ REFUSE`, and `no_flip==0 && flips==0 ⇒ INCONCLUSIVE` (427-433).

**The "benign chunk-0 seed" worry is unfounded.** `WINDOW_SEEDED_CHUNKS`
increments *only* when `initial_window.len() == MAX_WINDOW_SIZE` (32 KiB)
(gzip_chunk.rs:790-791). Chunk 0 in window-absent production has an **empty**
initial window, so it never increments; only a real 32 KiB pre-seed (the
`GZIPPY_SEED_WINDOWS` oracle) does. So `window_seeded=1` genuinely means "a chunk
routed to the clean engine," and REFUSE is correct — no false-REFUSE from a
legitimate chunk-0.

Caveats:
1. **Partial/empty sidecar → INCONCLUSIVE, not RAISE.** If the verbose file
   exists but lacks the patterns (truncated run, error before the end-of-run
   print), `parse_counters` returns `{}` → `seeding_guard` returns
   `(None, "NO COUNTER SIDECAR...")` → printed `[INCONCLUSIVE]` (555). It does
   **not** raise. A careless user can proceed past INCONCLUSIVE. The brief's
   exact concern ("reads as no sidecar => inconclusive but the user proceeds")
   is real. Consider making "sidecar present but unparseable" a distinct hard
   refusal vs "no sidecar at all."
2. **Binary `seeded>0` can't grade.** It cannot distinguish "a few chunks
   naturally seeded once windows propagated" from "oracle pre-seeded everything."
   For the *current* architecture this is fine — production passes an empty
   initial_window and never trips it — but it is brittle to any future eager-seed
   optimization that legitimately hands a 32 KiB window to a mid-stream chunk.

---

## C5 — ORACLE CONTAMINATION check: **UPHELD**

`oracle_overhead_guard` (436-459): `isal_oracle_chunks>0 && isal_oracle_fallbacks>0`
⇒ "ORACLE IMPURE ... wall is a BLEND" (448-451); the counters exist and are
emitted (chunk_fetcher.rs:760-763). `print_delta` additionally voids a
cross-tool verdict if either side is non-production (619-621). Selftest #7
exercises the impure-blend path. The `to_vec`/`oracle_copy`/`oracle_alloc`
name heuristic (454-458) is best-effort: no such span is currently wired, so it
won't false-positive, but it also wouldn't catch an alloc-overhead oracle that
used a different span name — a known, acceptable limitation. Upheld.

---

## C6 — SELF-VALIDATING (--selftest): **UPHELD-WITH-CAVEATS**

The positive control (717-734) genuinely **localizes**: it compares
`self_time_by_name` per name, so injecting +50% into `worker.decode` moves only
that name 1.5x while `consumer.writev` stays 1.0x — a smear would necessarily
move writev too, so a pass really does prove per-name isolation. Negative
control, seeding, oracle, empty-output, and end-to-end analyze() classes are all
covered. 21/21 reportedly passing is consistent with the code.

Caveats:
1. **The selftest never exercises the two weak spots above.** All synthetic
   traces are clean depth-0 sequences or single-level nests (`_synth_trace`,
   `_synth_nested`). None contains overlapping/non-nested spans, a B/E mismatch,
   or a worker thread with a span >= the consumer's. So the suite cannot catch
   (a) the C2 tautology, nor (b) the wall-thread inversion (attack #2). It
   validates the *named historical* failure classes, not the *structural* edge
   cases the brief asks about.
2. The positive control tests `self_time_by_name` directly, **not** the
   per-thread leaf breakdown or the busy+idle path — so the breakdown's
   localization is asserted nowhere.

---

## C7 — OFF==identity (read-only post-processor): **UPHELD-WITH-CAVEATS**

The analyzer only reads (`analyze`/`print_*` open files read-only; no production
import). The capture sets only `GZIPPY_TIMELINE`/`GZIPPY_VERBOSE` and
**sha-verifies** output ==ref (capture lines 88-90), with explicit empty-trace
and sha-mismatch gates (85-90). Byte-transparency holds.

Caveats:
1. **Tracing perturbs TIMING, not bytes.** Each span does a mutex + `format!` +
   buffered write (trace_v2.rs:158-189, 124-133). The *traced* wall is inflated
   vs the untraced production wall, so absolute wall numbers from a traced run
   are not the production wall — fine for relative/structural reading, but a
   reader must not quote the traced consumer span as "the production wall."
2. **Cross-tool delta assumes rapidgzip uses gzippy's span names.** `print_delta`
   (602-617) reads `right["consumer"]`'s compute/wait/output, but every prefix in
   the taxonomy is a gzippy name (`consumer.*`, `worker.*`, `ttp.*`). A genuine
   rapidgzip Chrome trace would classify **all-unknown**, making the right-side
   compute/wait split ~0 and the delta meaningless, unless rapidgzip is
   instrumented with identical names. The single-trace gzippy analysis is sound;
   the `vs A B` cross-tool path is not, as-is.

---

## Attack-target roundup

1. **busy+idle exactness / overlapping / zero-len / B-E mismatch** — coverage is
   exact *by construction* (leaf sweep) but *not by the assertion* (C2 tautology).
   Overlapping non-nested spans would corrupt LIFO pairing (pair_spans 218-222),
   inflating `mismatched` — which is only **warned** (546), never raised. RAII
   `SpanGuard` Drop makes intra-thread nesting clean in practice (real trace:
   ~0 mismatched), but a silent violation is *possible in principle* and the
   instrument would not raise on it. **Partial defect — see C2 fix.**
2. **consumer = max-span thread (`consumer_tid`, 355-360)** —
   **UPHELD-WITH-CAVEAT, this is the most dangerous heuristic.** Thread-pool
   workers are long-lived; a worker whose first event precedes and last event
   follows the consumer's would have a larger span and **steal the wall-critical
   label**, flipping the headline 98.5%-WAIT story into a compute story. It
   happened to pick correctly on the real trace, but it is not guaranteed.
   **Fix:** select the consumer structurally — the unique thread that owns
   `consumer.iter`/`consumer.drain` (outer frames are consumer-exclusive) — not
   by max span.
3. **WAIT taxonomy completeness** — complete; no blocking wait mislabeled compute
   (see C3). Only nits (take_prefetch comment, get_if_available conservatism).
4. **Routing guard fooled** — not fooled by the real line format; the soft spot
   is unparseable-sidecar → INCONCLUSIVE-not-raise (see C4 caveat 1).
5. **Positive control localizes, not smears** — yes (C6), but it doesn't cover
   the leaf-breakdown or the wall-thread selection.
6. **Descriptive != causal / over-claim** — **UPHELD-WITH-CAVEAT.** The tool is
   mostly disciplined (SUM!=wall labels 579/584, "wait is NOT serial work" note
   574-575, refuses seeded deltas 619-621). But it **nowhere states** that its
   output is DESCRIPTIVE structure, not a causal binder verdict, and that a
   binder claim still requires the `GZIPPY_SLOW_BOOTSTRAP` perturbation + locked
   guest. A reader could over-read the "WALL-CRITICAL THREAD" split or the
   SELF-time ranking as "X is the binder." **Fix:** print a one-line banner —
   "DESCRIPTIVE only; a binder verdict requires a causal perturbation."

---

## Required fixes (priority order)

1. **C2:** make `assert_busy_plus_idle_equals_span` non-tautological — derive
   `idle` from real empty-stack gaps and assert against it. As written it is a
   no-op and the "fails loud" claim is false.
2. **Attack #2:** select the consumer by ownership of `consumer.iter`/
   `consumer.drain`, not by max span, so a long-lived worker can't invert the
   WAIT/COMPUTE headline.
3. **C7/attack #6:** add a "DESCRIPTIVE, not causal" banner to `print_bundle`.
4. Minor: fix the `ttp.take_prefetch` comment (C3 caveat 1); make unparseable
   sidecar a distinct hard refusal (C4 caveat 1); gate or warn that `print_delta`
   needs identical instrumentation on the right-hand trace (C7 caveat 2).

With fixes #1 and #2 the instrument is sound for ending the campaign's
instrument-failure pattern. Until then, treat the printed busy+idle "assertion"
and the wall-critical thread label as **unguaranteed**, and never read a SELF
ranking or the wall-critical split as a binder verdict without a perturbation.

---

# RE-REVIEW (fixes)

Re-read of `per_thread_busy_idle` (262-332), `assert_busy_plus_idle_equals_span`
(335-359), `consumer_tid` (383-406), and selftest cases #1b and #11.

## Fix #1 — busy+idle==span made non-tautological: **UPHELD**

`covered` and `idle_gap` are now **disjoint independent accumulators** in the
sweep (309 vs 311): a slice charges `covered` *iff* `active` is non-empty, and
`idle_gap` *iff* `active` is empty. `idle` is no longer `span - busy`, so the old
algebraic identity is gone. The assertion now checks two relations against three
independently-built quantities:

- `busy == covered` (353). `busy` is the sum of the per-class buckets; `covered`
  is the single-charge-per-slice ground truth. These are coupled in the *current*
  loop (one class charge + one `covered` charge per slice), so they hold today —
  but that is the point: it pins the real invariant that **each slice is charged
  to exactly one bucket**. The canonical leaf-sweep double-count (charging
  ancestors *in addition to* the leaf, i.e. iterating `active` instead of taking
  `max depth`) would add to multiple class buckets while `covered` still
  increments once → `busy > covered` → **FIRES**. Under the old `idle=span-busy`
  form that same inflation was absorbed into a negative `idle` and the sum stayed
  `==span`, so it could never fire. That is the exact gap C2 named, now closed.
- `covered + idle == span` (356). Since the sorted boundary slices partition
  `[first,last]` and each lands in exactly one of the two disjoint branches, this
  validates the sweep's slice arithmetic (prev_time bookkeeping / stack mistrack);
  a dropped or mis-summed slice diverges and fires.

Selftest #1b proves sensitivity: corrupting `covered` fires *both* checks, and
corrupting `idle` independently fires `covered+idle!=span` — neither is possible
under a tautology. The assertion is wired into `analyze` (537-543) and raises.

Caveat (non-blocking): #1b corrupts the output dict, not the sweep, so it proves
the *assertion* bites but not end-to-end that a doubled-charge sweep produces the
divergence. The reasoning above shows it would; a stronger test would run an
actual all-`active` attribution and assert it raises. Verdict unaffected.

## Fix #2 — consumer by frame ownership, not max span: **UPHELD**

`consumer_tid` now accumulates `consumer.iter`/`consumer.drain` duration per
thread and returns the max **owner** with method `consumer-frame-owner`
(400-403). Because those outer frames are emitted only on the consumer thread,
the wall-critical label is now assigned structurally; a long-lived pool worker
that spans wider can no longer steal it. Selftest #11 is the direct disproof of
attack #2: worker tid=2 spans 0..2500 (wider than the consumer's 0..2000), yet
`consumer_tid` returns `(1,1)` via `consumer-frame-owner` — a max-span heuristic
would have picked tid=2 and inverted the WAIT story into a compute story. The
inversion risk is removed for any trace that carries the consumer frame; the
only residual is the degraded no-frame case, which now falls back to max-span
**with an explicit `FALLBACK-...` flag** that `print_bundle` surfaces as a `[WARN]`
(612-615). Honest and bounded.
