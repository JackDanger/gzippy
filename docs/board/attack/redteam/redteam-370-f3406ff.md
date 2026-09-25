# Red-team review — PR #370 (promotion-rule.md clause-5/6 size-spend amendment) + fulcrum `f3406ff` (size-spend ceiling, presumable PR #35)

Reviewer: hostile red-team pass, read-only, run 2026-09-05.
Attacked: (1) the 2026-09-05 clause-5/6 entries in
`.worktrees/rules/docs/promotion-rule.md` (commits `e97f54dd` 00:40:25, `3a890557`
00:44:54); (2) fulcrum `f3406ff` 00:59:32, branch
`owner-2026-09-05/size-spend-ceiling` (checked out in `fulcrum/`, presumed head of
fulcrum#35). Same author wrote the rule entries, the fulcrum commit, and
`CLAUDE.md`'s 2026-09-05 box within 20 minutes of each other, on the evening of
the same day as the owner arbitration.

Ground truth used: PR #367's frozen-box verdict (handoff-2026-09-04.md ADDENDUM,
L246-251): **NO-SHIP on the pre-amendment rule**, clause 4/6 both OK (two L2 wall
cells closed — `gzip:L2:T1` 1.25x→0.58x, `pigz:L2:T1` — improvement 3.9x vs
harm), **one size conviction: silesia L2/T1 0.9973 → 1.0000 vs libdeflate** (Δ
0.0027 = 0.27%). The amendment authorizes exactly that delta; the instrument
commit says so in its own changelog bullet and canonizes the number in a
selftest fixture (`promote.rs:3860`: "exactly 0.27% erosion").

Not verified (stated plainly): the PR #370 body and fulcrum#35 body (not in the
repo); whether PR #370 already carries the discipline checklist the handoff
claims (I can check the artifacts on disk only). Everything else below is cited
to files and lines read in full.

Verdict up front: **NO-GO for fulcrum#35 as-is.** Four of the findings below are
one-turn fixes (doc sentences, fixture re-pins, one additive guard, changelog
lines); two are structural (the unpriced aggregate spend needs an owner
decision; the rescore-merge gap needs its own PR). Details and exact fixes
follow.

---

## ATTACK 1 — Symmetry: would this amendment exist if #367 had SHIPped? (HIGH)

### The steelman against this review, written out first

The defense of the amendment is real, and a fair review must state it at full
strength:

- **The directive is independent of the verdict.** The owner's arbitration on
  the L2 fork ("I delete the pick-min … We can take a <1% hit to compression
  size but we will not lose on wall clock under any conditions",
  handoff-2026-09-04.md L237-239) is a *policy* decision about what the encoder
  is allowed to cost, not a re-fit of a measurement. Pick-min deletion makes a
  size give-away *unavoidable*: without a ceiling in the rule, every honest
  one-encode change eats "size is exact, any positive delta is harm"
  (`promote.rs:216-218`, pre-f3406ff wording) forever, and clause 4/6 can never
  be satisfied by any variant of the owner's own directive. A rule that
  structurally forbids the campaign owner's stated trade is mis-specified, and
  fixing a mis-specificaton that *would also have applied if #367 had SHIPped*
  passes the symmetry test by construction.
- **The number is not fitted to the trigger.** A verdict-rescuing ceiling would
  sit just above 0.27% (0.3%). The amendment took a round 1%, 3.7x above the
  motivating spend — the most defensible number available, and the one the
  directive's own "<1%" implies.
- **The direction was already settled 26 days earlier.** The 2026-08-10
  margin-floor redesign ("margin is capital", receipts #295/#296/#310) and the
  2026-08-11 residual-harm fix (the #310 double-billing receipt) both moved
  clause 6 from "sum all deterioration" toward "price what was authorized,
  price nothing twice". The size ceiling is the same doctrine extended to the
  axis the owner just re-priced. It also matches owner #356 ("if we have to
  tie, we tie on size", 2026-08-22).
- **Process letter satisfied.** Rule file changed in a separate worktree/branch
  (`e97f54dd`, PR #370), instrument in a separate repo/branch (`f3406ff`), and
  only then is the affected change re-adjudicated — exactly the 2026-07-28
  preamble's own protocol (promotion-rule.md L16-20: "The gate was NOT changed
  in the branch that failed it … and only then is that change re-evaluated").

### Rebuttal — where the steelman fails against the document's own tests

The rule file's proxy-correction table (promotion-rule.md L159-168) has four
legs. The author runs **one** of them:

1. **Timing — passed.** Separate branches, rules before re-adjudication, both
   PR-numbered. No dispute.
2. **Symmetry — unanswerable as stated, but the tell is in the fixtures.** The
   fulcrum selftest now contains a fixture whose *comment names the motivating
   number* (`promote.rs:3860` "exactly 0.27% erosion") and asserts the exact
   verdict the amendment was manufactured to produce (SHIP). Every future
   `selftest` re-certifies the motivating case as SHIP. The instrument has
   burned the triggering verdict into its gate. That is the shape the table
   exists to catch ("bless the present patch", L166).
3. **Backtest — not performed.** Nothing in PR #370's texts or f3406ff applies
   the new ceiling to the stored adjudications (#295/#296/#310, #366-#369) to
   show it predicts board progress better rather than flipping one verdict. The
   2026-08-10 redesign, by contrast, *has* a receipt list of past adjudications
   re-examined. This matchable standard was skipped.
4. **Counterfactual, written first — not written.** No text anywhere in
   promotion-rule.md's 2026-09-05 entries, CLAUDE.md's warning box, or the
   fulcrum CHANGELOG states what measured result would *revert* the ceiling.
   The 2026-08-10 redesign at least enumerated the mis-adjudications that
   grounded it.

The deeper rot: the amendment needs, and lacks, one honest sentence of
classification. The 2026-07-28 preamble (L3-4, L17-20) forbids adjusting the
rule "after a measurement comes back inconvenient" and says that "'my proxy was
wrong' is exactly what it would sound like from the inside either way" (L19-20)
— the one move the rule forbids outright. #367's conviction was **correct under the
rule as written** — size is exact integers, the erosion was real, the old rule
did exactly what it said. This is not a proxy correction; it is an owner
*re-pricing of the contract*. That distinction is never stated, and the
amendment instead wears the costume of a correction ("amended 2026-09-05 (owner
directive, verbatim receipt)"). Justified or not, the text must say plainly
which move it is, or the rule file accumulates indistinguishable entries — the
third time this document rewrites itself after an inconvenient result, the
"proxy correction vs excuse" table becomes unfalsifiable.

**Fix (one turn):** add to promotion-rule.md's 2026-09-05 entry: (a) the
explicit sentence "This is a policy re-trade by the owner, not a proxy
correction — #367's pre-amendment adjudication was correct under the old rule;"
(b) a one-paragraph backtest (which past adjudications would the ceiling have
changed — inspect stored `try.json` clause6/convictions; the expected result is
"only #367, and only its one cell", which is itself evidence *for* review, not
against); (c) a pre-committed revert trigger (e.g. "if the first
ceiling-blessed SHIP's nightly board shows a per-label size delta > 1% from
this spend, the ceiling reverts to exact-size"). Add the same receipt to the
fulcrum CHANGELOG entry (see Attack 5).

---

## ATTACK 2 — Double-spend on one lattice; the authorized size spend and the total-spend hole (HIGH)

### What the amendment actually implements

`SIZE_SPEND_CEILING = 0.01` (`promote.rs:273`), judged per cell: a passing size
cell with census delta ≤ 0.01 (+1e-12, `promote.rs:796`) is classified
`AUTHORIZED-<=1%-SIZE-SPEND` and `continue`s (`promote.rs:794-813`); the
clause-6 ledger then buckets *every* positive size delta ≤ the ceiling into
`excluded_size_spend` (`promote.rs:1053-1066`) rather than charging harm. Deltas
> 1% convict clause 5 (`promote.rs:814-821`) and stay harm (the re-pinned
(6b)/(6c') fixtures pin this).

Two consequences the commit message and rule text do not spell out:

1. **The ceiling is not an addition to the flat budget on the size axis — it
   deletes the flat budget's role there.** `erosion_budget()` is capped at
   0.005 (`promote.rs:260-262`), so *every* authorized spend (0 < Δ ≤ 1%) is
   above the flat budget on every passing size cell. Pre-f3406ff, (a) any Δ >
   flat-budget size erosion convicted clause 5 outright (that is how #367
   died), and (b) sub-budget size deltas were still charged harm at clause 6 —
   the old (6b) test said so verbatim: "an exact size regression on a winning
   cell is harm even INSIDE the flat budget". Post-f3406ff, the *entire* (0,
   1%] band is unpriced. The size axis goes from two backstops to zero, with
   no replacement total.
2. **Nothing bounds TOTAL spend.** The ceiling is per-cell only. `c6.harm` is
   the sum of `confirmed_real + size + undecided` (`promote.rs:1076`) — all
   authorized quantities are absent by design — and clause 6 auto-passes at
   `harm <= 0.0` or `improvement >= 2x harm` (`promote.rs:1097-1101`). The
   ledger loop has no aggregate constraint of any kind
   (`promote.rs:1048-1075`); `excluded_size_spend` is an output, never an
   input to any comparison. The rule file's own thousand-cuts promise — "the
   floor is a hard line no accumulation of 'harmless' degradations may cross"
   (promotion-rule.md L88-89) — is now simply false on the size axis:
   accumulation crosses nothing instrument-visible.

### The 36-cell trace (both authorizations eaten, and the residual zeroed)

Take a 36-cell lattice (each coordinate with wall and size readings; a change
in scope). The change closes **one** failing wall cell
(`gzip:L2:T1`-shaped, improvement 0.67 — clause 4 OK via closed cell). Then:

- **Size axis:** all 35 passing size cells erode by 0.0099 each (Δ 0.99%).
  Individually: each is ≤ ceiling, `authorized` (`promote.rs:796`), excluded
  (`promote.rs:1060-1062`). Aggregate: `excluded_size_spend = 0.3465 [35]`.
- **Wall axis:** each passing winning wall cell erodes within the margin floor.
  A census post already under `min(0.80, 1-3*floor)` is ACCEPTED *without
  confirmation* (`promote.rs:866-885`), confirmed-real posts that clear the
  floor are ACCEPTED (`promote.rs:921-930`), all bucketed into
  `excluded_margin_spend` (floor-priced). Aggregate: up to `0.80 − base` per
  cell — no aggregate cap either (pre-existing 2026-08-10 semantics, unchanged,
  but note they now *compound* with the size spend).

Adjudication of this census: clause 3 OK (no flips); clause 4 OK (closed cell);
clause 5 OK (all spends accepted or authorized — `promote.rs:1003-1011`);
**clause 6: `harm = 0.0000` → "improvement 0.6700 vs residual harm 0.0000 (>=2x
or no harm)" (`promote.rs:1097-1101`) → SHIP.** Byte-for-byte the same census
before f3406ff: clause 5 convicted on every 0.99% size cell that cleared its
budget, and clause 6 charged the sub-budget remainder → NO-SHIP.

The amendment's own receipt sentence — "the size budget is its replacement
pricing, not a new lever to spend twice" (promotion-rule.md L59-61) — is
unimplemented at the aggregate level: it prices *single cells* and prices
*nothing* jointly. Note also the "priced" word is false advertising for the
size bucket: the wall margin-spend is priced by *measured floor arithmetic*
(`post <= min(0.80, 1-3*floor)`); the size spend is priced by owner fiat with
no measurement procedure and no floor. Excluding a permission is not pricing
it.

### Can the size spend HIDE a wall harm via the exclusions list?

Direct answer: **not through the harm arithmetic** — the wall harm channels are
independent and still live: floor-REJECTED confirmed-real erosions charge harm
(`promote.rs:943-945`), UNDECIDED suspects charge their census deltas
(`promote.rs:958-985`), thin-margin confirmed breaches charge
(`promote.rs:893-920`), clause 3 flips are absolute. I checked the confirm
short-circuit too: excluding size spend makes verdicts *more* ship-shaped, so
the "skip confirms only when NO-SHIP even under best-case acquittal" rule
(`promote.rs:1898-1931`) short-circuits *less* often — audit-positive, no
hiding there.

What the size bucket *does* hide is the **total give-away**, and it does so by
joining the clause-6 line's exclusion family: `promote.rs:1079-1081` prints all
four excluded buckets under one prefix, "excluded as clause-5-priced:" — under
which the size spend, which was *not* priced by clause 5 (no floor, no confirm
— owner fiat), is indistinguishable to anyone reading the artifact line the way
the 2026-08-11 receipt is usually quoted in approval discussions. A run can now
end at `harm = 0.0000` with a four-line exclusion tail; clause 6 never runs a
Pareto comparison at all. The 2026-08-11 receipt's point "stands in both
directions" (promotion-rule.md L121-123: over-billing authorized spend is a
flat budget in disguise) — the under-billing direction is implemented as an
unbounded exclusion. Half a receipt.

**Fix:** this is the one finding that needs an owner decision, then one turn:
the directive says "we can take **a** <1% hit" — singular. Two coherent
implementations, and the repo must carry exactly one: either **(1) run-total
cap**: `excluded_size_spend` summed over all in-scope cells must be ≤ 0.01 per
run; overflow convicts clause 5 with a "total size spend exceeds the single <1%
hit the directive authorized" chain, keeps the per-cell ceiling as the local
edge; or **(2)** the owner re-states the per-cell reading in the rule file with
a measured total they ratify (any number at all — 1%, 5%, ∞ with a sentence
explaining why Pareto-over-failing-cells makes totals safe). Without one of
these, the amendment's thousand-cuts sentence should be deleted rather than
departed from. Add the compound gate-0 fixture (35 size spends + floor-priced
wall spends + one closure → expected verdict under the chosen policy), which is
currently absent precisely because it would pin the answer.

---

## ATTACK 3 — Edge semantics

### 3a. The ceiling edge: "≤ 1%" vs the owner's "< 1%" vs the code's "<= 1% + 1e-12" (LOW, must be settled in the same turn)

Three documents disagree about the half-open interval: the owner's verbatim
directive is "<1%" (handoff L237-239); the amended rule text adopts "<= 1% …
> 1% convicts" (promotion-rule.md L54-58); the code authorizes `delta <=
SIZE_SPEND_CEILING + 1e-12` (`promote.rs:796, :1060`) — that is, ≤ 1% **plus
fudge**. A Δ of exactly 0.0100 (achievable with exact integers: ours = 1.01 ×
rival bytes exactly) is authorized by code and rule text but exceeds the
directive's strict inequality; 0.010000001 convicts. Nothing pins either side
of the boundary in the selftest fixtures ((a) uses 0.0027, (b) uses 0.02 —
nothing at the edge), and the chain text prints `{:+.4}`
(`promote.rs:798-807`), which cannot distinguish 0.00997 (spend) from 0.01003
(convict) in the printed artifact — the raw census fields save the audit, but
the boundary is load-bearing precisely because #367 sits 0.0097 below it.

Adjacent, related, and cheap to state: the tie endpoint is real. Size pass
means ours ≤ rival bytes (`sizecensus.rs:233-251`, `bigger = gzippy_bytes >
rival_bytes`; matches the owner's #356 "tie on size"), so a cell can erode to
exactly 1.0000 and remain passing; the ceiling floor there is the *flip
border* itself. Good — but pin it.

**Fix:** write the interval convention down once in
`CLAUSE5_MARGIN_FLOOR_RULE` ("deltas ≤ 0.01 are spend; > 0.01 convict; the
ratio is ours/rival"), align the owner-quote paraphrase (or keep "<1%" and make
the code exclusive), and add two boundary fixtures: Δ = 0.0100 (authorized per
rule text) and Δ = 0.0100001 (convicts). The +1e-12 slop matches the
pre-existing flat-budget convention (`promote.rs:791, 1067`) but must be
declared, defaulting not to widen.

### 3b. The flag stage: the flat census budget still gates which size deltas get chains; sub-flag spends are itemized aggregates only (LOW)

The clause-5 loop enters a size cell only when its delta exceeds
`erosion_budget(base)` (`promote.rs:790-791`) — the ≤0.005 census flag still
decides *which* size deltas print a `[size-spend]` chain. The judgment is the
ceiling, the flag is the old budget: a sub-flag delta (e.g. base 0.990 →
0.9924, Δ 0.0024, budget 0.0025) prints **no per-cell chain** yet is still
excluded in the ledger — only the aggregate `authorized size spend <=1% 0.0024
[1]` line exists (`promote.rs:1060-1062, 1077-1096`). The (d) fixture
demonstrates this transit: one spend has a chain, one is ledger-only
(`promote.rs:3925-3939`). This is not a conviction hole — the ceiling is the
binding judgment either way — but it IS an audit asymmetry: a run can exclude
dozens of sub-flag size cells with zero per-cell audit lines. No analog flag is
*needed* for verdict correctness (unlike wall, no confirmation machinery exists
to gate); what is needed is symmetric itemization.

**Fix:** print one `[size-spend]` line per excluded cell regardless of the flag
(same text, minus the false "Δ > flat budget" clause for sub-flag deltas), or
append `excluded_size_spend` cell IDs to the clause-6 artifact block. Cheap.

### 3c. Multi-cell aggregation — does the per-cell cap dodge "death by a thousand cuts"? What limits TOTAL spend?

Answered in full in Attack 2: the ≤1%-per-cell cap *is exactly the dodge the
rule file's own clause promises to prevent* (promotion-rule.md L88-91). What
limits total spend today: clause 4 (needs one closure or 1% fail-gap drop —
orthogonal to give-away scale), clause 6 (harm==0 auto-pass), clause 8 (size is
exact — noise floor irrelevant), clause 7 (per-arch, same hole in each). **The
instrument imposes no total limit.** Six spends of 0.6% each are six
authorizations and zero charge; thirty-six spends are the same, priced nowhere
except an audit line.

### 3d. Rescore semantics: is rescore-with-new-rules a rule-winding hazard? (MEDIUM)

What f3406ff's rescore *does* when rescoring a stored NO-SHIP like #367: the
newly-authorized size cell changes clause 5 from FAIL to a `[size-spend]`
chain; but (i) any VOID cell in the artifact forces UNDECIDED with a re-run
demand (`promote.rs:488-493`, the 3755-3769 selftest), and (ii) any wall
suspect the *current* rules flag without a stored confirm is UNDECIDED via
`RESCORE_CANNOT_MEASURE` (`promote.rs:2783-2792, 2701`). #367's artifact
contains the VOIDed pair named in the handoff — so **the rescored verdict of
the motivating artifact is UNDECIDED, not SHIP**; the handoff's "Under the
amended ceiling that cell is an authorized spend and the trunk re-adjudicates"
(handoff L250-251) overstates what the instrument can do on paper. The re-ship
necessarily burns a full re-run (which the handoff itself schedules, L259-264).
Fine — but then the *winding* is: a loosening rule cannot flip stored NO-SHIP
straight to SHIP through rescore; it flips it to UNDECIDED, which is
re-run-until-green pressure applied by paperwork. The guards that exist:
`stored_verdict` echo + `verdict_matches_stored` (`promote.rs:2866-2868`), the
explicit mismatch print "rescored to X under the CURRENT rules (the rules or
floors changed…)" (`promote.rs:2965-2973`), append-only `try-rescore-N.json`
(`promote.rs:2913-2924`), floors-refusal (`r4`, `promote.rs:4135-4151`), and
the artifact carries which rule string adjudicated it
(`CLAUSE5_MARGIN_FLOOR_RULE`, `promote.rs:2855-2862`) plus the stamping
`fulcrum_commit` (`promote.rs:2870-2872`). Real guards. What is missing:

- **The LOOSEN direction is unpinned.** Gate-0 has only the tighten-direction
  fixture (r2: stored SHIP → rescored NO-SHIP, `promote.rs:4089-4114`). There
  is no fixture pinning "stored NO-SHIP + sub-ceiling size erosion → rescored
  to X". Worse, f3406ff **re-pinned every sub-ceiling payload in the suite to
  >1%** — (6b), (6c), (6c'), and (r2) all moved from Δ 0.0024/0.0045 to Δ
  0.02 — so after f3406ff there is *no fixture anywhere* covering the
  (0.005, 0.01] band, i.e. the exact class of stored-NO-SHIP artifacts that
  rescore will meet in the wild (that is #367's band). The suite certifies the
  new rule's conviction classes and dropped the ones that ever flipped.
- **No reclassification ledger.** The stored artifact carries the old
  `adjudication.clause6` (`promote.rs:2865`-copied from the live writer,
  L2295); the rescore can therefore computably itemize "verdict moved NO-SHIP
  → UNDECIDED because class X re-classified to AUTHORIZED-<=1%-SIZE-SPEND on
  cells [ids]". Today the human gets the clause-line diff only, which is not
  grep-safe for the winding question "how much did the rule change move?".

**Fix (one turn):** add the mirrored pin — a stored NO-SHIP artifact whose only
conviction is a sub-ceiling size erosion must rescore to the *expected* verdict
with the reclassification itemized (compute the clause6 diff between the
stored and recomputed artifacts and print it as `reclassifications`). Keep the
r2 fixture as-is. This is the honest mirror of the guard the 2026-08-11
receipt built for the other direction, and it costs one fixture plus one
serialized field.

### 3e. Discovered while attacking: the motivating fixtures test a census state the size axis cannot produce (HIGH)

Fixture (a) — the receipt-story test for the motivating #367 spend — is
`cell("size", 2, 1.000, false, 1.0027, false)` (`promote.rs:3860-3864`), and
(d) reuses the same payload (`promote.rs:3930-3931`): an **after ratio of
1.0027 flagged PASSING**. On the size axis that census state is impossible:
`sizecensus.rs:233-251` sets `bigger = gzippy_bytes > rival_bytes` — ours
bigger by 0.27% is a LOSS (failing), i.e. a clause-3 flip, not an authorized
spend. The fixture fleet itself disagrees on the convention: fixture (c) in the
same block encodes 0.999 → 1.001 as a flip (`promote.rs:3898-3912`). So the
two fixtures assume contradictory pass/fail semantics for size cells whose
after-ratio exceeds 1.0, and the SHIP-pinning half of the pair is the
impossible one. The real #367 payload is 0.9973 → 1.0000 (a
passing-at-the-tie spend, Δ 0.0027 — the tie endpoint under the "tie counts as
pass" bar). The suite should pin reality: **re-pin (a) and (d) to
`0.9973 → 1.0000, after_failing=false`** (still Δ 0.0027, still authorized,
SHIP preserved), and add `1.000 → 1.0027, after_failing=true → clause 3
convicts` as the impossible-payload guard. Without this, the motivating case
the rule change receipts is asserted by a test that has never tested it.

---

## ATTACK 4 — The rescore-merge gap: two stored artifacts, disjoint VOID sets, no honest operation merges them (MEDIUM-HIGH, needs its own PR)

`cmd_rescore` accepts only `--rescore <dir> [--layout-floors <tsv>]` — any
other argument is refused (`promote.rs:2928-2950`) — and nothing anywhere in
promotion code merges two stored try.json artifacts (grep: no merge op in
`promote.rs`; the only merge machinery in the instrument is
`wallcensus report`'s display merge, `wallcensus.rs:1160-1231`, and it
*time-duplicates* rows rather than adjudicating). Concretely the shape the next
try run of #367 will produce:

- run1: `{gzip:L9:T4, pigz:L6:T4}` VOID ({...} per-arm status VOID) — verdict
  UNDECIDED; refused total rerun to recover.
- run2: `{gzip:L2/T1 after, libdeflate:L2/T1 after, pigz:L9/T1 base}` VOID —
  verdict UNDECIDED; union of the two runs' VOID sets is disjoint, so every
  cell is OK in at least one artifact, and a merged artifact is fully decidable
  without any new measurement. Today the ONLY path to that union is re-running
  the full census (~10 h/attempt — the exact cost the rescore receipt exists to
  stop re-paying).

### The honest merge (spec to implement as its own fulcrum PR)

New subcommand: `fulcrum try --rescore DIR --merge-with DIR2 ...` (or
`try --merge DIR DIR2`), writing `try-merged.json` (+N), never overwriting —
the whole rescore discipline (`promote.rs:2886-2926`) applies unchanged.

**Refusals (hard, never warnings, never a skip flag).** Before any merging,
for every input artifact:

1. `after.bin_sha` and `base.bin_sha` must be byte-identical across inputs —
   the stored fields exist (`try.json` carries them verbatim,
   `promote.rs:2267-2268`). This is exactly the rule the census already
   enforces for banked merges: refuse on `ours_sha256` mismatch — "resuming
   here would merge cells from two different gzippy binaries into one census.
   Use a fresh --out DIR" (`wallcensus.rs:649-663`), and `wallcensus report`
   REFUSES when listed dirs carry different ours-shas
   (`wallcensus.rs:1198-1217`). The try merge must not be weaker than the
   machinery it borrows from.
2. `fulcrum_commit`, `method`, `n`, `arch`, `archs_required`, `levels`,
   `threads` and the `scope` block must be identical (the first three pin the
   measurement protocol; n and the grid pin clause 8's "stated n").
3. Corpus/rival identity: for every cell present in ≥ 2 artifacts, the **size
   arms' exact integer byte counts** (`base_field`/`after_field`) must be
   equal; size is exact, so equal byte counts ⇔ same corpus, same binary, same
   rival. Wall timings are NOT identity-comparable (they drift) and are
   excluded from the check.
4. Floors: the merged adjudication runs under ONE floors source. If both
   artifacts recorded floors metadata (`median_floor`, `cells_in_file`,
   `promote.rs:2279-2285`) and it disagrees, refuse naming
   `--layout-floors`; recorded paths that cannot load here refuse exactly like
   the rescore's floor rule (`promote.rs:2895-2911`, r4 fixture
   `4135-4151`). A floor is never borrowed from another coordinate — same law
   at merge time.

**Cell banking (whole-cell only, provenance always).**

- Never stitch arms: a cell is adopted only as a unit from the artifact where
  **both arms OK** — the paired-interleaved per-pair median exists only within
  one run's pairing (clause 8, promotion-rule.md L144-149); a base from run1
  against an after from run2 is not a measurement, it is arithmetic cosplay.
- A cell OK in exactly one artifact: adopt that whole cell verbatim, record
  `merged_from: "<artifact path>"`. VOID/ABSENT in one arm in that artifact
  the same way: adopt only OK-both-arms cells; a cell with one arm VOID and
  the other OK in the same artifact stays unadopted (VOID arms already demand
  their own re-run, `promote.rs:3049-3065`).
- Cells VOID/ABSENT in both: stay UNDECIDED, re-run reasons merged from both
  artifacts.
- Cells OK in both: the higher-`n` artifact wins; equal `n` ⇒ REFUSE ("two
  complete readings of the same cell — pick a survivor by hand; a silent
  coin-flip is how census drift becomes policy").

Then adjudicate ONCE from scratch on the merged cell set under the CURRENT
rules (recompute, never copy — the r2 discipline, `promote.rs:4089-4114`),
with the artifact stating `merge_of: [paths]`, per-cell provenance, the
`CLAUSE5_MARGIN_FLOOR_RULE` string that adjudicated it, and
`verdict_matches_stored` per input. Gate-0 pins: (m1) the disjoint-VOID union
produces the fully-decidable grid above with per-cell provenance; (m2) sha
mismatch refusal names both hashes; (m3) equal-n conflict refuses; (m4) a
merged artifact never overwrites anything.

This design is the *only* way the handoff's own next step ("re-measure the two
VOIDed wall cells … then the try re-run", L259-264) ever needs a single
measurement instead of two full ones after a partial VOID — the merge is not
speculative generality, it is the gap the campaign is about to fall into.

---

## ATTACK 5 — The instrument's own receipt standard, matched (MEDIUM) + two rule-document contradictions f3406ff leaves standing (HIGH)

### 5a. Every mid-campaign rule rewrite in fulcrum's history opens with measured receipts; this one cites only the owner's mouth

Read the CHANGELOG linearly: the 2026-08-10 margin-floor entry — itself an
owner-delegated redesign — opens with "Receipts: the #295/#296/#310
adjudications … 2/2 LAYOUT-ARTIFACT", the 2026-08-11 residual-harm entry with
"Receipt: the #310 run accepted 54 erosions … then failed clause 6 on harm
1.3537 of which 0.7487 was that same accepted spend", the `try --rescore`
entry with "Receipt: three ~10-hour reruns were burned", even `supervise` opens
with its OOM SIGKILL receipt. The 2026-09-05 size-spend entry opens with "The
campaign owner's directive (2026-09-05)" and cites **no** adjudication
evidence, **no** vertical PR reference, and **no** backtest — the one entry in
the file that re-prices the contract reads the least like its own
precedents. The rule file independently names the receipts (PR #370, the
clause-5/6 amendments), so the standard is one sentence plus one list away.

**Fix (one turn):** prepend to the fulcrum CHANGELOG's size-spend entry a
`Receipt:` paragraph: "#367's frozen-box verdict NO-SHIP'd on one size
conviction (silesia L2/T1 0.9973→1.0000 vs libdeflate, Δ 0.0027) with clauses
4/6 otherwise OK (improvement 3.9x vs harm) — a correct adjudication under the
pre-amendment rule; the ceiling re-prices the trade by owner directive rather
than correcting a mis-specified gate. Backtest against the stored
adjudications: [fill from Attack 1's backtest]." Add the PR references
(#35/#370) the sibling entries all carry.

### 5b. promotion-rule.md contradicts itself inside clause 5 (HIGH)

The 2026-09-05 amendment inserted the size-spend paragraph (promotion-rule.md
L54-64) and amended the clause-6 residual bullet (L115-127), but left the
clause-5 framing sentence directly above the insertion untouched — L50-52
still reads: "The rule now has two parts, both applying to **wall** cells only
— size cells are exact integers and unchanged: a size erosion or size flip
convicts directly, no confirmation involved." That sentence is now *false
under the same section* (it authorizes what L56-57 convicts, and the code's own
selftest (e) asserts the superseded wording "size cells exact and unchanged" is
GONE from artifacts — `promote.rs:3940-3944` — while it demonstrably survives
in the governing document). A human reading the rule top-down enforces the
pre-amendment rule and calls the instrument noncompliant; a human reading the
code sees the artifact rule-string changed and the governing doc didn't. The
rule file's first sentence declares itself "written down so it cannot be
adjusted after a measurement comes back inconvenient" (L3-4) — adjusted copies
that contradict each other are worse than either single state. **Fix:** one
rewritten sentence at L50-52: "…both applying to wall cells; size cells are
exact and judged by the 2026-09-05 authorized-spend ceiling below."

### 5c. The two 2026-09-05 lock-in documents define "wall never loses" on opposite sides of the instrument (HIGH)

CLAUDE.md's warning box (rules worktree, L84-85) writes: "**WALL NEVER LOSES**
(a wall regression is a conviction under any circumstances — clauses 3/5/7's
cross-layout machinery unchanged)". The instrument — which f3406ff did not
touch on the wall axis, by its own commit message — ACCEPTS confirmed-real
wall erosions on winning cells that clear the margin floor
(`promote.rs:866-885`, `921-930`), buckets them as
`excluded_margin_spend`, and charges no harm: a wall regression past the floor
is *not* a conviction, it is accepted spend. promotion-rule.md's own long-form
text agrees with the instrument ("A CONFIRMED-real erosion on a winning wall
cell … is acceptable iff … post_ratio <= min(0.80, …)", L76-83). So the owner
directive's operative semantic — does "we will not lose on wall clock under
any conditions" mean *never lose to the rival* (floor semantics, the
instrument's current reading: post 0.80 = still 20% faster than the rival) or
*never regress any cell versus base* (CLAUDE.md's hyperbole)? — is resolved,
in the author's favor, implicitly, in the *looser* direction, by not writing
anything. These are the two governing copies of the same directive written
20 minutes apart; they must say the same thing. If the floor's rival-anchored
reading is intended (it is what the instrument implements and what the #367
verdict's wall leg already exercises), fix CLAUDE.md's sentence to "wall may
be spent down to the floor and never crosses to the rival's side"; if the
absolute reading is intended, the 2026-08-10 margin-floor machinery is
falsified by this directive and f3406ff's "Wall rules … byte-for-byte
unchanged" is the wrong wall to hold still. Either way this is the owner's
sentence being stretched across two different rules; one of them gets to be
wrong on paper, and today it is whichever a human reads second.

---

## Sequencing, unpriced residues, and what I could not verify

- Sequencing the handoff records is fair to the author: rules worktree
  (`e97f54dd`), instrument (`f3406ff`), then re-adjudication is exactly the
  2026-07-28 protocol. I am not alleging branch hygiene — the findings above
  are about what the amendment prices, says, and pins, not where.
- I could not read PR #370's body or fulcrum#35's body (out of repo scope); if
  the discipline checklist and backtest live there, pull them into
  promotion-rule.md and the CHANGELOG, since the artifact readers will never
  see a PR body.
- `cargo test` numbers ("682 passed / 0 failed") are quoted by the commit; I
  did not run them per the read-only mandate. Every passing claim above that
  depends on code paths is cited line-by-line; the two behavioral claims not
  directly exercised by a fixture — the 36-spend compound SHIP and the stored
  NO-SHIP rescore flip — are the two fixtures my fix list demands.

## GO / NO-GO — fulcrum#35 (`f3406ff`) as-is

**NO-GO as-is.** The change is directional-right (it generalizes 2026-08-10's
margin-as-capital doctrine and honors a real owner trade), but it ships ten
findings — the table below; two of them change what live verdicts can say
(Attack 2's unbounded compound; Attack 3e's ceiling-vs-flip boundary as
exercised by the fixtures a human reviewer will trust):

| # | Finding | Severity | Fix |
|---|---------|----------|-----|
| 3e | Fixtures (a)/(d) pin SHIP for a size payload the census cannot produce (after-ratio > 1.0 flagged passing); (c) in the same block assumes the opposite convention | HIGH | Re-pin to 0.9973→1.0000 (tie-crossing, still Δ 0.0027); add impossible-payload flip fixture |
| 2 | No aggregate cap on authorized size spend; thousand-cuts protection gone on the size axis; compound demo ships at harm 0.0000 | HIGH | Run-total cap (`excluded_size_spend ≤ 0.01`/run) or an owner-ratified per-cell receipt with a stated total; + compound gate-0 fixture |
| 5b | promotion-rule.md L50-52 contradicts the amendment 2 lines below it | HIGH | One sentence |
| 5c | CLAUDE.md "a wall regression is a conviction under any circumstances" vs the margin floor the instrument still enforces | HIGH (doc) | Reconcile, owner sign-off on the reading |
| 5a | No Receipt/backtest/counterfactual — the entry skips its own history's standard | MEDIUM | One paragraph + backtest note |
| 3d | Rescore LOOSEN direction unpinned; sub-ceiling fixture class deleted by the re-pins; no reclassification ledger | MEDIUM | Mirrored pin + `reclassifications` field |
| 3a | "<1%" owner verbatim vs "<= 1%" rule text vs `+1e-12` code, no boundary pins | LOW | Declare half-open interval + two boundary pins |
| 3b | Sub-flag size spends itemized in aggregate only | LOW | Per-cell [size-spend] line |
| 2a | "authorized size spend <=1%" printed under the "excluded as clause-5-priced:" prefix (`promote.rs:1079-1081`) | LOW | Rename prefix "excluded (priced where authorized):" |
| 4 | No merge op for disjoint-VOID stored artifacts (next #367 rerun burns ~10 h) | MEDIUM (separate PR) | Banked merge per Attack 4 spec, with the `wallcensus`-style sha refusals |

Blocked items 2 and 5c need a one-line owner call (per-cell vs run-total;
rival-anchored vs absolute wall reading) — everything else is a same-day fix
that does not change the fulcrum branch's wall arithmetic. After 3e, 5b, and
either (2 or the owner receipt), respecify the changelog receipt, and this is a
GO.
