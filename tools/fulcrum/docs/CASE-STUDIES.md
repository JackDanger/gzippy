# Case studies — the measurement failures behind the invariants

Fulcrum's rules were not designed in the abstract. Each was extracted from a
real failure in a long performance-parity campaign (a gzip decoder being
ported to match a reference tool, measured for months on a frozen bench box).
`fulcrum invariants` names each rule for its scar; this file is the long form.
Project-specific names are kept verbatim — the value of a case study is that
it really happened.

## The half-rebased matrix (SINK-LAW)

A scorecard compared the tool against its comparator across a workload ×
thread matrix. Mid-campaign, the comparator's numbers were re-measured with a
regular-file output sink (the honest protocol — users get real output), but
the tool's own numbers stayed cached from /dev/null-sink runs. The published
"T1 0.973" ratio was a phantom composed of two different experiments. A
follow-up claim — "the tool is sink-insensitive, so its /dev/null numbers
stand" — was falsified when measured: ~110 ms/run of real output cost at T1.

Rule extracted: both arms of ANY comparison must carry the same sink class
*in the stored number itself*, and the tool must refuse mixed-sink ratios.
Corollary (the writev phantom): a FIFO sink with a draining reader behaves
like neither a file nor /dev/null; non-file sinks are flagged on sight.

## The stale anchor (FINGERPRINT-OR-NO-COMPARE, ledger)

A "0.98x parity" claim was computed against a *banked* comparator wall of
926.6 ms. A live, co-located run of the same comparator measured ~810 ms.
Every intermediate ratio in the campaign needed a re-base lens. The bank was
not wrong when it was written — the world had moved (comparator rebuilt,
conditions changed) and nothing forced the comparison to notice.

Rule extracted: every stored number carries its full measurement fingerprint;
a live number contradicting a *compatible* banked number is flagged
CONTRADICTS-LEDGER instead of silently ranked; the contradicting number is
banked pending-reconcile (never as an anchor) until a supersede record
resolves which side was wrong.

## The clock confound (FINGERPRINT-OR-NO-COMPARE)

A per-iteration cycle count "regressed" ~25% against a banked profile. The
code had not changed: the banked capture ran under a different frequency
state (turbo on) than the live one (frozen, no_turbo). TSC cycles are
fixed-rate; core clocks are not; the comparison was between two different
clocks. The structure (relative shares) matched perfectly — the tell that
this was a protocol mismatch, not a regression.

Rule extracted: protocol/freeze state are fingerprint fields; cross-state
ratios refuse. Structure-matches-but-absolute-diverges is surfaced as a
frequency-state hypothesis, not a regression.

## The thawed-box drift (FROZEN-OR-LABELED)

An A/B verdict flipped between 0.945 and 0.989 across sessions. The cause: a
bench-freeze TTL had lapsed mid-campaign and the freeze guard only WARNED.
The drift was caught by absolute-level sanity checking, not by the guard.

Rule extracted: a wall number from a thawed/loaded box is refused for
ranking (an explicit override downgrades refusal to a loud label on every
affected row). A readable-and-wrong freeze value is never overridable.

## The read-slurp false divergence (SHA-OR-VOID, structural checks)

A shell harness read "seconds sha" from a helper that had started emitting
"seconds sha rss". `read`'s last variable slurped the remainder of the line,
corrupting the sha with the appended rss — producing a false SHA-DIVERGENCE
on byte-perfect output. The inverse failure (trusting unverified bytes) is
worse: a speed win with wrong bytes is a loss.

Rule extracted: content verification must be structural (the harness aborts
on mismatch and the analyzer accounts for the verification), and the
verification machinery itself needs positive/negative controls.

## The bimodal comparator (SPREAD-RESOLUTION)

At high thread counts the comparator's wall distribution went bimodal
(scheduling regimes), and with N=21 samples a median could land on either
mode. Whole sessions were spent "measuring" deltas smaller than the spread.

Rule extracted: every verdict carries RESOLVED/UNRESOLVED with N-needed; a
sub-spread delta is never presented as a finding; bimodality is detected and
flagged on every sample set.

## Attribution that never converted (CAUSAL-OR-HYPOTHESIS)

Three separate "levers" — a 377 ms pair-drain, a per-block-end stop cost, a
key-mismatch re-key — dominated busy-time/latency/critical-path attribution
and moved the wall not at all when fixed. Separately, a nested-span
double-count manufactured a 62 ms "serial CRC cost" that was actually an
O(1) combine. Cycle-share profiles did not translate to wall share.

Rule extracted: attribution is a hypothesis generator. No row ranks as
actionable without a tool-executed causal A/B (same-binary kill-switch,
effect-verified); everything else is HYPOTHESIS plus the exact
pre-registered perturbation that would test it.

## The kill-switch that didn't switch (EFFECT-VERIFIED-OR-FLAGGED)

An allocator A/B read "the stats line prints in the knob arm" as proof of
engagement — but the line printed in BOTH arms (it was gated on the stats
env var, not the feature). Another harness built duplicate env entries;
env last-wins meant ZERO injection, silently, in what was reported as a
50%-slowdown arm.

Rule extracted: a kill-switch A/B is causal only when a counter predicate
proves the switch engaged in one arm and not the other; line *presence*
proves nothing; a failed predicate voids the A/B.

## The broken instruments (SELF-TEST-OR-NO-TRUST)

Two instruments were silently broken for days: a "removal oracle" that
quietly re-ran the work it claimed to remove (its ceiling was a tie by
construction), and a capture path that emitted empty output that was then
"analyzed". A coverage assertion (busy+idle==span) was once implemented as
a tautology — it could not fail.

Rule extracted: every analyzer ships synthetic-input self-tests with
positive AND negative controls, including corruption tests proving the trust
assertions FIRE; the engine labels its own output untrusted when its
self-test stamp is missing or stale for the current source.

## The mislabeled binary (derived fields, comparator/host identity)

A "bombshell" finding — the vendored fast path dormant in production — came
from a run whose binary was a differently-built artifact than labeled. The
counters were structurally impossible for the claimed build. Separately:
requested CPU pins are not always the pins a process runs under (cgroup
cpusets shrink masks silently).

Rule extracted: identity fields are DERIVED, never self-reported — binary
sha, sink class via stat, mask via kernel readback, freeze via sysfs,
comparator version via probe, host identity from the box itself. A
self-report contradicting its derivation is flagged as a lying manifest.
