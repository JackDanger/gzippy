# gzippy-isal PARITY GATE MANDATE (supervisor, from process retrospective 2026-06-06)

BIGGEST CURRENT RISK (process review): the gzippy-isal real-ISA-L "Design-A" tail
is deferred to last, and its byte-exact parity rests on the STEP-1b differential
(8d026a8, 10/10) — which proved pure ≡ ISA-L tail ONLY on the CURRENT two-phase
driver shape. The gzippy-native fold-to-one-engine will REWRITE the very driver
interface Design-A swaps into. So the fold can SILENTLY invalidate the only parity
proof we have, and PHASE 3's 3-way numbers could then be measured against a
subtly-diverged gzippy-isal baseline (this is incident #4 — the inverted
divergence map — recurring).

## REQUIRED before the fold is trusted / before Phase 3
1. HARDEN the STEP-1b pure-vs-ISA-L tail differential NOW (before the fold lands)
   to exercise the edge cases the fold could break: the `:486` rewind path,
   fixed-Huffman no-rewind, and the `== stop_hint` boundary (plans note
   lines ~178-181). Not just the happy path.
2. FREEZE it as a PERMANENT gate that runs against the FOLDED native driver (not
   only the pre-fold two-phase shape) — so any fold that diverges the tail FAILS
   the gate instead of passing silently.
3. PHASE 3 precondition: the gzippy-isal baseline used in the 3-way Fulcrum MUST
   pass this hardened differential against the folded driver + dual-sha
   028bd002...cb410f. No 3-way numbers against an unvalidated isal baseline.

## Status
Supervisor will inject this into the leader re-drive that does the fold, and
verify (independent advisor) that the hardened differential passes against the
folded driver BEFORE the gzippy-isal Design-A tail is built and BEFORE Phase 3.
