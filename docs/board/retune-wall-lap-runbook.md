# Retuned-L9 wall lap runbook (the box session)

1. SSO click (login loop armed). Developer account, us-east-1.
2. Launch c7a.4xlarge dedicated (subnet 0676cc1b07166ec50, SG sg-0837..., profile
   gzippy-adjudication-ssm-core, ami-0fef201115eefe936), user-data clips from
   /tmp/final-lap-v2.sh steps: fulcrum at d738eae + selftest gate; rivals build.
3. Clone gzippy at **origin/main (a29b87b7)** — the trunk now carries the
   passes2 retune (PR #373) AND the thread-free chunk grid (PR #374,
   `GRID_REF_THREADS`). The old sprint-branch checkout step is dead: the
   retune merged; the lap adjudicates the TRUNK as shipped.
4. calibrate at levels 9 threads 4 + floors; verify no VOID; full repair if VOID.
5. wall census:
   fulcrum try a29b87b7 --base aa682fcc --levels 9 --threads 1,4 --n 45
   --corpus /root/silesia.tar --rival gzip/pigz/libdeflate
   --layout-floors /root/wave-out/layout-floors-l9t4/layout_floors.tsv
   --out /root/wave-out/wave-retune9
   (aa682fcc = the pre-retune trunk the S2 adjudication pinned as base;
    the try-vs-base delta = PR #373's retune + PR #374's grid pin together.)
6. try --rescore; adjudicate:
   ABORT if pigz:silesia.tar:L9:T4:wall does not close or fail-gap fails <=1%.
   REVISED ABORT CONSEQUENCE (design v3.1 — the v1 revert-to-Lazy2 leg is
   DEAD by the +2.9% cap measurement): keep the size, declare the wall cell
   OPEN, and brief a NEW wall lever priced under the cap — the revert is not
   on the menu.
7. artifact: /root/wave-out/wave-retune9 pushed to S3 before termination and
   committed under `docs/board/records/<date-box>/final-lap/`.

## The STACKED leg (next box session, 2026-09-26)

The candidate set per the lever ledger (sprint-2026-09-25.md): the trunk's
passes2 retune STAYS; evaluate on one box:

1. **near-opt-parallel-flush ON** (`--features near-opt-parallel-flush` in
   both arms; DEFAULT OFF on trunk — the lever is adjudication-gated): flush
   N overlapping fill N+1; measured 0.71-0.74x of the serial flush share
   locally; the mid-chunk byte-identity precondition is zero-flip corpora
   (0 flips across 47 campaign-corpus flushes; the guard latches serial on
   ANY flip). Byte receipt: the parallel arms' streams must be byte-identical
   to the serial arms on the same corpora.
2. **d1: near-opt depth 100 at L9, passes 2** — one knob, `level.rs:505-507`;
   the local A/B on the probe corpus was FLAT (depth does not bind there), so
   the silesia leg is the decider. Byte receipt needed unchanged-or-≤cap per
   the retune's own census discipline.

Try invocation (per levels 2,9 protocol — fulcrum refuses single-level):
   fulcrum try trunk --base 908f587e --levels 2,9 --threads 1,4 --n 45
   --corpus /root/silesia.tar --rival gzip/pigz(+libdeflate)
   --layout-floors <fresh l9t4> --out /root/wave-out/wave-lever3

Receipt gates:
- pigz L9/T4 wall from 1.0822 toward <=1.0 (the lever's whole point);
- size census per cell <= the 1%-cap authorization;
- the feature-off arms byte-identical to trunk (the flush must be invisible).
