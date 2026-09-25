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
