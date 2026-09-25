# Retuned-L9 wall lap runbook (the box session)

1. SSO click (login loop armed). Developer account, us-east-1.
2. Launch c7a.4xlarge dedicated (subnet 0676cc1b07166ec50, SG sg-0837..., profile
   gzippy-adjudication-ssm-core, ami-0fef201115eefe936), user-data clips from
   /tmp/final-lap-v2.sh steps: fulcrum at d738eae + selftest gate; rivals build.
3. Clone gzippy at sprint/parity-and-losses (HEAD 604ae0ec / the retune).
4. calibrate at levels 9 threads 4 + floors; verify no VOID; full repair if VOID.
5. wall census:
   fulcrum try origin/lever-jiu... - no; the retuned branch tip: try 604ae0ec
   --base origin/main --levels 9 --threads 1,4 --n 45
   --corpus /root/silesia.tar --rival gzip/pigz/libdeflate
   --layout-floors /root/wave-out/layout-floors-l9t4/layout_floors.tsv
   --out /root/wave-out/wave-retune9
6. try --rescore; adjudicate:
   ABORT if pigz:silesia.tar:L9:T4:wall does not close or fail-gap fails <=1%;
   revert the retune and land the REVISED shape (Lazy2+depth-scaled at L9/T>1).
7. artifact: /root/wave-out/wave-retune9 pushed to S3 and committed here.
