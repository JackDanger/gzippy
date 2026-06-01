#!/usr/bin/env bash
# DEPRECATED (2026-05-31) — reported wall-clock on an UNLOCKED, UNGATED box.
#
# This was a compression-side L11 zopfli byte-identity + informal wall-clock
# check wired into `make ship` step 4. Its perf numbers were the kind of noisy,
# unverified absolutes the campaign post-mortem banned. Byte-identity belongs in
# the correctness tests; trustworthy wall numbers come ONLY from the gated
# clean-bench harness.
#
# USE INSTEAD:  scripts/bench/clean_bench.sh   (see scripts/bench/README.md)
# (referenced in memory; kept as a stub rather than deleted.)
echo "DEPRECATED — use scripts/bench/clean_bench.sh (see scripts/bench/README.md)" >&2
exit 1
